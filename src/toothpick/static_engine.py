"""
Toothpick static analysis & normalization engine (unit W03).

Implements ``PLAN-W03-static.md`` lines 20-31 and ``docs/design-document.md``
section 6:

* ``StaticEngine`` — the engine interface (also published as
  :class:`StaticEngineProtocol`) plus a Capstone-backed implementation.
* Multi-architecture disassembly: x86, x86_64, ARM64, PowerPC, MIPS.
* Container ingestion: PE, ELF, Switch NSO/NRO, and raw memory blobs
  (``--raw --arch <arch> --base <addr>``).
* Function detection: entry point, container symbols (PE exports / ELF symtab),
  direct call targets, and a post-terminator / post-padding sweep.
* Call-graph extraction: direct ``call`` edges, intra-function jump ``cfg``
  edges, and ``xref`` edges for branches leaving the analysed function.
* Instruction lifting: per-register def/use facts (``ssa_facts`` table) derived
  from Capstone register-access information.
* Syntactic normalization pre-pass: negative-literal normalization
  (``0xffffffff`` -> ``-1``) and conservative dead-assignment / NOP trimming.
* Results are written straight into ``toothpick.db`` via
  :class:`toothpick.storage.Database`, transactionally and idempotently.

Honesty notes (what this module deliberately does **not** claim):

* PyGhidra is an *optional* bridge (``prefer_ghidra=True`` and a usable Ghidra
  install). Without it ``decompiled_c`` stays ``NULL`` — this module never
  fabricates decompiler output.
* NSO/NRO parsing is best-effort header/layout parsing; an LZ4-compressed NSO
  ``.text`` is reported as a note and not decompressed here.
* Dead-assignment elimination is liveness-based and conservative: writes to
  return-value, stack, and flag registers are always treated as live, and
  side-effecting instructions (branches, calls, memory writers) are never
  removed.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import os
import struct
import time
from collections import deque
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import (
    Any,
    Deque,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    Union,
    runtime_checkable,
)

import capstone
from capstone import Cs

from toothpick.storage import (
    Database,
    EdgeKind,
    EdgeRecord,
    FunctionRecord,
    SsaFact,
)
from toothpick.triage import ContainerType

__all__ = [
    "StaticEngineError",
    "UnsupportedArchitectureError",
    "ContainerParseError",
    "GhidraUnavailableError",
    "ArchSpec",
    "Section",
    "ContainerInfo",
    "Instruction",
    "FunctionAnalysis",
    "AnalysisResult",
    "StaticEngineProtocol",
    "StaticEngine",
    "detect_container",
    "load_container",
    "arch_spec",
    "normalize_arch_name",
    "ghidra_available",
]

PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# exceptions
# ---------------------------------------------------------------------------


class StaticEngineError(Exception):
    """Base exception for static engine failures."""


class UnsupportedArchitectureError(StaticEngineError):
    """Raised when no disassembler exists for the requested architecture."""


class ContainerParseError(StaticEngineError):
    """Raised when the target container cannot be parsed."""


class GhidraUnavailableError(StaticEngineError):
    """Raised when the PyGhidra bridge was requested but is not usable."""


# ---------------------------------------------------------------------------
# architecture specification
# ---------------------------------------------------------------------------

_CS_ARM64 = getattr(capstone, "CS_ARCH_ARM64", getattr(capstone, "CS_ARCH_AARCH64", None))
_OP_IMM = getattr(capstone, "CS_OP_IMM", 2)

# alias -> (canonical arch, default pointer size)
_ARCH_ALIASES: Dict[str, Tuple[str, int]] = {
    "x86": ("x86", 4),
    "i386": ("x86", 4),
    "ia32": ("x86", 4),
    "x86_32": ("x86", 4),
    "x86_64": ("x86_64", 8),
    "x86-64": ("x86_64", 8),
    "x64": ("x86_64", 8),
    "amd64": ("x86_64", 8),
    "arm": ("arm", 4),
    "arm32": ("arm", 4),
    "arm64": ("arm64", 8),
    "aarch64": ("arm64", 8),
    "arm64e": ("arm64", 8),
    "ppc": ("ppc", 4),
    "powerpc": ("ppc", 4),
    "ppc32": ("ppc", 4),
    "ppc64": ("ppc", 8),
    "powerpc64": ("ppc", 8),
    "mips": ("mips", 4),
    "mips32": ("mips", 4),
    "mips64": ("mips", 8),
}

SUPPORTED_ARCHES: Tuple[str, ...] = ("x86", "x86_64", "arm", "arm64", "ppc", "mips")

# PE machine -> (arch, pointer size)
_PE_MACHINES: Dict[int, Tuple[str, int]] = {
    0x014C: ("x86", 4),
    0x8664: ("x86_64", 8),
    0x01C0: ("arm", 4),
    0x01C4: ("arm", 4),
    0xAA64: ("arm64", 8),
    0xA641: ("arm64", 8),
    0x0166: ("mips", 4),
    0x0169: ("mips", 4),
    0x01F0: ("ppc", 4),
    0x01F1: ("ppc", 4),
    0x01F2: ("ppc", 8),
}

# ELF e_machine -> (arch, pointer size)
_ELF_MACHINES: Dict[int, Tuple[str, int]] = {
    0x03: ("x86", 4),
    0x3E: ("x86_64", 8),
    0x28: ("arm", 4),
    0xB7: ("arm64", 8),
    0x14: ("ppc", 4),
    0x15: ("ppc", 8),
    0x08: ("mips", 4),
    0x0A: ("mips", 8),
}

# Capstone's group flags are not uniform across architectures, so mnemonic
# knowledge is kept per architecture for branches/returns.
_ARCH_CALL_MNEMONICS: Dict[str, Set[str]] = {
    "x86": {"call", "lcall"},
    "x86_64": {"call", "lcall"},
    "arm": {"bl", "blx"},
    "arm64": {"bl", "blraa", "blraaz", "blrab", "blrabz"},
    "ppc": {"bl", "bla", "bcl", "bcla", "bctrl", "bcctrl"},
    "mips": {"jal", "jalx", "bal", "bgezal", "bltzal", "bgezall"},
}

_ARCH_RET_MNEMONICS: Dict[str, Set[str]] = {
    "x86": {"ret", "retf", "iret", "iretd", "iretq", "sysret", "sysexit"},
    "x86_64": {"ret", "retf", "iret", "iretd", "iretq", "sysret", "sysexit"},
    "arm": {"bx", "pop", "ldm"},
    "arm64": {"ret", "eret", "blr"},
    "ppc": {"blr", "bclr", "rfi", "rfid"},
    "mips": {"jr"},
}

# unconditional branches: no fall-through after them
_ARCH_UNCOND_JUMPS: Dict[str, Set[str]] = {
    "x86": {"jmp", "ljmp"},
    "x86_64": {"jmp", "ljmp"},
    "arm": {"b", "bx"},
    "arm64": {"b", "br"},
    "ppc": {"b", "ba"},
    "mips": {"b", "j"},
}

_PADDING_MNEMONICS: Set[str] = {
    "nop",
    "nopw",
    "nopl",
    "hint",
    "ud0",
    "ud1",
    "ud2",
    "udf",
    "hlt",
    "int3",
    "int",
    "brk",
    "trap",
    "invalid",
}

# prologue-ish first mnemonics, used only to promote *external* branch targets
# (tail calls / split blocks) to function-start candidates
_ARCH_PROLOGUE_MNEMONICS: Dict[str, Set[str]] = {
    "x86": {"push", "endbr64", "endbr32", "sub", "mov", "lea", "xor", "test", "ret"},
    "x86_64": {"push", "endbr64", "endbr32", "sub", "mov", "lea", "xor", "test", "ret"},
    "arm": {"push", "stmdb", "sub", "mov", "bx", "ldr"},
    "arm64": {"stp", "sub", "mov", "pacibsp", "ret", "hint", "ldr"},
    "ppc": {"stwu", "mflr", "stw", "addi", "blr", "lis"},
    "mips": {"addiu", "sw", "lui", "jr", "move"},
}

# instructions whose semantics we do not model — never call these dead
_SIDE_EFFECT_MNEMONICS: Set[str] = {
    "syscall",
    "sysenter",
    "cpuid",
    "rdtsc",
    "rdtscp",
    "xgetbv",
    "mfence",
    "lfence",
    "sfence",
    "stosb",
    "stosd",
    "stosq",
    "movsb",
    "movsd",
    "movsq",
    "sc",
    "svc",
    "hvc",
    "smc",
    "mrs",
    "msr",
    "wrmsr",
    "rdmsr",
}


@dataclass(frozen=True)
class ArchSpec:
    """Capstone wiring plus the ABI facts the normalizer needs."""

    arch: str
    pointer_size: int
    alignment: int
    capstone_arch: int
    capstone_mode: int
    return_regs: Tuple[str, ...]
    stack_regs: Tuple[str, ...]
    flag_regs: Tuple[str, ...]

    @property
    def always_live(self) -> Set[str]:
        return set(self.return_regs) | set(self.stack_regs) | set(self.flag_regs)


def _canonical_reg(arch: str, reg: str) -> str:
    """Map architecture sub-registers to root register names for def-use tracking."""
    r = reg.lower().strip()
    if not r:
        return ""
    if arch in ("x86", "x86_64"):
        x86_map = {
            "rax": "rax", "eax": "rax", "ax": "rax", "al": "rax", "ah": "rax",
            "rbx": "rbx", "ebx": "rbx", "bx": "rbx", "bl": "rbx", "bh": "rbx",
            "rcx": "rcx", "ecx": "rcx", "cx": "rcx", "cl": "rcx", "ch": "rcx",
            "rdx": "rdx", "edx": "rdx", "dx": "rdx", "dl": "rdx", "dh": "rdx",
            "rsi": "rsi", "esi": "rsi", "si": "rsi", "sil": "rsi",
            "rdi": "rdi", "edi": "rdi", "di": "rdi", "dil": "rdi",
            "rbp": "rbp", "ebp": "rbp", "bp": "rbp", "bpl": "rbp",
            "rsp": "rsp", "esp": "rsp", "sp": "rsp", "spl": "rsp",
            "rflags": "rflags", "eflags": "rflags", "flags": "rflags",
        }
        if r in x86_map:
            return x86_map[r]
        for i in range(8, 16):
            if r in (f"r{i}", f"r{i}d", f"r{i}w", f"r{i}b"):
                return f"r{i}"
        return r
    if arch in ("arm64", "arm"):
        if r.startswith("w") and r[1:].isdigit():
            return f"x{r[1:]}"
        if r == "wsp":
            return "sp"
        return r
    if arch == "mips":
        if not r.startswith("$") and ("$" + r) in (
            "$zero", "$at", "$v0", "$v1", "$a0", "$a1", "$a2", "$a3",
            "$t0", "$t1", "$t2", "$t3", "$t4", "$t5", "$t6", "$t7",
            "$s0", "$s1", "$s2", "$s3", "$s4", "$s5", "$s6", "$s7",
            "$t8", "$t9", "$k0", "$k1", "$gp", "$sp", "$fp", "$ra",
        ):
            return "$" + r
        return r
    return r


def normalize_arch_name(name: str) -> Tuple[str, int]:
    """Map an operator-supplied arch name to ``(canonical_arch, pointer_size)``."""
    key = str(name).strip().lower().replace(" ", "").replace("-", "_")
    if key in _ARCH_ALIASES:
        return _ARCH_ALIASES[key]
    raise UnsupportedArchitectureError(
        f"unsupported architecture {name!r}; supported aliases: {sorted(_ARCH_ALIASES)}"
    )


def _build_spec(arch: str, pointer_size: int) -> ArchSpec:
    if arch == "x86":
        return ArchSpec(
            arch=arch,
            pointer_size=4,
            alignment=1,
            capstone_arch=capstone.CS_ARCH_X86,
            capstone_mode=capstone.CS_MODE_32,
            return_regs=("rax", "rdx", "eax", "edx", "ax", "dx"),
            stack_regs=("esp", "ebp", "sp", "bp"),
            flag_regs=("eflags", "flags"),
        )
    if arch == "x86_64":
        return ArchSpec(
            arch=arch,
            pointer_size=8,
            alignment=1,
            capstone_arch=capstone.CS_ARCH_X86,
            capstone_mode=capstone.CS_MODE_64,
            return_regs=("rax", "rdx", "eax", "edx", "ax", "dx"),
            stack_regs=("rsp", "rbp", "esp", "ebp", "sp", "bp"),
            flag_regs=("rflags", "eflags", "flags"),
        )
    if arch == "arm":
        return ArchSpec(
            arch=arch,
            pointer_size=4,
            alignment=4,
            capstone_arch=capstone.CS_ARCH_ARM,
            capstone_mode=capstone.CS_MODE_ARM,
            return_regs=("r0", "r1"),
            stack_regs=("sp", "fp", "r11", "r13", "lr"),
            flag_regs=("cpsr", "apsr"),
        )
    if arch == "arm64":
        if _CS_ARM64 is None:  # pragma: no cover - capstone build too old
            raise UnsupportedArchitectureError("this capstone build lacks CS_ARCH_ARM64")
        return ArchSpec(
            arch=arch,
            pointer_size=8,
            alignment=4,
            capstone_arch=_CS_ARM64,
            capstone_mode=capstone.CS_MODE_ARM,
            return_regs=("x0", "w0", "x1", "w1"),
            stack_regs=("sp", "wsp", "x29", "w29", "fp", "lr", "x30", "w30"),
            flag_regs=("nzcv",),
        )
    if arch == "ppc":
        mode = capstone.CS_MODE_64 if pointer_size == 8 else capstone.CS_MODE_32
        return ArchSpec(
            arch=arch,
            pointer_size=pointer_size,
            alignment=4,
            capstone_arch=capstone.CS_ARCH_PPC,
            capstone_mode=mode + capstone.CS_MODE_BIG_ENDIAN,
            return_regs=("r3", "r4"),
            stack_regs=("r1", "r31"),
            flag_regs=("cr0", "cr1", "cr2", "cr3", "cr4", "cr5", "cr6", "cr7", "xer"),
        )
    if arch == "mips":
        mode = capstone.CS_MODE_MIPS64 if pointer_size == 8 else capstone.CS_MODE_MIPS32
        return ArchSpec(
            arch=arch,
            pointer_size=pointer_size,
            alignment=4,
            capstone_arch=capstone.CS_ARCH_MIPS,
            capstone_mode=mode + capstone.CS_MODE_BIG_ENDIAN,
            return_regs=("$v0", "v0", "$a0", "a0"),
            stack_regs=("$sp", "sp", "$fp", "fp", "$gp", "gp"),
            flag_regs=(),
        )
    raise UnsupportedArchitectureError(
        f"unsupported architecture {arch!r}; supported: {list(SUPPORTED_ARCHES)}"
    )


def arch_spec(arch: str, pointer_size: Optional[int] = None) -> ArchSpec:
    """Build the :class:`ArchSpec` for an arch name (or alias)."""
    canonical, bits = normalize_arch_name(arch)
    return _build_spec(canonical, pointer_size or bits)


# ---------------------------------------------------------------------------
# container model
# ---------------------------------------------------------------------------


@dataclass
class Section:
    """One mapped region of the target."""

    name: str
    vaddr: int
    vsize: int
    data: bytes
    file_offset: int = 0
    is_exec: bool = False

    @property
    def end(self) -> int:
        return self.vaddr + max(self.vsize, len(self.data))

    def contains(self, addr: int) -> bool:
        return self.vaddr <= addr < self.end

    def has_bytes(self, addr: int) -> bool:
        return self.vaddr <= addr < self.vaddr + len(self.data)

    def bytes_at(self, addr: int) -> bytes:
        if not self.has_bytes(addr):
            return b""
        return self.data[addr - self.vaddr:]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "vaddr": self.vaddr,
            "vsize": self.vsize,
            "size": len(self.data),
            "file_offset": self.file_offset,
            "executable": self.is_exec,
        }


@dataclass
class ContainerInfo:
    """Parsed view of one target container."""

    path: str
    container: str
    arch: str
    base: int
    entry: int
    pointer_size: int
    sections: List[Section] = field(default_factory=list)
    symbols: Dict[int, str] = field(default_factory=dict)
    machine: str = ""
    sha256: str = ""
    is_raw: bool = False
    notes: List[str] = field(default_factory=list)

    @property
    def spec(self) -> ArchSpec:
        return _build_spec(self.arch, self.pointer_size)

    def exec_sections(self) -> List[Section]:
        return [s for s in self.sections if s.is_exec and s.data]

    def exec_section_with_bytes(self, addr: int) -> Optional[Section]:
        for sec in self.sections:
            if sec.is_exec and sec.has_bytes(addr):
                return sec
        return None

    def is_mapped(self, addr: int) -> bool:
        return any(sec.contains(addr) for sec in self.sections)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target": self.path,
            "container": self.container,
            "arch": self.arch,
            "base": self.base,
            "entry_point": self.entry,
            "pointer_size": self.pointer_size,
            "machine": self.machine,
            "is_raw": self.is_raw,
            "binary_sha256": self.sha256,
            "sections": [s.to_dict() for s in self.sections],
            "symbol_count": len(self.symbols),
            "notes": list(self.notes),
        }


def detect_container(data: bytes) -> str:
    """Classify raw bytes into a :class:`toothpick.triage.ContainerType` value."""
    if data[:2] == b"MZ":
        return ContainerType.PE.value
    if data[:4] == b"\x7fELF":
        return ContainerType.ELF.value
    if data[:4] == b"NSO0":
        return ContainerType.NSO.value
    if len(data) > 0x14 and data[0x10:0x14] == b"NRO0":
        return ContainerType.NRO.value
    if data[:4] == b"SCE\x00":
        return ContainerType.SELF.value
    if len(data) >= 0x100:
        first_text_off = _u32_be(data, 0)
        entry = _u32_be(data, 0xE0)
        if first_text_off == 0x100 and (0x80000000 <= entry < 0x81800000):
            return ContainerType.DOL.value
    return ContainerType.UNKNOWN.value


def _u16(data: bytes, off: int) -> int:
    return struct.unpack_from("<H", data, off)[0] if off + 2 <= len(data) else 0


def _u32(data: bytes, off: int) -> int:
    return struct.unpack_from("<I", data, off)[0] if off + 4 <= len(data) else 0


def _u32_be(data: bytes, off: int) -> int:
    return struct.unpack_from(">I", data, off)[0] if off + 4 <= len(data) else 0


def _u64(data: bytes, off: int) -> int:
    return struct.unpack_from("<Q", data, off)[0] if off + 8 <= len(data) else 0


def _parse_pe(data: bytes, path: str, notes: List[str]) -> ContainerInfo:
    """PE/PE32+ parser (stdlib only, tolerant of sloppy real-world headers)."""
    if data[:2] != b"MZ":
        raise ContainerParseError("not a PE image (missing MZ)")
    e_lfanew = _u32(data, 0x3C)
    if e_lfanew + 24 > len(data) or data[e_lfanew:e_lfanew + 4] != b"PE\x00\x00":
        raise ContainerParseError("not a PE image (missing PE signature)")

    machine = _u16(data, e_lfanew + 4)
    section_count = _u16(data, e_lfanew + 6)
    size_optional = _u16(data, e_lfanew + 20)
    opt_off = e_lfanew + 24
    magic = _u16(data, opt_off)

    if magic == 0x20B:  # PE32+
        fallback_optional, dd_off, image_base = 0xF0, 112, _u64(data, opt_off + 24)
        arch, bits = _PE_MACHINES.get(machine, ("x86_64", 8))
    elif magic == 0x10B:  # PE32
        fallback_optional, dd_off, image_base = 0xE0, 96, _u32(data, opt_off + 28)
        arch, bits = _PE_MACHINES.get(machine, ("x86", 4))
    elif magic == 0x107:  # ROM image
        fallback_optional, dd_off, image_base = 0xF0, 112, _u32(data, opt_off + 28)
        arch, bits = _PE_MACHINES.get(machine, ("mips", 4))
    else:
        raise ContainerParseError(f"unsupported PE optional-header magic 0x{magic:x}")

    if size_optional < 0x40 or opt_off + size_optional > len(data):
        notes.append(
            f"SizeOfOptionalHeader={size_optional} unusable; "
            f"using magic-derived size 0x{fallback_optional:x}"
        )
        size_optional = fallback_optional

    entry_rva = _u32(data, opt_off + 16)
    sec_off = opt_off + size_optional
    sections: List[Section] = []
    for i in range(section_count):
        base_off = sec_off + i * 40
        if base_off + 40 > len(data):
            notes.append(f"section header {i} truncated; stopping")
            break
        name = data[base_off:base_off + 8].split(b"\x00")[0].decode("latin-1", "replace")
        v_size = _u32(data, base_off + 8)
        v_addr = _u32(data, base_off + 12)
        raw_size = _u32(data, base_off + 16)
        raw_ptr = _u32(data, base_off + 20)
        sec_flags = _u32(data, base_off + 36)
        payload = data[raw_ptr:raw_ptr + raw_size] if raw_ptr < len(data) else b""
        sections.append(
            Section(
                name=name or f"sec{i}",
                vaddr=image_base + v_addr,
                vsize=max(v_size, len(payload)),
                data=payload,
                file_offset=raw_ptr,
                is_exec=bool(sec_flags & 0x20000000),
            )
        )

    symbols: Dict[int, str] = {}
    try:
        dir_count = _u32(data, opt_off + dd_off - 4)
        export_rva = _u32(data, opt_off + dd_off) if dir_count > 0 else 0
        if export_rva:
            symbols.update(_parse_pe_exports(data, export_rva, image_base, sections, notes))
    except (struct.error, IndexError):  # pragma: no cover - defensive
        notes.append("PE export directory parse failed")

    if not sections:
        raise ContainerParseError("PE image declares no usable sections")

    return ContainerInfo(
        path=path,
        container=ContainerType.PE.value,
        arch=arch,
        base=image_base,
        entry=image_base + entry_rva,
        pointer_size=bits,
        sections=sections,
        symbols=symbols,
        machine=f"0x{machine:04x}",
        notes=notes,
    )


def _rva_to_offset(rva: int, sections: Sequence[Section], base: int) -> Optional[int]:
    for sec in sections:
        if sec.vaddr <= base + rva < sec.vaddr + len(sec.data):
            return sec.file_offset + (base + rva - sec.vaddr)
    return None


def _parse_pe_exports(
    data: bytes,
    export_rva: int,
    image_base: int,
    sections: Sequence[Section],
    notes: List[str],
) -> Dict[int, str]:
    off = _rva_to_offset(export_rva, sections, image_base)
    if off is None or off + 40 > len(data):
        return {}
    name_count = _u32(data, off + 24)
    funcs_rva = _u32(data, off + 28)
    names_rva = _u32(data, off + 32)
    ordinals_rva = _u32(data, off + 36)
    if not (name_count and funcs_rva and names_rva):
        return {}
    names_off = _rva_to_offset(names_rva, sections, image_base)
    ords_off = _rva_to_offset(ordinals_rva, sections, image_base)
    funcs_off = _rva_to_offset(funcs_rva, sections, image_base)
    if names_off is None or ords_off is None or funcs_off is None:
        notes.append("PE export tables partially unmapped; symbols skipped")
        return {}
    out: Dict[int, str] = {}
    for i in range(min(name_count, 4096)):
        if names_off + i * 4 + 4 > len(data):
            break
        name_off = _rva_to_offset(_u32(data, names_off + i * 4), sections, image_base)
        if name_off is None:
            continue
        try:
            name = data[name_off:name_off + 256].split(b"\x00")[0].decode("ascii")
        except UnicodeDecodeError:
            continue
        ordinal = _u16(data, ords_off + i * 2)
        if funcs_off + ordinal * 4 + 4 > len(data):
            continue
        func_rva = _u32(data, funcs_off + ordinal * 4)
        if func_rva and name:
            out[image_base + func_rva] = name
    return out


def _parse_elf(data: bytes, path: str, notes: List[str]) -> ContainerInfo:
    """ELF parser built on pyelftools, with a program-header fallback."""
    try:
        from elftools.elf.elffile import ELFFile
        from elftools.elf.constants import SH_FLAGS
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise ContainerParseError(f"pyelftools unavailable: {exc}") from exc

    try:
        elf = ELFFile(BytesIO(data))
    except Exception as exc:
        raise ContainerParseError(f"ELF parse failed: {exc}") from exc

    machine = elf.header["e_machine"]
    bits = 8 if elf.elfclass == 64 else 4
    if isinstance(machine, int):
        e_machine = machine
    elif isinstance(machine, str):
        try:
            from elftools.elf.enums import ENUM_E_MACHINE

            e_machine = int(ENUM_E_MACHINE.get(machine, 0))
        except Exception:
            e_machine = 0
    else:
        e_machine = int(getattr(machine, "value", 0))
    arch, ptr = _ELF_MACHINES.get(e_machine, ("x86", bits))

    base = 0
    try:
        loads = [seg for seg in elf.iter_segments() if seg["p_type"] == "PT_LOAD"]
        if loads:
            base = min(int(seg["p_vaddr"]) for seg in loads) & ~0xFFF
    except Exception:  # pragma: no cover - defensive
        notes.append("ELF program headers unreadable; base assumed 0")

    sections: List[Section] = []
    try:
        for sec in elf.iter_sections():
            if not (int(sec["sh_flags"]) & SH_FLAGS.SHF_EXECINSTR):
                continue
            try:
                payload = sec.data()
            except Exception:  # SHT_NOBITS and friends
                continue
            sections.append(
                Section(
                    name=sec.name or f"sec{len(sections)}",
                    vaddr=int(sec["sh_addr"]) or base,
                    vsize=int(sec["sh_size"]),
                    data=payload,
                    file_offset=int(sec["sh_offset"]),
                    is_exec=True,
                )
            )
    except Exception as exc:  # pragma: no cover - defensive
        notes.append(f"ELF section headers unreadable ({exc}); using program headers")

    if not sections:
        try:
            for seg in elf.iter_segments():
                if seg["p_type"] != "PT_LOAD" or not (int(seg["p_flags"]) & 0x1):
                    continue
                sections.append(
                    Section(
                        name=f"seg{len(sections)}",
                        vaddr=int(seg["p_vaddr"]),
                        vsize=int(seg["p_memsz"]),
                        data=seg.data(),
                        file_offset=int(seg["p_offset"]),
                        is_exec=True,
                    )
                )
            if sections:
                notes.append("no SHF_EXECINSTR sections; fell back to PT_LOAD(PF_X)")
        except Exception as exc:
            raise ContainerParseError(f"ELF has no executable content: {exc}") from exc

    symbols: Dict[int, str] = {}
    try:
        symtab = elf.get_section_by_name(".symtab") or elf.get_section_by_name(".dynsym")
        if symtab is not None:
            for sym in symtab.iter_symbols():
                addr = int(sym["st_value"])
                if sym["st_info"]["type"] == "STT_FUNC" and addr and sym.name:
                    symbols[addr] = sym.name
    except Exception:  # pragma: no cover - stripped/odd binaries
        notes.append("ELF symbol table unreadable; symbol-based starts skipped")

    if not sections:
        raise ContainerParseError("ELF contains no executable sections")

    return ContainerInfo(
        path=path,
        container=ContainerType.ELF.value,
        arch=arch,
        base=base,
        entry=int(elf.header["e_entry"]),
        pointer_size=ptr,
        sections=sections,
        symbols=symbols,
        machine=str(machine),
        notes=notes,
    )


def _parse_nso(data: bytes, path: str, notes: List[str]) -> ContainerInfo:
    """Switch NSO (``NSO0``) parser with LZ4 decompression support."""
    notes.append("NSO container parsed with best-effort header layout")
    flags = _u32(data, 0x0C)
    text_decomp_size = _u32(data, 0x14)
    table_off = 0x40
    text_file_off = _u32(data, table_off)
    text_file_size = _u32(data, table_off + 8) or _u32(data, 0x14) or _u32(data, 0x1C)
    if not (0 < text_file_off < len(data)):
        text_file_off = 0x100
    raw_payload = data[text_file_off:min(len(data), text_file_off + max(text_file_size, 0))]

    if flags & 0x1:
        try:
            import lz4.block
            payload = lz4.block.decompress(raw_payload, uncompressed_size=text_decomp_size or (len(raw_payload) * 4))
            notes.append("NSO .text successfully decompressed with LZ4")
        except Exception as exc:
            notes.append(f"NSO LZ4 decompression failed: {exc}; using raw bytes")
            payload = raw_payload
    else:
        payload = raw_payload

    return ContainerInfo(
        path=path,
        container=ContainerType.NSO.value,
        arch="arm64",
        base=0,
        entry=0,
        pointer_size=8,
        sections=[
            Section(
                name=".nso.text",
                vaddr=0,
                vsize=max(text_decomp_size or text_file_size, len(payload)),
                data=payload,
                file_offset=text_file_off,
                is_exec=True,
            )
        ],
        symbols={},
        machine="NSO0 (Switch)",
        notes=notes,
    )


def _parse_nro(data: bytes, path: str, notes: List[str]) -> ContainerInfo:
    """Best-effort Switch NRO (``NRO0``) parser.

    ``NRO0`` sits at file offset 0x10; the executable ``.text`` normally starts
    at file offset 0x80, with sizes in the segment table at 0x20+.
    """
    notes.append("NRO container parsed with best-effort header layout")
    fixed_size = _u32(data, 0x18)
    text_off = _u32(data, 0x20)
    text_size = _u32(data, 0x24)
    if not (0 < text_off < len(data)) or text_size == 0:
        text_off = 0x80
        text_size = max(len(data) - text_off, 0)
        notes.append("NRO segment table unusable; assuming .text at file offset 0x80")
    payload = data[text_off:min(len(data), text_off + text_size)]
    return ContainerInfo(
        path=path,
        container=ContainerType.NRO.value,
        arch="arm64",
        base=0,
        entry=0,
        pointer_size=8,
        sections=[
            Section(
                name=".nro.text",
                vaddr=0,
                vsize=max(text_size, len(payload)),
                data=payload,
                file_offset=text_off,
                is_exec=True,
            )
        ],
        symbols={},
        machine=f"NRO0 (Switch, size={fixed_size})",
        notes=notes,
    )


def _parse_self(data: bytes, path: str, notes: List[str]) -> ContainerInfo:
    """PlayStation SELF container parser (extracts wrapped ELF payload)."""
    notes.append("PlayStation SELF container")
    elf_off = data.find(b"\x7fELF")
    if elf_off == -1:
        raise ContainerParseError("PlayStation SELF does not contain an unencrypted ELF payload")
    info = _parse_elf(data[elf_off:], path, notes)
    info.container = ContainerType.SELF.value
    info.machine = f"SELF ({info.machine})"
    return info


def _parse_dol(data: bytes, path: str, notes: List[str]) -> ContainerInfo:
    """GameCube / Wii DOL container parser."""
    notes.append("GameCube/Wii DOL container")
    if len(data) < 0x100:
        raise ContainerParseError("DOL image too small for header")
    text_offsets = [_u32_be(data, i * 4) for i in range(7)]
    data_offsets = [_u32_be(data, 0x1C + i * 4) for i in range(11)]
    text_addrs = [_u32_be(data, 0x48 + i * 4) for i in range(7)]
    data_addrs = [_u32_be(data, 0x64 + i * 4) for i in range(11)]
    text_sizes = [_u32_be(data, 0x90 + i * 4) for i in range(7)]
    data_sizes = [_u32_be(data, 0xAC + i * 4) for i in range(11)]
    entry = _u32_be(data, 0xE0)

    sections: List[Section] = []
    for i in range(7):
        off, addr, sz = text_offsets[i], text_addrs[i], text_sizes[i]
        if off and sz and off + sz <= len(data):
            sections.append(
                Section(
                    name=f".text{i}",
                    vaddr=addr,
                    vsize=sz,
                    data=data[off:off + sz],
                    file_offset=off,
                    is_exec=True,
                )
            )
    for i in range(11):
        off, addr, sz = data_offsets[i], data_addrs[i], data_sizes[i]
        if off and sz and off + sz <= len(data):
            sections.append(
                Section(
                    name=f".data{i}",
                    vaddr=addr,
                    vsize=sz,
                    data=data[off:off + sz],
                    file_offset=off,
                    is_exec=False,
                )
            )
    if not sections:
        raise ContainerParseError("DOL image declared no usable sections")

    base = min(s.vaddr for s in sections)
    return ContainerInfo(
        path=path,
        container=ContainerType.DOL.value,
        arch="ppc",
        base=base,
        entry=entry,
        pointer_size=4,
        sections=sections,
        symbols={},
        machine="PowerPC (GameCube/Wii DOL)",
        notes=notes,
    )


def load_container(
    target: Union[str, Path],
    *,
    is_raw: bool = False,
    raw_arch: Optional[str] = None,
    raw_base: Optional[int] = None,
) -> ContainerInfo:
    """Read ``target`` and return a parsed :class:`ContainerInfo`."""
    path = Path(target)
    if not path.exists():
        raise ContainerParseError(f"target not found: {path}")
    data = path.read_bytes()
    if not data:
        raise ContainerParseError(f"target is empty: {path}")
    sha = hashlib.sha256(data).hexdigest()
    notes: List[str] = []

    if is_raw:
        if not raw_arch:
            raise UnsupportedArchitectureError(
                f"raw targets require --arch (one of {list(SUPPORTED_ARCHES)})"
            )
        arch, bits = normalize_arch_name(raw_arch)
        base = int(raw_base or 0)
        return ContainerInfo(
            path=str(path),
            container=ContainerType.RAW.value,
            arch=arch,
            base=base,
            entry=base,
            pointer_size=bits,
            sections=[
                Section(
                    name=".raw",
                    vaddr=base,
                    vsize=len(data),
                    data=data,
                    file_offset=0,
                    is_exec=True,
                )
            ],
            symbols={},
            machine="raw",
            sha256=sha,
            is_raw=True,
            notes=["raw blob mapped as a single executable region"],
        )

    kind = detect_container(data)
    if kind == ContainerType.PE.value:
        info = _parse_pe(data, str(path), notes)
    elif kind == ContainerType.ELF.value:
        info = _parse_elf(data, str(path), notes)
    elif kind == ContainerType.NSO.value:
        info = _parse_nso(data, str(path), notes)
    elif kind == ContainerType.NRO.value:
        info = _parse_nro(data, str(path), notes)
    elif kind == ContainerType.SELF.value:
        info = _parse_self(data, str(path), notes)
    elif kind == ContainerType.DOL.value:
        info = _parse_dol(data, str(path), notes)
    else:
        hint = (
            "; pass --raw --arch <arch> --base <addr> to ingest as a blob"
            if not raw_arch
            else ""
        )
        raise ContainerParseError(
            f"UNKNOWN_FORMAT_NEED_HINT: cannot parse {path} as PE/ELF/NSO/NRO/SELF/DOL{hint}"
        )
    info.sha256 = sha
    return info


# ---------------------------------------------------------------------------
# instruction / function / result model
# ---------------------------------------------------------------------------


@dataclass
class Instruction:
    """One decoded instruction plus lifting metadata."""

    address: int
    size: int
    mnemonic: str
    op_str: str
    opcode: bytes = b""
    reads: Tuple[str, ...] = ()
    writes: Tuple[str, ...] = ()
    immediates: Tuple[Tuple[int, int], ...] = ()
    is_call: bool = False
    is_jump: bool = False
    is_ret: bool = False
    is_int: bool = False
    is_uncond_jump: bool = False
    target: Optional[int] = None

    @property
    def end(self) -> int:
        return self.address + self.size

    @property
    def text(self) -> str:
        return f"{self.mnemonic} {self.op_str}".strip()

    def has_memory_operand(self) -> bool:
        return "[" in self.op_str or "(" in self.op_str


@dataclass
class FunctionAnalysis:
    """Analysis result for a single function."""

    addr: int
    name: str
    size: int
    instructions: List[Instruction] = field(default_factory=list)
    call_targets: List[int] = field(default_factory=list)
    intra_jump_targets: List[int] = field(default_factory=list)
    external_jump_targets: List[int] = field(default_factory=list)
    cfg_edges: List[Tuple[int, int]] = field(default_factory=list)
    next_start: Optional[int] = None
    ir_pcode: str = ""
    decompiled_c: Optional[str] = None
    removed_nops: int = 0
    removed_dead_assignments: int = 0
    normalized_negative_literals: int = 0
    truncated: bool = False

    @property
    def instruction_count(self) -> int:
        return len(self.instructions)


@dataclass
class AnalysisResult:
    """Summary of one analysis run (JSON-ready through :meth:`to_dict`)."""

    target: str
    container: str
    arch: str
    base: int
    entry_point: int
    binary_sha256: str
    engine: str = "capstone"
    function_count: int = 0
    edge_count: int = 0
    call_edge_count: int = 0
    cfg_edge_count: int = 0
    xref_edge_count: int = 0
    instruction_count: int = 0
    ssa_fact_count: int = 0
    removed_nops: int = 0
    removed_dead_assignments: int = 0
    normalized_negative_literals: int = 0
    unmapped_branch_targets: int = 0
    ghidra_available: bool = False
    ghidra_used: bool = False
    static_only: bool = False
    elapsed_seconds: float = 0.0
    binary_metadata: Dict[str, Any] = field(default_factory=dict)
    db_stats: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target,
            "container": self.container,
            "arch": self.arch,
            "base": self.base,
            "entry_point": self.entry_point,
            "binary_sha256": self.binary_sha256,
            "engine": self.engine,
            "function_count": self.function_count,
            "edge_count": self.edge_count,
            "call_edge_count": self.call_edge_count,
            "cfg_edge_count": self.cfg_edge_count,
            "xref_edge_count": self.xref_edge_count,
            "instruction_count": self.instruction_count,
            "ssa_fact_count": self.ssa_fact_count,
            "normalization": {
                "removed_nops": self.removed_nops,
                "removed_dead_assignments": self.removed_dead_assignments,
                "normalized_negative_literals": self.normalized_negative_literals,
            },
            "unmapped_branch_targets": self.unmapped_branch_targets,
            "engine_availability": {
                "ghidra_available": self.ghidra_available,
                "ghidra_used": self.ghidra_used,
                "static_only": self.static_only,
            },
            "elapsed_seconds": round(self.elapsed_seconds, 4),
            "binary": self.binary_metadata,
            "db": self.db_stats,
            "notes": list(self.notes),
        }

    def to_json(self, indent: Optional[int] = 2) -> str:
        if indent is None:
            return json.dumps(self.to_dict(), separators=(",", ":"))
        return json.dumps(self.to_dict(), indent=indent)


@dataclass
class _ScanResult:
    functions: List[FunctionAnalysis] = field(default_factory=list)
    edges: List[Tuple[int, int, str]] = field(default_factory=list)
    ssa_facts: List[SsaFact] = field(default_factory=list)
    instruction_count: int = 0
    unmapped_branch_targets: int = 0
    notes: List[str] = field(default_factory=list)


@runtime_checkable
class StaticEngineProtocol(Protocol):
    """Interface every static head must satisfy (Capstone head, Ghidra head...)."""

    def analyze_binary(
        self,
        target: Union[str, Path],
        is_raw: bool = False,
        raw_arch: Optional[str] = None,
        raw_base: Optional[int] = None,
        static_only: bool = False,
    ) -> Dict[str, Any]:
        """Analyse ``target`` and return the JSON-ready summary."""
        ...


# ---------------------------------------------------------------------------
# optional PyGhidra bridge
# ---------------------------------------------------------------------------


def ghidra_available() -> bool:
    """True when PyGhidra *and* a Ghidra install look usable.

    Importing ``pyghidra`` alone is not enough (launching a project needs a real
    install directory), so this check is env/path based and never starts a JVM.
    """
    try:
        import pyghidra  # noqa: F401
    except Exception:
        return False
    for var in ("GHIDRA_INSTALL_DIR", "GHIDRA_HOME", "GHIDRA_DIR"):
        value = os.environ.get(var)
        if value and Path(value).exists():
            return True
    for candidate in (Path("C:/tools/ghidra"), Path("C:/tools/ghidra_12.1.4_PUBLIC")):
        if candidate.exists() and (candidate / "ghidraRun.bat").exists():
            os.environ["GHIDRA_INSTALL_DIR"] = str(candidate)
            return True
    return False


# ---------------------------------------------------------------------------
# engine
# ---------------------------------------------------------------------------


class StaticEngine:
    """Capstone-backed multi-architecture static analysis engine.

    Parameters
    ----------
    db_path:
        SQLite graph database to ingest into (created when missing).
    db:
        Pre-built :class:`~toothpick.storage.Database` (overrides ``db_path``).
    prefer_ghidra:
        Attempt the PyGhidra bridge for decompiled C when a Ghidra install is
        present; silently degrades to Capstone-only results otherwise.
    store_ssa_facts:
        Emit per-register def/use facts into ``ssa_facts``.
    """

    ENGINE_NAME = "capstone"

    def __init__(
        self,
        db_path: PathLike = "toothpick.db",
        *,
        db: Optional[Database] = None,
        prefer_ghidra: bool = False,
        static_only: bool = True,
        store_ssa_facts: bool = True,
        max_function_size: int = 1 << 16,
        max_instructions_per_function: int = 4096,
        max_functions: int = 200_000,
        max_ssa_facts_per_binary: int = 500_000,
        max_padding_bytes: int = 1 << 14,
        max_resync_attempts: int = 8,
        chunk_size: int = 256,
    ) -> None:
        self.db_path = Path(db_path)
        self._db = db if db is not None else Database(self.db_path)
        self.prefer_ghidra = prefer_ghidra
        self.static_only = static_only
        self.store_ssa_facts = store_ssa_facts
        self.max_function_size = max_function_size
        self.max_instructions_per_function = max_instructions_per_function
        self.max_functions = max_functions
        self.max_ssa_facts_per_binary = max_ssa_facts_per_binary
        self.max_padding_bytes = max_padding_bytes
        self.max_resync_attempts = max_resync_attempts
        self.chunk_size = max(1, chunk_size)
        self._last_container: Optional[ContainerInfo] = None

    # -- public API --------------------------------------------------------

    def analyze_binary(
        self,
        target: Union[str, Path],
        is_raw: bool = False,
        raw_arch: Optional[str] = None,
        raw_base: Optional[int] = None,
        static_only: bool = False,
        write_db: bool = True,
        prefer_ghidra: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Analyse ``target`` and return the JSON-ready summary dict."""
        result = self.analyze(
            target,
            is_raw=is_raw,
            raw_arch=raw_arch,
            raw_base=raw_base,
            static_only=static_only,
            write_db=write_db,
            prefer_ghidra=prefer_ghidra,
        )
        return result.to_dict()

    def analyze(
        self,
        target: Union[str, Path],
        *,
        is_raw: bool = False,
        raw_arch: Optional[str] = None,
        raw_base: Optional[int] = None,
        static_only: bool = False,
        write_db: bool = True,
        prefer_ghidra: Optional[bool] = None,
    ) -> AnalysisResult:
        """Full pass: parse -> disassemble -> normalize -> lift -> ingest."""
        started = time.perf_counter()
        container = load_container(
            target, is_raw=is_raw, raw_arch=raw_arch, raw_base=raw_base
        )
        self._last_container = container
        self._validate_entry(container)

        md = self._disassembler(container)
        scan = self._scan(container, md)

        want_ghidra = self.prefer_ghidra if prefer_ghidra is None else prefer_ghidra
        gh_ok = ghidra_available()
        gh_used = False
        notes = list(container.notes) + list(scan.notes)
        if want_ghidra and gh_ok:
            try:
                gh_used = self._ghidra_enrich(container, scan.functions)
            except GhidraUnavailableError as exc:
                notes.append(f"pyghidra bridge unavailable: {exc}")
        elif want_ghidra and not gh_ok:
            notes.append(
                "ghidra headless unavailable (no GHIDRA_INSTALL_DIR); "
                "capstone-only results"
            )

        counts: Dict[str, int] = {"call": 0, "cfg": 0, "xref": 0}
        for _src, _dst, kind in scan.edges:
            counts[kind] = counts.get(kind, 0) + 1

        result = AnalysisResult(
            target=container.path,
            container=container.container,
            arch=container.arch,
            base=container.base,
            entry_point=container.entry,
            binary_sha256=container.sha256,
            engine=self.ENGINE_NAME + ("+ghidra" if gh_used else ""),
            function_count=len(scan.functions),
            edge_count=len(scan.edges),
            call_edge_count=counts.get("call", 0),
            cfg_edge_count=counts.get("cfg", 0),
            xref_edge_count=counts.get("xref", 0),
            instruction_count=scan.instruction_count,
            ssa_fact_count=len(scan.ssa_facts),
            removed_nops=sum(f.removed_nops for f in scan.functions),
            removed_dead_assignments=sum(
                f.removed_dead_assignments for f in scan.functions
            ),
            normalized_negative_literals=sum(
                f.normalized_negative_literals for f in scan.functions
            ),
            unmapped_branch_targets=scan.unmapped_branch_targets,
            ghidra_available=gh_ok,
            ghidra_used=gh_used,
            static_only=bool(static_only) or self.static_only,
            binary_metadata=self._binary_metadata(container),
            notes=notes,
        )

        if write_db:
            self._ingest(self._db, container, scan, result)
            result.db_stats = self._db.stats(container.sha256)

        result.elapsed_seconds = time.perf_counter() - started
        return result

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> "StaticEngine":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    # -- container helpers -------------------------------------------------

    def _validate_entry(self, container: ContainerInfo) -> None:
        if container.entry and container.exec_section_with_bytes(container.entry) is None:
            container.notes.append(
                f"entry point 0x{container.entry:x} is not inside executable bytes"
            )

    def _binary_metadata(self, container: ContainerInfo) -> Dict[str, Any]:
        return {
            "container": container.container,
            "arch": container.arch,
            "pointer_size": container.pointer_size,
            "base": container.base,
            "entry_point": container.entry,
            "machine": container.machine,
            "is_raw": container.is_raw,
            "section_count": len(container.sections),
            "symbol_count": len(container.symbols),
        }

    def _disassembler(self, container: ContainerInfo) -> Cs:
        spec = container.spec
        try:
            md = Cs(spec.capstone_arch, spec.capstone_mode)
        except Exception as exc:  # pragma: no cover - capstone build issue
            raise UnsupportedArchitectureError(
                f"cannot create capstone disassembler for {spec.arch}: {exc}"
            ) from exc
        md.detail = True
        return md

    @staticmethod
    def _reg_name(md: Cs, reg_id: int) -> str:
        try:
            return (md.reg_name(reg_id) or "").lower()
        except Exception:  # pragma: no cover - defensive
            return ""

    # -- decoding ----------------------------------------------------------

    def _decode(self, md: Cs, data: bytes, addr: int) -> Optional[Instruction]:
        if not data:
            return None
        chunk = data[:16] if len(data) > 16 else data
        try:
            insn = next(md.disasm(chunk, addr, count=1))
        except StopIteration:
            return None
        except capstone.CsError:  # pragma: no cover - defensive
            return None
        return self._to_instruction(md, insn)

    def _to_instruction(self, md: Cs, insn: Any) -> Instruction:
        arch = self._last_container.arch if self._last_container else ""
        mnemonic = (insn.mnemonic or "").lower()
        try:
            reads_ids, writes_ids = insn.regs_access()
        except Exception:  # pragma: no cover - arch/detail combinations
            reads_ids, writes_ids = (), ()
        reads = tuple(dict.fromkeys(self._reg_name(md, r) for r in reads_ids))
        writes = tuple(dict.fromkeys(self._reg_name(md, w) for w in writes_ids))
        groups = set(insn.groups or ())

        is_call = bool(
            capstone.CS_GRP_CALL in groups
            or mnemonic in _ARCH_CALL_MNEMONICS.get(arch, set())
        )
        is_ret = bool(
            capstone.CS_GRP_RET in groups
            or capstone.CS_GRP_IRET in groups
            or mnemonic in _ARCH_RET_MNEMONICS.get(arch, set())
        )
        is_int = bool(
            capstone.CS_GRP_INT in groups or mnemonic in {"int3", "hlt", "ud2", "brk"}
        )
        is_jump = bool(
            capstone.CS_GRP_JUMP in groups or mnemonic in _ARCH_UNCOND_JUMPS.get(arch, set())
        )

        # ``jr $ra`` is a return; ``jr <reg>`` is an indirect jump
        if arch == "mips" and mnemonic == "jr":
            is_ret = "$ra" in (insn.op_str or "")
            is_jump = not is_ret
        # ARM64 ``blr`` is a return even when capstone tags it as a call
        if arch == "arm64" and mnemonic in ("blr", "ret"):
            is_ret, is_call = True, False
        if is_ret:
            is_call = False

        is_uncond = bool(mnemonic in _ARCH_UNCOND_JUMPS.get(arch, set()) and not is_call)

        target: Optional[int] = None
        immediates: List[Tuple[int, int]] = []
        for op in insn.operands or ():
            if op.type == _OP_IMM:
                width = int(getattr(op, "size", 0) or 0)
                immediates.append((int(op.imm), width))
        if (is_call or is_jump) and immediates:
            target = immediates[0][0]
        elif (is_call or is_jump) and not immediates:
            for op in insn.operands or ():
                if op.type == getattr(capstone, "CS_OP_MEM", 3):
                    mem = op.mem
                    base_reg = self._reg_name(md, mem.base) if mem.base else ""
                    if base_reg in ("rip", "ip"):
                        mem_target = int(insn.address + insn.size + mem.disp)
                    elif mem.disp and not mem.base and not mem.index:
                        mem_target = int(mem.disp)
                    else:
                        mem_target = None
                    if mem_target is not None:
                        if self._last_container is not None and self._last_container.is_mapped(mem_target):
                            ptr_val = self._deref_pointer(self._last_container, mem_target)
                            if ptr_val and self._last_container.exec_section_with_bytes(ptr_val) is not None:
                                target = ptr_val
                            else:
                                target = mem_target
                        else:
                            target = mem_target
                        break

        return Instruction(
            address=int(insn.address),
            size=int(insn.size),
            mnemonic=mnemonic,
            op_str=insn.op_str or "",
            opcode=bytes(insn.bytes or b""),
            reads=reads,
            writes=writes,
            immediates=tuple(immediates),
            is_call=is_call,
            is_jump=is_jump,
            is_ret=is_ret,
            is_int=is_int,
            is_uncond_jump=is_uncond,
            target=target,
        )

    @staticmethod
    def _deref_pointer(container: ContainerInfo, addr: int) -> Optional[int]:
        for sec in container.sections:
            if sec.has_bytes(addr):
                off = addr - sec.vaddr
                if container.pointer_size == 8 and off + 8 <= len(sec.data):
                    return struct.unpack_from("<Q", sec.data, off)[0]
                elif container.pointer_size == 4 and off + 4 <= len(sec.data):
                    return struct.unpack_from("<I", sec.data, off)[0]
        return None

    # -- scanning ----------------------------------------------------------

    def _scan(self, container: ContainerInfo, md: Cs) -> _ScanResult:
        exec_sections = container.exec_sections()
        if not exec_sections:
            return _ScanResult(notes=["no executable bytes found; nothing to analyse"])

        result = _ScanResult()
        pending: Deque[int] = deque()
        seen: Set[int] = set()
        edge_keys: Set[Tuple[int, int, str]] = set()

        def push(addr: int) -> None:
            if addr in seen or container.exec_section_with_bytes(addr) is None:
                return
            seen.add(addr)
            pending.append(addr)

        def add_edge(src: int, dst: int, kind: str) -> None:
            key = (src, dst, kind)
            if key not in edge_keys:
                edge_keys.add(key)
                result.edges.append(key)

        if container.entry:
            push(container.entry)
        for addr in sorted(container.symbols):
            push(addr)
        if not seen:
            push(exec_sections[0].vaddr)

        covered: List[FunctionAnalysis] = []
        while pending and len(covered) < self.max_functions:
            addr = pending.popleft()
            func = self._analyze_function(container, md, addr)
            if func is None or not func.instructions:
                continue
            covered.append(func)
            result.instruction_count += func.instruction_count

            for target in func.call_targets:
                if container.exec_section_with_bytes(target) is not None:
                    add_edge(func.addr, target, EdgeKind.CALL.value)
                    push(target)
                elif container.is_mapped(target):
                    add_edge(func.addr, target, EdgeKind.XREF.value)
                else:
                    result.unmapped_branch_targets += 1

            for src_site, dst_site, kind in func.cfg_edges:
                add_edge(src_site, dst_site, kind)
            if not func.cfg_edges:
                for target in func.intra_jump_targets:
                    add_edge(func.addr, target, EdgeKind.CFG.value)

            for target in func.external_jump_targets:
                if container.is_mapped(target):
                    add_edge(func.addr, target, EdgeKind.XREF.value)
                    if self._looks_like_prologue(container, md, target):
                        push(target)
                else:
                    result.unmapped_branch_targets += 1

            if func.next_start is not None:
                push(func.next_start)

        result.functions = covered
        if self.store_ssa_facts:
            result.ssa_facts = self._lift_ssa_facts(container, covered)
        if len(covered) >= self.max_functions:
            result.notes.append(
                f"function discovery stopped at max_functions={self.max_functions}"
            )
        return result

    def _analyze_function(
        self, container: ContainerInfo, md: Cs, start: int
    ) -> Optional[FunctionAnalysis]:
        section = container.exec_section_with_bytes(start)
        if section is None:
            return None
        base_data = section.bytes_at(start)
        if not base_data:
            return None

        limit = min(len(base_data), self.max_function_size)
        base_data = base_data[:limit + 16]

        func = FunctionAnalysis(
            addr=start,
            name=container.symbols.get(start) or f"sub_{start:x}",
            size=0,
        )
        pos = 0
        resync = 0
        call_targets: List[int] = []
        jump_sites: List[Tuple[int, int]] = []
        cfg_edges: List[Tuple[int, int, str]] = []
        max_intra_target = start

        while pos < limit and func.instruction_count < self.max_instructions_per_function:
            insn = self._decode(md, base_data[pos:pos + 16], start + pos)
            if insn is None:
                resync += 1
                if resync > self.max_resync_attempts:
                    break
                pos += 1
                continue
            resync = 0
            func.instructions.append(insn)
            pos += insn.size

            if insn.target is not None:
                if insn.is_call:
                    call_targets.append(insn.target)
                elif insn.is_jump:
                    jump_sites.append((insn.address, insn.target))
                    # Check if target is inside this function's reachable window
                    if start <= insn.target < start + limit:
                        max_intra_target = max(max_intra_target, insn.target)
                        cfg_edges.append((insn.address, insn.target, EdgeKind.CFG.value))
                        if not insn.is_uncond_jump and start + pos < start + limit:
                            # Conditional branch fallthrough edge
                            cfg_edges.append((insn.address, start + pos, EdgeKind.CFG.value))
                    else:
                        cfg_edges.append((insn.address, insn.target, EdgeKind.XREF.value))

            # Termination condition: true return, interrupt, or unconditional jump that exits function
            is_term = bool(insn.is_ret or insn.is_int or insn.is_uncond_jump)
            if is_term and (start + pos) > max_intra_target:
                func.next_start = self._skip_padding(container, section, md, start + pos)
                break

        if not func.instructions:
            return None

        func.size = func.instructions[-1].end - func.addr
        end = func.instructions[-1].end
        for site, target in jump_sites:
            if func.addr <= target < end:
                func.intra_jump_targets.append(target)
            else:
                func.external_jump_targets.append(target)
                if container.exec_section_with_bytes(target) is not None:
                    # External jump to executable code is a tail call
                    call_targets.append(target)
        func.call_targets = list(dict.fromkeys(call_targets))
        func.intra_jump_targets = list(dict.fromkeys(func.intra_jump_targets))
        func.external_jump_targets = list(dict.fromkeys(func.external_jump_targets))
        func.cfg_edges = list(dict.fromkeys(cfg_edges))

        self._normalize(func, container)
        func.ir_pcode = self._render_ir(func)
        if func.instruction_count >= self.max_instructions_per_function:
            func.truncated = True
        return func

    @staticmethod
    def _is_terminator(insn: Instruction) -> bool:
        return bool(insn.is_ret or insn.is_int or insn.is_uncond_jump)

    def _skip_padding(
        self, container: ContainerInfo, section: Section, md: Cs, addr: int
    ) -> Optional[int]:
        """Advance over NOP / zero / trap padding that follows a terminator."""
        data = section.data
        pos = addr - section.vaddr
        limit = min(len(data), pos + self.max_padding_bytes)
        while pos < limit:
            if data[pos] == 0:
                pos += 1
                continue
            insn = self._decode(md, data[pos:pos + 16], section.vaddr + pos)
            if insn is None or insn.mnemonic not in _PADDING_MNEMONICS:
                break
            pos += max(insn.size, 1)
        if pos >= len(data):
            return None
        return section.vaddr + pos

    def _looks_like_prologue(self, container: ContainerInfo, md: Cs, addr: int) -> bool:
        """Conservative promotion of external branch targets to function starts."""
        section = container.exec_section_with_bytes(addr)
        if section is None:
            return False
        data = section.bytes_at(addr)
        prologues = _ARCH_PROLOGUE_MNEMONICS.get(container.arch, set())
        for _ in range(3):
            insn = self._decode(md, data, addr)
            if insn is None:
                return False
            if insn.mnemonic in prologues or insn.is_ret or insn.is_int:
                return True
            data = data[insn.size:]
            if not data:
                break
        return False

    # -- normalization -----------------------------------------------------

    def _normalize(self, func: FunctionAnalysis, container: ContainerInfo) -> None:
        """Negative-literal normalization + conservative dead/NOP trimming."""
        spec = container.spec
        always_live = spec.always_live

        negative_literals = 0
        for insn in func.instructions:
            text, changed = self._normalize_negative_literals(insn)
            insn.op_str = text
            negative_literals += changed
        func.normalized_negative_literals = negative_literals

        live: Set[str] = set(always_live)
        remove: Set[int] = set()
        removed_nops = 0
        dead = 0
        for insn in reversed(func.instructions):
            if insn.mnemonic in _PADDING_MNEMONICS and not insn.op_str:
                remove.add(id(insn))
                removed_nops += 1
                continue
            reads = set(insn.reads)
            writes = set(insn.writes)
            barrier = (
                insn.is_call
                or insn.is_jump
                or insn.is_ret
                or insn.is_int
                or insn.has_memory_operand()
                or insn.mnemonic in _SIDE_EFFECT_MNEMONICS
            )
            if barrier:
                live = (live - writes) | reads | always_live
                continue
            if writes and not (writes & live):
                remove.add(id(insn))
                dead += 1
                continue
            live = (live - writes) | reads

        if remove:
            func.instructions = [i for i in func.instructions if id(i) not in remove]
        func.removed_nops = removed_nops
        func.removed_dead_assignments = dead

    def _normalize_negative_literals(self, insn: Instruction) -> Tuple[str, int]:
        """Rewrite unsigned ``0xffffffff``-style immediates as signed decimals.

        ``mov eaax, 0xffffffff`` becomes ``mov eax, -1``; a 64-bit
        ``0xffffffffffffffff`` becomes ``-1`` as well. Branch/call targets are
        left alone — an address is not a literal.
        """
        text = insn.op_str
        if not text or "0x" not in text or insn.is_call or insn.is_jump:
            return text, 0
        changed = 0
        for value, width in insn.immediates:
            if not width:
                continue
            mask = (1 << (8 * width)) - 1
            if value < 0:
                unsigned, signed = value & mask, value
            elif value & (1 << (8 * width - 1)):
                unsigned, signed = value, value - (1 << (8 * width))
            else:
                continue
            token = f"0x{unsigned:x}"
            idx = text.find(token)
            if idx < 0:
                continue
            text = text[:idx] + str(signed) + text[idx + len(token):]
            changed += 1
        return text, changed

    def _render_ir(self, func: FunctionAnalysis) -> str:
        """Normalized textual IR snapshot kept in ``funcs.ir_pcode``."""
        lines = [
            f"0x{insn.address:x}  {insn.mnemonic} {insn.op_str}".rstrip()
            for insn in func.instructions[:512]
        ]
        if len(func.instructions) > 512:
            lines.append(f"; ... {len(func.instructions) - 512} more instructions")
        return "\n".join(lines)

    # -- lifting (ssa facts) ----------------------------------------------

    def _lift_ssa_facts(
        self, container: ContainerInfo, functions: Sequence[FunctionAnalysis]
    ) -> List[SsaFact]:
        """Per-register def/use facts (a Chandra-style SSA sketch).

        Each register write becomes one fact with its *later* reads as use
        points; this is the fact shape ``ssa_facts`` is specified to hold, not a
        full SSA construction (no phi insertion — see the module honesty note).
        """
        facts: List[SsaFact] = []
        sha = container.sha256
        for func in functions:
            if len(facts) >= self.max_ssa_facts_per_binary:
                break
            # Group writes and reads by canonical register name to handle sub-registers
            reg_writes: Dict[str, List[int]] = {}
            reg_reads: Dict[str, List[int]] = {}
            for idx, insn in enumerate(func.instructions):
                for reg in insn.writes:
                    if reg:
                        canon = _canonical_reg(container.arch, reg)
                        reg_writes.setdefault(canon, []).append(idx)
                        if canon != reg:
                            reg_writes.setdefault(reg, []).append(idx)
                for reg in insn.reads:
                    if reg:
                        canon = _canonical_reg(container.arch, reg)
                        reg_reads.setdefault(canon, []).append(idx)
                        if canon != reg:
                            reg_reads.setdefault(reg, []).append(idx)

            total_len = len(func.instructions)
            for idx, insn in enumerate(func.instructions):
                for reg in insn.writes:
                    if not reg:
                        continue
                    canon = _canonical_reg(container.arch, reg)
                    writes = reg_writes.get(canon, ())
                    k_idx = bisect.bisect_right(writes, idx)
                    kill_limit = writes[k_idx] if k_idx < len(writes) else total_len
                    reads = reg_reads.get(canon, ())
                    u_start = bisect.bisect_right(reads, idx)
                    u_end = bisect.bisect_left(reads, kill_limit)
                    reaching_uses = tuple(dict.fromkeys(reads[u_start:u_end]))
                    facts.append(
                        SsaFact(
                            fact_id=f"{sha}:{func.addr:x}:{idx}:{reg}",
                            binary_sha256=sha,
                            func_addr=func.addr,
                            var_name=reg,
                            ssa_reg=f"{reg}#{idx}",
                            def_point=idx,
                            use_points=reaching_uses,
                        )
                    )
                    if len(facts) >= self.max_ssa_facts_per_binary:
                        break
        return facts

    # -- optional PyGhidra bridge -----------------------------------------

    def _ghidra_enrich(
        self, container: ContainerInfo, functions: Sequence[FunctionAnalysis]
    ) -> bool:
        """Attach decompiled C from PyGhidra when a Ghidra install is present."""
        try:
            import pyghidra
        except Exception as exc:
            raise GhidraUnavailableError(str(exc)) from exc
        try:
            with pyghidra.open_program(container.path, analyze=False) as flat_api:
                program = flat_api.getCurrentProgram()
                decompiler = flat_api.getDecompiler()
                if decompiler is None:
                    raise GhidraUnavailableError("Ghidra decompiler is not exposed")
                space = program.getAddressFactory().getDefaultAddressSpace()
                for func in functions:
                    try:
                        gfunc = flat_api.getFunctionAt(space.getAddress(func.addr))
                        if gfunc is None:
                            continue
                        func.decompiled_c = str(
                            decompiler.decompileFunction(gfunc, 30)
                            .getDecompiledFunction()
                            .getC()
                        )
                    except Exception:
                        continue
        except GhidraUnavailableError:
            raise
        except Exception as exc:
            raise GhidraUnavailableError(f"pyghidra run failed: {exc}") from exc
        return True

    # -- persistence -------------------------------------------------------

    def _ingest(
        self,
        db: Database,
        container: ContainerInfo,
        scan: _ScanResult,
        result: AnalysisResult,
    ) -> None:
        """Write functions/edges/ssa facts + receipts into ``toothpick.db``.

        Invalidate-then-rewrite is atomic (single transaction), so a replay of
        the same binary leaves exactly the new rows and never duplicates the
        old ones — the kill-resume contract of design-document section 5.
        """
        sha = container.sha256
        ssa_by_func: Dict[int, List[SsaFact]] = {}
        for fact in scan.ssa_facts:
            ssa_by_func.setdefault(fact.func_addr, []).append(fact)

        with db.transaction():
            invalidated = db.invalidate_binary(sha)
            db.set_binary_metadata(sha, self._binary_metadata(container))
            db.add_receipt(
                sha,
                container.entry or 0,
                "static_analysis_start",
                {
                    "invalidated": invalidated,
                    "container": container.container,
                    "arch": container.arch,
                },
            )

        def build_chunk(chunk: Sequence[FunctionAnalysis]) -> Tuple[
            List[FunctionRecord], List[EdgeRecord]
        ]:
            func_records: List[FunctionRecord] = []
            edge_records: List[EdgeRecord] = []
            for func in chunk:
                func_records.append(
                    FunctionRecord(
                        binary_sha256=sha,
                        func_addr=func.addr,
                        name=func.name,
                        size=func.size,
                        calling_convention=self._calling_convention(container),
                        prototype=f"{func.name}(void)",
                        ir_pcode=func.ir_pcode or None,
                        decompiled_c=func.decompiled_c,
                    )
                )
                for target in func.call_targets:
                    if container.exec_section_with_bytes(target) is not None:
                        edge_records.append(
                            EdgeRecord(sha, func.addr, target, EdgeKind.CALL)
                        )
                    elif container.is_mapped(target):
                        edge_records.append(
                            EdgeRecord(sha, func.addr, target, EdgeKind.XREF)
                        )
                for src_site, dst_site, kind in func.cfg_edges:
                    edge_records.append(EdgeRecord(sha, src_site, dst_site, kind))
                if not func.cfg_edges:
                    for target in func.intra_jump_targets:
                        edge_records.append(EdgeRecord(sha, func.addr, target, EdgeKind.CFG))
                for target in func.external_jump_targets:
                    if container.is_mapped(target):
                        edge_records.append(
                            EdgeRecord(sha, func.addr, target, EdgeKind.XREF)
                        )
            return func_records, edge_records

        chunk_index = 0
        for start in range(0, len(scan.functions), self.chunk_size):
            chunk = scan.functions[start:start + self.chunk_size]
            func_records, edge_records = build_chunk(chunk)
            with db.transaction():
                db.bulk_upsert_functions(func_records, phase="static_analysis")
                db.add_edges(edge_records)
                if self.store_ssa_facts:
                    for func in chunk:
                        facts = ssa_by_func.get(func.addr)
                        if facts:
                            db.add_ssa_facts(facts)
                db.add_receipt(
                    sha,
                    chunk[0].addr,
                    "static_analysis_batch",
                    {
                        "chunk_index": chunk_index,
                        "functions": len(chunk),
                        "edges": len(edge_records),
                        "first_func": f"0x{chunk[0].addr:x}",
                        "last_func": f"0x{chunk[-1].addr:x}",
                    },
                )
            chunk_index += 1

        with db.transaction():
            db.add_receipt(
                sha,
                container.entry or 0,
                "static_analysis",
                {
                    "status": "ok",
                    "functions": result.function_count,
                    "edges": result.edge_count,
                    "instructions": result.instruction_count,
                    "arch": container.arch,
                    "container": container.container,
                    "engine": result.engine,
                },
            )

    @staticmethod
    def _calling_convention(container: ContainerInfo) -> str:
        arch = container.arch
        is_pe = container.container == ContainerType.PE.value
        if arch == "x86_64":
            return "ms_x64" if is_pe else "sysv_amd64"
        if arch == "x86":
            return "stdcall" if is_pe else "cdecl"
        if arch == "arm64":
            return "aapcs64"
        if arch == "arm":
            return "aapcs"
        if arch == "ppc":
            return "ppc_eabi"
        if arch == "mips":
            return "mips_o32"
        return ""
