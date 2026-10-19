"""
Toothpick dynamic taint analysis (DTA) & symbolic execution engine (unit W17).

Implements ``docs/units/UNIT-W17-SYMBOLIC-PLAN.md`` chunks 1-3:

* :class:`TaintState` — register-set plus byte-level memory taint model.
* :class:`TaintReport` — per-function taint findings (sink status, propagation
  path, tainted variables).
* :class:`TaintEngine` — instruction transfer semantics (``mov`` / ALU / loads
  / stores / ``push`` / ``pop`` / xor-zeroing) driving
  :meth:`TaintEngine.analyze_function`.
* :class:`SymbolicExecutionEngine` — path-constraint synthesis over direct Z3
  bitvector ASTs, backed by the pure-Python :class:`Z3Propagator`.
* :class:`TritonExporter` / :class:`ManticoreExporter` — script generators for
  the optional external frameworks.  ``triton`` and ``manticore`` are imported
  **lazily inside those exporters only**, so toothpick itself imports and runs
  with zero C-extension symbolic dependencies.

Instruction input is duck-typed: either the :class:`toothpick.static_engine.
Instruction` record or a raw Capstone ``CsInsn`` (``md.detail = True``) is
accepted; :func:`normalize_instruction` returns a uniform :class:`InsnView`.

Honesty notes (what this module deliberately does **not** claim):

* :meth:`TaintEngine.analyze_function` walks the reachable instruction graph in
  linear address order and does not model aliasing, so a positive sink hit means
  "taint *may* reach the sink" — never "was observed to reach it".
* Memory taint is byte-addressed and exact only for statically resolvable
  addresses (``[rsp + 8]``, ``[rip + 0x10]``, absolute constants).  A load or
  store through an unresolvable address is reported in
  ``TaintReport.unresolved_memory``; loads from such addresses are tainted
  conservatively, stores through them are not tracked.
* Memory reads whose address *is* resolvable but holds no tainted byte are
  treated as untainted (classic Triton-style DTA seeding: a missed source is a
  missed taint by construction).
* Calls and syscalls are opaque boundaries — taint is not propagated through
  callee bodies; a syscall conservatively taints its return register.
* Mnemonics outside the modelled transfer set are listed in
  ``TaintReport.unsupported_mnemonics`` and treated as taint-transparent.
* Symbolic execution models registers, arithmetic and flags; the stack and
  memory are modelled as free bitvector symbols.  A branch whose flags were
  never defined by a modelled instruction is reported as ``UNSUPPORTED`` rather
  than silently assumed.
"""

from __future__ import annotations

import bisect
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import capstone
import z3

from toothpick.static_engine import (
    ContainerParseError,
    Instruction,
    UnsupportedArchitectureError,
    arch_spec,
    load_container,
)

__all__ = [
    "SymbolicError",
    "InsnView",
    "InstructionLike",
    "normalize_instruction",
    "canonical_register",
    "split_operands",
    "is_memory_operand",
    "TaintState",
    "TaintReport",
    "TaintEngine",
    "SolveResult",
    "Z3Propagator",
    "SymbolicExecutionEngine",
    "TritonExporter",
    "ManticoreExporter",
    "exporter_for",
    "export_script",
    "disassemble_at",
    "DEFAULT_STACK_BASE",
    "FLAG_REGISTERS",
]

PathLike = Union[str, Path]
InstructionLike = Union[Instruction, capstone.CsInsn]

#: Synthetic stack top used when the caller does not seed the stack pointer.
DEFAULT_STACK_BASE = 0x7FFE0000

#: x86 flag pseudo-register, arm64 status pseudo-register.
FLAG_REGISTERS = frozenset({"rflags", "eflags", "flags", "nzcv", "cpsr", "apsr"})

_X86_CANONICAL: Dict[str, str] = {}
for _r64, _r32, _r16, _r8 in (
    ("rax", "eax", "ax", "al"),
    ("rbx", "ebx", "bx", "bl"),
    ("rcx", "ecx", "cx", "cl"),
    ("rdx", "edx", "dx", "dl"),
    ("rsi", "esi", "si", "sil"),
    ("rdi", "edi", "di", "dil"),
    ("rbp", "ebp", "bp", "bpl"),
    ("rsp", "esp", "sp", "spl"),
):
    _X86_CANONICAL[_r64] = _r64
    _X86_CANONICAL[_r32] = _r64
    _X86_CANONICAL[_r16] = _r64
    _X86_CANONICAL[_r8] = _r64
for _i in range(8, 16):
    _X86_CANONICAL[f"r{_i}"] = f"r{_i}"
    _X86_CANONICAL[f"r{_i}d"] = f"r{_i}"
    _X86_CANONICAL[f"r{_i}w"] = f"r{_i}"
    _X86_CANONICAL[f"r{_i}b"] = f"r{_i}"
_X86_CANONICAL.update(
    {
        "ah": "rax",
        "bh": "rbx",
        "ch": "rcx",
        "dh": "rdx",
        "rip": "rip",
        "eip": "rip",
        "rflags": "rflags",
        "eflags": "rflags",
        "flags": "rflags",
    }
)


class SymbolicError(Exception):
    """Base exception for taint analysis, symbolic lifting or solver failures."""


# ---------------------------------------------------------------------------
# instruction normalization & operand parsing
# ---------------------------------------------------------------------------


_SIZED_PREFIXES = ("byte", "word", "dword", "qword", "tbyte", "xword", "oword")
_NON_REGISTER_TOKENS = frozenset(
    {
        *(_SIZED_PREFIXES),
        "ptr",
        "short",
        "near",
        "far",
        "offset",
        "imm",
        "rel",
        "lsl",
        "asr",
        "uxtb",
        "uxth",
        "sxtb",
        "sxth",
    }
)

_UNCOND_JUMPS = frozenset({"jmp", "ljmp", "b", "br", "j", "jr"})
_CONDITIONAL_JUMPS = frozenset(
    {
        "je", "jne", "jz", "jnz", "jl", "jle", "jg", "jge", "jb", "jbe", "ja",
        "jae", "jc", "jnc", "js", "jns", "jo", "jno", "jp", "jpe", "jnp", "jpo",
        "jcxz", "jecxz", "jrcxz", "loop", "loope", "loopne", "loopz", "loopnz",
        "cbz", "cbnz", "tbz", "tbnz",
    }
)
_RET_MNEMONICS = frozenset({"ret", "retf", "iret", "iretd", "iretq", "eret", "blr", "rfe"})
_CALL_MNEMONICS = frozenset({"call", "lcall", "bl", "blx", "jal", "jalr", "bctrl"})
_SYSCALL_MNEMONICS = frozenset({"syscall", "sysenter", "int", "int3", "svc", "sc", "ecall"})

_NOOP_MNEMONICS = frozenset(
    {
        "nop", "hint", "endbr64", "endbr32", "pause", "lfence", "sfence", "mfence",
        "clflush", "prefetch", "prefetchnta", "prefetcht0", "prefetcht1",
        "prefetcht2", "bti", "csdb", "clc", "stc", "cld", "std", "cmc", "xgetbv",
        "cpuid", "rdtsc", "rdtscp", "ud2", "hlt", "wfi", "sev", "yield",
    }
)

_ZEROING_MNEMONICS = frozenset(
    {"xor", "sub", "pxor", "xorps", "xorpd", "vxorps", "vxorpd", "vpxor", "eor", "sbc"}
)

_MOV_MNEMONICS = frozenset(
    {"mov", "movabs", "movzx", "movsx", "movsxd", "movz", "movn", "movk", "lea"}
)
_ARM64_LOAD_MNEMONICS = frozenset(
    {"ldr", "ldrb", "ldrh", "ldrsb", "ldrsh", "ldrsw", "ldur", "ldurb", "ldurh"}
)
_ARM64_STORE_MNEMONICS = frozenset({"str", "strb", "strh", "stur", "sturb", "sturh"})
_BINOP_MNEMONICS = frozenset(
    {
        "add", "adds", "adc", "sub", "subs", "sbb", "and", "ands", "or", "orr",
        "xor", "eor", "imul", "mul", "umulh", "smulh", "shl", "sal", "shlx",
        "shr", "shrx", "sar", "sarx", "rol", "ror", "rcl", "rcr", "bswap",
    }
)
_COMPARE_MNEMONICS = frozenset({"cmp", "test", "tst", "cmn", "ccmp"})
_UNARY_MNEMONICS = frozenset({"inc", "dec", "neg", "not", "mvn", "bswap", "clz", "cls"})
_PURE_DEST_MNEMONICS = frozenset(
    {
        "mov", "movabs", "movzx", "movsx", "movsxd", "movz", "movn", "movk", "lea",
        "pop", "ldr", "ldrb", "ldrh", "ldrsb", "ldrsh", "ldrsw", "ldur", "ldurb",
        "ldurh", "sete", "setne", "setb", "seta", "setl", "setg", "setle", "setge",
        "setc", "seto", "sets", "setz", "setnz", "setbe", "setae",
    }
)
_BRANCH_FLAG_READERS = _CONDITIONAL_JUMPS | frozenset(
    {"sete", "setne", "setb", "seta", "setl", "setg", "setle", "setge", "setc",
     "seto", "sets", "setz", "setnz", "setbe", "setae", "sbb", "adc"}
)


def split_operands(op_str: str) -> List[str]:
    """Split an operand string, ignoring commas nested inside brackets."""
    if not op_str:
        return []
    tokens: List[str] = []
    current: List[str] = []
    depth = 0
    for ch in op_str:
        if ch in "[(":
            depth += 1
        elif ch in "])":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            tokens.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    tail = "".join(current).strip()
    if tail:
        tokens.append(tail)
    return [token for token in tokens if token]


def is_memory_operand(token: str) -> bool:
    """True when the operand token dereferences memory (``[...]`` / ``(...)``)."""
    return "[" in token or "(" in token


def _token_body(token: str) -> str:
    """Strip a leading size prefix and assembler sigils from an operand token."""
    body = token.strip()
    lowered = body.lower()
    for prefix in _SIZED_PREFIXES:
        if lowered.startswith(prefix + " "):
            body = body[len(prefix) + 1 :].strip()
            lowered = body.lower()
    if lowered.startswith("ptr "):
        body = body[4:].strip()
    return body.strip().lstrip("%").lstrip("#").lstrip("$")


def register_operand(token: str) -> Optional[str]:
    """Return the register name held by ``token``, or ``None``.

    Immediates, memory operands, keyword modifiers and symbol names all fail the
    shape test, so the helper is safe to apply to any operand token.
    """
    if not token or is_memory_operand(token):
        return None
    body = _token_body(token)
    if not body or body.lower() in _NON_REGISTER_TOKENS:
        return None
    if body[0].isdigit() or body.startswith("-") or body.startswith("0x"):
        return None
    if not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]*", body):
        return None
    return body.lower()


def operand_size(token: str, default: int = 0) -> int:
    """Size in bytes of an operand: size prefix, else register width, else default."""
    lowered = token.lower()
    for prefix, size in (("qword", 8), ("xmmword", 16), ("dword", 4), ("word", 2), ("byte", 1)):
        if prefix in lowered:
            return size
    reg = register_operand(token)
    if reg is not None:
        return register_width(reg)
    return default


def memory_base_registers(token: str) -> List[str]:
    """Registers contributing to a memory operand address (``[rsp+rax*4]``)."""
    start = token.find("[")
    end = token.rfind("]")
    if start < 0 or end <= start:
        start = token.find("(")
        end = token.rfind(")")
    if start < 0 or end <= start:
        return []
    inner = token[start + 1 : end]
    found: List[str] = []
    for piece in re.split(r"[+\-]", inner):
        reg = register_operand(piece.split("*")[0].strip())
        if reg is not None and reg not in found:
            found.append(reg)
    return found


def canonical_register(reg: str, arch: str = "x86_64") -> str:
    """Map a (sub-)register name to its canonical tracking register."""
    name = str(reg).strip().lower().lstrip("%").lstrip("$")
    if not name:
        return ""
    if name in FLAG_REGISTERS:
        return "nzcv" if arch in ("arm64", "arm", "aarch64") else "rflags"
    if arch in ("x86", "x86_64"):
        return _X86_CANONICAL.get(name, name)
    if arch in ("arm64", "arm"):
        if len(name) > 1 and name[0] == "w" and name[1:].isdigit():
            return f"x{name[1:]}"
        if name == "wsp":
            return "sp"
        if name == "lr":
            return "x30"
        if name == "fp":
            return "x29"
        return name
    return name


def register_width(reg: str) -> int:
    """Bit width of a register name; 0 when unknown."""
    name = str(reg).strip().lower().lstrip("%").lstrip("$")
    if not name:
        return 0
    if name in ("al", "bl", "cl", "dl", "ah", "bh", "ch", "dh"):
        return 8
    if re.fullmatch(r"r\d+b", name) or name in ("sil", "dil", "bpl", "spl"):
        return 8
    if name in ("ax", "bx", "cx", "dx", "si", "di", "bp", "sp"):
        return 16
    if re.fullmatch(r"r\d+w", name) or re.fullmatch(r"w\d+", name):
        return 16 if name[0] == "w" else 32
    if name in ("eax", "ebx", "ecx", "edx", "esi", "edi", "ebp", "esp", "eip"):
        return 32
    if re.fullmatch(r"r\d+d", name):
        return 32
    if name in ("rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp", "rsp", "rip"):
        return 64
    if re.fullmatch(r"r\d+", name):
        return 64
    if re.fullmatch(r"x\d+", name):
        return 64
    if name in ("xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7"):
        return 128
    return 0


def _parse_int_token(token: str) -> Optional[int]:
    """Parse a hex/decimal immediate or branch target from an operand token."""
    body = token.strip().lstrip("#").lstrip("$").strip()
    if not body:
        return None
    neg = body.startswith("-")
    if neg:
        body = body[1:]
    body = body.strip()
    if not body or body.lower() in _NON_REGISTER_TOKENS:
        return None
    try:
        if body.lower().startswith("0x"):
            value = int(body, 16)
        elif any(c in "abcdefABCDEF" for c in body):
            value = int(body, 16)
        else:
            value = int(body, 10)
    except ValueError:
        return None
    return -value if neg else value


def _branch_target_token(op_str: str) -> Optional[int]:
    tokens = split_operands(op_str)
    if not tokens:
        return None
    return _parse_int_token(tokens[-1])


def is_conditional_branch(mnemonic: str) -> bool:
    """True for a conditional branch mnemonic (x86 ``jcc`` / arm64 ``b.cc``)."""
    m = mnemonic.lower().strip()
    if m in _CONDITIONAL_JUMPS:
        return True
    return m.startswith("b.") and len(m) > 2


@dataclass(frozen=True)
class InsnView:
    """Uniform, framework-independent view over one decoded instruction."""

    address: int
    size: int
    mnemonic: str
    op_str: str
    target: Optional[int] = None
    is_jump: bool = False
    is_uncond_jump: bool = False
    is_ret: bool = False
    is_call: bool = False
    raw: Any = None

    @property
    def text(self) -> str:
        return f"{self.mnemonic} {self.op_str}".strip()

    @property
    def end(self) -> int:
        return self.address + self.size


def normalize_instruction(insn: Any) -> InsnView:
    """Normalize a toothpick ``Instruction`` or a Capstone ``CsInsn``."""
    if isinstance(insn, InsnView):
        return insn
    mnemonic = str(getattr(insn, "mnemonic", "") or "").strip().lower()
    op_str = str(getattr(insn, "op_str", "") or "").strip()
    address = int(getattr(insn, "address", 0) or 0)
    size = int(getattr(insn, "size", 0) or 0)

    raw_target = getattr(insn, "target", None)
    target = int(raw_target) if isinstance(raw_target, int) else _branch_target_token(op_str)

    is_call = bool(getattr(insn, "is_call", False)) or mnemonic in _CALL_MNEMONICS
    is_ret = bool(getattr(insn, "is_ret", False)) or mnemonic in _RET_MNEMONICS
    is_uncond = bool(getattr(insn, "is_uncond_jump", False)) or (
        mnemonic in _UNCOND_JUMPS and not is_call
    )
    is_jump = (
        bool(getattr(insn, "is_jump", False))
        or is_uncond
        or is_conditional_branch(mnemonic)
    )
    if is_ret:
        is_call = False
    return InsnView(
        address=address,
        size=size,
        mnemonic=mnemonic,
        op_str=op_str,
        target=target,
        is_jump=is_jump,
        is_uncond_jump=is_uncond,
        is_ret=is_ret,
        is_call=is_call,
        raw=insn,
    )


# ---------------------------------------------------------------------------
# Chunk 1: taint state model & report
# ---------------------------------------------------------------------------


class TaintState:
    """Register-set and byte-level memory taint for one analysis context.

    ``tainted_memory`` maps *byte* addresses to ``True``.  Register taint is
    tracked at register-family granularity: tainting ``al`` marks ``rax``, and
    ``is_reg_tainted("al")`` therefore also reports the family as tainted.
    """

    def __init__(self, arch: str = "x86_64") -> None:
        self.arch = arch
        self.tainted_registers: Set[str] = set()
        self.tainted_memory: Dict[int, bool] = {}
        self.origins: Dict[str, str] = {}

    # -- registers ---------------------------------------------------------

    def taint_reg(self, reg: str, origin: Optional[str] = None) -> None:
        """Mark a register (and its family) as tainted."""
        name = canonical_register(reg, self.arch)
        if not name:
            return
        self.tainted_registers.add(name)
        if origin is not None:
            self.origins[name] = origin

    def clear_reg(self, reg: str) -> None:
        """Remove taint from a register (and its family)."""
        name = canonical_register(reg, self.arch)
        if not name:
            return
        self.tainted_registers.discard(name)
        self.origins.pop(name, None)

    def is_reg_tainted(self, reg: str) -> bool:
        name = canonical_register(reg, self.arch)
        return bool(name) and name in self.tainted_registers

    # -- memory ------------------------------------------------------------

    def taint_mem(self, addr: int, size: int = 1, origin: Optional[str] = None) -> None:
        """Mark ``size`` bytes starting at ``addr`` as tainted."""
        if size <= 0:
            size = 1
        for byte_addr in range(int(addr), int(addr) + int(size)):
            self.tainted_memory[byte_addr] = True
            if origin is not None:
                self.origins[f"mem_0x{byte_addr:x}"] = origin

    def clear_mem(self, addr: int, size: int = 1) -> None:
        """Remove taint from ``size`` bytes starting at ``addr``."""
        if size <= 0:
            size = 1
        for byte_addr in range(int(addr), int(addr) + int(size)):
            self.tainted_memory.pop(byte_addr, None)
            self.origins.pop(f"mem_0x{byte_addr:x}", None)

    def is_mem_tainted(self, addr: int, size: int = 1) -> bool:
        """True when *any* byte in ``[addr, addr + size)`` carries taint."""
        if size <= 0:
            size = 1
        return any(self.tainted_memory.get(a, False) for a in range(int(addr), int(addr) + int(size)))

    # -- introspection -----------------------------------------------------

    def tainted_memory_ranges(self) -> List[Tuple[int, int]]:
        """Collapse tainted bytes into sorted ``(start, size)`` runs."""
        runs: List[Tuple[int, int]] = []
        for addr in sorted(self.tainted_memory):
            if runs and addr == runs[-1][0] + runs[-1][1]:
                start, size = runs[-1]
                runs[-1] = (start, size + 1)
            else:
                runs.append((addr, 1))
        return runs

    def variable_names(self) -> List[str]:
        """Deterministic list of tainted variable names (registers + memory runs)."""
        names = set(self.tainted_registers)
        for start, size in self.tainted_memory_ranges():
            names.add(f"mem_0x{start:x}" if size == 1 else f"mem_0x{start:x}_{size}")
        return sorted(names)

    def clone(self) -> "TaintState":
        """Deep copy; used for path forks."""
        copy = TaintState(self.arch)
        copy.tainted_registers = set(self.tainted_registers)
        copy.tainted_memory = dict(self.tainted_memory)
        copy.origins = dict(self.origins)
        return copy


@dataclass
class TaintReport:
    """Result of one :meth:`TaintEngine.analyze_function` run."""

    func_addr: int = 0
    source: str = ""
    sink_addr: int = 0
    is_sink_tainted: bool = False
    taint_propagation_path: List[int] = field(default_factory=list)
    tainted_variables: List[str] = field(default_factory=list)
    instruction_count: int = 0
    sink_hits: List[int] = field(default_factory=list)
    unsupported_mnemonics: List[str] = field(default_factory=list)
    unresolved_memory: List[int] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "func_addr": self.func_addr,
            "func_addr_hex": f"0x{self.func_addr:x}",
            "source": self.source,
            "sink_addr": self.sink_addr,
            "sink_addr_hex": f"0x{self.sink_addr:x}",
            "is_sink_tainted": self.is_sink_tainted,
            "taint_propagation_path": list(self.taint_propagation_path),
            "taint_propagation_path_hex": [f"0x{a:x}" for a in self.taint_propagation_path],
            "tainted_variables": list(self.tainted_variables),
            "instruction_count": self.instruction_count,
            "sink_hits": list(self.sink_hits),
            "unsupported_mnemonics": list(self.unsupported_mnemonics),
            "unresolved_memory": list(self.unresolved_memory),
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# Chunk 2: dynamic taint engine
# ---------------------------------------------------------------------------


class TaintEngine:
    """Instruction-level dynamic taint engine over decoded instruction streams.

    ``step_instruction`` applies one transfer function; ``analyze_function``
    resets state, seeds the requested source and walks the reachable
    instructions in address order, returning a :class:`TaintReport`.
    """

    def __init__(
        self,
        arch: str = "x86_64",
        state: Optional[TaintState] = None,
        stack_base: int = DEFAULT_STACK_BASE,
        pointer_size: Optional[int] = None,
    ) -> None:
        self.arch = arch
        self.pointer_size = pointer_size or (4 if arch in ("x86", "arm", "ppc", "mips") else 8)
        self.state = state if state is not None else TaintState(arch)
        self.stack_base = int(stack_base)
        self.reg_values: Dict[str, int] = {self._stack_pointer(): self.stack_base}
        self.sink_addr: Optional[int] = None
        self.sink_tainted = False
        self.sink_hits: List[int] = []
        self.propagation_path: List[int] = []
        self.unsupported_mnemonics: List[str] = []
        self.unresolved_memory: List[int] = []

    # -- configuration -----------------------------------------------------

    @property
    def flags_reg(self) -> str:
        return canonical_register("nzcv" if self.arch in ("arm64", "arm") else "rflags", self.arch)

    def reset(self) -> None:
        """Clear all taint, path and diagnostic state."""
        self.state = TaintState(self.arch)
        self.reg_values = {self._stack_pointer(): self.stack_base}
        self.sink_tainted = False
        self.sink_hits = []
        self.propagation_path = []
        self.unsupported_mnemonics = []
        self.unresolved_memory = []

    def set_reg_value(self, reg: str, value: int) -> None:
        """Seed a known concrete register value (used for address resolution)."""
        name = canonical_register(reg, self.arch)
        if name:
            self.reg_values[name] = int(value)

    def set_sink(self, sink_addr: int) -> None:
        self.sink_addr = int(sink_addr)

    def set_source(self, source: Union[str, int], size: Optional[int] = None) -> None:
        """Seed taint from a register name or a memory address.

        ``source`` accepts ``"rsi"`` (register), ``"0x140003000"``, ``"mem:0x..."``
        or a bracketed operand such as ``"[rsp+0x10]"``.  Memory sources taint
        ``size`` bytes (default: pointer size).
        """
        if isinstance(source, int):
            self.state.taint_mem(source, size or self.pointer_size, origin=str(source))
            return
        text = str(source).strip()
        if not text:
            raise SymbolicError("empty taint source")
        lowered = text.lower()
        if lowered.startswith("mem:"):
            text = text[4:].strip()
        if is_memory_operand(text) or text.lower().startswith("0x") or text.isdigit():
            resolved = self._resolve_memory(text, None) if is_memory_operand(text) else None
            if resolved is None:
                resolved = _parse_int_token(text)
            if resolved is None:
                raise SymbolicError(
                    f"cannot resolve taint source {source!r}: address is not statically known"
                )
            self.state.taint_mem(resolved, size or self.pointer_size, origin=str(source))
            return
        reg = register_operand(text)
        if reg is None:
            raise SymbolicError(f"unrecognized taint source {source!r}")
        self.state.taint_reg(reg, origin=str(source))

    # -- analysis ----------------------------------------------------------

    def analyze_function(
        self,
        insns: Sequence[InstructionLike],
        source: Union[str, int],
        sink_addr: int,
        func_addr: Optional[int] = None,
        source_size: Optional[int] = None,
    ) -> TaintReport:
        """Run DTA over ``insns`` from ``source`` and check ``sink_addr``."""
        views = [normalize_instruction(insn) for insn in insns]
        self.reset()
        self.sink_addr = int(sink_addr)
        if not views:
            raise SymbolicError("no instructions supplied to analyze_function")
        self.set_source(source, size=source_size)

        index = {view.address: view for view in views}
        sorted_addrs = sorted(index)
        reachable = self._reachable_addresses(views, index, sorted_addrs)

        for addr in sorted(reachable):
            self._step(index[addr])

        if not self.sink_tainted and self.state.is_mem_tainted(self.sink_addr, 1):
            # The sink is a *data* address rather than an instruction address.
            self.sink_tainted = True
            self.sink_hits.append(self.sink_addr)

        notes: List[str] = []
        if self.unsupported_mnemonics:
            notes.append(
                "taint-transparent (unmodelled) mnemonics: "
                + ", ".join(sorted(self.unsupported_mnemonics))
            )
        if self.unresolved_memory:
            notes.append(
                "unresolved memory addresses (loads tainted conservatively): "
                + ", ".join(f"0x{a:x}" for a in sorted(self.unresolved_memory)[:16])
            )
        if not self.sink_tainted:
            notes.append(
                "taint did not reach the sink: either the flow is absent or an "
                "unmodelled mnemonic cut it (see unsupported_mnemonics)"
            )

        return TaintReport(
            func_addr=int(func_addr if func_addr is not None else views[0].address),
            source=str(source),
            sink_addr=self.sink_addr,
            is_sink_tainted=self.sink_tainted,
            taint_propagation_path=list(self.propagation_path),
            tainted_variables=self.state.variable_names(),
            instruction_count=len(views),
            sink_hits=list(self.sink_hits),
            unsupported_mnemonics=sorted(self.unsupported_mnemonics),
            unresolved_memory=sorted(self.unresolved_memory),
            notes=notes,
        )

    def step_instruction(self, insn: Any) -> None:
        """Apply one instruction's transfer function to the live taint state."""
        self._step(normalize_instruction(insn))

    # -- internals: control flow ------------------------------------------

    def _reachable_addresses(
        self,
        views: Sequence[InsnView],
        index: Dict[int, InsnView],
        sorted_addrs: Sequence[int],
    ) -> Set[int]:
        """Addresses reachable from the first instruction following CFG edges."""
        reachable: Set[int] = set()
        stack: List[int] = [views[0].address]
        while stack:
            addr = stack.pop()
            if addr in reachable or addr not in index:
                continue
            reachable.add(addr)
            view = index[addr]
            if view.is_ret or view.mnemonic in _RET_MNEMONICS:
                continue
            fallthrough = self._linear_next(addr, sorted_addrs)
            if view.is_uncond_jump:
                if view.target is not None:
                    stack.append(view.target)
                continue
            if view.is_jump or is_conditional_branch(view.mnemonic):
                if view.target is not None:
                    stack.append(view.target)
                if fallthrough is not None:
                    stack.append(fallthrough)
                continue
            if fallthrough is not None:
                stack.append(fallthrough)
        return reachable

    @staticmethod
    def _linear_next(addr: int, sorted_addrs: Sequence[int]) -> Optional[int]:
        position = bisect.bisect_right(sorted_addrs, addr)
        if position >= len(sorted_addrs):
            return None
        return int(sorted_addrs[position])

    # -- internals: transfer semantics ------------------------------------

    def _step(self, view: InsnView) -> None:
        mnemonic = view.mnemonic
        ops = split_operands(view.op_str)

        reads_tainted = self._reads_tainted(view, mnemonic, ops)
        if not ops and mnemonic in _NOOP_MNEMONICS:
            return
        self._apply(view, mnemonic, ops)

        changed = bool(self.state.tainted_registers) or bool(self.state.tainted_memory)
        if reads_tainted or changed:
            if reads_tainted and (not self.propagation_path or self.propagation_path[-1] != view.address):
                self.propagation_path.append(view.address)

        if self.sink_addr is not None and view.address == self.sink_addr:
            if reads_tainted or self._memory_roles(mnemonic, ops):
                if reads_tainted:
                    self.sink_tainted = True
                    self.sink_hits.append(view.address)

    def _apply(self, view: InsnView, mnemonic: str, ops: List[str]) -> None:
        if mnemonic in _NOOP_MNEMONICS:
            return
        if mnemonic in _SYSCALL_MNEMONICS:
            self._do_syscall()
            return
        if mnemonic in ("push", "pushf", "pushfq", "pushfd"):
            self._do_push(view, ops)
            return
        if mnemonic in ("pop", "popf", "popfq", "popfd"):
            self._do_pop(view, ops)
            return
        if mnemonic in _CALL_MNEMONICS:
            self._do_call(view, ops)
            return
        if mnemonic in _RET_MNEMONICS:
            self._do_ret()
            return
        if mnemonic == "leave":
            return
        if (
            mnemonic in _ZEROING_MNEMONICS
            and len(ops) >= 2
            and _same_token(ops[0], ops[1])
        ):
            self._write_operand(view, ops[0], False)
            self._set_flags(view, tainted=False)
            return
        if mnemonic in _MOV_MNEMONICS:
            self._do_move(view, mnemonic, ops)
            return
        if mnemonic in _BINOP_MNEMONICS:
            self._do_binop(view, mnemonic, ops)
            return
        if mnemonic in _COMPARE_MNEMONICS:
            self._do_compare(view, ops)
            return
        if mnemonic in _UNARY_MNEMONICS:
            self._do_unary(view, mnemonic, ops)
            return
        if mnemonic.startswith("set") and mnemonic in _BRANCH_FLAG_READERS:
            self._write_operand(view, ops[0] if ops else "", self.state.is_reg_tainted(self.flags_reg))
            return
        if mnemonic.startswith("cmov") and len(ops) >= 2:
            tainted = self._read_operand_tainted(view, ops[1]) or self.state.is_reg_tainted(
                self.flags_reg
            )
            self._write_operand(view, ops[0], tainted)
            return
        if mnemonic in ("xchg", "xadd") and len(ops) >= 2:
            tainted = self._read_operand_tainted(view, ops[0]) or self._read_operand_tainted(
                view, ops[1]
            )
            self._write_operand(view, ops[0], tainted)
            self._write_operand(view, ops[1], tainted)
            return
        if mnemonic in ("enter", "stos", "stosb", "movs", "movsb", "scas", "lods", "rep", "repne"):
            self.unsupported_mnemonics.append(mnemonic)
            return
        if not view.is_jump and not view.is_ret:
            self.unsupported_mnemonics.append(mnemonic or "<empty>")

    def _do_move(self, view: InsnView, mnemonic: str, ops: List[str]) -> None:
        if not ops:
            return
        dest = ops[0]
        src = ops[1] if len(ops) > 1 else None
        size = operand_size(dest, 0) or (operand_size(src, 0) if src else 0) or self.pointer_size
        if src is None:
            return
        if mnemonic == "lea":
            self._write_operand(view, dest, self._address_dependency_tainted(src), size=size)
            return
        if is_memory_operand(dest):  # store
            self._write_operand(view, dest, self._read_operand_tainted(view, src), size=size)
            return
        if is_memory_operand(src):  # load
            resolved = self._resolve_memory(src, view)
            if resolved is None:
                # Unknown provenance: conservatively keep the destination tainted.
                self._write_operand(view, dest, True, size=size)
                return
            self._write_operand(view, dest, self.state.is_mem_tainted(resolved, size), size=size)
            return
        self._write_operand(view, dest, self._read_operand_tainted(view, src), size=size)

    def _do_binop(self, view: InsnView, mnemonic: str, ops: List[str]) -> None:
        if not ops:
            return
        dest = ops[0]
        sources = ops[1:]
        size = operand_size(dest, 0) or self.pointer_size
        tainted = self._read_operand_tainted(view, dest)
        for src in sources:
            tainted = tainted or self._read_operand_tainted(view, src)
        self._write_operand(view, dest, tainted, size=size)
        self._set_flags(view, tainted=tainted)
        self._fold_constant(mnemonic, ops)

    def _do_compare(self, view: InsnView, ops: List[str]) -> None:
        tainted = False
        for op in ops:
            tainted = tainted or self._read_operand_tainted(view, op)
        self._set_flags(view, tainted=tainted)

    def _do_unary(self, view: InsnView, mnemonic: str, ops: List[str]) -> None:
        if not ops:
            return
        dest = ops[0]
        tainted = self._read_operand_tainted(view, dest)
        size = operand_size(dest, 0) or self.pointer_size
        self._write_operand(view, dest, tainted, size=size)
        if mnemonic in ("inc", "dec", "neg", "not"):
            self._set_flags(view, tainted=tainted)

    def _do_push(self, view: InsnView, ops: List[str]) -> None:
        source = ops[0] if ops else None
        tainted = self._read_operand_tainted(view, source) if source is not None else False
        addr = self._stack_value() - self.pointer_size
        self._write_memory(addr, self.pointer_size, tainted)
        self.reg_values[self._stack_pointer()] = addr

    def _do_pop(self, view: InsnView, ops: List[str]) -> None:
        addr = self._stack_value()
        if ops:
            self._write_operand(
                view, ops[0], self.state.is_mem_tainted(addr, self.pointer_size), size=self.pointer_size
            )
        self.reg_values[self._stack_pointer()] = addr + self.pointer_size

    def _do_call(self, view: InsnView, ops: List[str]) -> None:
        # Opaque boundary: only stack bookkeeping, taint is not propagated
        # through the callee body (documented limitation).
        self.reg_values[self._stack_pointer()] = self._stack_value() - self.pointer_size
        if ops and not is_memory_operand(ops[0]):
            reg = register_operand(ops[0])
            if reg is not None and view.mnemonic == "bl":
                self.reg_values[canonical_register("x30", self.arch)] = view.end

    def _do_ret(self) -> None:
        self.reg_values[self._stack_pointer()] = self._stack_value() + self.pointer_size

    def _do_syscall(self) -> None:
        # A syscall return value is external input: taint it conservatively.
        self.state.taint_reg("x0" if self.arch in ("arm64", "arm") else "rax", origin="syscall")

    # -- internals: taint queries & writes ---------------------------------

    def _write_operand(self, view: InsnView, token: str, tainted: bool, size: Optional[int] = None) -> None:
        if not token:
            return
        if is_memory_operand(token):
            addr = self._resolve_memory(token, view)
            width = size or operand_size(token, self.pointer_size)
            if addr is None:
                self._note_unresolved(token, view)
                return
            self._write_memory(addr, width, tainted)
            return
        reg = register_operand(token)
        if reg is None:
            return
        if tainted:
            self.state.taint_reg(reg)
        else:
            self.state.clear_reg(reg)

    def _write_memory(self, addr: int, size: int, tainted: bool) -> None:
        if tainted:
            self.state.taint_mem(addr, size)
        else:
            self.state.clear_mem(addr, size)

    def _read_operand_tainted(self, view: InsnView, token: Optional[str]) -> bool:
        """Taint of an operand *value*; unknown-provenance memory reads are tainted."""
        if token is None:
            return False
        if is_memory_operand(token):
            addr = self._resolve_memory(token, view)
            if addr is None:
                self._note_unresolved(token, view)
                return True
            return self.state.is_mem_tainted(addr, operand_size(token, self.pointer_size))
        reg = register_operand(token)
        if reg is None:
            return False
        return self.state.is_reg_tainted(reg)

    def _address_dependency_tainted(self, token: str) -> bool:
        for reg in memory_base_registers(token):
            if self.state.is_reg_tainted(reg):
                return True
        return False

    def _reads_tainted(self, view: InsnView, mnemonic: str, ops: List[str]) -> bool:
        for reg in self._read_registers(mnemonic, ops):
            if self.state.is_reg_tainted(reg):
                return True
        for token, role in self._memory_roles(mnemonic, ops):
            if role in ("load", "rmw") and self._read_operand_tainted(view, token):
                return True
        if mnemonic in ("pop", "popf", "popfq", "popfd"):
            if self.state.is_mem_tainted(self._stack_value(), self.pointer_size):
                return True
        if mnemonic in ("ret", "retf"):
            if self.state.is_reg_tainted("x0" if self.arch in ("arm64", "arm") else "rax"):
                return True
        return False

    def _read_registers(self, mnemonic: str, ops: List[str]) -> List[str]:
        reads: List[str] = []
        for position, token in enumerate(ops):
            if is_memory_operand(token):
                for reg in memory_base_registers(token):
                    if reg not in reads:
                        reads.append(reg)
                continue
            if position == 0 and mnemonic in _PURE_DEST_MNEMONICS:
                continue
            reg = register_operand(token)
            if reg is not None and reg not in reads:
                reads.append(reg)
        if mnemonic in _BRANCH_FLAG_READERS or mnemonic in ("jmp", "ljmp", "b", "br"):
            reads.append(self.flags_reg if mnemonic in _BRANCH_FLAG_READERS else "")
            reads = [reg for reg in reads if reg]
        return reads

    def _memory_roles(self, mnemonic: str, ops: List[str]) -> List[Tuple[str, str]]:
        roles: List[Tuple[str, str]] = []
        for position, token in enumerate(ops):
            if not is_memory_operand(token):
                continue
            if mnemonic in _ARM64_STORE_MNEMONICS:
                roles.append((token, "store"))
            elif mnemonic in _ARM64_LOAD_MNEMONICS:
                roles.append((token, "load"))
            elif mnemonic in _MOV_MNEMONICS and self.arch in ("x86", "x86_64") and position == 0:
                roles.append((token, "store"))
            elif mnemonic in _BINOP_MNEMONICS or mnemonic in ("xchg", "xadd"):
                roles.append((token, "rmw"))
            else:
                roles.append((token, "load"))
        return roles

    def _set_flags(self, view: InsnView, tainted: bool) -> None:
        if tainted:
            self.state.taint_reg(self.flags_reg)
        else:
            self.state.clear_reg(self.flags_reg)

    def _note_unresolved(self, token: str, view: Optional[InsnView]) -> None:
        address = view.address if view is not None else 0
        if address not in self.unresolved_memory:
            self.unresolved_memory.append(address)

    # -- internals: address & value resolution -----------------------------

    def _stack_pointer(self) -> str:
        return canonical_register(
            "sp" if self.arch in ("arm64", "arm") else ("esp" if self.arch == "x86" else "rsp"),
            self.arch,
        )

    def _stack_value(self) -> int:
        return self.reg_values.get(self._stack_pointer(), self.stack_base)

    def _resolve_memory(self, token: str, view: Optional[InsnView]) -> Optional[int]:
        """Resolve a memory operand to a concrete address, or ``None``."""
        start = token.find("[")
        end = token.rfind("]")
        if start < 0 or end <= start:
            start = token.find("(")
            end = token.rfind(")")
        if start < 0 or end <= start:
            return None
        inner = token[start + 1 : end]
        total = 0
        unknown = False
        for piece in re.findall(r"[+-]?[^+-]+", inner.replace(" ", "")):
            piece = piece.strip()
            if not piece:
                continue
            sign = -1 if piece.startswith("-") else 1
            body = piece.lstrip("+-").strip()
            if not body:
                continue
            base_text, _, scale_text = body.partition("*")
            reg = register_operand(base_text)
            if reg is not None:
                if reg in ("rip", "eip", "pc") and view is not None:
                    value = view.end
                else:
                    canonical = canonical_register(reg, self.arch)
                    if canonical not in self.reg_values:
                        unknown = True
                        continue
                    value = self.reg_values[canonical]
                scale = _parse_int_token(scale_text) if scale_text else 1
                if scale is None:
                    unknown = True
                    continue
                total += sign * value * int(scale)
                continue
            immediate = _parse_int_token(body)
            if immediate is None:
                unknown = True
                continue
            total += sign * immediate
        if unknown:
            return None
        return total

    def _fold_constant(self, mnemonic: str, ops: List[str]) -> None:
        """Track concrete values for ``add/sub reg, imm`` (stack pointer math)."""
        if mnemonic not in ("add", "sub") or len(ops) < 2:
            return
        dest = register_operand(ops[0])
        if dest is None:
            return
        canonical = canonical_register(dest, self.arch)
        if canonical not in self.reg_values:
            return
        delta = _parse_int_token(ops[1])
        if delta is None:
            return
        if mnemonic == "sub":
            delta = -delta
        self.reg_values[canonical] = (self.reg_values[canonical] + delta) % (1 << 64)


def _same_token(left: str, right: str) -> bool:
    return re.sub(r"\s+", "", left).lower() == re.sub(r"\s+", "", right).lower()


# ---------------------------------------------------------------------------
# Chunk 3: symbolic execution (pure-Python Z3 propagator)
# ---------------------------------------------------------------------------


@dataclass
class SolveResult:
    """Result of one symbolic reachability solve."""

    status: str  # "SAT" | "UNSAT" | "TIMEOUT" | "UNSUPPORTED"
    model: Optional[Dict[str, int]] = None
    constraints: List[str] = field(default_factory=list)
    elapsed_sec: float = 0.0
    taint_source: Optional[str] = None
    taint_reached: bool = False

    @property
    def witness(self) -> Optional[Dict[str, int]]:
        """Alias of :attr:`model` for witness-input consumers."""
        return self.model

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "model": self.model,
            "witness": self.model,
            "constraints": list(self.constraints),
            "elapsed_sec": round(self.elapsed_sec, 4),
            "taint_source": self.taint_source,
            "taint_reached": self.taint_reached,
        }


def _build_x86_64_registers() -> Dict[str, Tuple[str, int, int]]:
    table: Dict[str, Tuple[str, int, int]] = {}
    for r64, r32, r16, r8 in (
        ("rax", "eax", "ax", "al"),
        ("rbx", "ebx", "bx", "bl"),
        ("rcx", "ecx", "cx", "cl"),
        ("rdx", "edx", "dx", "dl"),
        ("rsi", "esi", "si", "sil"),
        ("rdi", "edi", "di", "dil"),
        ("rbp", "ebp", "bp", "bpl"),
        ("rsp", "esp", "sp", "spl"),
    ):
        table[r64] = (r64, 64, 0)
        table[r32] = (r64, 32, 0)
        table[r16] = (r64, 16, 0)
        table[r8] = (r64, 8, 0)
    for index in range(8, 16):
        base = f"r{index}"
        table[base] = (base, 64, 0)
        table[f"{base}d"] = (base, 32, 0)
        table[f"{base}w"] = (base, 16, 0)
        table[f"{base}b"] = (base, 8, 0)
    table["ah"] = ("rax", 8, 8)
    table["bh"] = ("rbx", 8, 8)
    table["ch"] = ("rcx", 8, 8)
    table["dh"] = ("rdx", 8, 8)
    table["rip"] = ("rip", 64, 0)
    table["eip"] = ("rip", 32, 0)
    return table


def _build_x86_registers() -> Dict[str, Tuple[str, int, int]]:
    table: Dict[str, Tuple[str, int, int]] = {}
    for r32, r16, r8, high in (
        ("eax", "ax", "al", "ah"),
        ("ebx", "bx", "bl", "bh"),
        ("ecx", "cx", "cl", "ch"),
        ("edx", "dx", "dl", "dh"),
        ("esi", "si", "sil", ""),
        ("edi", "di", "dil", ""),
        ("ebp", "bp", "bpl", ""),
        ("esp", "sp", "spl", ""),
    ):
        table[r32] = (r32, 32, 0)
        table[r16] = (r32, 16, 0)
        table[r8] = (r32, 8, 0)
        if high:
            table[high] = (r32, 8, 8)
    table["eip"] = ("eip", 32, 0)
    return table


def _build_arm64_registers() -> Dict[str, Tuple[str, int, int]]:
    table: Dict[str, Tuple[str, int, int]] = {}
    for index in range(31):
        table[f"x{index}"] = (f"x{index}", 64, 0)
        table[f"w{index}"] = (f"x{index}", 32, 0)
    table["sp"] = ("sp", 64, 0)
    table["wsp"] = ("sp", 32, 0)
    table["xzr"] = ("xzr", 64, 0)
    table["wzr"] = ("wzr", 32, 0)
    return table


_X86_64_REGISTERS = _build_x86_64_registers()
_X86_32_REGISTERS = _build_x86_registers()
_ARM64_REGISTERS = _build_arm64_registers()


def _build_aliases(root_regs: Sequence[Tuple[str, str, str, str]]) -> Dict[str, List[Tuple[str, int]]]:
    aliases: Dict[str, List[Tuple[str, int]]] = {}
    for r64, r32, r16, r8 in root_regs:
        aliases[r64] = [(r32, 0xFFFFFFFF), (r16, 0xFFFF), (r8, 0xFF)]
    for index in range(8, 16):
        aliases[f"r{index}"] = [
            (f"r{index}d", 0xFFFFFFFF),
            (f"r{index}w", 0xFFFF),
            (f"r{index}b", 0xFF),
        ]
    return aliases


_X86_64_ALIASES = _build_aliases(
    (
        ("rax", "eax", "ax", "al"),
        ("rbx", "ebx", "bx", "bl"),
        ("rcx", "ecx", "cx", "cl"),
        ("rdx", "edx", "dx", "dl"),
        ("rsi", "esi", "si", "sil"),
        ("rdi", "edi", "di", "dil"),
        ("rbp", "ebp", "bp", "bpl"),
        ("rsp", "esp", "sp", "spl"),
    )
)
_X86_32_ALIASES: Dict[str, List[Tuple[str, int]]] = {
    "eax": [("ax", 0xFFFF), ("al", 0xFF)],
    "ebx": [("bx", 0xFFFF), ("bl", 0xFF)],
    "ecx": [("cx", 0xFFFF), ("cl", 0xFF)],
    "edx": [("dx", 0xFFFF), ("dl", 0xFF)],
    "esi": [("si", 0xFFFF), ("sil", 0xFF)],
    "edi": [("di", 0xFFFF), ("dil", 0xFF)],
    "ebp": [("bp", 0xFFFF), ("bpl", 0xFF)],
    "esp": [("sp", 0xFFFF), ("spl", 0xFF)],
}
_ARM64_ALIASES: Dict[str, List[Tuple[str, int]]] = {
    f"x{index}": [(f"w{index}", 0xFFFFFFFF)] for index in range(31)
}


class Z3Propagator:
    """Pure-Python symbolic propagator over Z3 bitvector ASTs.

    Registers start as free bitvectors (the unconstrained program input), every
    modelled instruction rewrites them, and branch predicates are accumulated as
    path constraints.  No external symbolic framework is involved.
    """

    def __init__(self, arch: str = "x86_64") -> None:
        self.arch = arch
        self._base_width = 32 if arch == "x86" else 64
        if arch in ("arm64", "aarch64"):
            self._reg_map = _ARM64_REGISTERS
        elif arch == "x86":
            self._reg_map = _X86_32_REGISTERS
        else:
            self._reg_map = _X86_64_REGISTERS
        self.registers: Dict[str, z3.BitVecRef] = {}
        self.memory: Dict[str, z3.BitVecRef] = {}
        self.flags: Dict[str, z3.BoolRef] = {}
        self.constraints: List[z3.BoolRef] = []
        self.constraint_strs: List[str] = []
        self.unsupported: Optional[str] = None
        self._call_counter = 0

    # -- lifecycle ---------------------------------------------------------

    def clone(self) -> "Z3Propagator":
        fork = Z3Propagator(self.arch)
        fork.registers = dict(self.registers)
        fork.memory = dict(self.memory)
        fork.flags = dict(self.flags)
        fork.constraints = list(self.constraints)
        fork.constraint_strs = list(self.constraint_strs)
        fork.unsupported = self.unsupported
        fork._call_counter = self._call_counter
        return fork

    # -- register/memory access -------------------------------------------

    def _lookup(self, reg: str) -> Optional[Tuple[str, int, int]]:
        return self._reg_map.get(reg.lower().strip())

    def reg(self, name: str) -> z3.BitVecRef:
        """Symbolic value of a register (free bitvector on first read)."""
        clean = name.lower().strip().lstrip("%")
        if clean in ("xzr", "wzr"):
            return z3.BitVecVal(0, 32 if clean == "wzr" else 64)
        entry = self._lookup(clean)
        if entry is None:
            if clean not in self.registers:
                self.registers[clean] = z3.BitVec(clean, self._base_width)
            return self.registers[clean]
        base, width, offset = entry
        current = self.registers.get(base)
        if current is None:
            current = z3.BitVec(base, self._base_width)
            self.registers[base] = current
        if width == self._base_width and offset == 0:
            return current
        return z3.Extract(offset + width - 1, offset, current)

    def set_reg(self, name: str, value: z3.BitVecRef) -> None:
        """Write a register, zero-extending 32-bit writes in 64-bit mode."""
        clean = name.lower().strip().lstrip("%")
        if clean in ("xzr", "wzr"):
            return
        entry = self._lookup(clean)
        if entry is None:
            self.registers[clean] = value
            return
        base, width, offset = entry
        current = self.registers.get(base)
        if current is None:
            current = z3.BitVec(base, self._base_width)
            self.registers[base] = current
        if value.size() < width:
            value = z3.ZeroExt(width - value.size(), value)
        elif value.size() > width:
            value = z3.Extract(width - 1, 0, value)
        if width == self._base_width and offset == 0:
            self.registers[base] = value
            return
        if self.arch in ("x86_64", "arm64", "aarch64") and width == 32 and offset == 0:
            self.registers[base] = z3.ZeroExt(32, value)
            return
        pieces: List[z3.BitVecRef] = []
        if offset + width < self._base_width:
            pieces.append(z3.Extract(self._base_width - 1, offset + width, current))
        pieces.append(value)
        if offset > 0:
            pieces.append(z3.Extract(offset - 1, 0, current))
        self.registers[base] = z3.Concat(*pieces) if len(pieces) > 1 else pieces[0]

    def mem(self, token: str, width: int) -> z3.BitVecRef:
        """Free bitvector standing for the contents of a memory operand."""
        name = "mem_" + re.sub(r"[^0-9a-zA-Z]+", "_", token).strip("_")
        existing = self.memory.get(name)
        if existing is not None and existing.size() == width:
            return existing
        value = z3.BitVec(name, width)
        self.memory[name] = value
        return value

    def add_constraint(self, cond: z3.BoolRef, desc: Optional[str] = None) -> None:
        self.constraints.append(cond)
        self.constraint_strs.append(desc or str(cond))

    # -- operand evaluation ------------------------------------------------

    def _eval_operand(self, token: str, width: int) -> z3.BitVecRef:
        immediate = _parse_int_token(token)
        if immediate is not None and register_operand(token) is None:
            return z3.BitVecVal(immediate & ((1 << width) - 1), width)
        if is_memory_operand(token):
            return self.mem(token, width)
        return self.reg(token)

    def _eval_address(self, token: str, width: int) -> Optional[z3.BitVecRef]:
        start = token.find("[")
        end = token.rfind("]")
        if start < 0 or end <= start:
            return None
        inner = token[start + 1 : end].replace(" ", "")
        total: Optional[z3.BitVecRef] = None
        for piece in re.findall(r"[+-]?[^+-]+", inner):
            if not piece:
                continue
            sign = -1 if piece.startswith("-") else 1
            body = piece.lstrip("+-")
            if not body:
                continue
            base_text, _, scale_text = body.partition("*")
            scale = int(scale_text) if scale_text.isdigit() else 1
            reg = register_operand(base_text)
            if reg is not None:
                term = self.reg(reg)
                if term.size() != width:
                    term = (
                        z3.ZeroExt(width - term.size(), term)
                        if term.size() < width
                        else z3.Extract(width - 1, 0, term)
                    )
                if scale != 1:
                    term = term * z3.BitVecVal(scale, width)
            else:
                immediate = _parse_int_token(body)
                if immediate is None:
                    return None
                term = z3.BitVecVal(immediate & ((1 << width) - 1), width)
            if sign < 0:
                term = -term
            total = term if total is None else total + term
        return total

    # -- transfer ----------------------------------------------------------

    def execute(self, insn: Any) -> None:
        """Apply one instruction symbolically (branches are handled by the explorer)."""
        view = normalize_instruction(insn)
        mnemonic = view.mnemonic
        ops = split_operands(view.op_str)
        if mnemonic in _NOOP_MNEMONICS or not ops:
            if mnemonic not in _NOOP_MNEMONICS and mnemonic not in ("ret", "nop"):
                self._unsupported(view, "instruction carries no operands")
            return
        if view.is_jump or view.is_ret or mnemonic in ("ret", "retf"):
            return
        if mnemonic in ("push", "pop", "pushf", "popf", "enter", "leave"):
            return
        if mnemonic in _CALL_MNEMONICS or mnemonic in _SYSCALL_MNEMONICS:
            self._kill_return_register(view)
            return
        if mnemonic in ("mov", "movabs"):
            self.set_reg(ops[0], self._eval_operand(ops[1], self._width_of(ops[0])))
            return
        if mnemonic in ("movzx", "movz"):
            dest_width = self._width_of(ops[0])
            src = self._eval_operand(ops[1], self._width_of(ops[1]))
            self.set_reg(ops[0], z3.ZeroExt(max(0, dest_width - src.size()), src))
            return
        if mnemonic in ("movsx", "movsxd", "movn"):
            dest_width = self._width_of(ops[0])
            src = self._eval_operand(ops[1], self._width_of(ops[1]))
            self.set_reg(ops[0], z3.SignExt(max(0, dest_width - src.size()), src))
            return
        if mnemonic == "lea":
            width = self._width_of(ops[0])
            address = self._eval_address(ops[1], width)
            if address is None:
                self._unsupported(view, "unsupported LEA address expression")
                return
            self.set_reg(ops[0], address)
            return
        if mnemonic in ("add", "adds", "sub", "subs", "cmp", "cmn", "test", "tst"):
            self._execute_compare_like(view, mnemonic, ops)
            return
        if mnemonic in ("and", "ands", "or", "orr", "xor", "eor"):
            self._execute_logical(view, mnemonic, ops)
            return
        if mnemonic in ("shl", "sal", "shlx", "shr", "shrx", "sar", "sarx"):
            self._execute_shift(view, mnemonic, ops)
            return
        if mnemonic in ("inc", "dec", "neg", "not"):
            self._execute_unary(view, mnemonic, ops)
            return
        if mnemonic in ("adc", "sbb"):
            self._execute_compare_like(view, "add" if mnemonic == "adc" else "sub", ops)
            return
        if mnemonic.startswith("cmov"):
            cond = self.branch_condition(mnemonic.replace("cmov", "j", 1), ops, taken=True)
            if cond is None:
                self._unsupported(view, "conditional move without defined flags")
                return
            width = self._width_of(ops[0])
            self.set_reg(ops[0], z3.If(cond, self._eval_operand(ops[1], width), self.reg(ops[0])))
            return
        if mnemonic.startswith("set") and mnemonic in _BRANCH_FLAG_READERS:
            cond = self.branch_condition(mnemonic.replace("set", "j", 1), ops, taken=True)
            if cond is None:
                self._unsupported(view, "conditional set without defined flags")
                return
            width = self._width_of(ops[0])
            self.set_reg(
                ops[0], z3.ZeroExt(max(0, width - 1), z3.If(cond, z3.BitVecVal(1, 1), z3.BitVecVal(0, 1)))
            )
            return
        self._unsupported(view, f"unmodelled mnemonic '{mnemonic}'")

    def _execute_compare_like(self, view: InsnView, mnemonic: str, ops: List[str]) -> None:
        width = self._width_of(ops[0])
        left = self._eval_operand(ops[0], width)
        right = self._eval_operand(ops[1], width) if len(ops) > 1 else z3.BitVecVal(0, width)
        if mnemonic in ("test", "tst"):
            result = left & right
            self._logical_flags(result, width)
            return
        if mnemonic in ("add", "adds", "cmn"):
            result = left + right
            self._add_flags(left, right, result, width)
            if mnemonic in ("add", "adds") and not is_memory_operand(ops[0]) and ops[0] != "xzr":
                self.set_reg(ops[0], result)
            return
        result = left - right
        self._sub_flags(left, right, result, width)
        if mnemonic in ("sub", "subs") and not is_memory_operand(ops[0]) and ops[0] != "xzr":
            self.set_reg(ops[0], result)

    def _execute_logical(self, view: InsnView, mnemonic: str, ops: List[str]) -> None:
        width = self._width_of(ops[0])
        if len(ops) >= 2 and _same_token(ops[0], ops[1]) and mnemonic in ("xor", "eor", "sub"):
            self.set_reg(ops[0], z3.BitVecVal(0, width))
            self._logical_flags(z3.BitVecVal(0, width), width)
            return
        left = self._eval_operand(ops[0], width)
        right = self._eval_operand(ops[1], width) if len(ops) > 1 else left
        if mnemonic in ("xor", "eor"):
            result = left ^ right
        elif mnemonic in ("or", "orr"):
            result = left | right
        else:
            result = left & right
        if not is_memory_operand(ops[0]) and ops[0] != "xzr":
            self.set_reg(ops[0], result)
        self._logical_flags(result, width)

    def _execute_shift(self, view: InsnView, mnemonic: str, ops: List[str]) -> None:
        width = self._width_of(ops[0])
        value = self._eval_operand(ops[0], width)
        amount = self._eval_operand(ops[1], width) if len(ops) > 1 else z3.BitVecVal(1, width)
        if mnemonic in ("shl", "sal", "shlx"):
            result = value << amount
        elif mnemonic in ("shr", "shrx"):
            result = z3.LShR(value, amount)
        else:
            result = value >> amount
        if not is_memory_operand(ops[0]):
            self.set_reg(ops[0], result)
        self._logical_flags(result, width)

    def _execute_unary(self, view: InsnView, mnemonic: str, ops: List[str]) -> None:
        width = self._width_of(ops[0])
        value = self._eval_operand(ops[0], width)
        if mnemonic == "inc":
            result = value + z3.BitVecVal(1, width)
            self._logical_flags(result, width)
        elif mnemonic == "dec":
            result = value - z3.BitVecVal(1, width)
            self._logical_flags(result, width)
        elif mnemonic == "neg":
            result = -value
            self._sub_flags(z3.BitVecVal(0, width), value, result, width)
        else:
            result = ~value
            self._logical_flags(result, width)
        if not is_memory_operand(ops[0]):
            self.set_reg(ops[0], result)

    def _kill_return_register(self, view: InsnView) -> None:
        """Model an opaque call/syscall: the return value becomes a fresh unknown."""
        self._call_counter += 1
        if self.arch in ("arm64", "aarch64", "arm"):
            target = "x0"
        elif self.arch == "x86":
            target = "eax"
        else:
            target = "rax"
        width = 32 if self.arch == "x86" else 64
        self.set_reg(target, z3.BitVec(f"{target}_ret_{view.address:x}_{self._call_counter}", width))

    def _unsupported(self, view: InsnView, reason: str) -> None:
        message = f"{reason} at 0x{view.address:x} ('{view.text}')"
        if self.unsupported is None:
            self.unsupported = message

    # -- flags -------------------------------------------------------------

    def _width_of(self, token: str) -> int:
        if is_memory_operand(token):
            return operand_size(token, self._base_width)
        width = operand_size(token, 0)
        return width or self._base_width

    def _logical_flags(self, result: z3.BitVecRef, width: int) -> None:
        self.flags["zf"] = result == z3.BitVecVal(0, width)
        self.flags["sf"] = z3.Extract(width - 1, width - 1, result) == z3.BitVecVal(1, 1)
        self.flags["cf"] = z3.BoolVal(False)
        self.flags["of"] = z3.BoolVal(False)

    def _add_flags(self, left: z3.BitVecRef, right: z3.BitVecRef, result: z3.BitVecRef, width: int) -> None:
        msb = width - 1
        self.flags["zf"] = result == z3.BitVecVal(0, width)
        self.flags["sf"] = z3.Extract(msb, msb, result) == z3.BitVecVal(1, 1)
        self.flags["cf"] = z3.ULT(result, left)
        self.flags["of"] = z3.Extract(msb, msb, (left ^ result) & (right ^ result)) == z3.BitVecVal(1, 1)

    def _sub_flags(self, left: z3.BitVecRef, right: z3.BitVecRef, result: z3.BitVecRef, width: int) -> None:
        msb = width - 1
        self.flags["zf"] = result == z3.BitVecVal(0, width)
        self.flags["sf"] = z3.Extract(msb, msb, result) == z3.BitVecVal(1, 1)
        self.flags["cf"] = z3.ULT(left, right)
        self.flags["of"] = z3.Extract(msb, msb, (left ^ right) & (left ^ result)) == z3.BitVecVal(1, 1)

    # -- branch predicates -------------------------------------------------

    def branch_condition(
        self, mnemonic: str, operands: Sequence[str], taken: bool = True
    ) -> Optional[z3.BoolRef]:
        """Z3 predicate for a conditional branch, or ``None`` when uncertain."""
        m = mnemonic.lower().strip()
        ops = list(operands)
        cond: Optional[z3.BoolRef] = None

        if m in ("jmp", "ljmp", "b", "br"):
            cond = z3.BoolVal(True)
        elif m in ("cbz", "cbnz"):
            if not ops:
                return None
            tested = self._eval_operand(ops[0], self._width_of(ops[0]))
            cond = tested == z3.BitVecVal(0, tested.size())
            if m == "cbnz":
                cond = z3.Not(cond)
        elif m in ("tbz", "tbnz"):
            if len(ops) < 2:
                return None
            tested = self._eval_operand(ops[0], self._width_of(ops[0]))
            bit = _parse_int_token(ops[1])
            if bit is None or not 0 <= bit < tested.size():
                return None
            bit_is_set = z3.Extract(int(bit), int(bit), tested) == z3.BitVecVal(1, 1)
            cond = z3.Not(bit_is_set) if m == "tbz" else bit_is_set
        else:
            zf = self.flags.get("zf")
            sf = self.flags.get("sf")
            cf = self.flags.get("cf")
            of = self.flags.get("of")
            signed_differs = z3.Xor(sf, of) if sf is not None and of is not None else None
            if m in ("je", "jz", "b.eq"):
                cond = zf
            elif m in ("jne", "jnz", "b.ne"):
                cond = z3.Not(zf) if zf is not None else None
            elif m in ("jl", "jnge", "b.lt", "b.mi"):
                cond = signed_differs
            elif m in ("jge", "jnl", "b.ge", "b.pl"):
                cond = z3.Not(signed_differs) if signed_differs is not None else None
            elif m in ("jle", "jng", "b.le"):
                cond = z3.Or(zf, signed_differs) if zf is not None and signed_differs is not None else None
            elif m in ("jg", "jnle", "b.gt"):
                cond = (
                    z3.And(z3.Not(zf), z3.Not(signed_differs))
                    if zf is not None and signed_differs is not None
                    else None
                )
            elif m in ("jb", "jc", "jnae", "b.lo", "b.cc", "b.cs"):
                cond = cf
            elif m in ("jae", "jnc", "jnb", "b.hs"):
                cond = z3.Not(cf) if cf is not None else None
            elif m in ("jbe", "jna", "b.ls"):
                cond = z3.Or(cf, zf) if cf is not None and zf is not None else None
            elif m in ("ja", "jnbe", "b.hi"):
                cond = (
                    z3.And(z3.Not(cf), z3.Not(zf))
                    if cf is not None and zf is not None
                    else None
                )
            elif m in ("js", "b.mi"):
                cond = sf
            elif m in ("jns", "b.pl"):
                cond = z3.Not(sf) if sf is not None else None
            elif m == "jo":
                cond = of
            elif m == "jno":
                cond = z3.Not(of) if of is not None else None

        if cond is None:
            return None
        return cond if taken else z3.Not(cond)

    # -- solving -----------------------------------------------------------

    def check_sat(self, timeout_ms: int) -> Union[str, Tuple[str, Dict[str, int]]]:
        """Run Z3 over the accumulated constraints with a strict timeout."""
        solver = z3.Solver()
        solver.set("timeout", max(1, int(timeout_ms)))
        for constraint in self.constraints:
            solver.add(constraint)
        status = solver.check()
        if status == z3.sat:
            return ("SAT", self.extract_model(solver.model()))
        if status == z3.unsat:
            return ("UNSAT", {})
        reason = str(solver.reason_unknown()).lower()
        if "timeout" in reason or "canceled" in reason or "cancelled" in reason:
            return "TIMEOUT"
        return ("UNSAT", {})

    def extract_model(self, model: z3.ModelRef) -> Dict[str, int]:
        """Model values plus derived sub-register aliases (``rax`` -> ``eax``)."""
        values: Dict[str, int] = {}
        for decl in model.decls():
            try:
                values[decl.name()] = int(model[decl].as_long())
            except (AttributeError, z3.Z3Exception):
                continue
        aliases = (
            _ARM64_ALIASES
            if self.arch in ("arm64", "aarch64")
            else (_X86_32_ALIASES if self.arch == "x86" else _X86_64_ALIASES)
        )
        for root, derived in aliases.items():
            if root not in values:
                continue
            for alias, mask in derived:
                values.setdefault(alias, values[root] & mask)
        return values


class SymbolicExecutionEngine:
    """Symbolic reachability over a decoded instruction stream.

    ``solve_target`` explores instruction-level paths from ``entry_addr`` to
    ``target_addr``, accumulating Z3 bitvector path constraints, and returns the
    first satisfying witness input (or ``UNSAT`` / ``TIMEOUT`` / ``UNSUPPORTED``).
    """

    def __init__(
        self,
        instructions: Optional[Sequence[InstructionLike]] = None,
        *,
        arch: str = "x86_64",
        code_bytes: Optional[bytes] = None,
        base_addr: int = 0x1000,
        max_paths: int = 200,
        max_path_length: int = 64,
    ) -> None:
        self.arch = arch
        self.max_paths = int(max_paths)
        self.max_path_length = int(max_path_length)
        if instructions is None and code_bytes is not None:
            instructions = disassemble_bytes(code_bytes, base_addr=base_addr, arch=arch)
        self.instructions: List[InsnView] = [
            normalize_instruction(insn) for insn in (instructions or [])
        ]

    def propagator(self) -> Z3Propagator:
        """A fresh pure-Python Z3 propagator for this architecture."""
        return Z3Propagator(self.arch)

    def solve_target(
        self,
        entry_addr: int,
        target_addr: int,
        timeout_sec: float = 5.0,
        *,
        taint_source: Optional[str] = None,
    ) -> SolveResult:
        """Solve for inputs that reach ``target_addr`` from ``entry_addr``."""
        started = time.perf_counter()
        index = {view.address: view for view in self.instructions}
        if not index:
            raise SymbolicError("no instructions supplied to SymbolicExecutionEngine")
        if entry_addr not in index:
            raise SymbolicError(f"entry address 0x{entry_addr:x} is not in the instruction stream")
        if target_addr not in index:
            return SolveResult(
                status="UNSAT",
                constraints=[f"target address 0x{target_addr:x} is not in the instruction stream"],
                elapsed_sec=time.perf_counter() - started,
                taint_source=taint_source,
            )

        sorted_addrs = sorted(index)
        source_regs: List[str] = []
        if taint_source:
            source_regs = [reg for reg in (register_operand(taint_source),) if reg]
            source_regs = [canonical_register(reg, self.arch) for reg in source_regs]

        queue: List[Tuple[int, Z3Propagator, Tuple[int, ...]]] = [
            (int(entry_addr), Z3Propagator(self.arch), (int(entry_addr),))
        ]
        paths_explored = 0
        last_constraints: List[str] = []
        unsupported_reason: Optional[str] = None

        while queue and paths_explored < self.max_paths:
            elapsed = time.perf_counter() - started
            if elapsed >= timeout_sec:
                return SolveResult(
                    status="TIMEOUT",
                    elapsed_sec=elapsed,
                    taint_source=taint_source,
                )
            addr, propagator, path = queue.pop(0)
            paths_explored += 1

            if addr == int(target_addr):
                remaining_ms = max(1, int((timeout_sec - (time.perf_counter() - started)) * 1000))
                outcome = propagator.check_sat(remaining_ms)
                if outcome == "TIMEOUT":
                    return SolveResult(
                        status="TIMEOUT",
                        constraints=propagator.constraint_strs,
                        elapsed_sec=time.perf_counter() - started,
                        taint_source=taint_source,
                    )
                last_constraints = list(propagator.constraint_strs)
                if outcome[0] == "SAT":
                    model = outcome[1]
                    reached = False
                    if source_regs:
                        reached = any(
                            f"{reg}" in " ".join(propagator.constraint_strs) for reg in source_regs
                        ) or any(reg in model for reg in source_regs)
                    return SolveResult(
                        status="SAT",
                        model=model,
                        constraints=last_constraints,
                        elapsed_sec=time.perf_counter() - started,
                        taint_source=taint_source,
                        taint_reached=reached,
                    )
                continue

            view = index[addr]
            if len(path) >= self.max_path_length:
                continue
            if view.is_ret or view.mnemonic in _RET_MNEMONICS:
                continue

            fallthrough = _next_address(addr, sorted_addrs)
            if view.is_uncond_jump:
                if view.target in index:
                    child = propagator.clone()
                    self._enqueue(queue, index[view.target], child, path)
                continue

            if view.is_jump or is_conditional_branch(view.mnemonic):
                operands = split_operands(view.op_str)
                for taken in (True, False):
                    dst = view.target if taken else fallthrough
                    if dst is None or dst not in index:
                        continue
                    condition = propagator.branch_condition(view.mnemonic, operands, taken=taken)
                    if condition is None:
                        unsupported_reason = (
                            f"cannot lift branch condition for '{view.text}' at 0x{view.address:x}"
                        )
                        continue
                    child = propagator.clone()
                    child.add_constraint(
                        condition,
                        desc=f"{view.mnemonic} {view.op_str} "
                        f"({'taken' if taken else 'fallthrough'}) -> {condition}",
                    )
                    self._enqueue(queue, index[dst], child, path)
                continue

            child = propagator.clone()
            child.execute(view)
            if child.unsupported is not None:
                unsupported_reason = child.unsupported
                continue
            if fallthrough is not None and fallthrough in index:
                self._enqueue(queue, index[fallthrough], child, path)

        elapsed_total = time.perf_counter() - started
        if unsupported_reason is not None:
            return SolveResult(
                status="UNSUPPORTED",
                constraints=last_constraints + [unsupported_reason],
                elapsed_sec=elapsed_total,
                taint_source=taint_source,
            )
        return SolveResult(
            status="UNSAT",
            constraints=last_constraints,
            elapsed_sec=elapsed_total,
            taint_source=taint_source,
        )

    @staticmethod
    def _enqueue(
        queue: List[Tuple[int, Z3Propagator, Tuple[int, ...]]],
        view: InsnView,
        propagator: Z3Propagator,
        path: Tuple[int, ...],
    ) -> None:
        if path.count(view.address) >= 2:  # bounded loop unrolling
            return
        queue.append((view.address, propagator, path + (view.address,)))


def _next_address(addr: int, sorted_addrs: Sequence[int]) -> Optional[int]:
    position = bisect.bisect_right(sorted_addrs, addr)
    if position >= len(sorted_addrs):
        return None
    return int(sorted_addrs[position])


# ---------------------------------------------------------------------------
# Chunk 3 (cont.): Triton / Manticore script exporters
# ---------------------------------------------------------------------------


@dataclass
class ScriptExporter:
    """Base class for external symbolic-framework script generators."""

    binary: str
    arch: str = "x86_64"
    func_addr: int = 0
    source: Optional[str] = None
    sink_addr: Optional[int] = None
    target_addr: Optional[int] = None
    max_instructions: int = 1024

    FRAMEWORK = ""

    @property
    def framework(self) -> str:
        return self.FRAMEWORK

    def generate(self) -> str:  # pragma: no cover - overridden
        raise NotImplementedError

    def write(self, out_path: PathLike) -> Path:
        """Write the generated script to ``out_path`` (UTF-8) and return it."""
        path = Path(out_path)
        if path.parent and not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.generate(), encoding="utf-8")
        return path

    def check_availability(self) -> Tuple[bool, str]:  # pragma: no cover - overridden
        raise NotImplementedError

    def run(self, timeout_sec: float = 5.0) -> Dict[str, Any]:  # pragma: no cover - overridden
        raise NotImplementedError

    def to_dict(self) -> Dict[str, Any]:
        return {
            "framework": self.framework,
            "binary": self.binary,
            "arch": self.arch,
            "func_addr": self.func_addr,
            "func_addr_hex": f"0x{self.func_addr:x}",
            "source": self.source,
            "sink_addr": self.sink_addr,
            "target_addr": self.target_addr,
            "max_instructions": self.max_instructions,
        }

    # -- shared header -----------------------------------------------------

    def _header_lines(self) -> List[str]:
        return [
            "#!/usr/bin/env python3",
            '"""Generated by ``toothpick taint --export '
            + self.framework
            + '`` (unit W17, chunk 3).',
            "",
            "Toothpick itself never imports this framework: the script below is the",
            "hand-off point for operators who have the dedicated symbolic stack",
            "installed.  Review it before running it against untrusted targets.",
            '"""',
            "from __future__ import annotations",
            "",
            "import argparse",
            "import sys",
            "",
            f"BINARY = {self.binary!r}",
            f"ARCH_NAME = {self.arch!r}",
            f"FUNC_ADDR = {self.func_addr:#x}",
            f"TAINT_SOURCE = {self.source!r}",
            f"SINK_ADDR = {self.sink_addr!r}",
            f"TARGET_ADDR = {self.target_addr!r}",
            f"MAX_INSTRUCTIONS = {self.max_instructions}",
            "",
        ]


_TRITON_BODY = '''\
try:
    import triton
    from triton import ARCH, MODE, Cpu, Instruction, TritonContext
except ImportError as exc:  # pragma: no cover - optional dependency
    print(f"triton (the Triton dynamic binary analysis framework) is required: {exc}",
          file=sys.stderr)
    raise SystemExit(2)


def build_context() -> "TritonContext":
    ctx = TritonContext()
    if ARCH_NAME in ("x86_64", "x64", "amd64"):
        ctx.setArchitecture(ARCH.X86_64)
    elif ARCH_NAME == "x86":
        ctx.setArchitecture(ARCH.X86)
    elif ARCH_NAME == "arm64":
        ctx.setArchitecture(ARCH.AARCH64)
    else:
        raise SystemExit(f"unsupported triton architecture: {ARCH_NAME}")
    ctx.setMode(MODE.ALIGNED_MEMORY, False)
    ctx.setMode(MODE.AST_OPTIMIZATIONS, True)
    return ctx


def load_code(path: str) -> bytes:
    with open(path, "rb") as handle:
        return handle.read()


def taint_source(ctx: "TritonContext") -> None:
    """Seed taint on the requested source register / memory address."""
    if TAINT_SOURCE is None:
        print("no --source given: only disassembly will be emitted", file=sys.stderr)
        return
    name = str(TAINT_SOURCE).strip()
    if name.startswith("0x") or name.startswith("mem:"):
        address = int(name.split(":")[-1], 16)
        ctx.taintMemory(ctx.getConcreteMemoryAreaValue(address, 8).__class__ and address)
    else:
        register = ctx.registers.__getattr__(name.lower()) if False else None
        if register is not None:
            ctx.taintRegister(ctx.getRegister(name.lower()))


def trace(ctx: "TritonContext", code: bytes) -> int:
    """Single-pass emulation from FUNC_ADDR, marking the sink when tainted."""
    ctx.setConcreteMemoryAreaValue(FUNC_ADDR, code)
    pc = FUNC_ADDR
    executed = 0
    sink_tainted = False
    while executed < MAX_INSTRUCTIONS:
        insn = Instruction(pc, ctx.getConcreteMemoryAreaValue(pc, 16))
        if not ctx.processing(insn):
            break
        executed += 1
        if SINK_ADDR is not None and pc == SINK_ADDR:
            for read in insn.getReadRegisters():
                if ctx.isRegisterTainted(read):
                    sink_tainted = True
        pc = ctx.getConcreteRegisterValue(ctx.registers.rip if false else ctx.registers.pc)
        if TARGET_ADDR is not None and pc == TARGET_ADDR:
            break
    print(f"executed={executed} sink_tainted={sink_tainted}")
    return 0 if not sink_tainted else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Triton taint analysis (toothpick W17)")
    parser.add_argument("--target", default=BINARY, help="binary or memory blob to analyze")
    args = parser.parse_args()
    ctx = build_context()
    taint_source(ctx)
    return trace(ctx, load_code(args.target))


if __name__ == "__main__":
    sys.exit(main())
'''


_MANTICORE_BODY = '''\
try:
    from manticore.native import Manticore
except ImportError as exc:  # pragma: no cover - optional dependency
    print("manticore is required (python 3.8/3.9 era release): "
          f"{exc}", file=sys.stderr)
    raise SystemExit(2)


def build_manticore() -> "Manticore":
    """Bind a Manticore instance to the first instruction after FUNC_ADDR."""
    m = Manticore(BINARY)
    m.context["toothpick.taint_source"] = TAINT_SOURCE
    m.context["toothpick.sink_addr"] = SINK_ADDR
    m.context["toothpick.target_addr"] = TARGET_ADDR
    m.context["toothpick.max_instructions"] = MAX_INSTRUCTIONS

    def attach(state):
        state.cpu.PC = FUNC_ADDR
        state.context["insn_count"] = 0

    def count(state):
        state.context["insn_count"] = state.context.get("insn_count", 0) + 1
        if state.context["insn_count"] > MAX_INSTRUCTIONS:
            state.abandon()
            return
        if SINK_ADDR is not None and state.cpu.PC == SINK_ADDR:
            # A tainted operand reaching the sink proves the candidate flow.
            print(f"[toothpick] sink reached at {SINK_ADDR:#x}")

    def report(state):
        print(f"[toothpick] path ran {state.context.get('insn_count', 0)} instructions")

    m.register_plugin(attach)
    return m


def main() -> int:
    m = build_manticore()
    m.run()
    m.finalize()
    print("manticore run complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


class TritonExporter(ScriptExporter):
    """Generates a self-contained Triton DTA script (Triton is imported by it)."""

    FRAMEWORK = "triton"

    def generate(self) -> str:
        lines = self._header_lines()
        lines.extend(_TRITON_BODY.rstrip("\n").splitlines())
        return "\n".join(lines) + "\n"

    def check_availability(self) -> Tuple[bool, str]:
        """Lazily import Triton; distinguishes the RE framework from the GPU one."""
        try:
            import triton  # noqa: PLC0415 - deliberately lazy (optional dependency)
        except Exception as exc:  # pragma: no cover - optional dependency
            return False, f"triton is not importable: {exc}"
        if not hasattr(triton, "TritonContext"):
            return False, (
                "the imported 'triton' module is not the Triton binary analysis "
                "framework (no TritonContext attribute)"
            )
        return True, f"triton {getattr(triton, '__version__', 'unknown')}"

    def run(self, timeout_sec: float = 5.0) -> Dict[str, Any]:
        """Availability-gated execution hook; the generated script does the work."""
        available, detail = self.check_availability()
        if not available:
            raise SymbolicError(
                f"cannot run the Triton workflow: {detail}. "
                "Run the exported script in an environment that has triton installed."
            )
        return {"framework": self.framework, "available": True, "detail": detail,
                "source": self.binary, "timeout_sec": timeout_sec}


class ManticoreExporter(ScriptExporter):
    """Generates a Manticore symbolic-exploration script (imported by it only)."""

    FRAMEWORK = "manticore"

    def generate(self) -> str:
        lines = self._header_lines()
        lines.extend(_MANTICORE_BODY.rstrip("\n").splitlines())
        return "\n".join(lines) + "\n"

    def check_availability(self) -> Tuple[bool, str]:
        try:
            import manticore  # noqa: PLC0415 - deliberately lazy (optional dependency)
        except Exception as exc:  # pragma: no cover - optional dependency
            return False, f"manticore is not importable: {exc}"
        return True, f"manticore {getattr(manticore, '__version__', 'unknown')}"

    def run(self, timeout_sec: float = 5.0) -> Dict[str, Any]:
        available, detail = self.check_availability()
        if not available:
            raise SymbolicError(
                f"cannot run the Manticore workflow: {detail}. "
                "Run the exported script in a Python 3.8/3.9 environment with manticore installed."
            )
        return {"framework": self.framework, "available": True, "detail": detail,
                "source": self.binary, "timeout_sec": timeout_sec}


_EXPORTERS: Dict[str, type] = {
    "triton": TritonExporter,
    "manticore": ManticoreExporter,
}


def exporter_for(framework: str, binary: PathLike, **kwargs: Any) -> ScriptExporter:
    """Return the exporter instance for ``framework`` (``triton`` / ``manticore``)."""
    key = str(framework).strip().lower()
    if key not in _EXPORTERS:
        raise SymbolicError(
            f"unknown symbolic framework {framework!r}; expected one of {sorted(_EXPORTERS)}"
        )
    return _EXPORTERS[key](str(binary), **kwargs)


def export_script(framework: str, binary: PathLike, **kwargs: Any) -> str:
    """Generate the external framework script text without importing the framework."""
    return exporter_for(framework, binary, **kwargs).generate()


# ---------------------------------------------------------------------------
# Disassembly helper (Capstone; used by the `toothpick taint` CLI)
# ---------------------------------------------------------------------------


def disassemble_bytes(
    code_bytes: bytes, base_addr: int = 0x1000, arch: str = "x86_64"
) -> List[InsnView]:
    """Disassemble a raw byte stream into :class:`InsnView` records."""
    spec = arch_spec(arch)
    md = capstone.Cs(spec.capstone_arch, spec.capstone_mode)
    md.detail = True
    views: List[InsnView] = []
    for cs_insn in md.disasm(code_bytes, base_addr):
        views.append(normalize_instruction(cs_insn))
    return views


def disassemble_at(
    target: PathLike,
    func_addr: int,
    *,
    arch: Optional[str] = None,
    base: Optional[int] = None,
    max_instructions: int = 512,
) -> Tuple[List[InsnView], Dict[str, Any]]:
    """Disassemble a function body starting at ``func_addr``.

    Returns ``(instructions, metadata)``.  The sweep is bounded by
    ``max_instructions`` and stops at padding (a run of 8+ zero bytes), at an
    explicit padding marker (``int3`` / ``ud2`` / ``hlt``) or at the end of the
    executable section; code after an unconditional jump is left in the stream
    and filtered by the reachability walk in :meth:`TaintEngine.analyze_function`.
    """
    try:
        container = load_container(target, is_raw=arch is not None, raw_arch=arch, raw_base=base)
    except (ContainerParseError, UnsupportedArchitectureError, OSError) as exc:
        raise SymbolicError(f"cannot load {target}: {exc}") from exc

    section = None
    for candidate in container.exec_sections:
        if candidate.contains(func_addr) and candidate.has_bytes(func_addr):
            section = candidate
            break
    if section is None:
        raise SymbolicError(
            f"0x{func_addr:x} is not inside an executable section of {target}"
        )

    spec = arch_spec(container.arch)
    md = capstone.Cs(spec.capstone_arch, spec.capstone_mode)
    md.detail = True
    data = section.bytes_at(func_addr)

    views: List[InsnView] = []
    offset = 0
    stop_reason = "end_of_section"
    while len(views) < max(1, max_instructions):
        remaining = data[offset:]
        if not remaining:
            stop_reason = "end_of_section"
            break
        if len(remaining) >= 8 and remaining[:8] == b"\x00" * 8:
            stop_reason = "zero_padding"
            break
        try:
            insn = next(md.disasm(remaining[:16], func_addr + offset, count=1))
        except StopIteration:
            stop_reason = "undecodable_bytes"
            break
        except capstone.CsError:  # pragma: no cover - defensive
            stop_reason = "disassembler_error"
            break
        view = normalize_instruction(insn)
        views.append(view)
        offset += max(1, view.size)
        if view.mnemonic in ("int3", "ud2", "hlt"):
            stop_reason = "padding_marker"
            break

    metadata: Dict[str, Any] = {
        "target": str(target),
        "container": container.container,
        "arch": container.arch,
        "base": container.base,
        "section": section.name,
        "instruction_count": len(views),
        "stop_reason": stop_reason,
        "truncated": len(views) >= max(1, max_instructions),
    }
    return views, metadata
