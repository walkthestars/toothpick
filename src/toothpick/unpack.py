"""
Toothpick simple unpack pre-pass (unit W02).

Implements the pre-pass described by ``PLAN-W02-unpack.md`` and
``docs/design-document.md`` §2 ("Packer Triage & Simple De-obfuscation"):

* pass-through detection for already-unpacked containers,
* hard abort for hypervisor / VM packers (``PACKED_ABORT(VM_PROTECTED)``),
# w19 probe
* single-layer XOR loader de-obfuscation,
* Unicorn micro-emulation of the packer stub up to its tail jump, extracting the
  Original Entry Point (OEP) - and dumping the freshly written sections when the
  stub reconstructs the image in memory.

Recovery strategy for a ``packed_simple`` PE::

    static payload decompression  ->  file-level payload
    micro-emulation tail jump     ->  OEP (+ section snapshot when the stub wrote memory)

``UnpackMethod.UNICORN_TAILJUMP`` means the OEP came from micro-emulation;
``UnpackMethod.STATIC`` / ``UnpackMethod.STATIC_XOR`` are static (no-emulation)
recovery sub-methods (``STATIC_XOR`` is the single-layer XOR loader case).
"""

from __future__ import annotations

import bz2
import gzip
import lzma
import math
import zlib
import struct
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from toothpick.triage import PackedAbortError, TriageError

_UNICORN_AVAILABLE: bool = False

try:  # pragma: no cover - optional native dependency
    from unicorn import (
        UC_ARCH_X86,
        UC_HOOK_CODE,
        UC_HOOK_MEM_FETCH_UNMAPPED,
        UC_HOOK_MEM_WRITE,
        UC_MODE_32,
        UC_MODE_64,
        UC_PROT_ALL,
        Uc,
        UcError,
    )
    from unicorn.x86_const import (
        UC_X86_REG_EAX,
        UC_X86_REG_EIP,
        UC_X86_REG_ESP,
        UC_X86_REG_FS,
        UC_X86_REG_GDTR,
        UC_X86_REG_GS_BASE,
        UC_X86_REG_RAX,
        UC_X86_REG_RIP,
        UC_X86_REG_RSP,
    )

    _UNICORN_AVAILABLE = True
except Exception:  # pragma: no cover - unicorn==2.1.4 is a declared dependency
    _UNICORN_AVAILABLE = False


PAGE_SIZE = 0x1000

#: Instruction budget for packer-stub micro-emulation (design-document R7:
#: "Unicorn OEP loop >10M insns -> fail over to PACKED_ABORT(UNPACK_FAILED)").
DEFAULT_MAX_INSTRUCTIONS = 10_000_000

#: Hard cap for section snapshots pulled out of the emulator.
MAX_SECTION_DUMP = 64 * 1024 * 1024

UNPACK_FAILED = "PACKED_ABORT(UNPACK_FAILED)"
VM_PROTECTED = "PACKED_ABORT(VM_PROTECTED)"

#: Marker tables (mirroring ``toothpick.triage``).
VM_PROTECTION_MARKERS: Tuple[bytes, ...] = (
    b".vmp0",
    b".vmp1",
    b".vmp2",
    b"VMProtect",
    b"vmprotect",
    b".themida",
    b"Themida",
    b"themida",
    b"WinLicense",
    b"Denuvo",
    b"denuvo",
)

UPX_MARKERS: Tuple[bytes, ...] = (b"UPX0", b"UPX1", b"UPX2", b"UPX!")
PECOMPACT_MARKERS: Tuple[bytes, ...] = (b".pec1", b".pec2", b"PECompact2", b"PEC2")
ASPACK_MARKERS: Tuple[bytes, ...] = (b".aspack", b"ASPack", b".adata")
MPRESS_MARKERS: Tuple[bytes, ...] = (b"MPRESS1", b"MPRESS2")


class UnpackStatus(str, Enum):
    """Outcome of an unpack attempt."""

    UNPACKED = "unpacked"
    ALREADY_UNPACKED = "already_unpacked"
    FAILED = "failed"


class UnpackMethod(str, Enum):
    """How the artifact/OEP was recovered.

    ``STATIC`` and ``STATIC_XOR`` are the static family required by the plan's
    ``method: "static"|"unicorn_tailjump"`` contract; ``STATIC_XOR`` names the
    single-layer XOR loader sub-method.

    The serialized (wire) value always comes from :attr:`wire`, never from
    :attr:`value`: PLAN-W02 allows only ``"static"`` or ``"unicorn_tailjump"``
    on the wire, so ``STATIC_XOR`` serializes as ``"static"`` and its detail is
    carried by ``UnpackReport.method_detail`` / ``codec`` / ``xor_key``.
    """

    STATIC = "static"
    STATIC_XOR = "static_xor"
    UNICORN_TAILJUMP = "unicorn_tailjump"
    NONE = "none"

    @property
    def is_static(self) -> bool:
        return self in (UnpackMethod.STATIC, UnpackMethod.STATIC_XOR)

    @property
    def wire(self) -> str:
        """Plan-contract serialized value: ``"static"`` or ``"unicorn_tailjump"``."""
        if self.is_static:
            return UnpackMethod.STATIC.value
        return self.value


class UnpackError(TriageError):
    """Raised when the unpack pre-pass itself fails (not a VM-protection abort)."""


@dataclass
class UnpackReport:
    """Structured result of :func:`unpack_binary`."""

    status: UnpackStatus
    method: UnpackMethod = UnpackMethod.NONE
    method_detail: str = ""
    oep: Optional[int] = None
    packer: str = "none"
    target: str = ""
    output_path: Optional[str] = None
    codec: str = ""
    payload_size: int = 0
    xor_key: Optional[int] = None
    instruction_count: int = 0
    tailjumps: List[int] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.method_detail:
            self.method_detail = self.method.value

    @property
    def method_wire(self) -> str:
        """Plan-contract wire value: ``"static"`` or ``"unicorn_tailjump"``."""
        return self.method.wire

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "method": self.method.wire,
            "method_detail": self.method_detail,
            "oep": self.oep,
            "packer": self.packer,
            "target": self.target,
            "output_path": self.output_path,
            "codec": self.codec,
            "payload_size": self.payload_size,
            "xor_key": self.xor_key,
            "instruction_count": self.instruction_count,
            "tailjumps": list(self.tailjumps),
            "notes": list(self.notes),
        }

    def to_json(self, indent: int = 2) -> str:
        import json

        return json.dumps(self.to_dict(), indent=indent)


# --------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------


def _align_up(value: int, alignment: int) -> int:
    if alignment is None or alignment <= 0:
        return int(value)
    return (int(value) + alignment - 1) // alignment * alignment


def _detect_marker(data: bytes, markers: Tuple[bytes, ...]) -> Optional[bytes]:
    for marker in markers:
        if marker in data:
            return marker
    return None


def _abort_if_vm_protected(data: bytes) -> None:
    """Hypervisor/VM packers are out of scope for V1 -> hard abort."""
    if _detect_marker(data, VM_PROTECTION_MARKERS) is not None:
        raise PackedAbortError(VM_PROTECTED)


def _detect_packer(data: bytes) -> Optional[str]:
    if _detect_marker(data, UPX_MARKERS) is not None:
        return "upx"
    if _detect_marker(data, PECOMPACT_MARKERS) is not None:
        return "pecompact"
    if _detect_marker(data, ASPACK_MARKERS) is not None:
        return "aspack"
    if _detect_marker(data, MPRESS_MARKERS) is not None:
        return "mpress"
    if not data.startswith(b"MZ") and _detect_xor_key(data) is not None:
        return "xor"
    return None


# --------------------------------------------------------------------------------------
# single-layer XOR loaders
# --------------------------------------------------------------------------------------


def _detect_xor_key(data: bytes) -> Optional[int]:
    """Guess a single-byte XOR key whose plaintext starts with ``MZ``.

    A key ``k`` must satisfy ``data[0] ^ k == 'M'`` and ``data[1] ^ k == 'Z'``;
    two equations pin a unique candidate, so the check is exact rather than
    probabilistic. A structural PE check is applied when the decoded header
    carries a usable ``e_lfanew``.
    """
    if len(data) < 0x40 or data.startswith(b"MZ"):
        return None
    key = data[0] ^ 0x4D
    if data[1] ^ 0x5A != key:
        return None
    if not 0 <= key <= 0xFF:
        return None
    if not _xor_key_plausible(data, key):
        return None
    return key


def _xor_key_plausible(data: bytes, key: int) -> bool:
    head = bytes(b ^ key for b in data[:0x40])
    if head[:2] != b"MZ":
        return False
    e_lfanew = struct.unpack_from("<I", head, 0x3C)[0]
    if 0 < e_lfanew < 0x10000000 and e_lfanew + 4 <= len(data):
        signature = bytes(b ^ key for b in data[e_lfanew : e_lfanew + 4])
        if signature != b"PE\x00\x00":
            return False
    return True


def _unpack_xor(
    target_path: Path,
    data: bytes,
    key: int,
    output_path: Optional[Path],
    explicit: bool,
) -> UnpackReport:
    key &= 0xFF
    plain = bytes(b ^ key for b in data)
    report = UnpackReport(
        status=UnpackStatus.UNPACKED,
        method=UnpackMethod.STATIC_XOR,
        packer="xor",
        target=str(target_path),
        codec=f"xor1(0x{key:02x})",
        payload_size=len(plain),
        xor_key=key,
        notes=[
            "single-layer XOR loader de-obfuscated"
            + (" (key supplied by caller)" if explicit else " (key auto-detected)"),
        ],
    )
    pe = parse_pe(plain)
    if pe is not None and pe.entry_point_rva:
        report.oep = pe.image_base + pe.entry_point_rva
        report.notes.append("OEP read from de-obfuscated PE headers")
    else:
        report.notes.append("de-obfuscated payload has no parseable PE entry point")
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(plain)
        report.output_path = str(output_path)
    return report


# --------------------------------------------------------------------------------------
# PE container parsing (structural, tolerant of truncated fixtures)
# --------------------------------------------------------------------------------------


@dataclass
class PESection:
    index: int
    name: str
    virtual_size: int
    virtual_address: int
    raw_size: int
    raw_ptr: int
    characteristics: int
    header_offset: int


@dataclass
class PEImage:
    pe_offset: int
    machine: int
    num_sections: int
    optional_size: int
    magic: int
    entry_point_rva: int
    image_base: int
    file_alignment: int
    section_alignment: int
    size_of_image: int
    section_table_offset: int
    sections: List[PESection]

    @property
    def is_64(self) -> bool:
        return self.magic == 0x20B

    @property
    def header_end(self) -> int:
        return self.section_table_offset + 40 * self.num_sections

    def section_containing_rva(self, rva: int) -> Optional[PESection]:
        for section in self.sections:
            span = max(section.virtual_size, section.raw_size)
            if section.virtual_address <= rva < section.virtual_address + span:
                return section
        return None


def parse_pe(data: bytes) -> Optional[PEImage]:
    """Parse PE headers/section table; returns ``None`` when unusable."""
    if len(data) < 0x40 or data[:2] != b"MZ":
        return None
    e_lfanew = struct.unpack_from("<I", data, 0x3C)[0]
    if e_lfanew <= 0 or e_lfanew + 24 > len(data):
        return None
    if data[e_lfanew : e_lfanew + 4] != b"PE\x00\x00":
        return None

    machine, num_sections = struct.unpack_from("<HH", data, e_lfanew + 4)
    if not 0 < num_sections <= 96:
        return None
    optional_size = struct.unpack_from("<H", data, e_lfanew + 20)[0]
    optional_offset = e_lfanew + 24
    section_table_offset = optional_offset + optional_size
    if section_table_offset + 40 * num_sections > len(data):
        return None

    magic = 0
    entry_point_rva = 0
    image_base = 0
    file_alignment = 0x200
    section_alignment = 0x1000
    size_of_image = 0
    if optional_size >= 0x40:
        magic = struct.unpack_from("<H", data, optional_offset)[0]
        entry_point_rva = struct.unpack_from("<I", data, optional_offset + 16)[0]
        if magic == 0x20B:
            image_base = struct.unpack_from("<Q", data, optional_offset + 24)[0]
        elif magic == 0x10B:
            image_base = struct.unpack_from("<I", data, optional_offset + 28)[0]
        section_alignment = struct.unpack_from("<I", data, optional_offset + 32)[0]
        file_alignment = struct.unpack_from("<I", data, optional_offset + 36)[0]
        size_of_image = struct.unpack_from("<I", data, optional_offset + 56)[0]
    if file_alignment <= 0 or file_alignment > 0x10000:
        file_alignment = 0x200
    if section_alignment <= 0 or section_alignment > 0x10000:
        section_alignment = 0x1000

    sections: List[PESection] = []
    for index in range(num_sections):
        header_offset = section_table_offset + 40 * index
        raw_name = data[header_offset : header_offset + 8]
        name = raw_name.split(b"\x00", 1)[0].decode("latin-1", "replace")
        virtual_size, virtual_address, raw_size, raw_ptr = struct.unpack_from(
            "<IIII", data, header_offset + 8
        )
        characteristics = struct.unpack_from("<I", data, header_offset + 36)[0]
        sections.append(
            PESection(
                index=index,
                name=name,
                virtual_size=virtual_size,
                virtual_address=virtual_address,
                raw_size=raw_size,
                raw_ptr=raw_ptr,
                characteristics=characteristics,
                header_offset=header_offset,
            )
        )

    return PEImage(
        pe_offset=e_lfanew,
        machine=machine,
        num_sections=num_sections,
        optional_size=optional_size,
        magic=magic,
        entry_point_rva=entry_point_rva,
        image_base=image_base,
        file_alignment=file_alignment,
        section_alignment=section_alignment,
        size_of_image=size_of_image,
        section_table_offset=section_table_offset,
        sections=sections,
    )


# --------------------------------------------------------------------------------------
# payload decompression (UPX pack-header guided, codec auto-detected)
# --------------------------------------------------------------------------------------


def _payload_plausible(payload: Optional[bytes], expected_len: Optional[int] = None) -> bool:
    if not payload:
        return False
    if expected_len:
        return abs(len(payload) - expected_len) <= 8
    return payload[:2] == b"MZ" or len(payload) >= 512


def _deflate_raw(blob: bytes) -> bytes:
    decompressor = zlib.decompressobj(-15)
    return decompressor.decompress(bytes(blob))


def _lzma_raw(blob: bytes) -> bytes:
    filters = [{"id": lzma.FILTER_LZMA1, "dict_size": 1 << 24, "lc": 3, "lp": 0, "pb": 2}]
    decompressor = lzma.LZMADecompressor(format=lzma.FORMAT_RAW, filters=filters)
    return decompressor.decompress(bytes(blob))


#: UPX pack-header compression methods that use an NRV2* bitstream.
NRV2_METHODS = {2: "nrv2b", 5: "nrv2d", 8: "nrv2e"}


def _decode_with(codec: str, blob: bytes) -> Optional[bytes]:
    """Best-effort single-codec decode. Raises nothing: returns ``None`` on failure."""
    try:
        if codec == "zlib":
            return zlib.decompress(bytes(blob))
        if codec == "gzip":
            return gzip.decompress(bytes(blob))
        if codec == "bzip2":
            return bz2.decompress(bytes(blob))
        if codec == "lzma":
            return lzma.decompress(bytes(blob))
        if codec == "lzma_raw":
            return _lzma_raw(blob)
        if codec == "deflate_raw":
            return _deflate_raw(blob)
        if codec in ("nrv2b", "nrv2d", "nrv2e"):
            return _nrv2_decode(bytes(blob), codec)
    except Exception:  # noqa: BLE001 - any codec mismatch is a non-fatal miss
        return None
    return None


def _decompress_payload(
    blob: bytes,
    method_hint: Optional[int] = None,
    expected_len: Optional[int] = None,
) -> Tuple[str, Optional[bytes]]:
    """Try the pack-header method first, then a fixed codec ladder."""
    ladder: List[str] = []
    hint_codec = NRV2_METHODS.get(method_hint) if method_hint else None
    if hint_codec:
        ladder.append(hint_codec)
    ladder.extend(
        ["zlib", "gzip", "lzma", "lzma_raw", "bzip2", "deflate_raw", "nrv2b", "nrv2d", "nrv2e"]
    )
    for codec in ladder:
        decoded = _decode_with(codec, blob)
        if _payload_plausible(decoded, expected_len):
            return codec, decoded
    return "", None


def _scan_decompress(data: bytes, max_output: int = 64 * 1024 * 1024) -> Tuple[str, Optional[bytes]]:
    """Last-resort scan: locate a zlib/gzip/lzma stream that yields a PE image."""
    for offset in range(0, len(data) - 2):
        cmf = data[offset]
        flg = data[offset + 1]
        if cmf & 0x0F == 8 and ((cmf << 8) | flg) % 31 == 0:
            decoded = _decode_with("zlib", data[offset:])
            if decoded and len(decoded) < max_output and _payload_plausible(decoded):
                return "zlib", decoded
        if data[offset : offset + 2] == b"\x1f\x8b":
            decoded = _decode_with("gzip", data[offset:])
            if decoded and len(decoded) < max_output and _payload_plausible(decoded):
                return "gzip", decoded
        if data[offset : offset + 6] == b"\xfd7zXZ\x00":
            decoded = _decode_with("lzma", data[offset:])
            if decoded and len(decoded) < max_output and _payload_plausible(decoded):
                return "lzma", decoded
    return "", None


class _NRV2BitReader:
    """MSB-first bit reader used by the NRV2B/2D/2E packer streams."""

    def __init__(self, blob: bytes) -> None:
        self.blob = blob
        self.byte_pos = 0
        self.bits = 0
        self.bit_count = 0

    def bit(self) -> int:
        if self.bit_count == 0:
            if self.byte_pos >= len(self.blob):
                raise EOFError("nrv2: bitstream exhausted")
            self.bits = self.blob[self.byte_pos]
            self.byte_pos += 1
            self.bit_count = 8
        self.bit_count -= 1
        return (self.bits >> self.bit_count) & 1

    def bits_n(self, count: int) -> int:
        value = 0
        for _ in range(count):
            value = (value << 1) | self.bit()
        return value

    def byte(self) -> int:
        if self.byte_pos >= len(self.blob):
            raise EOFError("nrv2: literal stream exhausted")
        value = self.blob[self.byte_pos]
        self.byte_pos += 1
        return value


def _nrv2_decode(blob: bytes, variant: str) -> Optional[bytes]:
    """NRV2B/2D/2E decoder used by UPX stubs.

    Structural implementation of the UCL/UPX NRV2 core (bit flags, m_off
    encoding, extended match length, backwards byte copy). ``variant`` selects
    the NRV2B (`m_off == 2` reuses the last output byte), NRV2D (`delta = 1`)
    and NRV2E ("delta from two extra bits") offset refinements.
    """
    if not blob:
        return None
    reader = _NRV2BitReader(blob)
    out = bytearray()
    try:
        while True:
            # literal run
            while reader.bit():
                out.append(reader.byte())
            # offset
            m_off = 1
            while True:
                m_off = (m_off << 1) + reader.bit()
                if not reader.bit():
                    break
            if m_off == 2:
                if variant == "nrv2b":
                    m_off = out[-1] + 1 if out else 1
                elif variant == "nrv2d":
                    m_off = reader.bits_n(8) + 1
                else:  # nrv2e
                    m_off = reader.bits_n(8) + 1
                    if m_off == 1:
                        m_off = out[-1] + 1 if out else 1
            else:
                m_off = (m_off - 3) * 256 + reader.byte()
            if m_off == 0 or m_off == 0xFFFFFFFF:
                break  # end marker
            # match length
            m_len = reader.bit()
            m_len = (m_len << 1) + reader.bit()
            if m_len == 0:
                m_len = 1
                while True:
                    m_len = (m_len << 1) + reader.bit()
                    if not reader.bit():
                        break
                m_len += 2
            m_len += 1
            if m_off > len(out) or m_off <= 0:
                return None
            start = len(out) - m_off
            for index in range(m_len):
                out.append(out[start + index])
    except (EOFError, IndexError):
        return None
    return bytes(out) if out else None


@dataclass
class UPXInfo:
    offset: int
    version: int
    format: int
    method: int
    level: int
    original_size: int
    compressed_size: int
    original_file_size: int


def _find_upx_info(window: bytes, base_offset: int = 0) -> Optional[UPXInfo]:
    """Locate and validate the 32-byte ``UPX!`` pack header inside ``window``."""
    search = 0
    while True:
        found = window.find(b"UPX!", search)
        if found < 0:
            return None
        search = found + 1
        if found + 32 > len(window):
            continue
        version = window[found + 4]
        fmt = window[found + 5]
        method = window[found + 6]
        level = window[found + 7]
        original_size = struct.unpack_from("<I", window, found + 16)[0]
        compressed_size = struct.unpack_from("<I", window, found + 20)[0]
        original_file_size = struct.unpack_from("<I", window, found + 24)[0]
        if not 8 <= version <= 24:
            continue
        if not 1 <= fmt <= 64:
            continue
        if not (1 <= method <= 32 or method in NRV2_METHODS):
            continue
        if level > 10:
            continue
        if not 0 < original_size < (1 << 31):
            continue
        return UPXInfo(
            offset=base_offset + found,
            version=version,
            format=fmt,
            method=method,
            level=level,
            original_size=original_size,
            compressed_size=compressed_size,
            original_file_size=original_file_size,
        )


def _recover_upx_payload(
    data: bytes, pe: PEImage
) -> Tuple[Optional[bytes], str, List[str], Optional[UPXInfo]]:
    """Static UPX payload recovery: pack header guided, with a scan fallback."""
    notes: List[str] = []
    upx_sections = [s for s in pe.sections if s.name.upper().startswith("UPX")]
    if not upx_sections:
        return None, "", ["no UPX section headers found for static recovery"], None

    compressed_section = _compressed_section(pe, upx_sections)
    info: Optional[UPXInfo] = None
    if compressed_section is not None and compressed_section.raw_size:
        start = compressed_section.raw_ptr
        window = data[start : start + compressed_section.raw_size]
        info = _find_upx_info(window, base_offset=start)
        if info is not None:
            notes.append(
                f"UPX pack header found at 0x{info.offset:x} "
                f"(version={info.version}, method={info.method}, "
                f"u_len={info.original_size}, c_len={info.compressed_size})"
            )
        candidates: List[Tuple[str, bytes, Optional[int]]] = []
        if info is not None and 0 < info.compressed_size <= len(data) - start:
            candidates.append(
                (f"pack-header c_len={info.compressed_size}", data[start : start + info.compressed_size], info.method)
            )
        candidates.append(("raw section", window, info.method if info else None))
        expected = info.original_size if info is not None else None
        for label, blob, method_hint in candidates:
            codec, payload = _decompress_payload(blob, method_hint, expected)
            if payload is not None:
                notes.append(f"payload decompressed from {label} using {codec}")
                if expected and len(payload) != expected:
                    notes.append(
                        f"decoded size {len(payload)} differs from pack header u_len {expected}"
                    )
                return payload, codec, notes, info

    codec, payload = _scan_decompress(data)
    if payload is not None:
        notes.append(f"payload located by generic scan using {codec}")
        return payload, codec, notes, info
    notes.append("no payload could be decompressed from the UPX container")
    return None, "", notes, info


def _compressed_section(pe: PEImage, upx_sections: List[PESection]) -> Optional[PESection]:
    """Section holding the compressed data + stub (EP-bearing UPX section)."""
    entry = pe.section_containing_rva(pe.entry_point_rva) if pe.entry_point_rva else None
    if entry is not None and entry.name.upper().startswith("UPX"):
        return entry
    with_raw = [s for s in upx_sections if s.raw_size]
    if with_raw:
        return max(with_raw, key=lambda s: s.virtual_address)
    return None


# --------------------------------------------------------------------------------------
# Unicorn micro-emulation
# --------------------------------------------------------------------------------------


@dataclass
class EmulationResult:
    oep: Optional[int]
    instruction_count: int
    tailjumps: List[int]
    sections_snapshot: Dict[int, bytes]
    stop_reason: str


def _setup_teb_peb(mu: "Uc", is_64: bool, image_base: int) -> int:
    """Set up simulated Windows TEB/PEB pages and segment register (FS/GS).

    Allows packer stubs that query the PEB (e.g. for ImageBase, BeingDebugged,
    or API tables) via fs:[0x30] (x86) or gs:[0x60] (x64) to execute without faulting.
    """
    teb_addr = 0x7F010000 if is_64 else 0x70010000
    peb_addr = teb_addr + 0x800
    try:
        mu.mem_map(teb_addr, PAGE_SIZE, UC_PROT_ALL)
    except Exception:
        pass  # already mapped

    if is_64:
        teb_data = bytearray(PAGE_SIZE)
        struct.pack_into("<Q", teb_data, 0x30, teb_addr)  # Self
        struct.pack_into("<Q", teb_data, 0x60, peb_addr)  # ProcessEnvironmentBlock
        struct.pack_into("<B", teb_data, 0x800 + 2, 0)     # BeingDebugged = 0
        struct.pack_into("<Q", teb_data, 0x800 + 0x10, image_base)  # ImageBaseAddress
        try:
            mu.mem_write(teb_addr, bytes(teb_data))
            mu.reg_write(UC_X86_REG_GS_BASE, teb_addr)
        except Exception:
            pass
    else:
        teb_data = bytearray(PAGE_SIZE)
        struct.pack_into("<I", teb_data, 0x18, teb_addr)  # Self
        struct.pack_into("<I", teb_data, 0x30, peb_addr)  # ProcessEnvironmentBlock
        struct.pack_into("<B", teb_data, 0x800 + 2, 0)     # BeingDebugged = 0
        struct.pack_into("<I", teb_data, 0x800 + 8, image_base)  # ImageBaseAddress
        try:
            mu.mem_write(teb_addr, bytes(teb_data))
            gdt_addr = 0xC0000000
            try:
                mu.mem_map(gdt_addr, PAGE_SIZE, UC_PROT_ALL)
            except Exception:
                pass
            low = 0xFFF | ((teb_addr & 0xFFFF) << 16)
            high = ((teb_addr >> 16) & 0xFF) | (0xF2 << 8) | (((teb_addr >> 24) & 0xFF) << 24) | 0x400000
            fs_entry = struct.pack("<II", low, high)
            gdt_bytes = (b"\x00" * 8 * 7) + fs_entry
            mu.mem_write(gdt_addr, gdt_bytes)
            mu.reg_write(UC_X86_REG_GDTR, (0, gdt_addr, len(gdt_bytes) - 1, 0))
            mu.reg_write(UC_X86_REG_FS, (7 << 3) | 3)
        except Exception:
            pass
    return teb_addr


def emulate_tailjump_oep(
    code: bytes,
    load_addr: int,
    stub_start: int,
    stub_size: int,
    max_instructions: int = DEFAULT_MAX_INSTRUCTIONS,
    arch: str = "x86_64",
    stack_base: Optional[int] = None,
    stack_size: int = 0x20000,
) -> int:
    """Emulate a packer stub until it jumps out of ``[stub_start, stub_start+stub_size)``.

    The first instruction fetch outside the stub window (or an unmapped fetch,
    which is how a tail jump into a not-yet-mapped OEP surfaces) is the tail
    jump; its target is returned as the Original Entry Point.
    """
    if not _UNICORN_AVAILABLE:  # pragma: no cover - dependency is declared
        raise UnpackError("UNICORN_UNAVAILABLE: unicorn-engine is not installed")

    is_64 = str(arch).lower() in ("x86_64", "x64", "amd64", "64")
    mode = UC_MODE_64 if is_64 else UC_MODE_32
    reg_sp = UC_X86_REG_RSP if is_64 else UC_X86_REG_ESP
    reg_ip = UC_X86_REG_RIP if is_64 else UC_X86_REG_EIP

    code = bytes(code)
    mu = Uc(UC_ARCH_X86, mode)
    map_start = load_addr & ~(PAGE_SIZE - 1)
    region = max(len(code), int(stub_size), PAGE_SIZE)
    mu.mem_map(map_start, _align_up(load_addr + region - map_start, PAGE_SIZE), UC_PROT_ALL)
    if code:
        mu.mem_write(load_addr, code)

    if stack_base is None:
        stack_base = 0x7F000000 if is_64 else 0x70000000
    stack_start = stack_base & ~(PAGE_SIZE - 1)
    mu.mem_map(stack_start, _align_up(stack_size, PAGE_SIZE), UC_PROT_ALL)
    mu.reg_write(reg_sp, (stack_start + stack_size - 0x2000) & ~0xF)
    _setup_teb_peb(mu, is_64, load_addr)

    stub_end = int(stub_start) + max(int(stub_size), 1)
    state: Dict[str, Any] = {"oep": None, "count": 0, "reason": "end"}

    def _on_code(uc: "Uc", address: int, size: int, user: Any) -> None:
        state["count"] += 1
        if state["count"] > max_instructions:
            state["reason"] = "budget"
            uc.emu_stop()
            return
        if address < stub_start or address >= stub_end:
            state["oep"] = address
            state["reason"] = "tailjump"
            uc.emu_stop()

    def _on_unmapped_fetch(uc: "Uc", access: int, address: int, size: int, value: int, user: Any) -> bool:
        state["oep"] = address
        state["reason"] = "tailjump_unmapped"
        uc.emu_stop()
        return False

    mu.hook_add(UC_HOOK_CODE, _on_code)
    mu.hook_add(UC_HOOK_MEM_FETCH_UNMAPPED, _on_unmapped_fetch)
    try:
        mu.emu_start(load_addr, 0, timeout=0, count=0)
    except UcError:
        pass

    oep = state["oep"]
    if oep is None:
        if state["reason"] == "budget":
            raise UnpackError(
                f"{UNPACK_FAILED}: emulation budget of {max_instructions} instructions "
                "exhausted before a tail jump (design-document R7)"
            )
        try:
            ip = mu.reg_read(reg_ip)
        except UcError:
            ip = None
        if ip is not None and not (stub_start <= ip < stub_end):
            oep = int(ip)
    if oep is None:
        raise UnpackError(f"{UNPACK_FAILED}: no tail jump detected inside the packer stub")
    return int(oep)


def _emulate_pe_stub(
    data: bytes,
    pe: PEImage,
    max_instructions: int = DEFAULT_MAX_INSTRUCTIONS,
) -> EmulationResult:
    """Map a PE image into Unicorn, run the packer stub, and recover OEP + written sections."""
    if not _UNICORN_AVAILABLE:  # pragma: no cover - dependency is declared
        raise UnpackError("UNICORN_UNAVAILABLE: unicorn-engine is not installed")
    if not pe.entry_point_rva:
        raise UnpackError(f"{UNPACK_FAILED}: packed PE has no entry point")

    is_64 = pe.is_64
    mode = UC_MODE_64 if is_64 else UC_MODE_32
    reg_sp = UC_X86_REG_RSP if is_64 else UC_X86_REG_ESP
    reg_ip = UC_X86_REG_RIP if is_64 else UC_X86_REG_EIP
    reg_ax = UC_X86_REG_RAX if is_64 else UC_X86_REG_EAX

    image_base = pe.image_base
    header_end = min(len(data), max(pe.header_end, 0x200))
    image_size = max(
        pe.size_of_image,
        _align_up(header_end, pe.section_alignment),
        PAGE_SIZE,
    )
    image_size = _align_up(image_size, pe.section_alignment)
    if image_size > MAX_SECTION_DUMP:
        raise UnpackError(f"{UNPACK_FAILED}: image size 0x{image_size:x} too large to emulate")

    mu = Uc(UC_ARCH_X86, mode)
    map_start = image_base & ~(PAGE_SIZE - 1)
    mu.mem_map(map_start, _align_up(image_base + image_size - map_start, PAGE_SIZE), UC_PROT_ALL)
    mu.mem_write(image_base, data[: min(header_end, image_size)])

    for section in pe.sections:
        if not section.raw_size or not section.virtual_size:
            continue
        if section.virtual_address >= image_size or section.raw_ptr >= len(data):
            continue
        available = min(section.raw_size, section.virtual_size + 0x1000, len(data) - section.raw_ptr)
        if available <= 0:
            continue
        chunk = data[section.raw_ptr : section.raw_ptr + available]
        chunk = chunk[: max(0, image_size - section.virtual_address)]
        if chunk:
            mu.mem_write(image_base + section.virtual_address, chunk)

    stack_base = 0x7F000000 if is_64 else 0x70000000
    stack_start = stack_base & ~(PAGE_SIZE - 1)
    mu.mem_map(stack_start, _align_up(0x20000, PAGE_SIZE), UC_PROT_ALL)
    mu.reg_write(reg_sp, (stack_start + 0x20000 - 0x2000) & ~0xF)
    mu.reg_write(reg_ax, 0)
    _setup_teb_peb(mu, is_64, image_base)

    entry_va = image_base + pe.entry_point_rva
    entry_section = pe.section_containing_rva(pe.entry_point_rva)
    if entry_section is not None:
        stub_start = image_base + entry_section.virtual_address
        stub_end = stub_start + max(entry_section.raw_size, entry_section.virtual_size, 0x200)
    else:
        stub_start = entry_va
        stub_end = entry_va + 0x200
    stub_end = min(stub_end, image_base + image_size)

    state: Dict[str, Any] = {"oep": None, "count": 0, "reason": "end", "tailjumps": []}
    written_pages: set = set()

    def _on_mem_write(uc: "Uc", access: int, address: int, size: int, value: int, user: Any) -> None:
        written_pages.add(address & ~(PAGE_SIZE - 1))

    def _on_code(uc: "Uc", address: int, size: int, user: Any) -> None:
        state["count"] += 1
        if state["count"] > max_instructions:
            state["reason"] = "budget"
            uc.emu_stop()
            return
        if address == entry_va:
            return
        outside_stub = not (stub_start <= address < stub_end)
        written_code = state["count"] > 8 and (address & ~(PAGE_SIZE - 1)) in written_pages
        if outside_stub or written_code:
            state["oep"] = address
            state["tailjumps"].append(address)
            state["reason"] = "tailjump" if outside_stub else "tailjump_written_code"
            uc.emu_stop()

    def _on_unmapped_fetch(uc: "Uc", access: int, address: int, size: int, value: int, user: Any) -> bool:
        state["oep"] = address
        state["tailjumps"].append(address)
        state["reason"] = "tailjump_unmapped"
        uc.emu_stop()
        return False

    mu.hook_add(UC_HOOK_CODE, _on_code)
    mu.hook_add(UC_HOOK_MEM_WRITE, _on_mem_write)
    mu.hook_add(UC_HOOK_MEM_FETCH_UNMAPPED, _on_unmapped_fetch)
    try:
        mu.emu_start(entry_va, 0, timeout=0, count=0)
    except UcError:
        pass

    oep = state["oep"]
    if oep is None and state["reason"] == "budget":
        raise UnpackError(
            f"{UNPACK_FAILED}: emulation budget of {max_instructions} instructions exhausted "
            "before a tail jump (design-document R7)"
        )
    if oep is None:
        try:
            ip = int(mu.reg_read(reg_ip))
        except UcError:
            ip = None
        if ip is not None and not (stub_start <= ip < stub_end):
            oep = ip
            state["tailjumps"].append(ip)
            state["reason"] = "tailjump_after_fault"
    if oep is None:
        raise UnpackError(f"{UNPACK_FAILED}: packer stub never tail-jumped to an OEP")

    snapshot: Dict[int, bytes] = {}
    for section in pe.sections:
        if not section.virtual_size or section.virtual_address >= image_size:
            continue
        section_start = image_base + section.virtual_address
        section_end = section_start + min(section.virtual_size, image_size - section.virtual_address)
        pages = range(section_start & ~(PAGE_SIZE - 1), section_end, PAGE_SIZE)
        if not any(page in written_pages for page in pages):
            continue
        length = min(section_end - section_start, MAX_SECTION_DUMP)
        try:
            snapshot[section.index] = bytes(mu.mem_read(section_start, length))
        except UcError:
            continue

    return EmulationResult(
        oep=int(oep),
        instruction_count=int(state["count"]),
        tailjumps=[int(addr) for addr in state["tailjumps"]],
        sections_snapshot=snapshot,
        stop_reason=str(state["reason"]),
    )


# --------------------------------------------------------------------------------------
# PE rebuild
# --------------------------------------------------------------------------------------


def _apply_upx_payload(
    pe: PEImage, payload: bytes, contents: Dict[int, bytes]
) -> List[str]:
    """Distribute a decompressed UPX block over the packer's output sections."""
    notes: List[str] = []
    upx_sections = [s for s in pe.sections if s.name.upper().startswith("UPX")]
    targets = [s for s in upx_sections if s.raw_size == 0 and s.virtual_size]
    if not targets:
        targets = [s for s in pe.sections if s.raw_size == 0 and s.virtual_size]
    targets.sort(key=lambda s: s.virtual_address)
    if not targets:
        notes.append("no uninitialized output section available; payload kept as overlay")
        contents[-1] = payload
        return notes

    total = sum(s.virtual_size for s in targets)
    if len(payload) > total:
        notes.append(f"payload ({len(payload)} bytes) exceeds output sections ({total} bytes); truncated")
    cursor = 0
    for section in targets:
        take = min(section.virtual_size, max(0, len(payload) - cursor))
        contents[section.index] = payload[cursor : cursor + take]
        cursor += take
    # Drop the packer's own (compressed + stub) section from the output file.
    for section in upx_sections:
        if section.raw_size and section.index not in contents:
            contents[section.index] = b""
    return notes


def _build_unpacked_pe(
    data: bytes,
    pe: PEImage,
    contents: Dict[int, bytes],
    new_ep_rva: Optional[int],
) -> bytes:
    """Rebuild a PE file from header bytes + per-section content.

    ``contents[index] = b""`` marks a section to be dropped (packer stub).
    Sections absent from ``contents`` keep their original raw bytes.
    """
    file_align = pe.file_alignment or 0x200
    out = bytearray(data[: pe.header_end])
    offset = _align_up(len(out), file_align)
    out.extend(b"\x00" * (offset - len(out)))

    for section in pe.sections:
        if section.index in contents:
            content = contents[section.index]
            if not content:
                raw_size, raw_ptr = 0, 0
            else:
                raw_size = _align_up(len(content), file_align)
                raw_ptr = offset
                out.extend(content)
                out.extend(b"\x00" * (raw_size - len(content)))
                offset += raw_size
        elif section.raw_size and section.raw_ptr + section.raw_size <= len(data):
            raw_size, raw_ptr = section.raw_size, offset
            out.extend(data[section.raw_ptr : section.raw_ptr + section.raw_size])
            offset += raw_size
        else:
            raw_size, raw_ptr = 0, 0

        struct.pack_into("<II", out, section.header_offset + 16, raw_size, raw_ptr)
        characteristics = struct.unpack_from("<I", out, section.header_offset + 36)[0]
        if raw_size:
            characteristics &= ~0x00000080
        else:
            characteristics |= 0x00000080
        struct.pack_into("<I", out, section.header_offset + 36, characteristics)

    # Always compute and update SizeOfImage
    size_of_image = 0
    for section in pe.sections:
        content = contents.get(section.index)
        span = max(section.virtual_size, len(content) if content is not None else section.raw_size)
        size_of_image = max(size_of_image, section.virtual_address + span)
    size_of_image = max(size_of_image, pe.header_end)
    size_of_image = _align_up(size_of_image, pe.section_alignment or PAGE_SIZE)
    struct.pack_into("<I", out, pe.pe_offset + 24 + 56, size_of_image)

    if new_ep_rva is not None:
        struct.pack_into("<I", out, pe.pe_offset + 24 + 16, new_ep_rva & 0xFFFFFFFF)

    # Clean up Optional Header Data Directories: zero out dangling pointers to dropped packer sections
    opt_off = pe.pe_offset + 24
    if opt_off + 2 <= len(out):
        magic = struct.unpack_from("<H", out, opt_off)[0]
        dd_start = opt_off + (112 if magic == 0x20B else 96)
        opt_size = struct.unpack_from("<H", out, pe.pe_offset + 20)[0] if pe.pe_offset + 22 <= len(out) else 0
        rva_sizes_off = opt_off + (108 if magic == 0x20B else 92)
        num_dirs = struct.unpack_from("<I", out, rva_sizes_off)[0] if rva_sizes_off + 4 <= len(out) else 0
        if num_dirs == 0 and opt_size > (112 if magic == 0x20B else 96):
            num_dirs = min(16, (opt_size - (112 if magic == 0x20B else 96)) // 8)
        if num_dirs:
            dropped_spans = []
            for sec in pe.sections:
                if contents.get(sec.index) == b"":
                    dropped_spans.append((sec.virtual_address, sec.virtual_address + max(sec.virtual_size, sec.raw_size, 1)))

            for i in range(min(num_dirs, 16)):
                dir_entry_off = dd_start + i * 8
                if dir_entry_off + 8 <= len(out):
                    dir_rva, dir_sz = struct.unpack_from("<II", out, dir_entry_off)
                    if dir_rva:
                        in_dropped = any(start <= dir_rva < end for start, end in dropped_spans)
                        if in_dropped or dir_rva >= size_of_image:
                            struct.pack_into("<II", out, dir_entry_off, 0, 0)
    return bytes(out)


def _oep_rva(pe: PEImage, oep: Optional[int]) -> Tuple[Optional[int], List[str]]:
    notes: List[str] = []
    if oep is None:
        return None, notes
    rva = oep - pe.image_base
    span = max(pe.size_of_image, max((s.virtual_address + max(s.virtual_size, s.raw_size) for s in pe.sections), default=0))
    if 0 < rva < span:
        return rva, notes
    notes.append(
        f"emulated tail jump target 0x{oep:x} is outside the image; OEP left unset"
    )
    return None, notes


# --------------------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------------------


def _unpack_packed_simple(
    target_path: Path,
    data: bytes,
    pe: PEImage,
    packer: str,
    output_path: Optional[Path],
    emulate_stub: bool,
    max_instructions: int,
) -> UnpackReport:
    notes: List[str] = []
    payload: Optional[bytes] = None
    codec = ""
    info: Optional[UPXInfo] = None

    if packer == "upx":
        payload, codec, recover_notes, info = _recover_upx_payload(data, pe)
        notes.extend(recover_notes)

    oep: Optional[int] = None
    snapshot: Dict[int, bytes] = {}
    instruction_count = 0
    tailjumps: List[int] = []
    if emulate_stub:
        try:
            emulation = _emulate_pe_stub(data, pe, max_instructions)
            oep = emulation.oep
            snapshot = emulation.sections_snapshot
            instruction_count = emulation.instruction_count
            tailjumps = emulation.tailjumps
            notes.append(
                f"micro-emulation ran {instruction_count} instructions; "
                f"tail jump -> 0x{oep:x} ({emulation.stop_reason})"
            )
            if snapshot:
                notes.append(
                    "stub wrote "
                    + ", ".join(f"{pe.sections[i].name or i}" for i in sorted(snapshot))
                    + " in memory; section snapshot used for the unpacked image"
                )
        except UnpackError as exc:
            notes.append(f"micro-emulation unavailable: {exc}")

    if payload is None and not snapshot:
        raise PackedAbortError(UNPACK_FAILED)

    contents: Dict[int, bytes] = dict(snapshot)
    if payload is not None:
        contents.update({idx: chunk for idx, chunk in snapshot.items()})
        if snapshot:
            notes.append("static payload available; section snapshot takes precedence")
        else:
            notes.extend(_apply_upx_payload(pe, payload, contents))
    elif snapshot:
        for section in pe.sections:
            if section.raw_size and section.index not in contents:
                pass

    new_ep_rva, oep_notes = _oep_rva(pe, oep)
    notes.extend(oep_notes)
    if new_ep_rva is None and payload is not None:
        targets = [s for s in pe.sections if s.raw_size == 0 and s.virtual_size]
        if targets:
            new_ep_rva = min(targets, key=lambda s: s.virtual_address).virtual_address
            notes.append(
                f"OEP not recovered; entry point set to first output section RVA "
                f"0x{new_ep_rva:x}"
            )

    output_bytes = _build_unpacked_pe(data, pe, contents, new_ep_rva)
    method = UnpackMethod.UNICORN_TAILJUMP if oep is not None else UnpackMethod.STATIC
    report = UnpackReport(
        status=UnpackStatus.UNPACKED,
        method=method,
        oep=oep,
        packer=packer,
        target=str(target_path),
        codec=codec,
        payload_size=len(payload) if payload is not None else sum(len(v) for v in snapshot.values()),
        instruction_count=instruction_count,
        tailjumps=tailjumps,
        notes=notes,
    )
    if info is not None:
        report.notes.append(
            f"pack header reports original size {info.original_size} bytes "
            f"(version {info.version}, method {info.method})"
        )
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(output_bytes)
        report.output_path = str(output_path)
    return report


def unpack_binary(
    target_path: Path,
    output_path: Optional[Path] = None,
    xor_key: Optional[int] = None,
    *,
    auto_xor: bool = True,
    emulate_stub: bool = True,
    max_instructions: int = DEFAULT_MAX_INSTRUCTIONS,
) -> UnpackReport:
    """Simple unpack pre-pass entry point.

    Args:
        target_path: binary to inspect.
        output_path: optional path for the unpacked artifact.
        xor_key: explicit single-byte key for a single-layer XOR loader
            (auto-detected when omitted).
        auto_xor: allow XOR key auto-detection.
        emulate_stub: use Unicorn micro-emulation for OEP extraction.
        max_instructions: micro-emulation instruction budget.

    Raises:
        PackedAbortError: ``PACKED_ABORT(VM_PROTECTED)`` for VM-protected targets,
            ``PACKED_ABORT(UNPACK_FAILED)`` when a simple packer cannot be undone.
        UnpackError: on pre-pass internal failures (e.g. Unicorn unavailable).
    """
    target_path = Path(target_path)
    if not target_path.exists():
        raise FileNotFoundError(f"Target file not found: {target_path}")
    data = target_path.read_bytes()
    if not data:
        raise UnpackError(f"{UNPACK_FAILED}: empty target")

    _abort_if_vm_protected(data)

    if xor_key is not None:
        return _unpack_xor(target_path, data, int(xor_key), output_path, explicit=True)

    packer = _detect_packer(data)
    if packer == "xor":
        key = _detect_xor_key(data)
        if key is not None and auto_xor:
            return _unpack_xor(target_path, data, key, output_path, explicit=False)
        packer = None

    if packer in ("upx", "pecompact", "aspack", "mpress"):
        pe = parse_pe(data)
        if pe is None:
            raise PackedAbortError(
                f"{UNPACK_FAILED}: {packer} container is not a parseable PE image"
            )
        return _unpack_packed_simple(
            target_path, data, pe, packer, output_path, emulate_stub, max_instructions
        )

    # Pass-through: nothing to undo.
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(data)
    return UnpackReport(
        status=UnpackStatus.ALREADY_UNPACKED,
        method=UnpackMethod.NONE,
        packer="none",
        target=str(target_path),
        output_path=str(output_path) if output_path is not None else None,
        payload_size=len(data),
        notes=["no supported simple packer detected; target passes through unchanged"],
    )


# --------------------------------------------------------------------------------------
# W19: advanced VM-protection triage, OEP hunting, and PE dump reconstruction
# --------------------------------------------------------------------------------------

#: Sections whose Shannon entropy reaches this bound are flagged as virtualized/encrypted.
VM_ENTROPY_THRESHOLD = 7.20

#: Hard cap on reported VM dispatcher candidates (context budget).
MAX_VM_DISPATCHER_CANDIDATES = 64

#: Hard cap on reported OEP candidates.
MAX_OEP_CANDIDATES = 64

#: Bytes at the head of a section that count as a section transition for OEP fingerprinting.
OEP_TRANSITION_WINDOW = 0x40

#: How far back a VM dispatcher jump looks for dispatch-context instructions.
DISPATCHER_CONTEXT_LOOKBACK = 12


class VmProtectionType(str, Enum):
    """Known commercial VM protectors triaged by :class:`VmDetector`."""

    DENUVO = "denuvo"
    THEMIDA = "themida"
    VMPROTECT = "vmprotect"
    ENIGMA = "enigma"
    STEAM_STUB = "steam_stub"
    UNKNOWN = "unknown"


@dataclass
class SectionEntropyReport:
    """Per-section Shannon-entropy profile."""

    name: str
    virtual_address: int
    raw_size: int
    entropy: float
    is_encrypted_or_virtualized: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "virtual_address": self.virtual_address,
            "raw_size": self.raw_size,
            "entropy": round(self.entropy, 4),
            "is_encrypted_or_virtualized": self.is_encrypted_or_virtualized,
        }


@dataclass
class VmDetectionReport:
    """Structured VM-protection triage result (PLAN W19 section 2)."""

    protector: VmProtectionType
    confidence: float
    sections: List[SectionEntropyReport] = field(default_factory=list)
    vm_dispatcher_candidates: List[int] = field(default_factory=list)
    suggested_action: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "protector": self.protector.value,
            "confidence": round(self.confidence, 4),
            "sections": [section.to_dict() for section in self.sections],
            "vm_dispatcher_candidates": list(self.vm_dispatcher_candidates),
            "suggested_action": self.suggested_action,
        }

    def to_json(self, indent: int = 2) -> str:
        import json

        return json.dumps(self.to_dict(), indent=indent)


def _shannon_entropy(data: bytes) -> float:
    """Shannon entropy ``H = -sum p_i * log2(p_i)`` in bits/byte (``0.0`` when empty)."""
    blob = bytes(data or b"")
    if not blob:
        return 0.0
    counts = [0] * 256
    for byte in blob:
        counts[byte] += 1
    length = len(blob)
    entropy = 0.0
    for count in counts:
        if count:
            probability = count / length
            entropy -= probability * math.log2(probability)
    return entropy


def compute_shannon_entropy(data: bytes) -> float:
    """Public W19 entropy primitive (uniform data -> ``0.0``, random -> ~``8.0``)."""
    return _shannon_entropy(data)


def _section_window(data: bytes, pe: Optional[PEImage], section: PESection) -> bytes:
    """Bytes backing ``section`` in either the on-disk or the memory-dump layout.

    A blob at least ``SizeOfImage`` long is treated as an image-aligned memory dump
    (section bytes live at their *virtual* addresses); anything shorter is treated as
    an on-disk PE (section bytes live at ``PointerToRawData``).
    """
    if pe is not None and pe.size_of_image and len(data) >= pe.size_of_image:
        start = section.virtual_address
        span = max(section.virtual_size, section.raw_size)
        if 0 <= start < len(data) and span > 0:
            return bytes(data[start : min(len(data), start + span)])
    if section.raw_ptr and section.raw_size and section.raw_ptr < len(data):
        end = min(len(data), section.raw_ptr + section.raw_size)
        return bytes(data[section.raw_ptr : end])
    return b""


#: Structural protector markers. Section names live in the section table (plain bytes in
#: the image), so one substring search over the whole blob catches names and banners.
PROTECTOR_MARKERS: Dict[VmProtectionType, Tuple[bytes, ...]] = {
    VmProtectionType.VMPROTECT: (b".vmp0", b".vmp1", b".vmp2", b"VMProtect", b"vmprotect"),
    VmProtectionType.THEMIDA: (b".themida", b".winlice", b"Themida", b"themida", b"WinLicense"),
    VmProtectionType.DENUVO: (b"Denuvo", b"denuvo", b".denuvo", b"DenuvoProtect"),
    VmProtectionType.ENIGMA: (
        b".enigma",
        b".enigma1",
        b".enigma2",
        b"Enigma",
        b"enigma protector",
    ),
    # ``\xbe\x92\x9c\xbb\x91`` is the fixed SteamStub encrypted-section header banner.
    VmProtectionType.STEAM_STUB: (
        b"SteamStub",
        b"steamstub",
        b"\xbe\x92\x9c\xbb\x91",
    ),
}

#: Section names owned by a protector (strong structural evidence).
PROTECTOR_SECTION_NAMES: Dict[VmProtectionType, Tuple[str, ...]] = {
    VmProtectionType.VMPROTECT: (".vmp0", ".vmp1", ".vmp2"),
    VmProtectionType.THEMIDA: (".themida", ".winlice", ".tmd"),
    VmProtectionType.DENUVO: (".denuvo", ".dvl"),
    VmProtectionType.ENIGMA: (".enigma", ".enigma1", ".enigma2"),
    VmProtectionType.STEAM_STUB: (".bind", ".steam"),
}

#: Virtualized-section name prefixes used to down-weight OEP candidates.
VM_SECTION_PREFIXES: Tuple[str, ...] = (
    ".vmp",
    ".themida",
    ".winlice",
    ".enigma",
    ".denuvo",
    ".bind",
)

#: Recommended next move per protector family.
PROTECTOR_ACTIONS: Dict[VmProtectionType, str] = {
    VmProtectionType.DENUVO: (
        "denuvo: keep to triage only; VM handlers are entangled with anti-tamper, "
        "automatic unpacking is not viable"
    ),
    VmProtectionType.THEMIDA: (
        "themida: dump the process once the protector reaches native code, then "
        "--hunt-oep / --reconstruct-pe"
    ),
    VmProtectionType.VMPROTECT: (
        "vmprotect: hunt the OEP with --hunt-oep and rebuild a runnable image with "
        "--reconstruct-pe"
    ),
    VmProtectionType.ENIGMA: (
        "enigma: encrypted/virtualized sections detected; dump the process and "
        "reconstruct with --reconstruct-pe"
    ),
    VmProtectionType.STEAM_STUB: (
        "steam_stub: Steam DRM wrapper; dump after the stub decrypts .text, then "
        "--reconstruct-pe"
    ),
    VmProtectionType.UNKNOWN: (
        "unknown: high-entropy sections without a known protector signature; dump the "
        "process and reconstruct with --reconstruct-pe"
    ),
}

#: Register/memory-indirect ``jmp r/m`` opcodes used by VM dispatchers.
DISPATCHER_JMP_OPCODES: Tuple[bytes, ...] = (
    b"\xff\xe0",
    b"\xff\xe1",
    b"\xff\xe2",
    b"\xff\xe3",
    b"\xff\xe4",
    b"\xff\xe5",
    b"\xff\xe6",
    b"\xff\xe7",
    b"\xff\x25",
)

#: Multi-byte context tokens that must precede a VM dispatcher jump (a bare indirect
#: jump also occurs in ordinary code; a load/lea/arith into a register does not).
DISPATCHER_CONTEXT: Tuple[bytes, ...] = (
    b"\x48\x8b",
    b"\x4c\x8b",
    b"\x48\x8d",
    b"\x4c\x8d",
    b"\x48\x0f\xb6",
    b"\x4c\x0f\xb6",
    b"\x48\x03",
    b"\x48\x01",
    b"\x8b\xc0",
    b"\x8b\xc3",
    b"\x8b\xc5",
    b"\x8b\xc6",
    b"\x8b\xc7",
)


class VmDetector:
    """Structural VM-protection triage: section entropy + signature + dispatcher scan."""

    compute_shannon_entropy = staticmethod(compute_shannon_entropy)

    def __init__(
        self,
        entropy_threshold: float = VM_ENTROPY_THRESHOLD,
        dispatcher_limit: int = MAX_VM_DISPATCHER_CANDIDATES,
    ) -> None:
        self.entropy_threshold = float(entropy_threshold)
        self.dispatcher_limit = int(dispatcher_limit)

    def section_entropy_reports(self, data: bytes) -> List[SectionEntropyReport]:
        """Shannon entropy per PE section (one ``<raw>`` entry for non-PE blobs)."""
        blob = bytes(data or b"")
        pe = parse_pe(blob)
        if pe is None:
            entropy = compute_shannon_entropy(blob)
            return [
                SectionEntropyReport(
                    name="<raw>",
                    virtual_address=0,
                    raw_size=len(blob),
                    entropy=entropy,
                    is_encrypted_or_virtualized=(
                        len(blob) >= 256 and entropy >= self.entropy_threshold
                    ),
                )
            ]
        reports: List[SectionEntropyReport] = []
        for section in pe.sections:
            window = _section_window(blob, pe, section)
            entropy = compute_shannon_entropy(window)
            reports.append(
                SectionEntropyReport(
                    name=section.name,
                    virtual_address=section.virtual_address,
                    raw_size=len(window) or section.raw_size,
                    entropy=entropy,
                    is_encrypted_or_virtualized=(
                        len(window) >= 256 and entropy >= self.entropy_threshold
                    ),
                )
            )
        return reports

    def find_dispatcher_candidates(self, data: bytes) -> List[int]:
        """RVAs of register/memory-indirect jumps preceded by VM dispatch context."""
        blob = bytes(data or b"")
        pe = parse_pe(blob)
        if pe is None:
            return []
        found_rvas: List[int] = []
        for section in pe.sections:
            window = _section_window(blob, pe, section)
            if len(window) < 8:
                continue
            for opcode in DISPATCHER_JMP_OPCODES:
                start = 0
                while True:
                    hit = window.find(opcode, start)
                    if hit < 0:
                        break
                    start = hit + 1
                    lookback = max(0, hit - DISPATCHER_CONTEXT_LOOKBACK)
                    context = window[lookback:hit]
                    if not any(token in context for token in DISPATCHER_CONTEXT):
                        continue
                    rva = section.virtual_address + hit
                    if rva not in found_rvas:
                        found_rvas.append(rva)
        found_rvas.sort()
        return found_rvas[: self.dispatcher_limit]

    def _score_protectors(
        self,
        data: bytes,
        sections: List[SectionEntropyReport],
        dispatchers: List[int],
    ) -> Tuple[VmProtectionType, float]:
        """Deterministic marker/entropy/dispatcher scoring (declaration order breaks ties)."""
        names = [report.name.lower() for report in sections]
        virtualized = [report for report in sections if report.is_encrypted_or_virtualized]
        protector = VmProtectionType.UNKNOWN
        best = 0.0
        for candidate, markers in PROTECTOR_MARKERS.items():
            hits = sum(1 for marker in markers if marker in data)
            owned_names = PROTECTOR_SECTION_NAMES.get(candidate, ())
            owned = sum(
                1
                for name in names
                if any(name == want or name.startswith(want) for want in owned_names)
            )
            if not hits and not owned:
                continue
            score = min(hits, 3) * 0.12 + (0.45 if owned else 0.0)
            if virtualized:
                score += 0.15
            if dispatchers:
                score += 0.10
            if score > best:
                protector = candidate
                best = min(score, 0.98)
        if best > 0.0:
            return protector, best
        if virtualized:
            return VmProtectionType.UNKNOWN, min(0.30 + 0.05 * len(virtualized), 0.50)
        return VmProtectionType.UNKNOWN, 0.0

    def detect_vm_protection(self, data: bytes) -> Optional[VmDetectionReport]:
        """Classify the blob; ``None`` when no VM-protection evidence is present."""
        blob = bytes(data or b"")
        if not blob:
            return None
        sections = self.section_entropy_reports(blob)
        dispatchers = self.find_dispatcher_candidates(blob)
        protector, confidence = self._score_protectors(blob, sections, dispatchers)
        if protector is VmProtectionType.UNKNOWN and confidence <= 0.0:
            return None
        return VmDetectionReport(
            protector=protector,
            confidence=confidence,
            sections=sections,
            vm_dispatcher_candidates=dispatchers,
            suggested_action=PROTECTOR_ACTIONS[protector],
        )

    def analyze(self, data: bytes) -> VmDetectionReport:
        """CLI-facing wrapper that always returns a report (``unknown`` when clean)."""
        report = self.detect_vm_protection(data)
        if report is not None:
            return report
        blob = bytes(data or b"")
        return VmDetectionReport(
            protector=VmProtectionType.UNKNOWN,
            confidence=0.0,
            sections=self.section_entropy_reports(blob),
            vm_dispatcher_candidates=self.find_dispatcher_candidates(blob),
            suggested_action=PROTECTOR_ACTIONS[VmProtectionType.UNKNOWN],
        )


def detect_vm_protection(data: bytes) -> Optional[VmDetectionReport]:
    """Module-level wrapper around :meth:`VmDetector.detect_vm_protection`."""
    return VmDetector().detect_vm_protection(data)


@dataclass(frozen=True)
class PrologueSignature:
    """A compiler function prologue used as an OEP fingerprint."""

    name: str
    prefix: bytes
    wildcard: int = 0
    arch: str = "any"


#: Canonical prologues (PLAN W19 section 2): ``push rbp; mov rbp, rsp``, ``sub rsp, imm``,
#: ``push ebp; mov ebp, esp`` and their immediate offsets.
PROLOGUE_SIGNATURES: Tuple[PrologueSignature, ...] = (
    PrologueSignature("push rbp; mov rbp, rsp", b"\x55\x48\x89\xe5", arch="x86_64"),
    PrologueSignature("push rbp; mov rbp, rsp (alt)", b"\x55\x48\x8b\xec", arch="x86_64"),
    PrologueSignature(
        "push rbp; lea rbp, [rsp+imm8]",
        b"\x55\x48\x8d\x6c\x24",
        wildcard=1,
        arch="x86_64",
    ),
    PrologueSignature("sub rsp, imm8", b"\x48\x83\xec", wildcard=1, arch="x86_64"),
    PrologueSignature("sub rsp, imm32", b"\x48\x81\xec", wildcard=4, arch="x86_64"),
    PrologueSignature("push ebp; mov ebp, esp", b"\x55\x89\xe5", arch="x86"),
    PrologueSignature("push ebp; mov ebp, esp (alt)", b"\x55\x8b\xec", arch="x86"),
    PrologueSignature("sub esp, imm8", b"\x83\xec", wildcard=1, arch="x86"),
    PrologueSignature("sub esp, imm32", b"\x81\xec", wildcard=4, arch="x86"),
)


@dataclass
class OepCandidate:
    """One heuristic Original Entry Point candidate."""

    rva: int
    address: int
    section: str
    prologue: str
    confidence: float
    transition: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rva": self.rva,
            "address": self.address,
            "section": self.section,
            "prologue": self.prologue,
            "confidence": round(self.confidence, 4),
            "transition": self.transition,
        }


class OepHunter:
    """Heuristic Original Entry Point hunter (prologue scanning across transitions)."""

    def __init__(
        self,
        transition_window: int = OEP_TRANSITION_WINDOW,
        limit: int = MAX_OEP_CANDIDATES,
    ) -> None:
        self.transition_window = int(transition_window)
        self.limit = int(limit)

    def scan(self, data: bytes, pe: PEImage) -> List[OepCandidate]:
        """Scan every section for compiler prologues, best first."""
        blob = bytes(data or b"")
        if pe is None:
            return []
        target_arch = "x86_64" if pe.is_64 else "x86"
        best_by_rva: Dict[int, OepCandidate] = {}
        for section in sorted(pe.sections, key=lambda item: (item.virtual_address, item.index)):
            window = _section_window(blob, pe, section)
            if len(window) < 6:
                continue
            executable = bool(section.characteristics & 0x20000000)
            section_name = section.name.lower()
            virtualized_section = any(
                section_name.startswith(prefix) for prefix in VM_SECTION_PREFIXES
            )
            for signature in PROLOGUE_SIGNATURES:
                if signature.arch != "any" and signature.arch != target_arch:
                    continue
                span = len(signature.prefix) + signature.wildcard
                start = 0
                while True:
                    offset = window.find(signature.prefix, start)
                    if offset < 0:
                        break
                    start = offset + 1
                    if offset + span > len(window):
                        continue
                    transition = offset < self.transition_window
                    confidence = 0.5
                    if signature.arch == target_arch:
                        confidence += 0.1
                    if executable:
                        confidence += 0.2
                    if transition:
                        confidence += 0.25
                    if virtualized_section:
                        confidence -= 0.3
                    confidence = max(0.0, min(confidence, 0.99))
                    candidate = OepCandidate(
                        rva=section.virtual_address + offset,
                        address=pe.image_base + section.virtual_address + offset,
                        section=section.name,
                        prologue=signature.name,
                        confidence=confidence,
                        transition=transition,
                    )
                    previous = best_by_rva.get(candidate.rva)
                    if previous is None or candidate.confidence > previous.confidence:
                        best_by_rva[candidate.rva] = candidate
        candidates = list(best_by_rva.values())
        candidates.sort(key=lambda item: (-item.confidence, item.rva))
        return candidates[: self.limit]

    def hunt_oep(self, data: bytes, pe: PEImage) -> Optional[int]:
        """Virtual address of the most plausible OEP, or ``None`` when nothing matches."""
        candidates = self.scan(data, pe)
        if not candidates:
            return None
        return candidates[0].address


def hunt_oep(data: bytes, pe: PEImage) -> Optional[int]:
    """Module-level wrapper around :meth:`OepHunter.hunt_oep`."""
    return OepHunter().hunt_oep(data, pe)


class PeReconstructor:
    """Rebuild a runnable on-disk PE from an image-aligned memory dump."""

    @staticmethod
    def _dump_section(dump: bytes, pe: PEImage, section: PESection) -> bytes:
        """Virtual-layout bytes of ``section`` inside ``dump`` (file layout as fallback)."""
        span = section.virtual_size or section.raw_size
        if span <= 0:
            return b""
        start = section.virtual_address
        if start < len(dump):
            chunk = bytes(dump[start : min(len(dump), start + span)])
            if chunk:
                return chunk
        if section.raw_ptr and section.raw_ptr < len(dump):
            raw_span = section.raw_size or span
            return bytes(dump[section.raw_ptr : min(len(dump), section.raw_ptr + raw_span)])
        return b""

    @staticmethod
    def reconstruct_dump(dump_data: bytes, image_base: int, oep_rva: int) -> bytes:
        """Map virtual pages onto file alignment (Trap 1/Trap 3) and fix up headers.

        Every section's virtual bytes are copied to a ``FileAlignment``-padded raw
        pointer, ``SizeOfRawData``/``PointerToRawData`` are repacked at
        ``header_offset + 16`` (Trap 1), ``SizeOfImage`` and ``AddressOfEntryPoint`` are
        recalculated, and the file-offset-based Certificate Table directory is dropped.
        """
        dump = bytes(dump_data or b"")
        pe = parse_pe(dump)
        if pe is None:
            raise UnpackError(
                "RECONSTRUCT_FAILED: dump does not start with parseable PE headers"
            )
        file_alignment = pe.file_alignment if 0 < pe.file_alignment <= 0x10000 else 0x200
        section_alignment = (
            pe.section_alignment if 0 < pe.section_alignment <= 0x10000 else 0x1000
        )
        header_end = min(len(dump), max(pe.header_end, pe.pe_offset + 24 + pe.optional_size))
        out = bytearray(dump[:header_end])
        out.extend(b"\x00" * (_align_up(len(out), file_alignment) - len(out)))
        size_of_image = _align_up(max(len(out), 1), section_alignment)
        for section in pe.sections:
            content = PeReconstructor._dump_section(dump, pe, section)
            if content:
                raw_ptr = _align_up(len(out), file_alignment)
                raw_size = _align_up(len(content), file_alignment)
                out.extend(b"\x00" * (raw_ptr - len(out)))
                out.extend(content)
                out.extend(b"\x00" * (raw_size - len(content)))
            else:
                raw_size, raw_ptr = 0, 0
            struct.pack_into("<II", out, section.header_offset + 16, raw_size, raw_ptr)
            section_end = section.virtual_address + max(section.virtual_size, len(content))
            if section_end > 0:
                size_of_image = max(size_of_image, section_end)
        size_of_image = _align_up(size_of_image, section_alignment)

        optional_offset = pe.pe_offset + 24
        magic = pe.magic
        if optional_offset + 2 <= len(out):
            magic = struct.unpack_from("<H", out, optional_offset)[0] or magic
        if optional_offset + 60 <= len(out):
            struct.pack_into("<I", out, optional_offset + 16, int(oep_rva) & 0xFFFFFFFF)
            struct.pack_into("<I", out, optional_offset + 56, size_of_image)
        if image_base:
            if magic == 0x20B and optional_offset + 32 <= len(out):
                struct.pack_into(
                    "<Q", out, optional_offset + 24, int(image_base) & 0xFFFFFFFFFFFFFFFF
                )
            elif magic == 0x10B and optional_offset + 32 <= len(out):
                struct.pack_into("<I", out, optional_offset + 28, int(image_base) & 0xFFFFFFFF)

        # Certificate Table (directory 4) stores a *file* offset, which a virtual -> file
        # remap invalidates; zero it so the rebuilt image parses cleanly.
        directories_offset = optional_offset + (112 if magic == 0x20B else 96)
        count_offset = optional_offset + (108 if magic == 0x20B else 92)
        directory_count = 0
        if count_offset + 4 <= len(out):
            directory_count = struct.unpack_from("<I", out, count_offset)[0]
        if not directory_count and pe.optional_size >= (128 if magic == 0x20B else 112):
            tail = pe.optional_size - (112 if magic == 0x20B else 96)
            directory_count = min(16, tail // 8)
        for index in range(min(int(directory_count), 16)):
            entry = directories_offset + 8 * index
            if entry + 8 > len(out):
                break
            if index == 4:
                struct.pack_into("<II", out, entry, 0, 0)
        return bytes(out)


def _parse_oep_argument(value: Any) -> Optional[int]:
    """Parse a CLI ``--oep`` value (decimal, ``0x`` hex, or bare hex)."""
    if value is None:
        return None
    if isinstance(value, int):
        return int(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text, 0)
    except ValueError:
        return int(text, 16)


def unpack_analysis(
    target_path: Path,
    output_path: Optional[Path] = None,
    *,
    detect_vm: bool = False,
    hunt_oep: bool = False,
    reconstruct_pe: bool = False,
    oep: Optional[Any] = None,
) -> Dict[str, Any]:
    """W19 forensic entry point behind ``unpack --detect-vm/--hunt-oep/--reconstruct-pe``.

    Unlike :func:`unpack_binary` this path never raises ``PACKED_ABORT(VM_PROTECTED)``:
    for a VM-protected target the requested triage *is* the deliverable. The default
    (flagless) unpack path is untouched and keeps hard-aborting.
    """
    target_path = Path(target_path)
    if not target_path.exists():
        raise FileNotFoundError(f"Target file not found: {target_path}")
    data = target_path.read_bytes()
    if not data:
        raise UnpackError(f"{UNPACK_FAILED}: empty target")
    pe = parse_pe(data)
    result: Dict[str, Any] = {
        "target": str(target_path),
        "size": len(data),
        "image_base": pe.image_base if pe is not None else None,
    }
    if detect_vm:
        detection = VmDetector().analyze(data)
        result["vm_detection"] = detection.to_dict()
        result["vm_protected"] = detection.protector is not VmProtectionType.UNKNOWN
    if hunt_oep:
        if pe is None:
            result["oep"] = {
                "oep": None,
                "oep_rva": None,
                "prologue": None,
                "confidence": 0.0,
                "candidates": [],
                "note": "target is not a parseable PE image",
            }
        else:
            candidates = OepHunter().scan(data, pe)
            best = candidates[0] if candidates else None
            result["oep"] = {
                "oep": best.address if best is not None else None,
                "oep_rva": best.rva if best is not None else None,
                "prologue": best.prologue if best is not None else None,
                "confidence": best.confidence if best is not None else 0.0,
                "candidates": [item.to_dict() for item in candidates[:10]],
                "note": "" if best is not None else "no compiler prologue found in the image",
            }
    if reconstruct_pe:
        if pe is None:
            raise UnpackError(
                "RECONSTRUCT_FAILED: target does not contain parseable PE headers"
            )
        requested = _parse_oep_argument(oep)
        if requested is None:
            hunted = OepHunter().hunt_oep(data, pe)
            requested = hunted if hunted is not None else pe.image_base + pe.entry_point_rva
        oep_rva = requested - pe.image_base if requested >= pe.image_base else requested
        rebuilt = PeReconstructor.reconstruct_dump(data, pe.image_base, oep_rva)
        suffix = target_path.suffix or ".exe"
        destination = (
            Path(output_path)
            if output_path is not None
            else target_path.with_name(f"{target_path.stem}_reconstructed{suffix}")
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(rebuilt)
        parsed = parse_pe(rebuilt)
        result["reconstructed"] = {
            "output": str(destination),
            "size": len(rebuilt),
            "entry_point_rva": parsed.entry_point_rva if parsed is not None else None,
            "size_of_image": parsed.size_of_image if parsed is not None else None,
            "sections": [
                {
                    "name": section.name,
                    "raw_ptr": section.raw_ptr,
                    "raw_size": section.raw_size,
                }
                for section in (parsed.sections if parsed is not None else [])
            ],
        }
    return result
