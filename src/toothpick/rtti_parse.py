"""
Toothpick MSVC RTTI recovery -- parsing half (unit P3 split, Chunk 1).

Binding contract: ``docs/units/UNIT-P3-RTTI-PLAN.md`` section 1 (RTTI module) and
section 2 (interface specification).  Design anchors: ``docs/design-document.md``
section 1 ("no fabricated structs -- every class row traces to parsed bytes") and
section 19 / Risk R1 (bounded emission: default 50 scanned vtables, ceiling 1000).

This module is the *parsing* half of the RTTI module: constants, data contracts,
name recovery, the bounds-checked reader, the structure readers and
:func:`scan_vtables`.  It is byte-identical to the corresponding region of
``toothpick.rtti`` (a pure move, no redesign); the persistence half
(``store``/``get_class``/``Database`` handling) stays in ``toothpick.rtti``.

Recovered shape -- MSVC ABI, 32- and 64-bit:

    VFTable[-1]          -> RTTICompleteObjectLocator (COL)
    COL.type_descriptor  -> RTTITypeDescriptor (mangled name, e.g. ``.?AVFoo@@``)
    COL.class_hierarchy  -> RTTIClassHierarchyDescriptor (CHD)
    CHD.base_class_array[0 .. n-1] -> RTTIBaseClassDescriptor[] (one per base)

Link resolution differs per architecture and is *never guessed*:

* ``X86``: every COL/CHD/BCD link is a 4-byte **absolute virtual address**, and the
  ``vftable[-1]`` slot is a 4-byte absolute COL pointer.  The type-descriptor
  header is 8 bytes (``pVFTable`` + ``spare`` dwords).
* ``X64``: COL/CHD/BCD links are 4-byte **RVAs relative to the image base** (COL
  signature 1), while the ``vftable[-1]`` slot is an 8-byte absolute COL pointer
  and the type-descriptor header is 16 bytes (two quadwords).  ``sizeof(BCD)`` is
  0x1C on x64 -- the descriptor is dword-packed, not pointer-padded.

Epistemic rules enforced here:

* Every dereference is bounds-checked against the caller-supplied buffer.  A
  truncated, misaligned or out-of-image structure raises :class:`RttiError`; the
  candidate is *rejected* (counted in the ``rejected`` list of
  :func:`scan_vtables`) and **no partial class row is ever emitted**.
* A recovered name is the exact mangled string found in the image or a demangled
  form of it.  :func:`demangle_msvc_type` returns ``(raw, False)`` for anything
  it does not positively recognise, so no name is invented.
* Nothing here reads a file or scans memory: the caller supplies ``data`` and
  ``base_addr`` (the DB stores no raw section bytes, so the parser must never
  invent addresses).

Trap hygiene: this module never imports ``faulthandler`` (Trap 11 -- suites run
under ``-p no:faulthandler``) and it opens no SQLite handle at all, so the
Trap 7 ``[WinError 32]`` class of leak (a WAL handle left open for Windows
teardown) cannot originate here; persistence lives in ``toothpick.rtti``, which
closes every handle it opens in a ``finally``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

__all__ = [
    "DEFAULT_MAX_VTABLES",
    "MAX_BASE_CLASSES",
    "MAX_TYPE_NAME",
    "MAX_VTABLES",
    "PROVENANCE",
    "PTR_SIZE_X86",
    "PTR_SIZE_X64",
    "RTTI_TYPE_PREFIX",
    "BaseRef",
    "ClassRecord",
    "RttiArch",
    "RttiError",
    "ScanResult",
    "class_type_id",
    "demangle_msvc_type",
    "read_base_class_descriptor",
    "read_class_hierarchy_descriptor",
    "read_complete_object_locator",
    "read_type_descriptor",
    "resolve_max_vtables",
    "scan_vtables",
]

# --------------------------------------------------------------------------- #
# Constants (Risk R1 bounds + ABI sizes)
# --------------------------------------------------------------------------- #

#: Default cap on scanned vtables per call (R1: bounded emission).
DEFAULT_MAX_VTABLES = 50

#: Hard ceiling on scanned vtables per call -- a caller asking for 10_000 is
#: clamped, so the answer is still bounded (R1).
MAX_VTABLES = 1000

#: Hard ceiling on ``numBaseClasses`` of one recovered class (defensive bound:
#: a corrupt descriptor cannot make the parser walk an unbounded array).
MAX_BASE_CLASSES = 256

#: Longest type-descriptor name read before the structure is refused.
MAX_TYPE_NAME = 4096

#: Provenance stamped into every recovered ``types`` layout.
PROVENANCE = "msvc-rtti"

#: ``types.type_id`` namespace for recovered RTTI rows.
RTTI_TYPE_PREFIX = "rtti:"

PTR_SIZE_X86 = 4
PTR_SIZE_X64 = 8

#: Accepted ``CompleteObjectLocator.signature`` values.  1 marks the RVA-based
#: (x64) form; 0 the absolute-pointer form (x86, older x64 images).  The value is
#: recorded but *not* used to pick a resolution rule -- the architecture decides
#: that, deterministically.
_COL_SIGNATURES = (0, 1)

_TD_HEADER_X86 = 8
_TD_HEADER_X64 = 16
_COL_SIZE = {PTR_SIZE_X86: 20, PTR_SIZE_X64: 24}
_CHD_SIZE = 16
_BCD_SIZE = 28




class RttiError(Exception):
    """Raised when RTTI bytes cannot be parsed or persisted.

    A malformed/truncated structure raises this instead of returning a partial
    row; :func:`scan_vtables` turns it into a ``(addr, reason)`` rejection.
    """


class RttiArch(str, Enum):
    """MSVC ABI variant the parser must assume (never inferred from bytes)."""

    X86 = "x86"
    X64 = "x64"

    @classmethod
    def coerce(cls, value: Union["RttiArch", str]) -> "RttiArch":
        """Accept an :class:`RttiArch` or a known architecture spelling."""
        if isinstance(value, cls):
            return value
        if isinstance(value, bool):
            raise RttiError(f"architecture must be 'x86' or 'x64', got {value!r}")
        text = str(value).strip().lower()
        aliases = {
            "x86": cls.X86,
            "i386": cls.X86,
            "ia32": cls.X86,
            "32": cls.X86,
            "x64": cls.X64,
            "amd64": cls.X64,
            "x86_64": cls.X64,
            "64": cls.X64,
        }
        try:
            return aliases[text]
        except KeyError:
            raise RttiError(
                f"unsupported RTTI architecture {value!r}; expected 'x86' or 'x64'"
            ) from None

    @property
    def pointer_size(self) -> int:
        """Size of a ``vftable[-1]`` slot (4 on x86, 8 on x64)."""
        return PTR_SIZE_X86 if self is RttiArch.X86 else PTR_SIZE_X64

    @property
    def rva_links(self) -> bool:
        """True when COL/CHD/BCD links are RVAs relative to the image base."""
        return self is RttiArch.X64


# --------------------------------------------------------------------------- #
# Data contracts
# --------------------------------------------------------------------------- #


@dataclass
class BaseRef:
    """One base class of a recovered class (a ``BaseClassDescriptor``)."""

    mangled: str
    name: str = ""
    mdisp: int = 0


@dataclass
class ClassRecord:
    """One recovered class: identity, vtable site, and its base chain."""

    mangled: str
    name: str = ""
    demangled: bool = False
    vtable_addr: int = 0
    bases: List[BaseRef] = field(default_factory=list)
    col_addr: int = 0
    #: Extra parse evidence (documentation/query only; never required to exist).
    td_addr: int = 0
    #: Name offset inside the TypeDescriptor that validated (16 canonical).
    td_name_offset: int = 16
    chd_addr: int = 0
    arch: str = ""


class ScanResult(NamedTuple):
    """``(classes, rejected)`` -- unpackable, and attribute-addressable."""

    classes: List[ClassRecord]
    rejected: List[Tuple[int, str]]


def class_type_id(mangled: str) -> str:
    """The ``types.type_id`` a recovered class is stored under."""
    if not isinstance(mangled, str) or not mangled:
        raise RttiError("mangled name must be a non-empty string")
    return RTTI_TYPE_PREFIX + mangled


def resolve_max_vtables(value: Optional[int] = None) -> int:
    """Coerce a vtable cap: default :data:`DEFAULT_MAX_VTABLES`, clamp ceiling.

    Mirrors ``profile.resolve_top_n``: a non-positive or non-integer budget is a
    caller error (raised), an over-large budget is clamped to
    :data:`MAX_VTABLES` -- the answer stays bounded either way.
    """
    if value is None:
        return DEFAULT_MAX_VTABLES
    if isinstance(value, bool):
        raise RttiError(f"max_vtables must be an integer, got {value!r}")
    try:
        budget = int(value)
    except (TypeError, ValueError) as exc:
        raise RttiError(f"max_vtables must be an integer, got {value!r}") from exc
    if budget <= 0:
        raise RttiError(f"max_vtables must be positive, got {budget}")
    return min(budget, MAX_VTABLES)


# --------------------------------------------------------------------------- #
# Name recovery
# --------------------------------------------------------------------------- #


def demangle_msvc_type(text: str) -> Tuple[str, bool]:
    """Recover a readable class/struct name from a mangled MSVC RTTI string.

    Recognised forms are the two the plan names, ``.?AV<name>@@`` (class) and
    ``.?AU<name>@@`` (struct): the prefix and the single trailing ``@@``
    terminator are stripped and the remainder is returned verbatim, so nested
    scope/template decorations stay visible (``.?AV?$vec@VFoo@@@@`` ->
    ``'?$vec@VFoo@@'``).  Anything else -- other encodings, missing terminator,
    an empty body, a non-string -- returns ``(text, False)``: the raw bytes are
    never presented as a guessed name.
    """
    if not isinstance(text, str):
        raise RttiError(f"demangle_msvc_type expects str, got {type(text).__name__}")
    if not text.startswith((".?AV", ".?AU")):
        return text, False
    body = text[4:]
    if not body.endswith("@@"):
        return text, False
    name = body[:-2]
    if not name:
        return text, False
    return name, True


# --------------------------------------------------------------------------- #
# Bounds-checked reader
# --------------------------------------------------------------------------- #


def _is_printable_ascii(text: str) -> bool:
    return bool(text) and all(0x20 <= ord(ch) <= 0x7E for ch in text)


class _Reader:
    """Read-only view over caller-supplied image bytes plus its image base.

    Every accessor converts an absolute address into ``data`` and refuses
    anything outside ``[base, base + len(data))``; that refusal is the whole
    truncation story of this module (nothing reads past the buffer, so a partial
    row can never be returned).
    """

    __slots__ = ("data", "base", "end", "arch")

    def __init__(
        self, data: Union[bytes, bytearray, memoryview], base_addr: int, arch: RttiArch
    ) -> None:
        if isinstance(data, (bytes, bytearray, memoryview)):
            self.data = bytes(data)
        else:
            raise RttiError(f"data must be bytes-like, got {type(data).__name__}")
        if isinstance(base_addr, bool):
            raise RttiError(f"base_addr must be an integer, got {base_addr!r}")
        try:
            base = int(base_addr)
        except (TypeError, ValueError) as exc:
            raise RttiError(f"base_addr must be an integer, got {base_addr!r}") from exc
        if base < 0:
            raise RttiError(f"base_addr must be non-negative, got {base}")
        self.base = base
        self.end = base + len(self.data)
        self.arch = arch

    def offset(self, addr: int, size: int, what: str) -> int:
        """Convert an absolute address to a buffer offset, or refuse."""
        if isinstance(addr, bool):
            raise RttiError(f"{what}: address must be an integer, got {addr!r}")
        try:
            value = int(addr)
        except (TypeError, ValueError) as exc:
            raise RttiError(f"{what}: address must be an integer, got {addr!r}") from exc
        off = value - self.base
        if off < 0 or off + size > len(self.data):
            raise RttiError(
                f"{what}: address 0x{value:x} (+{size} bytes) is outside the supplied "
                f"buffer [0x{self.base:x}, 0x{self.end:x})"
            )
        return off

    def u32(self, addr: int, what: str) -> int:
        off = self.offset(addr, 4, what)
        return int.from_bytes(self.data[off : off + 4], "little")

    def u64(self, addr: int, what: str) -> int:
        off = self.offset(addr, 8, what)
        return int.from_bytes(self.data[off : off + 8], "little")

    def s32(self, addr: int, what: str) -> int:
        off = self.offset(addr, 4, what)
        return int.from_bytes(self.data[off : off + 4], "little", signed=True)

    def cstring(self, addr: int, what: str, limit: int = MAX_TYPE_NAME) -> str:
        """Read a NUL-terminated string; an unterminated one is refused."""
        off = self.offset(addr, 1, what)
        window = self.data[off : off + limit]
        stop = window.find(b"\x00")
        if stop < 0:
            raise RttiError(
                f"{what}: unterminated string at 0x{addr:x} (no NUL within {limit} bytes)"
            )
        return window[:stop].decode("utf-8", errors="replace")

    def resolve(self, link: int, what: str) -> int:
        """Resolve a structure link: RVA + base on x64, absolute VA on x86."""
        if self.arch.rva_links:
            if link == 0:
                raise RttiError(f"{what}: null RVA link")
            addr = self.base + link
        else:
            addr = link
        self.offset(addr, 1, what)
        return addr


# --------------------------------------------------------------------------- #
# Structure readers (each raises RttiError rather than degrade)
# --------------------------------------------------------------------------- #


def _read_locator(reader: _Reader, col_addr: int) -> Dict[str, Any]:
    arch = reader.arch
    size = _COL_SIZE[arch.pointer_size]
    reader.offset(col_addr, size, "CompleteObjectLocator")
    signature = reader.u32(col_addr, "COL.signature")
    if signature not in _COL_SIGNATURES:
        raise RttiError(
            f"CompleteObjectLocator at 0x{col_addr:x}: signature 0x{signature:x} "
            f"is not one of {list(_COL_SIGNATURES)}"
        )
    td_link = reader.u32(col_addr + 12, "COL.type_descriptor link")
    chd_link = reader.u32(col_addr + 16, "COL.class_hierarchy link")
    self_rva = reader.u32(col_addr + 20, "COL.self link") if arch.rva_links else 0
    return {
        "col_addr": int(col_addr),
        "signature": signature,
        "offset": reader.u32(col_addr + 4, "COL.offset"),
        "cd_offset": reader.u32(col_addr + 8, "COL.cd_offset"),
        "type_descriptor_link": td_link,
        "class_hierarchy_link": chd_link,
        "self_rva": self_rva,
        "self": (reader.base + self_rva) if self_rva else 0,
        "type_descriptor": reader.resolve(td_link, "COL.type_descriptor"),
        "class_hierarchy": reader.resolve(chd_link, "COL.class_hierarchy"),
        "arch": arch.value,
    }


def _read_type_descriptor(reader: _Reader, td_addr: int) -> Dict[str, Any]:
    arch = reader.arch
    header = _TD_HEADER_X64 if arch is RttiArch.X64 else _TD_HEADER_X86
    reader.offset(td_addr, header, "TypeDescriptor header")
    mangled = reader.cstring(td_addr + header, "TypeDescriptor name")
    name_offset = header
    if (not mangled or not _is_printable_ascii(mangled)) and arch is RttiArch.X64:
        # HOI4-evidence variant (field log Q1): some x64 descriptors carry
        # an 8-byte zero prefix (pVF/spare shifted to +8/+16, name at +24).
        # Engage ONLY when the prefix pattern holds strictly; otherwise the
        # canonical failure below stands.
        prefix = reader.u64(td_addr, "TypeDescriptor prefix")
        ptr = reader.u64(td_addr + 8, "TypeDescriptor pointer")
        if prefix == 0 and ptr != 0:
            candidate = reader.cstring(td_addr + 24, "TypeDescriptor name (+24)")
            if candidate and _is_printable_ascii(candidate):
                mangled = candidate
                name_offset = 24
    if not mangled:
        raise RttiError(f"TypeDescriptor at 0x{td_addr:x} has an empty name")
    if not _is_printable_ascii(mangled):
        raise RttiError(
            f"TypeDescriptor at 0x{td_addr:x} does not hold a printable name "
            f"({mangled[:32]!r})"
        )
    name, demangled = demangle_msvc_type(mangled)
    return {
        "td_addr": int(td_addr),
        "mangled": mangled,
        "name": name,
        "demangled": demangled,
        "td_name_offset": int(name_offset),
        "arch": arch.value,
    }


def _read_class_hierarchy(reader: _Reader, chd_addr: int) -> Dict[str, Any]:
    # Canonical CHD window (16 bytes: signature, attributes, numBaseClasses,
    # array link).  The x64 +24 fallback for the array link lives in
    # _recover_class, so this reader only ever applies the +12 rule.
    reader.offset(chd_addr, _CHD_SIZE, "ClassHierarchyDescriptor")
    num_bases = reader.u32(chd_addr + 8, "CHD.num_base_classes")
    if num_bases == 0:
        raise RttiError(
            f"ClassHierarchyDescriptor at 0x{chd_addr:x} declares no base classes"
        )
    if num_bases > MAX_BASE_CLASSES:
        raise RttiError(
            f"ClassHierarchyDescriptor at 0x{chd_addr:x} declares {num_bases} base "
            f"classes (ceiling {MAX_BASE_CLASSES})"
        )
    # The canonical +12 link is the only rule applied here, for x86 and x64
    # alike; an x64 image whose word at +12 does not decode as a BCD-pointer
    # array is retried at +24 by _recover_class (evidence there), never
    # partially here.
    array_link = reader.u32(chd_addr + 12, "CHD.base_class_array link")
    array_addr = reader.resolve(array_link, "CHD.base_class_array")
    # The entire array must be present before any element is trusted.
    reader.offset(array_addr, 4 * num_bases, "CHD.base_class_array contents")
    return {
        "chd_addr": int(chd_addr),
        "signature": reader.u32(chd_addr, "CHD.signature"),
        "attributes": reader.u32(chd_addr + 4, "CHD.attributes"),
        "num_base_classes": num_bases,
        "base_class_array_link": array_link,
        "base_class_array": array_addr,
        "arch": reader.arch.value,
    }


def _read_base_class(reader: _Reader, bcd_addr: int) -> Dict[str, Any]:
    reader.offset(bcd_addr, _BCD_SIZE, "BaseClassDescriptor")
    td_addr = reader.resolve(
        reader.u32(bcd_addr, "BCD.type_descriptor link"), "BCD.type_descriptor"
    )
    td = _read_type_descriptor(reader, td_addr)
    return {
        "bcd_addr": int(bcd_addr),
        "type_descriptor": td_addr,
        "mangled": td["mangled"],
        "name": td["name"],
        "demangled": td["demangled"],
        "num_contained_bases": reader.u32(bcd_addr + 4, "BCD.num_contained_bases"),
        "mdisp": reader.s32(bcd_addr + 8, "BCD.mdisp"),
        "pdisp": reader.s32(bcd_addr + 12, "BCD.pdisp"),
        "vdisp": reader.s32(bcd_addr + 16, "BCD.vdisp"),
        "attributes": reader.u32(bcd_addr + 20, "BCD.attributes"),
        "class_hierarchy": reader.resolve(
            reader.u32(bcd_addr + 24, "BCD.class_hierarchy link"),
            "BCD.class_hierarchy",
        ),
        "arch": reader.arch.value,
    }


def read_complete_object_locator(
    data: Union[bytes, bytearray, memoryview], base_addr: int, arch: Any, col_addr: int
) -> Dict[str, Any]:
    """Parse the COL at ``col_addr`` (x86: 5 dwords of absolute VAs; x64: 6 dwords of RVAs)."""
    reader = _Reader(data, base_addr, RttiArch.coerce(arch))
    return _read_locator(reader, int(col_addr))


def read_type_descriptor(
    data: Union[bytes, bytearray, memoryview], base_addr: int, arch: Any, td_addr: int
) -> Dict[str, Any]:
    """Parse the TypeDescriptor at ``td_addr`` (8-byte header on x86, 16 on x64)."""
    reader = _Reader(data, base_addr, RttiArch.coerce(arch))
    return _read_type_descriptor(reader, int(td_addr))


def read_class_hierarchy_descriptor(
    data: Union[bytes, bytearray, memoryview], base_addr: int, arch: Any, chd_addr: int
) -> Dict[str, Any]:
    """Parse the ClassHierarchyDescriptor at ``chd_addr``."""
    reader = _Reader(data, base_addr, RttiArch.coerce(arch))
    return _read_class_hierarchy(reader, int(chd_addr))


def read_base_class_descriptor(
    data: Union[bytes, bytearray, memoryview], base_addr: int, arch: Any, bcd_addr: int
) -> Dict[str, Any]:
    """Parse the BaseClassDescriptor at ``bcd_addr`` (28 dword-packed bytes)."""
    reader = _Reader(data, base_addr, RttiArch.coerce(arch))
    return _read_base_class(reader, int(bcd_addr))


# --------------------------------------------------------------------------- #
# Vtable scan
# --------------------------------------------------------------------------- #


def _coerce_candidates(vtable_addrs: Any) -> List[int]:
    """Ordered, de-duplicated candidate addresses (a lone int is accepted)."""
    if isinstance(vtable_addrs, bool):
        raise RttiError(f"vtable_addrs must be a sequence of addresses, got {vtable_addrs!r}")
    if isinstance(vtable_addrs, int):
        raw: Sequence[Any] = [vtable_addrs]
    elif isinstance(vtable_addrs, (str, bytes, bytearray)):
        raise RttiError(
            "vtable_addrs must be a sequence of integer addresses "
            f"(got {type(vtable_addrs).__name__}); parse hex text before calling"
        )
    else:
        try:
            raw = list(vtable_addrs)
        except TypeError as exc:
            raise RttiError(
                f"vtable_addrs must be a sequence of addresses, got {type(vtable_addrs).__name__}"
            ) from exc
    seen: Dict[int, None] = {}
    for item in raw:
        if isinstance(item, bool):
            raise RttiError(f"vtable address must be an integer, got {item!r}")
        try:
            value = int(item)
        except (TypeError, ValueError) as exc:
            raise RttiError(f"vtable address must be an integer, got {item!r}") from exc
        if value < 0:
            raise RttiError(f"vtable address must be non-negative, got {value}")
        seen.setdefault(value, None)
    return list(seen)


def _walk_bases(reader: _Reader, array_addr: int, num_bases: int) -> List[BaseRef]:
    """Walk one base-class array; any unreadable entry fails the whole walk."""
    reader.offset(array_addr, 4 * num_bases, "CHD.base_class_array contents")
    bases: List[BaseRef] = []
    for index in range(num_bases):
        entry_addr = array_addr + 4 * index
        base_addr = reader.resolve(
            reader.u32(entry_addr, f"base_class_array[{index}] link"),
            f"base_class_array[{index}]",
        )
        base = _read_base_class(reader, base_addr)
        bases.append(
            BaseRef(mangled=base["mangled"], name=base["name"], mdisp=base["mdisp"])
        )
    return bases


def _recover_class(reader: _Reader, vtable_addr: int) -> ClassRecord:
    arch = reader.arch
    slot_addr = vtable_addr - arch.pointer_size
    what = "vftable[-1] CompleteObjectLocator pointer"
    col_ptr = (
        reader.u64(slot_addr, what)
        if arch is RttiArch.X64
        else reader.u32(slot_addr, what)
    )
    locator = _read_locator(reader, col_ptr)
    descriptor = _read_type_descriptor(reader, locator["type_descriptor"])
    hierarchy = _read_class_hierarchy(reader, locator["class_hierarchy"])
    try:
        bases = _walk_bases(
            reader, hierarchy["base_class_array"], hierarchy["num_base_classes"]
        )
    except RttiError:
        if arch is not RttiArch.X64:
            raise
        # x64 field evidence (HOI4, imagebase 0x140000000): the word at +12 does
        # not decode as a BCD-pointer array for some binaries (anchor CHD
        # 0x142bc7fc8 raw u32s [0,0,2,0x2bc7fe0,0,0,0x2bc7ff8,0x2bc5390];
        # 0x142bc7ff8 is a valid BCD), while the canonical +12 holds for the
        # certified corpus.  So: prefer canonical +12; engage +24 ONLY when the
        # whole canonical walk failed (no partially read array is ever kept) and
        # the +24 array then walks in full -- anything else stays refused, and
        # x86 never falls back at all.
        alt_link = reader.u32(
            hierarchy["chd_addr"] + 24, "CHD.base_class_array link (+24)"
        )
        alt_array = reader.resolve(alt_link, "CHD.base_class_array (+24)")
        bases = _walk_bases(reader, alt_array, hierarchy["num_base_classes"])
    return ClassRecord(
        mangled=descriptor["mangled"],
        name=descriptor["name"],
        demangled=bool(descriptor["demangled"]),
        vtable_addr=int(vtable_addr),
        bases=bases,
        col_addr=locator["col_addr"],
        td_addr=descriptor["td_addr"],
        td_name_offset=int(descriptor.get("td_name_offset", 16)),
        chd_addr=hierarchy["chd_addr"],
        arch=arch.value,
    )


def scan_vtables(
    data: Union[bytes, bytearray, memoryview],
    base_addr: int,
    arch: Any,
    vtable_addrs: Any,
    max_vtables: Optional[int] = None,
) -> ScanResult:
    """Recover classes for each candidate vtable address in ``vtable_addrs``.

    For each candidate ``addr`` the ``vftable[-1]`` slot is read (8 bytes on x64,
    4 on x86), the COL is parsed, and the type descriptor / class hierarchy / base
    class descriptors are walked with full bounds checks.  Candidates are
    processed in caller order with exact duplicates dropped.

    Emission is bounded (Risk R1): at most ``max_vtables`` candidates are scanned
    -- default :data:`DEFAULT_MAX_VTABLES` (50), clamped to :data:`MAX_VTABLES`
    (1000) -- and every candidate beyond the bound is reported in ``rejected``
    with an explicit cap reason, so a caller can never mistake a bounded answer
    for an exhaustive one.

    Returns ``(classes, rejected)`` as a :class:`ScanResult`: ``classes`` in scan
    order and ``rejected`` as ``(candidate_addr, reason)``.  A candidate whose
    bytes are truncated/misaligned/out of image contributes *no* partial row.
    """
    engine = RttiArch.coerce(arch)
    budget = resolve_max_vtables(max_vtables)
    reader = _Reader(data, base_addr, engine)
    candidates = _coerce_candidates(vtable_addrs)

    rejected: List[Tuple[int, str]] = []
    if len(candidates) > budget:
        for addr in candidates[budget:]:
            rejected.append(
                (
                    addr,
                    f"vtable cap reached: {budget} of {len(candidates)} candidates "
                    f"scanned (R1 bound; ceiling {MAX_VTABLES})",
                )
            )
        candidates = candidates[:budget]

    classes: List[ClassRecord] = []
    for addr in candidates:
        try:
            classes.append(_recover_class(reader, addr))
        except RttiError as exc:
            rejected.append((addr, str(exc)))
    return ScanResult(classes=classes, rejected=rejected)
