"""
Toothpick ReClass data-structure automation (unit W18).

Implements ``docs/units/UNIT-W18-RECLASS-PLAN.md`` chunks 1-2:

* :class:`StructField` / :class:`StructLayout` -- the data contracts named by the
  plan (``offset``/``name``/``field_type``/``size``/``comment`` and
  ``name``/``total_size``/``fields``).
* :class:`StructSynthesizer` -- memory-layout synthesis with type-inference
  heuristics over a byte buffer (:meth:`StructSynthesizer.synthesize_from_memory`)
  or over disassembly/pointer-scan access hints
  (:meth:`StructSynthesizer.synthesize_from_accesses`).
* Exporters -- :meth:`StructLayout.to_reclass_xml` (ReClass.NET ``.rcnet``
  project), :meth:`StructLayout.to_cpp` (``#pragma pack(push, 1)`` + offset
  annotations + ``static_assert`` size gate), :meth:`StructLayout.to_csharp`
  (``[StructLayout(LayoutKind.Explicit)]`` + ``[FieldOffset(...)]``) and
  :meth:`StructLayout.to_json`.
* :func:`ingest_layout` -- persistence into the ``types`` table of
  ``toothpick.db`` (:class:`toothpick.storage.TypeRecord`), so a synthesized
  layout joins the global type lattice.

Inference rules (deterministic, first match wins at each offset):

1. offset 0 holding a pointer-shaped value -> ``vtable`` (``void**``);
2. printable ASCII run of >= 4 bytes terminated by NUL -> ``string``;
3. UTF-16LE run of >= 3 printable chars terminated by ``0x0000`` -> ``wstring``;
4. IEEE-754 single whose 32-bit integer reading is noise (``> 0x800000``) ->
   ``float``; same test one word wider for ``double``;
5. pointer-shaped value (aligned, user-space range, non-zero low dword on x64) ->
   ``pointer``;
6. a ``0x01`` byte followed by three zero bytes and then a non-zero field ->
   ``bool`` (a ``0x00`` slot is indistinguishable from padding and is reported as
   padding -- padding is the more informative inference);
7. a run of >= 4 zero bytes -> ``padding`` array;
8. a small positive value in a zero-padded slot -> ``int8`` / ``int16``;
9. otherwise ``int32``, promoted to ``int64`` only when the low dword is large
   *and* the high dword is zero or sign-extension.

Honesty notes -- these are heuristics, not proofs:

* Value-type inference from bytes is ambiguous in principle (``0x3F800000`` is
  both ``int32 1065353216`` and ``float 1.0``).  The rules above always pick one
  reading; :meth:`StructSynthesizer.synthesize_from_accesses` takes explicit type
  hints and is the way to force a reading.
* Pointer plausibility is range-based only.  Nothing here proves a value *is* a
  pointer, and a synthesized layout is never claimed to be the true C declaration
  of the target structure.
* Only byte windows that are actually present in the buffer are classified, so an
  inferred size is a property of one sample, not of the type system.

Layout synthesis is bounded (``MAX_STRUCT_SIZE``, ``MAX_FIELDS``) so a 4 GB
pointer-scan result can never blow up an agent's context window (Risk R1).
"""

from __future__ import annotations

import json
import re
import struct as _struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from toothpick.storage import Database, TypeRecord

__all__ = [
    "FIELD_TYPES",
    "MAX_FIELDS",
    "MAX_STRUCT_SIZE",
    "POINTER_SIZE_X86",
    "POINTER_SIZE_X64",
    "RC_NET_FILE_VERSION",
    "ReclassError",
    "StructField",
    "StructLayout",
    "StructSynthesizer",
    "ingest_layout",
]

#: File version stamped into the generated ReClass.NET project root.
RC_NET_FILE_VERSION = "4.3.0.0"

#: Hard ceiling for a synthesized layout (64 KiB) -- Risk R1 context bound.
MAX_STRUCT_SIZE = 0x1_0000

#: Hard ceiling for the number of fields in one synthesized layout.
MAX_FIELDS = 4096

POINTER_SIZE_X86 = 4
POINTER_SIZE_X64 = 8

#: A zero run of at least this many bytes is reported as a padding array.
PADDING_MIN_RUN = 4

#: A 32-bit integer reading above this magnitude is treated as "not an integer"
#: when a float reading is plausible (and vice versa).
_INT32_NOISE = 0x0080_0000
_INT64_NOISE = 1 << 32

#: The complete ``field_type`` vocabulary of this module.
FIELD_TYPES = frozenset(
    {
        "pointer",
        "vtable",
        "float",
        "double",
        "int8",
        "int16",
        "int32",
        "int64",
        "bool",
        "string",
        "wstring",
        "padding",
    }
)


class ReclassError(Exception):
    """Raised when a struct layout cannot be synthesized, validated or exported."""


# ---------------------------------------------------------------------------
# ReClass.NET / C++ / C# type mappings
# ---------------------------------------------------------------------------

_RC_NET_NODE_TYPES: Dict[str, str] = {
    "vtable": "Pointer",
    "pointer": "Pointer",
    "float": "Float",
    "double": "Double",
    "int8": "Int8",
    "int16": "Int16",
    "int32": "Int32",
    "int64": "Int64",
    "bool": "Bool",
    "string": "Utf8Text",
    "wstring": "Utf16Text",
}

_CPP_SCALAR_TYPES: Dict[str, str] = {
    "vtable": "void**",
    "pointer": "std::uintptr_t",
    "float": "float",
    "double": "double",
    "int8": "std::int8_t",
    "int16": "std::int16_t",
    "int32": "std::int32_t",
    "int64": "std::int64_t",
    "bool": "bool",
}

_CSHARP_SCALAR_TYPES: Dict[str, str] = {
    "vtable": "IntPtr",
    "pointer": "IntPtr",
    "float": "float",
    "double": "double",
    "int8": "sbyte",
    "int16": "short",
    "int32": "int",
    "int64": "long",
    "bool": "bool",
}

#: Prefix used for generated member names, per inferred field type.
_NAME_PREFIXES: Dict[str, str] = {
    "vtable": "vtable",
    "pointer": "ptr",
    "float": "f32",
    "double": "f64",
    "int8": "i8",
    "int16": "i16",
    "int32": "i32",
    "int64": "i64",
    "bool": "b",
    "string": "str",
    "wstring": "wstr",
    "padding": "pad",
}

_CPP_RESERVED = frozenset(
    """
    alignas alignof and and_eq asm auto bitand bitor bool break case catch char
    char16_t char32_t class compl const constexpr const_cast continue decltype
    default delete do double dynamic_cast else enum explicit export extern false
    float for friend goto if inline int long mutable namespace new noexcept not
    not_eq nullptr operator or or_eq private protected public register
    reinterpret_cast return short signed sizeof static static_assert static_cast
    struct switch template this throw true try typedef typeid typename union
    unsigned using virtual void volatile wchar_t while xor xor_eq
    """.split()
)

_IDENT_CLEAN_RE = re.compile(r"[^0-9A-Za-z_]")
_IDENT_RUN_RE = re.compile(r"_+")

#: Scan ``data_type`` -> synthesis hint (``--export-reclass``).
_SCAN_HINT_MAP: Dict[str, str] = {
    "pointer": "pointer",
    "i8": "int8",
    "u8": "int8",
    "i16": "int16",
    "u16": "int16",
    "i32": "int32",
    "u32": "int32",
    "i64": "int64",
    "u64": "int64",
    "f32": "float",
    "f64": "double",
    "string": "string",
    "hex": "padding",
}

#: Accepted type-hint aliases for :meth:`StructSynthesizer.synthesize_from_accesses`.
_HINT_ALIASES: Dict[str, str] = {
    "ptr": "pointer",
    "pointer": "pointer",
    "void*": "pointer",
    "void**": "pointer",
    "vtable": "vtable",
    "vtbl": "vtable",
    "f32": "float",
    "float": "float",
    "f64": "double",
    "double": "double",
    "i8": "int8",
    "u8": "int8",
    "byte": "int8",
    "char": "int8",
    "int8": "int8",
    "i16": "int16",
    "u16": "int16",
    "short": "int16",
    "int16": "int16",
    "i32": "int32",
    "u32": "int32",
    "int": "int32",
    "dword": "int32",
    "int32": "int32",
    "i64": "int64",
    "u64": "int64",
    "long": "int64",
    "qword": "int64",
    "int64": "int64",
    "bool": "bool",
    "string": "string",
    "str": "string",
    "utf8": "string",
    "wstring": "wstring",
    "utf16": "wstring",
    "wide": "wstring",
    "padding": "padding",
    "pad": "padding",
    "hex": "padding",
}

#: Fixed sizes for scalar hints; ``None`` means "use the synthesizer pointer size".
_HINT_FIXED_SIZES: Dict[str, Optional[int]] = {
    "vtable": None,
    "pointer": None,
    "float": 4,
    "double": 8,
    "int8": 1,
    "int16": 2,
    "int32": 4,
    "int64": 8,
    "bool": 1,
}


# ---------------------------------------------------------------------------
# Heuristic primitives
# ---------------------------------------------------------------------------


def _float_plausible(raw32: int) -> bool:
    """True when ``raw32`` reads as a normal, human-scale IEEE-754 single."""
    exponent = (raw32 >> 23) & 0xFF
    if exponent == 0 or exponent == 0xFF:  # subnormal / zero / inf / nan
        return False
    value = _struct.unpack("<f", _struct.pack("<I", raw32))[0]
    return 1e-6 <= abs(value) <= 1e12


def _double_plausible(raw64: int) -> bool:
    """True when ``raw64`` reads as a normal, human-scale IEEE-754 double."""
    exponent = (raw64 >> 52) & 0x7FF
    if exponent == 0 or exponent == 0x7FF:
        return False
    value = _struct.unpack("<d", _struct.pack("<Q", raw64))[0]
    return 1e-6 <= abs(value) <= 1e12


def _pointer_plausible(
    value: int,
    pointer_size: int,
    base_addr: int,
    buffer_len: int,
    module_ranges: Sequence[Tuple[int, int]],
) -> bool:
    """Range/alignment heuristic for "this word could be a pointer".

    A value inside a known module range (or inside the loaded buffer) is accepted
    on range alone; otherwise it must look like a user-space address: aligned,
    non-zero, above the 32-bit noise floor (on x64) and below the kernel split.
    On x64 a value whose low dword is zero is rejected -- that is exactly the
    shape a run of padding bytes takes when read as a pointer.
    """
    if value <= 0:
        return False
    if value % pointer_size:
        return False
    for start, end in module_ranges:
        if start <= value < end:
            return True
    if buffer_len and base_addr <= value < base_addr + buffer_len:
        return True
    if pointer_size == POINTER_SIZE_X64:
        if (value & 0xFFFF_FFFF) == 0:
            return False
        return (1 << 32) <= value < (1 << 48)
    return 0x1000 <= value < (1 << 32)


def _ascii_string_length(window: bytes) -> int:
    """Length (including NUL) of a leading ASCII string, or ``0`` when absent."""
    run = 0
    for byte in window:
        if 0x20 <= byte <= 0x7E:
            run += 1
        else:
            break
    if run < 4 or run >= len(window):
        return 0
    if window[run] != 0:
        return 0
    return run + 1


def _utf16_string_length(window: bytes) -> int:
    """Length (including the ``0x0000`` terminator) of a UTF-16LE string, else 0."""
    chars = 0
    index = 0
    while index + 1 < len(window):
        low = window[index]
        high = window[index + 1]
        if low == 0 and high == 0:
            return (chars + 1) * 2 if chars >= 3 else 0
        if high != 0 or not (0x20 <= low <= 0x7E):
            return 0
        chars += 1
        if chars > 256:
            return 0
        index += 2
    return 0


def _sanitize_identifier(name: str, fallback: str) -> str:
    """Turn an arbitrary field/struct name into a C/C++/C# safe identifier."""
    cleaned = _IDENT_CLEAN_RE.sub("_", name or "")
    cleaned = _IDENT_RUN_RE.sub("_", cleaned).strip("_")
    if not cleaned:
        cleaned = fallback
    if cleaned[0].isdigit():
        cleaned = f"f_{cleaned}"
    if cleaned.lower() in _CPP_RESERVED:
        cleaned = f"{cleaned}_field"
    return cleaned


def _parse_hint(hint: str) -> Tuple[str, Optional[int]]:
    """Parse ``"int32"`` / ``"pointer"`` / ``"string:0x20"`` into (kind, size)."""
    text = (hint or "").strip()
    if not text:
        raise ReclassError("empty type hint")
    base, sep, size_text = text.partition(":")
    key = base.strip().lower()
    if key not in _HINT_ALIASES:
        known = ", ".join(sorted(set(_HINT_ALIASES)))
        raise ReclassError(f"unknown type hint {hint!r}; expected one of: {known}")
    kind = _HINT_ALIASES[key]
    size: Optional[int] = None
    if sep:
        stripped = size_text.strip()
        try:
            size = int(stripped, 0)
        except ValueError:
            raise ReclassError(f"invalid byte size {stripped!r} in type hint {hint!r}") from None
        if size <= 0:
            raise ReclassError(f"byte size must be positive in type hint {hint!r}")
    return kind, size


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------


@dataclass
class StructField:
    """One member of a synthesized structure layout."""

    offset: int
    name: str
    field_type: str
    size: int
    comment: str = ""

    def __post_init__(self) -> None:
        if self.offset < 0:
            raise ReclassError(f"field offset must be non-negative, got {self.offset}")
        if self.size <= 0:
            raise ReclassError(f"field size must be positive, got {self.size}")
        if self.field_type not in FIELD_TYPES:
            raise ReclassError(
                f"unknown field type {self.field_type!r}; expected one of "
                f"{sorted(FIELD_TYPES)}"
            )

    @property
    def end(self) -> int:
        """Offset one past the last byte of this field."""
        return self.offset + self.size

    def to_dict(self) -> Dict[str, Any]:
        return {
            "offset": self.offset,
            "offset_hex": f"0x{self.offset:04x}",
            "name": self.name,
            "field_type": self.field_type,
            "size": self.size,
            "comment": self.comment,
        }

    def to_reclass_node_type(self) -> str:
        """ReClass.NET ``Node@Type`` for this field."""
        if self.field_type == "padding":
            return "Hex64" if self.size % 8 == 0 and self.size >= 8 else "Hex8"
        return _RC_NET_NODE_TYPES[self.field_type]

    def cpp_declaration(self, identifier: str) -> str:
        """One C++ member declaration for this field."""
        if self.field_type == "padding":
            return f"std::uint8_t {identifier}[0x{self.size:02X}];"
        if self.field_type == "string":
            return f"char {identifier}[0x{self.size:02X}];"
        if self.field_type == "wstring":
            return f"char16_t {identifier}[0x{self.size // 2:02X}];"
        return f"{_CPP_SCALAR_TYPES[self.field_type]} {identifier};"

    def csharp_declaration(self, identifier: str) -> str:
        """One C# member declaration for this field."""
        if self.field_type in ("padding", "string", "wstring"):
            return (
                f"[MarshalAs(UnmanagedType.ByValArray, SizeConst = {self.size})] "
                f"public byte[] {identifier};"
            )
        if self.field_type == "bool":
            return f"[MarshalAs(UnmanagedType.I1)] public bool {identifier};"
        return f"public {_CSHARP_SCALAR_TYPES[self.field_type]} {identifier};"


@dataclass
class StructLayout:
    """A synthesized structure layout plus its export surface."""

    name: str
    total_size: int
    fields: List[StructField] = field(default_factory=list)
    base_address: int = 0
    pointer_size: int = POINTER_SIZE_X64
    source: str = "synthesized"
    comment: str = ""

    # -- validation --------------------------------------------------------

    def validate(self) -> None:
        """Raise :class:`ReclassError` when the layout is not internally consistent."""
        if self.total_size <= 0:
            raise ReclassError("total_size must be positive")
        if self.total_size > MAX_STRUCT_SIZE:
            raise ReclassError(
                f"layout size 0x{self.total_size:x} exceeds MAX_STRUCT_SIZE "
                f"(0x{MAX_STRUCT_SIZE:x})"
            )
        if len(self.fields) > MAX_FIELDS:
            raise ReclassError(f"layout has {len(self.fields)} fields; MAX_FIELDS is {MAX_FIELDS}")
        previous_end = 0
        for index, item in enumerate(self.fields):
            if item.end > self.total_size:
                raise ReclassError(
                    f"field {item.name!r} ends at 0x{item.end:x}, past total_size "
                    f"0x{self.total_size:x}"
                )
            if index and item.offset < previous_end:
                raise ReclassError(
                    f"field {item.name!r} at 0x{item.offset:x} overlaps the previous "
                    f"field ending at 0x{previous_end:x}"
                )
            previous_end = max(previous_end, item.end)

    def field_at(self, offset: int) -> Optional[StructField]:
        """Field covering ``offset``, or ``None`` (paddings included)."""
        for item in self.fields:
            if item.offset <= offset < item.end:
                return item
        return None

    # -- serialization -----------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "total_size": self.total_size,
            "pointer_size": self.pointer_size,
            "base_address": self.base_address,
            "source": self.source,
            "comment": self.comment,
            "fields": [item.to_dict() for item in self.fields],
        }

    def to_json(self) -> str:
        """Deterministic JSON projection of the layout (stable key order)."""
        self.validate()
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def to_reclass_xml(self) -> str:
        """ReClass.NET (``.rcnet``) project XML for this layout.

        The root is ``<RCNetProject>`` carrying ``<Classes>/<Class>`` with one
        ``<Node Name= Type= Size= Offset= Comment= />`` per field, which is the
        shape ReClass.NET reads when the project is opened.
        """
        self.validate()
        root = ET.Element(
            "RCNetProject",
            {"FileVersion": RC_NET_FILE_VERSION, "Generator": "toothpick"},
        )
        classes = ET.SubElement(root, "Classes")
        class_attrs: Dict[str, str] = {
            "Name": self.name,
            "Size": f"0x{self.total_size:x}",
            "TotalSize": str(self.total_size),
            "PointerSize": str(self.pointer_size),
        }
        if self.base_address:
            class_attrs["Address"] = f"0x{self.base_address:x}"
        if self.comment:
            class_attrs["Comment"] = self.comment
        class_el = ET.SubElement(classes, "Class", class_attrs)
        nodes = ET.SubElement(class_el, "Nodes")
        for item in self.fields:
            node_attrs: Dict[str, str] = {
                "Name": item.name,
                "Type": item.to_reclass_node_type(),
                "Size": str(item.size),
                "Offset": f"0x{item.offset:04X}",
            }
            if item.comment:
                node_attrs["Comment"] = item.comment
            ET.SubElement(nodes, "Node", node_attrs)
        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        body = ET.tostring(root, encoding="unicode")
        return '<?xml version="1.0" encoding="utf-8"?>\n' + body + "\n"

    def to_cpp(self, namespace: Optional[str] = None) -> str:
        """C++ header source with offset annotations and a ``static_assert`` size gate."""
        self.validate()
        identifier = _sanitize_identifier(self.name, "SynthesizedStruct")
        declarations = [
            (
                item.cpp_declaration(
                    _sanitize_identifier(item.name, f"field_{item.offset:04X}")
                ),
                item,
            )
            for item in self.fields
        ]
        width = min(max((len(decl) for decl, _ in declarations), default=0), 56)
        lines: List[str] = [
            "// Auto-generated by toothpick (UNIT-W18 ReClass automation). Do not edit.",
            f"// Struct: {identifier}  size: 0x{self.total_size:x} ({self.total_size} bytes)",
        ]
        if self.comment:
            lines.append(f"// {self.comment}")
        lines.extend(["", "#pragma once", "", "#include <cstddef>", "#include <cstdint>", ""])
        if namespace:
            lines.append(f"namespace {_sanitize_identifier(namespace, 'toothpick')} {{")
            lines.append("")
        lines.append("#pragma pack(push, 1)")
        lines.append(f"struct {identifier} {{")
        for decl, item in declarations:
            body = f"    /* 0x{item.offset:04X} */ {decl.ljust(width)}"
            if item.comment:
                body = f"{body} // {item.comment}"
            lines.append(body.rstrip())
        lines.append("};")
        lines.append("#pragma pack(pop)")
        if namespace:
            lines.extend(["", f"}}  // namespace {namespace}"])
        lines.extend(
            [
                "",
                f'static_assert(sizeof({identifier}) == 0x{self.total_size:x}, '
                f'"{identifier} size mismatch");',
                "",
            ]
        )
        return "\n".join(lines)

    def to_csharp(self, namespace: str = "Toothpick.Generated") -> str:
        """C# struct source using explicit layout and per-field offsets."""
        self.validate()
        identifier = _sanitize_identifier(self.name, "SynthesizedStruct")
        ns = _sanitize_identifier(namespace, "Toothpick") if namespace else ""
        lines: List[str] = [
            "// Auto-generated by toothpick (UNIT-W18 ReClass automation). Do not edit.",
            f"// Struct: {identifier}  size: 0x{self.total_size:x} ({self.total_size} bytes)",
        ]
        if self.comment:
            lines.append(f"// {self.comment}")
        lines.extend(["", "using System;", "using System.Runtime.InteropServices;", ""])
        if ns:
            lines.append(f"namespace {ns}")
            lines.append("{")
            pad = "    "
        else:
            pad = ""
        lines.append(
            f"{pad}[StructLayout(LayoutKind.Explicit, Size = 0x{self.total_size:x}, Pack = 1)]"
        )
        lines.append(f"{pad}public struct {identifier}")
        lines.append(f"{pad}{{")
        for item in self.fields:
            member = _sanitize_identifier(item.name, f"field_{item.offset:04X}")
            line = f"{pad}    [FieldOffset(0x{item.offset:04X})] {item.csharp_declaration(member)}"
            if item.comment:
                line = f"{line}  // {item.comment}"
            lines.append(line)
        lines.append(f"{pad}}}")
        if ns:
            lines.append("}")
        lines.append("")
        return "\n".join(lines)

    # -- type lattice ingestion -------------------------------------------

    def to_type_record(
        self,
        binary_sha256: str = "",
        lattice_rank: int = 2,
        type_id: Optional[str] = None,
    ) -> TypeRecord:
        """Project this layout onto a ``types`` row of the global type lattice."""
        self.validate()
        return TypeRecord(
            type_id=type_id or self.name,
            name=self.name,
            kind="struct",
            size_bytes=self.total_size,
            layout=self.to_dict(),
            lattice_rank=lattice_rank,
            binary_sha256=binary_sha256,
        )


# ---------------------------------------------------------------------------
# Layout synthesis
# ---------------------------------------------------------------------------


class StructSynthesizer:
    """Synthesizes :class:`StructLayout` records from memory or access hints."""

    def __init__(
        self,
        pointer_size: int = POINTER_SIZE_X64,
        module_ranges: Optional[Sequence[Tuple[int, int]]] = None,
        base_address: int = 0,
    ) -> None:
        if pointer_size not in (POINTER_SIZE_X86, POINTER_SIZE_X64):
            raise ReclassError(f"pointer_size must be 4 or 8, got {pointer_size}")
        self.pointer_size = pointer_size
        self.module_ranges: Tuple[Tuple[int, int], ...] = tuple(
            (int(start), int(end)) for start, end in (module_ranges or ())
        )
        self.base_address = int(base_address)

    # -- memory buffer -----------------------------------------------------

    def synthesize_from_memory(
        self,
        buffer: bytes,
        base_addr: int = 0,
        name: str = "SynthesizedStruct",
        pointer_size: Optional[int] = None,
        module_ranges: Optional[Sequence[Tuple[int, int]]] = None,
    ) -> StructLayout:
        """Infer a layout from ``buffer`` (a memory image starting at ``base_addr``).

        The resulting fields are contiguous and sum exactly to ``len(buffer)``:
        every byte of the sample is accounted for, either as a typed field or as
        a padding array.
        """
        if isinstance(buffer, (bytearray, memoryview)):
            data = bytes(buffer)
        elif isinstance(buffer, bytes):
            data = buffer
        else:
            raise ReclassError(f"buffer must be bytes-like, got {type(buffer).__name__}")
        if not data:
            raise ReclassError("buffer is empty: nothing to synthesize")
        if len(data) > MAX_STRUCT_SIZE:
            raise ReclassError(
                f"buffer of {len(data)} bytes exceeds MAX_STRUCT_SIZE "
                f"(0x{MAX_STRUCT_SIZE:x})"
            )
        size_ptr = pointer_size or self.pointer_size
        if size_ptr not in (POINTER_SIZE_X86, POINTER_SIZE_X64):
            raise ReclassError(f"pointer_size must be 4 or 8, got {size_ptr}")
        ranges = tuple(module_ranges) if module_ranges is not None else self.module_ranges
        base = int(base_addr) or self.base_address

        fields: List[StructField] = []
        position = 0
        while position < len(data):
            if len(fields) >= MAX_FIELDS:
                fields.append(self._padding_field(position, len(data) - position))
                break
            field_type, size, comment = self._classify(data, position, base, ranges, size_ptr)
            size = max(1, min(size, len(data) - position))
            fields.append(
                StructField(
                    offset=position,
                    name=self._field_name(field_type, position),
                    field_type=field_type,
                    size=size,
                    comment=comment,
                )
            )
            position += size

        return StructLayout(
            name=name,
            total_size=len(data),
            fields=fields,
            base_address=base,
            pointer_size=size_ptr,
            source="synthesize_from_memory",
        )

    def _classify(
        self,
        data: bytes,
        position: int,
        base_addr: int,
        module_ranges: Sequence[Tuple[int, int]],
        pointer_size: int,
    ) -> Tuple[str, int, str]:
        """Classify the field starting at ``position`` -> (field_type, size, comment)."""
        window = data[position:]
        remaining = len(window)

        # 1) a pointer-shaped word at offset 0 is the virtual method table.
        if position == 0 and remaining >= pointer_size:
            value = int.from_bytes(window[:pointer_size], "little")
            if _pointer_plausible(
                value, pointer_size, base_addr, len(data), module_ranges
            ):
                return ("vtable", pointer_size, f"vtable (void**) 0x{value:x}")

        # 2/3) strings.
        ascii_len = _ascii_string_length(window)
        if ascii_len:
            return ("string", ascii_len, f"ASCII string ({ascii_len - 1} chars + NUL)")
        utf16_len = _utf16_string_length(window)
        if utf16_len:
            return (
                "wstring",
                utf16_len,
                f"UTF-16LE string ({utf16_len // 2 - 1} chars + NUL)",
            )

        # 4) IEEE-754 values, accepted only when the integer reading is noise.
        if remaining >= 4:
            raw32 = _struct.unpack_from("<I", data, position)[0]
            as_int32 = _struct.unpack_from("<i", data, position)[0]
            if _float_plausible(raw32) and abs(as_int32) > _INT32_NOISE:
                value = _struct.unpack_from("<f", data, position)[0]
                return ("float", 4, f"float {value!r}")
        if remaining >= 8:
            raw64 = _struct.unpack_from("<Q", data, position)[0]
            as_int64 = _struct.unpack_from("<q", data, position)[0]
            if _double_plausible(raw64) and abs(as_int64) > _INT64_NOISE:
                value = _struct.unpack_from("<d", data, position)[0]
                return ("double", 8, f"double {value!r}")

        # 5) pointers to heap/module ranges.
        if remaining >= pointer_size:
            value = int.from_bytes(window[:pointer_size], "little")
            if _pointer_plausible(
                value, pointer_size, base_addr, len(data), module_ranges
            ):
                return ("pointer", pointer_size, f"pointer 0x{value:x}")

        # 6) booleans: a true byte, three zero bytes, then a further field.
        if remaining >= 5 and window[0] == 1 and window[1:4] == b"\x00\x00\x00":
            if any(window[4:8]):
                return ("bool", 1, "bool true")

        # 7) padding arrays.
        zero_run = 0
        while zero_run < remaining and window[zero_run] == 0:
            zero_run += 1
        if zero_run >= PADDING_MIN_RUN:
            return ("padding", zero_run, f"padding ({zero_run} zero bytes)")
        if zero_run == remaining and zero_run > 0:
            return ("padding", zero_run, f"padding ({zero_run} zero bytes)")

        # 8) narrow integers sitting in a zero-padded slot.
        if remaining >= 6 and 0 < window[0] <= 0x7F and window[1:5] == b"\x00" * 4:
            return ("int8", 1, f"int8 {window[0]} (zero-padded slot)")
        if remaining >= 6:
            value16 = int.from_bytes(window[:2], "little")
            if 0 < value16 <= 0x7FFF and window[2:6] == b"\x00" * 4:
                return ("int16", 2, f"int16 {value16} (zero-padded slot)")

        # 9) remaining integers.
        if remaining >= 4:
            value32 = _struct.unpack_from("<i", data, position)[0]
            if abs(value32) <= _INT32_NOISE:
                return ("int32", 4, f"int32 {value32}")
            if remaining >= 8:
                high = _struct.unpack_from("<I", data, position + 4)[0]
                value64 = _struct.unpack_from("<q", data, position)[0]
                if high in (0, 0xFFFF_FFFF) and not (-(1 << 31) <= value64 < (1 << 31)):
                    return ("int64", 8, f"int64 {value64}")
            unsigned = _struct.unpack_from("<I", data, position)[0]
            return ("int32", 4, f"int32 0x{unsigned:08x} (unsigned reading)")
        if remaining >= 2:
            value16 = _struct.unpack_from("<h", data, position)[0]
            return ("int16", 2, f"int16 {value16}")
        value8 = _struct.unpack_from("<b", data, position)[0]
        return ("int8", 1, f"int8 {value8}")

    def _field_name(self, field_type: str, offset: int) -> str:
        if field_type == "vtable":
            return "vtable"
        return f"{_NAME_PREFIXES[field_type]}_{offset:04X}"

    @staticmethod
    def _padding_field(offset: int, size: int) -> StructField:
        return StructField(
            offset=offset,
            name=f"pad_{offset:04X}",
            field_type="padding",
            size=size,
            comment=f"padding ({size} zero bytes)",
        )

    # -- access hints ------------------------------------------------------

    def synthesize_from_accesses(
        self,
        offsets: List[Tuple[int, str]],
        total_size: int,
        name: str = "SynthesizedStruct",
        pointer_size: int = POINTER_SIZE_X64,
    ) -> StructLayout:
        """Build a layout from ``(offset, type_hint)`` pairs (disassembly/scan hints).

        Hints are explicit, so no guessing happens for the named offsets; the gaps
        between them become padding arrays.  Offsets must be non-negative and
        inside ``total_size``.  Duplicate or overlapping accesses are dropped (the
        first after sorting wins) rather than raising, because a scan can report
        the same slot twice.
        """
        if total_size <= 0:
            raise ReclassError("total_size must be positive")
        if total_size > MAX_STRUCT_SIZE:
            raise ReclassError(
                f"total_size 0x{total_size:x} exceeds MAX_STRUCT_SIZE "
                f"(0x{MAX_STRUCT_SIZE:x})"
            )
        if not offsets:
            raise ReclassError("no field accesses supplied")
        if pointer_size not in (POINTER_SIZE_X86, POINTER_SIZE_X64):
            raise ReclassError(f"pointer_size must be 4 or 8, got {pointer_size}")

        resolved: List[Tuple[int, str, int]] = []
        for entry in offsets:
            if not isinstance(entry, (tuple, list)) or len(entry) != 2:
                raise ReclassError(f"access entries must be (offset, hint) pairs, got {entry!r}")
            offset = int(entry[0])
            kind, explicit_size = _parse_hint(str(entry[1]))
            fixed = _HINT_FIXED_SIZES.get(kind)
            size = explicit_size or (pointer_size if fixed is None else fixed)
            if size is None or size <= 0:
                raise ReclassError(
                    f"type hint {entry[1]!r} needs an explicit byte size, e.g. '{entry[1]}:0x20'"
                )
            if offset < 0:
                raise ReclassError(f"access offset must be non-negative, got {offset}")
            if offset + size > total_size:
                raise ReclassError(
                    f"access at 0x{offset:x} (size {size}) leaves total_size 0x{total_size:x}"
                )
            resolved.append((offset, kind, size))

        resolved.sort(key=lambda item: (item[0], item[1], item[2]))
        fields: List[StructField] = []
        cursor = 0
        for offset, kind, size in resolved:
            if offset < cursor:
                continue  # duplicate / overlapping access: keep the first
            if len(fields) >= MAX_FIELDS:
                break
            if offset > cursor:
                fields.append(self._padding_field(cursor, offset - cursor))
            fields.append(
                StructField(
                    offset=offset,
                    name=self._field_name(kind, offset),
                    field_type=kind,
                    size=size,
                    comment=f"from access hint {kind!r}",
                )
            )
            cursor = offset + size
        if cursor < total_size:
            fields.append(self._padding_field(cursor, total_size - cursor))

        return StructLayout(
            name=name,
            total_size=total_size,
            fields=fields,
            pointer_size=pointer_size,
            source="synthesize_from_accesses",
        )

    # -- scan results ------------------------------------------------------

    def synthesize_from_scan_result(
        self,
        result: Any,
        name: Optional[str] = None,
    ) -> StructLayout:
        """Project a :class:`toothpick.scanner.ScanResult` onto a layout.

        Every match becomes a field of the type the scan searched for, addressed
        relative to the lowest match.  The result is bounded to
        ``MAX_STRUCT_SIZE`` bytes and ``MAX_FIELDS`` fields: a pointer scan can
        report matches across gigabytes of address space, and an export must
        never grow without bound (Risk R1).
        """
        matches = list(getattr(result, "matches", None) or [])
        if not matches:
            raise ReclassError("cannot export a ReClass layout: the scan produced no matches")
        data_type = str(getattr(result, "data_type", "i32")).lower()
        base_hint = _SCAN_HINT_MAP.get(data_type, "int8")
        ordered = sorted(matches, key=lambda match: int(match.address))
        base = int(ordered[0].address)
        accesses: List[Tuple[int, str]] = []
        end = 0
        truncated = 0
        for match in ordered:
            size = max(int(getattr(match, "size", 0) or 0), 1)
            offset = int(match.address) - base
            if offset + size > MAX_STRUCT_SIZE or len(accesses) >= MAX_FIELDS:
                truncated += 1
                continue
            hint = base_hint
            if base_hint in ("string", "padding"):
                hint = f"{base_hint}:0x{size:x}"
            accesses.append((offset, hint))
            end = max(end, offset + size)
        if not accesses:
            raise ReclassError("cannot export a ReClass layout: no match fits the export bound")
        layout_name = name or f"Scan_{getattr(result, 'session_id', 'session')}"
        pointer_size = POINTER_SIZE_X64
        if base_hint == "pointer":
            sizes = {max(int(getattr(m, "size", 0) or 0), 1) for m in ordered}
            pointer_size = POINTER_SIZE_X86 if sizes == {POINTER_SIZE_X86} else POINTER_SIZE_X64
        comment = (
            f"ReClass export of scan session {getattr(result, 'session_id', '?')} "
            f"({data_type}); {len(accesses)} of {len(ordered)} matches"
        )
        if truncated:
            comment += f", {truncated} beyond the 0x{MAX_STRUCT_SIZE:x} byte export bound"
        layout = self.synthesize_from_accesses(
            accesses,
            total_size=max(end, 1),
            name=layout_name,
            pointer_size=pointer_size,
        )
        layout.base_address = base
        layout.comment = comment
        return layout


# ---------------------------------------------------------------------------
# Type-lattice ingestion
# ---------------------------------------------------------------------------


def ingest_layout(
    db: Database,
    layout: StructLayout,
    binary_sha256: str = "",
    lattice_rank: int = 2,
    type_id: Optional[str] = None,
) -> str:
    """Persist ``layout`` into the ``types`` table of ``toothpick.db``.

    Returns the ``type_id`` used.  ``lattice_rank`` defaults to 2 (a synthesized
    aggregate outranks a primitive reading of the same bytes).
    """
    record = layout.to_type_record(
        binary_sha256=binary_sha256, lattice_rank=lattice_rank, type_id=type_id
    )
    db.upsert_type(record)
    return record.type_id
