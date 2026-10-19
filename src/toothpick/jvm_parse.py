"""Toothpick JVM structural ingest -- class-file / JAR parsing half (J1 split).

Verbatim parsing half of :mod:`toothpick.jvm` (milestone J1, module split):
the opcode-length tables, :class:`JvmError`, the ``Jvm*`` data contracts, the
bounds-checked :class:`_Reader`, the constant-pool walk (modified UTF-8
included), the instruction-length decoder and ``invoke*`` scan, the attribute /
field / method readers, :func:`parse_class_file`, the JAR readers and
:func:`parse_jar`, and the deterministic synthetic-ordinal and type-id helpers
through :func:`_display_name`.

Binding contract: ``docs/units/UNIT-J1-JVM-PLAN.md`` section 1 (Chunk 1) and
section 2 (interface specification); milestone J1 of ``docs/NEXT-TRACK.md``
("Class-file/JAR parsing; methods/fields/``invoke*`` callgraph into
``funcs``/``edges``.  No lifting, no SSA.").  Design anchors:
``docs/design-document.md`` section 1 ("no fabricated structs -- every class row
traces to parsed bytes") and section 7 (edge taxonomy: ``call``/``xref``/``cfg``).

This is a **pure-stdlib** (``int.from_bytes`` + ``zipfile``) *structural* reader
for the JVM class-file format (JVMS 4.1-4.7): class names, methods with their
``Code`` ``code_length``, fields, and the ``invoke*`` call sites -- and nothing
more.  There is no lifting, no SSA, no taint, no decompilation and no live-heap
grounding (the JVM boundary of ``docs/NEXT-TRACK.md``).

Non-fabrication rules carried over unchanged:

* **Full constant-pool tag coverage** ``1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15,
  16, 17, 18, 19, 20``; exotic tags are parsed *structurally* so the pool
  strides correctly and are never executed.  ``long``/``double`` consume two
  slots; an unknown tag raises :class:`JvmError` instead of mis-striding.
* **``invoke*`` sites are found by decoding the instruction stream**, not by
  scanning ``Code`` for bytes that happen to equal ``0xB6..0xBA``.
* **Every reference is resolved through the constant pool or refused.**  A
  malformed ``0xB6-0xB9`` operand raises :class:`JvmError`; a target that is not
  one of the parsed methods is left ``resolved=False`` for the storage half to
  count ``unresolved`` -- no placeholder row and no fake edge is created here.
* **Truncation and bad magic are hard failures.**  A partially parsed
  :class:`JvmClass` is never returned, so the storage half can build every row
  *before* opening a database.

Synthetic addresses are labelled as synthetic: ``funcs.func_addr`` is a
deterministic 1-based ordinal over methods in ``(class, method)`` sorted order
(:func:`method_addr_index`) -- never a bytecode offset, never a file offset and
never an address in any image.

This module owns no database and no storage import (the ``toothpick.storage``
imports stay in :mod:`toothpick.jvm`): it is pure bytes-in / objects-out, so the
J1 split stays testable without a database.
"""

from __future__ import annotations

import io
import struct
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

__all__ = [
    "CALLING_CONVENTION",
    "CP_TAG_NAMES",
    "INVOKE_NAMES",
    "INVOKE_OPS",
    "JVM_FIELD_PREFIX",
    "JVM_TYPE_PREFIX",
    "MAGIC",
    "JvmClass",
    "JvmError",
    "JvmField",
    "JvmInvoke",
    "JvmMethod",
    "class_type_id",
    "field_type_id",
    "method_addr_index",
    "parse_class_file",
    "parse_jar",
]

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: JVMS 4.1: every class file starts with this 4-byte magic.
MAGIC = b"\xca\xfe\xba\xbe"

#: Calling convention stamped on every JVM method row (J1 acceptance: a JVM
#: method is identifiable in ``funcs`` without inspecting the name).
CALLING_CONVENTION = "jvm"

#: Type-id namespace of JVM class rows (mirrors ``rtti:`` in ``rtti_store``).
JVM_TYPE_PREFIX = "jvm:"

#: Separator between a class half and a member half of a field type id.
JVM_FIELD_PREFIX = "#"

#: CP tags (JVMS 4.4).  Parsed structurally; never interpreted.
_CP_UTF8 = 1
_CP_INTEGER = 3
_CP_FLOAT = 4
_CP_LONG = 5
_CP_DOUBLE = 6
_CP_CLASS = 7
_CP_STRING = 8
_CP_FIELDREF = 9
_CP_METHODREF = 10
_CP_INTERFACE_METHODREF = 11
_CP_NAME_AND_TYPE = 12
_CP_METHOD_HANDLE = 15
_CP_METHOD_TYPE = 16
_CP_DYNAMIC = 17
_CP_INVOKE_DYNAMIC = 18
_CP_MODULE = 19
_CP_PACKAGE = 20

#: Human-readable tag names (error messages / layout evidence).
CP_TAG_NAMES: Dict[int, str] = {
    _CP_UTF8: "Utf8",
    _CP_INTEGER: "Integer",
    _CP_FLOAT: "Float",
    _CP_LONG: "Long",
    _CP_DOUBLE: "Double",
    _CP_CLASS: "Class",
    _CP_STRING: "String",
    _CP_FIELDREF: "Fieldref",
    _CP_METHODREF: "Methodref",
    _CP_INTERFACE_METHODREF: "InterfaceMethodref",
    _CP_NAME_AND_TYPE: "NameAndType",
    _CP_METHOD_HANDLE: "MethodHandle",
    _CP_METHOD_TYPE: "MethodType",
    _CP_DYNAMIC: "Dynamic",
    _CP_INVOKE_DYNAMIC: "InvokeDynamic",
    _CP_MODULE: "Module",
    _CP_PACKAGE: "Package",
}

#: The five invoke opcodes the J1 callgraph is built from (plan section 2).
INVOKE_OPS = frozenset({0xB6, 0xB7, 0xB8, 0xB9, 0xBA})

#: Opcode -> mnemonic, for diagnostics and layout evidence only.
INVOKE_NAMES: Dict[int, str] = {
    0xB6: "invokevirtual",
    0xB7: "invokespecial",
    0xB8: "invokestatic",
    0xB9: "invokeinterface",
    0xBA: "invokedynamic",
}

#: Revised opcodes that take no operand bytes (length 1).
_NO_OPERAND_RANGES: Tuple[Tuple[int, int], ...] = (
    (0x00, 0x0F),  # nop .. dconst_1
    (0x1A, 0x35),  # iload_0 .. saload
    (0x3B, 0x83),  # istore_0 .. lxor
    (0x85, 0x98),  # i2l .. dcmpg
    (0xAC, 0xB1),  # ireturn .. dreturn
    (0xBE, 0xBF),  # arraylength, athrow
    (0xC2, 0xC3),  # monitorenter, monitorexit
)

#: Revised opcodes with a 1-byte operand.
_ONE_OPERAND_RANGES: Tuple[Tuple[int, int], ...] = (
    (0x10, 0x10),  # bipush
    (0x12, 0x12),  # ldc (ONE operand byte: an 8-bit CP index, unlike ldc_w/ldc2_w)
    (0x15, 0x19),  # iload .. aload
    (0x36, 0x3A),  # istore .. astore
    (0xA9, 0xA9),  # ret
    (0xBC, 0xBC),  # newarray
)

#: Revised opcodes with a 2-byte operand.
_TWO_OPERAND_RANGES: Tuple[Tuple[int, int], ...] = (
    (0x11, 0x11),  # sipush
    (0x13, 0x14),  # ldc_w, ldc2_w (2-byte CP index each)
    (0x84, 0x84),  # iinc
    (0x99, 0xA8),  # ifeq .. jsr
    (0xB2, 0xB8),  # getstatic .. invokestatic
    (0xBB, 0xBB),  # new
    (0xBD, 0xBD),  # anewarray
    (0xC0, 0xC1),  # checkcast, instanceof
    (0xC6, 0xC7),  # ifnull, ifnonnull
)

#: Revised opcodes with a 3-byte operand.
_THREE_OPERAND_RANGES: Tuple[Tuple[int, int], ...] = ((0xC5, 0xC5),)  # multianewarray

#: Revised opcodes with a 4-byte operand (``invokeinterface`` declares its CP
#: index in the first two of them -- that is the slot the callgraph reads).
_FOUR_OPERAND_RANGES: Tuple[Tuple[int, int], ...] = (
    (0xB9, 0xBA),  # invokeinterface, invokedynamic
    (0xC8, 0xC9),  # goto_w, jsr_w
)

#: Opcodes whose length depends on their payload.
_VARIABLE_OPS = frozenset({0xAA, 0xAB, 0xC4})  # tableswitch, lookupswitch, wide

#: Sentinel for "reserved opcode, no legal length".
_UNKNOWN_LENGTH = 0xFF


def _build_opcode_lengths() -> List[int]:
    """One 256-entry table of fixed operand counts (``0xFF`` = variable/reserved)."""
    table = [_UNKNOWN_LENGTH] * 256
    for ranges, length in (
        (_NO_OPERAND_RANGES, 0),
        (_ONE_OPERAND_RANGES, 1),
        (_TWO_OPERAND_RANGES, 2),
        (_THREE_OPERAND_RANGES, 3),
        (_FOUR_OPERAND_RANGES, 4),
    ):
        for low, high in ranges:
            for opcode in range(low, high + 1):
                table[opcode] = length
    for opcode in _VARIABLE_OPS:
        table[opcode] = _UNKNOWN_LENGTH
    return table


_OPCODE_OPERANDS = _build_opcode_lengths()

#: ``wide``-modified opcodes that take a further 2 local-variable bytes.
_WIDE_TWO_BYTE = frozenset(
    {0x15, 0x16, 0x17, 0x18, 0x19, 0x36, 0x37, 0x38, 0x39, 0x3A, 0xA9}
)


class JvmError(Exception):
    """Raised for any class file / JAR this module refuses to read structurally.

    Refusal is the whole point: a truncated pool, an unknown tag, a reserved
    opcode or a bad magic never degrades into a partial :class:`JvmClass` or a
    partially written set of rows.
    """


# --------------------------------------------------------------------------- #
# Data contracts (plan section 2)
# --------------------------------------------------------------------------- #


@dataclass
class JvmMethod:
    """One ``methods[]`` entry (JVMS 4.6); ``code_len`` is the ``Code`` attribute's ``code_length``."""

    class_name: str
    name: str
    descriptor: str
    access_flags: int
    code_len: int


@dataclass
class JvmField:
    """One ``fields[]`` entry (JVMS 4.5).  Attributes are skipped structurally."""

    class_name: str
    name: str
    descriptor: str
    access_flags: int


@dataclass
class JvmInvoke:
    """One decoded ``invoke*`` instruction site (0xB6-0xBA) and its CP resolution.

    ``resolved`` means the constant pool yielded a *named target*: a class plus
    member name and descriptor for ``0xB6-0xB9``.  An ``invokedynamic`` (0xBA)
    site resolves to a name/descriptor only (its real target is a bootstrap
    method), so it is recorded ``resolved=False`` with ``reason='dynamic'`` and
    is counted ``unresolved`` by :func:`ingest` -- never given a fake class.
    """

    class_name: str
    method_name: str
    method_descriptor: str
    code_offset: int
    opcode: int
    cp_index: int
    cp_tag: int
    target_class: str
    target_name: str
    target_descriptor: str
    resolved: bool
    reason: str = ""


@dataclass
class JvmClass:
    """One parsed class file.

    The plan names ``(name, methods, fields)``; the remaining fields are
    additive parsed evidence with defaults, so that constructor contract still
    holds.  ``name`` is the *internal* name from the constant pool (``a/b/C``).
    """

    name: str
    methods: List[JvmMethod]
    fields: List[JvmField]
    invokes: List[JvmInvoke] = field(default_factory=list)
    access_flags: int = 0
    super_name: str = ""
    interfaces: List[str] = field(default_factory=list)
    minor_version: int = 0
    major_version: int = 0

    @property
    def dotted_name(self) -> str:
        """The internal name in source form (``a.b.C``) -- for display only."""
        return self.name.replace("/", ".")


# --------------------------------------------------------------------------- #
# Bounds-checked reader
# --------------------------------------------------------------------------- #


class _Reader:
    """Forward cursor over a byte range that refuses to read past its end."""

    __slots__ = ("_data", "_pos", "_label")

    def __init__(self, data: bytes, label: str) -> None:
        self._data = data
        self._pos = 0
        self._label = label

    @property
    def pos(self) -> int:
        return self._pos

    @property
    def size(self) -> int:
        return len(self._data)

    @property
    def remaining(self) -> int:
        return len(self._data) - self._pos

    def _need(self, count: int) -> None:
        if count < 0:
            raise JvmError(f"{self._label}: negative read of {count} byte(s)")
        if self._pos + count > len(self._data):
            raise JvmError(
                f"{self._label}: truncated at offset {self._pos}: "
                f"needed {count} byte(s), {self.remaining} remaining "
                f"(buffer is {len(self._data)} bytes)"
            )

    def u1(self) -> int:
        self._need(1)
        value = self._data[self._pos]
        self._pos += 1
        return value

    def u2(self) -> int:
        self._need(2)
        value = int.from_bytes(self._data[self._pos : self._pos + 2], "big")
        self._pos += 2
        return value

    def u4(self) -> int:
        self._need(4)
        value = int.from_bytes(self._data[self._pos : self._pos + 4], "big")
        self._pos += 4
        return value

    def read(self, count: int) -> bytes:
        self._need(count)
        chunk = self._data[self._pos : self._pos + count]
        self._pos += count
        return chunk

    def skip(self, count: int) -> None:
        """Advance over ``count`` bytes without materialising them."""
        self._need(count)
        self._pos += count

    def seek(self, pos: int) -> None:
        if pos < 0 or pos > len(self._data):
            raise JvmError(
                f"{self._label}: cannot seek to offset {pos} "
                f"(buffer is {len(self._data)} bytes)"
            )
        self._pos = pos


# --------------------------------------------------------------------------- #
# Constant pool
# --------------------------------------------------------------------------- #


@dataclass
class _CpEntry:
    """One constant-pool entry: ``(tag, a, b)`` with tag-specific operands."""

    tag: int
    a: Any = None
    b: Any = None

    def describe(self) -> str:
        return CP_TAG_NAMES.get(self.tag, f"tag {self.tag}")


@dataclass
class _ConstantPool:
    """The parsed constant pool of one class file (index 0 is unusable)."""

    entries: List[Optional[_CpEntry]]

    def entry(self, index: int) -> _CpEntry:
        if not isinstance(index, int) or index < 1 or index >= len(self.entries):
            raise JvmError(
                f"constant pool index {index!r} is out of range "
                f"(1..{max(len(self.entries) - 1, 0)})"
            )
        entry = self.entries[index]
        if entry is None:
            raise JvmError(
                f"constant pool index {index} is the unused second slot of a "
                "Long/Double entry"
            )
        return entry

    def utf8(self, index: int) -> str:
        entry = self.entry(index)
        if entry.tag != _CP_UTF8:
            raise JvmError(
                f"constant pool #{index} is a {entry.describe()} entry, expected Utf8"
            )
        return entry.a

    def class_name(self, index: int) -> str:
        """Internal name of a ``CONSTANT_Class`` entry."""
        entry = self.entry(index)
        if entry.tag != _CP_CLASS:
            raise JvmError(
                f"constant pool #{index} is a {entry.describe()} entry, expected Class"
            )
        return self.utf8(entry.a)

    def name_and_type(self, index: int) -> Tuple[str, str]:
        entry = self.entry(index)
        if entry.tag != _CP_NAME_AND_TYPE:
            raise JvmError(
                f"constant pool #{index} is a {entry.describe()} entry, "
                "expected NameAndType"
            )
        return self.utf8(entry.a), self.utf8(entry.b)

    def member_ref(self, index: int) -> Tuple[str, str, str]:
        """``(class, name, descriptor)`` of a Methodref/InterfaceMethodref/Fieldref."""
        entry = self.entry(index)
        if entry.tag not in (
            _CP_METHODREF,
            _CP_INTERFACE_METHODREF,
            _CP_FIELDREF,
        ):
            raise JvmError(
                f"constant pool #{index} is a {entry.describe()} entry, "
                "expected Methodref/InterfaceMethodref/Fieldref"
            )
        class_name = self.class_name(entry.a)
        name, descriptor = self.name_and_type(entry.b)
        return class_name, name, descriptor


def _decode_modified_utf8(raw: bytes, index: int) -> str:
    """Decode a ``CONSTANT_Utf8`` payload (JVMS 4.4.7 is modified UTF-8).

    Standard UTF-8 first; then ``surrogatepass`` so a supplementary character
    encoded as two surrogate code units (legal modified UTF-8, illegal UTF-8)
    still decodes.  Anything else is refused rather than replaced, because a
    replaced name would be a fabricated name.
    """
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        pass
    try:
        return raw.decode("utf-8", "surrogatepass")
    except UnicodeDecodeError as exc:
        raise JvmError(
            f"constant pool #{index}: Utf8 entry is not decodable ({exc})"
        ) from exc


def _parse_constant_pool(reader: _Reader) -> _ConstantPool:
    """Walk the pool with full tag coverage (plan section 1, bullet 1)."""
    count = reader.u2()
    if count < 1:
        raise JvmError(f"constant pool count {count} is invalid (must be >= 1)")
    entries: List[Optional[_CpEntry]] = [None] * count
    index = 1
    while index < count:
        offset = reader.pos
        tag = reader.u1()
        if tag == _CP_UTF8:
            length = reader.u2()
            raw = reader.read(length)
            entries[index] = _CpEntry(tag, _decode_modified_utf8(raw, index))
        elif tag in (_CP_INTEGER, _CP_FLOAT):
            entries[index] = _CpEntry(tag, reader.u4())
        elif tag in (_CP_LONG, _CP_DOUBLE):
            entries[index] = _CpEntry(tag, reader.read(8))
            index += 1
            if index >= count:
                raise JvmError(
                    f"constant pool {CP_TAG_NAMES[tag]} at #{index - 1} has no "
                    "second slot to occupy (JVMS 4.4.5)"
                )
            entries[index] = None
        elif tag in (_CP_CLASS, _CP_STRING, _CP_METHOD_TYPE, _CP_MODULE, _CP_PACKAGE):
            entries[index] = _CpEntry(tag, reader.u2())
        elif tag in (
            _CP_FIELDREF,
            _CP_METHODREF,
            _CP_INTERFACE_METHODREF,
            _CP_NAME_AND_TYPE,
            _CP_DYNAMIC,
            _CP_INVOKE_DYNAMIC,
        ):
            entries[index] = _CpEntry(tag, reader.u2(), reader.u2())
        elif tag == _CP_METHOD_HANDLE:
            entries[index] = _CpEntry(tag, reader.u1(), reader.u2())
        else:
            raise JvmError(
                f"unsupported constant-pool tag {tag} at index {index} "
                f"(offset {offset})"
            )
        index += 1
    return _ConstantPool(entries)


# --------------------------------------------------------------------------- #
# Instruction stream (variable-length opcodes)
# --------------------------------------------------------------------------- #


def _instruction_length(code: bytes, offset: int) -> int:
    """Total byte length of the instruction at ``code[offset]``.

    Raises :class:`JvmError` on a reserved opcode, a malformed switch or a
    truncated instruction -- all of which mean the ``code`` array is not a valid
    instruction stream and no honest call site can be derived from it.
    """
    opcode = code[offset]
    if opcode == 0xAA:  # tableswitch
        pad = (4 - ((offset + 1) % 4)) % 4
        base = offset + 1 + pad
        if base + 12 > len(code):
            raise JvmError(
                f"code at offset {offset}: tableswitch header is truncated "
                f"(need {base + 12} bytes, have {len(code)})"
            )
        low = int.from_bytes(code[base + 4 : base + 8], "big", signed=True)
        high = int.from_bytes(code[base + 8 : base + 12], "big", signed=True)
        if high < low:
            raise JvmError(
                f"code at offset {offset}: tableswitch high ({high}) < low ({low})"
            )
        total = 1 + pad + 12 + 4 * (high - low + 1)
        if offset + total > len(code):
            raise JvmError(
                f"code at offset {offset}: tableswitch jumps are truncated "
                f"(need {offset + total} bytes, have {len(code)})"
            )
        return total
    if opcode == 0xAB:  # lookupswitch
        pad = (4 - ((offset + 1) % 4)) % 4
        base = offset + 1 + pad
        if base + 8 > len(code):
            raise JvmError(
                f"code at offset {offset}: lookupswitch header is truncated "
                f"(need {base + 8} bytes, have {len(code)})"
            )
        pairs = int.from_bytes(code[base + 4 : base + 8], "big", signed=True)
        if pairs < 0:
            raise JvmError(
                f"code at offset {offset}: lookupswitch npairs is negative ({pairs})"
            )
        total = 1 + pad + 8 + 8 * pairs
        if offset + total > len(code):
            raise JvmError(
                f"code at offset {offset}: lookupswitch pairs are truncated "
                f"(need {offset + total} bytes, have {len(code)})"
            )
        return total
    if opcode == 0xC4:  # wide
        if offset + 2 > len(code):
            raise JvmError(f"code at offset {offset}: wide prefix is truncated")
        modified = code[offset + 1]
        if modified == 0x84:
            total = 6
        elif modified in _WIDE_TWO_BYTE:
            total = 4
        else:
            raise JvmError(
                f"code at offset {offset}: wide prefix on opcode "
                f"0x{modified:02X} is not a wide-modified instruction"
            )
        if offset + total > len(code):
            raise JvmError(
                f"code at offset {offset}: wide instruction is truncated "
                f"(need {offset + total} bytes, have {len(code)})"
            )
        return total

    operands = _OPCODE_OPERANDS[opcode]
    if operands == _UNKNOWN_LENGTH:
        raise JvmError(
            f"code at offset {offset}: reserved/unknown opcode 0x{opcode:02X}"
        )
    total = 1 + operands
    if offset + total > len(code):
        raise JvmError(
            f"code at offset {offset}: opcode 0x{opcode:02X} is truncated "
            f"(need {offset + total} bytes, have {len(code)})"
        )
    return total


def _resolve_invoke(
    cp: _ConstantPool,
    cp_index: int,
    opcode: int,
    class_name: str,
    method_name: str,
    descriptor: str,
    code_offset: int,
) -> JvmInvoke:
    """Resolve one ``invoke*`` operand through the constant pool, or refuse it."""
    entry = cp.entry(cp_index)
    base: Dict[str, Any] = {
        "class_name": class_name,
        "method_name": method_name,
        "method_descriptor": descriptor,
        "code_offset": code_offset,
        "opcode": opcode,
        "cp_index": cp_index,
        "cp_tag": entry.tag,
    }
    if opcode == 0xBA:
        if entry.tag != _CP_INVOKE_DYNAMIC:
            raise JvmError(
                f"{class_name}.{method_name}{descriptor} at code offset "
                f"{code_offset}: invokedynamic operand #{cp_index} is a "
                f"{entry.describe()} entry, expected InvokeDynamic"
            )
        name, target_descriptor = cp.name_and_type(entry.b)
        return JvmInvoke(
            target_class="",
            target_name=name,
            target_descriptor=target_descriptor,
            resolved=False,
            reason="dynamic",
            **base,
        )
    if entry.tag not in (_CP_METHODREF, _CP_INTERFACE_METHODREF):
        raise JvmError(
            f"{class_name}.{method_name}{descriptor} at code offset "
            f"{code_offset}: {INVOKE_NAMES.get(opcode, hex(opcode))} operand "
            f"#{cp_index} is a {entry.describe()} entry, expected "
            "Methodref/InterfaceMethodref"
        )
    target_class, target_name, target_descriptor = cp.member_ref(cp_index)
    return JvmInvoke(
        target_class=target_class,
        target_name=target_name,
        target_descriptor=target_descriptor,
        resolved=True,
        reason="",
        **base,
    )


def _scan_invokes(
    code: bytes,
    cp: _ConstantPool,
    class_name: str,
    method_name: str,
    descriptor: str,
) -> List[JvmInvoke]:
    """Decode ``code`` and collect every real ``invoke*`` site in offset order."""
    label = f"code of {class_name}.{method_name}{descriptor}"
    cursor = _Reader(code, label)
    found: List[JvmInvoke] = []
    while cursor.remaining:
        offset = cursor.pos
        opcode = cursor.u1()
        length = _instruction_length(code, offset)
        if opcode in INVOKE_OPS:
            operand = code[offset + 1 : offset + length]
            cp_index = int.from_bytes(operand[:2], "big")
            found.append(
                _resolve_invoke(
                    cp, cp_index, opcode, class_name, method_name, descriptor, offset
                )
            )
        cursor.seek(offset + length)
    return found


# --------------------------------------------------------------------------- #
# Class file structure
# --------------------------------------------------------------------------- #


def _skip_attributes(reader: _Reader, cp: _ConstantPool, count: int, label: str) -> None:
    """Walk ``count`` attributes, skipping each body structurally (bounds-checked)."""
    for _ in range(count):
        name = cp.utf8(reader.u2())
        length = reader.u4()
        end = reader.pos + length
        if end > reader.size:
            raise JvmError(
                f"{label}: attribute {name!r} ({length} bytes) overruns the class "
                f"file at offset {reader.pos} (file is {reader.size} bytes)"
            )
        reader.seek(end)


def _read_fields(
    reader: _Reader, cp: _ConstantPool, class_name: str
) -> List[JvmField]:
    count = reader.u2()
    fields: List[JvmField] = []
    for _ in range(count):
        access_flags = reader.u2()
        name = cp.utf8(reader.u2())
        descriptor = cp.utf8(reader.u2())
        attribute_count = reader.u2()
        _skip_attributes(
            reader, cp, attribute_count, f"field {class_name}.{name}:{descriptor}"
        )
        fields.append(
            JvmField(
                class_name=class_name,
                name=name,
                descriptor=descriptor,
                access_flags=access_flags,
            )
        )
    return fields


def _read_methods(
    reader: _Reader, cp: _ConstantPool, class_name: str
) -> Tuple[List[JvmMethod], List[JvmInvoke]]:
    count = reader.u2()
    methods: List[JvmMethod] = []
    invokes: List[JvmInvoke] = []
    for _ in range(count):
        access_flags = reader.u2()
        name = cp.utf8(reader.u2())
        descriptor = cp.utf8(reader.u2())
        attribute_count = reader.u2()
        label = f"method {class_name}.{name}{descriptor}"
        code_len = 0
        code_seen = False
        for _ in range(attribute_count):
            attribute_name = cp.utf8(reader.u2())
            attribute_length = reader.u4()
            end = reader.pos + attribute_length
            if end > reader.size:
                raise JvmError(
                    f"{label}: attribute {attribute_name!r} ({attribute_length} "
                    f"bytes) overruns the class file at offset {reader.pos} "
                    f"(file is {reader.size} bytes)"
                )
            if attribute_name == "Code":
                if code_seen:
                    raise JvmError(f"{label}: duplicate Code attribute")
                code_seen = True
                if attribute_length < 8:
                    raise JvmError(
                        f"{label}: Code attribute is too short "
                        f"({attribute_length} bytes, need >= 8)"
                    )
                reader.u2()  # max_stack
                reader.u2()  # max_locals
                code_len = reader.u4()
                if reader.pos + code_len > end:
                    raise JvmError(
                        f"{label}: code_length {code_len} overruns its Code "
                        f"attribute ({attribute_length} bytes)"
                    )
                code = reader.read(code_len)
                invokes.extend(_scan_invokes(code, cp, class_name, name, descriptor))
            reader.seek(end)
        methods.append(
            JvmMethod(
                class_name=class_name,
                name=name,
                descriptor=descriptor,
                access_flags=access_flags,
                code_len=code_len,
            )
        )
    return methods, invokes


def parse_class_file(data: bytes) -> JvmClass:
    """Parse one class file's structure (JVMS 4.1-4.7) or raise :class:`JvmError`.

    Returns classes, methods (with ``Code`` ``code_length``), fields and the
    decoded ``invoke*`` sites.  Nothing is lifted, interpreted or executed, and
    ``data`` is never written anywhere: the caller owns it.  Any truncation, bad
    magic, unknown constant-pool tag, malformed attribute length or invalid
    instruction raises -- a partial :class:`JvmClass` is never returned.

    Trailing bytes after the class attributes are ignored (some obfuscators pad);
    everything the parser *did* consume is expressed in the returned object.
    """
    if isinstance(data, (bytearray, memoryview)):
        raw = bytes(data)
    elif isinstance(data, bytes):
        raw = data
    else:
        raise JvmError(
            f"parse_class_file expects bytes, got {type(data).__name__}"
        )
    if len(raw) < len(MAGIC) or raw[:4] != MAGIC:
        head = raw[:4].hex() if raw else "<empty>"
        raise JvmError(
            f"bad magic: expected CAFEBABE, got {head} "
            f"({len(raw)} byte(s) supplied)"
        )

    reader = _Reader(raw, "class file")
    reader.skip(4)
    minor_version = reader.u2()
    major_version = reader.u2()
    cp = _parse_constant_pool(reader)
    access_flags = reader.u2()
    this_index = reader.u2()
    if this_index == 0:
        raise JvmError("class file: this_class index is 0 (JVMS 4.1 forbids it)")
    name = cp.class_name(this_index)
    super_index = reader.u2()
    super_name = cp.class_name(super_index) if super_index else ""
    interface_count = reader.u2()
    interfaces: List[str] = []
    for _ in range(interface_count):
        interface_index = reader.u2()
        if interface_index == 0:
            raise JvmError(f"class {name}: interfaces[] holds the reserved index 0")
        interfaces.append(cp.class_name(interface_index))

    fields = _read_fields(reader, cp, name)
    methods, invokes = _read_methods(reader, cp, name)
    class_attribute_count = reader.u2()
    _skip_attributes(reader, cp, class_attribute_count, f"class {name}")

    return JvmClass(
        name=name,
        methods=methods,
        fields=fields,
        invokes=invokes,
        access_flags=access_flags,
        super_name=super_name,
        interfaces=interfaces,
        minor_version=minor_version,
        major_version=major_version,
    )


# --------------------------------------------------------------------------- #
# JAR
# --------------------------------------------------------------------------- #

#: Prefix skipped silently inside an archive (signed metadata, not code).
_META_INF = "META-INF/"


def _read_jar_bytes(source: Union[bytes, bytearray, memoryview, str, Path]) -> bytes:
    if isinstance(source, (bytearray, memoryview)):
        return bytes(source)
    if isinstance(source, bytes):
        return source
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise JvmError(f"JAR source not found: {path}")
        try:
            return path.read_bytes()
        except OSError as exc:
            raise JvmError(f"cannot read JAR {path}: {exc}") from exc
    raise JvmError(
        "parse_jar expects bytes or a filesystem path, got "
        f"{type(source).__name__}"
    )


def parse_jar(
    source: Union[bytes, bytearray, memoryview, str, Path]
) -> Tuple[List[JvmClass], Dict[str, Any]]:
    """Parse every ``*.class`` entry of a JAR, deterministically.

    Entries are visited in ``filename`` sorted order (Trap 10), so two runs over
    the same archive -- or two archives with the same members -- yield the same
    class order and therefore the same synthetic ordinals.  ``META-INF/`` entries,
    directory entries and non-``.class`` members are skipped silently and
    *counted* in the returned diagnostics (``{entries, skipped, classes,
    format}``).  A ``.class`` entry that cannot be parsed raises
    :class:`JvmError` naming the entry: a JAR is never partially ingested.
    """
    data = _read_jar_bytes(source)
    classes: List[JvmClass] = []
    entries = 0
    skipped = 0
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            infos = sorted(archive.infolist(), key=lambda info: info.filename)
            for info in infos:
                entry_name = info.filename
                entries += 1
                if (
                    info.is_dir()
                    or entry_name.startswith(_META_INF)
                    or not entry_name.endswith(".class")
                ):
                    skipped += 1
                    continue
                try:
                    blob = archive.read(info)
                except (zipfile.BadZipFile, RuntimeError, OSError, ValueError, KeyError, EOFError, struct.error) as exc:
                    raise JvmError(f"JAR entry {entry_name}: unreadable ({exc})") from exc
                try:
                    classes.append(parse_class_file(blob))
                except JvmError as exc:
                    raise JvmError(f"JAR entry {entry_name}: {exc}") from exc
    except (zipfile.BadZipFile, ValueError, KeyError, EOFError, struct.error) as exc:
        raise JvmError(f"not a readable JAR/ZIP archive ({len(data)} bytes): {exc}") from exc
    return classes, {
        "format": "jar",
        "entries": entries,
        "skipped": skipped,
        "classes": len(classes),
    }


# --------------------------------------------------------------------------- #
# Deterministic synthetic ordinals
# --------------------------------------------------------------------------- #


def _sorted_classes(classes: Iterable[JvmClass]) -> List[JvmClass]:
    """Class order of the synthetic address space: by internal name, stable."""
    return sorted(classes, key=lambda cls: cls.name)


def _sorted_methods(methods: Iterable[JvmMethod]) -> List[JvmMethod]:
    """Method order inside a class: by ``(name, descriptor)``, total and stable."""
    return sorted(
        methods,
        key=lambda method: (
            method.name,
            method.descriptor,
            method.access_flags,
            method.code_len,
        ),
    )


def _sorted_fields(fields: Iterable[JvmField]) -> List[JvmField]:
    return sorted(
        fields,
        key=lambda field: (field.name, field.descriptor, field.access_flags),
    )


def _ordered_methods(
    classes: Iterable[JvmClass],
) -> Iterator[Tuple[int, JvmClass, JvmMethod]]:
    """Yield ``(ordinal, class, method)`` from 1 in sorted class/method order.

    This is the single definition of the synthetic address space: both
    :func:`ingest` and :func:`method_addr_index` consume it, so the ordinals a
    caller computes cannot drift from the ordinals that were written.
    """
    ordinal = 0
    for cls in _sorted_classes(classes):
        for method in _sorted_methods(cls.methods):
            ordinal += 1
            yield ordinal, cls, method


def method_addr_index(
    classes: Iterable[JvmClass],
) -> Dict[Tuple[str, str, str], int]:
    """``{(class, method, descriptor): ordinal}`` for the synthetic address space.

    Ordinals are assigned from 1 over methods in ``(class, method)`` sorted
    order, so the mapping is a pure function of the parsed classes -- re-ingesting
    the same archive reproduces the same addresses.  A duplicated identical
    ``(class, name, descriptor)`` keeps its first ordinal (duplicate method
    definitions are not legal Java, and inventing a second address would be
    fabrication).
    """
    index: Dict[Tuple[str, str, str], int] = {}
    for ordinal, cls, method in _ordered_methods(classes):
        index.setdefault((cls.name, method.name, method.descriptor), ordinal)
    return index


def class_type_id(internal_name: str) -> str:
    """``type_id`` of a JVM class row: ``jvm:<internal/name>``."""
    return f"{JVM_TYPE_PREFIX}{internal_name}"


def field_type_id(internal_name: str, field_name: str, descriptor: str) -> str:
    """``type_id`` of a JVM field row: ``jvm:<internal/name>#<field>:<descriptor>``."""
    return (
        f"{JVM_TYPE_PREFIX}{internal_name}{JVM_FIELD_PREFIX}"
        f"{field_name}:{descriptor}"
    )


def _display_name(class_name: str, member: Optional[str] = None) -> str:
    """Source-shaped name (``a.b.C`` / ``a.b.C.m``) for the ``name`` column."""
    dotted = class_name.replace("/", ".")
    return dotted if member is None else f"{dotted}.{member}"
