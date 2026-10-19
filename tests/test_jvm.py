"""J1 JVM structural ingest tests: struct-built class-file / JAR fixtures, no JDK.

Binding contract: ``docs/units/UNIT-J1-JVM-PLAN.md`` section 1 (Chunk 1/2b) and
section 2 (interface specification); milestone J1 of ``docs/NEXT-TRACK.md``
("class-file/JAR parsing; methods/fields/``invoke*`` callgraph into
``funcs``/``edges``.  No lifting, no SSA").

Every fixture below is a *synthetic* class file assembled with :mod:`struct` at
explicit JVMS 4.1-4.7 offsets and archived with :mod:`zipfile` in memory: no JDK,
no ``javac``, no real artifact is read and no address is invented.  A recovered
row is only ever what was placed in the byte buffer -- the epistemic rule the
module is built on.

Covered here (plan section 3, Chunk 2b):

* minimal class parse -- internal/dotted name, methods with ``Code``
  ``code_length``, fields, versions, ``super``;
* the five invoke opcodes (``0xB6``-``0xBA``) decoded from the instruction
  stream in offset order, and the CP tag each one resolved through;
* callgraph edges (``kind='call'``) for every ``invoke*`` whose target is one of
  the ingested methods, and ``unresolved`` counting for every other site
  (external/JDK callee, ``invokedynamic`` bootstrap) with nothing fabricated;
* multi-class JAR determinism -- entries visited in sorted order,
  ``META-INF/``/directory/non-``.class`` members skipped and counted, and a
  stable synthetic ordinal map that does not depend on ZIP member order;
* refusal: bad magic, every truncation length, a unusable ``.class`` member of a
  JAR (named in the error), all before a single row is written;
* ``types`` rows: one ``jvm:<class>`` ``kind='class'`` row with method/field
  layout, one ``jvm:<class>#<field>:<descriptor>`` ``kind='field'`` row per
  field;
* replace-per-sha / per-namespace semantics and re-ingest ordinal stability;
* Trap 7 (every SQLite handle closed and the tmp db file removable) and Trap 11
  (no ``faulthandler`` anywhere; ``pyproject.toml`` keeps the pin).
"""

from __future__ import annotations

import io
import re
import sqlite3
import struct
import zipfile
from pathlib import Path

import pytest

from toothpick import jvm, jvm_parse
from toothpick.jvm import (
    CALLING_CONVENTION,
    JVM_FIELD_PREFIX,
    JVM_TYPE_PREFIX,
    MAGIC,
    JvmClass,
    JvmError,
    ingest,
    method_addr_index,
    parse_class_file,
    parse_jar,
)
from toothpick.storage import (
    Database,
    EdgeKind,
    EdgeRecord,
    FunctionRecord,
    TypeRecord,
)

SHA = "a1" * 32
SHA_OTHER = "b2" * 32

# --- access flags / opcodes / CP tags (JVMS 4.5, 6.5, 4.4) ------------------ #

ACC_PUBLIC = 0x0001
ACC_STATIC = 0x0008
ACC_INTERFACE = 0x0200
ACC_ABSTRACT = 0x0400

OP_INVOKEVIRTUAL = 0xB6
OP_INVOKESPECIAL = 0xB7
OP_INVOKESTATIC = 0xB8
OP_INVOKEINTERFACE = 0xB9
OP_INVOKEDYNAMIC = 0xBA
OP_RETURN = 0xB1

CP_UTF8 = 1
CP_INTEGER = 3
CP_FLOAT = 4
CP_LONG = 5
CP_DOUBLE = 6
CP_CLASS = 7
CP_STRING = 8
CP_FIELDREF = 9
CP_METHODREF = 10
CP_INTERFACE_METHODREF = 11
CP_NAME_AND_TYPE = 12
CP_METHOD_HANDLE = 15
CP_METHOD_TYPE = 16
CP_DYNAMIC = 17
CP_INVOKE_DYNAMIC = 18
CP_MODULE = 19
CP_PACKAGE = 20

MAJOR_VERSION = 52  # Java 8 class-file version


# --------------------------------------------------------------------------- #
# struct-built fixtures (no JDK)
# --------------------------------------------------------------------------- #


class _ConstantPool:
    """A slot-accurate constant pool: index ``i`` is ``_slots[i - 1]``.

    ``long``/``double`` occupy two slots (JVMS 4.4.5) and the second is emitted
    as an empty continuation in :meth:`to_bytes`, exactly as the reader must
    stride over it.
    """

    def __init__(self) -> None:
        self._slots: list[bytes | None] = []
        self._cache: dict[tuple, int] = {}

    def _add(self, entry: bytes, key: tuple) -> int:
        self._slots.append(entry)
        index = len(self._slots)
        self._cache[key] = index
        return index

    def utf8(self, text: str) -> int:
        key = ("utf8", text)
        if key in self._cache:
            return self._cache[key]
        payload = text.encode("utf-8")
        return self._add(bytes([CP_UTF8]) + struct.pack(">H", len(payload)) + payload, key)

    def class_(self, name: str) -> int:
        key = ("class", name)
        if key in self._cache:
            return self._cache[key]
        return self._add(struct.pack(">BH", CP_CLASS, self.utf8(name)), key)

    def integer(self, value: int) -> int:
        key = ("int", value)
        if key in self._cache:
            return self._cache[key]
        return self._add(struct.pack(">Bi", CP_INTEGER, value), key)

    def float_(self, value: float) -> int:
        key = ("float", value)
        if key in self._cache:
            return self._cache[key]
        return self._add(struct.pack(">Bf", CP_FLOAT, value), key)

    def long_(self, value: int) -> int:
        key = ("long", value)
        if key in self._cache:
            return self._cache[key]
        index = self._add(struct.pack(">Bq", CP_LONG, value), key)
        self._slots.append(None)  # JVMS 4.4.5: the second slot is unusable
        return index

    def double_(self, value: float) -> int:
        key = ("double", value)
        if key in self._cache:
            return self._cache[key]
        index = self._add(struct.pack(">Bd", CP_DOUBLE, value), key)
        self._slots.append(None)
        return index

    def string(self, text: str) -> int:
        key = ("string", text)
        if key in self._cache:
            return self._cache[key]
        return self._add(struct.pack(">BH", CP_STRING, self.utf8(text)), key)

    def name_and_type(self, name: str, descriptor: str) -> int:
        key = ("nat", name, descriptor)
        if key in self._cache:
            return self._cache[key]
        return self._add(
            struct.pack(">BHH", CP_NAME_AND_TYPE, self.utf8(name), self.utf8(descriptor)),
            key,
        )

    def fieldref(self, class_name: str, name: str, descriptor: str) -> int:
        key = ("fieldref", class_name, name, descriptor)
        if key in self._cache:
            return self._cache[key]
        return self._add(
            struct.pack(
                ">BHH", CP_FIELDREF, self.class_(class_name),
                self.name_and_type(name, descriptor),
            ),
            key,
        )

    def methodref(self, class_name: str, name: str, descriptor: str) -> int:
        key = ("methodref", class_name, name, descriptor)
        if key in self._cache:
            return self._cache[key]
        return self._add(
            struct.pack(
                ">BHH", CP_METHODREF, self.class_(class_name),
                self.name_and_type(name, descriptor),
            ),
            key,
        )

    def interface_methodref(self, class_name: str, name: str, descriptor: str) -> int:
        key = ("imethodref", class_name, name, descriptor)
        if key in self._cache:
            return self._cache[key]
        return self._add(
            struct.pack(
                ">BHH", CP_INTERFACE_METHODREF, self.class_(class_name),
                self.name_and_type(name, descriptor),
            ),
            key,
        )

    def method_handle(self, kind: int, class_name: str, name: str, descriptor: str) -> int:
        key = ("mhandle", kind, class_name, name, descriptor)
        if key in self._cache:
            return self._cache[key]
        return self._add(
            struct.pack(
                ">BBH", CP_METHOD_HANDLE, kind,
                self.methodref(class_name, name, descriptor),
            ),
            key,
        )

    def method_type(self, descriptor: str) -> int:
        key = ("mtype", descriptor)
        if key in self._cache:
            return self._cache[key]
        return self._add(struct.pack(">BH", CP_METHOD_TYPE, self.utf8(descriptor)), key)

    def dynamic(self, name: str, descriptor: str, bootstrap: int = 0) -> int:
        key = ("dynamic", name, descriptor, bootstrap)
        if key in self._cache:
            return self._cache[key]
        return self._add(
            struct.pack(
                ">BHH", CP_DYNAMIC, bootstrap, self.name_and_type(name, descriptor)
            ),
            key,
        )

    def invoke_dynamic(self, name: str, descriptor: str, bootstrap: int = 0) -> int:
        key = ("indy", name, descriptor, bootstrap)
        if key in self._cache:
            return self._cache[key]
        return self._add(
            struct.pack(
                ">BHH", CP_INVOKE_DYNAMIC, bootstrap, self.name_and_type(name, descriptor)
            ),
            key,
        )

    def module(self, name: str) -> int:
        key = ("module", name)
        if key in self._cache:
            return self._cache[key]
        return self._add(struct.pack(">BH", CP_MODULE, self.utf8(name)), key)

    def package(self, name: str) -> int:
        key = ("package", name)
        if key in self._cache:
            return self._cache[key]
        return self._add(struct.pack(">BH", CP_PACKAGE, self.utf8(name)), key)

    def to_bytes(self) -> bytes:
        count = len(self._slots) + 1
        return struct.pack(">H", count) + b"".join(slot or b"" for slot in self._slots)


class _ClassBuilder:
    """Assembles one class file: pool, this/super, interfaces, fields, methods."""

    def __init__(
        self,
        name: str,
        *,
        super_name: str = "java/lang/Object",
        access: int = ACC_PUBLIC,
        major: int = MAJOR_VERSION,
        minor: int = 0,
    ) -> None:
        self.cp = _ConstantPool()
        self.name = name
        self.super_name = super_name
        self.access = access
        self.major = major
        self.minor = minor
        self.interfaces: list[str] = []
        self.fields: list[tuple[int, str, str]] = []
        self.methods: list[tuple[int, str, str, bytes, int, int]] = []

    def add_interface(self, name: str) -> "_ClassBuilder":
        self.interfaces.append(name)
        return self

    def add_field(
        self, name: str, descriptor: str, access: int = ACC_PUBLIC
    ) -> "_ClassBuilder":
        self.fields.append((access, name, descriptor))
        return self

    def add_method(
        self,
        name: str,
        descriptor: str,
        *,
        code: bytes = b"",
        access: int = ACC_PUBLIC,
        max_stack: int = 4,
        max_locals: int = 4,
    ) -> "_ClassBuilder":
        self.methods.append((access, name, descriptor, code, max_stack, max_locals))
        return self

    def to_bytes(self) -> bytes:
        body = bytearray()
        super_index = self.cp.class_(self.super_name) if self.super_name else 0
        body += struct.pack(">HHH", self.access, self.cp.class_(self.name), super_index)
        body += struct.pack(">H", len(self.interfaces))
        for iface in self.interfaces:
            body += struct.pack(">H", self.cp.class_(iface))
        body += struct.pack(">H", len(self.fields))
        for access, name, descriptor in self.fields:
            body += struct.pack(
                ">HHHH", access, self.cp.utf8(name), self.cp.utf8(descriptor), 0
            )
        body += struct.pack(">H", len(self.methods))
        for access, name, descriptor, code, max_stack, max_locals in self.methods:
            attribute_count = 1 if code else 0
            body += struct.pack(
                ">HHHH",
                access,
                self.cp.utf8(name),
                self.cp.utf8(descriptor),
                attribute_count,
            )
            if code:
                payload = (
                    struct.pack(">HHI", max_stack, max_locals, len(code))
                    + code
                    + struct.pack(">HH", 0, 0)  # no exception table, no sub-attributes
                )
                body += struct.pack(">HI", self.cp.utf8("Code"), len(payload)) + payload
        body += struct.pack(">H", 0)  # class attributes: none
        header = (
            MAGIC
            + struct.pack(">HH", self.minor, self.major)
            + self.cp.to_bytes()
        )
        return header + bytes(body)


def _invoke(opcode: int, cp_index: int) -> bytes:
    """One ``invoke*`` instruction: opcode + 2 CP bytes + 2 operand-tail bytes."""
    tail = b"\x00\x00" if opcode in (OP_INVOKEINTERFACE, OP_INVOKEDYNAMIC) else b""
    return bytes([opcode]) + struct.pack(">H", cp_index) + tail


def _plain_class(
    name: str,
    *,
    super_name: str = "java/lang/Object",
    methods: tuple[tuple[str, str, bytes], ...] = (("run", "()V", bytes([OP_RETURN])),),
    fields: tuple[tuple[str, str], ...] = (),
    access: int = ACC_PUBLIC,
) -> bytes:
    builder = _ClassBuilder(name, super_name=super_name, access=access)
    for field_name, field_descriptor in fields:
        builder.add_field(field_name, field_descriptor)
    for method_name, method_descriptor, code in methods:
        builder.add_method(method_name, method_descriptor, code=code)
    return builder.to_bytes()


def _main_class_bytes() -> bytes:
    """``app/Main`` exercising all five invoke opcodes plus every CP tag class.

    Methods: ``<init>``, ``helper``, ``init0``, ``main``, ``render``.  The pool
    deliberately carries ``Integer``/``Float``/``Long``/``Double`` (two slots),
    ``String``, ``Fieldref``, ``NameAndType``, ``MethodHandle``, ``MethodType``,
    ``Dynamic``, ``Module`` and ``Package`` entries even though the recorded
    structure never reads them: the reader must stride over all of them to reach
    ``this_class``, which is what the name assertion below proves.
    """
    builder = _ClassBuilder("app/Main", super_name="app/Base")
    cp = builder.cp

    cp.integer(7)
    cp.float_(1.5)
    cp.long_(1 << 40)
    cp.double_(2.5)
    cp.string("hello")
    cp.fieldref("app/Main", "count", "I")
    cp.name_and_type("helper", "()V")  # duplicate: must be re-used, not re-added
    cp.method_handle(6, "app/Main", "helper", "()V")
    cp.method_type("()V")
    cp.dynamic("callSite", "()V", bootstrap=0)
    cp.module("app.module")
    cp.package("app/pkg")

    init0 = cp.methodref("app/Main", "init0", "()V")
    obj_init = cp.methodref("java/lang/Object", "<init>", "()V")
    helper = cp.methodref("app/Main", "helper", "()V")
    render = cp.methodref("app/Main", "render", "()V")
    list_size = cp.interface_methodref("java/util/List", "size", "()I")
    indy = cp.invoke_dynamic("bootstrap$0", "()V", bootstrap=0)

    builder.add_field("count", "I", access=ACC_PUBLIC | ACC_STATIC)
    builder.add_field("label", "Ljava/lang/String;")
    builder.add_method(
        "<init>",
        "()V",
        code=_invoke(OP_INVOKESPECIAL, init0)
        + _invoke(OP_INVOKESPECIAL, obj_init)
        + bytes([OP_RETURN]),
    )
    builder.add_method("init0", "()V", code=bytes([OP_RETURN]))
    builder.add_method("helper", "()V", code=bytes([OP_RETURN]))
    builder.add_method("render", "()V", code=bytes([OP_RETURN]))
    builder.add_method(
        "main",
        "([Ljava/lang/String;)V",
        access=ACC_PUBLIC | ACC_STATIC,
        code=_invoke(OP_INVOKESTATIC, helper)
        + _invoke(OP_INVOKEVIRTUAL, render)
        + _invoke(OP_INVOKEINTERFACE, list_size)
        + _invoke(OP_INVOKEDYNAMIC, indy)
        + bytes([OP_RETURN]),
    )
    return builder.to_bytes()


def _derived_class_bytes() -> bytes:
    """``app/Derived``: ``run()V`` invokespecial-calls ``app/Base.run()V``."""
    builder = _ClassBuilder("app/Derived", super_name="app/Base")
    base_run = builder.cp.methodref("app/Base", "run", "()V")
    builder.add_field("id", "I")
    builder.add_method(
        "run", "()V", code=_invoke(OP_INVOKESPECIAL, base_run) + bytes([OP_RETURN])
    )
    builder.add_method("extra", "()V", code=bytes([OP_RETURN]))
    return builder.to_bytes()


def _jar_bytes(items: tuple[tuple[str, bytes], ...]) -> bytes:
    """Archive ``items`` in the given order (insertion order is the ZIP order)."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, payload in items:
            archive.writestr(name, payload)
    return buffer.getvalue()


def _multi_class_jar(*, reversed_order: bool = False) -> bytes:
    items: list[tuple[str, bytes]] = [
        ("app/Base.class", _plain_class("app/Base", fields=(("ttl", "I"),))),
        ("app/Derived.class", _derived_class_bytes()),
        ("z/Z.class", _plain_class("z/Z", methods=(("zeta", "()V", bytes([OP_RETURN])),))),
        ("META-INF/MANIFEST.MF", b"Manifest-Version: 1.0\n"),
        ("README.txt", b"not bytecode\n"),
        ("sub/", b""),
    ]
    if reversed_order:
        items.reverse()
    return _jar_bytes(tuple(items))


def _interface_jar() -> bytes:
    iface = (
        _ClassBuilder("app/Iface", access=ACC_PUBLIC | ACC_INTERFACE | ACC_ABSTRACT)
        .add_method("run", "()V", access=ACC_PUBLIC | ACC_ABSTRACT)
        .to_bytes()
    )
    builder = _ClassBuilder("app/Impl")
    iface_run = builder.cp.interface_methodref("app/Iface", "run", "()V")
    impl = (
        builder.add_interface("app/Iface")
        .add_method(
            "run", "()V", code=_invoke(OP_INVOKEINTERFACE, iface_run) + bytes([OP_RETURN])
        )
        .to_bytes()
    )
    return _jar_bytes((("app/Iface.class", iface), ("app/Impl.class", impl)))


# --------------------------------------------------------------------------- #
# 1. minimal class parse
# --------------------------------------------------------------------------- #


def test_parse_minimal_class_file_recovers_name_methods_and_fields() -> None:
    data = _main_class_bytes()
    assert data[:4] == MAGIC

    cls = parse_class_file(data)
    assert cls.name == "app/Main"
    assert cls.dotted_name == "app.Main"
    assert cls.super_name == "app/Base"
    assert cls.interfaces == []
    assert (cls.minor_version, cls.major_version) == (0, MAJOR_VERSION)
    assert cls.access_flags == ACC_PUBLIC

    assert [(m.class_name, m.name, m.descriptor) for m in cls.methods] == [
        ("app/Main", "<init>", "()V"),
        ("app/Main", "init0", "()V"),
        ("app/Main", "helper", "()V"),
        ("app/Main", "render", "()V"),
        ("app/Main", "main", "([Ljava/lang/String;)V"),
    ]
    by_name = {m.name: m for m in cls.methods}
    # code_len is the Code attribute's code_length, nothing else.
    assert by_name["helper"].code_len == 1
    assert by_name["main"].code_len == 3 + 3 + 5 + 5 + 1
    assert by_name["main"].access_flags == ACC_PUBLIC | ACC_STATIC
    assert by_name["<init>"].code_len == 3 + 3 + 1

    assert [(f.class_name, f.name, f.descriptor, f.access_flags) for f in cls.fields] == [
        ("app/Main", "count", "I", ACC_PUBLIC | ACC_STATIC),
        ("app/Main", "label", "Ljava/lang/String;", ACC_PUBLIC),
    ]

    # Parsing the same bytes twice yields the same structure (no hidden state).
    assert parse_class_file(data) == cls
    # The plan's positional constructor contract still holds.
    assert JvmClass("a/B", [], []).dotted_name == "a.B"


# --------------------------------------------------------------------------- #
# 2. invoke* decoding (all five opcodes, offset order, CP tag per site)
# --------------------------------------------------------------------------- #


def test_invoke_sites_decode_all_five_opcodes_in_offset_order() -> None:
    cls = parse_class_file(_main_class_bytes())
    assert [(s.opcode, s.target_class, s.target_name, s.resolved) for s in cls.invokes] == [
        (OP_INVOKESPECIAL, "app/Main", "init0", True),
        (OP_INVOKESPECIAL, "java/lang/Object", "<init>", True),
        (OP_INVOKESTATIC, "app/Main", "helper", True),
        (OP_INVOKEVIRTUAL, "app/Main", "render", True),
        (OP_INVOKEINTERFACE, "java/util/List", "size", True),
        (OP_INVOKEDYNAMIC, "", "bootstrap$0", False),
    ]
    assert [s.code_offset for s in cls.invokes] == [0, 3, 0, 3, 6, 11]
    assert [s.class_name for s in cls.invokes] == [
        "app/Main",
        "app/Main",
        "app/Main",
        "app/Main",
        "app/Main",
        "app/Main",
    ]
    assert [s.cp_tag for s in cls.invokes] == [
        CP_METHODREF,
        CP_METHODREF,
        CP_METHODREF,
        CP_METHODREF,
        CP_INTERFACE_METHODREF,
        CP_INVOKE_DYNAMIC,
    ]
    assert cls.invokes[3].target_descriptor == "()V"
    assert cls.invokes[4].target_descriptor == "()I"
    assert cls.invokes[5].reason == "dynamic"
    assert cls.invokes[5].target_descriptor == "()V"


# --------------------------------------------------------------------------- #
# 3. ingest: funcs rows / synthetic ordinals / get_function
# --------------------------------------------------------------------------- #


def test_ingest_class_bytes_writes_funcs_and_get_function_finds_method(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "jvm-funcs.db"
    data = _main_class_bytes()
    db = Database(db_path)
    try:
        result = ingest(db, SHA, data)
        assert result["binary_sha256"] == SHA
        assert result["classes"] == 1
        assert result["methods"] == 5
        assert result["fields"] == 2
        assert result["diagnostics"] == {
            "format": "class",
            "entries": 1,
            "skipped": 0,
            "classes": 1,
        }

        rows = {r.func_addr: r for r in db.get_functions(SHA)}
        assert sorted(rows) == [1, 2, 3, 4, 5]
        for ordinal, method in zip(
            [1, 2, 3, 4, 5], ["<init>", "helper", "init0", "main", "render"]
        ):
            assert rows[ordinal].name == f"app.Main.{method}"
            assert rows[ordinal].calling_convention == CALLING_CONVENTION == "jvm"
        assert rows[4].prototype == "([Ljava/lang/String;)V"
        assert rows[4].size == 17
        assert rows[4].ir_pcode is None and rows[4].decompiled_c is None

        # J1 acceptance: a method is found by ordinal + name.
        found = db.get_function(SHA, 4)
        assert isinstance(found, FunctionRecord)
        assert found.name == "app.Main.main"

        # The written ordinals are exactly the documented synthetic map.
        assert method_addr_index([parse_class_file(data)]) == {
            ("app/Main", "<init>", "()V"): 1,
            ("app/Main", "helper", "()V"): 2,
            ("app/Main", "init0", "()V"): 3,
            ("app/Main", "main", "([Ljava/lang/String;)V"): 4,
            ("app/Main", "render", "()V"): 5,
        }
    finally:
        db.close()
    db_path.unlink()  # Trap 7: no handle outlives the test


# --------------------------------------------------------------------------- #
# 4. callgraph edges + unresolved counting
# --------------------------------------------------------------------------- #


def test_ingest_writes_call_edges_for_resolved_invokes_only(tmp_path: Path) -> None:
    db_path = tmp_path / "jvm-edges.db"
    db = Database(db_path)
    try:
        result = ingest(db, SHA, _main_class_bytes())
        assert result["invokes"] == 6
        assert result["unresolved"] == 3
        assert result["edges"] == 3

        edges = db.get_edges(SHA, kind=EdgeKind.CALL)
        assert [(e.src_addr, e.dst_addr) for e in edges] == [(1, 3), (4, 2), (4, 5)]
        assert {e.kind for e in edges} == {EdgeKind.CALL}
        assert db.count_edges(SHA) == 3
        assert db.get_callees(SHA, 4) == [2, 5]
        assert db.get_callers(SHA, 3) == [1]
        assert db.get_callees(SHA, 1) == [3]
        assert db.get_callees(SHA, 2) == []
    finally:
        db.close()
    db_path.unlink()


def test_unresolved_invokes_are_counted_and_never_fabricated(tmp_path: Path) -> None:
    cls = parse_class_file(_main_class_bytes())
    resolved_in_cp = [s for s in cls.invokes if s.resolved]
    assert len(resolved_in_cp) == 5  # CP-resolved ...
    dynamic = [s for s in cls.invokes if s.opcode == OP_INVOKEDYNAMIC]
    assert len(dynamic) == 1 and dynamic[0].resolved is False

    db_path = tmp_path / "jvm-unresolved.db"
    db = Database(db_path)
    try:
        result = ingest(db, SHA, cls)
        # ... but only three targets are methods of this ingest: <init> ->
        # java/lang/Object.<init>, main -> java/util/List.size and the
        # invokedynamic bootstrap have no row and no edge.
        assert result["unresolved"] == 3
        assert result["edges"] == 3
        assert len(db.get_functions(SHA)) == 5
        edges = db.get_edges(SHA)
        assert len(edges) == 3
        assert 0 not in [e.dst_addr for e in edges]  # no placeholder ordinal

        layout = db.get_type("jvm:app/Main", SHA).layout
        stats = {m["name"]: m for m in layout["methods"]}
        assert (stats["<init>"]["invokes"], stats["<init>"]["unresolved"]) == (2, 1)
        assert (stats["main"]["invokes"], stats["main"]["unresolved"]) == (4, 2)
        assert (stats["helper"]["invokes"], stats["helper"]["unresolved"]) == (0, 0)
        assert layout["invoke_targets"] == [
            "app/Main.helper:()V",
            "app/Main.init0:()V",
            "app/Main.render:()V",
        ]
    finally:
        db.close()
    db_path.unlink()


def test_invokeinterface_resolves_when_target_class_is_ingested(tmp_path: Path) -> None:
    iface, impl = parse_jar(_interface_jar())[0]
    assert (iface.name, impl.name) == ("app/Iface", "app/Impl")
    assert impl.interfaces == ["app/Iface"]
    site = parse_class_file(
        (
            _ClassBuilder("app/Impl")
            .add_interface("app/Iface")
            .add_method("run", "()V", code=b"\xb1")
            .to_bytes()
        )
    )
    del site  # the interface-method ref is asserted on the parsed JAR below
    assert [s.opcode for s in impl.invokes] == [OP_INVOKEINTERFACE]
    assert impl.invokes[0].cp_tag == CP_INTERFACE_METHODREF
    assert impl.invokes[0].resolved is True

    db_path = tmp_path / "jvm-iface.db"
    db = Database(db_path)
    try:
        result = ingest(db, SHA, _interface_jar())
        assert (result["classes"], result["methods"], result["edges"]) == (2, 2, 1)
        assert result["unresolved"] == 0
        # Ordinals: app/Iface.run -> 1, app/Impl.run -> 2 (class then method sort).
        assert method_addr_index(parse_jar(_interface_jar())[0]) == {
            ("app/Iface", "run", "()V"): 1,
            ("app/Impl", "run", "()V"): 2,
        }
        assert db.get_callees(SHA, 2) == [1]
        assert db.get_function(SHA, 1).name == "app.Iface.run"
    finally:
        db.close()
    db_path.unlink()


# --------------------------------------------------------------------------- #
# 5. types rows: class layout + field rows
# --------------------------------------------------------------------------- #


def test_class_rows_in_types_carry_method_and_field_layout(tmp_path: Path) -> None:
    db_path = tmp_path / "jvm-types.db"
    db = Database(db_path)
    try:
        result = ingest(db, SHA, _main_class_bytes())
        assert result["type_ids"] == [
            "jvm:app/Main",
            "jvm:app/Main#count:I",
            "jvm:app/Main#label:Ljava/lang/String;",
        ]
        record = db.get_type("jvm:app/Main", SHA)
        assert isinstance(record, TypeRecord)
        assert (record.name, record.kind, record.size_bytes) == ("app.Main", "class", 0)
        assert record.binary_sha256 == SHA
        assert record.lattice_rank == 3

        layout = record.layout
        assert layout["provenance"] == "jvm-class"
        assert layout["internal_name"] == "app/Main"
        assert layout["synthetic_addr"] is True
        assert layout["super"] == "app/Base"
        assert layout["major_version"] == MAJOR_VERSION
        assert layout["methods"] == [
            {
                "name": "<init>",
                "descriptor": "()V",
                "access_flags": ACC_PUBLIC,
                "code_len": 7,
                "func_addr": 1,
                "invokes": 2,
                "unresolved": 1,
            },
            {
                "name": "helper",
                "descriptor": "()V",
                "access_flags": ACC_PUBLIC,
                "code_len": 1,
                "func_addr": 2,
                "invokes": 0,
                "unresolved": 0,
            },
            {
                "name": "init0",
                "descriptor": "()V",
                "access_flags": ACC_PUBLIC,
                "code_len": 1,
                "func_addr": 3,
                "invokes": 0,
                "unresolved": 0,
            },
            {
                "name": "main",
                "descriptor": "([Ljava/lang/String;)V",
                "access_flags": ACC_PUBLIC | ACC_STATIC,
                "code_len": 17,
                "func_addr": 4,
                "invokes": 4,
                "unresolved": 2,
            },
            {
                "name": "render",
                "descriptor": "()V",
                "access_flags": ACC_PUBLIC,
                "code_len": 1,
                "func_addr": 5,
                "invokes": 0,
                "unresolved": 0,
            },
        ]
        assert layout["fields"] == [
            {"name": "count", "descriptor": "I", "access_flags": ACC_PUBLIC | ACC_STATIC},
            {
                "name": "label",
                "descriptor": "Ljava/lang/String;",
                "access_flags": ACC_PUBLIC,
            },
        ]
    finally:
        db.close()
    db_path.unlink()


def test_field_rows_land_in_types_with_field_kind(tmp_path: Path) -> None:
    assert JVM_TYPE_PREFIX == "jvm:" and JVM_FIELD_PREFIX == "#"

    db_path = tmp_path / "jvm-fields.db"
    db = Database(db_path)
    try:
        ingest(db, SHA, _main_class_bytes())
        count = db.get_type("jvm:app/Main#count:I", SHA)
        assert (count.name, count.kind) == ("app.Main.count", "field")
        assert count.layout == {
            "provenance": "jvm-field",
            "class": "app/Main",
            "internal_name": "app/Main",
            "name": "count",
            "descriptor": "I",
            "access_flags": ACC_PUBLIC | ACC_STATIC,
        }
        label = db.get_type("jvm:app/Main#label:Ljava/lang/String;", SHA)
        assert (label.name, label.kind) == ("app.Main.label", "field")
        assert label.layout["descriptor"] == "Ljava/lang/String;"

        kinds = {r.type_id: r.kind for r in db.get_types(SHA)}
        assert kinds == {
            "jvm:app/Main": "class",
            "jvm:app/Main#count:I": "field",
            "jvm:app/Main#label:Ljava/lang/String;": "field",
        }
    finally:
        db.close()
    db_path.unlink()


# --------------------------------------------------------------------------- #
# 6. refusal: bad magic, truncation, unusable JAR member
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"\x00",
        b"\x01\x02\x03\x04",
        b"NOTMAGIC" + b"\x00" * 12,
        b"\xca\xfe\xba\xbf" + _main_class_bytes()[4:],
    ],
)
def test_bad_magic_and_other_garbage_are_refused(payload: bytes) -> None:
    with pytest.raises(JvmError) as excinfo:
        parse_class_file(payload)
    message = str(excinfo.value)
    assert "bad magic" in message
    assert "CAFEBABE" in message


def test_truncated_class_file_is_refused_at_every_length() -> None:
    data = _main_class_bytes()
    assert len(data) > 100
    for length in range(1, len(data)):
        prefix = data[:length]
        with pytest.raises(JvmError) as excinfo:
            parse_class_file(prefix)
        message = str(excinfo.value)
        assert (
            "truncat" in message
            or "bad magic" in message
            or "overruns" in message
        ), (length, message)

    # A bare magic word parses no further: the refusal is about bytes, not size.
    with pytest.raises(JvmError) as excinfo:
        parse_class_file(MAGIC)
    assert "truncated" in str(excinfo.value)


def test_ingest_refuses_bad_input_without_writing_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "jvm-refuse.db"
    db = Database(db_path)
    try:
        for bad_sha in ("", "   ", 17, None, True):
            with pytest.raises(JvmError):
                ingest(db, bad_sha, _main_class_bytes())
        for bad_source in (b"", b"NOTMAGIC", b"PK\x03\x04garbage", 3.5, object()):
            with pytest.raises(JvmError):
                ingest(db, SHA, bad_source)
        with pytest.raises(JvmError) as excinfo:
            ingest(db, SHA, tmp_path / "missing.jar")
        assert "not found" in str(excinfo.value)
        # Nothing was created or written by any refused call.
        assert db.count_functions(SHA) == 0
        assert db.count_edges(SHA) == 0
        assert db.count_types(SHA) == 0
    finally:
        db.close()
    db_path.unlink()


def test_jar_with_a_bad_class_entry_is_refused_by_entry_name(tmp_path: Path) -> None:
    good = _plain_class("app/Good")
    archive = _jar_bytes(
        (
            ("app/Bad.class", b"NOTACLASSFILE"),
            ("app/Good.class", good),
            ("app/Good.class.txt", good),
        )
    )
    with pytest.raises(JvmError) as excinfo:
        parse_jar(archive)
    assert "app/Bad.class" in str(excinfo.value)

    db_path = tmp_path / "jvm-badjar.db"
    db = Database(db_path)
    try:
        with pytest.raises(JvmError):
            ingest(db, SHA, archive)
        assert db.count_functions(SHA) == 0
        assert db.count_types(SHA) == 0
    finally:
        db.close()
    db_path.unlink()


# --------------------------------------------------------------------------- #
# 7. JAR determinism: sorted entries, skip counts, stable ordinals
# --------------------------------------------------------------------------- #


def test_jar_entries_are_visited_in_sorted_order_and_skips_counted() -> None:
    forward = _multi_class_jar()
    classes, diagnostics = parse_jar(forward)
    assert diagnostics == {"format": "jar", "entries": 6, "skipped": 3, "classes": 3}
    assert [c.name for c in classes] == ["app/Base", "app/Derived", "z/Z"]
    assert classes[0].name == "app/Base" and classes[0].fields[0].name == "ttl"

    reversed_classes, reversed_diagnostics = parse_jar(_multi_class_jar(reversed_order=True))
    assert reversed_diagnostics == diagnostics
    assert [c.name for c in reversed_classes] == [c.name for c in classes]
    assert method_addr_index(reversed_classes) == method_addr_index(classes)
    assert method_addr_index(classes) == {
        ("app/Base", "run", "()V"): 1,
        ("app/Derived", "extra", "()V"): 2,
        ("app/Derived", "run", "()V"): 3,
        ("z/Z", "zeta", "()V"): 4,
    }

    # parse_jar accepts bytes and a path, and answers identically for both.
    path = Path(__file__).parent / "_jvm_tmp_fixture.jar"
    try:
        path.write_bytes(forward)
        from_path, path_diagnostics = parse_jar(path)
        assert path_diagnostics == diagnostics
        assert [c.name for c in from_path] == [c.name for c in classes]
    finally:
        path.unlink(missing_ok=True)

    # A single class file is not a JAR: refuse rather than guess.
    with pytest.raises(JvmError):
        parse_jar(_plain_class("app/Base"))


def test_jar_member_order_does_not_change_ingested_rows(tmp_path: Path) -> None:
    first = tmp_path / "order-a.db"
    second = tmp_path / "order-b.db"
    db_a = Database(first)
    db_b = Database(second)
    try:
        result_a = ingest(db_a, SHA, _multi_class_jar())
        result_b = ingest(db_b, SHA_OTHER, _multi_class_jar(reversed_order=True))
        for key in ("classes", "methods", "fields", "edges", "invokes", "unresolved"):
            assert result_a[key] == result_b[key]
        assert (result_a["classes"], result_a["methods"], result_a["edges"]) == (3, 4, 1)
        def shape(rows: list) -> list:
            return [
                (r.func_addr, r.name, r.size, r.prototype, r.calling_convention)
                for r in rows
            ]
        assert shape(db_a.get_functions(SHA)) == shape(db_b.get_functions(SHA_OTHER))
        assert shape(db_a.get_functions(SHA)) == [
            (1, "app.Base.run", 1, "()V", "jvm"),
            (2, "app.Derived.extra", 1, "()V", "jvm"),
            (3, "app.Derived.run", 4, "()V", "jvm"),
            (4, "z.Z.zeta", 1, "()V", "jvm"),
        ]
        assert db_a.get_edges(SHA) == [
            EdgeRecord(binary_sha256=SHA, src_addr=3, dst_addr=1, kind=EdgeKind.CALL)
        ]
        assert db_b.get_edges(SHA_OTHER) == [
            EdgeRecord(
                binary_sha256=SHA_OTHER, src_addr=3, dst_addr=1, kind=EdgeKind.CALL
            )
        ]
        assert db_b.get_callees(SHA_OTHER, 3) == [1]
    finally:
        db_a.close()
        db_b.close()
    first.unlink()
    second.unlink()


# --------------------------------------------------------------------------- #
# 8. replace semantics + ordinal stability across re-ingest
# --------------------------------------------------------------------------- #


def test_re_ingest_is_idempotent_and_ordinals_are_stable(tmp_path: Path) -> None:
    db_path = tmp_path / "jvm-reingest.db"
    archive = _multi_class_jar()
    db = Database(db_path)
    try:
        first = ingest(db, SHA, archive)
        funcs = db.get_functions(SHA)
        edges = db.get_edges(SHA)
        types = {r.type_id: r.layout for r in db.get_types(SHA)}

        second = ingest(db, SHA, archive)
        assert second["replaced"] == first["methods"]
        for key in ("classes", "methods", "fields", "edges", "invokes", "unresolved"):
            assert second[key] == first[key]
        assert db.get_functions(SHA) == funcs
        assert db.get_edges(SHA) == edges
        assert {r.type_id: r.layout for r in db.get_types(SHA)} == types
        assert method_addr_index(parse_jar(archive)[0]) == {
            ("app/Base", "run", "()V"): 1,
            ("app/Derived", "extra", "()V"): 2,
            ("app/Derived", "run", "()V"): 3,
            ("z/Z", "zeta", "()V"): 4,
        }
        assert {
            (record.name.rsplit(".", 1)[0].replace(".", "/"), record.name.rsplit(".", 1)[1]):
            record.func_addr
            for record in funcs
        } == {
            ("app/Base", "run"): 1,
            ("app/Derived", "extra"): 2,
            ("app/Derived", "run"): 3,
            ("z/Z", "zeta"): 4,
        }
    finally:
        db.close()
    db_path.unlink()


def test_ingest_replaces_prior_rows_per_sha_and_leaves_other_namespaces(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "jvm-replace.db"
    big = _multi_class_jar()
    small = _jar_bytes((("app/Base.class", _plain_class("app/Base")),))
    db = Database(db_path)
    try:
        # Rows the jvm namespace must never touch: another calling convention,
        # another binary's jvm rows, and a non-jvm: type lattice row.
        db.upsert_function(
            FunctionRecord(
                binary_sha256=SHA,
                func_addr=0x1000,
                name="native_helper",
                calling_convention="cdecl",
            )
        )
        db.add_edge(SHA, 0x1000, 0x2000, EdgeKind.CALL)
        db.upsert_type(
            TypeRecord(
                type_id="rtti:?AVFoo@@",
                name="Foo",
                kind="class",
                binary_sha256=SHA,
            )
        )
        ingest(db, SHA_OTHER, _plain_class("other/O"))

        first = ingest(db, SHA, big)
        assert (first["methods"], first["fields"], first["edges"]) == (4, 2, 1)
        assert db.count_functions(SHA) == 5  # 4 jvm + 1 cdecl

        second = ingest(db, SHA, small)
        assert second["replaced"] == first["methods"]
        assert second["methods"] == 1
        # func_addr ordering: the jvm ordinal 1 sorts before the cdecl 0x1000.
        assert [f.name for f in db.get_functions(SHA)] == ["app.Base.run", "native_helper"]
        assert db.count_functions(SHA) == 2  # 1 jvm + 1 cdecl
        # ... and the call edge sourced by a replaced ordinal is gone with it,
        # while the cdecl edge survives.
        assert db.get_edges(SHA) == [
            EdgeRecord(binary_sha256=SHA, src_addr=0x1000, dst_addr=0x2000, kind=EdgeKind.CALL)
        ]
        # jvm:% type rows are replace-per-sha; foreign type rows are untouched.
        assert {r.type_id for r in db.get_types(SHA)} == {"jvm:app/Base", "rtti:?AVFoo@@"}
        # Another binary's jvm rows are a different sha and stay intact.
        assert db.count_functions(SHA_OTHER) == 1
        assert {r.type_id for r in db.get_types(SHA_OTHER)} == {"jvm:other/O"}
    finally:
        db.close()
    db_path.unlink()


# --------------------------------------------------------------------------- #
# 9. hygiene: Trap 7 (handles), Trap 11 (faulthandler), stdlib-only surface
# --------------------------------------------------------------------------- #


def test_trap7_handles_are_closed_for_every_target_kind(tmp_path: Path) -> None:
    data = _plain_class("app/Base")

    # A Database target: the store owns its handle and is closed by the caller.
    db_path = tmp_path / "handles" / "store.db"
    db = Database(db_path)
    try:
        assert ingest(db, SHA, data)["methods"] == 1
    finally:
        db.close()
    assert db.closed is True
    db_path.unlink()  # PermissionError [WinError 32] if a WAL handle leaked

    # A raw connection target: the caller's handle and state stay the caller's.
    conn_path = tmp_path / "handles" / "conn.db"
    conn = sqlite3.connect(str(conn_path), isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        assert ingest(conn, SHA, data)["methods"] == 1
        assert conn.row_factory is sqlite3.Row
        assert conn.isolation_level is None
        assert conn.in_transaction is False
    finally:
        conn.close()
    conn_path.unlink()

    # A path target bootstraps funcs/edges/types and releases its own handle.
    path_target = tmp_path / "handles" / "fresh.db"
    assert ingest(path_target, SHA, data)["methods"] == 1
    assert path_target.exists()
    path_target.unlink()
    assert list(tmp_path.glob("handles/*.db-wal")) == []
    assert list(tmp_path.glob("handles/*.db-shm")) == []

    # create=False over an absent path refuses instead of creating it.
    absent = tmp_path / "handles" / "absent.db"
    with pytest.raises(JvmError) as excinfo:
        ingest(absent, SHA, data, create=False)
    assert "database not found" in str(excinfo.value)
    assert not absent.exists()

    # A closed Database is refused, not silently reopened.
    closed = Database(tmp_path / "handles" / "closed.db")
    closed.close()
    with pytest.raises(JvmError):
        ingest(closed, SHA, data)


def test_no_jdk_no_faulthandler_and_public_surface_reexports() -> None:
    root = Path(__file__).resolve().parents[1]
    jvm_source = Path(jvm.__file__).read_text(encoding="utf-8")
    parse_source = Path(jvm_parse.__file__).read_text(encoding="utf-8")

    # Trap 11: no faulthandler import anywhere, and the pin stays in pyproject.
    for source in (jvm_source, parse_source):
        assert re.search(r"^\s*(import|from)\s+faulthandler\b", source, re.M) is None
    assert "-p no:faulthandler" in (root / "pyproject.toml").read_text(encoding="utf-8")

    # No JDK and no subprocess: the parse half is bytes-in / objects-out.
    for source in (jvm_source, parse_source):
        assert "subprocess" not in source

    # The split keeps the parsing entry points on ``toothpick.jvm`` unchanged.
    assert jvm.parse_class_file is jvm_parse.parse_class_file
    assert jvm.parse_jar is jvm_parse.parse_jar
    assert jvm.method_addr_index is jvm_parse.method_addr_index
    assert jvm.JvmClass is jvm_parse.JvmClass
    assert jvm.JvmError is jvm_parse.JvmError
    for name in ("ingest", "parse_class_file", "parse_jar", "method_addr_index", "MAGIC"):
        assert name in jvm.__all__
    assert jvm.MAGIC == MAGIC == b"\xca\xfe\xba\xbe"
