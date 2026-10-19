"""Toothpick JVM structural ingest -- class-file / JAR -> ``funcs``/``edges``/``types``.
Storage half of the J1 module split: class-file / JAR structural parsing now
lives verbatim in :mod:`toothpick.jvm_parse` (the opcode-length tables,
:class:`JvmError`, the ``Jvm*`` data contracts, the bounds-checked reader, the
constant-pool and instruction walk, the attribute/field/method readers,
:func:`parse_class_file`, :func:`parse_jar`, and the deterministic synthetic
ordinal / type-id helpers).  Everything that parses is imported from there and
re-exported here, so ``toothpick.jvm`` keeps its public surface unchanged; what
remains in this module is the *storage* half -- the minimal
``funcs``/``edges``/``types`` DDLs, the connection/transaction discipline, and
:func:`ingest`, which turns already-parsed classes into rows.

Binding contract: ``docs/units/UNIT-J1-JVM-PLAN.md`` section 1 (Chunk 1, this
drive) and section 2 (interface specification); milestone J1 of
``docs/NEXT-TRACK.md`` ("Class-file/JAR parsing; methods/fields/``invoke*``
callgraph into ``funcs``/``edges``.  No lifting, no SSA.").  Design anchors:
``docs/design-document.md`` section 1 ("no fabricated structs -- every class row
traces to parsed bytes") and section 7 (edge taxonomy: ``call``/``xref``/``cfg``).

Every row written here traces to a byte range the parsing half either consumed
or refused: there is no lifting, no SSA, no taint, no decompilation and no
live-heap grounding (the JVM boundary of ``docs/NEXT-TRACK.md``).  The
non-fabrication rules themselves -- full constant-pool tag coverage,
``invoke*`` sites found by decoding the instruction stream rather than by
scanning for opcode-valued bytes, every reference resolved through the constant
pool or refused, truncation and bad magic as hard failures -- are documented
with the code that enforces them, in :mod:`toothpick.jvm_parse`.

Synthetic addresses are labelled as synthetic: ``funcs.func_addr`` is a
deterministic 1-based ordinal over methods in ``(class, method)`` sorted order
(:func:`method_addr_index`) -- never a bytecode offset, never a file offset and
never an address in any image.

Buffer/DB hygiene: this module never imports ``faulthandler`` (Trap 11 -- the
suite runs under ``-p no:faulthandler``) and every SQLite handle it opens is
closed in a ``finally`` (Trap 7 -- a leaked WAL handle surfaces on Windows
teardown as ``PermissionError: [WinError 32]``).  Writes use
``ON CONFLICT ... DO UPDATE/DO NOTHING`` (Trap 8) and every iteration order is
sorted (Trap 10), so a re-ingest converges instead of accumulating duplicates.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from toothpick.jvm_parse import (
    CALLING_CONVENTION,
    CP_TAG_NAMES,
    INVOKE_NAMES,
    INVOKE_OPS,
    JVM_FIELD_PREFIX,
    JVM_TYPE_PREFIX,
    MAGIC,
    JvmClass,
    JvmError,
    JvmField,
    JvmInvoke,
    JvmMethod,
    _display_name,
    _ordered_methods,
    _sorted_classes,
    _sorted_fields,
    _sorted_methods,
    class_type_id,
    field_type_id,
    method_addr_index,
    parse_class_file,
    parse_jar,
)
from toothpick.storage import Database, EdgeKind, EdgeRecord, FunctionRecord, TypeRecord

__all__ = [
    "CALLING_CONVENTION",
    "CP_TAG_NAMES",
    "INVOKE_NAMES",
    "INVOKE_OPS",
    "JVM_FIELD_PREFIX",
    "JVM_TYPE_PREFIX",
    "MAGIC",
    "MINIMAL_EDGES_DDL",
    "MINIMAL_FUNCS_DDL",
    "MINIMAL_TYPES_DDL",
    "JvmClass",
    "JvmError",
    "JvmField",
    "JvmInvoke",
    "JvmMethod",
    "TargetLike",
    "class_type_id",
    "field_type_id",
    "ingest",
    "method_addr_index",
    "parse_class_file",
    "parse_jar",
]

# --------------------------------------------------------------------------- #
# Parsing is imported, not re-implemented (J1 split)
# --------------------------------------------------------------------------- #

# Everything from the opcode-length tables through :func:`_display_name` --
# ``JvmError``, the ``Jvm*`` data contracts, ``_Reader``, the constant-pool
# walk, the instruction-length decoder and ``invoke*`` scan, the attribute /
# field / method readers, :func:`parse_class_file`, the JAR readers and
# :func:`parse_jar`, and the deterministic ordinal / type-id helpers -- now
# lives verbatim in ``toothpick.jvm_parse`` and is bound by the import above,
# so ``jvm.JvmClass is jvm_parse.JvmClass`` and no parsing logic is duplicated
# or redefined here.  The names listed in ``__all__`` are re-exported
# unchanged, so existing ``from toothpick.jvm import ...`` call sites keep
# working across the split.
# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #

#: The ``funcs`` DDL, mirrored from ``toothpick.storage`` (``_SCHEMA["funcs"]``)
#: so a standalone db path can be bootstrapped without importing private storage
#: internals -- the same precedent as ``profile.MINIMAL_FUNCS_DDL``.
MINIMAL_FUNCS_DDL = """
CREATE TABLE IF NOT EXISTS funcs (
    binary_sha256      TEXT    NOT NULL,
    func_addr          INTEGER NOT NULL,
    name               TEXT    NOT NULL DEFAULT '',
    size               INTEGER NOT NULL DEFAULT 0,
    calling_convention TEXT    NOT NULL DEFAULT '',
    prototype          TEXT    NOT NULL DEFAULT '',
    ir_pcode           TEXT,
    decompiled_c       TEXT,
    PRIMARY KEY (binary_sha256, func_addr)
)
"""

#: The ``edges`` DDL, mirrored verbatim (including the ``kind`` CHECK).
MINIMAL_EDGES_DDL = """
CREATE TABLE IF NOT EXISTS edges (
    binary_sha256 TEXT    NOT NULL,
    src_addr      INTEGER NOT NULL,
    dst_addr      INTEGER NOT NULL,
    kind          TEXT    NOT NULL CHECK (kind IN ('call', 'xref', 'cfg')),
    PRIMARY KEY (binary_sha256, src_addr, dst_addr, kind)
)
"""

#: The ``types`` DDL, mirrored from ``toothpick.storage`` (``_SCHEMA["types"]``).
MINIMAL_TYPES_DDL = """
CREATE TABLE IF NOT EXISTS types (
    type_id       TEXT    NOT NULL,
    name          TEXT    NOT NULL DEFAULT '',
    kind          TEXT    NOT NULL DEFAULT 'unknown',
    size_bytes    INTEGER NOT NULL DEFAULT 0,
    layout_json   TEXT,
    lattice_rank  INTEGER NOT NULL DEFAULT 0,
    binary_sha256 TEXT    NOT NULL DEFAULT '',
    PRIMARY KEY (binary_sha256, type_id)
)
"""

#: Byte-identical to ``Database._FUNC_UPSERT`` (same columns, same conflict key).
_FUNC_UPSERT = """
INSERT INTO funcs (
    binary_sha256, func_addr, name, size,
    calling_convention, prototype, ir_pcode, decompiled_c
) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT (binary_sha256, func_addr) DO UPDATE SET
    name               = excluded.name,
    size               = excluded.size,
    calling_convention = excluded.calling_convention,
    prototype          = excluded.prototype,
    ir_pcode           = excluded.ir_pcode,
    decompiled_c       = excluded.decompiled_c
"""

#: Byte-identical to ``Database._EDGE_UPSERT``.
_EDGE_UPSERT = """
INSERT INTO edges (binary_sha256, src_addr, dst_addr, kind)
VALUES (?, ?, ?, ?)
ON CONFLICT (binary_sha256, src_addr, dst_addr, kind) DO NOTHING
"""

#: Byte-identical to ``Database.upsert_type``'s statement.
_TYPES_UPSERT = (
    "INSERT INTO types (type_id, name, kind, size_bytes, layout_json, "
    "lattice_rank, binary_sha256) VALUES (?, ?, ?, ?, ?, ?, ?) "
    "ON CONFLICT (binary_sha256, type_id) DO UPDATE SET name = excluded.name, "
    "kind = excluded.kind, size_bytes = excluded.size_bytes, "
    "layout_json = excluded.layout_json, lattice_rank = excluded.lattice_rank, "
    "binary_sha256 = excluded.binary_sha256"
)

#: Replace-per-sha, namespace-scoped: only rows this module owns are touched.
_SELECT_PRIOR_FUNCS = (
    "SELECT func_addr FROM funcs WHERE binary_sha256 = ? "
    "AND calling_convention = ?"
)
_DELETE_PRIOR_EDGES = (
    "DELETE FROM edges WHERE binary_sha256 = ? AND kind = 'call' AND src_addr IN "
    "(SELECT func_addr FROM funcs WHERE binary_sha256 = ? "
    "AND calling_convention = ?)"
)
_DELETE_PRIOR_FUNCS = (
    "DELETE FROM funcs WHERE binary_sha256 = ? AND calling_convention = ?"
)
_DELETE_PRIOR_TYPES = (
    "DELETE FROM types WHERE binary_sha256 = ? AND type_id LIKE ?"
)

TargetLike = Union[Database, sqlite3.Connection, str, Path]


def _require_sha(sha: Any) -> str:
    if isinstance(sha, bool) or not isinstance(sha, str):
        raise JvmError(f"sha must be a non-empty string, got {sha!r}")
    text = sha.strip()
    if not text:
        raise JvmError("sha must be a non-empty string")
    return text


@contextmanager
def _write_connection(
    target: TargetLike, *, create: bool
) -> Iterator[sqlite3.Connection]:
    """Yield a write connection inside one transaction; always closed/released.

    A :class:`Database` contributes its own ``transaction()`` (Trap 7: the handle
    belongs to the store and is released by it).  A caller-supplied
    :class:`sqlite3.Connection` keeps its own ``row_factory``/``isolation_level``,
    but the transaction boundary is ours: a connection with no transaction open --
    at any ``isolation_level`` -- runs ``BEGIN IMMEDIATE`` ... ``COMMIT`` here, and
    one already inside a caller transaction nests as a ``SAVEPOINT``, so only the
    rows of this ingest roll back on failure.  A closed :class:`Database` or
    connection is refused rather than silently reopened.  A path is opened, put in
    WAL mode and closed here, in a ``finally``.
    """
    if isinstance(target, Database):
        if target.closed:
            raise JvmError("database is closed")
        with target.transaction() as conn:
            yield conn
        return
    if isinstance(target, sqlite3.Connection):
        conn = target
        try:
            in_transaction = conn.in_transaction
        except sqlite3.ProgrammingError as exc:
            raise JvmError(f"connection is closed: {exc}") from exc
        if in_transaction:
            # The caller owns the open transaction: this batch nests inside it as a
            # savepoint, so a failed ingest rolls back only its own rows and leaves
            # the caller's transaction usable.
            conn.execute("SAVEPOINT toothpick_jvm_ingest")
            try:
                yield conn
            except BaseException:
                conn.execute("ROLLBACK TO toothpick_jvm_ingest")
                conn.execute("RELEASE toothpick_jvm_ingest")
                raise
            else:
                conn.execute("RELEASE toothpick_jvm_ingest")
            return
        # Nothing was open, whatever the isolation_level says: the boundary is ours,
        # so a successful ingest is committed (durable) before this call returns --
        # exactly as for a path target -- and a failure rolls back.
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        else:
            if conn.in_transaction:
                conn.execute("COMMIT")
        return
    path = Path(target)
    if not path.exists() and not create:
        raise JvmError(f"database not found: {path}")
    if str(path.parent) not in ("", "."):
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), isolation_level=None)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 10000")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        else:
            conn.execute("COMMIT")
    except sqlite3.Error as exc:
        raise JvmError(f"jvm ingest failed for {path}: {exc}") from exc
    finally:
        conn.close()


def _ensure_tables(conn: sqlite3.Connection) -> None:
    """Create ``funcs``/``edges``/``types`` if absent (``IF NOT EXISTS``, never alters)."""
    try:
        for ddl in (MINIMAL_FUNCS_DDL, MINIMAL_EDGES_DDL, MINIMAL_TYPES_DDL):
            conn.execute(ddl)
    except sqlite3.Error as exc:
        raise JvmError(f"cannot ensure funcs/edges/types tables: {exc}") from exc


def _coerce_classes(
    source: Any,
) -> Tuple[List[JvmClass], Dict[str, Any]]:
    """Turn a class file, a JAR (bytes or path), or parsed classes into classes."""
    if isinstance(source, JvmClass):
        return [source], {"format": "class", "entries": 1, "skipped": 0, "classes": 1}
    if isinstance(source, (bytes, bytearray, memoryview)):
        return _parse_blob(bytes(source), label="")
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise JvmError(f"source not found: {path}")
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise JvmError(f"cannot read {path}: {exc}") from exc
        return _parse_blob(data, label=f"{path}: ")
    if isinstance(source, Iterable):
        items = list(source)
        for item in items:
            if not isinstance(item, JvmClass):
                raise JvmError(
                    "ingest source must contain JvmClass instances, got "
                    f"{type(item).__name__}"
                )
        return items, {
            "format": "classes",
            "entries": len(items),
            "skipped": 0,
            "classes": len(items),
        }
    raise JvmError(
        f"unsupported ingest source type {type(source).__name__} "
        "(expected class bytes, JAR bytes, a path, or JvmClass instances)"
    )


def _parse_blob(data: bytes, *, label: str) -> Tuple[List[JvmClass], Dict[str, Any]]:
    """Sniff class file vs JAR from the first four bytes (never from the name)."""
    if len(data) >= 4 and data[:4] == MAGIC:
        return [parse_class_file(data)], {
            "format": "class",
            "entries": 1,
            "skipped": 0,
            "classes": 1,
        }
    if len(data) >= 4 and data[:2] == b"PK":
        classes, diagnostics = parse_jar(data)
        if label:
            diagnostics = {**diagnostics, "source": label}
        return classes, diagnostics
    if not data:
        raise JvmError(f"{label}empty source: no class file or JAR bytes")
    raise JvmError(
        f"{label}unrecognised source: first bytes {data[:4].hex()} are neither a "
        "class file (CAFEBABE) nor a ZIP/JAR (PK..)"
    )


def _class_layout(
    cls: JvmClass,
    method_stats: Dict[Tuple[str, str, str], Dict[str, int]],
    ordinals: Dict[Tuple[str, str, str], int],
    targets: Sequence[str],
) -> Dict[str, Any]:
    """The ``layout`` of one class row: parsed structure, explicitly labelled."""
    methods: List[Dict[str, Any]] = []
    for method in _sorted_methods(cls.methods):
        key = (cls.name, method.name, method.descriptor)
        stats = method_stats.get(key, {"invokes": 0, "unresolved": 0})
        methods.append(
            {
                "name": method.name,
                "descriptor": method.descriptor,
                "access_flags": int(method.access_flags),
                "code_len": int(method.code_len),
                "func_addr": ordinals.get(key),
                "invokes": int(stats.get("invokes", 0)),
                "unresolved": int(stats.get("unresolved", 0)),
            }
        )
    fields = [
        {
            "name": item.name,
            "descriptor": item.descriptor,
            "access_flags": int(item.access_flags),
        }
        for item in _sorted_fields(cls.fields)
    ]
    return {
        "provenance": "jvm-class",
        "internal_name": cls.name,
        "name": cls.dotted_name,
        "synthetic_addr": True,
        "access_flags": int(cls.access_flags),
        "minor_version": int(cls.minor_version),
        "major_version": int(cls.major_version),
        "super": cls.super_name,
        "interfaces": list(cls.interfaces),
        "methods": methods,
        "fields": fields,
        "invoke_targets": list(targets),
    }


def _field_layout(cls: JvmClass, item: JvmField) -> Dict[str, Any]:
    """The ``layout`` of one field row (the field member of one class file)."""
    return {
        "provenance": "jvm-field",
        "class": cls.name,
        "internal_name": cls.name,
        "name": item.name,
        "descriptor": item.descriptor,
        "access_flags": int(item.access_flags),
    }


def ingest(
    target: TargetLike,
    sha: Any,
    source: Any,
    *,
    create: bool = True,
    lattice_rank: int = 3,
) -> Dict[str, Any]:
    """Ingest a class file / JAR (or already-parsed classes) into the graph DB.

    ``source`` may be class bytes, JAR bytes, a filesystem path, a
    :class:`JvmClass`, or an iterable of :class:`JvmClass`; bytes and paths are
    sniffed (``CAFEBABE`` vs ``PK``), never trusted by extension.

    What lands where (plan section 1, bullet 1):

    * every method -> one ``funcs`` row: ``calling_convention="jvm"``,
      ``prototype=<descriptor>``, ``size=code_len`` (the ``Code`` attribute's
      ``code_length``, *not* an address), and ``func_addr`` = the deterministic
      synthetic 1-based ordinal of :func:`method_addr_index` -- a repeated
      ``(class, name, descriptor)`` keeps the first ordinal and gets no second row
      (``duplicates`` counts those), so the addresses written are exactly the
      addresses that map reports;
    * every ``invoke*`` site whose target is one of the ingested methods -> one
      ``edges`` row of kind ``call`` (``src`` = caller ordinal, ``dst`` = callee
      ordinal); every other site (JDK/external callee, ``invokedynamic``
      bootstrap) increments ``unresolved`` and writes nothing;
    * every class -> one ``types`` row ``type_id='jvm:<internal name>'``,
      ``kind='class'``, ``layout={methods, fields, ...}``; every field also gets
      its own ``jvm:<class>#<field>:<descriptor>`` row with ``kind='field'``, so
      a field inventory is queryable without decoding the class layout.  Both
      trace to the parsed field/method tables -- no synthetic members.

    Semantics are *replace per sha and per namespace*: in one transaction the
    previous ``calling_convention='jvm'`` funcs of ``sha`` are deleted (with the
    ``call`` edges whose source is one of those ordinals) and every ``jvm:%``
    type row is replaced, so a re-ingest converges instead of accumulating
    duplicates.  Rows belonging to other producers -- other calling conventions,
    other binaries' rows, non-``jvm:`` lattice rows -- are never touched.  All
    rows are built before the transaction opens, so a refused class file leaves
    the database exactly as it was.

    Returns ``{classes, methods, fields, edges, invokes, unresolved, duplicates,
    skipped, replaced, type_ids, binary_sha256, diagnostics}``.
    """
    binary = _require_sha(sha)
    if isinstance(lattice_rank, bool) or not isinstance(lattice_rank, int):
        raise JvmError(f"lattice_rank must be an integer, got {lattice_rank!r}")

    parsed, diagnostics = _coerce_classes(source)
    classes = list(parsed)

    ordinals: Dict[Tuple[str, str, str], int] = {}
    func_rows: List[Tuple[Any, ...]] = []
    duplicates = 0
    for ordinal, cls, method in _ordered_methods(classes):
        key = (cls.name, method.name, method.descriptor)
        if key in ordinals:
            # A repeated definition (not legal Java) keeps its first ordinal,
            # exactly as :func:`method_addr_index` documents: a second row for it
            # would invent an address that map does not contain.
            duplicates += 1
            continue
        ordinals[key] = ordinal
        func_rows.append(
            FunctionRecord(
                binary_sha256=binary,
                func_addr=ordinal,
                name=_display_name(cls.name, method.name),
                size=int(method.code_len),
                calling_convention=CALLING_CONVENTION,
                prototype=method.descriptor,
                ir_pcode=None,
                decompiled_c=None,
            ).to_row()
        )

    method_stats: Dict[Tuple[str, str, str], Dict[str, int]] = defaultdict(
        lambda: {"invokes": 0, "unresolved": 0}
    )
    invoke_targets: Dict[str, Set[str]] = defaultdict(set)
    edge_rows: List[Tuple[Any, ...]] = []
    seen_edges: Set[Tuple[int, int]] = set()
    total_invokes = 0
    unresolved = 0
    for cls in _sorted_classes(classes):
        for site in cls.invokes:
            total_invokes += 1
            key = (site.class_name, site.method_name, site.method_descriptor)
            stats = method_stats[key]
            stats["invokes"] += 1
            src = ordinals.get(key)
            dst: Optional[int] = None
            if site.resolved:
                dst = ordinals.get(
                    (site.target_class, site.target_name, site.target_descriptor)
                )
            if src is None or dst is None:
                stats["unresolved"] += 1
                unresolved += 1
                continue
            invoke_targets[site.class_name].add(
                f"{site.target_class}.{site.target_name}:{site.target_descriptor}"
            )
            if (src, dst) in seen_edges:
                continue
            seen_edges.add((src, dst))
            edge_rows.append(
                EdgeRecord(
                    binary_sha256=binary,
                    src_addr=src,
                    dst_addr=dst,
                    kind=EdgeKind.CALL,
                ).to_row()
            )

    type_rows: List[Tuple[Any, ...]] = []
    for cls in _sorted_classes(classes):
        type_rows.append(
            TypeRecord(
                type_id=class_type_id(cls.name),
                name=cls.dotted_name,
                kind="class",
                size_bytes=0,
                layout=_class_layout(
                    cls,
                    method_stats,
                    ordinals,
                    sorted(invoke_targets.get(cls.name, ())),
                ),
                lattice_rank=lattice_rank,
                binary_sha256=binary,
            ).to_row()
        )
    field_rows = 0
    for cls in _sorted_classes(classes):
        for item in _sorted_fields(cls.fields):
            field_rows += 1
            type_rows.append(
                TypeRecord(
                    type_id=field_type_id(cls.name, item.name, item.descriptor),
                    name=_display_name(cls.name, item.name),
                    kind="field",
                    size_bytes=0,
                    layout=_field_layout(cls, item),
                    lattice_rank=lattice_rank,
                    binary_sha256=binary,
                ).to_row()
            )

    with _write_connection(target, create=create) as conn:
        _ensure_tables(conn)
        try:
            prior = [
                int(row[0])
                for row in conn.execute(
                    _SELECT_PRIOR_FUNCS, (binary, CALLING_CONVENTION)
                ).fetchall()
            ]
            if prior:
                conn.execute(
                    _DELETE_PRIOR_EDGES, (binary, binary, CALLING_CONVENTION)
                )
            cursor = conn.execute(_DELETE_PRIOR_FUNCS, (binary, CALLING_CONVENTION))
            replaced = cursor.rowcount if cursor.rowcount is not None else 0
            conn.execute(_DELETE_PRIOR_TYPES, (binary, f"{JVM_TYPE_PREFIX}%"))
            if func_rows:
                conn.executemany(_FUNC_UPSERT, func_rows)
            if edge_rows:
                conn.executemany(_EDGE_UPSERT, edge_rows)
            if type_rows:
                conn.executemany(_TYPES_UPSERT, type_rows)
        except sqlite3.Error as exc:
            raise JvmError(
                f"jvm ingest failed for {binary}: {exc} "
                "(is the funcs/edges/types schema at the current version?)"
            ) from exc

    return {
        "binary_sha256": binary,
        "classes": len(classes),
        "methods": len(func_rows),
        "fields": field_rows,
        "edges": len(edge_rows),
        "invokes": total_invokes,
        "unresolved": unresolved,
        "duplicates": duplicates,
        "replaced": max(replaced, 0),
        "skipped": int(diagnostics.get("skipped", 0) or 0),
        "type_ids": sorted({str(row[0]) for row in type_rows}),
        "diagnostics": dict(diagnostics),
    }
