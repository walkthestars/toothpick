"""
Toothpick MSVC RTTI recovery -- persistence half (unit P3 split).

Binding contract: ``docs/units/UNIT-P3-RTTI-PLAN.md`` section 1 (RTTI module:
``store`` / ``get_class``) and section 2 (interface specification).  Design
anchors: ``docs/design-document.md`` section 1 ("no fabricated structs -- every
class row traces to parsed bytes") and section 19 / Risk R1 (bounded emission:
default 50 scanned vtables, ceiling 1000).

This module is the *persistence* half of the RTTI module: the ``types``-table
upsert (:func:`store`), the read-back join (:func:`get_class`), and the
``Database`` / path / connection discipline both of them share.  Everything
from the section banner down is byte-identical to the corresponding region of
``toothpick.rtti`` (a pure move, no redesign).  The parsing half -- constants,
data contracts, name recovery, the bounds-checked reader, the structure readers
and ``scan_vtables`` -- lives in :mod:`toothpick.rtti_parse`, and the taxonomy
(:class:`ClassRecord`, :class:`BaseRef`, :class:`ScanResult`,
:class:`RttiError`, :class:`RttiArch`, :func:`class_type_id`,
:func:`demangle_msvc_type`) plus its constants (``PROVENANCE``,
``RTTI_TYPE_PREFIX``) are imported from there rather than duplicated, so there
is exactly one definition of each.

What is persisted and why it is honest:

* A class row is built from a :class:`ClassRecord` the caller recovered from
  image bytes, and carries the vtable/COL/TD/CHD addresses and the base chain
  as evidence (``provenance: 'msvc-rtti'``).  A base that was only *referenced*
  gets its own row marked ``inferred: true``, so a hierarchy is queryable as
  parent rows instead of as bare strings -- and never as a fabricated layout.
* Semantics are *replace per sha*: every ``rtti:%`` row of that binary is
  deleted in the same transaction before the new rows land, so re-scanning an
  image converges instead of accumulating duplicates, while non-RTTI lattice
  rows and other binaries' rows are untouched.

Trap hygiene: this module never imports ``faulthandler`` (Trap 11 -- suites run
under ``-p no:faulthandler``) and every SQLite handle it opens is closed in a
``finally`` (Trap 7 -- a leaked WAL handle surfaces on Windows teardown as
``PermissionError: [WinError 32]``).
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from toothpick.rtti_parse import (
    PROVENANCE,
    RTTI_TYPE_PREFIX,
    BaseRef,
    ClassRecord,
    RttiArch,
    RttiError,
    ScanResult,
    class_type_id,
    demangle_msvc_type,
)
from toothpick.storage import Database, TypeRecord

try:
    from toothpick.demangle import full_demangle as _undname
except Exception:  # pragma: no cover - import-time safety
    _undname = None  # type: ignore[assignment]


@lru_cache(maxsize=8192)
def _full_name(mangled: str) -> Optional[str]:
    """undname expansion of *mangled*, cached; ``None`` when unavailable."""
    if _undname is None or not isinstance(mangled, str) or not mangled:
        return None
    try:
        return _undname(mangled)
    except Exception:
        return None

#: Public surface of the persistence half.  ``BaseRef`` / ``ScanResult`` are
#: re-exported (imported, never redefined) so callers of the pre-split
#: ``toothpick.rtti`` keep one import site for the RTTI taxonomy.
__all__ = [
    "BaseRef",
    "ClassRecord",
    "RttiArch",
    "RttiError",
    "ScanResult",
    "TargetLike",
    "class_type_id",
    "get_class",
    "store",
]


#: The ``types`` DDL, mirrored from ``toothpick.storage`` (``_SCHEMA["types"]``)
#: so a standalone db path can be bootstrapped without importing private storage
#: internals -- the same precedent as ``profile.MINIMAL_FUNCS_DDL``.
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

#: Byte-identical to ``Database.upsert_type``'s statement (same conflict key):
#: rows written here and rows written through ``Database.upsert_type`` are the
#: same lattice rows.
_TYPES_UPSERT = (
    "INSERT INTO types (type_id, name, kind, size_bytes, layout_json, "
    "lattice_rank, binary_sha256) VALUES (?, ?, ?, ?, ?, ?, ?) "
    "ON CONFLICT (binary_sha256, type_id) DO UPDATE SET name = excluded.name, "
    "kind = excluded.kind, size_bytes = excluded.size_bytes, "
    "layout_json = excluded.layout_json, lattice_rank = excluded.lattice_rank, "
    "binary_sha256 = excluded.binary_sha256"
)

_TYPES_DELETE_NAMESPACE = (
    "DELETE FROM types WHERE binary_sha256 = ? AND type_id LIKE 'rtti:%'"
)

TargetLike = Union[Database, sqlite3.Connection, str, Path]


# --------------------------------------------------------------------------- #
# Persistence (types table)
# --------------------------------------------------------------------------- #


def _require_sha(sha: Any) -> str:
    if isinstance(sha, bool) or not isinstance(sha, str):
        raise RttiError(f"sha must be a non-empty string, got {sha!r}")
    text = sha.strip()
    if not text:
        raise RttiError("sha must be a non-empty string")
    return text


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
    ).fetchone()
    return row is not None


@contextmanager
def _read_connection(target: TargetLike) -> Iterator[sqlite3.Connection]:
    """Yield a read connection; anything opened here is closed in ``finally``."""
    if isinstance(target, Database):
        if target.closed:
            raise RttiError("database is closed")
        with target.connection() as conn:
            yield conn
        return
    if isinstance(target, sqlite3.Connection):
        yield target
        return
    path = Path(target)
    if not path.exists():
        raise RttiError(f"database not found: {path}")
    conn = sqlite3.connect(str(path), isolation_level=None)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 10000")
        yield conn
    except sqlite3.Error as exc:
        raise RttiError(f"rtti read failed for {path}: {exc}") from exc
    finally:
        conn.close()


@contextmanager
def _write_connection(
    target: TargetLike, *, create: bool
) -> Iterator[sqlite3.Connection]:
    """Yield a write connection inside one transaction; always closed/released.

    A :class:`Database` contributes its own ``transaction()`` (Trap 7: the
    handle belongs to the store and is released by it).  A caller-supplied
    :class:`sqlite3.Connection` keeps its own ``row_factory``/``isolation_level``
    -- only the transaction boundary is ours.  A path is opened, put in WAL mode
    and closed here.
    """
    if isinstance(target, Database):
        if target.closed:
            raise RttiError("database is closed")
        with target.transaction() as conn:
            yield conn
        return
    if isinstance(target, sqlite3.Connection):
        conn = target
        owned = conn.isolation_level is None and not conn.in_transaction
        if owned:
            conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except BaseException:
            if owned and conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        else:
            if owned and conn.in_transaction:
                conn.execute("COMMIT")
        return
    path = Path(target)
    if not path.exists() and not create:
        raise RttiError(f"database not found: {path}")
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
        raise RttiError(f"rtti write failed for {path}: {exc}") from exc
    finally:
        conn.close()


@contextmanager
def _types_writer(
    target: TargetLike, *, create: bool
) -> Iterator[sqlite3.Connection]:
    """A write connection whose ``types`` table is guaranteed to exist."""
    with _write_connection(target, create=create) as conn:
        try:
            conn.execute(MINIMAL_TYPES_DDL)
        except sqlite3.Error as exc:
            raise RttiError(f"cannot ensure the types table: {exc}") from exc
        yield conn


def _record(record: ClassRecord) -> ClassRecord:
    if not isinstance(record, ClassRecord):
        raise RttiError(f"classes must contain ClassRecord, got {type(record).__name__}")
    if not isinstance(record.mangled, str) or not record.mangled:
        raise RttiError("ClassRecord.mangled must be a non-empty string")
    return record


def _class_layout(record: ClassRecord) -> Dict[str, Any]:
    return {
        "provenance": PROVENANCE,
        "inferred": False,
        "arch": record.arch,
        "mangled": record.mangled,
        "demangled": bool(record.demangled),
        "full_name": _full_name(record.mangled),
        "vtable_addr": int(record.vtable_addr),
        "col_addr": int(record.col_addr),
        "td_addr": int(record.td_addr),
        "td_name_offset": int(record.td_name_offset),
        "chd_addr": int(record.chd_addr),
        "hierarchy": {
            "bases": [base.mangled for base in record.bases],
            "names": {base.mangled: base.name for base in record.bases},
            "mdisp": {base.mangled: int(base.mdisp) for base in record.bases},
        },
    }


def _base_layout(
    mangled: str, arch: str, referenced_by: Sequence[str]
) -> Dict[str, Any]:
    name, demangled = demangle_msvc_type(mangled)
    return {
        "provenance": PROVENANCE,
        "inferred": True,
        "arch": arch,
        "mangled": mangled,
        "demangled": demangled,
        "full_name": _full_name(mangled),
        "name": name if demangled else "",
        "referenced_by": sorted(referenced_by),
    }


def store(
    target: TargetLike,
    sha: Any,
    classes: Iterable[ClassRecord],
    *,
    arch: Optional[Any] = None,
    lattice_rank: int = 3,
    create: bool = True,
) -> Dict[str, Any]:
    """Upsert recovered classes (plus inferred base rows) into ``types``.

    Each class becomes ``type_id = 'rtti:<mangled>'``, ``kind = 'class'`` with a
    ``layout_json`` carrying ``provenance: 'msvc-rtti'``, the vtable/COL/CHD
    addresses and the ``hierarchy`` (``bases`` list plus per-base ``mdisp``).
    Every referenced base that was not itself scanned gets its own row with
    ``inferred: true`` in the layout, so a hierarchy is queryable as parent rows
    rather than as bare strings.

    Semantics are *replace per sha*: every ``rtti:%`` row of ``sha`` is deleted in
    the same transaction before the new rows land -- a crash leaves the previous
    scan intact rather than a half-replaced one -- while other (non-RTTI) lattice
    rows and other binaries' rows are untouched.  Re-scanning the same image
    converges to the same rows.

    ``target`` may be a :class:`~toothpick.storage.Database`, a raw
    :class:`sqlite3.Connection`, or a db path (``create=True`` bootstraps a fresh
    path with the ``types`` table, mirroring ``profile.import_samples``).  Rows are
    written with the exact statement/conflict key ``Database.upsert_type`` uses.

    Returns ``{classes, bases, updated, replaced, type_ids, binary_sha256}``.
    """
    binary = _require_sha(sha)
    if isinstance(lattice_rank, bool) or not isinstance(lattice_rank, int):
        raise RttiError(f"lattice_rank must be an integer, got {lattice_rank!r}")

    records = [_record(record) for record in classes]
    scanned = {record.mangled for record in records}
    arch_text = ""
    if arch is not None:
        arch_text = RttiArch.coerce(arch).value
    elif records:
        arch_text = next((record.arch for record in records if record.arch), "")

    class_rows: List[Tuple[Any, ...]] = []
    for record in records:
        layout = _class_layout(
            record if record.arch else ClassRecord(
                mangled=record.mangled,
                name=record.name,
                demangled=record.demangled,
                vtable_addr=record.vtable_addr,
                bases=record.bases,
                col_addr=record.col_addr,
                td_addr=record.td_addr,
                td_name_offset=record.td_name_offset,
                chd_addr=record.chd_addr,
                arch=arch_text,
            )
        )
        class_rows.append(
            TypeRecord(
                type_id=class_type_id(record.mangled),
                name=record.name,
                kind="class",
                size_bytes=0,
                layout=layout,
                lattice_rank=lattice_rank,
                binary_sha256=binary,
            ).to_row()
        )

    referenced: Dict[str, List[str]] = {}
    for record in records:
        for base in record.bases:
            if base.mangled in scanned:
                continue
            referenced.setdefault(base.mangled, [])
            if record.mangled not in referenced[base.mangled]:
                referenced[base.mangled].append(record.mangled)

    base_rows: List[Tuple[Any, ...]] = []
    for mangled in sorted(referenced):
        layout = _base_layout(mangled, arch_text, referenced[mangled])
        base_rows.append(
            TypeRecord(
                type_id=class_type_id(mangled),
                name=layout["name"],
                kind="class",
                size_bytes=0,
                layout=layout,
                lattice_rank=lattice_rank,
                binary_sha256=binary,
            ).to_row()
        )

    with _types_writer(target, create=create) as conn:
        try:
            cursor = conn.execute(_TYPES_DELETE_NAMESPACE, (binary,))
            replaced = cursor.rowcount if cursor.rowcount is not None else 0
            if base_rows:
                conn.executemany(_TYPES_UPSERT, base_rows)
            if class_rows:
                conn.executemany(_TYPES_UPSERT, class_rows)
        except sqlite3.Error as exc:
            raise RttiError(
                f"rtti store failed for {binary}: {exc} "
                "(is the types table at the current schema version?)"
            ) from exc

    type_ids = sorted(
        {row[0] for row in class_rows} | {row[0] for row in base_rows}
    )
    return {
        "binary_sha256": binary,
        "classes": len(class_rows),
        "bases": len(base_rows),
        "updated": len(class_rows) + len(base_rows),
        "replaced": max(replaced, 0),
        "type_ids": type_ids,
    }


def _layout_of(row: Any) -> Dict[str, Any]:
    raw = row["layout_json"]
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def get_class(
    target: TargetLike, sha: Any, name_or_mangled: Any
) -> Optional[Dict[str, Any]]:
    """Look one recovered class up by demangled ``name``, mangled name, or type id.

    Accepts ``Foo``, ``.?AVFoo@@`` or ``rtti:?AVFoo@@``-style ids alike, scoped to
    ``sha``.  Base links are joined back onto their own rows, so the result
    carries each base's recovered name / ``mdisp`` and whether that parent row was
    itself scanned or only inferred.  Returns ``None`` when nothing matches; a
    store without a ``types`` table reads as ``None`` -- reads never create state.
    """
    binary = _require_sha(sha)
    if not isinstance(name_or_mangled, str):
        raise RttiError(
            f"name_or_mangled must be a string, got {type(name_or_mangled).__name__}"
        )
    query = name_or_mangled.strip()
    if not query:
        raise RttiError("name_or_mangled must be a non-empty string")
    bare = query[len(RTTI_TYPE_PREFIX) :] if query.startswith(RTTI_TYPE_PREFIX) else query
    exact_id = class_type_id(bare)

    with _read_connection(target) as conn:
        if not _table_exists(conn, "types"):
            return None
        rows = conn.execute(
            "SELECT type_id, name, kind, layout_json, lattice_rank FROM types "
            "WHERE binary_sha256 = ? AND (type_id IN (?, ?) OR name = ?)",
            (binary, query, exact_id, query),
        ).fetchall()
        if not rows:
            return None

        def rank(row: Any) -> Tuple[int, int, str]:
            layout = _layout_of(row)
            exact = 0 if row["type_id"] == exact_id else 1
            inferred = 1 if layout.get("inferred") else 0
            return (exact, inferred, row["type_id"])

        chosen = sorted(rows, key=rank)[0]
        layout = _layout_of(chosen)
        hierarchy = layout.get("hierarchy") or {}
        base_mangleds = list(hierarchy.get("bases") or [])
        base_names = dict(hierarchy.get("names") or {})
        base_mdisps = dict(hierarchy.get("mdisp") or {})

        bases: List[Dict[str, Any]] = []
        base_type_ids: List[str] = []
        for mangled in base_mangleds:
            base_type_id = class_type_id(mangled)
            base_type_ids.append(base_type_id)
            parent = conn.execute(
                "SELECT type_id, name, layout_json FROM types "
                "WHERE binary_sha256 = ? AND type_id = ?",
                (binary, base_type_id),
            ).fetchone()
            parent_layout = _layout_of(parent) if parent is not None else {}
            base_name = base_names.get(mangled, "")
            if not base_name:
                recovered, ok = demangle_msvc_type(mangled)
                base_name = recovered if ok else ""
            if parent is not None and (parent["name"] or ""):
                base_name = parent["name"]
            bases.append(
                {
                    "mangled": mangled,
                    "name": base_name,
                    "mdisp": int(base_mdisps.get(mangled, 0)),
                    "inferred": bool(parent_layout.get("inferred")),
                    "known": parent is not None,
                }
            )

    result: Dict[str, Any] = {
        "type_id": chosen["type_id"],
        "name": chosen["name"] or "",
        "kind": chosen["kind"] or "unknown",
        "mangled": layout.get("mangled", bare),
        "demangled": bool(layout.get("demangled")),
        "arch": layout.get("arch", ""),
        "provenance": layout.get("provenance", ""),
        "inferred": bool(layout.get("inferred")),
        "vtable_addr": int(layout.get("vtable_addr", 0) or 0),
        "col_addr": int(layout.get("col_addr", 0) or 0),
        "td_addr": int(layout.get("td_addr", 0) or 0),
        "chd_addr": int(layout.get("chd_addr", 0) or 0),
        "lattice_rank": int(chosen["lattice_rank"] or 0),
        "bases": bases,
        "base_type_ids": base_type_ids,
        "referenced_by": list(layout.get("referenced_by") or []),
        "binary_sha256": binary,
        "layout": layout,
    }
    return result
