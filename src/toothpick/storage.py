"""
Toothpick SQLite graph storage engine (``toothpick.db``).

Implements the W03 storage contract (``PLAN-W03-static.md`` lines 7-19,
``docs/design-document.md`` sections 5 and 7):

* WAL-mode SQLite (``PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;``).
* Six core tables: ``metadata``, ``funcs``, ``edges``, ``ssa_facts``,
  ``types``, ``receipts``.
* Exactly-once cursor semantics per ``(binary_sha256, func_addr)``: re-analysis
  overwrites the row via ``ON CONFLICT ... DO UPDATE`` and never duplicates it.
* Transactional per-function writes (``transaction()``) with receipt commits, so
  a kill in the middle leaves no partial function/edge rows and a replay
  resumes cleanly (receipt ids are deterministic, so a replay updates the
  receipt row instead of appending a duplicate).

This module is deliberately stdlib-only (``sqlite3``) so the storage layer can
be tested without any disassembly backend.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4
from enum import Enum
from hashlib import sha256
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

__all__ = [
    "SCHEMA_VERSION",
    "StorageError",
    "SchemaError",
    "EdgeKind",
    "FunctionRecord",
    "EdgeRecord",
    "SsaFact",
    "TypeRecord",
    "VersionRecord",
    "Database",
    "make_receipt_id",
]

SCHEMA_VERSION = "w03.2"

PathLike = Union[str, Path]


class StorageError(Exception):
    """Base exception for storage failures."""


class SchemaError(StorageError):
    """Raised when the on-disk schema cannot be created or migrated."""


class EdgeKind(str, Enum):
    """Edge taxonomy from ``docs/design-document.md`` section 7.

    ``call`` is the call-graph edge (function -> callee function),
    ``cfg`` is an intra-function control-flow edge (function -> local block),
    ``xref`` is a reference from an instruction site to a code address.
    """

    CALL = "call"
    XREF = "xref"
    CFG = "cfg"

    @classmethod
    def coerce(cls, value: Union["EdgeKind", str]) -> "EdgeKind":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).lower())
        except ValueError as exc:
            raise StorageError(
                f"invalid edge kind {value!r}; expected one of {[k.value for k in cls]}"
            ) from exc


@dataclass
class FunctionRecord:
    """One ``funcs`` row (one analysed function of one binary)."""

    binary_sha256: str
    func_addr: int
    name: str = ""
    size: int = 0
    calling_convention: str = ""
    prototype: str = ""
    ir_pcode: Optional[str] = None
    decompiled_c: Optional[str] = None

    def to_row(self) -> Tuple[Any, ...]:
        return (
            self.binary_sha256,
            int(self.func_addr),
            self.name,
            int(self.size),
            self.calling_convention,
            self.prototype,
            self.ir_pcode,
            self.decompiled_c,
        )

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "FunctionRecord":
        return cls(
            binary_sha256=row["binary_sha256"],
            func_addr=int(row["func_addr"]),
            name=row["name"] or "",
            size=int(row["size"] or 0),
            calling_convention=row["calling_convention"] or "",
            prototype=row["prototype"] or "",
            ir_pcode=row["ir_pcode"],
            decompiled_c=row["decompiled_c"],
        )


@dataclass(frozen=True)
class EdgeRecord:
    """One ``edges`` row."""

    binary_sha256: str
    src_addr: int
    dst_addr: int
    kind: Union[EdgeKind, str] = EdgeKind.CALL

    def to_row(self) -> Tuple[Any, ...]:
        return (
            self.binary_sha256,
            int(self.src_addr),
            int(self.dst_addr),
            EdgeKind.coerce(self.kind).value,
        )


@dataclass(frozen=True)
class SsaFact:
    """One ``ssa_facts`` row."""

    fact_id: str
    binary_sha256: str
    func_addr: int
    var_name: str = ""
    ssa_reg: str = ""
    def_point: int = -1
    use_points: Sequence[int] = field(default_factory=tuple)

    def to_row(self) -> Tuple[Any, ...]:
        return (
            self.fact_id,
            self.binary_sha256,
            int(self.func_addr),
            self.var_name,
            self.ssa_reg,
            int(self.def_point),
            json.dumps(list(self.use_points)),
        )


@dataclass
class TypeRecord:
    """One ``types`` row of the global type lattice."""

    type_id: str
    name: str = ""
    kind: str = "unknown"
    size_bytes: int = 0
    layout: Optional[Dict[str, Any]] = None
    lattice_rank: int = 0
    binary_sha256: str = ""

    def to_row(self) -> Tuple[Any, ...]:
        return (
            self.type_id,
            self.name,
            self.kind,
            int(self.size_bytes),
            json.dumps(self.layout) if self.layout is not None else None,
            int(self.lattice_rank),
            self.binary_sha256,
        )

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "TypeRecord":
        layout = row["layout_json"]
        bin_sha = ""
        try:
            bin_sha = row["binary_sha256"] or ""
        except (IndexError, KeyError):
            pass
        return cls(
            type_id=row["type_id"],
            name=row["name"] or "",
            kind=row["kind"] or "unknown",
            size_bytes=int(row["size_bytes"] or 0),
            layout=json.loads(layout) if layout else None,
            lattice_rank=int(row["lattice_rank"] or 0),
            binary_sha256=bin_sha,
        )


@dataclass
class VersionRecord:
    """One ``versions`` row (similarity/diff mapping across binary revisions)."""

    binary_sha256: str
    target_sha256: str
    similarity_score: float = 0.0
    embedding_blob: Optional[bytes] = None
    offset_map: Optional[Dict[str, Any]] = None

    def to_row(self) -> Tuple[Any, ...]:
        return (
            self.binary_sha256,
            self.target_sha256,
            float(self.similarity_score),
            self.embedding_blob,
            json.dumps(self.offset_map) if self.offset_map is not None else None,
        )

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "VersionRecord":
        om = row["offset_map_json"]
        return cls(
            binary_sha256=row["binary_sha256"],
            target_sha256=row["target_sha256"],
            similarity_score=float(row["similarity_score"] or 0.0),
            embedding_blob=row["embedding_blob"],
            offset_map=json.loads(om) if om else None,
        )


# --- DDL ------------------------------------------------------------------
# The column lists and constraints below are the binding schema from
# PLAN-W03-static.md lines 11-16; the indexes are an implementation addition
# (design-document section 7 requires an index on (binary, addr)).

_SCHEMA: Tuple[Tuple[str, str], ...] = (
    (
        "metadata",
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key   TEXT PRIMARY KEY,
            value TEXT
        )
        """,
    ),
    (
        "funcs",
        """
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
        """,
    ),
    (
        "edges",
        """
        CREATE TABLE IF NOT EXISTS edges (
            binary_sha256 TEXT    NOT NULL,
            src_addr      INTEGER NOT NULL,
            dst_addr      INTEGER NOT NULL,
            kind          TEXT    NOT NULL CHECK (kind IN ('call', 'xref', 'cfg')),
            PRIMARY KEY (binary_sha256, src_addr, dst_addr, kind)
        )
        """,
    ),
    (
        "ssa_facts",
        """
        CREATE TABLE IF NOT EXISTS ssa_facts (
            fact_id       TEXT PRIMARY KEY,
            binary_sha256 TEXT    NOT NULL,
            func_addr     INTEGER NOT NULL,
            var_name      TEXT    NOT NULL DEFAULT '',
            ssa_reg       TEXT    NOT NULL DEFAULT '',
            def_point     INTEGER NOT NULL DEFAULT -1,
            use_points    TEXT
        )
        """,
    ),
    (
        "types",
        """
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
        """,
    ),
    (
        "receipts",
        """
        CREATE TABLE IF NOT EXISTS receipts (
            receipt_id    TEXT PRIMARY KEY,
            binary_sha256 TEXT NOT NULL,
            func_addr     INTEGER NOT NULL,
            phase         TEXT NOT NULL,
            timestamp     TEXT NOT NULL,
            payload_json  TEXT
        )
        """,
    ),
    (
        "versions",
        """
        CREATE TABLE IF NOT EXISTS versions (
            binary_sha256    TEXT    NOT NULL,
            target_sha256    TEXT    NOT NULL,
            similarity_score REAL    NOT NULL DEFAULT 0.0,
            embedding_blob   BLOB,
            offset_map_json  TEXT,
            PRIMARY KEY (binary_sha256, target_sha256)
        )
        """,
    ),
    (
        "trace_events",
        """
        CREATE TABLE IF NOT EXISTS trace_events (
            event_id      TEXT PRIMARY KEY,
            binary_sha256 TEXT    NOT NULL,
            func_addr     INTEGER NOT NULL,
            timestamp     TEXT    NOT NULL,
            event_type    TEXT    NOT NULL,
            payload_json  TEXT
        )
        """,
    ),
)

_INDEXES: Tuple[str, ...] = (
    "CREATE INDEX IF NOT EXISTS idx_funcs_binary ON funcs (binary_sha256, func_addr)",
    "CREATE INDEX IF NOT EXISTS idx_edges_src ON edges (binary_sha256, src_addr, kind)",
    "CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges (binary_sha256, dst_addr, kind)",
    "CREATE INDEX IF NOT EXISTS idx_ssa_func ON ssa_facts (binary_sha256, func_addr)",
    "CREATE INDEX IF NOT EXISTS idx_types_binary ON types (binary_sha256)",
    "CREATE INDEX IF NOT EXISTS idx_receipts_func ON receipts (binary_sha256, func_addr, phase)",
    "CREATE INDEX IF NOT EXISTS idx_versions_binary ON versions (binary_sha256)",
    "CREATE INDEX IF NOT EXISTS idx_trace_events_func ON trace_events (binary_sha256, func_addr)",
)

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

_EDGE_UPSERT = """
INSERT INTO edges (binary_sha256, src_addr, dst_addr, kind)
VALUES (?, ?, ?, ?)
ON CONFLICT (binary_sha256, src_addr, dst_addr, kind) DO NOTHING
"""

_SSA_UPSERT = """
INSERT INTO ssa_facts (fact_id, binary_sha256, func_addr, var_name, ssa_reg,
                       def_point, use_points)
VALUES (?, ?, ?, ?, ?, ?, ?)
ON CONFLICT (fact_id) DO UPDATE SET
    var_name   = excluded.var_name,
    ssa_reg    = excluded.ssa_reg,
    def_point  = excluded.def_point,
    use_points = excluded.use_points
"""

_RECEIPT_UPSERT = """
INSERT INTO receipts (receipt_id, binary_sha256, func_addr, phase, timestamp, payload_json)
VALUES (?, ?, ?, ?, ?, ?)
ON CONFLICT (receipt_id) DO UPDATE SET
    phase        = excluded.phase,
    timestamp    = excluded.timestamp,
    payload_json = excluded.payload_json
"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def make_receipt_id(binary_sha256: str, func_addr: int, phase: str, payload_json: str) -> str:
    """Deterministic receipt id for one ``{func_id, phase, receipt}`` commit.

    Determinism is deliberate: a re-analysis (or a kill-and-replay) of the same
    ``(binary, func, phase, payload)`` updates the existing receipt row instead
    of appending a duplicate, which is what makes resume idempotent.
    """
    digest = sha256(
        f"{binary_sha256}|{int(func_addr)}|{phase}|{payload_json}".encode("utf-8")
    ).hexdigest()
    return f"rcpt_{digest[:40]}"


class Database:
    """WAL-mode SQLite graph store.

    Usage::

        db = Database(Path("toothpick.db"))
        db.upsert_function(record)
        with db.transaction():          # atomic per-function batch
            db.upsert_function(a)
            db.add_edge(sha, 0x1000, 0x2000, EdgeKind.CALL)
        db.add_receipt(sha, 0x1000, "static_analysis", {"status": "ok"})
    """

    def __init__(
        self,
        db_path: PathLike = "toothpick.db",
        *,
        create: bool = True,
        busy_timeout_ms: int = 10_000,
    ) -> None:
        self.db_path = Path(db_path)
        if str(self.db_path.parent) not in ("", "."):
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        try:
            self._conn = sqlite3.connect(
                str(self.db_path), isolation_level=None, check_same_thread=False
            )
        except sqlite3.Error as exc:  # pragma: no cover - platform dependent
            raise StorageError(f"cannot open database {self.db_path}: {exc}") from exc
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(f"PRAGMA busy_timeout = {int(busy_timeout_ms)}")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA synchronous = NORMAL")
        self._conn.execute("PRAGMA foreign_keys = ON")
        if create:
            self.initialize()

    # -- lifecycle ---------------------------------------------------------

    def initialize(self) -> None:
        """Create tables/indexes and stamp schema metadata (idempotent)."""
        try:
            with self._lock:
                for _name, ddl in _SCHEMA:
                    self._conn.execute(ddl)
                self._migrate_types_primary_key()
                for idx in _INDEXES:
                    self._conn.execute(idx)
                self._conn.execute(
                    "INSERT INTO metadata (key, value) VALUES ('schema_version', ?) "
                    "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
                    (SCHEMA_VERSION,),
                )
        except sqlite3.Error as exc:
            raise SchemaError(f"schema initialization failed: {exc}") from exc

    def _migrate_types_primary_key(self) -> None:
        """Evolve a pre-``w03.2`` ``types`` table to a per-binary primary key.

        Older databases keyed ``types`` on ``type_id`` alone, so the same
        ``type_id`` arriving from a second binary overwrote the first binary's
        row: ``WHERE binary_sha256 = ?`` could not isolate binaries and
        ``DELETE ... WHERE binary_sha256 = ?`` could not reclaim one binary's
        types.  SQLite cannot change a primary key in place, so the table is
        rebuilt (rename, create, copy, drop) inside the same initialize step;
        the replacement DDL comes from :data:`_SCHEMA`, one source of truth.
        """
        row = self._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'types'"
        ).fetchone()
        current = " ".join((row["sql"] or "").split()).upper() if row is not None else ""
        if "PRIMARY KEY (BINARY_SHA256, TYPE_ID)" in current:
            return
        columns = [
            r["name"] for r in self._conn.execute("PRAGMA table_info(types)").fetchall()
        ]
        if "binary_sha256" not in columns:
            self._conn.execute(
                "ALTER TABLE types ADD COLUMN binary_sha256 TEXT NOT NULL DEFAULT ''"
            )
        self._conn.execute("ALTER TABLE types RENAME TO types_pre_migration")
        self._conn.execute(dict(_SCHEMA)["types"])
        self._conn.execute(
            "INSERT OR IGNORE INTO types (type_id, name, kind, size_bytes, "
            "layout_json, lattice_rank, binary_sha256) "
            "SELECT type_id, name, kind, size_bytes, layout_json, lattice_rank, "
            "binary_sha256 FROM types_pre_migration"
        )
        self._conn.execute("DROP TABLE types_pre_migration")

    def close(self) -> None:
        with self._lock:
            conn, self._conn = self._conn, None  # type: ignore[assignment]
            if conn is not None:
                conn.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    @property
    def closed(self) -> bool:
        return self._conn is None

    def _require_conn(self) -> sqlite3.Connection:
        conn = self._conn
        if conn is None:
            raise StorageError("database is closed")
        return conn

    def _new_connection(self) -> sqlite3.Connection:
        # WAL mode is a persistent property of the database file, so a second
        # connection reports ``journal_mode == wal`` immediately.
        conn = sqlite3.connect(str(self.db_path), isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        return conn

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Yield a fresh :class:`sqlite3.Connection` for reads/queries.

        Commits on clean exit, rolls back and closes on error.
        """
        conn = self._new_connection()
        try:
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Atomic batch: ``BEGIN IMMEDIATE`` ... ``COMMIT`` / ``ROLLBACK``.

        A crash (or exception) inside the block leaves no partial funcs/edges/
        receipts behind, which is the kill-resume guarantee of
        ``docs/design-document.md`` section 5: every per-function write and its
        receipt commit or vanish together.
        """
        with self._lock:
            conn = self._require_conn()
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                try:
                    conn.execute("ROLLBACK")
                except sqlite3.Error:  # pragma: no cover - already rolled back
                    pass
                raise
            else:
                conn.execute("COMMIT")

    # -- low-level helpers -------------------------------------------------

    def _execute(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._require_conn().execute(sql, params)

    def _execute_many(self, sql: str, rows: Iterable[Sequence[Any]]) -> int:
        with self._lock:
            cur = self._require_conn().executemany(sql, rows)
            return cur.rowcount if cur.rowcount is not None else 0

    # -- metadata ----------------------------------------------------------

    def set_metadata(self, key: str, value: Any) -> None:
        text = value if isinstance(value, str) else json.dumps(value)
        self._execute(
            "INSERT INTO metadata (key, value) VALUES (?, ?) "
            "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
            (key, text),
        )

    def get_metadata(self, key: str, default: Optional[str] = None) -> Optional[str]:
        row = self._execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
        return row["value"] if row is not None else default

    def delete_metadata(self, key: str) -> int:
        return self._execute("DELETE FROM metadata WHERE key = ?", (key,)).rowcount or 0

    def set_binary_metadata(self, binary_sha256: str, metadata: Dict[str, Any]) -> None:
        """Record container/arch facts for one binary (JSON under ``binary:<sha>``)."""
        self.set_metadata(f"binary:{binary_sha256}", metadata)

    def get_binary_metadata(self, binary_sha256: str) -> Dict[str, Any]:
        raw = self.get_metadata(f"binary:{binary_sha256}")
        return json.loads(raw) if raw else {}

    def invalidate_binary(self, binary_sha256: str) -> Dict[str, int]:
        """Delete every row belonging to ``binary_sha256``.

        Used before a (re-)analysis so a replay starts from a clean slate.
        Does not open its own transaction: wrap the call in
        :meth:`transaction` to make invalidation + rewrite atomic.
        """
        deleted: Dict[str, int] = {}
        for table in ("funcs", "edges", "ssa_facts", "receipts", "types", "versions", "trace_events"):
            cur = self._execute(f"DELETE FROM {table} WHERE binary_sha256 = ?", (binary_sha256,))
            deleted[table] = cur.rowcount or 0
        return deleted

    # -- funcs -------------------------------------------------------------

    def upsert_function(
        self,
        record: FunctionRecord,
        *,
        phase: str = "func_upsert",
        receipt_payload: Optional[Dict[str, Any]] = None,
        write_receipt: bool = True,
    ) -> int:
        """Insert or overwrite one function row (exactly-once semantics).

        Returns 1. When ``write_receipt`` is true the accompanying receipt is
        committed too, so the row is reproducible after a resume.
        """
        if not record.binary_sha256:
            raise StorageError("FunctionRecord.binary_sha256 is required")
        self._execute(_FUNC_UPSERT, record.to_row())
        if write_receipt:
            payload = receipt_payload if receipt_payload is not None else {
                "name": record.name,
                "size": int(record.size),
                "ir_pcode": bool(record.ir_pcode),
            }
            self.add_receipt(record.binary_sha256, record.func_addr, phase, payload)
        return 1

    def bulk_upsert_functions(
        self,
        records: Sequence[FunctionRecord],
        *,
        phase: str = "static_analysis",
        write_receipts: bool = True,
    ) -> int:
        """Batched upsert with the same semantics as :meth:`upsert_function`.

        Call inside :meth:`transaction` for atomic chunks.
        """
        rows = [r.to_row() for r in records]
        if not rows:
            return 0
        self._execute_many(_FUNC_UPSERT, rows)
        if write_receipts:
            now = _utc_now()
            out = []
            for rec in records:
                payload = {
                    "name": rec.name,
                    "size": int(rec.size),
                    "ir_pcode": bool(rec.ir_pcode),
                }
                payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
                out.append(
                    (
                        make_receipt_id(rec.binary_sha256, rec.func_addr, phase, payload_json),
                        rec.binary_sha256,
                        int(rec.func_addr),
                        phase,
                        now,
                        payload_json,
                    )
                )
            self._execute_many(_RECEIPT_UPSERT, out)
        return len(rows)

    def get_function(self, binary_sha256: str, func_addr: int) -> Optional[FunctionRecord]:
        row = self._execute(
            "SELECT * FROM funcs WHERE binary_sha256 = ? AND func_addr = ?",
            (binary_sha256, int(func_addr)),
        ).fetchone()
        return FunctionRecord.from_row(row) if row is not None else None

    def get_functions(self, binary_sha256: str) -> List[FunctionRecord]:
        rows = self._execute(
            "SELECT * FROM funcs WHERE binary_sha256 = ? ORDER BY func_addr",
            (binary_sha256,),
        ).fetchall()
        return [FunctionRecord.from_row(r) for r in rows]

    def count_functions(self, binary_sha256: str) -> int:
        row = self._execute(
            "SELECT COUNT(*) AS n FROM funcs WHERE binary_sha256 = ?", (binary_sha256,)
        ).fetchone()
        return int(row["n"])

    def delete_function(self, binary_sha256: str, func_addr: int) -> int:
        cur = self._execute(
            "DELETE FROM funcs WHERE binary_sha256 = ? AND func_addr = ?",
            (binary_sha256, int(func_addr)),
        )
        return cur.rowcount or 0

    # -- edges -------------------------------------------------------------

    def add_edge(
        self,
        binary_sha256: str,
        src_addr: int,
        dst_addr: int,
        kind: Union[EdgeKind, str] = EdgeKind.CALL,
    ) -> int:
        """Insert an edge; a duplicate for the same primary key is ignored."""
        cur = self._execute(
            _EDGE_UPSERT,
            (binary_sha256, int(src_addr), int(dst_addr), EdgeKind.coerce(kind).value),
        )
        return cur.rowcount or 0

    def add_edges(self, edges: Sequence[EdgeRecord]) -> int:
        if not edges:
            return 0
        return self._execute_many(_EDGE_UPSERT, [e.to_row() for e in edges])

    def get_edges(
        self,
        binary_sha256: str,
        *,
        kind: Optional[Union[EdgeKind, str]] = None,
        src_addr: Optional[int] = None,
        dst_addr: Optional[int] = None,
    ) -> List[EdgeRecord]:
        sql = "SELECT * FROM edges WHERE binary_sha256 = ?"
        params: List[Any] = [binary_sha256]
        if kind is not None:
            sql += " AND kind = ?"
            params.append(EdgeKind.coerce(kind).value)
        if src_addr is not None:
            sql += " AND src_addr = ?"
            params.append(int(src_addr))
        if dst_addr is not None:
            sql += " AND dst_addr = ?"
            params.append(int(dst_addr))
        sql += " ORDER BY src_addr, dst_addr, kind"
        rows = self._execute(sql, params).fetchall()
        return [
            EdgeRecord(
                binary_sha256=r["binary_sha256"],
                src_addr=int(r["src_addr"]),
                dst_addr=int(r["dst_addr"]),
                kind=EdgeKind(r["kind"]),
            )
            for r in rows
        ]

    def get_callees(self, binary_sha256: str, src_addr: int) -> List[int]:
        """Direct call targets of ``src_addr`` in ascending address order."""
        rows = self._execute(
            "SELECT DISTINCT dst_addr FROM edges "
            "WHERE binary_sha256 = ? AND src_addr = ? AND kind = 'call' "
            "ORDER BY dst_addr",
            (binary_sha256, int(src_addr)),
        ).fetchall()
        return [int(r["dst_addr"]) for r in rows]

    def get_callers(self, binary_sha256: str, dst_addr: int) -> List[int]:
        """Call sites that target ``dst_addr`` in ascending address order."""
        rows = self._execute(
            "SELECT DISTINCT src_addr FROM edges "
            "WHERE binary_sha256 = ? AND dst_addr = ? AND kind = 'call' "
            "ORDER BY src_addr",
            (binary_sha256, int(dst_addr)),
        ).fetchall()
        return [int(r["src_addr"]) for r in rows]

    def count_edges(self, binary_sha256: str) -> int:
        row = self._execute(
            "SELECT COUNT(*) AS n FROM edges WHERE binary_sha256 = ?", (binary_sha256,)
        ).fetchone()
        return int(row["n"])

    # -- ssa facts ---------------------------------------------------------

    def add_ssa_fact(self, fact: SsaFact) -> int:
        self._execute(_SSA_UPSERT, fact.to_row())
        return 1

    def add_ssa_facts(self, facts: Sequence[SsaFact]) -> int:
        if not facts:
            return 0
        return self._execute_many(_SSA_UPSERT, [f.to_row() for f in facts])

    def get_ssa_facts(self, binary_sha256: str, func_addr: int) -> List[Dict[str, Any]]:
        rows = self._execute(
            "SELECT * FROM ssa_facts WHERE binary_sha256 = ? AND func_addr = ? "
            "ORDER BY def_point, fact_id",
            (binary_sha256, int(func_addr)),
        ).fetchall()
        return [
            {
                "fact_id": r["fact_id"],
                "binary_sha256": r["binary_sha256"],
                "func_addr": int(r["func_addr"]),
                "var_name": r["var_name"],
                "ssa_reg": r["ssa_reg"],
                "def_point": int(r["def_point"]),
                "use_points": json.loads(r["use_points"]) if r["use_points"] else [],
            }
            for r in rows
        ]

    def count_ssa_facts(self, binary_sha256: Optional[str] = None) -> int:
        if binary_sha256 is None:
            return int(self._execute("SELECT COUNT(*) AS n FROM ssa_facts").fetchone()["n"])
        return int(
            self._execute(
                "SELECT COUNT(*) AS n FROM ssa_facts WHERE binary_sha256 = ?",
                (binary_sha256,),
            ).fetchone()["n"]
        )

    # -- types (lattice) ---------------------------------------------------

    def upsert_type(self, record: TypeRecord) -> int:
        self._execute(
            "INSERT INTO types (type_id, name, kind, size_bytes, layout_json, lattice_rank, binary_sha256) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (binary_sha256, type_id) DO UPDATE SET name = excluded.name, "
            "kind = excluded.kind, size_bytes = excluded.size_bytes, "
            "layout_json = excluded.layout_json, lattice_rank = excluded.lattice_rank, "
            "binary_sha256 = excluded.binary_sha256",
            record.to_row(),
        )
        return 1

    def get_type(
        self, type_id: str, binary_sha256: Optional[str] = None
    ) -> Optional[TypeRecord]:
        """One ``types`` row by ``type_id`` (scoped to one binary when given).

        ``type_id`` is unique only *within* a binary -- the table is keyed on
        ``(binary_sha256, type_id)`` -- so without ``binary_sha256`` the lowest
        binary sha wins.  Deterministic, but pass the sha when it matters.
        """
        if binary_sha256 is None:
            row = self._execute(
                "SELECT * FROM types WHERE type_id = ? ORDER BY binary_sha256 LIMIT 1",
                (type_id,),
            ).fetchone()
        else:
            row = self._execute(
                "SELECT * FROM types WHERE binary_sha256 = ? AND type_id = ?",
                (binary_sha256, type_id),
            ).fetchone()
        return TypeRecord.from_row(row) if row is not None else None

    def get_types(self, binary_sha256: Optional[str] = None) -> List[TypeRecord]:
        if binary_sha256 is None:
            rows = self._execute("SELECT * FROM types ORDER BY type_id").fetchall()
        else:
            rows = self._execute(
                "SELECT * FROM types WHERE binary_sha256 = ? ORDER BY type_id",
                (binary_sha256,),
            ).fetchall()
        return [TypeRecord.from_row(r) for r in rows]

    def count_types(self, binary_sha256: Optional[str] = None) -> int:
        if binary_sha256 is None:
            return int(self._execute("SELECT COUNT(*) AS n FROM types").fetchone()["n"])
        return int(
            self._execute(
                "SELECT COUNT(*) AS n FROM types WHERE binary_sha256 = ?",
                (binary_sha256,),
            ).fetchone()["n"]
        )

    # -- receipts ----------------------------------------------------------

    def add_receipt(
        self,
        binary_sha256: str,
        func_addr: int,
        phase: str,
        payload: Optional[Dict[str, Any]] = None,
        *,
        timestamp: Optional[str] = None,
        receipt_id: Optional[str] = None,
    ) -> str:
        """Commit one ``{func_id, phase, receipt}`` record.

        The receipt id is deterministic (:func:`make_receipt_id`), so replaying
        an analysis updates the row rather than appending a duplicate.
        """
        payload = payload if payload is not None else {}
        payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        rid = receipt_id or make_receipt_id(binary_sha256, func_addr, phase, payload_json)
        ts = timestamp or _utc_now()
        self._execute(
            _RECEIPT_UPSERT,
            (rid, binary_sha256, int(func_addr), phase, ts, payload_json),
        )
        return rid

    def get_receipts(
        self, binary_sha256: str, func_addr: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        if func_addr is None:
            rows = self._execute(
                "SELECT * FROM receipts WHERE binary_sha256 = ? ORDER BY timestamp, receipt_id",
                (binary_sha256,),
            ).fetchall()
        else:
            rows = self._execute(
                "SELECT * FROM receipts WHERE binary_sha256 = ? AND func_addr = ? "
                "ORDER BY timestamp, receipt_id",
                (binary_sha256, int(func_addr)),
            ).fetchall()
        return [self._receipt_row_to_dict(r) for r in rows]

    def insert_receipt(self, receipt: Any) -> str:
        """Insert a GroundReceipt or structured receipt record."""
        if hasattr(receipt, "to_row"):
            self._execute(_RECEIPT_UPSERT, receipt.to_row())
            return str(receipt.receipt_id)
        if isinstance(receipt, dict):
            return self.add_receipt(
                binary_sha256=receipt["binary_sha256"],
                func_addr=receipt["func_addr"],
                phase=receipt.get("phase", "ground"),
                payload=receipt.get("payload", receipt),
                timestamp=receipt.get("timestamp"),
                receipt_id=receipt.get("receipt_id"),
            )
        raise StorageError(f"Unsupported receipt object: {type(receipt)}")

    def count_receipts(self, binary_sha256: Optional[str] = None) -> int:
        if binary_sha256 is None:
            return int(self._execute("SELECT COUNT(*) AS n FROM receipts").fetchone()["n"])
        return int(
            self._execute(
                "SELECT COUNT(*) AS n FROM receipts WHERE binary_sha256 = ?",
                (binary_sha256,),
            ).fetchone()["n"]
        )

    def latest_receipt(self, binary_sha256: str, phase: str) -> Optional[Dict[str, Any]]:
        row = self._execute(
            "SELECT * FROM receipts WHERE binary_sha256 = ? AND phase = ? "
            "ORDER BY timestamp DESC LIMIT 1",
            (binary_sha256, phase),
        ).fetchone()
        return self._receipt_row_to_dict(row) if row is not None else None

    def has_receipt(self, binary_sha256: str, phase: str) -> bool:
        return self.latest_receipt(binary_sha256, phase) is not None

    @staticmethod
    def _receipt_row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "receipt_id": row["receipt_id"],
            "binary_sha256": row["binary_sha256"],
            "func_addr": int(row["func_addr"]),
            "phase": row["phase"],
            "timestamp": row["timestamp"],
            "payload": json.loads(row["payload_json"]) if row["payload_json"] else {},
        }

    # -- versions ----------------------------------------------------------

    def add_version(self, record: VersionRecord) -> int:
        self._execute(
            "INSERT INTO versions (binary_sha256, target_sha256, similarity_score, embedding_blob, offset_map_json) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT (binary_sha256, target_sha256) DO UPDATE SET "
            "similarity_score = excluded.similarity_score, "
            "embedding_blob = excluded.embedding_blob, "
            "offset_map_json = excluded.offset_map_json",
            record.to_row(),
        )
        return 1

    def get_versions(self, binary_sha256: str) -> List[VersionRecord]:
        rows = self._execute(
            "SELECT * FROM versions WHERE binary_sha256 = ? ORDER BY target_sha256",
            (binary_sha256,),
        ).fetchall()
        return [VersionRecord.from_row(r) for r in rows]

    def count_versions(self, binary_sha256: Optional[str] = None) -> int:
        if binary_sha256 is None:
            return int(self._execute("SELECT COUNT(*) AS n FROM versions").fetchone()["n"])
        return int(
            self._execute(
                "SELECT COUNT(*) AS n FROM versions WHERE binary_sha256 = ?",
                (binary_sha256,),
            ).fetchone()["n"]
        )

    # -- trace events (dynamic receipts) -----------------------------------

    def add_trace_event(
        self,
        binary_sha256: str,
        func_addr: int,
        event_type: str,
        payload: Optional[Dict[str, Any]] = None,
        *,
        timestamp: Optional[str] = None,
        event_id: Optional[str] = None,
    ) -> str:
        payload = payload if payload is not None else {}
        payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        ts = timestamp or _utc_now()
        # A trace is history, not state: two identical events (same binary,
        # function, type, payload, even timestamp) are still two events, so the
        # generated id carries a uuid4 nonce instead of hashing content alone.
        # A caller-supplied ``event_id`` stays deterministic and idempotent --
        # exactly what a kill-and-replay of the same event wants.
        eid = event_id or f"trac_{uuid4().hex}"
        self._execute(
            "INSERT INTO trace_events (event_id, binary_sha256, func_addr, timestamp, event_type, payload_json) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (event_id) DO UPDATE SET "
            "timestamp = excluded.timestamp, "
            "payload_json = excluded.payload_json",
            (eid, binary_sha256, int(func_addr), ts, event_type, payload_json),
        )
        return eid

    def get_trace_events(
        self, binary_sha256: str, func_addr: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        if func_addr is None:
            rows = self._execute(
                "SELECT * FROM trace_events WHERE binary_sha256 = ? ORDER BY timestamp, event_id",
                (binary_sha256,),
            ).fetchall()
        else:
            rows = self._execute(
                "SELECT * FROM trace_events WHERE binary_sha256 = ? AND func_addr = ? "
                "ORDER BY timestamp, event_id",
                (binary_sha256, int(func_addr)),
            ).fetchall()
        return [
            {
                "event_id": r["event_id"],
                "binary_sha256": r["binary_sha256"],
                "func_addr": int(r["func_addr"]),
                "timestamp": r["timestamp"],
                "event_type": r["event_type"],
                "payload": json.loads(r["payload_json"]) if r["payload_json"] else {},
            }
            for r in rows
        ]

    def count_trace_events(self, binary_sha256: Optional[str] = None) -> int:
        if binary_sha256 is None:
            return int(self._execute("SELECT COUNT(*) AS n FROM trace_events").fetchone()["n"])
        return int(
            self._execute(
                "SELECT COUNT(*) AS n FROM trace_events WHERE binary_sha256 = ?",
                (binary_sha256,),
            ).fetchone()["n"]
        )

    # -- aggregate stats ---------------------------------------------------

    def journal_mode(self) -> str:
        return str(self._execute("PRAGMA journal_mode").fetchone()[0]).lower()

    def integrity_check(self) -> List[str]:
        """Run SQLite PRAGMA integrity_check and return findings ('ok' if healthy)."""
        rows = self._execute("PRAGMA integrity_check").fetchall()
        return [str(r[0]) for r in rows]

    def table_names(self) -> List[str]:
        rows = self._execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        return [r["name"] for r in rows]

    def column_names(self, table: str) -> List[str]:
        if table not in self.table_names():
            raise StorageError(f"unknown table {table!r}")
        rows = self._execute(f"PRAGMA table_info({table})").fetchall()
        return [r["name"] for r in rows]

    def stats(self, binary_sha256: Optional[str] = None) -> Dict[str, Any]:
        """DB statistics for the ``toothpick static`` summary JSON."""
        funcs = (
            int(self._execute("SELECT COUNT(*) AS n FROM funcs").fetchone()["n"])
            if binary_sha256 is None
            else self.count_functions(binary_sha256)
        )
        edges = (
            int(self._execute("SELECT COUNT(*) AS n FROM edges").fetchone()["n"])
            if binary_sha256 is None
            else self.count_edges(binary_sha256)
        )
        try:
            size_bytes = self.db_path.stat().st_size
        except OSError:  # pragma: no cover - path disappeared
            size_bytes = 0
        return {
            "db_path": str(self.db_path),
            "journal_mode": self.journal_mode(),
            "schema_version": self.get_metadata("schema_version"),
            "size_bytes": size_bytes,
            "func_count": funcs,
            "edge_count": edges,
            "ssa_fact_count": self.count_ssa_facts(),
            "type_count": self.count_types(),
            "receipt_count": self.count_receipts(),
            "version_count": self.count_versions(),
            "trace_event_count": self.count_trace_events(),
            "binary_count": int(
                self._execute(
                    "SELECT COUNT(DISTINCT binary_sha256) AS n FROM funcs"
                ).fetchone()["n"]
            ),
        }
