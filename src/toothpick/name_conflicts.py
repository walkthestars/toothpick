"""Toothpick pending/resolved name-conflict queue (grounded names are protected).

Binding contract: a function name recovered from bytes (a "grounded" name) may
never be silently overwritten by an IDA hint or by any other ungrounded rename.
Every such attempt is either a silent *fill* (the existing name was empty or a
synthetic placeholder) or a **pending conflict record** the analyst resolves
later.  The queue is deliberately small and stdlib-only:

* :func:`ensure_schema` owns this module's DDL (``name_conflicts``) and mirrors
  the minimal ``funcs`` DDL -- the same precedent as
  ``profile.MINIMAL_FUNCS_DDL`` / ``jvm.MINIMAL_FUNCS_DDL`` -- so a standalone
  db path can be grounded without importing private ``storage`` internals.  It
  is idempotent (``CREATE TABLE IF NOT EXISTS``) and it never migrates, stamps,
  or rewrites a row of an existing ``toothpick.db``.
* :func:`report_conflict` is the single write path.  Its outcomes are explicit
  (:class:`ConflictOutcome`): ``filled`` (no record), ``pending`` (recorded),
  ``unchanged``, ``ignored`` (the proposal is not a grounded name), and
  ``already_resolved``.  It never changes a grounded ``funcs.name``; only
  :func:`resolve_conflict` with ``accept_proposed`` does, and only after the
  recorded existing name still matches the row (so a resolution can never be
  applied on top of a name that drifted out of band).
* Conflict ids are deterministic on ``(binary_sha256, func_addr, existing,
  proposed)`` (:func:`conflict_id`), so re-reporting the same pair -- the
  ingest-ida consumer will replay a lot -- converges on one pending row instead
  of accumulating duplicates.
* Resolved rows are terminal: they are never re-opened, never re-inserted, and
  a second :func:`resolve_conflict` on one is an error.
* Reads (:func:`list_pending`, :func:`list_resolved`, :func:`list_conflicts`,
  :func:`counts`) are the consumer surface for ``ingest-ida``; they never create
  a file and they raise :class:`NameConflictError` when the store has no
  ``name_conflicts`` table, so a caller cannot mistake "no queue" for "no
  conflicts".

Trap hygiene: this module never imports ``faulthandler`` (house trap 11) and
every SQLite handle it opens is closed in a ``finally`` (house trap 7 -- a
leaked WAL handle surfaces on Windows teardown as ``PermissionError``).
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

from toothpick.storage import Database

__all__ = [
    "MINIMAL_FUNCS_DDL",
    "NAME_CONFLICTS_DDL",
    "NAME_CONFLICTS_INDEXES",
    "ConflictOutcome",
    "ConflictStatus",
    "NameConflict",
    "NameConflictError",
    "ReportResult",
    "Resolution",
    "TargetLike",
    "conflict_id",
    "counts",
    "ensure_schema",
    "is_grounded_name",
    "list_conflicts",
    "list_pending",
    "list_resolved",
    "report_conflict",
    "resolve_conflict",
]

PathLike = Union[str, Path]

#: Anything a caller may hand to this module: a :class:`Database`, a raw
#: :class:`sqlite3.Connection` (whose isolation/row-factory are never touched),
#: or a db path.
TargetLike = Union[Database, sqlite3.Connection, PathLike]

#: Conflict ids look like ``nc_<40 hex>``: long enough to be collision-safe,
#: short enough to read in a log line.
_ID_PREFIX = "nc_"
_ID_DIGEST_CHARS = 40

#: ``funcs`` DDL, mirrored from ``toothpick.storage`` (``_SCHEMA['funcs']``,
#: trimmed to the columns this module needs) so a standalone store can be
#: grounded without importing private storage internals.  When the store was
#: created by :class:`toothpick.storage.Database` this statement is a no-op --
#: ``CREATE TABLE IF NOT EXISTS`` never rewrites an existing table.
MINIMAL_FUNCS_DDL = """
CREATE TABLE IF NOT EXISTS funcs (
    binary_sha256 TEXT    NOT NULL,
    func_addr     INTEGER NOT NULL,
    name          TEXT    NOT NULL DEFAULT '',
    PRIMARY KEY (binary_sha256, func_addr)
)
"""

#: The one table this module owns (acceptance criterion 1).  ``status`` is
#: ``pending`` or ``resolved``; ``resolution`` is NULL until the analyst picks
#: ``keep_existing`` or ``accept_proposed``.
NAME_CONFLICTS_DDL = """
CREATE TABLE IF NOT EXISTS name_conflicts (
    conflict_id    TEXT PRIMARY KEY,
    binary_sha256  TEXT    NOT NULL,
    func_addr      INTEGER NOT NULL,
    existing_name  TEXT    NOT NULL,
    proposed_name  TEXT    NOT NULL,
    source         TEXT    NOT NULL DEFAULT '',
    status         TEXT    NOT NULL DEFAULT 'pending',
    resolution     TEXT,
    created_at     TEXT    NOT NULL,
    resolved_at    TEXT,
    resolved_by    TEXT    NOT NULL DEFAULT ''
)
"""

NAME_CONFLICTS_INDEXES: Tuple[str, ...] = (
    # The ingest-ida read: "everything still pending, per binary, in address order".
    "CREATE INDEX IF NOT EXISTS idx_name_conflicts_pending "
    "ON name_conflicts (status, binary_sha256, func_addr)",
    # The drift check: does this (binary, func) already have a queue entry?
    "CREATE INDEX IF NOT EXISTS idx_name_conflicts_func "
    "ON name_conflicts (binary_sha256, func_addr)",
)

#: Column list applied to every ``SELECT`` so rows bind positionally (a
#: caller-supplied connection keeps its own ``row_factory``).
_CONFLICT_COLUMNS = (
    "conflict_id, binary_sha256, func_addr, existing_name, proposed_name, "
    "source, status, resolution, created_at, resolved_at, resolved_by"
)

_INSERT_CONFLICT = (
    "INSERT INTO name_conflicts "
    f"({_CONFLICT_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
    "ON CONFLICT (conflict_id) DO NOTHING"
)

_SELECT_CONFLICT = f"SELECT {_CONFLICT_COLUMNS} FROM name_conflicts"

#: Fill-in / insert for a name that was empty or synthetic.  ``DO UPDATE SET
#: name`` only touches ``name``, so size/prototype/decompilation of an existing
#: row are preserved.
_UPSERT_FUNC_NAME = (
    "INSERT INTO funcs (binary_sha256, func_addr, name) VALUES (?, ?, ?) "
    "ON CONFLICT (binary_sha256, func_addr) DO UPDATE SET name = excluded.name"
)

_RESOLVE_CONFLICT = (
    "UPDATE name_conflicts SET status = 'resolved', resolution = ?, "
    "resolved_at = ?, resolved_by = ? WHERE conflict_id = ?"
)

#: Placeholder-name taxonomy, byte-identical in behaviour to
#: ``rebase._is_synthetic_name`` (that module imports the whole static engine,
#: so the taxonomy is mirrored here rather than imported) plus the empty case.
_SYNTHETIC_PREFIXES: Tuple[str, ...] = ("sub_", "func_", "m_", "0x")
_SYNTHETIC_NAMES: Tuple[str, ...] = ("targetclass", "targetmethod", "unnamed")


class NameConflictError(Exception):
    """Raised for a malformed call, a missing table, or an illegal transition."""


class ConflictStatus(str, Enum):
    """Lifecycle of one queue row.  ``RESOLVED`` is terminal."""

    PENDING = "pending"
    RESOLVED = "resolved"

    @classmethod
    def coerce(cls, value: Union["ConflictStatus", str, None]) -> "ConflictStatus":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError as exc:
            raise NameConflictError(
                f"invalid conflict status {value!r}; expected one of "
                f"{[member.value for member in cls]}"
            ) from exc


class Resolution(str, Enum):
    """The analyst's decision on a pending conflict."""

    KEEP_EXISTING = "keep_existing"
    ACCEPT_PROPOSED = "accept_proposed"

    @classmethod
    def coerce(cls, value: Union["Resolution", str]) -> "Resolution":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError as exc:
            raise NameConflictError(
                f"invalid resolution {value!r}; expected one of "
                f"{[member.value for member in cls]}"
            ) from exc


class ConflictOutcome(str, Enum):
    """What :func:`report_conflict` did with one proposal.

    ``FILLED``   -- the existing name was empty or synthetic; it was filled in
                    and *no* queue row was written (acceptance criterion 2).
    ``PENDING``  -- a grounded, differing name is at stake; a pending row exists
                    (created now, or already there from an earlier report).
    ``UNCHANGED``-- the proposal equals the name already in place; nothing to do.
    ``ALREADY_RESOLVED`` -- this exact pair was resolved earlier; terminal, so
                    the earlier resolution stands.
    ``IGNORED``  -- the proposal is not a grounded name, so it may neither
                    overwrite a grounded name nor fill an empty one.
    """

    FILLED = "filled"
    PENDING = "pending"
    UNCHANGED = "unchanged"
    ALREADY_RESOLVED = "already_resolved"
    IGNORED = "ignored"


@dataclass(frozen=True)
class NameConflict:
    """One ``name_conflicts`` row."""

    conflict_id: str
    binary_sha256: str
    func_addr: int
    existing_name: str
    proposed_name: str
    source: str = ""
    status: ConflictStatus = ConflictStatus.PENDING
    resolution: Optional[Resolution] = None
    created_at: str = ""
    resolved_at: Optional[str] = None
    resolved_by: str = ""

    @property
    def pending(self) -> bool:
        return self.status is ConflictStatus.PENDING

    @property
    def resolved(self) -> bool:
        return self.status is ConflictStatus.RESOLVED

    @classmethod
    def from_row(cls, row: Sequence[Any]) -> "NameConflict":
        raw_resolution = row[7]
        return cls(
            conflict_id=str(row[0]),
            binary_sha256=str(row[1]),
            func_addr=int(row[2]),
            existing_name=str(row[3] or ""),
            proposed_name=str(row[4] or ""),
            source=str(row[5] or ""),
            status=ConflictStatus.coerce(row[6]),
            resolution=(
                None if raw_resolution in (None, "") else Resolution.coerce(raw_resolution)
            ),
            created_at=str(row[8] or ""),
            resolved_at=(None if row[9] in (None, "") else str(row[9])),
            resolved_by=str(row[10] or ""),
        )


@dataclass(frozen=True)
class ReportResult:
    """Outcome of :func:`report_conflict` for one proposal.

    ``name`` is the grounded name *in effect after the call*: the filled-in
    proposed name for ``FILLED``, the untouched existing name for
    ``PENDING``/``ALREADY_RESOLVED``/``UNCHANGED``, and the existing name (or
    the empty string) for ``IGNORED``.
    """

    outcome: ConflictOutcome
    binary_sha256: str
    func_addr: int
    existing_name: str
    proposed_name: str
    name: str
    conflict: Optional[NameConflict] = None
    conflict_id: Optional[str] = None
    recorded: bool = False


def is_grounded_name(name: Any) -> bool:
    """True when ``name`` carries information (a real symbol or analyst label).

    Empty, whitespace-only, and synthetic placeholder names (``sub_140001000``,
    ``func_1``, ``m_0``, ``0x140001000``, ``unnamed``, ``TargetClass``, ...) are
    *not* grounded, so they may be filled in silently -- and a proposal that is
    not grounded may never overwrite or fill anything.
    """
    if name is None:
        return False
    text = str(name).strip()
    if not text:
        return False
    low = text.lower()
    if low.startswith(_SYNTHETIC_PREFIXES):
        return False
    if low in _SYNTHETIC_NAMES:
        return False
    return True


def conflict_id(
    binary_sha256: str, func_addr: int, existing_name: Any, proposed_name: Any
) -> str:
    """Deterministic id for one ``(binary, func, existing, proposed)`` tuple.

    Determinism is deliberate (acceptance criterion 3): a replayed IDA ingest of
    the same hint updates nothing and appends nothing, so the queue stays a set
    of distinct disagreements rather than a log of attempts.
    """
    payload = (
        f"{binary_sha256}|{int(func_addr)}|"
        f"{_name_text(existing_name)}|{_name_text(proposed_name)}"
    )
    return _ID_PREFIX + sha256(payload.encode("utf-8")).hexdigest()[:_ID_DIGEST_CHARS]


# --------------------------------------------------------------------------- #
# validation helpers
# --------------------------------------------------------------------------- #


def _require_sha(sha: Any) -> str:
    if isinstance(sha, bool) or not isinstance(sha, str) or not sha.strip():
        raise NameConflictError(f"binary_sha256 must be a non-empty string, got {sha!r}")
    return sha.strip()


def _require_addr(func_addr: Any) -> int:
    if isinstance(func_addr, bool) or not isinstance(func_addr, int):
        raise NameConflictError(f"func_addr must be an int, got {func_addr!r}")
    if func_addr < 0:
        raise NameConflictError(f"func_addr must not be negative, got {func_addr!r}")
    return int(func_addr)


def _require_proposed(name: Any) -> str:
    if name is None or not str(name).strip():
        raise NameConflictError("proposed_name must be a non-empty string")
    return str(name).strip()


def _name_text(name: Any) -> str:
    return "" if name is None else str(name).strip()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


# --------------------------------------------------------------------------- #
# connection discipline (every handle closed in ``finally``)
# --------------------------------------------------------------------------- #


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
    ).fetchone()
    return row is not None


@contextmanager
def _read_connection(target: TargetLike) -> Iterator[sqlite3.Connection]:
    """Yield a read connection for ``name_conflicts``; never creates a file."""
    if isinstance(target, Database):
        with target.connection() as conn:
            if not _table_exists(conn, "name_conflicts"):
                raise NameConflictError(
                    "name_conflicts table is missing; call "
                    "name_conflicts.ensure_schema() first"
                )
            yield conn
        return
    if isinstance(target, sqlite3.Connection):
        if not _table_exists(target, "name_conflicts"):
            raise NameConflictError(
                "name_conflicts table is missing; call "
                "name_conflicts.ensure_schema() first"
            )
        yield target
        return
    path = Path(target)
    if not path.exists():
        raise NameConflictError(f"database not found: {path}")
    conn = sqlite3.connect(str(path), isolation_level=None)
    try:
        conn.execute("PRAGMA busy_timeout = 10000")
        if not _table_exists(conn, "name_conflicts"):
            raise NameConflictError(
                f"{path} has no name_conflicts table; call "
                "name_conflicts.ensure_schema() first"
            )
        yield conn
    finally:
        conn.close()


@contextmanager
def _write_connection(
    target: TargetLike, *, create: bool = False
) -> Iterator[sqlite3.Connection]:
    """Yield a write connection inside a transaction; always released.

    A ``Database`` contributes its own ``transaction()`` (so callers get the
    engine's locking and receipt discipline); a raw connection only gets a
    ``BEGIN IMMEDIATE`` when it owns no transaction of its own; a path is opened
    WAL-mode and committed/rolled back here.
    """
    if isinstance(target, Database):
        with target.transaction() as conn:
            yield conn
        return
    if isinstance(target, sqlite3.Connection):
        conn = target
        began = False
        if conn.isolation_level is None and not conn.in_transaction:
            conn.execute("BEGIN IMMEDIATE")
            began = True
        try:
            yield conn
        except BaseException:
            if began and conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        else:
            if began and conn.in_transaction:
                conn.execute("COMMIT")
        return
    path = Path(target)
    if not path.exists() and not create:
        raise NameConflictError(f"database not found: {path}")
    parent = str(path.parent)
    if parent not in ("", "."):
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), isolation_level=None)
    try:
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
        raise NameConflictError(f"name-conflict write failed for {path}: {exc}") from exc
    finally:
        conn.close()


def _require_funcs(conn: sqlite3.Connection) -> None:
    if not _table_exists(conn, "funcs"):
        raise NameConflictError(
            "funcs table is missing; call name_conflicts.ensure_schema() first"
        )


# --------------------------------------------------------------------------- #
# schema
# --------------------------------------------------------------------------- #


def ensure_schema(target: TargetLike) -> None:
    """Create ``name_conflicts`` (+ indexes) and the minimal ``funcs`` (idempotent).

    Accepts a :class:`Database`, a raw :class:`sqlite3.Connection`, or a db path
    (a path is created when missing -- "ensure" is explicit about that).  This is
    the only DDL this module executes: an existing ``toothpick.db`` keeps its
    schema version, its rows, and its storage-owned tables exactly as they were.
    """
    with _write_connection(target, create=True) as conn:
        conn.execute(MINIMAL_FUNCS_DDL)
        conn.execute(NAME_CONFLICTS_DDL)
        for index in NAME_CONFLICTS_INDEXES:
            conn.execute(index)


# --------------------------------------------------------------------------- #
# write path
# --------------------------------------------------------------------------- #


def _func_name(conn: sqlite3.Connection, sha: str, func_addr: int) -> Optional[str]:
    row = conn.execute(
        "SELECT name FROM funcs WHERE binary_sha256 = ? AND func_addr = ?",
        (sha, func_addr),
    ).fetchone()
    return None if row is None else str(row[0] or "")


def _select_conflict(conn: sqlite3.Connection, cid: str) -> Optional[NameConflict]:
    row = conn.execute(f"{_SELECT_CONFLICT} WHERE conflict_id = ?", (cid,)).fetchone()
    return None if row is None else NameConflict.from_row(row)


def report_conflict(
    target: TargetLike,
    binary_sha256: str,
    func_addr: int,
    proposed_name: str,
    *,
    source: str = "ida-hint",
) -> ReportResult:
    """Queue (or refuse) one ungrounded rename proposal for one function.

    Never writes ``funcs.name`` when the incumbent name is grounded; never
    records anything when the incumbent name is empty or synthetic (that is a
    silent fill, criterion 2); never duplicates a pending row (criterion 3).
    """
    sha = _require_sha(binary_sha256)
    addr = _require_addr(func_addr)
    proposed = _require_proposed(proposed_name)
    origin = _name_text(source)

    with _write_connection(target) as conn:
        _require_funcs(conn)
        if not _table_exists(conn, "name_conflicts"):
            raise NameConflictError(
                "name_conflicts table is missing; call "
                "name_conflicts.ensure_schema() first"
            )

        existing = _func_name(conn, sha, addr)

        # (a) an ungrounded proposal is not a rename at all.
        if not is_grounded_name(proposed):
            return ReportResult(
                outcome=ConflictOutcome.IGNORED,
                binary_sha256=sha,
                func_addr=addr,
                existing_name=existing or "",
                proposed_name=proposed,
                name=existing or "",
            )

        # (b) nothing to do when the name already in place is the proposal.
        if existing is not None and existing.strip() == proposed:
            return ReportResult(
                outcome=ConflictOutcome.UNCHANGED,
                binary_sha256=sha,
                func_addr=addr,
                existing_name=existing,
                proposed_name=proposed,
                name=existing,
            )

        # (c) empty / synthetic incumbent: fill silently, record nothing.
        if existing is None or not is_grounded_name(existing):
            conn.execute(_UPSERT_FUNC_NAME, (sha, addr, proposed))
            return ReportResult(
                outcome=ConflictOutcome.FILLED,
                binary_sha256=sha,
                func_addr=addr,
                existing_name=existing or "",
                proposed_name=proposed,
                name=proposed,
            )

        # (d) grounded incumbent vs a different grounded proposal: the queue.
        cid = conflict_id(sha, addr, existing, proposed)
        settled = _select_conflict(conn, cid)
        if settled is not None and settled.resolved:
            return ReportResult(
                outcome=ConflictOutcome.ALREADY_RESOLVED,
                binary_sha256=sha,
                func_addr=addr,
                existing_name=existing,
                proposed_name=proposed,
                name=existing,
                conflict=settled,
                conflict_id=cid,
            )
        if settled is not None:
            return ReportResult(
                outcome=ConflictOutcome.PENDING,
                binary_sha256=sha,
                func_addr=addr,
                existing_name=existing,
                proposed_name=proposed,
                name=existing,
                conflict=settled,
                conflict_id=cid,
            )

        created_at = _utc_now()
        conn.execute(
            _INSERT_CONFLICT,
            (
                cid,
                sha,
                addr,
                existing,
                proposed,
                origin,
                ConflictStatus.PENDING.value,
                None,
                created_at,
                None,
                "",
            ),
        )
        stored = _select_conflict(conn, cid)
        assert stored is not None  # the row was just written in this transaction
        return ReportResult(
            outcome=ConflictOutcome.PENDING,
            binary_sha256=sha,
            func_addr=addr,
            existing_name=existing,
            proposed_name=proposed,
            name=existing,
            conflict=stored,
            conflict_id=cid,
            recorded=True,
        )


def resolve_conflict(
    target: TargetLike,
    conflict_id_value: str,
    resolution: Union[Resolution, str],
    *,
    resolved_by: str = "",
) -> NameConflict:
    """Transition one pending conflict to resolved (terminal).

    ``keep_existing`` leaves ``funcs.name`` alone; ``accept_proposed`` writes the
    proposed name, but only while the row still holds the name recorded in the
    conflict -- otherwise the resolution is refused and rolled back, so a stale
    decision can never clobber a name that was grounded in the meantime.
    """
    cid = _name_text(conflict_id_value)
    if not cid:
        raise NameConflictError("conflict_id must be a non-empty string")
    decision = Resolution.coerce(resolution)
    actor = _name_text(resolved_by)

    with _write_connection(target) as conn:
        _require_funcs(conn)
        current = _select_conflict(conn, cid)
        if current is None:
            raise NameConflictError(f"unknown conflict id {cid!r}")
        if current.resolved:
            raise NameConflictError(
                f"conflict {cid} is already resolved as "
                f"{current.resolution.value if current.resolution else '?'} "
                "(resolved rows are terminal)"
            )
        if decision is Resolution.ACCEPT_PROPOSED:
            live = _func_name(conn, current.binary_sha256, current.func_addr)
            if live is None or live != current.existing_name:
                raise NameConflictError(
                    f"conflict {cid} is stale: funcs.name for "
                    f"{current.binary_sha256[:12]}..@0x{current.func_addr:x} is "
                    f"{(live if live is not None else '<no row>')!r}, expected "
                    f"{current.existing_name!r}"
                )
            conn.execute(
                _UPSERT_FUNC_NAME,
                (current.binary_sha256, current.func_addr, current.proposed_name),
            )
        conn.execute(
            _RESOLVE_CONFLICT, (decision.value, _utc_now(), actor, cid)
        )
        resolved = _select_conflict(conn, cid)
        assert resolved is not None
        return resolved


# --------------------------------------------------------------------------- #
# read surface
# --------------------------------------------------------------------------- #


def _normalize_limit(limit: Optional[int]) -> Optional[int]:
    if limit is None:
        return None
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise NameConflictError(f"limit must be an int or None, got {limit!r}")
    if limit < 0:
        raise NameConflictError(f"limit must not be negative, got {limit!r}")
    return int(limit)


def list_conflicts(
    target: TargetLike,
    *,
    status: Union[ConflictStatus, str, None] = None,
    binary_sha256: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[NameConflict]:
    """Queue rows, oldest-first per ``(binary, address, id)``, with optional filters."""
    where: List[str] = []
    params: List[Any] = []
    if status is not None:
        where.append("status = ?")
        params.append(ConflictStatus.coerce(status).value)
    if binary_sha256 is not None:
        where.append("binary_sha256 = ?")
        params.append(_require_sha(binary_sha256))
    sql = (
        f"{_SELECT_CONFLICT}"
        + (f" WHERE {' AND '.join(where)}" if where else "")
        + " ORDER BY binary_sha256 ASC, func_addr ASC, conflict_id ASC"
    )
    bound = _normalize_limit(limit)
    if bound is not None:
        sql += " LIMIT ?"
        params.append(bound)
    with _read_connection(target) as conn:
        rows = conn.execute(sql, tuple(params)).fetchall()
    return [NameConflict.from_row(row) for row in rows]


def list_pending(
    target: TargetLike,
    *,
    binary_sha256: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[NameConflict]:
    """Unresolved conflicts -- the ingest-ida consumer's work list."""
    return list_conflicts(
        target,
        status=ConflictStatus.PENDING,
        binary_sha256=binary_sha256,
        limit=limit,
    )


def list_resolved(
    target: TargetLike,
    *,
    binary_sha256: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[NameConflict]:
    """Settled conflicts, kept as the audit trail of what was decided."""
    return list_conflicts(
        target,
        status=ConflictStatus.RESOLVED,
        binary_sha256=binary_sha256,
        limit=limit,
    )


def counts(target: TargetLike, *, binary_sha256: Optional[str] = None) -> Dict[str, int]:
    """``{"pending": n, "resolved": n, "total": n}`` for the queue (bounded)."""
    where = ""
    params: List[Any] = []
    if binary_sha256 is not None:
        where = " WHERE binary_sha256 = ?"
        params.append(_require_sha(binary_sha256))
    sql = (
        "SELECT status, COUNT(*) FROM name_conflicts"
        + where
        + " GROUP BY status"
    )
    with _read_connection(target) as conn:
        rows = conn.execute(sql, tuple(params)).fetchall()
    tally = {
        ConflictStatus.PENDING.value: 0,
        ConflictStatus.RESOLVED.value: 0,
    }
    for row in rows:
        key = str(row[0])
        tally[key] = tally.get(key, 0) + int(row[1])
    total = sum(tally.values())
    return {
        "pending": tally[ConflictStatus.PENDING.value],
        "resolved": tally[ConflictStatus.RESOLVED.value],
        "total": total,
    }
