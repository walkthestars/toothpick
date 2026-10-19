"""Failing-first contract tests for :mod:`toothpick.name_conflicts`.

Written before the module existed, from the GOAL/ACCEPTANCE text alone, so the
expected values below are the spec, not a description of the implementation:

* criterion (2) -- an empty or synthetic ``sub_*`` existing name is filled
  silently (no record); a differing *real* name creates a pending record and the
  existing ``funcs`` row keeps its name;
* criterion (3) -- conflict ids are deterministic on
  ``(binary_sha256, func_addr, existing, proposed)`` and re-reporting the same
  tuple never duplicates a pending row;
* criterion (4) -- ``resolve`` moves pending -> resolved
  (``keep_existing`` / ``accept_proposed``) and resolved rows are terminal;
* criterion (5) -- ``list_pending`` / ``list_resolved`` / ``list_conflicts`` /
  ``counts`` are the read surface a future ``ingest-ida`` consumer calls.

House traps respected: no ``faulthandler`` import, every SQLite handle closed
in a ``finally`` (Windows WAL teardown), stdlib-only.
"""

from __future__ import annotations

import ast
import sqlite3
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

import pytest

from toothpick.name_conflicts import (
    MINIMAL_FUNCS_DDL,
    NAME_CONFLICTS_DDL,
    ConflictOutcome,
    ConflictStatus,
    NameConflictError,
    Resolution,
    conflict_id,
    counts,
    ensure_schema,
    is_grounded_name,
    list_conflicts,
    list_pending,
    list_resolved,
    report_conflict,
    resolve_conflict,
)
from toothpick.storage import SCHEMA_VERSION, Database, FunctionRecord

SHA_A = "a" * 64
SHA_B = "b" * 64

ADDR_ONE = 0x140001000
ADDR_TWO = 0x140002000
ADDR_THREE = 0x140003000

GROUNDED = "player_update"
GROUNDED_TWO = "player_draw"
HINT = "ida_hint_name"
HINT_TWO = "ida_hint_other"

CONFLICT_COLUMNS = {
    "conflict_id",
    "binary_sha256",
    "func_addr",
    "existing_name",
    "proposed_name",
    "source",
    "status",
    "resolution",
    "created_at",
    "resolved_at",
    "resolved_by",
}


# --------------------------------------------------------------------------- #
# helpers (each opens and closes its own handle)
def _store(
    tmp_path: Path,
    rows: Sequence[Tuple[str, int, str]] = (),
    *,
    db_name: str = "toothpick.db",
) -> Path:
    """``<db_name>`` holding the module schema and the given ``funcs`` rows."""
    path = tmp_path / db_name
    ensure_schema(path)
    if rows:
        conn = sqlite3.connect(str(path), isolation_level=None)
        try:
            conn.executemany(
                "INSERT INTO funcs (binary_sha256, func_addr, name) VALUES (?, ?, ?)",
                list(rows),
            )
        finally:
            conn.close()
    return path


def _func_name(path: Path, sha: str, addr: int) -> Optional[str]:
    conn = sqlite3.connect(str(path), isolation_level=None)
    try:
        row = conn.execute(
            "SELECT name FROM funcs WHERE binary_sha256 = ? AND func_addr = ?",
            (sha, addr),
        ).fetchone()
        return None if row is None else str(row[0])
    finally:
        conn.close()


def _set_func_name(path: Path, sha: str, addr: int, name: str) -> None:
    """An out-of-band grounding write, to prove the queue notices drift."""
    conn = sqlite3.connect(str(path), isolation_level=None)
    try:
        conn.execute(
            "UPDATE funcs SET name = ? WHERE binary_sha256 = ? AND func_addr = ?",
            (name, sha, addr),
        )
    finally:
        conn.close()


def _conflict_row_count(path: Path) -> int:
    conn = sqlite3.connect(str(path), isolation_level=None)
    try:
        return int(conn.execute("SELECT COUNT(*) FROM name_conflicts").fetchone()[0])
    finally:
        conn.close()


def _tables(path: Path) -> set[str]:
    conn = sqlite3.connect(str(path), isolation_level=None)
    try:
        return {
            str(row[0])
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# (1) the module owns its own idempotent DDL
# --------------------------------------------------------------------------- #


def test_ensure_schema_owns_its_ddl_and_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "toothpick.db"
    ensure_schema(path)
    ensure_schema(path)  # idempotent: a second call must not raise or duplicate

    assert "CREATE TABLE IF NOT EXISTS name_conflicts" in NAME_CONFLICTS_DDL
    assert "CREATE TABLE IF NOT EXISTS funcs" in MINIMAL_FUNCS_DDL

    tables = _tables(path)
    assert "name_conflicts" in tables
    # ``funcs`` is mirrored (never migrated) so a standalone store can be grounded;
    # the same precedent as ``profile.MINIMAL_FUNCS_DDL`` / ``jvm.MINIMAL_FUNCS_DDL``.
    assert "funcs" in tables

    conn = sqlite3.connect(str(path), isolation_level=None)
    try:
        columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(name_conflicts)")}
        indexes = {
            str(row[1])
            for row in conn.execute("PRAGMA index_list(name_conflicts)")
        }
    finally:
        conn.close()
    assert columns == CONFLICT_COLUMNS
    assert any(name.startswith("idx_name_conflicts") for name in indexes)

    assert counts(path) == {"pending": 0, "resolved": 0, "total": 0}


def test_ensure_schema_does_not_touch_existing_storage_bookkeeping(tmp_path: Path) -> None:
    path = tmp_path / "toothpick.db"
    db = Database(path)
    try:
        ensure_schema(db)
        db.upsert_function(
            FunctionRecord(binary_sha256=SHA_A, func_addr=ADDR_ONE, name=GROUNDED, size=32)
        )
        with db.connection() as conn:
            stamped = conn.execute(
                "SELECT value FROM metadata WHERE key = 'schema_version'"
            ).fetchone()[0]
        result = report_conflict(db, SHA_A, ADDR_ONE, HINT, source="ida-hint")
        assert result.outcome is ConflictOutcome.PENDING
    finally:
        db.close()

    assert stamped == SCHEMA_VERSION
    assert _func_name(path, SHA_A, ADDR_ONE) == GROUNDED
    assert {
        "metadata",
        "funcs",
        "edges",
        "ssa_facts",
        "types",
        "receipts",
        "versions",
        "trace_events",
        "name_conflicts",
    } <= _tables(path)


def test_reads_refuse_a_store_without_the_module_ddl(tmp_path: Path) -> None:
    plain = tmp_path / "plain.db"
    conn = sqlite3.connect(str(plain), isolation_level=None)
    try:
        conn.execute(MINIMAL_FUNCS_DDL)
    finally:
        conn.close()

    with pytest.raises(NameConflictError):
        counts(plain)
    with pytest.raises(NameConflictError):
        list_pending(plain)

    missing = tmp_path / "missing.db"
    with pytest.raises(NameConflictError):
        counts(missing)
    assert not missing.exists()  # a read never creates the file


# --------------------------------------------------------------------------- #
# (2) silent fill vs pending record
# --------------------------------------------------------------------------- #


def test_empty_or_synthetic_existing_names_fill_silently(tmp_path: Path) -> None:
    path = _store(
        tmp_path,
        [
            (SHA_A, ADDR_ONE, "sub_140001000"),
            (SHA_A, ADDR_TWO, ""),
        ],
    )

    synthetic = report_conflict(path, SHA_A, ADDR_ONE, GROUNDED, source="ida-hint")
    assert synthetic.outcome is ConflictOutcome.FILLED
    assert synthetic.name == GROUNDED
    assert synthetic.conflict is None
    assert synthetic.conflict_id is None
    assert synthetic.recorded is False
    assert _func_name(path, SHA_A, ADDR_ONE) == GROUNDED

    empty = report_conflict(path, SHA_A, ADDR_TWO, GROUNDED_TWO, source="ida-hint")
    assert empty.outcome is ConflictOutcome.FILLED
    assert _func_name(path, SHA_A, ADDR_TWO) == GROUNDED_TWO

    # no row at all behaves like an empty name
    absent = report_conflict(path, SHA_A, ADDR_THREE, GROUNDED, source="ida-hint")
    assert absent.outcome is ConflictOutcome.FILLED
    assert _func_name(path, SHA_A, ADDR_THREE) == GROUNDED

    assert _conflict_row_count(path) == 0
    assert list_pending(path) == []
    assert counts(path) == {"pending": 0, "resolved": 0, "total": 0}


def test_differing_real_names_create_pending_and_keep_existing(tmp_path: Path) -> None:
    path = _store(tmp_path, [(SHA_A, ADDR_ONE, GROUNDED)])

    result = report_conflict(path, SHA_A, ADDR_ONE, HINT, source="ida-hint")

    assert result.outcome is ConflictOutcome.PENDING
    assert result.recorded is True
    assert result.name == GROUNDED
    assert result.binary_sha256 == SHA_A
    assert result.func_addr == ADDR_ONE
    assert result.existing_name == GROUNDED
    assert result.proposed_name == HINT
    assert result.conflict is not None
    assert result.conflict.status is ConflictStatus.PENDING
    assert result.conflict.pending is True
    assert result.conflict.resolution is None
    assert result.conflict.source == "ida-hint"
    assert result.conflict.resolved_at is None
    assert result.conflict.created_at
    assert result.conflict_id == conflict_id(SHA_A, ADDR_ONE, GROUNDED, HINT)

    # the invariant: the grounded name survives
    assert _func_name(path, SHA_A, ADDR_ONE) == GROUNDED
    assert [row.conflict_id for row in list_pending(path)] == [result.conflict_id]
    assert counts(path) == {"pending": 1, "resolved": 0, "total": 1}


def test_identical_or_ungrounded_proposals_are_noops(tmp_path: Path) -> None:
    path = _store(tmp_path, [(SHA_A, ADDR_ONE, GROUNDED), (SHA_A, ADDR_TWO, "")])

    same = report_conflict(path, SHA_A, ADDR_ONE, GROUNDED)
    assert same.outcome is ConflictOutcome.UNCHANGED
    assert same.conflict is None
    assert _func_name(path, SHA_A, ADDR_ONE) == GROUNDED

    # an ungrounded proposal may never overwrite a grounded name ...
    for faked in ("sub_140001000", "func_140001000", "m_140001000", "0x140001000", "unnamed"):
        ignored = report_conflict(path, SHA_A, ADDR_ONE, faked, source="ida-hint")
        assert ignored.outcome is ConflictOutcome.IGNORED, faked
        assert ignored.conflict is None
        assert _func_name(path, SHA_A, ADDR_ONE) == GROUNDED
    # ... and may never fill an empty one either
    ignored_empty = report_conflict(path, SHA_A, ADDR_TWO, "sub_140002000")
    assert ignored_empty.outcome is ConflictOutcome.IGNORED
    assert _func_name(path, SHA_A, ADDR_TWO) == ""

    assert _conflict_row_count(path) == 0

    with pytest.raises(NameConflictError):
        report_conflict(path, SHA_A, ADDR_ONE, "")
    with pytest.raises(NameConflictError):
        report_conflict(path, "", ADDR_ONE, HINT)


def test_is_grounded_name_taxonomy() -> None:
    assert is_grounded_name(GROUNDED) is True
    assert is_grounded_name("IDAPython_Callback") is True
    for placeholder in ("", "   ", "sub_140001000", "func_1", "m_1", "0x140001000", "unnamed", "TargetClass", "TargetMethod"):
        assert is_grounded_name(placeholder) is False, placeholder


# --------------------------------------------------------------------------- #
# (3) deterministic ids, no duplicate pending rows
# --------------------------------------------------------------------------- #


def test_conflict_ids_are_deterministic_and_reports_are_idempotent(tmp_path: Path) -> None:
    path = _store(tmp_path, [(SHA_A, ADDR_ONE, GROUNDED)])

    expected = conflict_id(SHA_A, ADDR_ONE, GROUNDED, HINT)
    assert expected.startswith("nc_")
    assert len(expected) == 43
    assert expected == conflict_id(SHA_A, ADDR_ONE, GROUNDED, HINT)  # pure function
    assert expected != conflict_id(SHA_A, ADDR_TWO, GROUNDED, HINT)
    assert expected != conflict_id(SHA_A, ADDR_ONE, GROUNDED, HINT_TWO)
    assert expected != conflict_id(SHA_A, ADDR_ONE, GROUNDED_TWO, HINT)
    assert expected != conflict_id(SHA_B, ADDR_ONE, GROUNDED, HINT)

    first = report_conflict(path, SHA_A, ADDR_ONE, HINT, source="ida-hint")
    again = report_conflict(path, SHA_A, ADDR_ONE, HINT, source="other-source")

    assert first.conflict_id == expected
    assert again.conflict_id == expected
    assert again.outcome is ConflictOutcome.PENDING
    assert again.conflict is not None and first.conflict is not None
    assert again.conflict.created_at == first.conflict.created_at  # no re-insert
    assert _conflict_row_count(path) == 1
    assert counts(path) == {"pending": 1, "resolved": 0, "total": 1}

    other = report_conflict(path, SHA_A, ADDR_ONE, HINT_TWO, source="ida-hint")
    assert other.conflict_id == conflict_id(SHA_A, ADDR_ONE, GROUNDED, HINT_TWO)
    assert other.conflict_id != expected
    assert counts(path) == {"pending": 2, "resolved": 0, "total": 2}


# --------------------------------------------------------------------------- #
# (4) resolve: pending -> resolved, terminal
# --------------------------------------------------------------------------- #


def test_resolve_keep_existing_leaves_the_grounded_name(tmp_path: Path) -> None:
    path = _store(tmp_path, [(SHA_A, ADDR_ONE, GROUNDED)])
    pending = report_conflict(path, SHA_A, ADDR_ONE, HINT, source="ida-hint")

    resolved = resolve_conflict(
        path, pending.conflict_id, Resolution.KEEP_EXISTING, resolved_by="analyst"
    )

    assert resolved.status is ConflictStatus.RESOLVED
    assert resolved.resolved is True
    assert resolved.resolution is Resolution.KEEP_EXISTING
    assert resolved.resolved_at
    assert resolved.resolved_by == "analyst"
    assert resolved.existing_name == GROUNDED
    assert resolved.proposed_name == HINT

    assert _func_name(path, SHA_A, ADDR_ONE) == GROUNDED
    assert list_pending(path) == []
    assert [row.conflict_id for row in list_resolved(path)] == [pending.conflict_id]
    assert counts(path) == {"pending": 0, "resolved": 1, "total": 1}


def test_resolve_accept_proposed_writes_the_proposed_name(tmp_path: Path) -> None:
    path = _store(tmp_path, [(SHA_A, ADDR_ONE, GROUNDED)])
    pending = report_conflict(path, SHA_A, ADDR_ONE, HINT, source="ida-hint")

    resolved = resolve_conflict(path, pending.conflict_id, "accept_proposed")

    assert resolved.status is ConflictStatus.RESOLVED
    assert resolved.resolution is Resolution.ACCEPT_PROPOSED
    assert _func_name(path, SHA_A, ADDR_ONE) == HINT
    assert list_pending(path) == []
    assert counts(path, binary_sha256=SHA_A) == {"pending": 0, "resolved": 1, "total": 1}


def test_resolved_rows_are_terminal(tmp_path: Path) -> None:
    path = _store(tmp_path, [(SHA_A, ADDR_ONE, GROUNDED)])
    pending = report_conflict(path, SHA_A, ADDR_ONE, HINT, source="ida-hint")
    resolve_conflict(path, pending.conflict_id, Resolution.KEEP_EXISTING)

    with pytest.raises(NameConflictError):
        resolve_conflict(path, pending.conflict_id, Resolution.ACCEPT_PROPOSED)

    re_reported = report_conflict(path, SHA_A, ADDR_ONE, HINT, source="ida-hint")
    assert re_reported.outcome is ConflictOutcome.ALREADY_RESOLVED
    assert re_reported.conflict is not None
    assert re_reported.conflict.status is ConflictStatus.RESOLVED
    assert re_reported.conflict.resolution is Resolution.KEEP_EXISTING
    assert _func_name(path, SHA_A, ADDR_ONE) == GROUNDED
    assert counts(path) == {"pending": 0, "resolved": 1, "total": 1}
    accepted_path = _store(tmp_path, [(SHA_B, ADDR_ONE, GROUNDED)], db_name="accepted.db")
    accepted = report_conflict(accepted_path, SHA_B, ADDR_ONE, HINT, source="ida-hint")
    resolve_conflict(accepted_path, accepted.conflict_id, Resolution.ACCEPT_PROPOSED)
    again = report_conflict(accepted_path, SHA_B, ADDR_ONE, HINT, source="ida-hint")
    assert again.outcome is ConflictOutcome.UNCHANGED
    assert _func_name(accepted_path, SHA_B, ADDR_ONE) == HINT
    assert counts(accepted_path) == {"pending": 0, "resolved": 1, "total": 1}


def test_resolve_rejects_unknown_ids_and_bad_resolutions(tmp_path: Path) -> None:
    path = _store(tmp_path, [(SHA_A, ADDR_ONE, GROUNDED)])
    pending = report_conflict(path, SHA_A, ADDR_ONE, HINT, source="ida-hint")

    with pytest.raises(NameConflictError):
        resolve_conflict(path, "nc_does_not_exist", Resolution.KEEP_EXISTING)
    with pytest.raises(NameConflictError):
        resolve_conflict(path, pending.conflict_id, "maybe")
    assert [row.conflict_id for row in list_pending(path)] == [pending.conflict_id]


def test_accept_proposed_refuses_a_stale_funcs_row(tmp_path: Path) -> None:
    path = _store(tmp_path, [(SHA_A, ADDR_ONE, GROUNDED)])
    pending = report_conflict(path, SHA_A, ADDR_ONE, HINT, source="ida-hint")

    _set_func_name(path, SHA_A, ADDR_ONE, "grounded_by_someone_else")

    with pytest.raises(NameConflictError):
        resolve_conflict(path, pending.conflict_id, Resolution.ACCEPT_PROPOSED)
    # the failed transition rolled back: the row is still pending, the name intact
    assert _func_name(path, SHA_A, ADDR_ONE) == "grounded_by_someone_else"
    assert [row.conflict_id for row in list_pending(path)] == [pending.conflict_id]

    # keep_existing never writes funcs, so it stays available
    resolved = resolve_conflict(path, pending.conflict_id, Resolution.KEEP_EXISTING)
    assert resolved.status is ConflictStatus.RESOLVED
    assert _func_name(path, SHA_A, ADDR_ONE) == "grounded_by_someone_else"


# --------------------------------------------------------------------------- #
# (5) read surface for the future ingest-ida consumer
# --------------------------------------------------------------------------- #


def test_list_pending_orders_filters_and_counts(tmp_path: Path) -> None:
    path = _store(
        tmp_path,
        [
            (SHA_A, ADDR_TWO, GROUNDED_TWO),
            (SHA_A, ADDR_ONE, GROUNDED),
            (SHA_B, ADDR_ONE, GROUNDED),
        ],
    )
    report_conflict(path, SHA_A, ADDR_TWO, HINT)
    report_conflict(path, SHA_A, ADDR_ONE, HINT)
    report_conflict(path, SHA_B, ADDR_ONE, HINT)

    assert [row.func_addr for row in list_pending(path, binary_sha256=SHA_A)] == [
        ADDR_ONE,
        ADDR_TWO,
    ]
    assert len(list_pending(path)) == 3
    assert len(list_pending(path, limit=2)) == 2
    assert len(list_conflicts(path, status="pending")) == 3
    assert [row.binary_sha256 for row in list_conflicts(path, status=ConflictStatus.PENDING, limit=1)] == [SHA_A]

    assert counts(path) == {"pending": 3, "resolved": 0, "total": 3}
    assert counts(path, binary_sha256=SHA_A) == {"pending": 2, "resolved": 0, "total": 2}
    assert counts(path, binary_sha256="c" * 64) == {"pending": 0, "resolved": 0, "total": 0}


def test_targets_may_be_database_path_or_raw_connection(tmp_path: Path) -> None:
    path = tmp_path / "toothpick.db"
    db = Database(path)
    try:
        ensure_schema(db)
        db.upsert_function(
            FunctionRecord(binary_sha256=SHA_A, func_addr=ADDR_ONE, name=GROUNDED, size=16)
        )
        result = report_conflict(db, SHA_A, ADDR_ONE, HINT, source="ida-hint")
        assert result.outcome is ConflictOutcome.PENDING
        assert counts(db) == {"pending": 1, "resolved": 0, "total": 1}
    finally:
        db.close()

    conn = sqlite3.connect(str(path), isolation_level=None)
    try:
        resolved = resolve_conflict(conn, result.conflict_id, Resolution.KEEP_EXISTING)
        assert resolved.status is ConflictStatus.RESOLVED
    finally:
        conn.close()
    assert _func_name(path, SHA_A, ADDR_ONE) == GROUNDED


# --------------------------------------------------------------------------- #
# (7) stdlib-only module, no new dependencies
# --------------------------------------------------------------------------- #


def test_module_imports_only_stdlib_and_repo_storage() -> None:
    import toothpick.name_conflicts as name_conflicts

    source = Path(name_conflicts.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])

    assert roots <= (set(sys.stdlib_module_names) | {"toothpick"})
    assert not ({"capstone", "lief", "z3", "unicorn", "frida", "mcp", "pytest"} & roots)
    assert "faulthandler" not in roots
