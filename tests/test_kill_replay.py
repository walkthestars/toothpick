"""Crash-consistency and kill-at-random replay test suite.

Governed by design-document.md §5 (State Management & Persistence Architecture)
and execution-plan.md §4 (CI/Verification specification).

Enforces:
1. Atomicity & Crash Consistency: Simulated process kill or failure during
   static analysis or receipt insertion leaves WAL database in an uncorrupted state.
2. Integrity Check: PRAGMA integrity_check passes ('ok') upon restart.
3. Replay Idempotence: Full replay after crash or interruption maintains
   exactly-once cursor semantics on (binary_sha256, func_addr) with zero duplicate rows.
4. Trace Event Replay Idempotence: Deterministic event_id replay preserves consistency.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List
from unittest.mock import patch

import pytest

from toothpick.static_engine import StaticEngine
from toothpick.storage import Database, FunctionRecord


def _build_synthetic_binary(num_funcs: int = 50) -> bytes:
    """Generate raw binary bytes containing `num_funcs` distinct synthetic functions.

    Each function: xor eax, eax (31 c0) ; ret (c3) + 13 bytes padding = 16 bytes.
    """
    code = bytearray()
    for _ in range(num_funcs):
        code.extend(b"\x31\xc0\xc3" + b"\x90" * 13)
    return bytes(code)


def test_kill_at_random_crash_consistency(tmp_path: Path) -> None:
    """Simulate ungraceful process termination mid-analysis and verify WAL integrity."""
    db_path = tmp_path / "kill_test.db"
    bin_path = tmp_path / "synthetic.bin"
    bin_bytes = _build_synthetic_binary(num_funcs=40)
    bin_path.write_bytes(bin_bytes)

    # Use chunk_size = 10 so 40 functions will require 4 separate transaction batches
    engine = StaticEngine(db_path=db_path, chunk_size=10)

    # Hook bulk_upsert_functions to inject a crash during the 3rd batch
    call_count = 0
    orig_bulk_upsert = Database.bulk_upsert_functions

    def crashing_bulk_upsert(funcs: List[FunctionRecord], phase: str = "static_analysis") -> int:
        nonlocal call_count
        call_count += 1
        if call_count == 3:
            # Crash simulation inside transaction
            raise RuntimeError("CRASH_INJECTION: Simulated SIGKILL during chunk insertion")
        return orig_bulk_upsert(engine._db, funcs, phase=phase)

    with patch.object(Database, "bulk_upsert_functions", side_effect=crashing_bulk_upsert):
        with pytest.raises(RuntimeError, match="CRASH_INJECTION"):
            engine.analyze_binary(bin_path, is_raw=True, raw_arch="x86_64", raw_base=0x1000)

    # Confirm the process "crashed" on the 3rd chunk
    assert call_count == 3

    sha = hashlib.sha256(bin_bytes).hexdigest()

    # Close any lingering references and open a fresh database handle
    fresh_db = Database(db_path)
    try:
        # Verify WAL mode and SQLite integrity
        assert fresh_db.journal_mode() == "wal"
        integrity = fresh_db.integrity_check()
        assert integrity == ["ok"], f"Database corrupt after crash: {integrity}"

        # In WAL mode, the first 2 chunks committed (20 funcs), while chunk 3 rolled back
        assert fresh_db.count_functions(sha) == 20
    finally:
        fresh_db.close()


def test_cursor_reanalysis_idempotence(tmp_path: Path) -> None:
    """Verify that re-analyzing an interrupted or completed binary is completely idempotent."""
    db_path = tmp_path / "reanalysis_idempotent.db"
    bin_path = tmp_path / "synthetic_50.bin"
    bin_bytes = _build_synthetic_binary(num_funcs=50)
    bin_path.write_bytes(bin_bytes)

    engine = StaticEngine(db_path=db_path, chunk_size=10)

    # Phase 1: Crash during chunk 2
    call_count = 0
    orig_bulk_upsert = Database.bulk_upsert_functions

    def crash_chunk_2(funcs: List[FunctionRecord], phase: str = "static_analysis") -> int:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("CRASH_INJECTION: Interrupted during batch 2")
        return orig_bulk_upsert(engine._db, funcs, phase=phase)

    with patch.object(Database, "bulk_upsert_functions", side_effect=crash_chunk_2):
        with pytest.raises(RuntimeError, match="CRASH_INJECTION"):
            engine.analyze_binary(bin_path, is_raw=True, raw_arch="x86_64", raw_base=0x1000)

    # Phase 2: Resume / Re-run analysis on the exact same binary (normal completion)
    result = engine.analyze_binary(bin_path, is_raw=True, raw_arch="x86_64", raw_base=0x1000)
    assert result["function_count"] == 50

    sha = result["binary_sha256"]
    db = Database(db_path)
    try:
        assert db.integrity_check() == ["ok"]
        # Exactly 50 functions, no duplicate rows
        assert db.count_functions(sha) == 50

        # Phase 3: Re-run a second time to test idempotence on already completed database
        result2 = engine.analyze_binary(bin_path, is_raw=True, raw_arch="x86_64", raw_base=0x1000)
        assert result2["function_count"] == 50
        assert db.count_functions(sha) == 50
    finally:
        db.close()


def test_trace_events_replay_idempotence(tmp_path: Path) -> None:
    """Verify that deterministic trace event replay with caller-supplied event_id is idempotent."""
    db_path = tmp_path / "trace_idempotence.db"
    db = Database(db_path)
    sha = "sha_trace_replay"

    try:
        # First write batch with deterministic IDs
        for i in range(10):
            db.add_trace_event(
                binary_sha256=sha,
                func_addr=0x1000 + i * 0x10,
                event_type="test_event",
                payload={"iter": i},
                event_id=f"evt_{i:04d}",
            )

        assert db.count_trace_events(sha) == 10

        # Replay the same events with identical event_ids (simulating recovery replay)
        for i in range(10):
            db.add_trace_event(
                binary_sha256=sha,
                func_addr=0x1000 + i * 0x10,
                event_type="test_event",
                payload={"iter": i, "replayed": True},
                event_id=f"evt_{i:04d}",
            )

        # Count MUST remain exactly 10, not 20
        assert db.count_trace_events(sha) == 10
        assert db.integrity_check() == ["ok"]

        events = db.get_trace_events(sha)
        assert len(events) == 10
        assert all(e["payload"]["replayed"] is True for e in events)
    finally:
        db.close()
