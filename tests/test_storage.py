from pathlib import Path
import pytest
from toothpick.storage import Database, EdgeKind, FunctionRecord, TypeRecord, VersionRecord


def test_database_initialization_and_wal(tmp_path: Path) -> None:
    db_path = tmp_path / "toothpick.db"
    db = Database(db_path)
    
    # Assert WAL mode
    with db.connection() as conn:
        cursor = conn.cursor()
        mode = cursor.execute("PRAGMA journal_mode;").fetchone()[0]
        assert mode.lower() == "wal"

        # Check tables
        tables = [
            row[0]
            for row in cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table';"
            ).fetchall()
        ]
        for expected in ["metadata", "funcs", "edges", "ssa_facts", "types", "receipts", "versions", "trace_events"]:
            assert expected in tables


def test_database_idempotent_func_upsert(tmp_path: Path) -> None:
    db_path = tmp_path / "toothpick.db"
    db = Database(db_path)
    sha = "test_sha256_hash"

    fn1 = FunctionRecord(
        binary_sha256=sha,
        func_addr=0x140001000,
        name="sub_140001000",
        size=64,
        calling_convention="__fastcall",
        prototype="void sub_140001000()",
        ir_pcode="pcode_v1",
        decompiled_c="void sub_140001000() { return; }",
    )
    db.upsert_function(fn1)

    # Verify single row
    retrieved = db.get_function(sha, 0x140001000)
    assert retrieved is not None
    assert retrieved.name == "sub_140001000"
    assert retrieved.ir_pcode == "pcode_v1"

    # Re-insert with updated decompilation (overwrite, no duplicate)
    fn2 = FunctionRecord(
        binary_sha256=sha,
        func_addr=0x140001000,
        name="player_update",
        size=64,
        calling_convention="__fastcall",
        prototype="void player_update()",
        ir_pcode="pcode_v2",
        decompiled_c="void player_update() { do_work(); }",
    )
    db.upsert_function(fn2)

    retrieved2 = db.get_function(sha, 0x140001000)
    assert retrieved2 is not None
    assert retrieved2.name == "player_update"
    assert retrieved2.ir_pcode == "pcode_v2"
    assert db.count_functions(sha) == 1


def test_database_edges_query(tmp_path: Path) -> None:
    db_path = tmp_path / "toothpick.db"
    db = Database(db_path)
    sha = "sha_edges"

    # A calls B and C
    db.add_edge(sha, src_addr=0x1000, dst_addr=0x2000, kind=EdgeKind.CALL)
    db.add_edge(sha, src_addr=0x1000, dst_addr=0x3000, kind=EdgeKind.CALL)
    # B calls C
    db.add_edge(sha, src_addr=0x2000, dst_addr=0x3000, kind=EdgeKind.CALL)

    callees_of_a = db.get_callees(sha, 0x1000)
    assert set(callees_of_a) == {0x2000, 0x3000}

    callers_of_c = db.get_callers(sha, 0x3000)
    assert set(callers_of_c) == {0x1000, 0x2000}


def test_database_receipt_commit(tmp_path: Path) -> None:
    db_path = tmp_path / "toothpick.db"
    db = Database(db_path)
    sha = "sha_receipts"

    db.add_receipt(
        binary_sha256=sha,
        func_addr=0x1000,
        phase="static_analysis",
        payload={"status": "ok", "instructions": 12},
    )

    receipts = db.get_receipts(sha, 0x1000)
    assert len(receipts) == 1
    assert receipts[0]["phase"] == "static_analysis"
    assert receipts[0]["payload"]["instructions"] == 12


def test_storage_types_with_binary_sha256(tmp_path: Path) -> None:
    db_path = tmp_path / "types.db"
    db = Database(db_path)
    sha1 = "sha_bin_1"
    sha2 = "sha_bin_2"

    t1 = TypeRecord(
        type_id="type_player_state",
        name="PlayerState",
        kind="struct",
        size_bytes=128,
        layout={"fields": [{"name": "health", "offset": 0, "type": "int32"}]},
        lattice_rank=2,
        binary_sha256=sha1,
    )
    t2 = TypeRecord(
        type_id="type_enemy_state",
        name="EnemyState",
        kind="struct",
        size_bytes=64,
        layout={"fields": [{"name": "hp", "offset": 0, "type": "int32"}]},
        lattice_rank=1,
        binary_sha256=sha2,
    )
    db.upsert_type(t1)
    db.upsert_type(t2)

    assert db.count_types() == 2
    assert db.count_types(sha1) == 1
    assert db.count_types(sha2) == 1

    types_sha1 = db.get_types(sha1)
    assert len(types_sha1) == 1
    assert types_sha1[0].name == "PlayerState"
    assert types_sha1[0].binary_sha256 == sha1


def test_storage_versions_table(tmp_path: Path) -> None:
    db_path = tmp_path / "versions.db"
    db = Database(db_path)
    sha_src = "sha_source_v1"
    sha_tgt = "sha_target_v2"

    rec = VersionRecord(
        binary_sha256=sha_src,
        target_sha256=sha_tgt,
        similarity_score=0.94,
        embedding_blob=b"\x01\x02\x03\x04",
        offset_map={"0x1000": "0x1040", "0x2000": "0x2050"},
    )
    db.add_version(rec)

    assert db.count_versions(sha_src) == 1
    versions = db.get_versions(sha_src)
    assert len(versions) == 1
    assert versions[0].target_sha256 == sha_tgt
    assert versions[0].similarity_score == pytest.approx(0.94)
    assert versions[0].embedding_blob == b"\x01\x02\x03\x04"
    assert versions[0].offset_map == {"0x1000": "0x1040", "0x2000": "0x2050"}


def test_storage_trace_events_table(tmp_path: Path) -> None:
    db_path = tmp_path / "traces.db"
    db = Database(db_path)
    sha = "sha_trace"

    eid1 = db.add_trace_event(
        binary_sha256=sha,
        func_addr=0x1000,
        event_type="enter",
        payload={"args": [1, 2]},
    )
    eid2 = db.add_trace_event(
        binary_sha256=sha,
        func_addr=0x1000,
        event_type="exit",
        payload={"retval": 3},
    )

    assert eid1.startswith("trac_")
    assert eid2.startswith("trac_")
    assert eid1 != eid2
    assert db.count_trace_events(sha) == 2
    events = db.get_trace_events(sha, 0x1000)
    assert len(events) == 2
    assert {e["event_type"] for e in events} == {"enter", "exit"}


def test_storage_invalidate_binary_cleans_all_tables(tmp_path: Path) -> None:
    db_path = tmp_path / "invalidate.db"
    db = Database(db_path)
    sha = "sha_to_invalidate"

    db.upsert_function(FunctionRecord(binary_sha256=sha, func_addr=0x1000))
    db.add_edge(sha, 0x1000, 0x2000, EdgeKind.CALL)
    db.upsert_type(TypeRecord(type_id="t1", binary_sha256=sha))
    db.add_version(VersionRecord(binary_sha256=sha, target_sha256="tgt"))
    db.add_trace_event(binary_sha256=sha, func_addr=0x1000, event_type="hook")
    db.add_receipt(binary_sha256=sha, func_addr=0x1000, phase="test")

    deleted = db.invalidate_binary(sha)
    assert deleted["funcs"] == 1
    assert deleted["edges"] == 1
    assert deleted["types"] == 1
    assert deleted["versions"] == 1
    assert deleted["trace_events"] == 1
    assert deleted["receipts"] == 2
