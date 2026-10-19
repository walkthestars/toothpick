"""
Tests for UNIT-W09: Version Rebase Engine (toothpick.rebase).

Validates:
1. Exact match on identical binary version (100% similarity, 100% auto-port).
2. Rebase rehearsal on synthetic version pair with relocated RVAs (>= 80% auto-port target TBM-3).
3. Modified function detection: basic block edits trigger 0.5 <= sim < 0.99 (REVIEW_REQUIRED per §16 #2).
4. Unmatched new and deleted functions handling.
5. Rebase report persistence to SQLite versions table (VersionRecord) and receipts table.
6. Porting mod configuration: port_mod_config updates target_addr to new binary RVA cleanly.
7. Refusal to auto-port unverified / review-required offsets in port_mod_config.
8. CLI invocation of toothpick rebase with JSON report output and --out file.
9. MCP tool toothpick_rebase execution via create_mcp_server.
10. Multi-binary SHA disambiguation (Trap 20).
11. SQLite handle cleanup on teardown (Trap 7).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pytest
from click.testing import CliRunner

from toothpick.cli import cli
from toothpick.mcp_server import create_mcp_server
from toothpick.mod_builder import ModConfig
from toothpick.rebase import RebaseEngine, RebaseError, RebaseReport
from toothpick.storage import Database, EdgeKind, FunctionRecord, SsaFact

OLD_SHA = "1" * 64
NEW_SHA = "2" * 64
THIRD_SHA = "3" * 64


@pytest.fixture
def test_db(tmp_path: Path) -> Database:
    """Isolated SQLite database fixture."""
    db_file = tmp_path / "rebase_test.db"
    db = Database(db_file, create=True)
    yield db
    db.close()


def _populate_sample_binary(
    db: Database,
    sha: str,
    base_addr: int = 0x14001000,
    count: int = 10,
    prefix: str = "Func",
) -> List[int]:
    """Helper to populate sample functions, CFG edges, and SSA facts."""
    addrs: List[int] = []
    for i in range(count):
        addr = base_addr + (i * 0x100)
        addrs.append(addr)
        name = f"{prefix}_{i}"
        db.upsert_function(
            FunctionRecord(
                binary_sha256=sha,
                func_addr=addr,
                name=name,
                size=64 + (i * 8),
                calling_convention="ms_x64",
                prototype=f"int {name}(int a, int b)",
            ),
            write_receipt=False,
        )
        # Add basic block CFG edges
        db.add_edge(sha, addr, addr + 0x20, EdgeKind.CFG)
        db.add_edge(sha, addr + 0x20, addr + 0x40, EdgeKind.CFG)

        # Add SSA facts (register defs/uses)
        db.add_ssa_fact(
            SsaFact(
                fact_id=f"{sha}:{addr:x}:0:rax",
                binary_sha256=sha,
                func_addr=addr,
                var_name="rax",
                ssa_reg="rax#0",
                def_point=0,
                use_points=[1, 2],
            )
        )
        db.add_ssa_fact(
            SsaFact(
                fact_id=f"{sha}:{addr:x}:1:rcx",
                binary_sha256=sha,
                func_addr=addr,
                var_name="rcx",
                ssa_reg="rcx#1",
                def_point=1,
                use_points=[2],
            )
        )
    return addrs


def test_exact_rebase_identical_binary(test_db: Database) -> None:
    """Test rebasing against the exact same binary yields 1.0 similarity and 100% auto-port."""
    _populate_sample_binary(test_db, OLD_SHA, base_addr=0x14001000, count=5)

    with RebaseEngine(db=test_db) as engine:
        report = engine.rebase(old_sha256=OLD_SHA, new_target=OLD_SHA)
        assert report.total_old_funcs == 5
        assert report.total_new_funcs == 5
        assert report.auto_ported == 5
        assert report.review_required == 0
        assert report.unmatched == 0
        assert report.auto_port_rate == 1.0
        assert len(report.offset_map) == 5
        for m in report.matches:
            assert m.similarity == 1.0
            assert m.status == "AUTO_PORTED"


def test_rebase_rehearsal_relocated_functions_tbm_3(test_db: Database) -> None:
    """Rebase rehearsal on synthetic version pair with shifted RVAs (TBM-3 >= 80% auto-port)."""
    # Old version: 10 functions at 0x14001000
    old_addrs = _populate_sample_binary(test_db, OLD_SHA, base_addr=0x14001000, count=10)

    # New version: 10 functions shifted by +0x5000 (at 0x14006000), identical structure and names
    new_addrs = _populate_sample_binary(test_db, NEW_SHA, base_addr=0x14006000, count=10)

    with RebaseEngine(db=test_db) as engine:
        report = engine.rebase(old_sha256=OLD_SHA, new_target=NEW_SHA)
        assert report.total_old_funcs == 10
        assert report.total_new_funcs == 10
        # All 10 functions should auto-port
        assert report.auto_ported == 10
        assert report.auto_port_rate >= 0.80  # TBM-3 requirement
        assert report.auto_port_rate == 1.0

        # Verify offset map correctly maps old RVAs to new shifted RVAs
        for i in range(10):
            old_hex = f"0x{old_addrs[i]:x}"
            expected_new = new_addrs[i]
            assert report.offset_map[old_hex] == expected_new


def test_modified_functions_require_review(test_db: Database) -> None:
    """Test that functions with altered CFG/registers are flagged REVIEW_REQUIRED (RS-2)."""
    # Old function: synthetic name to force graph comparison
    test_db.upsert_function(
        FunctionRecord(
            binary_sha256=OLD_SHA,
            func_addr=0x1000,
            name="sub_1000",
            size=64,
        ),
        write_receipt=False,
    )
    test_db.add_edge(OLD_SHA, 0x1000, 0x1020, EdgeKind.CFG)
    test_db.add_ssa_fact(
        SsaFact(
            fact_id="old:1",
            binary_sha256=OLD_SHA,
            func_addr=0x1000,
            var_name="rax",
            def_point=0,
        )
    )

    # New function: modified size and additional CFG edges/registers
    test_db.upsert_function(
        FunctionRecord(
            binary_sha256=NEW_SHA,
            func_addr=0x2000,
            name="sub_2000",
            size=128,  # size ratio 0.5
        ),
        write_receipt=False,
    )
    test_db.add_edge(NEW_SHA, 0x2000, 0x2020, EdgeKind.CFG)
    test_db.add_edge(NEW_SHA, 0x2020, 0x2040, EdgeKind.CFG)
    test_db.add_edge(NEW_SHA, 0x2040, 0x2060, EdgeKind.CFG)
    test_db.add_ssa_fact(
        SsaFact(
            fact_id="new:1",
            binary_sha256=NEW_SHA,
            func_addr=0x2000,
            var_name="rax",
            def_point=0,
        )
    )
    test_db.add_ssa_fact(
        SsaFact(
            fact_id="new:2",
            binary_sha256=NEW_SHA,
            func_addr=0x2000,
            var_name="rdx",
            def_point=1,
        )
    )

    with RebaseEngine(db=test_db) as engine:
        report = engine.rebase(old_sha256=OLD_SHA, new_target=NEW_SHA, threshold=0.99)
        assert len(report.matches) == 1
        m = report.matches[0]
        # Similarity should be intermediate (>= 0.50 but < 0.99)
        assert 0.50 <= m.similarity < 0.99
        assert m.status == "REVIEW_REQUIRED"
        assert report.auto_ported == 0
        assert report.review_required == 1
        assert "0x1000" not in report.offset_map


def test_unmatched_new_and_deleted_functions(test_db: Database) -> None:
    """Test handling of deleted functions in old binary and new additions in new binary."""
    # Old has func1 and func2
    test_db.upsert_function(
        FunctionRecord(binary_sha256=OLD_SHA, func_addr=0x1000, name="SharedFunc", size=32),
        write_receipt=False,
    )
    test_db.upsert_function(
        FunctionRecord(binary_sha256=OLD_SHA, func_addr=0x1500, name="DeletedFunc", size=100),
        write_receipt=False,
    )

    # New has SharedFunc and AddedFunc (no DeletedFunc)
    test_db.upsert_function(
        FunctionRecord(binary_sha256=NEW_SHA, func_addr=0x2000, name="SharedFunc", size=32),
        write_receipt=False,
    )
    test_db.upsert_function(
        FunctionRecord(binary_sha256=NEW_SHA, func_addr=0x2500, name="AddedFunc", size=500),
        write_receipt=False,
    )

    with RebaseEngine(db=test_db) as engine:
        report = engine.rebase(old_sha256=OLD_SHA, new_target=NEW_SHA)
        assert report.auto_ported == 1
        assert report.unmatched == 1
        assert 0x1500 in report.unmatched_old
        assert 0x2500 in report.unmatched_new


def test_rebase_persistence_versions_and_receipts(test_db: Database) -> None:
    """Test persistence into versions and receipts tables."""
    _populate_sample_binary(test_db, OLD_SHA, base_addr=0x1000, count=3)
    _populate_sample_binary(test_db, NEW_SHA, base_addr=0x5000, count=3)

    with RebaseEngine(db=test_db) as engine:
        report = engine.rebase(old_sha256=OLD_SHA, new_target=NEW_SHA)
    assert report.auto_ported == 3

    # Check versions table
    row = test_db._conn.execute(
        "SELECT * FROM versions WHERE binary_sha256 = ? AND target_sha256 = ?",
        (OLD_SHA, NEW_SHA),
    ).fetchone()
    assert row is not None
    assert row["similarity_score"] == 1.0
    om = json.loads(row["offset_map_json"])
    assert len(om) == 3

    # Check receipts table
    receipts = [
        r for r in test_db.get_receipts(OLD_SHA, 0)
        if r["phase"] == "rebase"
    ]
    assert len(receipts) == 1
    assert receipts[0]["payload"]["auto_ported"] == 3


def test_port_mod_config_success(test_db: Database) -> None:
    """Test porting ModConfig target_addr to updated binary RVA."""
    old_target = 0x14001000
    new_target = 0x14005000

    test_db.upsert_function(
        FunctionRecord(binary_sha256=OLD_SHA, func_addr=old_target, name="PlayerController.Update", size=64),
        write_receipt=False,
    )
    test_db.upsert_function(
        FunctionRecord(binary_sha256=NEW_SHA, func_addr=new_target, name="PlayerController.Update", size=64),
        write_receipt=False,
    )

    with RebaseEngine(db=test_db) as engine:
        report = engine.rebase(old_sha256=OLD_SHA, new_target=NEW_SHA)

    config = ModConfig(
        plugin_name="SpeedMod",
        target_addr=old_target,
        target_class="PlayerController",
        target_method="Update",
        binary_sha256=OLD_SHA,
    )

    ported_config = RebaseEngine.port_mod_config(config, report)
    assert ported_config.target_addr == new_target
    assert ported_config.binary_sha256 == NEW_SHA
    assert ported_config.plugin_name == "SpeedMod"


def test_port_mod_config_refusal_when_unmatched(test_db: Database) -> None:
    """Test port_mod_config raises RebaseError when target offset is not auto-ported."""
    # Create an empty report with no auto-ported offsets
    empty_report = RebaseReport(
        old_sha256=OLD_SHA,
        new_sha256=NEW_SHA,
        total_old_funcs=1,
        total_new_funcs=0,
        auto_ported=0,
        review_required=0,
        unmatched=1,
        auto_port_rate=0.0,
        offset_map={},
    )

    config = ModConfig(
        plugin_name="SpeedMod",
        target_addr=0x14001000,
        target_class="PlayerController",
        target_method="Update",
    )

    with pytest.raises(RebaseError, match="was not auto-ported"):
        RebaseEngine.port_mod_config(config, empty_report)


def test_cli_rebase_json_report(tmp_path: Path) -> None:
    """Test CLI subcommand 'toothpick rebase' with output report."""
    runner = CliRunner()
    db_file = tmp_path / "cli_rebase.db"

    with Database(db_file, create=True) as db:
        _populate_sample_binary(db, OLD_SHA, base_addr=0x1000, count=4)
        _populate_sample_binary(db, NEW_SHA, base_addr=0x9000, count=4)

    out_json = tmp_path / "rebase_report.json"
    res = runner.invoke(
        cli,
        [
            "rebase",
            NEW_SHA,
            "--old-sha",
            OLD_SHA,
            "--db",
            str(db_file),
            "--out",
            str(out_json),
        ],
    )
    assert res.exit_code == 0, res.output
    data = json.loads(res.output)
    assert data["status"] == "success"
    assert data["auto_ported"] == 4
    assert data["auto_port_rate"] == 1.0

    # Verify file was written
    assert out_json.is_file()
    saved_data = json.loads(out_json.read_text(encoding="utf-8"))
    assert saved_data["total_old_funcs"] == 4


def test_mcp_toothpick_rebase(tmp_path: Path) -> None:
    """Test MCP tool 'toothpick_rebase' execution."""
    db_file = tmp_path / "mcp_rebase.db"
    with Database(db_file, create=True) as db:
        _populate_sample_binary(db, OLD_SHA, base_addr=0x1000, count=3)
        _populate_sample_binary(db, NEW_SHA, base_addr=0x7000, count=3)

    server = create_mcp_server(db_file)
    res = server.call_tool(
        "toothpick_rebase",
        {
            "old_sha256": OLD_SHA,
            "new_sha256": NEW_SHA,
            "limit": 10,
        },
    )
    assert res["tool"] == "toothpick_rebase"
    assert res["auto_ported"] == 3
    assert res["auto_port_rate"] == 1.0
    assert len(res["matches"]) == 3


def test_ambiguous_sha_inference_trap_20(tmp_path: Path) -> None:
    """Test that ambiguous baseline SHA inference raises error (Trap 20)."""
    runner = CliRunner()
    db_file = tmp_path / "ambig.db"
    with Database(db_file, create=True) as db:
        _populate_sample_binary(db, OLD_SHA, base_addr=0x1000, count=2)
        _populate_sample_binary(db, THIRD_SHA, base_addr=0x3000, count=2)

    # Calling rebase without --old-sha when 2 candidate binaries exist must fail
    res = runner.invoke(
        cli,
        [
            "rebase",
            NEW_SHA,
            "--db",
            str(db_file),
        ],
    )
    assert res.exit_code == 1
    assert "Cannot infer baseline binary SHA-256" in res.output


def test_sqlite_handle_cleanup_trap_7(tmp_path: Path) -> None:
    """Verify that RebaseEngine cleanly releases database locks (Trap 7)."""
    db_file = tmp_path / "trap7_rebase.db"
    with Database(db_file, create=True) as db:
        _populate_sample_binary(db, OLD_SHA, base_addr=0x1000, count=2)
        _populate_sample_binary(db, NEW_SHA, base_addr=0x5000, count=2)

    with RebaseEngine(db=db_file) as engine:
        engine.rebase(old_sha256=OLD_SHA, new_target=NEW_SHA)

    # Deleting db_file must succeed without Windows file lock errors
    db_file.unlink()
    assert not db_file.exists()
