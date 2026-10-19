"""
Tests for Toothpick Token & Cost Observability Ledger and Debug Overlay (W10).

Governing specs:
- design-document.md §14 (Observability: events.log pipeline trace + token/cost
  ledger per model class + toothpick show <addr> overlay; no telemetry leaves host).
- design-document.md §1, §9, §17 (Risk R6: LLM cost overrun -> token ledger + phase caps).
- execution-plan.md §1 (W10: Token/cost ledger + events.log + toothpick show).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Generator

import pytest
from click.testing import CliRunner

from toothpick.cli import cli
from toothpick.mcp_server import McpToolError, ToothpickMcpServer
from toothpick.observability import (
    DEFAULT_PRICING,
    CostEstimate,
    FunctionInspector,
    InspectionError,
    ModelClass,
    ObservabilityLedger,
    TokenUsage,
)
from toothpick.storage import Database, FunctionRecord, SsaFact, TypeRecord


@pytest.fixture
def test_db(tmp_path: Path) -> Generator[Database, None, None]:
    """Create a fully populated test database with functions, edges, SSA facts, receipts, and types."""
    db_path = tmp_path / "toothpick_test.db"
    db = Database(db_path)
    sha = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    # 1. Functions
    db.upsert_function(
        FunctionRecord(
            binary_sha256=sha,
            func_addr=0x1000,
            name="TargetFunc",
            size=48,
            calling_convention="win64",
            prototype="int TargetFunc(int arg0, int arg1)",
            decompiled_c="int TargetFunc(int a0, int a1) {\n    return a0 + a1;\n}",
            ir_pcode="(unique, 0x0, 4) = INT_ADD (register, rcx, 4), (register, rdx, 4)\n",
        )
    )
    db.upsert_function(
        FunctionRecord(
            binary_sha256=sha,
            func_addr=0x1100,
            name="CallerFunc",
            size=32,
            calling_convention="win64",
            prototype="void CallerFunc()",
            decompiled_c="void CallerFunc() { TargetFunc(1, 2); }",
        )
    )
    db.upsert_function(
        FunctionRecord(
            binary_sha256=sha,
            func_addr=0x2000,
            name="CalleeFunc",
            size=16,
            calling_convention="win64",
            prototype="void CalleeFunc()",
            decompiled_c="void CalleeFunc() {}",
        )
    )

    # 2. Edges
    db.add_edge(sha, src_addr=0x1100, dst_addr=0x1000, kind="call")
    db.add_edge(sha, src_addr=0x1000, dst_addr=0x2000, kind="call")
    db.add_edge(sha, src_addr=0x1000, dst_addr=0x1010, kind="cfg")
    db.add_edge(sha, src_addr=0x1010, dst_addr=0x1025, kind="cfg")

    # 3. SSA facts
    db.add_ssa_facts(
        [
            SsaFact(
                fact_id="fact_1",
                binary_sha256=sha,
                func_addr=0x1000,
                var_name="arg0",
                ssa_reg="rcx",
                def_point=0x1000,
                use_points=[0x1005, 0x1010],
            ),
            SsaFact(
                fact_id="fact_2",
                binary_sha256=sha,
                func_addr=0x1000,
                var_name="arg1",
                ssa_reg="rdx",
                def_point=0x1000,
                use_points=[0x1005],
            ),
            SsaFact(
                fact_id="fact_3",
                binary_sha256=sha,
                func_addr=0x1000,
                var_name="res",
                ssa_reg="rax",
                def_point=0x1005,
                use_points=[0x1025],
            ),
        ]
    )

    # 4. Receipts
    db.add_receipt(
        binary_sha256=sha,
        func_addr=0x1000,
        phase="ground",
        payload={"verifier_status": "PASS", "live_reads": {"rax": 3}},
        receipt_id="rec_ground_01",
    )
    db.add_receipt(
        binary_sha256=sha,
        func_addr=0x1000,
        phase="solve",
        payload={"status": "sat", "model": {"arg0": 42}},
        receipt_id="rec_solve_01",
    )

    # 5. Types
    db.upsert_type(
        TypeRecord(
            type_id="type_player",
            name="PlayerState",
            kind="struct",
            size_bytes=64,
            layout={"health": 0, "mana": 4},
            lattice_rank=2,
            binary_sha256=sha,
        )
    )

    # 6. Trace Events
    db.add_trace_event(
        binary_sha256=sha,
        func_addr=0x1000,
        event_type="pipeline_milestone",
        payload={"phase": "static", "action": "decompiled"},
        event_id="trac_init_01",
    )

    try:
        yield db
    finally:
        db.close()


def test_token_usage_and_cost_estimation() -> None:
    """Verify TokenUsage property calculations and ModelPricing pricing logic."""
    usage = TokenUsage(prompt_tokens=1_000, completion_tokens=500, cached_tokens=200)
    assert usage.total_tokens == 1_500
    assert usage.to_dict()["total_tokens"] == 1_500
    assert usage.to_dict()["cached_tokens"] == 200

    # Test roundtrip
    restored = TokenUsage.from_dict(usage.to_dict())
    assert restored.prompt_tokens == 1_000
    assert restored.completion_tokens == 500
    assert restored.cached_tokens == 200

    # Pricing calculations
    spark_pricing = DEFAULT_PRICING[ModelClass.ORCHESTRATOR.value]
    cost = spark_pricing.calculate_cost(usage)
    # prompt: 1000 / 1M * $3.00 = $0.003
    # completion: 500 / 1M * $15.00 = $0.0075
    # total = $0.0105
    assert round(cost.prompt_cost, 6) == 0.003
    assert round(cost.completion_cost, 6) == 0.0075
    assert round(cost.total_cost, 6) == 0.0105

    # CostEstimate serialization roundtrip
    cost_dict = cost.to_dict()
    assert cost_dict["total_cost"] == 0.0105
    restored_cost = CostEstimate.from_dict(cost_dict)
    assert round(restored_cost.total_cost, 6) == 0.0105


def test_record_llm_call_and_pricing_resolution(tmp_path: Path) -> None:
    """Verify ObservabilityLedger records LLM calls and computes costs per model."""
    events_log = tmp_path / "events.log"
    with ObservabilityLedger(events_log_path=events_log, log_to_db=False) as ledger:
        # 1. Spark 1.3 orchestrator
        evt1 = ledger.record_llm_call(
            model_name="spark-1.3",
            model_class=ModelClass.ORCHESTRATOR,
            phase="orchestrate",
            prompt_tokens=10_000,
            completion_tokens=2_000,
            binary_sha256="sha_a",
        )
        assert evt1.model_class == "orchestrator"
        assert evt1.token_usage is not None
        assert evt1.cost_estimate is not None
        # 10k * $3/1M = $0.03, 2k * $15/1M = $0.03 -> total $0.06
        assert round(evt1.cost_estimate.total_cost, 4) == 0.06

        # 2. DeepSeek V4.1 bulk worker
        evt2 = ledger.record_llm_call(
            model_name="deepseek-v4.1-flash",
            model_class=ModelClass.BULK,
            phase="static",
            prompt_tokens=100_000,
            completion_tokens=10_000,
            binary_sha256="sha_a",
        )
        assert evt2.model_class == "bulk"
        assert evt2.token_usage is not None
        assert evt2.cost_estimate is not None
        # 100k * $0.14/1M = $0.014, 10k * $0.28/1M = $0.0028 -> total $0.0168
        assert round(evt2.cost_estimate.total_cost, 4) == 0.0168

    # Ensure file exists and contains 2 lines
    assert events_log.exists()
    lines = [line.strip() for line in events_log.read_text(encoding="utf-8").split("\n") if line.strip()]
    assert len(lines) == 2


def test_events_log_file_persistence_jsonl(tmp_path: Path) -> None:
    """Verify events.log appends valid JSON Lines with required fields."""
    events_log = tmp_path / "test_events.log"
    with ObservabilityLedger(events_log_path=events_log, log_to_db=False) as ledger:
        e1 = ledger.record_phase_event(phase="triage", event_type="phase_start", binary_sha256="sha_1")
        e2 = ledger.record_phase_event(
            phase="triage",
            event_type="phase_end",
            binary_sha256="sha_1",
            payload={"detected_engine": "unity-il2cpp"},
        )

    lines = [json.loads(line) for line in events_log.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 2
    assert lines[0]["event_id"] == e1.event_id
    assert lines[0]["phase"] == "triage"
    assert lines[0]["event_type"] == "phase_start"
    assert lines[1]["event_id"] == e2.event_id
    assert lines[1]["payload"]["detected_engine"] == "unity-il2cpp"


def test_sqlite_trace_events_dual_write(tmp_path: Path) -> None:
    """Verify dual-write to SQLite trace_events table and retrieval."""
    db_path = tmp_path / "dual_write.db"
    events_log = tmp_path / "dual_write.log"
    sha = "test_sha_123"

    with ObservabilityLedger(db=db_path, events_log_path=events_log) as ledger:
        evt = ledger.record_llm_call(
            model_name="glm-5.3",
            model_class=ModelClass.SOLVER,
            phase="solve",
            prompt_tokens=5_000,
            completion_tokens=1_000,
            binary_sha256=sha,
            func_addr=0x1000,
        )

        # Retrieve events via ledger
        events = ledger.get_events(binary_sha256=sha)
        assert len(events) == 1
        assert events[0].event_id == evt.event_id
        assert events[0].func_addr == 0x1000
        assert events[0].model_class == "solver"
        assert events[0].token_usage is not None
        assert events[0].token_usage.total_tokens == 6_000

    # Verify directly in SQLite DB
    db = Database(db_path)
    count = db.count_trace_events(sha)
    db.close()
    assert count == 1


def test_ledger_summary_aggregations(tmp_path: Path) -> None:
    """Verify get_summary computes token totals and breakdowns across classes and phases."""
    events_log = tmp_path / "summary_test.log"
    sha = "summary_sha"

    with ObservabilityLedger(events_log_path=events_log, log_to_db=False) as ledger:
        # Orchestrator in query phase
        ledger.record_llm_call(
            model_name="spark-1.3",
            model_class=ModelClass.ORCHESTRATOR,
            phase="query",
            prompt_tokens=2_000,
            completion_tokens=500,
            binary_sha256=sha,
        )
        # Solver in solve phase
        ledger.record_llm_call(
            model_name="glm-5.3",
            model_class=ModelClass.SOLVER,
            phase="solve",
            prompt_tokens=10_000,
            completion_tokens=2_000,
            binary_sha256=sha,
        )
        # Bulk in static phase
        ledger.record_llm_call(
            model_name="deepseek-v4.1-flash",
            model_class=ModelClass.BULK,
            phase="static",
            prompt_tokens=50_000,
            completion_tokens=5_000,
            binary_sha256=sha,
        )

        summary = ledger.get_summary(binary_sha256=sha)
        assert summary.total_events == 3
        assert summary.total_tokens == (2500 + 12000 + 55000)
        assert summary.total_cost > 0.0

        # Class breakdown
        assert summary.by_model_class["orchestrator"]["events"] == 1
        assert summary.by_model_class["orchestrator"]["total_tokens"] == 2500
        assert summary.by_model_class["solver"]["events"] == 1
        assert summary.by_model_class["solver"]["total_tokens"] == 12000
        assert summary.by_model_class["bulk"]["events"] == 1
        assert summary.by_model_class["bulk"]["total_tokens"] == 55000

        # Phase breakdown
        assert "query" in summary.by_phase
        assert "solve" in summary.by_phase
        assert "static" in summary.by_phase
        assert summary.by_phase["solve"]["tokens"] == 12000

        # Formatted table string check
        table_str = summary.format_table()
        assert "TOOTHPICK OBSERVABILITY LEDGER SUMMARY" in table_str
        assert "ORCHESTRATOR" in table_str
        assert "SOLVER" in table_str
        assert "BULK" in table_str
        assert "query" in table_str


def test_function_inspector_complete_overlay(test_db: Database) -> None:
    """Verify FunctionInspector produces a complete FunctionOverlay with all subsystems."""
    sha = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    with FunctionInspector(db=test_db) as inspector:
        overlay = inspector.inspect(func_addr=0x1000, binary_sha256=sha)
        assert overlay.func_addr == 0x1000
        assert overlay.name == "TargetFunc"
        assert overlay.calling_convention == "win64"
        assert overlay.size == 48

        # Call graph & CFG
        assert 0x1100 in overlay.callers
        assert 0x2000 in overlay.callees
        assert len(overlay.cfg_edges) >= 2

        # SSA facts
        assert len(overlay.ssa_facts) == 3
        var_names = [f["var_name"] for f in overlay.ssa_facts]
        assert "arg0" in var_names
        assert "res" in var_names

        # Receipts
        assert len(overlay.receipts) == 3
        phases = [r["phase"] for r in overlay.receipts]
        assert "ground" in phases
        assert "solve" in phases
        assert "func_upsert" in phases

        # Types
        assert len(overlay.types) >= 1
        assert overlay.types[0]["name"] == "PlayerState"

        # Trace events
        assert len(overlay.trace_events) >= 1

        # Format text representation
        text = overlay.format_text()
        assert "TOOTHPICK FUNCTION DEBUG OVERLAY: 0x1000 (TargetFunc)" in text
        assert "Incoming Callers" in text
        assert "Outgoing Callees" in text
        assert "DATA FLOW & SSA FACTS" in text
        assert "GROUNDING RECEIPTS" in text
        assert "TYPE STATE" in text

        # Dict representation
        d = overlay.to_dict()
        assert d["func_addr_hex"] == "0x1000"
        assert len(d["callers"]) == 1
        assert d["callers"][0] == "0x1100"


def test_function_inspector_error_cases(tmp_path: Path) -> None:
    """Verify FunctionInspector handles missing functions and ambiguous SHAs per Trap 20."""
    empty_db = Database(tmp_path / "empty.db")
    with FunctionInspector(db=empty_db) as inspector:
        with pytest.raises(InspectionError, match="No binary analysis records"):
            inspector.inspect(0x1000)
    empty_db.close()

    # Ambiguous SHA test: insert records for two different binaries
    multi_db = Database(tmp_path / "multi.db")
    multi_db.upsert_function(
        FunctionRecord(binary_sha256="sha_1", func_addr=0x1000, name="F1")
    )
    multi_db.upsert_function(
        FunctionRecord(binary_sha256="sha_2", func_addr=0x1000, name="F2")
    )

    with FunctionInspector(db=multi_db) as inspector:
        with pytest.raises(InspectionError, match="Ambiguous binary SHA-256"):
            inspector.inspect(0x1000)

        # Explicit SHA works
        overlay = inspector.inspect(0x1000, binary_sha256="sha_1")
        assert overlay.name == "F1"

        # Non-existent function raises InspectionError
        with pytest.raises(InspectionError, match="Function 0x9999 not found"):
            inspector.inspect(0x9999, binary_sha256="sha_1")

    multi_db.close()


def test_cli_show_text_and_json(test_db: Database, tmp_path: Path) -> None:
    """Verify toothpick show CLI renders both human-readable text and JSON."""
    db_file = test_db.db_path
    runner = CliRunner()

    # 1. Text format
    res = runner.invoke(cli, ["show", "0x1000", "--db", str(db_file)])
    assert res.exit_code == 0
    assert "TOOTHPICK FUNCTION DEBUG OVERLAY: 0x1000 (TargetFunc)" in res.output
    assert "Incoming Callers" in res.output

    # 2. JSON format
    res_json = runner.invoke(cli, ["show", "0x1000", "--db", str(db_file), "--format", "json"])
    assert res_json.exit_code == 0
    data = json.loads(res_json.output)
    assert data["func_addr_hex"] == "0x1000"
    assert data["name"] == "TargetFunc"
    assert len(data["callers"]) == 1

    # 3. Invalid address
    res_err = runner.invoke(cli, ["show", "not_a_valid_hex_address_!@#", "--db", str(db_file)])
    assert res_err.exit_code == 1
    assert "Error:" in res_err.output


def test_cli_ledger_table_and_json(tmp_path: Path) -> None:
    """Verify toothpick ledger CLI renders both summary table and JSON output."""
    events_log = tmp_path / "cli_events.log"
    with ObservabilityLedger(events_log_path=events_log, log_to_db=False) as ledger:
        ledger.record_llm_call(
            model_name="spark-1.3",
            model_class=ModelClass.ORCHESTRATOR,
            phase="orchestration",
            prompt_tokens=4_000,
            completion_tokens=1_000,
        )

    runner = CliRunner()

    # 1. Table format
    res = runner.invoke(cli, ["ledger", "--events-log", str(events_log)])
    assert res.exit_code == 0
    assert "TOOTHPICK OBSERVABILITY LEDGER SUMMARY" in res.output
    assert "ORCHESTRATOR" in res.output

    # 2. JSON format
    res_json = runner.invoke(cli, ["ledger", "--events-log", str(events_log), "--json"])
    assert res_json.exit_code == 0
    data = json.loads(res_json.output)
    assert data["total_events"] == 1
    assert data["total_tokens"] == 5_000
    assert data["by_model_class"]["orchestrator"]["events"] == 1


def test_mcp_show_and_ledger_tools(test_db: Database, tmp_path: Path) -> None:
    """Verify toothpick_show and toothpick_ledger MCP tools over server interface."""
    server = ToothpickMcpServer(test_db, use_sdk=False)

    # 1. toothpick_show text format
    res_show = server.call_tool("toothpick_show", {"func_addr": 0x1000})
    assert res_show["tool"] == "toothpick_show"
    assert res_show["func_addr_hex"] == "0x1000"
    assert "TOOTHPICK FUNCTION DEBUG OVERLAY" in res_show["overlay"]

    # 2. toothpick_show json format
    res_show_json = server.call_tool("toothpick_show", {"func_addr": "0x1000", "format": "json"})
    assert res_show_json["format"] == "json"
    assert isinstance(res_show_json["overlay"], dict)
    assert res_show_json["overlay"]["name"] == "TargetFunc"

    # 3. Missing function raises McpToolError
    with pytest.raises(McpToolError, match="not found"):
        server.call_tool("toothpick_show", {"func_addr": 0xdead})

    # 4. toothpick_ledger tool
    res_ledger = server.call_tool("toothpick_ledger", {})
    assert res_ledger["tool"] == "toothpick_ledger"
    assert "summary" in res_ledger
    assert "table" in res_ledger


def test_func_addr_zero_vs_none_distinction(tmp_path: Path) -> None:
    """Verify func_addr=0 is preserved as address 0 and func_addr=None is preserved as None."""
    db_path = tmp_path / "zero_addr.db"
    events_log = tmp_path / "zero_addr.log"
    sha = "zero_sha"

    with ObservabilityLedger(db=db_path, events_log_path=events_log) as ledger:
        # Event with explicit func_addr=0 (e.g. entrypoint/reset vector)
        evt_zero = ledger.record_phase_event(
            phase="triage",
            event_type="entry_probed",
            binary_sha256=sha,
            func_addr=0,
        )
        # Event with no function (global pipeline milestone)
        evt_none = ledger.record_phase_event(
            phase="triage",
            event_type="pipeline_milestone",
            binary_sha256=sha,
            func_addr=None,
        )

        events = ledger.get_events(binary_sha256=sha)
        assert len(events) == 2
        e_map = {e.event_id: e for e in events}
        assert e_map[evt_zero.event_id].func_addr == 0
        assert e_map[evt_none.event_id].func_addr is None
