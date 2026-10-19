"""
Tests for UNIT-W06: Dynamic Grounding Harness (ground).

Covers:
- Test 1: Static fallback mode emits deterministic ungrounded receipt without VM/emulator.
- Test 2: Static fallback enriches receipt with DB metadata (name, prototype, SSA variables).
- Test 3: Mocked Frida session simulates live attach and register/stack capture.
- Test 4: Mocked Frida session simulates spawn, intercept, resume, and kill cleanup.
- Test 5: Timeout handling when breakpoint is not hit within threshold.
- Test 6: Attach failure handles process errors cleanly.
- Test 7: Spawn failure handles spawn errors cleanly.
- Test 8: Spawn succeeds but attach fails ensures suspended PID is killed.
- Test 9: Frida script compilation error detaches session cleanly.
- Test 10: CLI static-only invocation exits 0 and emits valid JSON receipt.
- Test 11: CLI missing execution mode fails with exit code 1.
- Test 12: CLI conflicting execution modes fail with exit code 1.
- Test 13: CLI invalid address hex fails with exit code 1.
- Test 14: CLI nonexistent database fails with exit code 1.
- Test 15: GroundReceipt serialization round-trip across to_row, to_dict, to_json, and from_row.
- Test 16: Harness context manager cleanly releases Windows SQLite file locks (Trap 7).
- Test 17: Missing frida module surfaces informative FridaAttachError.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from toothpick.cli import cli
from toothpick.ground import (
    FridaAttachError,
    FridaRunner,
    GroundError,
    GroundReceipt,
    GroundTimeoutError,
    GroundingHarness,
)
from toothpick.storage import Database, FunctionRecord, SsaFact


# ---------------------------------------------------------------------------
# Mock Frida Classes
# ---------------------------------------------------------------------------


class MockScript:
    """Simulates a Frida script object."""

    def __init__(self, registers: Optional[Dict[str, str]] = None, auto_fire: bool = True) -> None:
        self.message_handler: Optional[Callable[[Dict[str, Any], Any], None]] = None
        self.registers = registers or {
            "rax": "0x1337",
            "rbx": "0x42",
            "rsp": "0x7fffffffe000",
            "rip": "0x140001000",
        }
        self.stack = ["0x140002000", "0x140003000", "0x0", "0x0"]
        self.auto_fire = auto_fire
        self.loaded = False
        self.unloaded = False

    def on(self, event_name: str, handler: Callable[[Dict[str, Any], Any], None]) -> None:
        if event_name == "message":
            self.message_handler = handler

    def load(self) -> None:
        self.loaded = True
        if self.auto_fire and self.message_handler is not None:
            self.message_handler(
                {
                    "type": "send",
                    "payload": {
                        "status": "HIT",
                        "registers": self.registers,
                        "stack": self.stack,
                    },
                },
                None,
            )

    def unload(self) -> None:
        self.unloaded = True


class MockSession:
    """Simulates a Frida session object."""

    def __init__(self, script: Optional[MockScript] = None) -> None:
        self.script = script or MockScript()
        self.detached = False
        self.created_scripts: List[MockScript] = []

    def create_script(self, source: str) -> MockScript:
        self.created_scripts.append(self.script)
        return self.script

    def detach(self) -> None:
        self.detached = True


class MockFrida:
    """Simulates top-level frida module functions."""

    def __init__(self, session: Optional[MockSession] = None) -> None:
        self.session = session or MockSession()
        self.attached_pids: List[int] = []
        self.spawned: List[str] = []
        self.resumed: List[int] = []
        self.killed: List[int] = []
        self.next_pid = 9876

    def attach(self, pid: int) -> MockSession:
        self.attached_pids.append(pid)
        return self.session

    def spawn(self, target_path: str) -> int:
        self.spawned.append(target_path)
        pid = self.next_pid
        self.next_pid += 1
        return pid

    def resume(self, pid: int) -> None:
        self.resumed.append(pid)

    def kill(self, pid: int) -> None:
        self.killed.append(pid)


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------


def test_ground_static_only_basic() -> None:
    """Static fallback emits an ungrounded receipt without any live VM or DB."""
    harness = GroundingHarness()
    receipt = harness.fallback_read(addr=0x140001000)

    assert receipt.ground_type == "STATIC_FALLBACK"
    assert receipt.verifier_status == "UNGROUNDED"
    assert receipt.func_addr == 0x140001000
    assert receipt.live_reads["mode"] == "STATIC_FALLBACK"
    assert "reason" in receipt.live_reads
    assert receipt.receipt_id.startswith("rec_")


def test_ground_static_only_with_database_enrichment(tmp_path: Path) -> None:
    """Static fallback pulls function name, prototype, and SSA variable facts from DB."""
    db_path = tmp_path / "test_enrich.db"
    db = Database(db_path, create=True)
    sha = "abc123sha"
    func_addr = 0x140001000

    # Insert a dummy function record
    func = FunctionRecord(
        binary_sha256=sha,
        func_addr=func_addr,
        name="test_target_func",
        size=64,
        calling_convention="__fastcall",
        prototype="int test_target_func(int a, int b)",
        ir_pcode="pcode",
        decompiled_c="int test_target_func(int a, int b) { return a + b; }",
    )
    db.upsert_function(func)

    # Insert SSA facts
    db.add_ssa_fact(
        SsaFact(
            fact_id="fact_1",
            binary_sha256=sha,
            func_addr=func_addr,
            var_name="v0_param",
            ssa_reg="rcx",
            def_point=0x140001000,
            use_points=[0x140001008],
        )
    )
    db.add_ssa_fact(
        SsaFact(
            fact_id="fact_2",
            binary_sha256=sha,
            func_addr=func_addr,
            var_name="v1_result",
            ssa_reg="rax",
            def_point=0x140001010,
            use_points=[],
        )
    )

    with GroundingHarness(db=db, binary_sha256=sha) as harness:
        receipt = harness.fallback_read(addr=func_addr)

        assert receipt.ground_type == "STATIC_FALLBACK"
        assert receipt.verifier_status == "UNGROUNDED"
        assert receipt.live_reads["name"] == "test_target_func"
        assert receipt.live_reads["size"] == 64
        assert receipt.live_reads["calling_convention"] == "__fastcall"
        assert receipt.live_reads["prototype"] == "int test_target_func(int a, int b)"
        assert receipt.live_reads["ssa_variables"] == ["v0_param", "v1_result"]

        # Confirm receipt was persisted to database
        stored = harness.get_receipts(func_addr=func_addr)
        assert len(stored) == 1
        assert stored[0].receipt_id == receipt.receipt_id
        assert stored[0].verifier_status == "UNGROUNDED"
        assert stored[0].ground_type == "STATIC_FALLBACK"

    db.close()


def test_ground_mock_frida_attach(tmp_path: Path) -> None:
    """Mocked Frida attach traps at target address and records register states."""
    db_path = tmp_path / "attach.db"
    db = Database(db_path, create=True)
    sha = "test_sha_attach"

    mock_script = MockScript(
        registers={"rax": "0x1337", "rsp": "0x7fffffffe000", "rip": "0x140001000"}
    )
    mock_session = MockSession(script=mock_script)
    mock_frida = MockFrida(session=mock_session)
    runner = FridaRunner(frida_module=mock_frida)

    with GroundingHarness(db=db, binary_sha256=sha, frida_runner=runner) as harness:
        receipt = harness.attach_and_read(pid=1234, addr=0x140001000, timeout=1.0)

        assert receipt.ground_type == "FRIDA_LIVE"
        assert receipt.verifier_status == "GROUNDED"
        assert receipt.live_reads["registers"]["rax"] == "0x1337"
        assert receipt.live_reads["registers"]["rsp"] == "0x7fffffffe000"
        assert len(receipt.live_reads["stack"]) == 4

        assert mock_frida.attached_pids == [1234]
        assert mock_session.detached is True
        assert mock_script.unloaded is True

        # Verify persisted receipt in DB
        receipts = harness.get_receipts(0x140001000)
        assert len(receipts) == 1
        assert receipts[0].verifier_status == "GROUNDED"
        assert receipts[0].ground_type == "FRIDA_LIVE"

    db.close()


def test_ground_mock_frida_spawn() -> None:
    """Mocked Frida spawn launches suspended process, attaches, resumes, and kills."""
    mock_script = MockScript(registers={"rax": "0x42"})
    mock_session = MockSession(script=mock_script)
    mock_frida = MockFrida(session=mock_session)
    runner = FridaRunner(frida_module=mock_frida)

    harness = GroundingHarness(frida_runner=runner)
    receipt = harness.spawn_and_read(target_path="game.exe", addr=0x140005000, timeout=1.0)

    assert receipt.ground_type == "FRIDA_LIVE"
    assert receipt.verifier_status == "GROUNDED"
    assert receipt.live_reads["registers"]["rax"] == "0x42"

    assert mock_frida.spawned == ["game.exe"]
    pid = mock_frida.attached_pids[0]
    assert pid == 9876
    assert mock_frida.resumed == [pid]
    assert mock_frida.killed == [pid]
    assert mock_session.detached is True


def test_ground_timeout() -> None:
    """When breakpoint is not hit within timeout, GroundTimeoutError is raised."""
    # Script does not auto-fire messages
    mock_script = MockScript(auto_fire=False)
    mock_session = MockSession(script=mock_script)
    mock_frida = MockFrida(session=mock_session)
    runner = FridaRunner(frida_module=mock_frida)

    harness = GroundingHarness(frida_runner=runner)
    with pytest.raises(GroundTimeoutError, match="not hit within 0.0 seconds"):
        harness.attach_and_read(pid=555, addr=0x140001000, timeout=0.005)

    assert mock_session.detached is True


def test_ground_attach_failure() -> None:
    """Frida attach failure raises FridaAttachError cleanly."""
    mock_frida = MockFrida()

    def fail_attach(pid: int) -> MockSession:
        raise RuntimeError("Process 9999 not found")

    mock_frida.attach = fail_attach  # type: ignore[assignment]
    runner = FridaRunner(frida_module=mock_frida)

    harness = GroundingHarness(frida_runner=runner)
    with pytest.raises(FridaAttachError, match="Cannot attach to PID 9999"):
        harness.attach_and_read(pid=9999, addr=0x140001000)


def test_ground_spawn_failure() -> None:
    """Frida spawn failure raises FridaAttachError cleanly."""
    mock_frida = MockFrida()

    def fail_spawn(target: str) -> int:
        raise PermissionError("Access denied")

    mock_frida.spawn = fail_spawn  # type: ignore[assignment]
    runner = FridaRunner(frida_module=mock_frida)

    harness = GroundingHarness(frida_runner=runner)
    with pytest.raises(FridaAttachError, match="Cannot spawn restricted.exe"):
        harness.spawn_and_read(target_path="restricted.exe", addr=0x140001000)


def test_ground_spawn_attach_failure_kills_pid() -> None:
    """When spawn succeeds but attach fails, the suspended process is killed."""
    mock_frida = MockFrida()

    def fail_attach(pid: int) -> MockSession:
        raise RuntimeError("Failed to attach to newly spawned process")

    mock_frida.attach = fail_attach  # type: ignore[assignment]
    runner = FridaRunner(frida_module=mock_frida)

    harness = GroundingHarness(frida_runner=runner)
    with pytest.raises(FridaAttachError, match="Cannot spawn target.exe"):
        harness.spawn_and_read(target_path="target.exe", addr=0x140001000)

    assert 9876 in mock_frida.killed


def test_ground_script_compilation_failure() -> None:
    """Frida script compilation failure detaches session and raises FridaAttachError."""
    mock_session = MockSession()

    def fail_create_script(src: str) -> MockScript:
        raise RuntimeError("V8 syntax parse error")

    mock_session.create_script = fail_create_script  # type: ignore[assignment]
    mock_frida = MockFrida(session=mock_session)
    runner = FridaRunner(frida_module=mock_frida)

    harness = GroundingHarness(frida_runner=runner)
    with pytest.raises(FridaAttachError, match="Frida script compilation failed"):
        harness.attach_and_read(pid=123, addr=0x140001000)

    assert mock_session.detached is True


def test_ground_cli_static_only(tmp_path: Path) -> None:
    """toothpick ground CLI in --static-only mode returns exit code 0 and valid JSON."""
    db_path = tmp_path / "cli_ground.db"
    db = Database(db_path, create=True)
    db.close()

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "ground",
            "0x140001000",
            "--static-only",
            "--db",
            str(db_path),
        ],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["ground_type"] == "STATIC_FALLBACK"
    assert data["verifier_status"] == "UNGROUNDED"
    assert data["func_addr_hex"] == "0x140001000"
    assert data["receipt_id"].startswith("rec_")

    # Confirm record in database
    db_check = Database(db_path, create=False)
    receipts = db_check.get_receipts(data["binary_sha256"], 0x140001000)
    assert len(receipts) == 1
    assert receipts[0]["receipt_id"] == data["receipt_id"]
    db_check.close()


def test_ground_cli_missing_mode_error() -> None:
    """Omitting --pid, --spawn, and --static-only fails with code 1."""
    runner = CliRunner()
    result = runner.invoke(cli, ["ground", "0x140001000"])
    assert result.exit_code == 1
    assert "Error: Must specify at least one of --pid, --spawn, or --static-only" in result.output


def test_ground_cli_mutually_exclusive_error() -> None:
    """Passing multiple execution modes fails with code 1."""
    runner = CliRunner()
    result = runner.invoke(cli, ["ground", "0x140001000", "--pid", "1234", "--static-only"])
    assert result.exit_code == 1
    assert "Error: Mutually exclusive modes" in result.output


def test_ground_cli_invalid_addr_error() -> None:
    """Invalid address format fails with code 1."""
    runner = CliRunner()
    result = runner.invoke(cli, ["ground", "not_a_hex_addr", "--static-only"])
    assert result.exit_code == 1
    assert "Error: invalid address" in result.output


def test_ground_cli_missing_db_error(tmp_path: Path) -> None:
    """Nonexistent database path fails with code 1."""
    nonexistent = tmp_path / "does_not_exist.db"
    runner = CliRunner()
    result = runner.invoke(
        cli, ["ground", "0x140001000", "--static-only", "--db", str(nonexistent)]
    )
    assert result.exit_code == 1
    assert "Error: database not found" in result.output


def test_ground_receipt_serialization_roundtrip(tmp_path: Path) -> None:
    """GroundReceipt serialization round-trips cleanly across to_row, to_dict, to_json, and from_row."""
    receipt = GroundReceipt(
        receipt_id="rec_deadbeef1234",
        binary_sha256="sha256_mock_hash",
        func_addr=0x140001000,
        ground_type="FRIDA_LIVE",
        timestamp=1700000000.0,
        live_reads={"registers": {"rax": "0x1"}},
        verifier_status="GROUNDED",
        phase="ground",
    )

    # 1. to_dict
    d = receipt.to_dict()
    assert d["receipt_id"] == "rec_deadbeef1234"
    assert d["func_addr_hex"] == "0x140001000"
    assert d["verifier_status"] == "GROUNDED"

    # 2. to_json
    js = receipt.to_json()
    assert json.loads(js)["func_addr"] == 0x140001000

    # 3. SQLite persistence and from_row
    db_path = tmp_path / "roundtrip.db"
    db = Database(db_path, create=True)
    db.insert_receipt(receipt)

    stored = db.get_receipts("sha256_mock_hash", 0x140001000)
    assert len(stored) == 1
    reconstructed = GroundReceipt.from_row(stored[0])
    assert reconstructed.receipt_id == receipt.receipt_id
    assert reconstructed.binary_sha256 == receipt.binary_sha256
    assert reconstructed.func_addr == receipt.func_addr
    assert reconstructed.ground_type == "FRIDA_LIVE"
    assert reconstructed.verifier_status == "GROUNDED"
    assert reconstructed.live_reads == {"registers": {"rax": "0x1"}}
    db.close()


def test_ground_harness_context_manager_locks(tmp_path: Path) -> None:
    """GroundingHarness context manager cleanly closes SQLite handles to avoid Windows lock (Trap 7)."""
    db_path = tmp_path / "trap7_test.db"
    db = Database(db_path, create=True)
    db.close()

    with GroundingHarness(db=db_path) as harness:
        _ = harness.fallback_read(0x140001000)

    # Database file should be immediately deletable without Windows permission error
    db_path.unlink()
    assert not db_path.exists()


def test_ground_frida_missing_import_error() -> None:
    """When frida is not installed, FridaRunner._get_frida raises FridaAttachError."""
    runner = FridaRunner()
    with patch.dict("sys.modules", {"frida": None}):
        with pytest.raises(FridaAttachError, match="frida module is not installed"):
            runner._get_frida()


def test_ground_sha_inference_ambiguous_db_refused(tmp_path: Path) -> None:
    """P1: multiple binaries in the DB must not silently pick an arbitrary SHA."""
    db_path = tmp_path / "ambiguous.db"
    db = Database(db_path, create=True)
    for sha in ("sha_binary_a", "sha_binary_b"):
        db.upsert_function(
            FunctionRecord(
                binary_sha256=sha,
                func_addr=0x140001000,
                name=f"func_{sha}",
                size=16,
            )
        )

    with GroundingHarness(db=db) as harness:
        with pytest.raises(GroundError, match="Cannot infer binary SHA-256"):
            harness.fallback_read(addr=0x140001000)

    db.close()


def test_ground_sha_inference_single_binary(tmp_path: Path) -> None:
    """A DB holding exactly one binary still auto-infers its SHA-256."""
    db_path = tmp_path / "single.db"
    db = Database(db_path, create=True)
    db.upsert_function(
        FunctionRecord(
            binary_sha256="sha_only_one",
            func_addr=0x140001000,
            name="only_func",
            size=8,
        )
    )

    with GroundingHarness(db=db) as harness:
        receipt = harness.fallback_read(addr=0x140001000)

    assert receipt.binary_sha256 == "sha_only_one"
    assert receipt.ground_type == "STATIC_FALLBACK"
    db.close()


def test_ground_sha_inference_empty_db_yields_empty(tmp_path: Path) -> None:
    """An empty DB yields no binary attribution instead of guessing."""
    db_path = tmp_path / "empty.db"
    db = Database(db_path, create=True)

    with GroundingHarness(db=db) as harness:
        receipt = harness.fallback_read(addr=0x140001000)

    assert receipt.binary_sha256 == ""
    db.close()


def test_ground_cli_ambiguous_sha_fails(tmp_path: Path) -> None:
    """CLI --static-only against a multi-binary DB exits 1 with an explicit error."""
    db_path = tmp_path / "cli_ambiguous.db"
    db = Database(db_path, create=True)
    for sha in ("sha_binary_a", "sha_binary_b"):
        db.upsert_function(
            FunctionRecord(
                binary_sha256=sha,
                func_addr=0x140001000,
                name=f"func_{sha}",
                size=16,
            )
        )
    db.close()

    runner = CliRunner()
    result = runner.invoke(
        cli, ["ground", "0x140001000", "--static-only", "--db", str(db_path)]
    )
    assert result.exit_code == 1
    assert "Error: Cannot infer binary SHA-256" in result.output


def test_ground_cli_explicit_sha_overrides_inference(tmp_path: Path) -> None:
    """An explicit --sha makes a multi-binary DB unambiguous."""
    db_path = tmp_path / "cli_explicit.db"
    db = Database(db_path, create=True)
    for sha in ("sha_binary_a", "sha_binary_b"):
        db.upsert_function(
            FunctionRecord(
                binary_sha256=sha,
                func_addr=0x140001000,
                name=f"func_{sha}",
                size=16,
            )
        )
    db.close()

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "ground",
            "0x140001000",
            "--static-only",
            "--db",
            str(db_path),
            "--sha",
            "sha_binary_b",
        ],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["binary_sha256"] == "sha_binary_b"
    assert data["verifier_status"] == "UNGROUNDED"
