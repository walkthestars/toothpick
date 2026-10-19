"""W16 DBI tracing engine tests: CLI, generators, parsers, receipts (chunks 4-5).

Companion to ``tests/test_dbi_tracer.py`` (chunk 3, Unicorn engine): this
module covers the ``toothpick dbi`` command surface, the Pin/DynamoRIO
scaffolding generators, trace-log parser fidelity, and SQLite receipt
persistence.  All fixtures are synthetic; no Pin, DynamoRIO, or target
binaries are required.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from click.testing import CliRunner

from toothpick.cli import cli
from toothpick.dbi import (
    ENGINE_DYNAMORIO,
    ENGINE_PIN,
    ENGINE_UNICORN,
    MEMORY_READ,
    MEMORY_WRITE,
    DbiToolGenerator,
    DbiTraceResult,
    TraceParser,
)

#: ``mov rax, 5 ; add rax, 2 ; ret`` (x86_64, also used by the tracer suite).
_MOV_ADD_RET_STUB = bytes.fromhex("48c7c0050000004883c002c3")

#: Stub with stack traffic for the memory-accuracy test.
_STACK_OPS_STUB = bytes.fromhex("554889e54883ec10c745fc2a0000008b45fcc95dc3")

_FAKE_SHA = "b" * 64


def _write_stub(path: Path, blob: bytes = _MOV_ADD_RET_STUB) -> Path:
    path.write_bytes(blob)
    return path


# ---------------------------------------------------------------------------
# 1. Unicorn trace on a synthetic x86_64 stub (via CLI).
# ---------------------------------------------------------------------------


def test_dbi_unicorn_trace_on_synthetic_stub(tmp_path: Path) -> None:
    """``dbi --target`` emulates a stub and reports instructions and BBs."""
    stub = _write_stub(tmp_path / "stub.bin")
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["dbi", "--target", str(stub), "--engine", "unicorn", "--trace-bbs"],
    )
    assert result.exit_code == 0, f"dbi --target failed: {result.output}"
    payload = json.loads(result.output)
    assert payload["command"] == "dbi"
    assert payload["mode"] == "unicorn_trace"
    assert payload["phase"] == "dbi_trace"
    assert payload["verifier_status"] == "GROUNDED"
    trace = payload["trace"]
    assert trace["engine"] == ENGINE_UNICORN
    assert trace["total_instructions"] == 3
    assert trace["executed_bbs"], "expected executed basic blocks"
    assert trace["target_sha256"] == hashlib.sha256(_MOV_ADD_RET_STUB).hexdigest()


# ---------------------------------------------------------------------------
# 2. Memory read/write tracking accuracy.
# ---------------------------------------------------------------------------


def test_dbi_memory_tracking_accuracy(tmp_path: Path) -> None:
    """Stack push/mov traffic is recorded with read/write kinds and sizes."""
    from toothpick.dbi import UnicornDbiTracer

    stub = _write_stub(tmp_path / "stack.bin", _STACK_OPS_STUB)
    tracer = UnicornDbiTracer()
    result = tracer.trace(stub)
    assert result.memory_accesses, "expected recorded memory accesses"
    kinds = {entry.kind for entry in result.memory_accesses}
    assert kinds <= {MEMORY_READ, MEMORY_WRITE}
    assert MEMORY_WRITE in kinds, "expected at least one stack write"
    assert all(entry.size > 0 for entry in result.memory_accesses)
    assert all(entry.mem_addr > 0 for entry in result.memory_accesses)


# ---------------------------------------------------------------------------
# 3. Pintool / DynamoRIO template generation.
# ---------------------------------------------------------------------------


def test_dbi_tool_template_generation(tmp_path: Path) -> None:
    """Both generators emit C++ sources plus build configs."""
    pin_source = DbiToolGenerator.generate_pintool("memtrace", tmp_path / "pin")
    assert pin_source.is_file()
    assert pin_source.suffix == ".cpp"
    assert (tmp_path / "pin" / "Makefile").is_file()
    assert "memtrace" in pin_source.read_text(encoding="utf-8").lower()

    dr_source = DbiToolGenerator.generate_dynamorio_client(
        "bbtrace", tmp_path / "dr"
    )
    assert dr_source.is_file()
    assert (tmp_path / "dr" / "CMakeLists.txt").is_file()
    assert "bbtrace" in dr_source.read_text(encoding="utf-8").lower()


# ---------------------------------------------------------------------------
# 4. Trace log parser fidelity (round-trip through the tdbi/1 format).
# ---------------------------------------------------------------------------

_PIN_LOG = """\
# toothpick-dbi engine=pin target_sha256={sha} elapsed_sec=0.01
I 0x400000 7 mov rax, 5
I 0x400007 4 add rax, 2
I 0x40000b 1 ret
B 0x400000 12 3
M 0x400000 W 0x7ffffff0 8 2a00000000000000
E instructions=3 bbs=1 elapsed_sec=0.01
""".format(sha=_FAKE_SHA)


def test_dbi_trace_log_parser_fidelity(tmp_path: Path) -> None:
    """A hand-written tdbi/1 log parses to the exact trace it describes."""
    log = tmp_path / "pin_trace.log"
    log.write_text(_PIN_LOG, encoding="utf-8")

    parsed = TraceParser.parse_pin_log(log)
    assert isinstance(parsed, DbiTraceResult)
    assert parsed.engine == ENGINE_PIN
    assert parsed.total_instructions == 3
    assert parsed.executed_bbs == [0x400000]
    assert parsed.target_sha256 == _FAKE_SHA
    assert len(parsed.memory_accesses) == 1
    access = parsed.memory_accesses[0]
    assert access.kind == MEMORY_WRITE
    assert access.mem_addr == 0x7FFFFFF0
    assert access.size == 8

    instructions = TraceParser.parse_pin_instructions(log)
    assert [entry.mnemonic for entry in instructions] == ["mov", "add", "ret"]

    # The DynamoRIO parser reads the same schema deterministically.
    reparsed = TraceParser.parse_dynamorio_log(log)
    assert reparsed.engine == ENGINE_DYNAMORIO
    assert reparsed.total_instructions == parsed.total_instructions
    assert reparsed.executed_bbs == parsed.executed_bbs


# ---------------------------------------------------------------------------
# 5. Receipt persistence to SQLite.
# ---------------------------------------------------------------------------


def test_dbi_trace_receipt_persisted(tmp_path: Path) -> None:
    """``dbi --target --db`` stores a GROUNDED dbi_trace receipt."""
    from toothpick.storage import Database

    stub = _write_stub(tmp_path / "stub.bin")
    db_path = tmp_path / "dbi.db"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "dbi",
            "--target",
            str(stub),
            "--engine",
            "unicorn",
            "--trace-bbs",
            "--db",
            str(db_path),
            "--sha",
            _FAKE_SHA,
        ],
    )
    assert result.exit_code == 0, f"dbi --target --db failed: {result.output}"
    payload = json.loads(result.output)
    assert payload["receipt"]["phase"] == "dbi_trace"
    assert payload["receipt"]["binary_sha256"] == _FAKE_SHA

    with Database(db_path) as db:
        receipts = db.get_receipts(_FAKE_SHA)
    assert receipts, "expected at least one persisted receipt"
    phases = set()
    for receipt in receipts:
        try:
            body = json.loads(receipt["payload_json"])
        except (KeyError, TypeError, ValueError):
            body = receipt
        phases.add(receipt.get("phase", body.get("phase")))
        assert body.get("verifier_status", "GROUNDED") == "GROUNDED"
    assert "dbi_trace" in phases


# ---------------------------------------------------------------------------
# 6. CLI surface: help, generation, parse, and usage errors.
# ---------------------------------------------------------------------------


def test_dbi_help_lists_modes(tmp_path: Path) -> None:
    """``dbi --help`` documents all three modes."""
    runner = CliRunner()
    result = runner.invoke(cli, ["dbi", "--help"])
    assert result.exit_code == 0, f"dbi --help failed: {result.output}"
    for flag in ("--target", "--generate", "--parse", "--tool", "--db"):
        assert flag in result.output


def test_dbi_generate_and_parse_modes(tmp_path: Path) -> None:
    """``--generate`` scaffolds a project; ``--parse`` ingests its log shape."""
    runner = CliRunner()
    generated = runner.invoke(
        cli,
        [
            "dbi",
            "--generate",
            "pin",
            "--tool",
            "inscount",
            "--out",
            str(tmp_path / "pintool"),
        ],
    )
    assert generated.exit_code == 0, f"dbi --generate failed: {generated.output}"
    assert (tmp_path / "pintool").is_dir()

    log = tmp_path / "trace.log"
    log.write_text(_PIN_LOG, encoding="utf-8")
    parsed = runner.invoke(cli, ["dbi", "--parse", str(log)])
    assert parsed.exit_code == 0, f"dbi --parse failed: {parsed.output}"
    payload = json.loads(parsed.output)
    assert payload["mode"] == "parse"
    assert payload["trace"]["total_instructions"] == 3

    missing_mode = runner.invoke(cli, ["dbi"])
    assert missing_mode.exit_code != 0

    exclusive = runner.invoke(
        cli,
        ["dbi", "--target", str(tmp_path / "x"), "--generate", "pin"],
    )
    assert exclusive.exit_code != 0
