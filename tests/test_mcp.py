from pathlib import Path
import pytest
from click.testing import CliRunner
from toothpick.cli import cli
from toothpick.mcp_server import create_mcp_server
from toothpick.storage import Database, EdgeKind, FunctionRecord


@pytest.fixture
def test_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "test_mcp.db"
    db = Database(db_path)
    sha = "mcp_sha"

    db.upsert_function(FunctionRecord(
        binary_sha256=sha,
        func_addr=0x1000,
        name="root_fn",
        size=32,
        calling_convention="__fastcall",
        prototype="void root_fn()",
        ir_pcode="",
        decompiled_c="",
    ))
    db.upsert_function(FunctionRecord(
        binary_sha256=sha,
        func_addr=0x2000,
        name="child_fn",
        size=16,
        calling_convention="__fastcall",
        prototype="void child_fn()",
        ir_pcode="",
        decompiled_c="",
    ))
    db.add_edge(sha, 0x1000, 0x2000, EdgeKind.CALL)
    return db_path


def test_mcp_server_tools_registered(test_db: Path) -> None:
    server = create_mcp_server(test_db)
    # Server should define tool handlers
    assert hasattr(server, "list_tools") or hasattr(server, "call_tool")


def test_cli_query_callees(test_db: Path) -> None:
    runner = CliRunner()
    res = runner.invoke(
        cli,
        ["query", "callees", "0x1000", "--db", str(test_db), "--sha", "mcp_sha"],
    )
    assert res.exit_code == 0
    assert "0x2000" in res.output or "8192" in res.output or "child_fn" in res.output


def test_cli_query_callers(test_db: Path) -> None:
    runner = CliRunner()
    res = runner.invoke(
        cli,
        ["query", "callers", "0x2000", "--db", str(test_db), "--sha", "mcp_sha"],
    )
    assert res.exit_code == 0
    assert "0x1000" in res.output or "4096" in res.output or "root_fn" in res.output


def test_mcp_get_func_max_lines(test_db: Path) -> None:
    db = Database(test_db)
    decomp = "\n".join([f"line_{i}();" for i in range(100)])
    db.upsert_function(FunctionRecord(
        binary_sha256="mcp_sha",
        func_addr=0x3000,
        name="large_fn",
        decompiled_c=decomp,
    ))
    server = create_mcp_server(test_db)
    res_default = server.call_tool("toothpick_get_func", {"func_addr": 0x3000})
    assert res_default["function"]["truncated"] is False

    res_bounded = server.call_tool("toothpick_get_func", {"func_addr": 0x3000, "max_lines": 15})
    assert res_bounded["function"]["truncated"] is True
    assert "truncated 85 lines" in res_bounded["function"]["decompiled_c"]


def test_cli_query_func_max_lines(test_db: Path) -> None:
    db = Database(test_db)
    decomp = "\n".join([f"line_{i}();" for i in range(50)])
    db.upsert_function(FunctionRecord(
        binary_sha256="mcp_sha",
        func_addr=0x4000,
        name="fn_lines",
        decompiled_c=decomp,
    ))
    runner = CliRunner()
    res = runner.invoke(
        cli,
        ["query", "func", "0x4000", "--db", str(test_db), "--sha", "mcp_sha", "--max-lines", "5"],
    )
    assert res.exit_code == 0
    assert "truncated 45 lines" in res.output


def test_mcp_cli_sdk_flag(test_db: Path) -> None:
    runner = CliRunner()
    res = runner.invoke(cli, ["mcp", "--db", str(test_db), "--sdk"], input="")
    assert res.exit_code == 0

