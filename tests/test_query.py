from pathlib import Path
from typing import Tuple

import pytest
from toothpick.query import QueryEngine, SliceDirection
from toothpick.storage import Database, EdgeKind, FunctionRecord, SsaFact

@pytest.fixture
def populated_db(tmp_path: Path) -> Tuple[Database, str]:
    db_path = tmp_path / "toothpick.db"
    db = Database(db_path)
    sha = "test_sha_query"

    # Add 3 functions
    f1 = FunctionRecord(
        binary_sha256=sha,
        func_addr=0x1000,
        name="main",
        size=32,
        calling_convention="__cdecl",
        prototype="int main()",
        ir_pcode="",
        decompiled_c="",
    )
    f2 = FunctionRecord(
        binary_sha256=sha,
        func_addr=0x2000,
        name="process_input",
        size=48,
        calling_convention="__fastcall",
        prototype="void process_input(int)",
        ir_pcode="",
        decompiled_c="",
    )
    f3 = FunctionRecord(
        binary_sha256=sha,
        func_addr=0x3000,
        name="helper",
        size=16,
        calling_convention="__fastcall",
        prototype="void helper()",
        ir_pcode="",
        decompiled_c="",
    )
    db.upsert_function(f1)
    db.upsert_function(f2)
    db.upsert_function(f3)

    # Edges: main -> process_input, main -> helper, process_input -> helper
    db.add_edge(sha, 0x1000, 0x2000, EdgeKind.CALL)
    db.add_edge(sha, 0x1000, 0x3000, EdgeKind.CALL)
    db.add_edge(sha, 0x2000, 0x3000, EdgeKind.CALL)

    # SSA facts in f2
    # fact 1: def rax at 0x2005, used at 0x2010, 0x2015
    # fact 2: def rcx at 0x2000, used at 0x2005
    db.add_ssa_fact(SsaFact(
        fact_id="f2_fact_1",
        binary_sha256=sha,
        func_addr=0x2000,
        var_name="rax",
        ssa_reg="rax_1",
        def_point=0x2005,
        use_points=[0x2010, 0x2015],
    ))
    db.add_ssa_fact(SsaFact(
        fact_id="f2_fact_2",
        binary_sha256=sha,
        func_addr=0x2000,
        var_name="rcx",
        ssa_reg="rcx_0",
        def_point=0x2000,
        use_points=[0x2005],
    ))

    return db, sha


def test_query_callers_and_callees(populated_db: Tuple[Database, str]) -> None:
    db, sha = populated_db
    engine = QueryEngine(db)
    callees = engine.callees(sha, 0x1000)
    assert len(callees) == 2
    callee_addrs = [c["func_addr"] for c in callees]
    assert set(callee_addrs) == {0x2000, 0x3000}

    callers = engine.callers(sha, 0x3000)
    assert len(callers) == 2
    caller_addrs = [c["func_addr"] for c in callers]
    assert set(caller_addrs) == {0x1000, 0x2000}


def test_query_def_use(populated_db: Tuple[Database, str]) -> None:
    db, sha = populated_db
    engine = QueryEngine(db)

    facts = engine.def_use(sha, 0x2000, var_name="rax")
    assert len(facts) == 1
    assert facts[0]["def_point"] == 0x2005
    assert facts[0]["use_points"] == [0x2010, 0x2015]


def test_query_backward_slice(populated_db: Tuple[Database, str]) -> None:
    db, sha = populated_db
    engine = QueryEngine(db)

    # Slice backward from rax at 0x2005
    slc = engine.slice(sha, 0x2000, target="rax", direction=SliceDirection.BACKWARD)
    assert len(slc) >= 1
    assert any(step["var_name"] == "rax" for step in slc)


def test_slice_idempotence_property(populated_db: Tuple[Database, str]) -> None:
    db, sha = populated_db
    engine = QueryEngine(db)

    run1 = engine.slice(sha, 0x2000, target="rax", direction=SliceDirection.BACKWARD)
    run2 = engine.slice(sha, 0x2000, target="rax", direction=SliceDirection.BACKWARD)
    assert run1 == run2


def test_10x_context_cap_enforcement(tmp_path: Path) -> None:
    # Build dense 1,000 caller edges to a single hub function
    db_path = tmp_path / "dense.db"
    db = Database(db_path)
    sha = "dense_sha"
    hub_addr = 0x50000

    db.upsert_function(FunctionRecord(
        binary_sha256=sha,
        func_addr=hub_addr,
        name="hub",
        size=10,
        calling_convention="",
        prototype="",
        ir_pcode="",
        decompiled_c="",
    ))

    for i in range(1000):
        caller_addr = 0x10000 + (i * 0x10)
        db.add_edge(sha, caller_addr, hub_addr, EdgeKind.CALL)

    engine = QueryEngine(db)
    # Default limit is 50
    callers_default = engine.callers(sha, hub_addr)
    assert len(callers_default) == 50

    # Custom limit is respected
    callers_custom = engine.callers(sha, hub_addr, limit=20)
    assert len(callers_custom) == 20


def test_query_cached_connection_no_thrashing(tmp_path: Path) -> None:
    db_path = tmp_path / "conn_test.db"
    db = Database(db_path)
    sha = "test_sha"
    db.upsert_function(FunctionRecord(binary_sha256=sha, func_addr=0x1000, name="foo"))

    engine = QueryEngine(db_path)
    conn1 = engine._get_read_conn()
    conn2 = engine._get_read_conn()
    assert conn1 is conn2  # Reuses same cached connection

    # Repeated queries reuse the connection
    for _ in range(10):
        engine.binaries()

    assert engine._cached_conn is conn1
    engine.close()
    assert engine._cached_conn is None


def test_query_get_function_max_lines_truncation(tmp_path: Path) -> None:
    db_path = tmp_path / "func_trunc.db"
    db = Database(db_path)
    sha = "trunc_sha"

    decompiled = "\n".join([f"line_{i}();" for i in range(100)])
    pcode = "\n".join([f"pcode_op_{i}" for i in range(50)])

    db.upsert_function(FunctionRecord(
        binary_sha256=sha,
        func_addr=0x1000,
        name="huge_func",
        size=1024,
        decompiled_c=decompiled,
        ir_pcode=pcode,
    ))

    engine = QueryEngine(db)

    # Without max_lines, full text is returned
    full_res = engine.get_function(sha, 0x1000)
    assert full_res is not None
    assert full_res["truncated"] is False
    assert len(full_res["decompiled_c"].splitlines()) == 100

    # With max_lines=10, decompiled_c is capped at 10 lines + notice
    bounded_res = engine.get_function(sha, 0x1000, max_lines=10)
    assert bounded_res is not None
    assert bounded_res["truncated"] is True
    lines = bounded_res["decompiled_c"].splitlines()
    assert len(lines) == 11
    assert "truncated 90 lines" in lines[-1]
