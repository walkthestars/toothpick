"""CONS-01 consumer tests: icepick schema-v1 extract -> toothpick.db.

Failing-first TDD for src/toothpick/ida_consumer.py. Every test spells the
contract from docs/cons-01-handoff.md (icepick side) that the drive could not
deliver: merge, vendored schema pin, queue-routed conflicts, coverage gate,
idempotent re-ingest, graceful degradation, CLI surface.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from toothpick import name_conflicts
from toothpick.query import QueryEngine
from toothpick.storage import Database, FunctionRecord

ida_consumer = pytest.importorskip("toothpick.ida_consumer")

SHA_A = "a" * 64
SHA_B = "b" * 64
IDB_SHA = "c" * 64

A1 = 0x140001000
A2 = 0x140002000
A3 = 0x140003000
EXTERNAL = 0x14001A2B0


def _doc(*, sha_note: str = IDB_SHA, funcs=None, edges=None) -> dict:
    if funcs is None:
        funcs = [
            {
                "ea": hex(A1),
                "name": "sub_140001000",
                "name_source": "ida",
                "size": 64,
                "callers": [hex(A2)],
                "callees": [hex(A2), hex(A3)],
                "decompiled": "int __fastcall sub_140001000(int a1)\n{\n  return a1 + 1;\n}\n",
                "decompile_status": "ok",
            },
            {
                "ea": hex(A2),
                "name": "_strlen",
                "name_source": "flirt",
                "size": 48,
                "callers": [hex(A1)],
                "callees": [hex(EXTERNAL)],
                "decompiled": "size_t __cdecl _strlen(const char *s)\n{\n  return 1;\n}\n",
                "decompile_status": "ok",
            },
            {
                "ea": hex(A3),
                "name": "sub_140003000",
                "name_source": "ida",
                "size": 412,
                "callers": [hex(A1)],
                "callees": [],
                "decompiled": None,
                "decompile_status": "failed",
                "decompile_error": "DECOMPILE_FAILED(too_big)",
            },
        ]
    if edges is None:
        edges = [
            {"src": hex(A1), "dst": hex(A2), "kind": "call"},
            {"src": hex(A1), "dst": hex(A3), "kind": "call"},
            {"src": hex(A2), "dst": hex(EXTERNAL), "kind": "tail_call"},
            {"src": hex(A2), "dst": hex(A3), "kind": "vtable"},
        ]
    return {
        "schema_version": "v1",
        "kind": "extract",
        "generator": "icepick.extract 0.1.0 (idalib)",
        "provenance": {
            "ida_version": "9.5.0.250616",
            "idb_sha256": sha_note,
            "auto_analysis": True,
            "backend": "idalib",
            "generator": "icepick.extract 0.1.0 (idalib)",
        },
        "funcs": funcs,
        "edges": edges,
    }


def _db(path: Path) -> Database:
    db = Database(path, create=True)
    name_conflicts.ensure_schema(db)
    return db


# -- merge round-trip --------------------------------------------------------


def test_merge_round_trip_with_provenance(tmp_path: Path) -> None:
    db = _db(tmp_path / "tp.db")
    report = ida_consumer.ingest_extract_doc(db, _doc(), binary_sha256=SHA_A)
    assert report["funcs_ingested"] == 3
    assert report["edges_ingested"] == 4
    assert report["conflicts_pending"] == 0
    assert report["warnings"] == []

    engine = QueryEngine(db)
    try:
        callees = {c["func_addr"] for c in engine.callees(SHA_A, A1)}
        assert callees == {A2, A3}
        callers = {c["func_addr"] for c in engine.callers(SHA_A, A2)}
        assert A1 in callers
        got = engine.get_function(SHA_A, A2)
        assert got is not None
        assert got["name"] == "_strlen"
        assert got["has_decompiled_c"] is True
        assert got["has_ir_pcode"] is False  # IDA rows carry no SSA/pcode
        assert "size_t __cdecl _strlen" in (got["decompiled_c"] or "")
    finally:
        engine.close()

    receipts = db.get_receipts(SHA_A, A2)
    phases = {r["phase"] for r in receipts}
    assert "ida_extract" in phases
    payload = next(r for r in receipts if r["phase"] == "ida_extract")
    body = payload["payload"]
    assert body["verifier_status"] == "UNGROUNDED"
    assert body["backend"] == "idalib"


def test_vendored_schema_pin_and_digest() -> None:
    assert ida_consumer.SCHEMA_VERSION == "v1"
    assert "icepick" in ida_consumer.VENDORED_FROM
    assert ida_consumer.validate_extract(_doc()) == []
    first = ida_consumer.extract_digest(_doc())
    assert first == ida_consumer.extract_digest(_doc())
    assert len(first) == 64


# -- refusal paths -----------------------------------------------------------


def test_schema_invalid_doc_is_refused_with_reason(tmp_path: Path) -> None:
    db = _db(tmp_path / "tp.db")
    bad = _doc()
    bad["schema_version"] = "v2"
    with pytest.raises(ida_consumer.IdaConsumerError, match="[Rr]eject|schema"):
        ida_consumer.ingest_extract_doc(db, bad, binary_sha256=SHA_A)
    assert db.count_functions(SHA_A) == 0


def test_non_object_doc_is_refused(tmp_path: Path) -> None:
    db = _db(tmp_path / "tp.db")
    with pytest.raises(ida_consumer.IdaConsumerError):
        ida_consumer.ingest_extract_doc(db, ["not", "a", "doc"], binary_sha256=SHA_A)


# -- conflict routing (through the name_conflicts queue BY NAME) -------------


def test_grounded_name_never_silently_overwritten(tmp_path: Path) -> None:
    db = _db(tmp_path / "tp.db")
    db.upsert_function(
        FunctionRecord(
            binary_sha256=SHA_A,
            func_addr=A1,
            name="Engine_Init",
            size=64,
            prototype="void Engine_Init()",
        ),
        write_receipt=False,
    )
    doc = _doc()
    doc["funcs"][0]["name"] = "ida_render_frame"
    report = ida_consumer.ingest_extract_doc(db, doc, binary_sha256=SHA_A)
    assert report["conflicts_pending"] == 1

    engine = QueryEngine(db)
    try:
        got = engine.get_function(SHA_A, A1)
        assert got is not None
        assert got["name"] == "Engine_Init"  # grounded name kept
    finally:
        engine.close()

    pending = name_conflicts.list_conflicts(
        db, status="pending", binary_sha256=SHA_A
    )
    assert len(pending) == 1
    assert pending[0].existing_name == "Engine_Init"
    assert pending[0].proposed_name == "ida_render_frame"


def test_empty_name_silently_filled_no_conflict(tmp_path: Path) -> None:
    db = _db(tmp_path / "tp.db")
    db.upsert_function(
        FunctionRecord(binary_sha256=SHA_A, func_addr=A1, name="", size=10),
        write_receipt=False,
    )
    doc = _doc()
    doc["funcs"][0]["name"] = "ida_render_frame"
    report = ida_consumer.ingest_extract_doc(db, doc, binary_sha256=SHA_A)
    assert report["conflicts_pending"] == 0
    # A1 (empty incumbent) and A2 (missing row) fill; A3's synthetic
    # sub_* label is ungrounded so the queue ignores it, but the row is
    # still created under its IDA-given label for decompile/edge queries.
    assert report["conflicts_filled"] == 2
    assert db.get_function(SHA_A, A1).name == "ida_render_frame"
    assert db.get_function(SHA_A, A3).name == "sub_140003000"


# -- coverage gate (MW3 lesson: warn-and-continue, never silent success) -----


def test_zero_edges_at_scale_warns_and_records_coverage(tmp_path: Path) -> None:
    funcs = [
        {
            "ea": hex(0x140000000 + i * 0x100),
            "name": f"sub_140{i:06x}",
            "size": 32,
            "decompiled": f"void f{i}(){{}}\n",
            "decompile_status": "ok",
        }
        for i in range(60)
    ]
    db = _db(tmp_path / "tp.db")
    report = ida_consumer.ingest_extract_doc(
        db, _doc(funcs=funcs, edges=[]), binary_sha256=SHA_A
    )
    assert report["funcs_ingested"] == 60
    assert report["edges_ingested"] == 0
    assert any("edge" in w.lower() for w in report["warnings"])
    cov = db.get_receipts(SHA_A, 0)
    assert any(r["phase"] == "ida_extract_coverage" for r in cov)


def test_high_failure_rate_warns_but_ingests(tmp_path: Path) -> None:
    funcs = [
        {
            "ea": hex(0x140000000 + i * 0x100),
            "name": f"real_name_{i}",
            "size": 32,
            "decompiled": None,
            "decompile_status": "failed",
            "decompile_error": "DECOMPILE_FAILED(timeout)",
        }
        for i in range(10)
    ]
    edges = [
        {"src": hex(0x140000000), "dst": hex(0x140000100), "kind": "call"}
    ]
    db = _db(tmp_path / "tp.db")
    report = ida_consumer.ingest_extract_doc(
        db, _doc(funcs=funcs, edges=edges), binary_sha256=SHA_A
    )
    assert report["funcs_ingested"] == 10
    assert any("decompile" in w.lower() for w in report["warnings"])


# -- idempotence --------------------------------------------------------------


def test_reingest_is_idempotent(tmp_path: Path) -> None:
    db = _db(tmp_path / "tp.db")
    first = ida_consumer.ingest_extract_doc(db, _doc(), binary_sha256=SHA_A)
    second = ida_consumer.ingest_extract_doc(db, _doc(), binary_sha256=SHA_A)
    assert first["digest"] == second["digest"]
    assert second["funcs_ingested"] == 3
    assert db.count_functions(SHA_A) == 3
    assert db.count_edges(SHA_A) == 4
    assert (
        len(
            name_conflicts.list_conflicts(
                db, status="pending", binary_sha256=SHA_A
            )
        )
        == 0
    )

    eng1 = QueryEngine(db)
    try:
        before = eng1.callees(SHA_A, A1)
    finally:
        eng1.close()
    eng2 = QueryEngine(db)
    try:
        after = eng2.callees(SHA_A, A1)
    finally:
        eng2.close()
    assert before == after


# -- graceful degradation -----------------------------------------------------


def test_def_use_and_slice_empty_for_ida_only_funcs(tmp_path: Path) -> None:
    db = _db(tmp_path / "tp.db")
    ida_consumer.ingest_extract_doc(db, _doc(), binary_sha256=SHA_A)
    engine = QueryEngine(db)
    try:
        assert engine.def_use(SHA_A, A1) == []
        assert engine.slice(SHA_A, A1, target="rax") == []
    finally:
        engine.close()


def test_mixed_capstone_and_ida_db_queries(tmp_path: Path) -> None:
    db = _db(tmp_path / "tp.db")
    ida_consumer.ingest_extract_doc(db, _doc(), binary_sha256=SHA_A)
    db.upsert_function(
        FunctionRecord(
            binary_sha256=SHA_A,
            func_addr=0x5000,
            name="cap_func",
            size=20,
            ir_pcode="(pcode here)",
        ),
        write_receipt=False,
    )
    db._execute(
        "INSERT INTO ssa_facts (fact_id, binary_sha256, func_addr, var_name,"
        " ssa_reg, def_point, use_points) VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("f1", SHA_A, 0x5000, "rax", "rax_1", 0x5000, "[20480]"),
    )
    engine = QueryEngine(db)
    try:
        assert len(engine.def_use(SHA_A, 0x5000)) == 1
        assert engine.def_use(SHA_A, A1) == []
        assert engine.get_function(SHA_A, 0x5000)["has_ir_pcode"] is True
        assert engine.get_function(SHA_A, A1)["has_ir_pcode"] is False
    finally:
        engine.close()


# -- edge-kind mapping ---------------------------------------------------------


def test_icepick_edge_kinds_map_into_toothpick_taxonomy(
    tmp_path: Path,
) -> None:
    db = _db(tmp_path / "tp.db")
    ida_consumer.ingest_extract_doc(db, _doc(), binary_sha256=SHA_A)
    rows = db.get_edges(SHA_A)
    kinds = {(r.src_addr, r.dst_addr): r.kind.value for r in rows}
    assert kinds[(A1, A2)] == "call"
    assert kinds[(A2, EXTERNAL)] == "call"  # tail_call is still a call
    assert kinds[(A2, A3)] == "xref"  # vtable refs land as xrefs


# -- sha handling --------------------------------------------------------------


def test_sha_defaults_to_idb_with_receipt_note(tmp_path: Path) -> None:
    db = _db(tmp_path / "tp.db")
    report = ida_consumer.ingest_extract_doc(db, _doc())
    assert report["binary_sha256"] == IDB_SHA
    assert any("idb_sha256" in w for w in report["warnings"])
    assert db.count_functions(IDB_SHA) == 3


# -- CLI surface -----------------------------------------------------------------


def test_cli_ingest_ida_round_trip(tmp_path: Path) -> None:
    from click.testing import CliRunner

    from toothpick.cli import cli

    doc_path = tmp_path / "extract.json"
    doc_path.write_text(json.dumps(_doc()), encoding="utf-8")
    db_path = tmp_path / "tp.db"
    runner = CliRunner()
    result = runner.invoke(
        cli, ["ingest-ida", str(doc_path), "--db", str(db_path), "--sha", SHA_B]
    )
    assert result.exit_code == 0, result.output
    body = json.loads(result.output)
    assert body["funcs_ingested"] == 3
    assert body["binary_sha256"] == SHA_B
    assert Database(db_path, create=False).count_functions(SHA_B) == 3


def test_cli_refuses_invalid_doc(tmp_path: Path) -> None:
    from click.testing import CliRunner

    from toothpick.cli import cli

    bad = _doc()
    bad["schema_version"] = "v2"
    doc_path = tmp_path / "bad.json"
    doc_path.write_text(json.dumps(bad), encoding="utf-8")
    db_path = tmp_path / "tp.db"
    runner = CliRunner()
    result = runner.invoke(cli, ["ingest-ida", str(doc_path), "--db", str(db_path)])
    assert result.exit_code == 1
    assert not db_path.exists()
