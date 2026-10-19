"""End-to-End Pipeline Smoke Test across Golden Binary Fixtures.

Governed by execution-plan.md §4 (CI/Verification specification)
and design-document.md §1, §5, §13, §15, §16.

Walks the complete lifecycle across synthetic golden binaries:
1.  Triage (toothpick.triage)
2.  Unpack (toothpick.unpack)
3.  Static Disassembly & Graph Ingestion (toothpick.static_engine)
4.  Storage & Query (toothpick.storage, toothpick.query)
5.  SMT Reachability & Constraint Solving (toothpick.solver)
6.  Dynamic Scanner / Grounding Gate (toothpick.scanner)
7.  Mod Generation with Grounding Verification (toothpick.mod_builder)
8.  Cross-Version Differential Rebase (toothpick.rebase)
9.  Observability Ledger Dual-Writer (toothpick.observability)
10. CLI Surface Invocations (toothpick.cli)
"""

from __future__ import annotations

import json
from pathlib import Path
import struct

from click.testing import CliRunner

from toothpick.cli import cli
from toothpick.mod_builder import ModBuilder
from toothpick.observability import FunctionInspector, ObservabilityLedger
from toothpick.query import QueryEngine
from toothpick.rebase import RebaseEngine
from toothpick.scanner import BufferMemoryBackend, MemoryScanner
from toothpick.solver import SMTSolver
from toothpick.static_engine import StaticEngine
from toothpick.storage import Database
from toothpick.triage import (
    Architecture,
    ContainerType,
    EngineType,
    PackerStatus,
    triage_binary,
)
from toothpick.unpack import UnpackStatus, unpack_binary


def _build_golden_pe(base_addr: int = 0x140000000, extra_shift: int = 0) -> bytes:
    """Build a minimal valid 64-bit PE executable containing two linked functions:

    1. Function 1 at base + 0x1000 + extra_shift: calls Function 2 and returns.
    2. Function 2 at base + 0x1020 + extra_shift: compares eax with 0x42, branches, and returns.
    """
    dos_header = bytearray(b"MZ" + b"\x00" * 58)
    dos_header += struct.pack("<I", 0x40)
    pe_sig = b"PE\x00\x00"
    file_header = struct.pack("<HHIIIHH", 0x8664, 1, 0, 0, 0, 0xF0, 0x0022)

    optional_header = bytearray(0xF0)
    struct.pack_into("<H", optional_header, 0, 0x20B)  # PE32+
    struct.pack_into("<I", optional_header, 16, 0x1000 + extra_shift)  # EntryPoint RVA
    struct.pack_into("<Q", optional_header, 24, base_addr)  # ImageBase
    struct.pack_into("<I", optional_header, 32, 0x1000)  # SectionAlignment
    struct.pack_into("<I", optional_header, 36, 0x200)  # FileAlignment
    struct.pack_into("<I", optional_header, 56, 0x4000)  # SizeOfImage

    # Code bytes:
    # 0x1000:
    # sub rsp, 0x28 (48 83 ec 28)
    # call rel32 (e8 17 00 00 00) -> calls 0x1020
    # add rsp, 0x28 (48 83 c4 28)
    # ret (c3)
    # nops (18 bytes) -> brings to 0x1020
    f1_code = (
        b"\x48\x83\xec\x28"
        b"\xe8\x17\x00\x00\x00"
        b"\x48\x83\xc4\x28"
        b"\xc3"
        + b"\x90" * 18
    )

    # 0x1020:
    # cmp eax, 0x42 (83 f8 42)
    # je 0x102a (74 05)
    # xor eax, eax (31 c0)
    # ret (c3)
    # nop, nop (90 90)
    # mov eax, 1 (b8 01 00 00 00)
    # ret (c3)
    # padding
    f2_code = (
        b"\x83\xf8\x42"
        b"\x74\x05"
        b"\x31\xc0"
        b"\xc3"
        b"\x90\x90"
        b"\xb8\x01\x00\x00\x00"
        b"\xc3"
        + b"\x90" * 16
    )

    full_code = f1_code + f2_code

    # Section Header (.text)
    section_header = bytearray(40)
    section_header[0:5] = b".text"
    struct.pack_into(
        "<IIII",
        section_header,
        8,
        len(full_code),
        0x1000 + extra_shift,
        len(full_code),
        0x200,
    )
    struct.pack_into("<I", section_header, 36, 0x60000020)  # CODE | EXECUTE | READ

    headers = (
        bytes(dos_header)
        + pe_sig
        + file_header
        + bytes(optional_header)
        + bytes(section_header)
    )
    padding = b"\x00" * (0x200 - len(headers))
    return headers + padding + full_code


def test_end_to_end_pipeline_golden_smoke(tmp_path: Path) -> None:
    """Execute complete end-to-end integration walkthrough across all subsystems."""
    bin_v1_path = tmp_path / "golden_v1.exe"
    bin_v1_bytes = _build_golden_pe(base_addr=0x140000000)
    bin_v1_path.write_bytes(bin_v1_bytes)

    db_path = tmp_path / "toothpick_smoke.db"
    events_log = tmp_path / "events.log"

    # 1. TRIAGE
    triage_report = triage_binary(bin_v1_path)
    assert triage_report.container == ContainerType.PE
    assert triage_report.arch == Architecture.X86_64
    assert triage_report.engine == EngineType.NATIVE
    assert triage_report.protection.status == PackerStatus.UNPACKED

    # 2. UNPACK PRE-PASS
    unpack_report = unpack_binary(bin_v1_path)
    assert unpack_report.status == UnpackStatus.ALREADY_UNPACKED
    assert unpack_report.packer == "none"

    # 3. STATIC ENGINE
    static_eng = StaticEngine(db_path=db_path)
    static_result = static_eng.analyze_binary(bin_v1_path)
    sha_v1 = static_result["binary_sha256"]
    assert static_result["function_count"] >= 2
    assert static_result["edge_count"] >= 1

    # 4. STORAGE & QUERY
    db = Database(db_path)
    try:
        funcs = db.get_functions(sha_v1)
        assert len(funcs) >= 2

        f1_addr = 0x140001000
        f2_addr = 0x140001020

        # Callee edge verified via QueryEngine
        query_eng = QueryEngine(db=db)
        callee_records = query_eng.callees(sha_v1, f1_addr)
        callee_addrs = [c["func_addr"] for c in callee_records]
        assert f2_addr in callee_addrs

        # 5. SMT REACHABILITY SOLVER
        with SMTSolver(db=db) as solver:
            solve_res = solver.solve_path(
                func_addr=f2_addr,
                timeout_sec=5.0,
                binary_sha256=sha_v1,
            )
            assert solve_res.status in ("SAT", "UNSAT")

        # 6. DYNAMIC SCANNER / GROUNDING GATE
        # Memory region containing code
        region_data = bin_v1_bytes[0x200:]
        mem_backend = BufferMemoryBackend(data=region_data, base_address=f1_addr)
        with MemoryScanner(db=db) as scanner:
            scan_res = scanner.first_scan(
                backend=mem_backend,
                value="83 f8 42",
                data_type="hex",
                binary_sha256=sha_v1,
            )
            assert scan_res.total_matches >= 1
            assert scan_res.matches[0].address == f2_addr

        # Record verified dynamic receipt into DB (Design §16 #1)
        receipt_id = db.add_receipt(
            binary_sha256=sha_v1,
            func_addr=f2_addr,
            phase="ground",
            payload={
                "verifier_status": "GROUNDED",
                "matched_addr": hex(f2_addr),
            },
        )
        assert receipt_id

        # 7. MOD BUILDER
        mod_out = tmp_path / "mod_output"
        with ModBuilder(db=db, binary_sha256=sha_v1) as builder:
            # Epistemic Grounding Gate check passes because receipt is GROUNDED
            mod_artifact = builder.build(
                target=f2_addr,
                out_dir=mod_out,
                name="SmokeMod",
                hook_type="postfix",
            )
            assert mod_artifact.is_grounded is True
            assert mod_artifact.csproj_file.exists()
            assert mod_artifact.source_file.exists()
            source_code = mod_artifact.source_file.read_text(encoding="utf-8")
            assert "SmokeMod" in source_code
            assert "0x140001020" in source_code

        # 8. CROSS-VERSION DIFFERENTIAL REBASE
        # Synthesize v2 binary with code shifted by 0x1000 bytes
        bin_v2_path = tmp_path / "golden_v2.exe"
        bin_v2_bytes = _build_golden_pe(base_addr=0x140000000, extra_shift=0x1000)
        bin_v2_path.write_bytes(bin_v2_bytes)
        static_result_v2 = static_eng.analyze_binary(bin_v2_path)
        sha_v2 = static_result_v2["binary_sha256"]

        with RebaseEngine(db=db) as rebase_eng:
            rebase_report = rebase_eng.rebase(old_sha256=sha_v1, new_target=sha_v2)
            assert rebase_report.auto_ported >= 1
            assert len(rebase_report.matches) >= 1

        # 9. OBSERVABILITY LEDGER & DUAL-WRITER
        ledger = ObservabilityLedger(db=db, events_log_path=events_log)
        event = ledger.record_llm_call(
            model_name="deepseek-chat",
            model_class="CHEAP_FAST",
            phase="smoke_test",
            prompt_tokens=100,
            completion_tokens=50,
            binary_sha256=sha_v1,
            func_addr=f2_addr,
            payload={"verified": True},
        )
        assert event.event_id.startswith("trac_")
        assert events_log.exists()
        assert len(events_log.read_text().splitlines()) >= 1

        # Inspect function overlay
        inspector = FunctionInspector(db=db)
        overlay = inspector.inspect(func_addr=f2_addr, binary_sha256=sha_v1)
        assert overlay.func_addr == f2_addr
        assert len(overlay.receipts) >= 1
        assert len(overlay.trace_events) >= 1

    finally:
        db.close()

    # 10. CLI SURFACE SANITY
    runner = CliRunner()

    res = runner.invoke(cli, ["triage", str(bin_v1_path)])
    assert res.exit_code == 0
    assert "pe" in res.output.lower()

    res = runner.invoke(cli, ["unpack", str(bin_v1_path)])
    assert res.exit_code == 0
    assert "already_unpacked" in res.output.lower()

    res = runner.invoke(cli, ["query", "callees", hex(0x140001000), "--db", str(db_path), "--sha", sha_v1])
    assert res.exit_code == 0
    data = json.loads(res.output)
    callee_targets = [c["func_addr"] for c in data.get("results", [])]
    assert 0x140001020 in callee_targets

    res = runner.invoke(cli, ["show", hex(0x140001020), "--db", str(db_path), "--sha", sha_v1])
    assert res.exit_code == 0
    assert "0x140001020" in res.output

    res = runner.invoke(cli, ["ledger", "--db", str(db_path)])
    assert res.exit_code == 0
    assert "summary" in res.output.lower()
