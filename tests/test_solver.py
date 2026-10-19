"""
Tests for UNIT-W05: SMT Reachability & Constraint Solver.

Covers:
- Test 1: Simple satisfiable branch condition (cmp eax, 0x1337; je target -> eax == 0x1337).
- Test 2: Compound range check (0x10 <= eax <= 0x20).
- Test 3: Contradictory / unsatisfiable path detection (status == "UNSAT").
- Test 4: Strict timeout fallback gate (status == "TIMEOUT").
- Test 5: End-to-end CLI test using toothpick solve on toothpick.db.
- Test 6: Arithmetic mutations and 64-bit comparisons.
- Test 7: ARM64 conditional branches (b.eq, cbz).
- Test 8: Flag-setting test eax, eax instructions.
- Test 9: Sub-register model resolution.
- Test 10: Error handling for missing functions and nonexistent databases.
- Test 11: Register-only branch tests (cbz/cbnz) lift their own tested operand.
- Test 12: A register test missing its operand surfaces as UNSUPPORTED, not silence.
- Test 13: Unconditional jump edges are taken, not constrained to false.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import z3
from click.testing import CliRunner

from toothpick.cli import cli
from toothpick.solver import (
    Instruction,
    SMTBranchLifter,
    SMTSolver,
    SymbolicState,
    UnsupportedInstructionError,
)
from toothpick.static_engine import StaticEngine


# ---------------------------------------------------------------------------
# Test Fixtures & Helpers
# ---------------------------------------------------------------------------


def _build_minimal_pe_x64(code_bytes: bytes) -> bytes:
    import struct

    dos_header = bytearray(b"MZ" + b"\x00" * 58)
    dos_header += struct.pack("<I", 0x40)
    pe_sig = b"PE\x00\x00"
    file_header = struct.pack("<HHIIIHH", 0x8664, 1, 0, 0, 0, 0, 0x0022)
    optional_header = bytearray(0xF0)
    struct.pack_into("<H", optional_header, 0, 0x20B)  # PE32+
    struct.pack_into("<I", optional_header, 16, 0x1000)  # EntryPoint RVA
    struct.pack_into("<Q", optional_header, 24, 0x140000000)  # ImageBase
    struct.pack_into("<I", optional_header, 32, 0x1000)  # SectionAlignment
    struct.pack_into("<I", optional_header, 36, 0x200)  # FileAlignment
    struct.pack_into("<I", optional_header, 56, 0x3000)  # SizeOfImage

    section_header = bytearray(40)
    section_header[0:5] = b".text"
    struct.pack_into(
        "<IIII",
        section_header,
        8,
        max(len(code_bytes), 0x1000),
        0x1000,
        len(code_bytes),
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
    return headers + padding + code_bytes


# ---------------------------------------------------------------------------
# Test 1: Simple Satisfiable Branch Condition
# ---------------------------------------------------------------------------


def test_solve_simple_branch() -> None:
    """Verifies that cmp eax, 0x1337; je target solves to eax == 0x1337."""
    insns = [
        Instruction(0x1000, 5, "cmp", "eax, 0x1337"),
        Instruction(0x1005, 2, "je", "0x1020", is_jump=True, target=0x1020),
        Instruction(0x1007, 2, "xor", "eax, eax"),
        Instruction(0x1009, 1, "ret", "", is_ret=True),
        Instruction(0x1020, 5, "mov", "eax, 1"),
        Instruction(0x1025, 1, "ret", "", is_ret=True),
    ]

    solver = SMTSolver(arch="x86_64")
    res = solver.solve_instructions(insns, target_addr=0x1020)

    assert res.status == "SAT"
    assert res.model is not None
    assert res.model["eax"] == 0x1337
    assert res.model["rax"] == 0x1337
    assert len(res.constraints) > 0
    assert any("4919" in c or "1337" in c for c in res.constraints)


# ---------------------------------------------------------------------------
# Test 2: Compound Range Check
# ---------------------------------------------------------------------------


def test_solve_compound_range_check() -> None:
    """Verifies that chained checks solve for range 0x10 <= eax <= 0x20."""
    # If eax < 0x10, fail (0x1030)
    # If eax > 0x20, fail (0x1030)
    # Fallthrough reaches target (0x100A)
    insns = [
        Instruction(0x1000, 3, "cmp", "eax, 0x10"),
        Instruction(0x1003, 2, "jl", "0x1030", is_jump=True, target=0x1030),
        Instruction(0x1005, 3, "cmp", "eax, 0x20"),
        Instruction(0x1008, 2, "jg", "0x1030", is_jump=True, target=0x1030),
        Instruction(0x100A, 5, "mov", "eax, 1"),
        Instruction(0x100F, 1, "ret", "", is_ret=True),
        Instruction(0x1030, 2, "xor", "eax, eax"),
        Instruction(0x1032, 1, "ret", "", is_ret=True),
    ]

    solver = SMTSolver(arch="x86_64")
    res = solver.solve_instructions(insns, target_addr=0x100A)

    assert res.status == "SAT"
    assert res.model is not None
    assert 0x10 <= res.model["eax"] <= 0x20
    assert len(res.constraints) == 2


# ---------------------------------------------------------------------------
# Test 3: Contradictory Unsatisfiable Branches
# ---------------------------------------------------------------------------


def test_solve_contradictory_branches() -> None:
    """Verifies that mutually exclusive path conditions return UNSAT."""
    # Requires eax >= 0x10 AND eax <= 0x5
    insns = [
        Instruction(0x1000, 3, "cmp", "eax, 0x10"),
        Instruction(0x1003, 2, "jl", "0x1030", is_jump=True, target=0x1030),
        Instruction(0x1005, 3, "cmp", "eax, 0x5"),
        Instruction(0x1008, 2, "jg", "0x1030", is_jump=True, target=0x1030),
        Instruction(0x100A, 5, "mov", "eax, 1"),
        Instruction(0x100F, 1, "ret", "", is_ret=True),
        Instruction(0x1030, 2, "xor", "eax, eax"),
        Instruction(0x1032, 1, "ret", "", is_ret=True),
    ]

    solver = SMTSolver(arch="x86_64")
    res = solver.solve_instructions(insns, target_addr=0x100A)

    assert res.status == "UNSAT"
    assert res.model is None


# ---------------------------------------------------------------------------
# Test 4: Strict Timeout Fallback Gate
# ---------------------------------------------------------------------------


def test_solve_timeout_fallback() -> None:
    """Verifies that solver aborts cleanly with TIMEOUT status within time budget."""
    insns = [
        Instruction(0x1000, 3, "cmp", "eax, 0x10"),
        Instruction(0x1003, 2, "jl", "0x1030", is_jump=True, target=0x1030),
        Instruction(0x100A, 1, "ret", "", is_ret=True),
        Instruction(0x1030, 1, "ret", "", is_ret=True),
    ]

    solver = SMTSolver(arch="x86_64")
    # Set an infinitesimally small timeout
    res = solver.solve_instructions(insns, target_addr=0x100A, timeout_sec=0.0000001)

    assert res.status == "TIMEOUT"
    assert res.model is None
    assert res.elapsed_sec < 1.0


# ---------------------------------------------------------------------------
# Test 5: End-to-End CLI Integration with toothpick.db
# ---------------------------------------------------------------------------


def test_solve_cli_end_to_end(tmp_path: Path) -> None:
    """Ingests a binary with a conditional branch into DB, then runs toothpick solve CLI."""
    # Function at 0x140001000:
    # 0x140001000: 81 f8 37 13 00 00       cmp eax, 0x1337
    # 0x140001006: 74 03                   je 0x14000100b
    # 0x140001008: 31 c0                   xor eax, eax
    # 0x14000100a: c3                      ret
    # 0x14000100b: b8 01 00 00 00          mov eax, 1
    # 0x140001010: c3                      ret
    code = (
        b"\x81\xf8\x37\x13\x00\x00"
        b"\x74\x03"
        b"\x31\xc0\xc3"
        b"\xb8\x01\x00\x00\x00\xc3"
    )
    exe_path = tmp_path / "branch_target.exe"
    exe_path.write_bytes(_build_minimal_pe_x64(code))
    db_path = tmp_path / "toothpick.db"

    # Ingest using StaticEngine
    engine = StaticEngine(db_path=db_path)
    res = engine.analyze(exe_path)
    assert res.function_count >= 1

    runner = CliRunner()
    # 1. Run toothpick solve with explicit target-block
    result = runner.invoke(
        cli,
        [
            "solve",
            "0x140001000",
            "--target-block",
            "0x14000100b",
            "--db",
            str(db_path),
            "--timeout",
            "5.0",
        ],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["command"] == "solve"
    assert data["status"] == "SAT"
    assert data["model"]["eax"] == 0x1337
    assert data["target_block_hex"] == "0x14000100b"

    # 2. Run toothpick solve without explicit target-block (defaults to branch target)
    result_default = runner.invoke(
        cli,
        [
            "solve",
            "0x140001000",
            "--db",
            str(db_path),
        ],
    )
    assert result_default.exit_code == 0, result_default.output
    data_def = json.loads(result_default.output)
    assert data_def["status"] == "SAT"
    assert data_def["model"]["eax"] == 0x1337


# ---------------------------------------------------------------------------
# Test 6: Arithmetic Mutations and Register Flow
# ---------------------------------------------------------------------------


def test_solve_arithmetic_mutations() -> None:
    """Verifies that arithmetic operations before comparison correctly back-propagate."""
    # input in ecx
    # mov eax, ecx
    # add eax, 5
    # sub eax, 2  (net: ecx + 3)
    # cmp eax, 0x20
    # je 0x1020
    insns = [
        Instruction(0x1000, 2, "mov", "eax, ecx"),
        Instruction(0x1002, 3, "add", "eax, 5"),
        Instruction(0x1005, 3, "sub", "eax, 2"),
        Instruction(0x1008, 3, "cmp", "eax, 0x20"),
        Instruction(0x100B, 2, "je", "0x1020", is_jump=True, target=0x1020),
        Instruction(0x100D, 1, "ret", "", is_ret=True),
        Instruction(0x1020, 1, "ret", "", is_ret=True),
    ]

    solver = SMTSolver(arch="x86_64")
    res = solver.solve_instructions(insns, target_addr=0x1020)

    assert res.status == "SAT"
    assert res.model is not None
    # ecx + 3 == 0x20 (32) => ecx == 29 (0x1D)
    assert res.model["ecx"] == 29


# ---------------------------------------------------------------------------
# Test 7: ARM64 Branch Condition Lifting
# ---------------------------------------------------------------------------


def test_solve_arm64_conditions() -> None:
    """Verifies ARM64 cmp/b.eq and cbz condition lifting."""
    # ARM64: cmp w0, #0x42; b.eq 0x2000
    insns_arm = [
        Instruction(0x1000, 4, "cmp", "w0, #0x42"),
        Instruction(0x1004, 4, "b.eq", "0x2000", is_jump=True, target=0x2000),
        Instruction(0x1008, 4, "ret", "", is_ret=True),
        Instruction(0x2000, 4, "ret", "", is_ret=True),
    ]

    solver = SMTSolver(arch="arm64")
    res = solver.solve_instructions(insns_arm, target_addr=0x2000)

    assert res.status == "SAT"
    assert res.model is not None
    assert res.model["w0"] == 0x42

    # ARM64: cbz x1, 0x3000
    insns_cbz = [
        Instruction(0x1000, 4, "cbz", "x1, 0x3000", is_jump=True, target=0x3000),
        Instruction(0x1004, 4, "ret", "", is_ret=True),
        Instruction(0x3000, 4, "ret", "", is_ret=True),
    ]
    res_cbz = solver.solve_instructions(insns_cbz, target_addr=0x3000)
    assert res_cbz.status == "SAT"
    assert res_cbz.model is not None
    assert res_cbz.model["x1"] == 0


# ---------------------------------------------------------------------------
# Test 8: Flag-Setting test eax, eax
# ---------------------------------------------------------------------------


def test_solve_test_flag_setting() -> None:
    """Verifies that test eax, eax followed by jz/jnz sets correct constraints."""
    # test eax, eax; jz 0x1020 (requires eax == 0)
    insns_jz = [
        Instruction(0x1000, 2, "test", "eax, eax"),
        Instruction(0x1002, 2, "jz", "0x1020", is_jump=True, target=0x1020),
        Instruction(0x1004, 1, "ret", "", is_ret=True),
        Instruction(0x1020, 1, "ret", "", is_ret=True),
    ]
    solver = SMTSolver(arch="x86_64")
    res = solver.solve_instructions(insns_jz, target_addr=0x1020)
    assert res.status == "SAT"
    assert res.model is not None
    assert res.model["eax"] == 0

    # test eax, eax; jnz 0x1020 (requires eax != 0)
    insns_jnz = [
        Instruction(0x1000, 2, "test", "eax, eax"),
        Instruction(0x1002, 2, "jnz", "0x1020", is_jump=True, target=0x1020),
        Instruction(0x1004, 1, "ret", "", is_ret=True),
        Instruction(0x1020, 1, "ret", "", is_ret=True),
    ]
    res_jnz = solver.solve_instructions(insns_jnz, target_addr=0x1020)
    assert res_jnz.status == "SAT"
    assert res_jnz.model is not None
    assert res_jnz.model["eax"] != 0


# ---------------------------------------------------------------------------
# Test 9: Sub-Register Resolution in 64-Bit Mode
# ---------------------------------------------------------------------------


def test_solve_subregisters() -> None:
    """Verifies that 64-bit comparison also populates 32/16/8-bit subregisters in model."""
    insns = [
        Instruction(0x1000, 10, "cmp", "rax, 0x1122334455667788"),
        Instruction(0x100A, 2, "je", "0x1020", is_jump=True, target=0x1020),
        Instruction(0x100C, 1, "ret", "", is_ret=True),
        Instruction(0x1020, 1, "ret", "", is_ret=True),
    ]

    solver = SMTSolver(arch="x86_64")
    res = solver.solve_instructions(insns, target_addr=0x1020)

    assert res.status == "SAT"
    assert res.model is not None
    assert res.model["rax"] == 0x1122334455667788
    assert res.model["eax"] == 0x55667788
    assert res.model["ax"] == 0x7788
    assert res.model["al"] == 0x88


# ---------------------------------------------------------------------------
# Test 10: Error Handling & Invalid Targets
# ---------------------------------------------------------------------------


def test_solve_error_handling(tmp_path: Path) -> None:
    """Verifies graceful handling of nonexistent databases and unmapped addresses."""
    runner = CliRunner()
    # Nonexistent DB
    res = runner.invoke(cli, ["solve", "0x1000", "--db", str(tmp_path / "nonexistent.db")])
    assert res.exit_code == 1
    assert "not found" in res.output

    # Target block not in CFG
    insns = [
        Instruction(0x1000, 1, "ret", "", is_ret=True),
    ]
    solver = SMTSolver()
    res_unreachable = solver.solve_instructions(insns, target_addr=0x9999)
    assert res_unreachable.status == "UNSAT"


# ---------------------------------------------------------------------------
# Test 11: Register-Only Branch Tests Lift Their Own Operand
# ---------------------------------------------------------------------------


def test_cbz_lifts_register_from_own_operand() -> None:
    """cbz/cbnz constrain the register they name, never a stale comparison.

    Regression for the W05 P1 finding: cbz/cbnz used to read
    SymbolicState.last_cmp, so a cbz with no preceding comparison emitted no
    constraint at all and a cbz after an unrelated cmp constrained the wrong
    register.
    """
    lifter = SMTBranchLifter(arch="arm64")

    # (a) No comparison recorded anywhere: cbz x1 must still force x1 == 0.
    fresh = SymbolicState(arch="arm64")
    cond = lifter.lift_branch_condition(
        Instruction(0x1000, 4, "cbz", "x1, 0x3000", is_jump=True, target=0x3000),
        fresh,
        taken=True,
    )
    assert cond is not None
    s = z3.Solver()
    s.add(cond, fresh.get_reg("x1") != 0)
    assert s.check() == z3.unsat

    s_sat = z3.Solver()
    s_sat.add(cond, fresh.get_reg("x1") == 0)
    assert s_sat.check() == z3.sat

    # (b) An unrelated earlier comparison must not be reused for the branch.
    stale = SymbolicState(arch="arm64")
    lifter.lift_instruction(Instruction(0x2000, 4, "cmp", "w2, #0x42"), stale)
    cond_stale = lifter.lift_branch_condition(
        Instruction(0x2004, 4, "cbz", "w1, 0x3000", is_jump=True, target=0x3000),
        stale,
        taken=True,
    )
    assert cond_stale is not None
    s2 = z3.Solver()
    s2.add(cond_stale, stale.get_reg("w1") != 0)
    assert s2.check() == z3.unsat

    # The stale cmn/cmp result is irrelevant to the cbz predicate.
    s3 = z3.Solver()
    s3.add(cond_stale, stale.get_reg("w1") == 0, stale.get_reg("w2") == 0x42)
    assert s3.check() == z3.sat

    # (c) cbnz negates the same tested register.
    state_cbnz = SymbolicState(arch="arm64")
    cond_cbnz = lifter.lift_branch_condition(
        Instruction(0x3000, 4, "cbnz", "x3, 0x4000", is_jump=True, target=0x4000),
        state_cbnz,
        taken=True,
    )
    assert cond_cbnz is not None
    s4 = z3.Solver()
    s4.add(cond_cbnz, state_cbnz.get_reg("x3") == 0)
    assert s4.check() == z3.unsat


# ---------------------------------------------------------------------------
# Test 12: Missing Tested Register Is Reported, Not Silently Ignored
# ---------------------------------------------------------------------------


def test_register_test_without_operand_is_unsupported() -> None:
    """A register test with no tested-register operand must not fall silent."""
    lifter = SMTBranchLifter(arch="arm64")
    with pytest.raises(UnsupportedInstructionError):
        lifter.lift_branch_condition(
            Instruction(0x3000, 4, "cbz", "", is_jump=True, target=0x4000),
            SymbolicState(arch="arm64"),
            taken=True,
        )


# ---------------------------------------------------------------------------
# Test 13: Unconditional Jump Edges Are Taken, Not Constrained False
# ---------------------------------------------------------------------------


def test_unconditional_jump_edge_is_taken() -> None:
    """An uncond jmp edge must contribute True, never an unsatisfiable edge."""
    insns = [
        Instruction(0x1000, 3, "cmp", "eax, 0x10"),
        Instruction(0x1003, 2, "jne", "0x1010", is_jump=True, target=0x1010),
        Instruction(
            0x1005, 2, "jmp", "0x2000", is_jump=True, is_uncond_jump=True, target=0x2000
        ),
        Instruction(0x1007, 1, "ret", "", is_ret=True),
        Instruction(0x1010, 1, "ret", "", is_ret=True),
        Instruction(0x2000, 1, "ret", "", is_ret=True),
    ]
    res = SMTSolver(arch="x86_64").solve_instructions(insns, target_addr=0x2000)

    assert res.status == "SAT"
    assert res.model is not None
    assert res.model["eax"] == 0x10
