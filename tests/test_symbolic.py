"""W16+1 symbolic tests: dynamic taint analysis & Z3 solving (UNIT-W17).

All fixtures are inline x86_64 byte stubs disassembled with the module's own
:func:`toothpick.symbolic.disassemble_bytes` — no external binaries, no
Triton/Manticore installs (exporters are pure text emitters).
"""

from __future__ import annotations

from toothpick.symbolic import (
    SymbolicExecutionEngine,
    TaintEngine,
    disassemble_bytes,
)

_BASE = 0x1000

#: ``mov rax, rsi ; add rax, 1`` — taint flows rsi -> rax.
_MOV_ADD = bytes.fromhex("4889f04883c001")

#: ``mov rax, rsi ; xor rax, rax ; mov rbx, rax`` — zeroing clears the taint
#: before the final mov reads rax.
_MOV_XOR = bytes.fromhex("4889f04831c04889c3")

#: ``mov [rsp], rsi ; mov rax, [rsp]`` — taint round-trips through the stack.
_STACK_ROUNDTRIP = bytes.fromhex("48893424488b0424")

#: ``mov rax, rsi ; cmp rax, 0`` — the cmp reads tainted rax.
_MOV_CMP = bytes.fromhex("4889f04883f800")

#: ``cmp rdi, 5 ; je taken ; mov eax, 0 ; ret ; taken: mov eax, 1 ; ret``.
_BRANCH = bytes.fromhex(
    "4883ff057406"  # cmp rdi, 5 / je +6
    "b800000000"  # mov eax, 0
    "c3"  # ret
    "b801000000"  # taken: mov eax, 1
    "c3"  # ret
)
_TAKEN_ADDR = _BASE + 12


def test_register_taint_propagates_across_mov_and_add() -> None:
    """Taint seeded at ``rsi`` reaches ``rax`` through mov/add."""
    engine = TaintEngine()
    insns = disassemble_bytes(_MOV_ADD, base_addr=_BASE)
    assert [view.mnemonic for view in insns] == ["mov", "add"]
    report = engine.analyze_function(insns, source="rsi", sink_addr=_BASE + 3)
    assert report.is_sink_tainted
    assert report.taint_propagation_path, "expected a propagation path"
    assert any("rax" in var for var in report.tainted_variables)


def test_xor_self_clears_taint() -> None:
    """``xor rax, rax`` kills the taint despite the tainted source."""
    engine = TaintEngine()
    insns = disassemble_bytes(_MOV_XOR, base_addr=_BASE)
    assert [view.mnemonic for view in insns] == ["mov", "xor", "mov"]
    report = engine.analyze_function(insns, source="rsi", sink_addr=_BASE + 6)
    assert not report.is_sink_tainted
    assert not engine.state.is_reg_tainted("rax")


def test_memory_load_store_taint_round_trip() -> None:
    """Taint stored to ``[rsp]`` is picked back up by the load into rax."""
    engine = TaintEngine()
    insns = disassemble_bytes(_STACK_ROUNDTRIP, base_addr=_BASE)
    report = engine.analyze_function(insns, source="rsi", sink_addr=_BASE + 4)
    assert report.is_sink_tainted
    assert engine.state.is_reg_tainted("rax")


def test_taint_sink_detection_at_conditional_use() -> None:
    """An instruction consuming tainted state at the sink address hits."""
    engine = TaintEngine()
    insns = disassemble_bytes(_MOV_CMP, base_addr=_BASE)
    sink = _BASE + 3
    report = engine.analyze_function(insns, source="rsi", sink_addr=sink)
    assert report.is_sink_tainted
    assert sink in report.sink_hits

    # The same sink with an untainted source stays clean.
    clean = TaintEngine().analyze_function(insns, source="rbx", sink_addr=sink)
    assert not clean.is_sink_tainted


def test_symbolic_witness_input_recovery() -> None:
    """Z3 recovers the input reaching the taken branch (rdi == 5)."""
    engine = SymbolicExecutionEngine(
        code_bytes=_BRANCH, base_addr=_BASE, arch="x86_64"
    )
    result = engine.solve_target(
        entry_addr=_BASE, target_addr=_TAKEN_ADDR, taint_source="rdi"
    )
    assert result.status == "SAT", f"expected SAT, got {result}"
    assert result.model, "expected a witness model"
    assert result.model.get("rdi") == 5
    assert result.taint_reached
