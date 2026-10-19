"""W16 DBI tracer tests: the Unicorn execution engine (UNIT-W16 chunk 3).

The x86_64 stubs below are assembled inline as raw machine-code bytes -- no
external assembler, no checked-in target binaries -- so the suite stays
dependency-free apart from the declared Unicorn/Capstone engines and exercises
the ``UC_HOOK_CODE`` / ``UC_HOOK_MEM_READ`` / ``UC_HOOK_MEM_WRITE`` paths of
:class:`toothpick.dbi.UnicornDbiTracer`.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from toothpick.dbi import (
    ENGINE_UNICORN,
    MEMORY_READ,
    MEMORY_WRITE,
    STOP_INSTRUCTION_BUDGET,
    STOP_RETURN_SENTINEL,
    UnicornDbiTracer,
)

#: ``mov rax, 5 ; add rax, 2 ; ret`` (7 + 4 + 1 bytes, x86_64).
_MOV_ADD_RET_STUB = bytes.fromhex("48c7c0050000004883c002c3")

#: ``push rbp ; mov rbp, rsp ; sub rsp, 0x10 ; mov dword ptr [rbp-4], 0x2a ;
#: mov eax, dword ptr [rbp-4] ; leave ; pop rbp ; ret`` (x86_64).
_STACK_OPS_STUB = bytes.fromhex("554889e54883ec10c745fc2a0000008b45fcc95dc3")

#: Load address the tracer maps a raw code blob at.
_FLAT_CODE_BASE = 0x400000


def test_trace_synthetic_x86_64_stub_yields_instructions_and_basic_blocks(
    tmp_path: Path,
) -> None:
    """A mov/add/ret stub traces to a positive count and non-empty BB list."""
    tracer = UnicornDbiTracer()
    result = tracer.trace(_MOV_ADD_RET_STUB)

    assert result.engine == ENGINE_UNICORN
    assert result.total_instructions > 0
    assert result.executed_bbs
    # mov, add, ret -- then the synthetic sentinel page absorbs the return.
    assert result.total_instructions == 3
    assert result.executed_bbs[0] == _FLAT_CODE_BASE
    assert [entry.mnemonic for entry in tracer.last_instructions] == [
        "mov",
        "add",
        "ret",
    ]
    assert tracer.last_stop_reason == STOP_RETURN_SENTINEL
    assert result.target_sha256 == hashlib.sha256(_MOV_ADD_RET_STUB).hexdigest()

    # The same stub handed over as a file path traces identically.
    stub_path = tmp_path / "stub_x86_64.bin"
    stub_path.write_bytes(_MOV_ADD_RET_STUB)
    from_file = tracer.trace(stub_path)
    assert from_file.engine == ENGINE_UNICORN
    assert from_file.total_instructions == result.total_instructions
    assert from_file.target_sha256 == hashlib.sha256(stub_path.read_bytes()).hexdigest()

    # ... and the instruction budget bounds the run instead of letting it spin.
    budgeted_tracer = UnicornDbiTracer(max_instructions=2)
    budgeted = budgeted_tracer.trace(_MOV_ADD_RET_STUB)
    assert budgeted.total_instructions == 2
    assert budgeted_tracer.last_stop_reason == STOP_INSTRUCTION_BUDGET


def test_trace_stack_ops_stub_records_memory_accesses() -> None:
    """The memory hooks record the stack traffic of a push/mov/leave stub."""
    tracer = UnicornDbiTracer()
    result = tracer.trace(_STACK_OPS_STUB)

    assert result.total_instructions > 0
    assert result.memory_accesses, "expected at least one recorded memory access"
    assert all(
        entry.kind in (MEMORY_READ, MEMORY_WRITE) for entry in result.memory_accesses
    )

    writes = result.memory_writes
    reads = result.memory_reads
    assert writes, "expected the stack store (push/mov) to be recorded as a write"
    assert reads, "expected the stack loads (mov/leave/pop/ret) to be recorded as reads"

    # ``mov dword ptr [rbp - 4], 0x2a`` -- the written value is captured.
    stored = [entry for entry in writes if entry.size == 4 and entry.value]
    assert stored, "expected the 4-byte stack store to carry its written value"
    assert stored[0].value[:4] == b"\x2a\x00\x00\x00"
    assert stored[0].address == _FLAT_CODE_BASE + 0x08
    assert result.total_instructions == len(tracer.last_instructions)
