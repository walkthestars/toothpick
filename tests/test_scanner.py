"""
Tests for UNIT-W07: Bounded Memory Scanner (scan).

Covers:
- Test 1: Primitive numeric value scanning (i8, u8, i16, u16, i32, u32, i64, u64, f32, f64).
- Test 2: Wildcard hex byte pattern scanning ("48 89 ? 24 ? 57").
- Test 3: String scanning (ASCII and UTF-8).
- Test 4: Pointer reference scanning with offset tolerance.
- Test 5: Top-K context budget enforcement (10k matches strictly capped to limit) (Risk R1).
- Test 6: Deterministic address sorting across matches (Trap 10).
- Test 7: Iterative filtering (exact, increased, decreased, changed, unchanged).
- Test 8: Session persistence & cross-session retrieval via SQLite toothpick.db.
- Test 9: Mocked FridaMemoryBackend live scanning.
- Test 10: CLI scan on file dump exits 0, returns valid JSON, and persists receipt.
- Test 11: CLI pointer scan mode.
- Test 12: CLI missing target error handling.
- Test 13: CLI mutually exclusive targets error handling.
- Test 14: CLI missing file error handling.
- Test 15: CLI invalid pattern error handling.
- Test 16: Windows SQLite handle closure avoiding file locks (Trap 7).
- Test 17: Multi-binary ambiguous SHA inference refusal (Trap 20).
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, Dict, List

import pytest
from click.testing import CliRunner

from toothpick.cli import cli
from toothpick.scanner import (
    BufferMemoryBackend,
    FridaMemoryBackend,
    InvalidPatternError,
    MemoryScanner,
    ScannerError,
    SessionNotFoundError,
)
from toothpick.storage import Database, FunctionRecord


# ---------------------------------------------------------------------------
# Test Helpers & Mock Frida Classes
# ---------------------------------------------------------------------------


class MockFridaScriptExports:
    def __init__(self, ranges: List[Dict[str, Any]], memory_chunks: Dict[str, bytes]) -> None:
        self._ranges = ranges
        self._chunks = memory_chunks

    def get_ranges(self) -> List[Dict[str, Any]]:
        return self._ranges

    def read(self) -> bytes:
        # Returns the first available chunk for read
        for data in self._chunks.values():
            return data
        return b""


class MockFridaScript:
    def __init__(self, exports: MockFridaScriptExports) -> None:
        self.exports_sync = exports

    def load(self) -> None:
        pass


class MockFridaSession:
    def __init__(self, exports: MockFridaScriptExports) -> None:
        self.exports = exports
        self.detached = False

    def create_script(self, source: str) -> MockFridaScript:
        return MockFridaScript(self.exports)

    def detach(self) -> None:
        self.detached = True


class MockFridaModule:
    def __init__(self, session: MockFridaSession) -> None:
        self.session = session
        self.attached_pids: List[int] = []

    def attach(self, pid: int) -> MockFridaSession:
        self.attached_pids.append(pid)
        return self.session


# ---------------------------------------------------------------------------
# Unit & Property Tests
# ---------------------------------------------------------------------------


def test_scan_numeric_i32() -> None:
    """Scans for 32-bit integers in an in-memory buffer."""
    buf = bytearray(b"\x00" * 1024)
    # Place 1337 (0x00000539) at offsets 0x100 and 0x250
    struct.pack_into("<i", buf, 0x100, 1337)
    struct.pack_into("<i", buf, 0x250, 1337)

    backend = BufferMemoryBackend(bytes(buf), base_address=0x140000000)
    scanner = MemoryScanner()
    result = scanner.first_scan(backend, value=1337, data_type="i32")

    assert result.data_type == "i32"
    assert result.total_matches == 2
    assert result.returned_matches == 2
    assert not result.is_truncated
    assert [m.address for m in result.matches] == [0x140000100, 0x140000250]
    assert [m.address_hex for m in result.matches] == ["0x140000100", "0x140000250"]


def test_scan_numeric_types() -> None:
    """Tests scanning across various integer and float representations."""
    buf = bytearray(b"\x00" * 512)
    struct.pack_into("<b", buf, 0x10, -42)
    struct.pack_into("<H", buf, 0x20, 50000)
    struct.pack_into("<Q", buf, 0x40, 0x1122334455667788)
    struct.pack_into("<f", buf, 0x80, 3.14159)
    struct.pack_into("<d", buf, 0x100, 2.7182818284)

    backend = BufferMemoryBackend(bytes(buf), base_address=0x1000)
    scanner = MemoryScanner()

    # i8
    r_i8 = scanner.first_scan(backend, value=-42, data_type="i8")
    assert r_i8.total_matches == 1
    assert r_i8.matches[0].address == 0x1010

    # u16
    r_u16 = scanner.first_scan(backend, value=50000, data_type="u16")
    assert r_u16.total_matches == 1
    assert r_u16.matches[0].address == 0x1020

    # u64
    r_u64 = scanner.first_scan(backend, value=0x1122334455667788, data_type="u64")
    assert r_u64.total_matches == 1
    assert r_u64.matches[0].address == 0x1040

    # f32
    r_f32 = scanner.first_scan(backend, value=3.14159, data_type="f32")
    assert r_f32.total_matches >= 1
    assert r_f32.matches[0].address == 0x1080

    # f64
    r_f64 = scanner.first_scan(backend, value=2.7182818284, data_type="f64")
    assert r_f64.total_matches == 1
    assert r_f64.matches[0].address == 0x1100


def test_scan_hex_wildcard_pattern() -> None:
    """Scans for byte patterns with wildcards ('?' and '??')."""
    # Pattern: 48 89 5c 24 ? 57 48 83 ec ?
    buf = (
        b"\x90\x90"
        b"\x48\x89\x5c\x24\x08\x57\x48\x83\xec\x20"
        b"\x90\x90"
        b"\x48\x89\x5c\x24\x10\x57\x48\x83\xec\x30"
    )
    backend = BufferMemoryBackend(buf, base_address=0x140001000)
    scanner = MemoryScanner()

    pattern = "48 89 5c 24 ? 57 48 83 ec ?"
    result = scanner.first_scan(backend, value=pattern, data_type="hex")

    assert result.total_matches == 2
    assert result.matches[0].address == 0x140001002
    assert result.matches[1].address == 0x14000100e


def test_scan_string() -> None:
    """Scans for UTF-8 string values in memory."""
    buf = b"HEADER\x00\x00PlayerController\x00\x00SomeOtherData\x00PlayerController\x00"
    backend = BufferMemoryBackend(buf, base_address=0x2000)
    scanner = MemoryScanner()

    result = scanner.first_scan(backend, value="PlayerController", data_type="string")
    assert result.total_matches == 2
    assert result.matches[0].address == 0x2008
    assert result.matches[1].address == 0x2028

    # Empty string raises InvalidPatternError
    with pytest.raises(InvalidPatternError, match="cannot be empty"):
        scanner.first_scan(backend, value="", data_type="string")


def test_scan_pointers() -> None:
    """Scans memory for pointers referencing a target struct address."""
    buf = bytearray(b"\x00" * 512)
    target_struct = 0x140050000
    # Pointer 1: exact base
    struct.pack_into("<Q", buf, 0x10, target_struct)
    # Pointer 2: within offset tolerance (target_struct + 0x30)
    struct.pack_into("<Q", buf, 0x80, target_struct + 0x30)
    # Pointer 3: outside tolerance
    struct.pack_into("<Q", buf, 0x120, target_struct + 0x2000)

    backend = BufferMemoryBackend(bytes(buf), base_address=0x140000000)
    scanner = MemoryScanner()

    result = scanner.scan_pointers(
        backend, target_address=target_struct, offset_tolerance=0x100, pointer_size=8
    )

    assert result.total_matches == 2
    assert result.matches[0].address == 0x140000010
    assert result.matches[1].address == 0x140000080


def test_scan_top_k_limit_cap_enforcement() -> None:
    """Enforces strict top-K budget cap on dense memory fixtures (Risk R1)."""
    # Buffer with 10,000 matches of repeating 32-bit int 0x42
    target_val = 0x42
    packed = struct.pack("<i", target_val)
    buf = packed * 10000  # 40,000 bytes, 10,000 occurrences

    backend = BufferMemoryBackend(buf, base_address=0x100000)
    scanner = MemoryScanner()

    # Request default limit 50
    result = scanner.first_scan(backend, value=target_val, data_type="i32", limit=50)

    assert result.total_matches == 10000
    assert result.returned_matches == 50
    assert len(result.matches) == 50
    assert result.is_truncated is True

    # Request custom limit 150
    result_150 = scanner.first_scan(backend, value=target_val, data_type="i32", limit=150)
    assert result_150.returned_matches == 150
    assert len(result_150.matches) == 150
    assert result_150.is_truncated is True

    # Request limit exceeding ceiling 1,000 is capped to 1,000
    result_over = scanner.first_scan(backend, value=target_val, data_type="i32", limit=5000)
    assert result_over.returned_matches == 1000
    assert len(result_over.matches) == 1000


def test_scan_deterministic_address_sorting() -> None:
    """Matches are sorted strictly deterministically by address (Trap 10)."""
    buf = bytearray(b"\x00" * 1000)
    target = 999
    # Insert matches out of address order
    struct.pack_into("<i", buf, 600, target)
    struct.pack_into("<i", buf, 100, target)
    struct.pack_into("<i", buf, 300, target)

    backend = BufferMemoryBackend(bytes(buf), base_address=0x0)
    scanner = MemoryScanner()
    res = scanner.first_scan(backend, value=target, data_type="i32")

    addresses = [m.address for m in res.matches]
    assert addresses == [100, 300, 600]
    assert addresses == sorted(addresses)


def test_scan_iterative_filtering() -> None:
    """Simulates multi-step variable filtering (health: 100 -> 95 -> 70)."""
    buf = bytearray(b"\x00" * 1024)
    # Three variables initially at 100
    struct.pack_into("<i", buf, 0x100, 100)  # Real health
    struct.pack_into("<i", buf, 0x200, 100)  # Max health (will stay 100)
    struct.pack_into("<i", buf, 0x300, 100)  # Score (will increase to 150)

    backend = BufferMemoryBackend(buf, base_address=0x140000000)
    scanner = MemoryScanner()

    # Step 1: Initial scan for 100
    r1 = scanner.first_scan(backend, value=100, data_type="i32")
    assert r1.total_matches == 3
    session_id = r1.session_id

    # Mutate memory:
    struct.pack_into("<i", buf, 0x100, 95)   # Decreased to 95
    struct.pack_into("<i", buf, 0x200, 100)  # Unchanged
    struct.pack_into("<i", buf, 0x300, 150)  # Increased

    # Step 2: Filter for decreased values
    r2 = scanner.filter_scan(session_id, comparison="decreased")
    assert r2.total_matches == 1
    assert r2.matches[0].address == 0x140000100
    assert r2.matches[0].value == 95

    # Mutate memory again:
    struct.pack_into("<i", buf, 0x100, 70)

    # Step 3: Exact filter for 70
    r3 = scanner.filter_scan(session_id, new_value=70, comparison="exact")
    assert r3.total_matches == 1
    assert r3.matches[0].address == 0x140000100
    assert r3.matches[0].value == 70


def test_scan_session_persistence_in_db(tmp_path: Path) -> None:
    """Scan sessions persist in SQLite receipts table and recover across scanner instances."""
    db_path = tmp_path / "scan_persist.db"
    db = Database(db_path, create=True)

    buf = bytearray(b"\x00" * 512)
    struct.pack_into("<i", buf, 0x50, 42)
    struct.pack_into("<i", buf, 0x100, 42)
    backend = BufferMemoryBackend(bytes(buf), base_address=0x1000)

    # 1. Run first_scan with DB attached
    with MemoryScanner(db=db) as scanner1:
        res1 = scanner1.first_scan(backend, value=42, data_type="i32")
        session_id = res1.session_id

    # 2. Open completely new scanner with DB, no in-memory session cache
    # Mutate memory: offset 0x50 changes to 99, 0x100 stays 42
    buf2 = bytearray(bytes(buf))
    struct.pack_into("<i", buf2, 0x50, 99)
    backend2 = BufferMemoryBackend(bytes(buf2), base_address=0x1000)

    with MemoryScanner(db=db) as scanner2:
        # Re-filters against session loaded from SQLite
        res2 = scanner2.filter_scan(
            session_id=session_id, new_value=99, comparison="exact", backend=backend2
        )
        assert res2.total_matches == 1
        assert res2.matches[0].address == 0x1050

    db.close()


def test_scan_frida_memory_backend_mock() -> None:
    """Mocked FridaMemoryBackend queries ranges and reads memory blocks."""
    test_chunk = b"\x00" * 16 + b"\x39\x05\x00\x00" + b"\x00" * 16
    mock_ranges = [{"base": "0x140000000", "size": len(test_chunk), "protection": "rw-", "file": ""}]
    mock_chunks = {"0x140000000": test_chunk}

    exports = MockFridaScriptExports(mock_ranges, mock_chunks)
    session = MockFridaSession(exports)
    frida_mod = MockFridaModule(session)

    backend = FridaMemoryBackend(pid=1234, frida_module=frida_mod)
    scanner = MemoryScanner()
    res = scanner.first_scan(backend, value=1337, data_type="i32")

    assert res.total_matches == 1
    assert res.matches[0].address == 0x140000010
    backend.close()
    assert session.detached is True


def test_scan_cli_file_dump(tmp_path: Path) -> None:
    """toothpick scan CLI on raw memory dump file returns valid JSON and persists receipt."""
    dump_path = tmp_path / "game_dump.raw"
    db_path = tmp_path / "cli_scan.db"
    db = Database(db_path, create=True)
    db.close()

    buf = bytearray(b"\x00" * 256)
    struct.pack_into("<i", buf, 0x40, 777)
    dump_path.write_bytes(bytes(buf))

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "scan",
            "777",
            "--file",
            str(dump_path),
            "--type",
            "i32",
            "--base",
            "0x140000000",
            "--db",
            str(db_path),
        ],
    )

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["total_matches"] == 1
    assert data["matches"][0]["address_hex"] == "0x140000040"
    assert data["matches"][0]["value"] == 777

    # Confirm receipt was written into SQLite
    db_check = Database(db_path, create=False)
    receipts = db_check.get_receipts("", func_addr=0x140000040)
    assert len(receipts) == 1
    assert receipts[0]["phase"] == "scan"
    db_check.close()


def test_scan_cli_pointer_search(tmp_path: Path) -> None:
    """toothpick scan CLI in pointer search mode."""
    dump_path = tmp_path / "pointer_dump.raw"
    buf = bytearray(b"\x00" * 128)
    struct.pack_into("<Q", buf, 0x20, 0x140010000)
    dump_path.write_bytes(bytes(buf))

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "scan",
            "0x140010000",
            "--type",
            "pointer",
            "--file",
            str(dump_path),
            "--base",
            "0x140000000",
        ],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["total_matches"] == 1
    assert data["matches"][0]["address_hex"] == "0x140000020"


def test_scan_cli_error_missing_targets() -> None:
    """Omitting --file, --pid, and --session fails with code 1."""
    runner = CliRunner()
    res = runner.invoke(cli, ["scan", "100"])
    assert res.exit_code == 1
    assert "Error: Must specify either --file, --pid, or an active --session" in res.output


def test_scan_cli_error_mutually_exclusive_targets(tmp_path: Path) -> None:
    """Specifying both --file and --pid fails with code 1."""
    dump_file = tmp_path / "dump.raw"
    dump_file.write_bytes(b"\x00" * 10)
    runner = CliRunner()
    res = runner.invoke(cli, ["scan", "100", "--file", str(dump_file), "--pid", "1234"])
    assert res.exit_code == 1
    assert "Error: Mutually exclusive targets" in res.output


def test_scan_cli_error_missing_file(tmp_path: Path) -> None:
    """Nonexistent --file path fails with code 1."""
    nonexistent = tmp_path / "missing.raw"
    runner = CliRunner()
    res = runner.invoke(cli, ["scan", "100", "--file", str(nonexistent)])
    assert res.exit_code == 1
    assert "Error: target file not found" in res.output


def test_scan_cli_error_invalid_pattern(tmp_path: Path) -> None:
    """Malformed numeric pattern fails with code 1."""
    dump_file = tmp_path / "dump.raw"
    dump_file.write_bytes(b"\x00" * 10)
    runner = CliRunner()
    res = runner.invoke(cli, ["scan", "not_a_number", "--type", "i32", "--file", str(dump_file)])
    assert res.exit_code == 1
    assert "Error: Cannot parse" in res.output


def test_scan_context_manager_locks(tmp_path: Path) -> None:
    """MemoryScanner context manager cleanly closes SQLite handles (Trap 7)."""
    db_path = tmp_path / "trap7_scanner.db"
    db = Database(db_path, create=True)
    db.close()

    backend = BufferMemoryBackend(b"\x00" * 10)
    with MemoryScanner(db=db_path) as scanner:
        _ = scanner.first_scan(backend, value=0, data_type="i8")

    db_path.unlink()
    assert not db_path.exists()


def test_scan_ambiguous_db_sha_inference(tmp_path: Path) -> None:
    """Multi-binary DB raises ScannerError on ambiguous SHA inference (Trap 20)."""
    db_path = tmp_path / "ambiguous_scan.db"
    db = Database(db_path, create=True)
    for sha in ("sha_a", "sha_b"):
        db.upsert_function(
            FunctionRecord(
                binary_sha256=sha,
                func_addr=0x1000,
                name=f"func_{sha}",
                size=16,
            )
        )

    backend = BufferMemoryBackend(b"\x00" * 10)
    with MemoryScanner(db=db) as scanner:
        with pytest.raises(ScannerError, match="Cannot infer binary SHA-256"):
            scanner.first_scan(backend, value=0, data_type="i8")

    db.close()


def test_scan_session_not_found_error() -> None:
    """Nonexistent session ID in filter_scan raises SessionNotFoundError."""
    scanner = MemoryScanner()
    with pytest.raises(SessionNotFoundError, match="Active scan session not found: missing_id"):
        scanner.filter_scan(session_id="missing_id", new_value=100)

