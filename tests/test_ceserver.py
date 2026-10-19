"""
Tests for UNIT-W14: CEServer Memory Bridge & Client (toothpick.ceserver).

Validates:
1. TCP binary handshake and version retrieval (CMD_GETVERSION).
2. Process handle acquisition, closure, and invalid PID rejection.
3. Memory read and write fidelity over TCP socket stream.
4. VirtualQueryEx region querying and full address space enumeration.
5. Full integration with Bounded Memory Scanner (MemoryScanner) for first_scan and filter_scan.
6. CLI scan invocation using --backend ceserver.
7. CLI ground invocation using --backend ceserver and receipt persistence into database.
8. Connection failure and timeout error handling (CEServerConnectionError).
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest
from click.testing import CliRunner

from toothpick.ceserver import (
    CEServerClient,
    CEServerConnectionError,
    CEServerMemoryBackend,
    CEServerProtocolError,
    MockCEServer,
)
from toothpick.cli import cli
from toothpick.scanner import MemoryRegion, MemoryScanner
from toothpick.storage import Database

TEST_PID = 1234
TEST_BASE = 0x140000000


@pytest.fixture
def mock_server() -> MockCEServer:
    """Fixture providing an active MockCEServer with pre-populated process memory."""
    server = MockCEServer()
    # 4KB buffer with known integer values
    buf = bytearray(b"\x00" * 4096)
    struct.pack_into("<i", buf, 0x100, 1337)      # Value 1337 at offset 0x100 (addr 0x140000100)
    struct.pack_into("<i", buf, 0x200, 42)        # Value 42 at offset 0x200
    struct.pack_into("<f", buf, 0x300, 99.5)      # Float 99.5 at offset 0x300
    server.add_process(
        pid=TEST_PID,
        memory=bytes(buf),
        base_address=TEST_BASE,
        protection="rw-",
    )
    server.start()
    yield server
    server.stop()


@pytest.fixture
def test_db(tmp_path: Path) -> Database:
    """Isolated SQLite database fixture."""
    db_file = tmp_path / "test_ceserver.db"
    db = Database(db_file, create=True)
    yield db
    db.close()


def test_mock_ceserver_handshake(mock_server: MockCEServer) -> None:
    """Verify TCP handshake and version query."""
    with CEServerClient() as client:
        client.connect(host="127.0.0.1", port=mock_server.port, timeout=2.0)
        assert client.connected
        version = client.get_version()
        assert version == 1


def test_process_open_and_handle(mock_server: MockCEServer) -> None:
    """Verify opening and closing process handles."""
    with CEServerClient() as client:
        client.connect(host="127.0.0.1", port=mock_server.port, timeout=2.0)

        # Valid PID
        handle = client.open_process(TEST_PID)
        assert handle > 0

        # Architecture check
        arch = client.get_architecture(handle)
        assert arch == 1  # x86_64

        # Close handle
        closed = client.close_handle(handle)
        assert closed

        # Invalid PID
        with pytest.raises(CEServerProtocolError):
            client.open_process(99999)


def test_read_and_write_memory(mock_server: MockCEServer) -> None:
    """Verify memory reading and writing fidelity."""
    with CEServerClient() as client:
        client.connect(host="127.0.0.1", port=mock_server.port, timeout=2.0)
        handle = client.open_process(TEST_PID)

        # Read integer at 0x140000100
        val_bytes = client.read_memory(handle, TEST_BASE + 0x100, 4)
        assert len(val_bytes) == 4
        assert struct.unpack("<i", val_bytes)[0] == 1337

        # Write new integer
        written = client.write_memory(handle, TEST_BASE + 0x100, struct.pack("<i", 8888))
        assert written == 4

        # Re-read
        new_bytes = client.read_memory(handle, TEST_BASE + 0x100, 4)
        assert struct.unpack("<i", new_bytes)[0] == 8888

        client.close_handle(handle)


def test_virtual_query_and_enumeration(mock_server: MockCEServer) -> None:
    """Verify virtual query and region enumeration."""
    with CEServerClient() as client:
        client.connect(host="127.0.0.1", port=mock_server.port, timeout=2.0)
        handle = client.open_process(TEST_PID)

        # Query region at base
        reg = client.virtual_query(handle, TEST_BASE)
        assert reg is not None
        assert reg.base == TEST_BASE
        assert reg.size == 4096
        assert "rw" in reg.protection

        # Query beyond mapped space
        reg_oob = client.virtual_query(handle, TEST_BASE + 0x100000)
        assert reg_oob is None

        # Enumerate all regions
        regions = client.enumerate_regions(handle)
        assert len(regions) == 1
        assert regions[0].base == TEST_BASE

        client.close_handle(handle)


def test_ceserver_memory_backend_with_scanner(
    mock_server: MockCEServer, test_db: Database
) -> None:
    """Verify MemoryScanner integration with CEServerMemoryBackend."""
    with CEServerMemoryBackend(
        host="127.0.0.1", port=mock_server.port, pid=TEST_PID
    ) as backend:
        with MemoryScanner(db=test_db) as scanner:
            # First scan for 1337
            res = scanner.first_scan(
                backend=backend,
                value="1337",
                data_type="i32",
                limit=10,
            )
            assert res.total_matches == 1
            assert res.matches[0].address == TEST_BASE + 0x100
            assert res.matches[0].value == 1337

            # Refine scan after modifying value
            backend.write_memory(TEST_BASE + 0x100, struct.pack("<i", 1400))
            res_refine = scanner.filter_scan(
                session_id=res.session_id,
                new_value="1400",
                comparison="exact",
                limit=10,
                backend=backend,
            )
            assert res_refine.total_matches == 1
            assert res_refine.matches[0].value == 1400


def test_cli_scan_ceserver(mock_server: MockCEServer, test_db: Database) -> None:
    """Verify CLI scan command with --backend ceserver."""
    runner = CliRunner()
    res = runner.invoke(
        cli,
        [
            "scan",
            "1337",
            "--backend",
            "ceserver",
            "--host",
            "127.0.0.1",
            "--port",
            str(mock_server.port),
            "--pid",
            str(TEST_PID),
            "--type",
            "i32",
            "--limit",
            "10",
        ],
    )
    assert res.exit_code == 0
    data = json.loads(res.output)
    assert data["total_matches"] == 1
    assert data["matches"][0]["address_hex"] == f"0x{TEST_BASE + 0x100:x}"


def test_cli_ground_ceserver(mock_server: MockCEServer, test_db: Database) -> None:
    """Verify CLI ground command with --backend ceserver and database receipt storage."""
    runner = CliRunner()
    target_addr = f"0x{TEST_BASE + 0x100:x}"
    res = runner.invoke(
        cli,
        [
            "ground",
            target_addr,
            "--backend",
            "ceserver",
            "--host",
            "127.0.0.1",
            "--port",
            str(mock_server.port),
            "--pid",
            str(TEST_PID),
            "--db",
            str(test_db.db_path),
        ],
    )
    assert res.exit_code == 0
    data = json.loads(res.output)
    assert data["ground_type"] == "CESERVER_LIVE"
    assert data["verifier_status"] == "GROUNDED"
    assert data["live_reads"]["backend"] == "ceserver"

    # Verify receipt persisted in SQLite
    receipts = test_db.get_receipts(binary_sha256="", func_addr=TEST_BASE + 0x100)
    assert len(receipts) == 1
    assert receipts[0]["phase"] == "ground"


def test_connection_failure_and_timeout() -> None:
    """Verify connection failure on dead port raises CEServerConnectionError."""
    with CEServerClient() as client:
        # Non-listening port
        with pytest.raises(CEServerConnectionError):
            client.connect(host="127.0.0.1", port=59998, timeout=0.5)


def test_protection_mapping_variants(mock_server: MockCEServer) -> None:
    """Verify decoding of different memory protection attributes including PAGE_NOACCESS."""
    mock_server.add_process(
        pid=9999,
        memory=b"\x00" * 0x1000,
        base_address=0x200000000,
        protection="---",
    )
    with CEServerClient() as client:
        client.connect(host="127.0.0.1", port=mock_server.port, timeout=2.0)
        h = client.open_process(9999)
        reg = client.virtual_query(h, 0x200000000)
        assert reg is not None
        assert reg.protection == "---"
        client.close_handle(h)


def test_disjoint_regions_enumeration_with_gaps(mock_server: MockCEServer) -> None:
    """Verify enumerate_regions discovers disjoint memory regions separated by unmapped gaps."""
    proc_pid = 7777
    r1_base = 0x10000
    r1_size = 0x1000
    r2_base = 0x1000000  # 16 MB boundary
    r2_size = 0x2000
    mock_server.add_process(
        pid=proc_pid,
        memory=b"\x00" * 0x1000,
        base_address=r1_base,
        regions=[
            MemoryRegion(base=r1_base, size=r1_size, protection="rw-", name="r1"),
            MemoryRegion(base=r2_base, size=r2_size, protection="r-x", name="r2"),
        ],
    )
    with CEServerClient() as client:
        client.connect(host="127.0.0.1", port=mock_server.port, timeout=2.0)
        h = client.open_process(proc_pid)

        # Probing addr=0 (unmapped null-page gap before r1) returns r1
        reg0 = client.virtual_query(h, 0)
        assert reg0 is not None
        assert reg0.base == r1_base

        # Probing within the 16MB gap between r1 and r2 returns r2
        reg_gap = client.virtual_query(h, 0x50000)
        assert reg_gap is not None
        assert reg_gap.base == r2_base

        # Probing beyond r2 returns None
        reg_past = client.virtual_query(h, 0x2000000)
        assert reg_past is None

        # Full enumeration discovers both regions across unmapped gaps
        all_regs = client.enumerate_regions(h)
        assert len(all_regs) == 2
        assert all_regs[0].base == r1_base
        assert all_regs[1].base == r2_base

        client.close_handle(h)


def test_injected_client_autoconnect(mock_server: MockCEServer) -> None:
    """Verify CEServerMemoryBackend auto-connects an injected client that is not yet connected."""
    client = CEServerClient()
    assert not client.connected
    backend = CEServerMemoryBackend(
        host="127.0.0.1",
        port=mock_server.port,
        pid=TEST_PID,
        client=client,
    )
    with backend:
        assert client.connected
        regs = backend.enumerate_regions()
        assert len(regs) > 0


def test_cli_ground_ceserver_mutual_exclusivity() -> None:
    """Verify CLI rejects conflicting --spawn or --static-only flags with --backend ceserver."""
    runner = CliRunner()
    res_spawn = runner.invoke(
        cli,
        [
            "ground",
            "0x140000000",
            "--backend",
            "ceserver",
            "--spawn",
            "target.exe",
        ],
    )
    assert res_spawn.exit_code != 0
    assert "cannot be used with --backend ceserver" in res_spawn.output

    res_static = runner.invoke(
        cli,
        [
            "ground",
            "0x140000000",
            "--backend",
            "ceserver",
            "--static-only",
        ],
    )
    assert res_static.exit_code != 0
    assert "cannot be used with --backend ceserver" in res_static.output



