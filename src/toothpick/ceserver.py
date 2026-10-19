"""
Toothpick CEServer memory backend and protocol client (unit W14).

Implements the W14 CEServer bridge specification per UNIT-W14-CESERVER-PLAN.md,
design-document.md §4, §9, §11, §16 #1, and Risk R1/R5:
- Binary TCP client for Cheat Engine Server (CEServer) protocol.
- Command codecs: CMD_GETVERSION (0), CMD_CLOSECONNECTION (1), CMD_OPENPROCESS (3),
  CMD_CLOSEHANDLE (4), CMD_READPROCESSMEMORY (5), CMD_WRITEPROCESSMEMORY (6),
  CMD_VIRTUALQUERYEX (7), CMD_GETARCHITECTURE (10).
- CEServerMemoryBackend implementing MemoryBackend for MemoryScanner and GroundingHarness.
- MockCEServer: Pure-Python threaded loopback daemon for zero-external-dependency CI testing.
"""

from __future__ import annotations

import select
import socket
import struct
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from toothpick.scanner import (
    MemoryBackend,
    MemoryReadError,
    MemoryRegion,
    ScannerError,
)

__all__ = [
    "CMD_GETVERSION",
    "CMD_CLOSECONNECTION",
    "CMD_OPENPROCESS",
    "CMD_CLOSEHANDLE",
    "CMD_READPROCESSMEMORY",
    "CMD_WRITEPROCESSMEMORY",
    "CMD_VIRTUALQUERYEX",
    "CMD_GETARCHITECTURE",
    "CEServerError",
    "CEServerConnectionError",
    "CEServerProtocolError",
    "CEServerClient",
    "CEServerMemoryBackend",
    "MockProcess",
    "MockCEServer",
]

# ---------------------------------------------------------------------------
# CEServer Protocol Command Codes
# ---------------------------------------------------------------------------

CMD_GETVERSION = 0
CMD_CLOSECONNECTION = 1
CMD_OPENPROCESS = 3
CMD_CLOSEHANDLE = 4
CMD_READPROCESSMEMORY = 5
CMD_WRITEPROCESSMEMORY = 6
CMD_VIRTUALQUERYEX = 7
CMD_GETARCHITECTURE = 10

# Windows / CE Memory Protection Constants
PAGE_NOACCESS = 0x01
PAGE_READONLY = 0x02
PAGE_READWRITE = 0x04
PAGE_EXECUTE = 0x10
PAGE_EXECUTE_READ = 0x20
PAGE_EXECUTE_READWRITE = 0x40


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class CEServerError(ScannerError):
    """Base exception for CEServer client and protocol failures."""


class CEServerConnectionError(CEServerError):
    """Raised when connecting to CEServer fails or times out."""


class CEServerProtocolError(CEServerError):
    """Raised when CEServer returns invalid data or unexpected framing."""


# ---------------------------------------------------------------------------
# CEServer Binary Protocol Client
# ---------------------------------------------------------------------------


class CEServerClient:
    """Synchronous binary TCP client for Cheat Engine Server (ceserver)."""

    def __init__(self) -> None:
        self.sock: Optional[socket.socket] = None
        self.host: str = "127.0.0.1"
        self.port: int = 52736
        self.timeout: float = 5.0
        self.connected: bool = False

    def connect(
        self, host: str = "127.0.0.1", port: int = 52736, timeout: float = 5.0
    ) -> None:
        """Connect to remote CEServer daemon."""
        self.host = host
        self.port = port
        self.timeout = timeout
        self.close()
        s: Optional[socket.socket] = None
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(timeout)
            s.connect((host, port))
            self.sock = s
            self.connected = True
        except Exception as exc:
            if s is not None:
                try:
                    s.close()
                except Exception:
                    pass
            self.close()
            raise CEServerConnectionError(
                f"Failed to connect to CEServer at {host}:{port}: {exc}"
            ) from exc

    def close(self) -> None:
        """Close socket connection and release handles."""
        if self.sock is not None:
            try:
                if self.connected:
                    # Best-effort graceful disconnect notification
                    self.sock.sendall(struct.pack("<B", CMD_CLOSECONNECTION))
            except Exception:
                pass
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None
        self.connected = False

    def __enter__(self) -> CEServerClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _send_exact(self, data: bytes) -> None:
        """Send complete buffer to server."""
        if not self.connected or self.sock is None:
            raise CEServerConnectionError("Not connected to CEServer")
        try:
            self.sock.sendall(data)
        except Exception as exc:
            self.close()
            raise CEServerConnectionError(f"Socket send failed: {exc}") from exc

    def _recv_exact(self, size: int) -> bytes:
        """Receive exactly `size` bytes from server, looping on partial reads."""
        if not self.connected or self.sock is None:
            raise CEServerConnectionError("Not connected to CEServer")
        buf = bytearray()
        while len(buf) < size:
            try:
                chunk = self.sock.recv(size - len(buf))
            except socket.timeout as exc:
                self.close()
                raise CEServerConnectionError(f"Socket receive timed out: {exc}") from exc
            except Exception as exc:
                self.close()
                raise CEServerConnectionError(f"Socket receive failed: {exc}") from exc

            if not chunk:
                self.close()
                raise CEServerConnectionError(
                    f"Connection closed by server while waiting for {size - len(buf)} bytes"
                )
            buf.extend(chunk)
        return bytes(buf)

    def get_version(self) -> int:
        """Query server protocol version (CMD_GETVERSION)."""
        self._send_exact(struct.pack("<B", CMD_GETVERSION))
        raw = self._recv_exact(4)
        (version,) = struct.unpack("<I", raw)
        return version

    def open_process(self, pid: int) -> int:
        """Open a process handle by PID (CMD_OPENPROCESS). Returns handle > 0."""
        self._send_exact(struct.pack("<Bi", CMD_OPENPROCESS, int(pid)))
        raw = self._recv_exact(4)
        (handle,) = struct.unpack("<i", raw)
        if handle <= 0:
            raise CEServerProtocolError(f"CEServer failed to open PID {pid} (handle={handle})")
        return handle

    def close_handle(self, handle: int) -> bool:
        """Close an opened process handle (CMD_CLOSEHANDLE)."""
        self._send_exact(struct.pack("<Bi", CMD_CLOSEHANDLE, int(handle)))
        raw = self._recv_exact(4)
        (res,) = struct.unpack("<i", raw)
        return res != 0

    def read_memory(self, handle: int, address: int, size: int) -> bytes:
        """Read bytes from remote process memory (CMD_READPROCESSMEMORY)."""
        if size <= 0:
            return b""
        self._send_exact(
            struct.pack("<BiQI", CMD_READPROCESSMEMORY, int(handle), int(address), int(size))
        )
        raw_len = self._recv_exact(4)
        (bytes_read,) = struct.unpack("<I", raw_len)
        if bytes_read == 0:
            return b""
        return self._recv_exact(bytes_read)

    def write_memory(self, handle: int, address: int, data: bytes) -> int:
        """Write bytes to remote process memory (CMD_WRITEPROCESSMEMORY)."""
        if not data:
            return 0
        hdr = struct.pack(
            "<BiQI", CMD_WRITEPROCESSMEMORY, int(handle), int(address), len(data)
        )
        self._send_exact(hdr + data)
        raw_written = self._recv_exact(4)
        (bytes_written,) = struct.unpack("<I", raw_written)
        return bytes_written

    def virtual_query(self, handle: int, address: int) -> Optional[MemoryRegion]:
        """Query memory region attributes at address (CMD_VIRTUALQUERYEX).

        Per CEServer / VirtualQueryEx protocol specification, returns the memory region
        containing `address`, or the next mapped region at or above `address` when `address`
        falls within an unmapped gap. Returns None only when address is beyond all mapped memory.
        """
        self._send_exact(
            struct.pack("<BiQ", CMD_VIRTUALQUERYEX, int(handle), int(address))
        )
        (found,) = struct.unpack("<B", self._recv_exact(1))
        if found == 0:
            return None

        # Unpack base (8), size (8), protection (4), type (4)
        payload = self._recv_exact(24)
        base, size, prot_int, _ = struct.unpack("<QQII", payload)

        base_prot = prot_int & 0xFF
        if base_prot in (PAGE_EXECUTE_READWRITE, 0x80):
            prot_str = "rwx"
        elif base_prot == PAGE_EXECUTE_READ:
            prot_str = "r-x"
        elif base_prot == PAGE_EXECUTE:
            prot_str = "--x"
        elif base_prot in (PAGE_READWRITE, 0x08):
            prot_str = "rw-"
        elif base_prot == PAGE_READONLY:
            prot_str = "r--"
        elif base_prot == PAGE_NOACCESS:
            prot_str = "---"
        else:
            # Fallback to readable so unclassified mapped regions remain inspectable
            prot_str = "r--"

        return MemoryRegion(base=base, size=size, protection=prot_str, name="ceserver")

    def enumerate_regions(
        self, handle: int, max_regions: int = 1000
    ) -> List[MemoryRegion]:
        """Enumerate virtual memory regions within remote process.

        Starts querying from address 0. Because virtual_query returns the next mapped region
        at or above the queried address, unmapped gaps (including the null page gap at 0)
        are automatically skipped by jumping to each region's base + size. Enumeration
        terminates when virtual_query returns None (indicating no further mapped regions
        exist) or when max_regions is reached.
        """
        regions: List[MemoryRegion] = []
        addr = 0
        while len(regions) < max_regions:
            reg = self.virtual_query(handle, addr)
            if reg is None:
                break
            regions.append(reg)
            next_addr = reg.base + reg.size
            if next_addr <= addr or next_addr >= 0x7FFFFFFFFFFFFFFF:
                break
            addr = next_addr
        return regions

    def get_architecture(self, handle: int) -> int:
        """Query remote target architecture (0=x86, 1=x86_64, 2=arm, 3=arm64)."""
        self._send_exact(struct.pack("<Bi", CMD_GETARCHITECTURE, int(handle)))
        (arch,) = struct.unpack("<B", self._recv_exact(1))
        return arch


# ---------------------------------------------------------------------------
# CEServer Memory Backend (MemoryBackend integration)
# ---------------------------------------------------------------------------


class CEServerMemoryBackend(MemoryBackend):
    """Memory backend connecting to CEServer for live remote scanning."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 52736,
        pid: int = 0,
        client: Optional[CEServerClient] = None,
        timeout: float = 5.0,
    ) -> None:
        self.host = host
        self.port = port
        self.pid = pid
        self.timeout = timeout
        self._owned_client = client is None
        self.client = client or CEServerClient()
        self.handle: Optional[int] = None
        self._connected = False

    def connect(self) -> None:
        """Ensure connection and process handle are established."""
        if not self._connected:
            if not self.client.connected:
                self.client.connect(self.host, self.port, self.timeout)
            if self.handle is None:
                self.handle = self.client.open_process(self.pid)
            self._connected = True

    def enumerate_regions(self) -> List[MemoryRegion]:
        """Enumerate virtual memory regions of target remote process."""
        self.connect()
        assert self.handle is not None
        return self.client.enumerate_regions(self.handle)

    def read_memory(self, address: int, size: int) -> bytes:
        """Read bytes from remote address space."""
        self.connect()
        assert self.handle is not None
        data = self.client.read_memory(self.handle, address, size)
        if len(data) < size:
            raise MemoryReadError(
                f"CEServer read incomplete: requested {size} bytes at 0x{address:x}, got {len(data)}"
            )
        return data

    def write_memory(self, address: int, data: bytes) -> int:
        """Write bytes to remote address space."""
        self.connect()
        assert self.handle is not None
        return self.client.write_memory(self.handle, address, data)

    def close(self) -> None:
        """Cleanly close remote handle and socket."""
        if self._connected and self.handle is not None:
            try:
                self.client.close_handle(self.handle)
            except Exception:
                pass
            self.handle = None
        if self._owned_client:
            self.client.close()
        self._connected = False

    def __enter__(self) -> CEServerMemoryBackend:
        self.connect()
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Mock CEServer Daemon for Testing & CI
# ---------------------------------------------------------------------------


@dataclass
class MockProcess:
    """Mock process memory model."""

    pid: int
    name: str = "mock_game.exe"
    regions: List[MemoryRegion] = field(default_factory=list)
    memory: bytearray = field(default_factory=bytearray)
    base_address: int = 0x140000000

    def get_chunk(self, address: int, size: int) -> bytes:
        rel = address - self.base_address
        if rel < 0 or rel >= len(self.memory):
            return b""
        end = min(rel + size, len(self.memory))
        return bytes(self.memory[rel:end])

    def put_chunk(self, address: int, data: bytes) -> int:
        rel = address - self.base_address
        if rel < 0 or rel >= len(self.memory):
            return 0
        end = min(rel + len(data), len(self.memory))
        written = end - rel
        self.memory[rel:end] = data[:written]
        return written


class MockCEServer:
    """Multi-threaded mock server simulating CEServer binary protocol on localhost."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0) -> None:
        self.host = host
        self.port = port
        self.server_sock: Optional[socket.socket] = None
        self.thread: Optional[threading.Thread] = None
        self.running = False
        self.processes: Dict[int, MockProcess] = {}
        self._handle_map: Dict[int, int] = {}  # handle -> pid
        self._next_handle = 100

    def add_process(
        self,
        pid: int,
        memory: bytes,
        base_address: int = 0x140000000,
        protection: str = "rw-",
        regions: Optional[List[MemoryRegion]] = None,
    ) -> None:
        """Register a mock process with simulated memory."""
        proc_regions = regions or [
            MemoryRegion(
                base=base_address,
                size=len(memory),
                protection=protection,
                name=f"proc_{pid}",
            )
        ]
        proc_regions = sorted(proc_regions, key=lambda r: r.base)
        proc = MockProcess(
            pid=pid,
            regions=proc_regions,
            memory=bytearray(memory),
            base_address=base_address,
        )
        self.processes[pid] = proc

    def start(self) -> None:
        """Start mock daemon on ephemeral or specified port."""
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((self.host, self.port))
        s.listen(5)
        self.server_sock = s
        self.port = s.getsockname()[1]
        self.running = True

        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        """Stop mock daemon and close server socket."""
        self.running = False
        if self.server_sock is not None:
            try:
                self.server_sock.close()
            except Exception:
                pass
            self.server_sock = None
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            self.thread = None

    def __enter__(self) -> MockCEServer:
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop()

    def _run_loop(self) -> None:
        assert self.server_sock is not None
        while self.running:
            try:
                r, _, _ = select.select([self.server_sock], [], [], 0.2)
                if not r:
                    continue
                client, _ = self.server_sock.accept()
                t = threading.Thread(
                    target=self._handle_client, args=(client,), daemon=True
                )
                t.start()
            except Exception:
                break

    @staticmethod
    def _recv_all(sock: socket.socket, count: int) -> bytes:
        buf = bytearray()
        while len(buf) < count:
            chunk = sock.recv(count - len(buf))
            if not chunk:
                break
            buf.extend(chunk)
        return bytes(buf)

    def _handle_client(self, client: socket.socket) -> None:
        client.settimeout(2.0)
        try:
            while self.running:
                cmd_raw = client.recv(1)
                if not cmd_raw:
                    break
                cmd = cmd_raw[0]

                if cmd == CMD_GETVERSION:
                    client.sendall(struct.pack("<I", 1))

                elif cmd == CMD_CLOSECONNECTION:
                    break

                elif cmd == CMD_OPENPROCESS:
                    raw_pid = self._recv_all(client, 4)
                    if len(raw_pid) < 4:
                        break
                    (pid,) = struct.unpack("<i", raw_pid)
                    if pid in self.processes:
                        handle = self._next_handle
                        self._next_handle += 1
                        self._handle_map[handle] = pid
                        client.sendall(struct.pack("<i", handle))
                    else:
                        client.sendall(struct.pack("<i", 0))

                elif cmd == CMD_CLOSEHANDLE:
                    raw_h = self._recv_all(client, 4)
                    if len(raw_h) < 4:
                        break
                    (handle,) = struct.unpack("<i", raw_h)
                    self._handle_map.pop(handle, None)
                    client.sendall(struct.pack("<i", 1))

                elif cmd == CMD_READPROCESSMEMORY:
                    hdr = self._recv_all(client, 16)  # handle (4) + address (8) + size (4)
                    if len(hdr) < 16:
                        break
                    handle, address, size = struct.unpack("<iQI", hdr)
                    pid = self._handle_map.get(handle)
                    if pid is not None and pid in self.processes:
                        proc = self.processes[pid]
                        chunk = proc.get_chunk(address, size)
                        client.sendall(struct.pack("<I", len(chunk)) + chunk)
                    else:
                        client.sendall(struct.pack("<I", 0))

                elif cmd == CMD_WRITEPROCESSMEMORY:
                    hdr = self._recv_all(client, 16)  # handle (4) + address (8) + size (4)
                    if len(hdr) < 16:
                        break
                    handle, address, size = struct.unpack("<iQI", hdr)
                    data = self._recv_all(client, size)

                    pid = self._handle_map.get(handle)
                    if pid is not None and pid in self.processes:
                        proc = self.processes[pid]
                        written = proc.put_chunk(address, data)
                        client.sendall(struct.pack("<I", written))
                    else:
                        client.sendall(struct.pack("<I", 0))

                elif cmd == CMD_VIRTUALQUERYEX:
                    hdr = self._recv_all(client, 12)  # handle (4) + address (8)
                    if len(hdr) < 12:
                        break
                    handle, address = struct.unpack("<iQ", hdr)
                    pid = self._handle_map.get(handle)
                    matched_region: Optional[MemoryRegion] = None
                    if pid is not None and pid in self.processes:
                        proc = self.processes[pid]
                        for reg in proc.regions:
                            if reg.base <= address < reg.base + reg.size:
                                matched_region = reg
                                break
                            if reg.base > address:
                                matched_region = reg
                                break

                    if matched_region is not None:
                        prot_int = PAGE_READWRITE
                        if matched_region.protection == "rwx":
                            prot_int = PAGE_EXECUTE_READWRITE
                        elif matched_region.protection == "r-x":
                            prot_int = PAGE_EXECUTE_READ
                        elif matched_region.protection == "r--":
                            prot_int = PAGE_READONLY
                        elif matched_region.protection == "---":
                            prot_int = PAGE_NOACCESS
                        elif matched_region.protection == "--x":
                            prot_int = PAGE_EXECUTE
                        payload = struct.pack(
                            "<BQQII",
                            1,
                            matched_region.base,
                            matched_region.size,
                            prot_int,
                            0x1000,
                        )
                        client.sendall(payload)
                    else:
                        client.sendall(struct.pack("<B", 0))

                elif cmd == CMD_GETARCHITECTURE:
                    self._recv_all(client, 4)  # handle
                    client.sendall(struct.pack("<B", 1))  # x86_64
        except Exception:
            pass
        finally:
            try:
                client.close()
            except Exception:
                pass
