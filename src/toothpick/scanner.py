"""
Toothpick bounded memory scanner (unit W07).

Implements the W07 bounded memory scanning contract per UNIT-W07-SCAN-PLAN.md,
design-document.md §11, and Risk R1 (Context explosion on dense graphs/buffers):
- Fast pattern, primitive value (integers, floats), string, and pointer reference scanning.
- Memory backends for offline memory dumps / byte buffers and live Frida processes.
- Iterative filtering (first scan -> refine scan on exact/increased/decreased/changed/unchanged values).
- Mandatory top-K context budget enforcement (default 50, maximum ceiling 1,000)
  and deterministic address sorting per Trap 9 and Trap 10.
- Persistence of verifiable memory scan receipts directly into toothpick.db.
"""

from __future__ import annotations

import json
import re
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4

from toothpick.ground import GroundReceipt
from toothpick.storage import Database

__all__ = [
    "ScannerError",
    "InvalidPatternError",
    "SessionNotFoundError",
    "MemoryReadError",
    "MemoryRegion",
    "ScanMatch",
    "ScanResult",
    "MemoryBackend",
    "BufferMemoryBackend",
    "FridaMemoryBackend",
    "MemoryScanner",
]

DEFAULT_SCAN_LIMIT = 50
MAX_SCAN_LIMIT = 1000


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ScannerError(Exception):
    """Base exception for memory scanner failures."""


class InvalidPatternError(ScannerError):
    """Raised when a pattern or search value cannot be parsed."""


class SessionNotFoundError(ScannerError):
    """Raised when an iterative filter scan references a missing session ID."""


class MemoryReadError(ScannerError):
    """Raised when a memory backend cannot read target address space."""


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MemoryRegion:
    """A contiguous virtual memory region."""

    base: int
    size: int
    protection: str = "rw-"
    name: str = ""

    @property
    def end(self) -> int:
        return self.base + self.size


@dataclass
class ScanMatch:
    """A single memory match found during a scan."""

    address: int
    value: Any
    size: int
    region_base: int = 0
    protection: str = "rw-"
    address_hex: str = field(init=False)

    def __post_init__(self) -> None:
        self.address_hex = f"0x{self.address:x}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "address": self.address,
            "address_hex": self.address_hex,
            "value": self.value,
            "size": self.size,
            "region_base": self.region_base,
            "protection": self.protection,
        }


@dataclass
class ScanResult:
    """Summary of a bounded memory scan operation."""

    session_id: str
    pattern_or_value: Any
    data_type: str
    total_matches: int
    returned_matches: int
    is_truncated: bool
    limit: int
    matches: List[ScanMatch] = field(default_factory=list)
    elapsed_sec: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "pattern_or_value": self.pattern_or_value,
            "data_type": self.data_type,
            "total_matches": self.total_matches,
            "returned_matches": self.returned_matches,
            "is_truncated": self.is_truncated,
            "limit": self.limit,
            "elapsed_sec": round(self.elapsed_sec, 4),
            "matches": [m.to_dict() for m in self.matches],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str)


# ---------------------------------------------------------------------------
# Type Parsers & Encoders
# ---------------------------------------------------------------------------


_NUMERIC_FORMATS: Dict[str, Tuple[str, int]] = {
    "i8": ("<b", 1),
    "u8": ("<B", 1),
    "i16": ("<h", 2),
    "u16": ("<H", 2),
    "i32": ("<i", 4),
    "u32": ("<I", 4),
    "i64": ("<q", 8),
    "u64": ("<Q", 8),
    "f32": ("<f", 4),
    "f64": ("<d", 8),
}


def _parse_numeric_value(val: Any, data_type: str) -> Tuple[bytes, Any]:
    """Parse and pack a numeric value into bytes."""
    fmt, size = _NUMERIC_FORMATS[data_type]
    try:
        if data_type in ("f32", "f64"):
            num = float(val)
        elif isinstance(val, str):
            num = int(val, 0)
        else:
            num = int(val)
        packed = struct.pack(fmt, num)
        return packed, num
    except (ValueError, struct.error) as exc:
        raise InvalidPatternError(f"Cannot parse {val!r} as {data_type}: {exc}") from exc


def _decode_numeric_value(raw: bytes, data_type: str) -> Any:
    """Unpack raw bytes into a python numeric value."""
    fmt, _ = _NUMERIC_FORMATS[data_type]
    try:
        return struct.unpack(fmt, raw)[0]
    except struct.error:
        return None


def _compile_hex_pattern(pattern_str: str) -> Tuple[re.Pattern[bytes], int]:
    """Compile a hex wildcard pattern like '48 89 5c 24 ? 57' into a regex."""
    tokens = pattern_str.strip().split()
    if not tokens:
        raise InvalidPatternError("Empty hex pattern")

    parts: List[bytes] = []
    pattern_len = 0
    for tok in tokens:
        if tok in ("?", "??"):
            parts.append(b".")
            pattern_len += 1
        else:
            try:
                b = bytes.fromhex(tok)
                parts.append(re.escape(b))
                pattern_len += len(b)
            except ValueError as exc:
                raise InvalidPatternError(f"Invalid hex token in pattern: {tok}") from exc

    regex_bytes = b"".join(parts)
    return re.compile(regex_bytes, re.DOTALL), pattern_len


# ---------------------------------------------------------------------------
# Memory Backends
# ---------------------------------------------------------------------------


class MemoryBackend:
    """Base interface for readable memory spaces."""

    def enumerate_regions(self) -> List[MemoryRegion]:
        raise NotImplementedError

    def read_memory(self, address: int, size: int) -> bytes:
        raise NotImplementedError


class BufferMemoryBackend(MemoryBackend):
    """Memory backend over an in-memory byte buffer or raw file dump."""

    def __init__(
        self,
        data: bytes,
        base_address: int = 0x0,
        protection: str = "rw-",
        name: str = "buffer",
    ) -> None:
        self.data = data
        self.base_address = base_address
        self.protection = protection
        self.name = name

    def enumerate_regions(self) -> List[MemoryRegion]:
        if not self.data:
            return []
        return [
            MemoryRegion(
                base=self.base_address,
                size=len(self.data),
                protection=self.protection,
                name=self.name,
            )
        ]

    def read_memory(self, address: int, size: int) -> bytes:
        rel_offset = address - self.base_address
        if rel_offset < 0 or rel_offset + size > len(self.data):
            raise MemoryReadError(
                f"Address 0x{address:x} (len {size}) out of buffer range "
                f"[0x{self.base_address:x}, 0x{self.base_address + len(self.data):x})"
            )
        return self.data[rel_offset : rel_offset + size]

    @classmethod
    def from_file(
        cls,
        path: Union[str, Path],
        base_address: int = 0x0,
        protection: str = "rw-",
    ) -> BufferMemoryBackend:
        p = Path(path)
        if not p.is_file():
            raise MemoryReadError(f"Dump file not found: {p}")
        return cls(
            data=p.read_bytes(),
            base_address=base_address,
            protection=protection,
            name=p.name,
        )


class FridaMemoryBackend(MemoryBackend):
    """Memory backend connecting to a live process via Frida."""

    def __init__(self, pid: int, frida_module: Optional[Any] = None) -> None:
        self.pid = pid
        self._frida = frida_module
        self._session: Optional[Any] = None

    def _get_session(self) -> Any:
        if self._session is not None:
            return self._session
        if self._frida is None:
            try:
                import frida

                self._frida = frida
            except ImportError as exc:
                raise ScannerError(
                    "frida module is not installed; use --file for offline dumps"
                ) from exc
        try:
            self._session = self._frida.attach(self.pid)
            return self._session
        except Exception as exc:
            raise ScannerError(f"Cannot attach to PID {self.pid}: {exc}") from exc

    def close(self) -> None:
        if self._session is not None:
            try:
                self._session.detach()
            except Exception:
                pass
            self._session = None

    def enumerate_regions(self) -> List[MemoryRegion]:
        """Query target process memory ranges via Frida script RPC."""
        session = self._get_session()
        js_code = """
        rpc.exports = {
            get_ranges: function() {
                var readable = [];
                var all = Process.enumerateRanges('---');
                for (var i = 0; i < all.length; i++) {
                    var r = all[i];
                    var prot = r.protection || '';
                    if (prot.indexOf('r') === -1) continue;
                    readable.push({
                        base: r.base.toString(),
                        size: r.size,
                        protection: prot,
                        file: r.file ? r.file.path : ""
                    });
                }
                return readable;
            },
            read: function(addrStr, length) {
                return Memory.readByteArray(ptr(addrStr), length);
            }
        };
        """
        try:
            script = session.create_script(js_code)
            script.load()
            raw_ranges = script.exports_sync.get_ranges()
            regions: List[MemoryRegion] = []
            for r in raw_ranges:
                protection = str(r.get("protection", "r--") or "r--")
                if "r" not in protection.lower():
                    continue
                raw_base = r["base"]
                base_int = int(raw_base, 16) if isinstance(raw_base, str) else int(raw_base)
                regions.append(
                    MemoryRegion(
                        base=base_int,
                        size=int(r["size"]),
                        protection=r.get("protection", "r--"),
                        name=r.get("file", ""),
                    )
                )
            return regions
        except Exception as exc:
            raise MemoryReadError(f"Failed to enumerate Frida process ranges: {exc}") from exc

    def read_memory(self, address: int, size: int) -> bytes:
        session = self._get_session()
        js_code = f"""
        rpc.exports = {{
            read: function() {{
                const buf = Memory.readByteArray(ptr("0x{address:x}"), {size});
                return buf;
            }}
        }};
        """
        try:
            script = session.create_script(js_code)
            script.load()
            buf = script.exports_sync.read()
            return bytes(buf) if buf else b""
        except Exception as exc:
            raise MemoryReadError(f"Frida read failed at 0x{address:x}: {exc}") from exc


# ---------------------------------------------------------------------------
# Scan Session (Iterative State)
# ---------------------------------------------------------------------------


@dataclass
class _ScanSessionState:
    session_id: str
    data_type: str
    matches: List[int]  # All matched addresses
    last_values: Dict[int, Any]  # address -> last recorded value
    backend: MemoryBackend


# ---------------------------------------------------------------------------
# Bounded Memory Scanner Engine
# ---------------------------------------------------------------------------


class MemoryScanner:
    """Performs bounded memory scans and iterative variable tracking."""

    def __init__(self, db: Optional[Union[Database, str, Path]] = None) -> None:
        self.db: Optional[Database] = None
        self._owned_db = False
        if db is not None:
            if isinstance(db, Database):
                self.db = db
            else:
                self.db = Database(Path(db), create=False)
                self._owned_db = True
        self._sessions: Dict[str, _ScanSessionState] = {}

    def close(self) -> None:
        if self._owned_db and self.db is not None:
            self.db.close()
            self.db = None

    def __enter__(self) -> MemoryScanner:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def first_scan(
        self,
        backend: MemoryBackend,
        value: Any,
        data_type: str = "i32",
        limit: int = DEFAULT_SCAN_LIMIT,
        binary_sha256: Optional[str] = None,
    ) -> ScanResult:
        """Scan memory regions for an initial value, string, or pattern."""
        t0 = time.perf_counter()
        bounded_limit = max(1, min(limit, MAX_SCAN_LIMIT))
        session_id = f"scan_{uuid4().hex[:12]}"

        regions = backend.enumerate_regions()
        all_matches: List[ScanMatch] = []
        last_values: Dict[int, Any] = {}

        if data_type in _NUMERIC_FORMATS:
            packed_target, target_num = _parse_numeric_value(value, data_type)
            val_size = len(packed_target)
            step = 1 if val_size == 1 else (2 if val_size == 2 else 4)

            for region in regions:
                try:
                    chunk = backend.read_memory(region.base, region.size)
                except Exception:
                    continue

                idx = 0
                while True:
                    idx = chunk.find(packed_target, idx)
                    if idx == -1:
                        break
                    addr = region.base + idx
                    all_matches.append(
                        ScanMatch(
                            address=addr,
                            value=target_num,
                            size=val_size,
                            region_base=region.base,
                            protection=region.protection,
                        )
                    )
                    last_values[addr] = target_num
                    idx += step

        elif data_type == "string":
            str_bytes = value.encode("utf-8") if isinstance(value, str) else bytes(value)
            val_size = len(str_bytes)
            if val_size == 0:
                raise InvalidPatternError("String search value cannot be empty")

            for region in regions:
                try:
                    chunk = backend.read_memory(region.base, region.size)
                except Exception:
                    continue

                idx = 0
                while True:
                    idx = chunk.find(str_bytes, idx)
                    if idx == -1:
                        break
                    addr = region.base + idx
                    all_matches.append(
                        ScanMatch(
                            address=addr,
                            value=value,
                            size=val_size,
                            region_base=region.base,
                            protection=region.protection,
                        )
                    )
                    last_values[addr] = value
                    idx += 1

        elif data_type == "hex":
            pattern_regex, pat_len = _compile_hex_pattern(str(value))
            for region in regions:
                try:
                    chunk = backend.read_memory(region.base, region.size)
                except Exception:
                    continue

                for m in pattern_regex.finditer(chunk):
                    addr = region.base + m.start()
                    hex_val = m.group(0).hex()
                    all_matches.append(
                        ScanMatch(
                            address=addr,
                            value=hex_val,
                            size=pat_len,
                            region_base=region.base,
                            protection=region.protection,
                        )
                    )
                    last_values[addr] = hex_val

        else:
            raise InvalidPatternError(
                f"Unsupported data_type {data_type!r}. Supported: "
                f"{', '.join(list(_NUMERIC_FORMATS.keys()) + ['string', 'hex'])}"
            )

        # Trap 10: Sort strictly deterministically by address
        all_matches.sort(key=lambda m: m.address)

        # Store session state for iterative filtering
        self._sessions[session_id] = _ScanSessionState(
            session_id=session_id,
            data_type=data_type,
            matches=[m.address for m in all_matches],
            last_values=last_values,
            backend=backend,
        )

        total_count = len(all_matches)
        returned = all_matches[:bounded_limit]
        is_truncated = total_count > bounded_limit
        elapsed = time.perf_counter() - t0

        result = ScanResult(
            session_id=session_id,
            pattern_or_value=value,
            data_type=data_type,
            total_matches=total_count,
            returned_matches=len(returned),
            is_truncated=is_truncated,
            limit=bounded_limit,
            matches=returned,
            elapsed_sec=elapsed,
        )

        self._maybe_persist_receipt(result, binary_sha256)
        return result

    def filter_scan(
        self,
        session_id: str,
        new_value: Optional[Any] = None,
        comparison: str = "exact",
        limit: int = DEFAULT_SCAN_LIMIT,
        binary_sha256: Optional[str] = None,
        backend: Optional[MemoryBackend] = None,
    ) -> ScanResult:
        """Refine a previous scan session against updated memory values."""
        t0 = time.perf_counter()
        session = self._sessions.get(session_id)
        if session is None and self.db is not None:
            row = self.db._conn.execute(
                "SELECT payload_json FROM receipts WHERE receipt_id = ?",
                (session_id,),
            ).fetchone()
            if row is not None and row["payload_json"]:
                payload = json.loads(row["payload_json"])
                live_reads = payload.get("live_reads", {})
                d_type = live_reads.get("data_type", "i32")
                matches_data = live_reads.get("matches", [])
                addrs = [m["address"] for m in matches_data]
                last_vals = {m["address"]: m["value"] for m in matches_data}
                if backend is not None:
                    session = _ScanSessionState(
                        session_id=session_id,
                        data_type=d_type,
                        matches=addrs,
                        last_values=last_vals,
                        backend=backend,
                    )
                    self._sessions[session_id] = session

        if session is None:
            raise SessionNotFoundError(f"Active scan session not found: {session_id}")

        if backend is not None:
            session.backend = backend

        bounded_limit = max(1, min(limit, MAX_SCAN_LIMIT))
        data_type = session.data_type
        backend = session.backend
        prev_addresses = session.matches
        prev_values = session.last_values

        if data_type not in _NUMERIC_FORMATS:
            raise ScannerError(
                f"Iterative filtering only supported for numeric types, not {data_type}"
            )

        _, val_size = _NUMERIC_FORMATS[data_type]
        target_num: Optional[Any] = None
        if new_value is not None:
            _, target_num = _parse_numeric_value(new_value, data_type)

        surviving_matches: List[ScanMatch] = []
        new_last_values: Dict[int, Any] = {}

        for addr in prev_addresses:
            try:
                raw_bytes = backend.read_memory(addr, val_size)
            except Exception:
                continue

            current_val = _decode_numeric_value(raw_bytes, data_type)
            if current_val is None:
                continue

            old_val = prev_values.get(addr)
            keep = False

            if comparison == "exact":
                if target_num is not None and current_val == target_num:
                    keep = True
            elif comparison == "increased":
                if old_val is not None and current_val > old_val:
                    keep = True
            elif comparison == "decreased":
                if old_val is not None and current_val < old_val:
                    keep = True
            elif comparison == "changed":
                if old_val is not None and current_val != old_val:
                    keep = True
            elif comparison == "unchanged":
                if old_val is not None and current_val == old_val:
                    keep = True
            else:
                raise ScannerError(
                    f"Unknown comparison operator: {comparison!r}. "
                    "Valid: exact, increased, decreased, changed, unchanged"
                )

            if keep:
                surviving_matches.append(
                    ScanMatch(
                        address=addr,
                        value=current_val,
                        size=val_size,
                    )
                )
                new_last_values[addr] = current_val

        surviving_matches.sort(key=lambda m: m.address)

        # Update session state with filtered subset
        session.matches = [m.address for m in surviving_matches]
        session.last_values = new_last_values

        total_count = len(surviving_matches)
        returned = surviving_matches[:bounded_limit]
        is_truncated = total_count > bounded_limit
        elapsed = time.perf_counter() - t0

        result = ScanResult(
            session_id=session_id,
            pattern_or_value=new_value if new_value is not None else comparison,
            data_type=data_type,
            total_matches=total_count,
            returned_matches=len(returned),
            is_truncated=is_truncated,
            limit=bounded_limit,
            matches=returned,
            elapsed_sec=elapsed,
        )

        self._maybe_persist_receipt(result, binary_sha256)
        return result

    def scan_pointers(
        self,
        backend: MemoryBackend,
        target_address: int,
        offset_tolerance: int = 0x1000,
        pointer_size: int = 8,
        limit: int = DEFAULT_SCAN_LIMIT,
        binary_sha256: Optional[str] = None,
    ) -> ScanResult:
        """Scan readable memory for pointers pointing to target_address (+/- offset_tolerance)."""
        t0 = time.perf_counter()
        bounded_limit = max(1, min(limit, MAX_SCAN_LIMIT))
        session_id = f"ptr_{uuid4().hex[:12]}"
        ptr_fmt = "<Q" if pointer_size == 8 else "<I"

        regions = backend.enumerate_regions()
        matches: List[ScanMatch] = []
        target_min = target_address
        target_max = target_address + offset_tolerance

        for region in regions:
            try:
                chunk = backend.read_memory(region.base, region.size)
            except Exception:
                continue

            for offset in range(0, len(chunk) - pointer_size + 1, pointer_size):
                ptr_val = struct.unpack_from(ptr_fmt, chunk, offset)[0]
                if target_min <= ptr_val <= target_max:
                    matches.append(
                        ScanMatch(
                            address=region.base + offset,
                            value=f"0x{ptr_val:x}",
                            size=pointer_size,
                            region_base=region.base,
                            protection=region.protection,
                        )
                    )

        matches.sort(key=lambda m: m.address)
        total_count = len(matches)
        returned = matches[:bounded_limit]
        is_truncated = total_count > bounded_limit
        elapsed = time.perf_counter() - t0

        result = ScanResult(
            session_id=session_id,
            pattern_or_value=f"0x{target_address:x}",
            data_type="pointer",
            total_matches=total_count,
            returned_matches=len(returned),
            is_truncated=is_truncated,
            limit=bounded_limit,
            matches=returned,
            elapsed_sec=elapsed,
        )

        self._maybe_persist_receipt(result, binary_sha256)
        return result

    def _maybe_persist_receipt(
        self, result: ScanResult, explicit_sha: Optional[str] = None
    ) -> None:
        """Persist a scan receipt into the SQLite database if active."""
        if self.db is None:
            return

        sha = explicit_sha or ""
        if not sha:
            rows = self.db._conn.execute("SELECT DISTINCT binary_sha256 FROM funcs").fetchall()
            shas = [r["binary_sha256"] for r in rows if r["binary_sha256"]]
            if len(shas) == 1:
                sha = shas[0]
            elif len(shas) > 1:
                raise ScannerError(
                    "Cannot infer binary SHA-256: database contains "
                    f"{len(shas)} distinct binaries ({', '.join(sorted(shas))}); "
                    "pass an explicit --sha"
                )

        func_addr = result.matches[0].address if result.matches else 0
        receipt = GroundReceipt(
            receipt_id=result.session_id,
            binary_sha256=sha,
            func_addr=func_addr,
            ground_type="MEMORY_SCAN",
            timestamp=time.time(),
            live_reads=result.to_dict(),
            verifier_status="GROUNDED" if result.total_matches > 0 else "UNGROUNDED",
            phase="scan",
        )
        self.db.insert_receipt(receipt)
