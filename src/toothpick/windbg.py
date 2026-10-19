"""
Toothpick W15: WinDbg-cdb headless debugger bridge and minidump crash analyzer.

Implements ``docs/units/UNIT-W15-WINDBG-PLAN.md`` (milestone 5, "Full Deck"
dynamic backends) against ``design-document.md`` §6/§9/§11/§12/§16 #1 and
Risk R2 (target corruption) / R5 (no live host available):

* :class:`CdbEngine` drives ``cdb.exe`` (the console WinDbg) headlessly in a
  **non-invasive** mode: ``cdb -pv -p <pid> -c "<commands>"``.  ``-pv`` keeps
  the debugger from ever touching the target's memory image with a
  breakpoint, so a running game is inspected (registers, call stack, module
  map) without tripping anti-debug heuristics or dying on detach.
* Every subprocess run is guarded by an explicit timeout; on expiry the whole
  process tree is terminated so a hung ``cdb`` cannot leak a debuggee handle.
* :class:`MinidumpReader` is a dependency-free parser for the Microsoft
  minidump container (``MDMP`` magic ``0x504d444d``) covering the
  ``ExceptionStream``, ``ThreadListStream``, ``ModuleListStream`` and
  ``SystemInfoStream``.  Crash analysis therefore works completely offline on
  any host, with no Windows SDK or Debugging Tools installation (Risk R5).
* Findings are emitted as dynamic grounding receipts
  (``phase="cdb_trace"``, ``verifier_status="GROUNDED"``) so crash evidence
  enters ``toothpick.db`` on the same footing as Frida/CEServer grounding.

The CDB text parsers (``r`` / ``kp`` / ``lm``) mirror the real console layouts
of the Debugging Tools; unrelated commands in the same session are ignored
rather than misparsed.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import struct
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from pathlib import PureWindowsPath
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

__all__ = [
    "MDMP_MAGIC",
    "MDMP_SIGNATURE",
    "STREAM_THREAD_LIST",
    "STREAM_MODULE_LIST",
    "STREAM_MEMORY_LIST",
    "STREAM_EXCEPTION",
    "STREAM_SYSTEM_INFO",
    "STREAM_MEMORY64_LIST",
    "STREAM_NAMES",
    "EXCEPTION_NAMES",
    "CDB_PHASE",
    "DEFAULT_ATTACH_COMMANDS",
    "DEFAULT_CDB_TIMEOUT",
    "CdbError",
    "CdbNotFoundError",
    "CdbTimeoutError",
    "CdbParseError",
    "CdbRegisters",
    "CdbStackFrame",
    "CdbModule",
    "CdbSessionResult",
    "CrashReport",
    "MinidumpHeader",
    "MinidumpDirectory",
    "MinidumpException",
    "MinidumpThread",
    "MinidumpReader",
    "CdbEngine",
    "parse_cdb_registers",
    "parse_cdb_call_stack",
    "parse_cdb_modules",
    "parse_x64_context",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Microsoft minidump container signature (little endian ``MDMP``).
MDMP_MAGIC = 0x504D444D
MDMP_SIGNATURE = b"MDMP"

STREAM_THREAD_LIST = 3
STREAM_MODULE_LIST = 4
STREAM_MEMORY_LIST = 5
STREAM_EXCEPTION = 6
STREAM_SYSTEM_INFO = 7
STREAM_MEMORY64_LIST = 9

STREAM_NAMES: Dict[int, str] = {
    STREAM_THREAD_LIST: "ThreadListStream",
    STREAM_MODULE_LIST: "ModuleListStream",
    STREAM_MEMORY_LIST: "MemoryListStream",
    STREAM_EXCEPTION: "ExceptionStream",
    STREAM_SYSTEM_INFO: "SystemInfoStream",
    STREAM_MEMORY64_LIST: "Memory64ListStream",
}

#: Windows NTSTATUS crash codes worth naming in a receipt.
EXCEPTION_NAMES: Dict[int, str] = {
    0x80000003: "STATUS_BREAKPOINT",
    0x80000004: "STATUS_SINGLE_STEP",
    0xC0000005: "STATUS_ACCESS_VIOLATION",
    0xC0000006: "STATUS_IN_PAGE_ERROR",
    0xC0000017: "STATUS_NO_MEMORY",
    0xC000001D: "STATUS_ILLEGAL_INSTRUCTION",
    0xC0000025: "STATUS_NONCONTINUABLE_EXCEPTION",
    0xC000008C: "STATUS_ARRAY_BOUNDS_EXCEEDED",
    0xC000008D: "STATUS_FLOAT_DENORMAL_OPERAND",
    0xC0000094: "STATUS_INTEGER_DIVIDE_BY_ZERO",
    0xC0000095: "STATUS_INTEGER_OVERFLOW",
    0xC0000096: "STATUS_PRIVILEGED_INSTRUCTION",
    0xC00000FD: "STATUS_STACK_OVERFLOW",
    0xC0000374: "STATUS_HEAP_CORRUPTION",
    0xC0000409: "STATUS_STACK_BUFFER_OVERRUN",
    0xC0000420: "STATUS_ASSERTION_FAILURE",
    0xE06D7363: "CPP_EXCEPTION (0xE06D7363)",
}

#: Receipt phase stamped by every W15 crash/live attachment receipt.
CDB_PHASE = "cdb_trace"

#: Non-invasive default session: registers, call stack, module map, quit.
DEFAULT_ATTACH_COMMANDS: Tuple[str, ...] = ("r", "kp", "lm", "q")

DEFAULT_CDB_TIMEOUT = 30.0

#: Well-known Debugging Tools locations (checked after PATH / env overrides).
_CDB_CANDIDATE_PATHS: Tuple[str, ...] = (
    r"C:\Program Files (x86)\Windows Kits\10\Debuggers\x64\cdb.exe",
    r"C:\Program Files (x86)\Windows Kits\10\Debuggers\x86\cdb.exe",
    r"C:\Program Files\Windows Kits\10\Debuggers\x64\cdb.exe",
    r"C:\Program Files\Windows Kits\10\Debuggers\x86\cdb.exe",
    r"C:\Program Files (x86)\Windows Kits\8.1\Debuggers\x64\cdb.exe",
    r"C:\Program Files (x86)\Windows Kits\8.1\Debuggers\x86\cdb.exe",
    r"C:\Debuggers\cdb.exe",
)

_CDB_KITS_ROOT = r"C:\Program Files (x86)\Windows Kits\10\Debuggers"

#: Environment variables honoured as an explicit cdb override.
_CDB_ENV_VARS: Tuple[str, ...] = ("TOOTHPICK_CDB", "CDB_PATH", "WINDBG_CDB")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class CdbError(Exception):
    """Base error for CDB process failures, timeouts and parse errors."""


class CdbNotFoundError(CdbError):
    """``cdb.exe`` could not be located on this host."""


class CdbTimeoutError(CdbError):
    """A CDB session exceeded its timeout budget and was terminated."""


class CdbParseError(CdbError):
    """CDB text or minidump binary data could not be parsed."""


# ---------------------------------------------------------------------------
# CDB data contracts
# ---------------------------------------------------------------------------

#: Canonical x64 register set, in declaration order.
REGISTER_FIELDS: Tuple[str, ...] = (
    "rax",
    "rbx",
    "rcx",
    "rdx",
    "rsi",
    "rdi",
    "rbp",
    "rsp",
    "r8",
    "r9",
    "r10",
    "r11",
    "r12",
    "r13",
    "r14",
    "r15",
    "rip",
    "eflags",
)

#: 32-bit / alias spellings CDB may print -> canonical x64 register.
_REGISTER_ALIASES: Dict[str, str] = {
    "eax": "rax",
    "ebx": "rbx",
    "ecx": "rcx",
    "edx": "rdx",
    "esi": "rsi",
    "edi": "rdi",
    "ebp": "rbp",
    "esp": "rsp",
    "eip": "rip",
    "efl": "eflags",
}
for _reg in REGISTER_FIELDS:
    _REGISTER_ALIASES[_reg] = _reg

_REGISTER_RE = re.compile(
    r"\b(?P<name>r(?:ax|bx|cx|dx|si|di|bp|sp|ip|[89]|1[0-5])"
    r"|e(?:ax|bx|cx|dx|si|di|bp|sp|ip|fl)|eflags)"
    r"\s*[=:]\s*(?P<value>0x[0-9A-Fa-f]+|[0-9A-Fa-f]+)",
    re.IGNORECASE,
)

#: x64 ``CONTEXT`` offsets (winnt.h) used by minidump thread contexts.
_X64_CONTEXT_RAX = 0x78
_X64_CONTEXT_ORDER: Tuple[str, ...] = (
    "rax",
    "rcx",
    "rdx",
    "rbx",
    "rsp",
    "rbp",
    "rsi",
    "rdi",
    "r8",
    "r9",
    "r10",
    "r11",
    "r12",
    "r13",
    "r14",
    "r15",
)
_X64_CONTEXT_RIP = 0xF8
_X64_CONTEXT_EFLAGS = 0x44
_X64_CONTEXT_MIN_SIZE = _X64_CONTEXT_RIP + 8


@dataclass
class CdbRegisters:
    """x64 register snapshot captured from ``r`` or a minidump CONTEXT."""

    rax: int = 0
    rbx: int = 0
    rcx: int = 0
    rdx: int = 0
    rsi: int = 0
    rdi: int = 0
    rbp: int = 0
    rsp: int = 0
    r8: int = 0
    r9: int = 0
    r10: int = 0
    r11: int = 0
    r12: int = 0
    r13: int = 0
    r14: int = 0
    r15: int = 0
    rip: int = 0
    eflags: int = 0

    def get(self, name: str) -> int:
        """Return one register by canonical or alias name (0 when unknown)."""
        canonical = _REGISTER_ALIASES.get(str(name).lower().strip())
        if canonical is None:
            return 0
        return int(getattr(self, canonical))

    def to_dict(self) -> Dict[str, Any]:
        """JSON-ready mapping of every register plus hex convenience keys."""
        payload: Dict[str, Any] = {}
        for name in REGISTER_FIELDS:
            value = int(getattr(self, name))
            payload[name] = value
            payload[f"{name}_hex"] = f"0x{value:x}"
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_cdb_text(cls, text: str) -> "CdbRegisters":
        """Parse a CDB ``r`` register dump.

        CDB prints bare hexadecimal without an ``0x`` prefix
        (``rax=0000000000000042``), so unprefixed values are read as hex.
        Unknown mnemonics (``cs``, ``iopl``) are ignored.
        """
        return parse_cdb_registers(text)

    @classmethod
    def from_x64_context(cls, blob: bytes) -> "CdbRegisters":
        """Decode an x64 ``CONTEXT`` record (minidump thread context)."""
        return parse_x64_context(blob)


@dataclass
class CdbStackFrame:
    """One unwound frame from a CDB ``k`` / ``kp`` / ``kv`` listing."""

    frame_num: int
    child_sp: int = 0
    ret_addr: int = 0
    call_site: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_num": self.frame_num,
            "child_sp": self.child_sp,
            "child_sp_hex": f"0x{self.child_sp:x}",
            "ret_addr": self.ret_addr,
            "ret_addr_hex": f"0x{self.ret_addr:x}",
            "call_site": self.call_site,
        }


@dataclass
class CdbModule:
    """A loaded module (image) in the target process."""

    name: str
    base_addr: int = 0
    end_addr: int = 0
    path: str = ""

    @property
    def size(self) -> int:
        return max(0, int(self.end_addr) - int(self.base_addr))

    def contains(self, addr: int) -> bool:
        return int(self.base_addr) != 0 and int(self.base_addr) <= int(addr) < int(self.end_addr)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "base_addr": self.base_addr,
            "base_addr_hex": f"0x{self.base_addr:x}",
            "end_addr": self.end_addr,
            "end_addr_hex": f"0x{self.end_addr:x}",
            "size": self.size,
            "path": self.path,
        }


@dataclass
class CdbSessionResult:
    """Outcome of one non-invasive CDB session (``-pv -p <pid>``)."""

    pid: int
    commands: List[str] = field(default_factory=list)
    command_line: str = ""
    exit_code: int = 0
    timed_out: bool = False
    duration_seconds: float = 0.0
    raw_output: str = ""
    stderr: str = ""
    cdb_executable: Optional[str] = None
    registers: Optional[CdbRegisters] = None
    stack_trace: List[CdbStackFrame] = field(default_factory=list)
    modules: List[CdbModule] = field(default_factory=list)

    def verify_address(self, addr: int) -> Tuple[bool, Optional[CdbModule]]:
        """Return whether ``addr`` resolves inside a loaded module of this session."""
        for module in self.modules:
            if module.contains(addr):
                return True, module
        return False, None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pid": self.pid,
            "mode": "non_invasive_attach",
            "commands": list(self.commands),
            "command_line": self.command_line,
            "exit_code": self.exit_code,
            "timed_out": self.timed_out,
            "duration_seconds": self.duration_seconds,
            "cdb_executable": self.cdb_executable,
            "registers": self.registers.to_dict() if self.registers else None,
            "stack_trace": [f.to_dict() for f in self.stack_trace],
            "stack_depth": len(self.stack_trace),
            "modules": [m.to_dict() for m in self.modules],
            "module_count": len(self.modules),
            "stderr": self.stderr,
        }


@dataclass
class CrashReport:
    """Structured crash finding from a minidump (or a live ``.ecxr`` capture)."""

    exception_code: int
    exception_name: str
    faulting_ip: int
    faulting_address: Optional[int] = None
    registers: CdbRegisters = field(default_factory=CdbRegisters)
    stack_trace: List[CdbStackFrame] = field(default_factory=list)
    modules: List[CdbModule] = field(default_factory=list)
    parameters: List[int] = field(default_factory=list)
    faulting_operation: Optional[str] = None
    thread_id: Optional[int] = None
    source: str = "minidump"
    dump_path: Optional[str] = None
    has_exception_stream: bool = True
    threads: List["MinidumpThread"] = field(default_factory=list)
    system_info: Dict[str, Any] = field(default_factory=dict)
    streams: Dict[str, Any] = field(default_factory=dict)

    def faulting_module(self) -> Optional[CdbModule]:
        """Module containing the faulting instruction pointer, if any."""
        for module in self.modules:
            if module.contains(self.faulting_ip):
                return module
        return None

    def to_dict(self) -> Dict[str, Any]:
        module = self.faulting_module()
        return {
            "exception_code": self.exception_code,
            "exception_code_hex": f"0x{self.exception_code:08x}",
            "exception_name": self.exception_name,
            "faulting_ip": self.faulting_ip,
            "faulting_ip_hex": f"0x{self.faulting_ip:x}",
            "faulting_address": self.faulting_address,
            "faulting_address_hex": (
                f"0x{self.faulting_address:x}" if self.faulting_address is not None else None
            ),
            "faulting_operation": self.faulting_operation,
            "faulting_module": module.name if module else None,
            "faulting_module_offset": (
                f"0x{self.faulting_ip - module.base_addr:x}" if module else None
            ),
            "thread_id": self.thread_id,
            "parameters": list(self.parameters),
            "registers": self.registers.to_dict(),
            "stack_trace": [f.to_dict() for f in self.stack_trace],
            "stack_depth": len(self.stack_trace),
            "modules": [m.to_dict() for m in self.modules],
            "module_count": len(self.modules),
            "threads": [t.to_dict() for t in self.threads],
            "system_info": dict(self.system_info),
            "source": self.source,
            "dump_path": self.dump_path,
            "has_exception_stream": self.has_exception_stream,
            "streams": dict(self.streams),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str)

    def receipt_payload(self, ground_type: str = "CDB_CRASH_DUMP") -> Dict[str, Any]:
        """Dynamic receipt payload (``phase="cdb_trace"``, GROUNDED)."""
        return {
            "ground_type": ground_type,
            "verifier_status": "GROUNDED",
            "phase": CDB_PHASE,
            "crash": self.to_dict(),
        }

    def to_receipt(
        self,
        binary_sha256: str = "",
        func_addr: Optional[int] = None,
        receipt_id: Optional[str] = None,
        ground_type: str = "CDB_CRASH_DUMP",
    ) -> Dict[str, Any]:
        """Build a ``toothpick.db`` receipts-table record for this crash."""
        return {
            "receipt_id": receipt_id,
            "binary_sha256": binary_sha256 or "",
            "func_addr": int(self.faulting_ip if func_addr is None else func_addr),
            "phase": CDB_PHASE,
            "payload": self.receipt_payload(ground_type=ground_type),
        }


# ---------------------------------------------------------------------------
# Minidump binary format
# ---------------------------------------------------------------------------


@dataclass
class MinidumpHeader:
    """``MINIDUMP_HEADER`` (32 bytes)."""

    signature: int
    version: int
    number_of_streams: int
    stream_directory_rva: int
    checksum: int
    time_date_stamp: int
    flags: int
    stream_directory_size: int = 0

    @property
    def is_valid(self) -> bool:
        return self.signature == MDMP_MAGIC

    @property
    def magic_hex(self) -> str:
        return f"0x{self.signature:08x}"

    @property
    def version_major(self) -> int:
        return (self.version >> 16) & 0xFFFF

    @property
    def version_minor(self) -> int:
        return self.version & 0xFFFF

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signature": self.signature,
            "signature_hex": self.magic_hex,
            "signature_ascii": self.signature.to_bytes(4, "little").decode("ascii", "replace"),
            "version": self.version,
            "version_major": self.version_major,
            "version_minor": self.version_minor,
            "number_of_streams": self.number_of_streams,
            "stream_directory_rva": self.stream_directory_rva,
            "stream_directory_size": self.stream_directory_size,
            "checksum": self.checksum,
            "time_date_stamp": self.time_date_stamp,
            "flags": self.flags,
        }


@dataclass
class MinidumpDirectory:
    """One ``MINIDUMP_DIRECTORY`` entry (stream type + location)."""

    stream_type: int
    data_size: int
    rva: int

    @property
    def stream_name(self) -> str:
        return STREAM_NAMES.get(self.stream_type, f"Stream{self.stream_type}")

    @property
    def end_rva(self) -> int:
        return self.rva + self.data_size

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stream_type": self.stream_type,
            "stream_name": self.stream_name,
            "data_size": self.data_size,
            "rva": self.rva,
        }


@dataclass
class MinidumpException:
    """``MINIDUMP_EXCEPTION_STREAM`` (thread id + exception record + context)."""

    thread_id: int
    exception_code: int
    exception_flags: int = 0
    exception_record: int = 0
    exception_address: int = 0
    number_parameters: int = 0
    parameters: List[int] = field(default_factory=list)
    thread_context_rva: int = 0
    thread_context_size: int = 0

    @property
    def exception_name(self) -> str:
        return exception_name(self.exception_code)

    @property
    def faulting_address(self) -> Optional[int]:
        """First exception parameter (the dereferenced address) when present."""
        if self.number_parameters >= 1 and self.parameters:
            return int(self.parameters[0])
        return None

    @property
    def faulting_operation(self) -> Optional[str]:
        """Access type for access violations: read / write / execute."""
        if self.number_parameters >= 2 and len(self.parameters) >= 2:
            return {0: "read", 1: "write", 8: "execute"}.get(int(self.parameters[1]))
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "thread_id": self.thread_id,
            "exception_code": self.exception_code,
            "exception_code_hex": f"0x{self.exception_code:08x}",
            "exception_name": self.exception_name,
            "exception_flags": self.exception_flags,
            "exception_record": self.exception_record,
            "exception_address": self.exception_address,
            "exception_address_hex": f"0x{self.exception_address:x}",
            "number_parameters": self.number_parameters,
            "parameters": list(self.parameters),
            "faulting_address": self.faulting_address,
            "faulting_operation": self.faulting_operation,
            "thread_context_rva": self.thread_context_rva,
            "thread_context_size": self.thread_context_size,
        }


@dataclass
class MinidumpThread:
    """``MINIDUMP_THREAD`` (identity plus stack and context locations)."""

    thread_id: int
    suspend_count: int = 0
    priority_class: int = 0
    priority: int = 0
    teb: int = 0
    stack_start: int = 0
    stack_size: int = 0
    stack_rva: int = 0
    context_rva: int = 0
    context_size: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "thread_id": self.thread_id,
            "suspend_count": self.suspend_count,
            "priority_class": self.priority_class,
            "priority": self.priority,
            "teb": self.teb,
            "teb_hex": f"0x{self.teb:x}",
            "stack_start": self.stack_start,
            "stack_start_hex": f"0x{self.stack_start:x}",
            "stack_size": self.stack_size,
            "stack_rva": self.stack_rva,
            "context_rva": self.context_rva,
            "context_size": self.context_size,
        }


def exception_name(code: int) -> str:
    """Human-readable NTSTATUS name for one exception code."""
    code = int(code) & 0xFFFFFFFF
    return EXCEPTION_NAMES.get(code, f"UNKNOWN_EXCEPTION_0x{code:08X}")


def _align_up(value: int, alignment: int) -> int:
    if alignment <= 1:
        return value
    remainder = value % alignment
    return value if remainder == 0 else value + (alignment - remainder)


class MinidumpReader:
    """Pure-Python Microsoft minidump (``MDMP``) reader.

    ``source`` may be a path or raw ``bytes`` (useful for fixtures and for
    dumps received over the wire).  Every structural failure raises
    :class:`CdbError` so a hostile or truncated dump can never surface as an
    ``struct.error`` / ``IndexError`` to a caller.
    """

    MAGIC = MDMP_MAGIC

    HEADER_STRUCT = struct.Struct("<4sIIIIIQ")
    DIRECTORY_STRUCT = struct.Struct("<III")
    EXCEPTION_HEAD_STRUCT = struct.Struct("<II")
    EXCEPTION_RECORD_STRUCT = struct.Struct("<IIQQII")
    LOCATION_STRUCT = struct.Struct("<II")
    THREAD_STRUCT = struct.Struct("<IIIIQQIIII")
    MODULE_STRUCT = struct.Struct("<QIIII52s")
    MODULE_TAIL_STRUCT = struct.Struct("<IIIIQQ")
    MODULE_LIST_HEAD_STRUCT = struct.Struct("<I")
    SYSTEM_INFO_STRUCT = struct.Struct("<HHHBBIIIIIHH24s")
    MEMORY64_HEAD_STRUCT = struct.Struct("<Q")

    HEADER_SIZE = HEADER_STRUCT.size
    EXCEPTION_MAXIMUM_PARAMETERS = 15

    def __init__(self, source: Union[str, os.PathLike, bytes, bytearray, memoryview]) -> None:
        self.source = source
        self.path: Optional[Path] = None
        if isinstance(source, (bytes, bytearray, memoryview)):
            self._data = bytes(source)
        else:
            self.path = Path(source)
            try:
                self._data = self.path.read_bytes()
            except OSError as exc:
                raise CdbError(f"cannot read minidump {self.path}: {exc}") from exc
        if len(self._data) < self.HEADER_SIZE:
            raise CdbError(
                f"minidump truncated: {len(self._data)} bytes is shorter than the "
                f"{self.HEADER_SIZE}-byte MINIDUMP_HEADER"
            )
        self._header: Optional[MinidumpHeader] = None
        self._directory: Optional[List[MinidumpDirectory]] = None
        self._exception: Optional[MinidumpException] = None
        self._threads: Optional[List[MinidumpThread]] = None
        self._modules: Optional[List[CdbModule]] = None

    # -- header / directory -------------------------------------------------

    @property
    def data(self) -> bytes:
        return self._data

    def read_header(self) -> MinidumpHeader:
        """Parse and validate the ``MINIDUMP_HEADER`` at offset 0."""
        if self._header is not None:
            return self._header
        (signature, version, num_streams, dir_rva, checksum, timestamp, flags) = (
            self.HEADER_STRUCT.unpack_from(self._data, 0)
        )
        if signature != MDMP_SIGNATURE:
            raise CdbError(
                "invalid minidump magic "
                f"0x{int.from_bytes(signature, 'little'):08x} (expected 0x{MDMP_MAGIC:08x} 'MDMP')"
            )
        header = MinidumpHeader(
            signature=int.from_bytes(signature, "little"),
            version=version,
            number_of_streams=num_streams,
            stream_directory_rva=dir_rva,
            checksum=checksum,
            time_date_stamp=timestamp,
            flags=flags,
            stream_directory_size=num_streams * self.DIRECTORY_STRUCT.size,
        )
        self._header = header
        return header

    def read_directory(self) -> List[MinidumpDirectory]:
        """Parse every ``MINIDUMP_DIRECTORY`` entry of the stream directory."""
        if self._directory is not None:
            return self._directory
        header = self.read_header()
        entries: List[MinidumpDirectory] = []
        for index in range(header.number_of_streams):
            entry_offset = header.stream_directory_rva + index * self.DIRECTORY_STRUCT.size
            if entry_offset + self.DIRECTORY_STRUCT.size > len(self._data):
                raise CdbError(
                    f"stream directory entry {index} at rva 0x{entry_offset:x} "
                    f"is outside the {len(self._data)}-byte dump"
                )
            stream_type, data_size, rva = self.DIRECTORY_STRUCT.unpack_from(self._data, entry_offset)
            if data_size > 0 and rva + data_size > len(self._data):
                raise CdbError(
                    f"stream {STREAM_NAMES.get(stream_type, stream_type)} "
                    f"(size {data_size}, rva 0x{rva:x}) extends past the end of the "
                    f"{len(self._data)}-byte dump"
                )
            entries.append(MinidumpDirectory(stream_type=stream_type, data_size=data_size, rva=rva))
        self._directory = entries
        return entries

    def stream_bytes(self, stream_type: int) -> Optional[bytes]:
        """Raw payload of one stream, or ``None`` when the stream is absent."""
        for entry in self.read_directory():
            if entry.stream_type == stream_type:
                return self._data[entry.rva : entry.rva + entry.data_size]
        return None

    def directory_summary(self) -> List[Dict[str, Any]]:
        return [entry.to_dict() for entry in self.read_directory()]

    # -- helpers ------------------------------------------------------------

    def _slice(self, rva: int, size: int, what: str) -> bytes:
        if rva < 0 or size < 0 or rva + size > len(self._data):
            raise CdbError(
                f"{what} at rva 0x{rva:x} (size {size}) is outside the "
                f"{len(self._data)}-byte dump"
            )
        return self._data[rva : rva + size]

    def read_string(self, rva: int) -> str:
        """Read a ``MINIDUMP_STRING`` (ULONG32 length in bytes + UTF-16LE)."""
        if rva == 0:
            return ""
        head = self._slice(rva, 4, "MINIDUMP_STRING length")
        (length,) = struct.unpack("<I", head)
        if length == 0:
            return ""
        if length > 0x10000:
            raise CdbError(f"implausible MINIDUMP_STRING length {length} at rva 0x{rva:x}")
        raw = self._slice(rva + 4, length + 2, "MINIDUMP_STRING buffer")
        return raw[:length].decode("utf-16-le", "replace")

    # -- streams ------------------------------------------------------------

    def parse_exception(self) -> Optional[MinidumpException]:
        """Parse ``ExceptionStream`` (type 6), or ``None`` when absent."""
        if self._exception is not None:
            return self._exception
        blob = self.stream_bytes(STREAM_EXCEPTION)
        if blob is None:
            return None
        if len(blob) < self.EXCEPTION_HEAD_STRUCT.size + self.EXCEPTION_RECORD_STRUCT.size:
            raise CdbError(
                f"ExceptionStream is {len(blob)} bytes; expected at least "
                f"{self.EXCEPTION_HEAD_STRUCT.size + self.EXCEPTION_RECORD_STRUCT.size}"
            )
        thread_id, _alignment = self.EXCEPTION_HEAD_STRUCT.unpack_from(blob, 0)
        offset = self.EXCEPTION_HEAD_STRUCT.size
        (
            code,
            flags,
            exception_record,
            exception_address,
            number_parameters,
            _unused,
        ) = self.EXCEPTION_RECORD_STRUCT.unpack_from(blob, offset)
        offset += self.EXCEPTION_RECORD_STRUCT.size
        params_size = self.EXCEPTION_MAXIMUM_PARAMETERS * 8
        if offset + params_size > len(blob):
            raise CdbError("ExceptionStream truncated inside ExceptionInformation")
        parameters = list(struct.unpack_from("<15Q", blob, offset))
        offset += params_size
        if offset + self.LOCATION_STRUCT.size > len(blob):
            raise CdbError("ExceptionStream truncated before ThreadContext location")
        context_size, context_rva = self.LOCATION_STRUCT.unpack_from(blob, offset)
        exception = MinidumpException(
            thread_id=thread_id,
            exception_code=code,
            exception_flags=flags,
            exception_record=exception_record,
            exception_address=exception_address,
            number_parameters=min(int(number_parameters), self.EXCEPTION_MAXIMUM_PARAMETERS),
            parameters=parameters[: min(int(number_parameters), self.EXCEPTION_MAXIMUM_PARAMETERS)],
            thread_context_rva=context_rva,
            thread_context_size=context_size,
        )
        self._exception = exception
        return exception

    def parse_threads(self) -> List[MinidumpThread]:
        """Parse ``ThreadListStream`` (type 3) into :class:`MinidumpThread` records."""
        if self._threads is not None:
            return self._threads
        blob = self.stream_bytes(STREAM_THREAD_LIST)
        if blob is None:
            self._threads = []
            return self._threads
        if len(blob) < 4:
            raise CdbError("ThreadListStream is shorter than its count field")
        (count,) = self.MODULE_LIST_HEAD_STRUCT.unpack_from(blob, 0)
        expected = 4 + count * self.THREAD_STRUCT.size
        if count and len(blob) < expected:
            raise CdbError(
                f"ThreadListStream declares {count} threads "
                f"({expected} bytes) but only {len(blob)} bytes are present"
            )
        threads: List[MinidumpThread] = []
        for index in range(count):
            offset = 4 + index * self.THREAD_STRUCT.size
            (
                thread_id,
                suspend_count,
                priority_class,
                priority,
                teb,
                stack_start,
                stack_size,
                stack_rva,
                context_size,
                context_rva,
            ) = self.THREAD_STRUCT.unpack_from(blob, offset)
            threads.append(
                MinidumpThread(
                    thread_id=thread_id,
                    suspend_count=suspend_count,
                    priority_class=priority_class,
                    priority=priority,
                    teb=teb,
                    stack_start=stack_start,
                    stack_size=stack_size,
                    stack_rva=stack_rva,
                    context_rva=context_rva,
                    context_size=context_size,
                )
            )
        self._threads = threads
        return threads

    def parse_modules(self) -> List[CdbModule]:
        """Parse ``ModuleListStream`` (type 4) into :class:`CdbModule` records."""
        if self._modules is not None:
            return self._modules
        blob = self.stream_bytes(STREAM_MODULE_LIST)
        if blob is None:
            self._modules = []
            return self._modules
        if len(blob) < 4:
            raise CdbError("ModuleListStream is shorter than its count field")
        (count,) = self.MODULE_LIST_HEAD_STRUCT.unpack_from(blob, 0)
        record_size = self.MODULE_STRUCT.size + self.MODULE_TAIL_STRUCT.size
        expected = 4 + count * record_size
        if count and len(blob) < expected:
            raise CdbError(
                f"ModuleListStream declares {count} modules "
                f"({expected} bytes) but only {len(blob)} bytes are present"
            )
        modules: List[CdbModule] = []
        for index in range(count):
            offset = 4 + index * record_size
            base_addr, size_of_image, _checksum, _timestamp, name_rva, _version = (
                self.MODULE_STRUCT.unpack_from(blob, offset)
            )
            path = ""
            try:
                path = self.read_string(name_rva)
            except CdbError:
                path = ""
            # Minidump image paths are Windows paths by definition (MDMP is a
            # Windows format). `Path` splits on the host's separator, so on
            # POSIX it returns the whole `C:\\Games\\game.exe` string instead
            # of the basename; `PureWindowsPath` splits the same everywhere.
            name = PureWindowsPath(path).name if path else f"module_{index}"
            modules.append(
                CdbModule(
                    name=name,
                    base_addr=int(base_addr),
                    end_addr=int(base_addr) + int(size_of_image),
                    path=path,
                )
            )
        self._modules = modules
        return modules

    def parse_system_info(self) -> Dict[str, Any]:
        """Parse ``SystemInfoStream`` (type 7) into a plain mapping."""
        blob = self.stream_bytes(STREAM_SYSTEM_INFO)
        if blob is None or len(blob) < self.SYSTEM_INFO_STRUCT.size:
            return {}
        (
            processor_architecture,
            processor_level,
            processor_revision,
            number_of_processors,
            product_type,
            major_version,
            minor_version,
            build_number,
            platform_id,
            csd_version_rva,
            suite_mask,
            _reserved2,
            _cpu_union,
        ) = self.SYSTEM_INFO_STRUCT.unpack_from(blob, 0)
        arch_names = {0: "x86", 5: "arm", 6: "ia64", 9: "x64", 12: "arm64"}
        return {
            "processor_architecture": processor_architecture,
            "processor_architecture_name": arch_names.get(
                processor_architecture, f"unknown_{processor_architecture}"
            ),
            "processor_level": processor_level,
            "processor_revision": processor_revision,
            "number_of_processors": number_of_processors,
            "product_type": product_type,
            "os_version": f"{major_version}.{minor_version}.{build_number}",
            "major_version": major_version,
            "minor_version": minor_version,
            "build_number": build_number,
            "platform_id": platform_id,
            "csd_version_rva": csd_version_rva,
            "suite_mask": suite_mask,
        }

    def parse_registers(self) -> CdbRegisters:
        """Registers of the crashing thread, else the first thread's context."""
        exception = self.parse_exception()
        if exception is not None and exception.thread_context_size:
            try:
                blob = self._slice(
                    exception.thread_context_rva,
                    exception.thread_context_size,
                    "exception thread CONTEXT",
                )
            except CdbError:
                blob = b""
            if len(blob) >= _X64_CONTEXT_MIN_SIZE:
                return parse_x64_context(blob)
        for thread in self.parse_threads():
            if thread.context_size >= _X64_CONTEXT_MIN_SIZE:
                try:
                    blob = self._slice(thread.context_rva, thread.context_size, "thread CONTEXT")
                except CdbError:
                    continue
                return parse_x64_context(blob)
        return CdbRegisters()

    # -- report -------------------------------------------------------------

    def analyze(self) -> CrashReport:
        """Produce a :class:`CrashReport` from the dump."""
        header = self.read_header()
        exception = self.parse_exception()
        threads = self.parse_threads()
        modules = self.parse_modules()
        system_info = self.parse_system_info()
        registers = self.parse_registers()
        if exception is not None:
            exception_code = int(exception.exception_code)
            faulting_ip = int(exception.exception_address) or registers.rip
            faulting_address = exception.faulting_address
            faulting_operation = exception.faulting_operation
            thread_id: Optional[int] = exception.thread_id
            parameters = list(exception.parameters)
        else:
            exception_code = 0
            faulting_ip = registers.rip
            faulting_address = None
            faulting_operation = None
            thread_id = threads[0].thread_id if threads else None
            parameters = []
        return CrashReport(
            exception_code=exception_code,
            exception_name=(
                exception.exception_name if exception is not None else "NO_EXCEPTION_RECORD"
            ),
            faulting_ip=faulting_ip,
            faulting_address=faulting_address,
            registers=registers,
            stack_trace=[],
            modules=modules,
            parameters=parameters,
            faulting_operation=faulting_operation,
            thread_id=thread_id,
            source="minidump",
            dump_path=str(self.path) if self.path is not None else None,
            has_exception_stream=exception is not None,
            threads=threads,
            system_info=system_info,
            streams={
                "header": header.to_dict(),
                "directory": self.directory_summary(),
            },
        )


def parse_x64_context(blob: Union[bytes, bytearray, memoryview]) -> CdbRegisters:
    """Decode an x64 ``CONTEXT`` record into :class:`CdbRegisters`."""
    data = bytes(blob)
    if len(data) < _X64_CONTEXT_MIN_SIZE:
        raise CdbParseError(
            f"x64 CONTEXT record is {len(data)} bytes; need at least {_X64_CONTEXT_MIN_SIZE}"
        )
    registers = CdbRegisters()
    for index, name in enumerate(_X64_CONTEXT_ORDER):
        value = struct.unpack_from("<Q", data, _X64_CONTEXT_RAX + index * 8)[0]
        setattr(registers, name, int(value))
    registers.rip = int(struct.unpack_from("<Q", data, _X64_CONTEXT_RIP)[0])
    registers.eflags = int(struct.unpack_from("<I", data, _X64_CONTEXT_EFLAGS)[0])
    return registers


# ---------------------------------------------------------------------------
# CDB text parsers (r / kp / lm)
# ---------------------------------------------------------------------------


def parse_cdb_registers(text: str) -> CdbRegisters:
    """Parse a CDB ``r`` dump into :class:`CdbRegisters`.

    Bare values are hexadecimal (CDB prints them that way); ``0x`` prefixes are
    accepted too.  Unknown mnemonics (``cs``, ``iopl``, ``nv up ei``) are
    ignored so a full session transcript can be handed in verbatim.
    """
    registers = CdbRegisters()
    if not text:
        return registers
    for match in _REGISTER_RE.finditer(text):
        canonical = _REGISTER_ALIASES.get(match.group("name").lower())
        if canonical is None:
            continue
        raw = match.group("value")
        try:
            value = int(raw, 16)
        except ValueError:  # pragma: no cover - regex constrains the shape
            continue
        setattr(registers, canonical, value & 0xFFFFFFFFFFFFFFFF)
    return registers


_STACK_FRAME_RE = re.compile(
    r"^\s*(?P<frame>\d+)\s+"
    r"(?P<sp>[0-9A-Fa-f]{6,16}(?:`[0-9A-Fa-f]{8})?)\s+"
    r"(?P<ret>[0-9A-Fa-f]{6,16}(?:`[0-9A-Fa-f]{8})?)\s*"
    r"(?P<site>\S.*?)?\s*$"
)

_MODULE_LINE_RE = re.compile(
    r"^\s*(?P<start>[0-9A-Fa-f]{6,16}(?:`[0-9A-Fa-f]{8})?)\s+"
    r"(?P<end>[0-9A-Fa-f]{6,16}(?:`[0-9A-Fa-f]{8})?)\s+"
    r"(?P<name>[^\s]+)\s*"
    r"(?P<rest>.*?)\s*$"
)


def _hex_operand(token: str) -> int:
    """Convert a CDB hex operand (optionally ``hi`lo``) into an int."""
    return int(token.replace("`", ""), 16)


def parse_cdb_call_stack(text: str) -> List[CdbStackFrame]:
    """Parse a CDB ``kp`` / ``k`` listing into :class:`CdbStackFrame` records.

    Recognises the canonical ``# Child-SP  RetAddr  Call Site`` table, with or
    without the backtick hex separators, and ignores header/annotation lines.
    """
    frames: List[CdbStackFrame] = []
    if not text:
        return frames
    for line in text.splitlines():
        if not line.strip():
            continue
        lowered = line.strip().lower()
        if lowered.startswith(("#", "unloaded modules", "***", "child-sp")):
            continue
        match = _STACK_FRAME_RE.match(line)
        if not match:
            continue
        site = (match.group("site") or "").strip()
        frames.append(
            CdbStackFrame(
                frame_num=int(match.group("frame")),
                child_sp=_hex_operand(match.group("sp")),
                ret_addr=_hex_operand(match.group("ret")),
                call_site=site,
            )
        )
    frames.sort(key=lambda f: f.frame_num)
    return frames


def _looks_like_path(token: str) -> bool:
    if not token:
        return False
    return ("\\" in token) or ("/" in token) or (":" in token)


def parse_cdb_modules(text: str) -> List[CdbModule]:
    """Parse a CDB ``lm`` listing into :class:`CdbModule` records.

    Handles the ``start end module name [path]`` table produced by ``lm`` and
    ``lm v``; the trailing ``(deferred)`` marker is stripped and only genuine
    image paths are recorded in :attr:`CdbModule.path`.
    """
    modules: List[CdbModule] = []
    if not text:
        return modules
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("Unloaded", "start", "***")):
            continue
        match = _MODULE_LINE_RE.match(line)
        if not match:
            continue
        name = match.group("name")
        if name.lower() in {"module", "name", "(deferred)"}:
            continue
        rest = (match.group("rest") or "").strip()
        if rest.startswith("(deferred)"):
            rest = rest[len("(deferred)") :].strip()
        path = rest if _looks_like_path(rest) else ""
        base = _hex_operand(match.group("start"))
        end = _hex_operand(match.group("end"))
        if base == 0 and end == 0:
            continue
        modules.append(CdbModule(name=name, base_addr=base, end_addr=max(base, end), path=path))
    modules.sort(key=lambda m: (m.base_addr, m.name))
    return modules


# ---------------------------------------------------------------------------
# CDB process driver
# ---------------------------------------------------------------------------


class CdbEngine:
    """Headless driver for the WinDbg console debugger (``cdb.exe``).

    ``cdb_path`` may be an explicit executable path, a sequence argv prefix
    (an executable plus fixed arguments, used by tests and by wrapper
    launchers), or ``None`` to auto-discover the Debugging Tools install.
    """

    def __init__(
        self,
        cdb_path: Optional[Union[str, os.PathLike, Sequence[Union[str, os.PathLike]]]] = None,
        timeout: float = DEFAULT_CDB_TIMEOUT,
        symbol_path: Optional[str] = None,
        extra_env: Optional[Dict[str, str]] = None,
    ) -> None:
        self._argv_prefix: Optional[List[str]] = self._normalize_path(cdb_path)
        self.timeout = float(timeout)
        self.symbol_path = symbol_path
        self.extra_env = dict(extra_env or {})
        self.last_command_line: str = ""

    # -- discovery ----------------------------------------------------------

    @staticmethod
    def find_cdb_executable(
        explicit: Optional[Union[str, os.PathLike]] = None,
    ) -> Optional[Path]:
        """Locate ``cdb.exe``: explicit path, env override, PATH, then kits."""
        candidates: List[Path] = []
        if explicit:
            candidates.append(Path(explicit))
        for var in _CDB_ENV_VARS:
            value = os.environ.get(var)
            if value:
                candidates.append(Path(value))
        for which in (shutil.which("cdb.exe"), shutil.which("cdb")):
            if which:
                candidates.append(Path(which))
        candidates.extend(Path(p) for p in _CDB_CANDIDATE_PATHS)
        kits_root = Path(_CDB_KITS_ROOT)
        if kits_root.is_dir():
            try:
                for version_dir in sorted(
                    (d for d in kits_root.iterdir() if d.is_dir()), reverse=True
                ):
                    for arch in ("x64", "x86", "arm64"):
                        candidates.append(version_dir / arch / "cdb.exe")
            except OSError:  # pragma: no cover - unreadable kits root
                pass
        for candidate in candidates:
            try:
                if candidate.is_file():
                    return candidate
            except OSError:  # pragma: no cover - permission denied on probe
                continue
        return None

    @staticmethod
    def _normalize_path(
        cdb_path: Optional[Union[str, os.PathLike, Sequence[Union[str, os.PathLike]]]],
    ) -> Optional[List[str]]:
        if cdb_path is None:
            found = CdbEngine.find_cdb_executable()
            return [str(found)] if found is not None else None
        if isinstance(cdb_path, (str, os.PathLike)):
            return [str(cdb_path)]
        prefix = [str(part) for part in cdb_path]
        return prefix or None

    @property
    def executable(self) -> Optional[Path]:
        """First element of the argv prefix when it is a real file."""
        if not self._argv_prefix:
            return None
        candidate = Path(self._argv_prefix[0])
        return candidate if candidate.is_file() else None

    @property
    def available(self) -> bool:
        return self._argv_prefix is not None

    def _argv_for(self, args: Sequence[str]) -> List[str]:
        if not self._argv_prefix:
            raise CdbNotFoundError(
                "cdb.exe was not found: install the Windows Debugging Tools, add it to "
                "PATH, or set TOOTHPICK_CDB to the executable"
            )
        return list(self._argv_prefix) + [str(a) for a in args]

    # -- subprocess runner --------------------------------------------------

    def _run_process(
        self, argv: Sequence[str], timeout: Optional[float] = None
    ) -> Tuple[int, str, str, bool, float]:
        """Run ``argv`` under a hard timeout; kill the tree on expiry.

        Returns ``(exit_code, stdout, stderr, timed_out, duration)``.
        """
        budget = float(self.timeout if timeout is None else timeout)
        env = None
        if self.extra_env:
            env = dict(os.environ)
            env.update(self.extra_env)
        creationflags = 0
        if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
            creationflags = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
        self.last_command_line = " ".join(argv)
        started = time.monotonic()
        try:
            process = subprocess.Popen(
                list(argv),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=creationflags,
                env=env,
            )
        except OSError as exc:
            raise CdbError(f"failed to launch {argv[0]!r}: {exc}") from exc
        try:
            stdout, stderr = process.communicate(timeout=max(0.1, budget))
        except subprocess.TimeoutExpired as exc:
            self._terminate_tree(process)
            duration = time.monotonic() - started
            partial_out = exc.stdout or ""
            partial_err = exc.stderr or ""
            if isinstance(partial_out, bytes):  # pragma: no cover - text mode
                partial_out = partial_out.decode("utf-8", "replace")
            if isinstance(partial_err, bytes):  # pragma: no cover - text mode
                partial_err = partial_err.decode("utf-8", "replace")
            return (
                -1,
                partial_out,
                partial_err or f"cdb session timed out after {budget:g}s",
                True,
                duration,
            )
        return int(process.returncode or 0), stdout or "", stderr or "", False, time.monotonic() - started

    @staticmethod
    def _terminate_tree(process: subprocess.Popen) -> None:
        """Kill a timed-out CDB session and its debuggee children, best effort."""
        pid = getattr(process, "pid", None)
        if os.name == "nt" and pid:
            try:
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(pid)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=10,
                    check=False,
                )
            except (OSError, subprocess.SubprocessError):  # pragma: no cover - taskkill absent
                pass
        try:
            process.kill()
        except OSError:  # pragma: no cover - already gone
            pass
        try:
            process.communicate(timeout=10)
        except (subprocess.TimeoutExpired, ValueError, OSError):  # pragma: no cover
            pass

    # -- commands -----------------------------------------------------------

    def build_attach_command(self, pid: int, commands: Sequence[str]) -> List[str]:
        """argv for a non-invasive attach: ``cdb -pv -p <pid> -c "<cmds>"``."""
        argv = self._argv_for(["-pv", "-p", str(int(pid)), "-c", "; ".join(commands)])
        if self.symbol_path:
            argv.extend(["-y", self.symbol_path])
        return argv

    def attach_non_invasive(
        self,
        pid: int,
        commands: Optional[Sequence[str]] = None,
        timeout: Optional[float] = None,
    ) -> CdbSessionResult:
        """Attach non-invasively, run ``commands``, and parse the transcript.

        The debugger never writes to the target (``-pv``), so no breakpoint or
        patch is installed and the game keeps running after ``q`` detaches.
        Raises :class:`CdbTimeoutError` after terminating a hung session.
        """
        try:
            pid_int = int(pid)
        except (TypeError, ValueError) as exc:
            raise CdbError(f"invalid pid {pid!r}") from exc
        if pid_int <= 0:
            raise CdbError(f"invalid pid {pid_int}: must be a positive process id")
        command_list = [str(c).strip() for c in (commands or DEFAULT_ATTACH_COMMANDS) if str(c).strip()]
        if not command_list:
            raise CdbError("no CDB commands supplied; pass at least 'q' to detach")
        argv = self.build_attach_command(pid_int, command_list)
        exit_code, stdout, stderr, timed_out, duration = self._run_process(argv, timeout=timeout)
        if timed_out:
            budget = float(self.timeout if timeout is None else timeout)
            raise CdbTimeoutError(
                f"cdb non-invasive attach to pid {pid_int} timed out after {budget:g}s; "
                "the CDB process tree was terminated"
            )
        session = CdbSessionResult(
            pid=pid_int,
            commands=command_list,
            command_line=self.last_command_line,
            exit_code=exit_code,
            timed_out=False,
            duration_seconds=duration,
            raw_output=stdout,
            stderr=stderr,
            cdb_executable=argv[0],
            registers=parse_cdb_registers(stdout),
            stack_trace=parse_cdb_call_stack(stdout),
            modules=parse_cdb_modules(stdout),
        )
        return session

    def analyze_crash_dump(self, dump_path: Union[str, os.PathLike]) -> CrashReport:
        """Analyze a ``.dmp`` completely offline via :class:`MinidumpReader`.

        No ``cdb.exe`` and no Windows SDK are required: the parse is pure
        Python, so crash triage works on any host (Risk R5).
        """
        path = Path(dump_path)
        if not path.is_file():
            raise CdbError(f"crash dump not found: {path}")
        return MinidumpReader(path).analyze()

    # -- parser entry points (delegating to the module functions) -----------

    @staticmethod
    def parse_registers(text: str) -> CdbRegisters:
        """Parse a CDB ``r`` dump."""
        return parse_cdb_registers(text)

    @staticmethod
    def parse_call_stack(text: str) -> List[CdbStackFrame]:
        """Parse a CDB ``kp`` call stack."""
        return parse_cdb_call_stack(text)

    @staticmethod
    def parse_modules(text: str) -> List[CdbModule]:
        """Parse a CDB ``lm`` module listing."""
        return parse_cdb_modules(text)

    @staticmethod
    def verify_address(
        addr: int, modules: Sequence[CdbModule]
    ) -> Tuple[bool, Optional[CdbModule]]:
        """Return whether ``addr`` falls inside one of ``modules``."""
        address = int(addr)
        for module in modules:
            if module.contains(address):
                return True, module
        return False, None
