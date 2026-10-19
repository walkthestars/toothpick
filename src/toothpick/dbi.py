"""
Toothpick W16: Dynamic Binary Instrumentation (DBI) tracing engine.

Implements chunks 1-3 of ``docs/units/UNIT-W16-DBI-PLAN.md`` (data models, Pin /
DynamoRIO tool generators, trace log parsers) plus the :class:`UnicornDbiTracer`
offline emulator.  Governing specs: ``design-document.md`` A16/A19/A111/A116 #1;
execution-plan W16; Risk R2 (target corruption), R5 (no live host / no engine
installed), R7 (runaway instrumentation).

What this module is
-------------------
* :class:`InstructionTraceEntry`, :class:`MemoryTraceEntry` and
  :class:`DbiTraceResult` -- the DBI trace schema (chunk 1).
* :class:`DbiToolGenerator` -- writes self-contained Intel PIN pintool and
  DynamoRIO client C++ projects (source + build config) for the ``bbtrace``,
  ``inscount`` and ``memtrace`` instrumentation modes (chunk 2).  No Pin or
  DynamoRIO installation is required to *generate* the projects, so this path
  is CI-safe on a bare host (Risk R5).
* :class:`TraceParser` -- reads the ``tdbi/1`` text trace logs that the
  generated tools emit and reconstructs :class:`DbiTraceResult` objects
  (chunk 3, parsers only).  The parser also tolerates the classic Pin
  ``Count <n>`` line, DynamoRIO ``drcov``-style module/BB-table banners and
  ``[tid=...]`` / ``pin:`` / ``dr:`` line prefixes.
* :class:`UnicornDbiTracer` -- offline Unicorn emulation of a raw code blob or of
  a file image, recording executed instructions (``UC_HOOK_CODE``) and memory
  reads/writes (``UC_HOOK_MEM_READ`` / ``UC_HOOK_MEM_WRITE``) on every run
  (chunk 3, runner).

Unicorn execution path (chunk 3, runner)
----------------------------------------
:class:`UnicornDbiTracer` emulates a target under Unicorn and returns a
:class:`DbiTraceResult` with ``engine == "unicorn"``.  Every run is bounded four
ways -- the instruction budget, a wall-clock guard re-checked inside the code
hook, Unicorn's own microsecond timeout, and a synthetic sentinel page that
catches a stray ``ret`` -- so a hostile or buggy target can never spin on the
host (Risk R2/R7).  ``unicorn``, ``capstone`` and ``lief`` are imported *inside*
:meth:`UnicornDbiTracer.trace`, so ``import toothpick.dbi`` never pulls a native
dependency.  CLI wiring (chunk 4) lands in a follow-up drive.

Trace log format (``tdbi/1``)
-----------------------------
Line oriented UTF-8 text; blank lines and ``#`` comments are ignored unless the
comment is the ``# toothpick-dbi`` banner, which carries ``key=value``
metadata.  Record lines (whitespace separated):

``I <insn_addr> <insn_size> <mnemonic> [operands]``
    One executed instruction (:class:`InstructionTraceEntry`).
``B <bb_addr> [<bb_size> [<insn_count>]]``
    One executed basic block; the optional third field is the block's
    instruction count and feeds ``total_instructions`` when no summary exists.
``M <insn_addr> <R|W> <mem_addr> <size> [<value>]``
    One memory access; ``value`` is the little-endian hex byte string of the
    accessed memory (``-`` when the engine could not read it).
``E instructions=<n> bbs=<n> elapsed_sec=<f>``
    End-of-run summary emitted by every generated tool.
``Count <n>``
    Classic Pin ``inscount`` counter line (kept for tool compatibility).

Addresses are written with a ``0x`` prefix; the parser also accepts ``1234h``
and bare hex forms containing ``a``-``f``.  ``engine`` on the returned
:class:`DbiTraceResult` is decided by the *calling* parser
(``parse_pin_log`` -> ``"pin"``), because the file was selected for that
engine; a contradicting ``engine=`` banner token is ignored.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

__all__ = [
    "DBI_LOG_MAGIC",
    "DBI_LOG_FORMAT_VERSION",
    "DBI_PHASE",
    "ENGINE_PIN",
    "ENGINE_DYNAMORIO",
    "ENGINE_UNICORN",
    "ENGINES",
    "TOOL_BBTRACE",
    "TOOL_INSCOUNT",
    "TOOL_MEMTRACE",
    "SUPPORTED_TOOL_TYPES",
    "MEMORY_READ",
    "MEMORY_WRITE",
    "DEFAULT_MAX_INSTRUCTIONS",
    "MAX_TRACE_INSTRUCTIONS",
    "STOP_COMPLETED",
    "STOP_CPU_EXCEPTION",
    "STOP_FELL_OFF_CODE",
    "STOP_INSTRUCTION_BUDGET",
    "STOP_INVALID_INSTRUCTION",
    "STOP_PROTECTED_MEMORY",
    "STOP_RETURN_SENTINEL",
    "STOP_TIMEOUT",
    "STOP_UNMAPPED_MEMORY",
    "DbiError",
    "DbiToolGenerationError",
    "DbiParseError",
    "InstructionTraceEntry",
    "MemoryTraceEntry",
    "DbiTraceResult",
    "DbiToolGenerator",
    "TraceParser",
    "UnicornDbiTracer",
    "encode_memory_value",
    "decode_memory_value",
]

PathLike = Union[str, Path]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Banner written as the first line of every generated trace log.
DBI_LOG_MAGIC = "# toothpick-dbi"
#: Trace log format version; the parser only understands this version.
DBI_LOG_FORMAT_VERSION = "tdbi/1"
#: Receipt phase used when a DBI trace is persisted to ``toothpick.db`` (chunk 4).
DBI_PHASE = "dbi_trace"

ENGINE_PIN = "pin"
ENGINE_DYNAMORIO = "dynamorio"
ENGINE_UNICORN = "unicorn"
ENGINES = (ENGINE_PIN, ENGINE_DYNAMORIO, ENGINE_UNICORN)

TOOL_BBTRACE = "bbtrace"
TOOL_INSCOUNT = "inscount"
TOOL_MEMTRACE = "memtrace"
SUPPORTED_TOOL_TYPES = (TOOL_BBTRACE, TOOL_INSCOUNT, TOOL_MEMTRACE)

MEMORY_READ = "read"
MEMORY_WRITE = "write"

#: Default instruction budget for a single tracing run (plan A2 signature).
DEFAULT_MAX_INSTRUCTIONS = 1_000_000
#: Hard ceiling for emulated tracing (docs/traps.md Trap 6 -- anti-emulation loops).
MAX_TRACE_INSTRUCTIONS = 10_000_000

_UNKNOWN_VALUE_TOKENS = frozenset(
    {"", "-", "?", "na", "n/a", "none", "null", "unknown", "unavailable"}
)
_READ_TOKENS = frozenset({"r", "rd", "read", "load", "ld"})
_WRITE_TOKENS = frozenset({"w", "wr", "write", "store", "st"})

_RE_HEX_DIGITS = re.compile(r"[0-9a-f]+")
_RE_DEC_DIGITS = re.compile(r"[0-9]+")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class DbiError(Exception):
    """Base exception for DBI execution, generation and trace parsing errors."""


class DbiToolGenerationError(DbiError):
    """Raised when a pintool / DynamoRIO client project cannot be generated."""


class DbiParseError(DbiError):
    """Raised when a DBI trace log is missing, unreadable or malformed."""


# ---------------------------------------------------------------------------
# Small coercion helpers (shared by the models and the parsers)
# ---------------------------------------------------------------------------


def _as_path(value: PathLike) -> Path:
    """Return ``value`` as a :class:`pathlib.Path` without touching the disk."""
    if isinstance(value, Path):
        return value
    return Path(str(value))


def _check_non_negative(value: int, token: str) -> int:
    if value < 0:
        raise DbiParseError(f"negative numeric field {token!r}")
    return value


def _parse_address(token: Any) -> int:
    """Parse an address field.

    Accepts ``0x1f``, ``1f`` (bare hex containing ``a``-``f``), ``1234``
    (decimal) and MASM style ``1234h``.
    """
    text = str(token).strip()
    if not text:
        raise DbiParseError("missing address field")
    lowered = text.lower()
    try:
        if lowered.endswith("h") and _RE_HEX_DIGITS.fullmatch(lowered[:-1]):
            return _check_non_negative(int(lowered[:-1], 16), text)
        if lowered.startswith(("0x", "+0x", "-0x")):
            return _check_non_negative(int(lowered, 16), text)
        if lowered.startswith(("0b", "+0b", "-0b")):
            return _check_non_negative(int(lowered, 2), text)
        if _RE_HEX_DIGITS.fullmatch(lowered) and re.search(r"[a-f]", lowered):
            return _check_non_negative(int(lowered, 16), text)
        return _check_non_negative(int(lowered, 10), text)
    except ValueError as exc:
        raise DbiParseError(f"unparsable address field {token!r}") from exc


def _parse_size(token: Any) -> int:
    """Parse a byte-size / count field (decimal, or explicit ``0x`` hex)."""
    text = str(token).strip()
    if not text:
        raise DbiParseError("missing size field")
    lowered = text.lower()
    try:
        if lowered.startswith(("0x", "+0x", "-0x")):
            return _check_non_negative(int(lowered, 16), text)
        return _check_non_negative(int(lowered, 10), text)
    except ValueError as exc:
        raise DbiParseError(f"unparsable size field {token!r}") from exc


def _try_parse_size(token: Any) -> Optional[int]:
    try:
        return _parse_size(token)
    except DbiParseError:
        return None


def _parse_float(token: Any) -> float:
    try:
        value = float(str(token).strip())
    except (TypeError, ValueError) as exc:
        raise DbiParseError(f"unparsable float field {token!r}") from exc
    if value != value:  # NaN guard
        raise DbiParseError(f"nan in float field {token!r}")
    return value


def _coerce_int(value: Any, default: int = 0) -> int:
    """Best-effort int coercion used when rehydrating JSON payloads."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        try:
            return _parse_address(text)
        except DbiParseError:
            return default
    return default


def _coerce_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        try:
            return float(text)
        except ValueError:
            return default
    return default


def _coerce_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, (bytearray, memoryview)):
        return bytes(value)
    if isinstance(value, (list, tuple)):
        out = bytearray()
        for item in value:
            try:
                out.append(int(item) & 0xFF)
            except (TypeError, ValueError):
                return b""
        return bytes(out)
    if isinstance(value, str):
        try:
            return decode_memory_value(value)
        except DbiParseError:
            return b""
    return b""


def encode_memory_value(value: Union[bytes, bytearray, memoryview]) -> str:
    """Render ``value`` as the ``0x``-prefixed little-endian hex of ``tdbi/1``.

    An empty byte string encodes to ``"-"`` (engine could not observe the
    memory contents).
    """
    data = bytes(value)
    if not data:
        return "-"
    return "0x" + data.hex()


def decode_memory_value(text: str, size: int = 0) -> bytes:
    """Decode a ``tdbi/1`` memory value field into raw bytes.

    ``size`` (when > 0) truncates over-long payloads to the recorded access
    width; unknown markers (``-``, ``?``, ``na``, ...) decode to ``b""``.
    """
    token = str(text).strip()
    if token.lower() in _UNKNOWN_VALUE_TOKENS:
        return b""
    lowered = token.lower()
    if lowered.startswith(("0x", "-0x", "+0x")):
        digits = lowered.split("0x", 1)[1]
    elif lowered.endswith("h"):
        digits = lowered[:-1]
    else:
        digits = lowered
    if not _RE_HEX_DIGITS.fullmatch(digits):
        raise DbiParseError(f"unparsable memory value field {text!r}")
    if len(digits) % 2:
        digits = "0" + digits
    try:
        data = bytes.fromhex(digits)
    except ValueError as exc:  # pragma: no cover - guarded by the regex above
        raise DbiParseError(f"unparsable memory value field {text!r}") from exc
    if size > 0 and len(data) > size:
        data = data[:size]
    return data


# ---------------------------------------------------------------------------
# Chunk 1 -- Data models & trace schema
# ---------------------------------------------------------------------------


@dataclass
class InstructionTraceEntry:
    """One executed instruction recorded by ``bbtrace`` / ``memtrace`` tools."""

    address: int
    size: int = 0
    mnemonic: str = ""
    operands: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "address": int(self.address),
            "address_hex": f"0x{int(self.address):x}",
            "size": int(self.size),
            "mnemonic": self.mnemonic,
            "operands": self.operands,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InstructionTraceEntry":
        return cls(
            address=_coerce_int(data.get("address")),
            size=_coerce_int(data.get("size")),
            mnemonic=str(data.get("mnemonic", "")),
            operands=str(data.get("operands", "")),
        )


@dataclass
class MemoryTraceEntry:
    """One memory read or write observed while tracing.

    ``address`` is the address of the instrumented instruction, ``mem_addr``
    the accessed location and ``value`` the raw little-endian bytes (``b""``
    when the engine could not read the contents).
    """

    address: int
    kind: str
    mem_addr: int
    size: int = 0
    value: bytes = b""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "address": int(self.address),
            "address_hex": f"0x{int(self.address):x}",
            "kind": self.kind,
            "mem_addr": int(self.mem_addr),
            "mem_addr_hex": f"0x{int(self.mem_addr):x}",
            "size": int(self.size),
            "value": encode_memory_value(self.value),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryTraceEntry":
        kind = normalize_memory_kind(str(data.get("kind", MEMORY_READ)))
        return cls(
            address=_coerce_int(data.get("address")),
            kind=kind,
            mem_addr=_coerce_int(data.get("mem_addr")),
            size=_coerce_int(data.get("size")),
            value=_coerce_bytes(data.get("value", b"")),
        )


@dataclass
class DbiTraceResult:
    """Structured result of one DBI tracing run (plan A2 trace schema)."""

    engine: str
    target_sha256: str = ""
    total_instructions: int = 0
    executed_bbs: List[int] = field(default_factory=list)
    memory_accesses: List[MemoryTraceEntry] = field(default_factory=list)
    elapsed_sec: float = 0.0

    @property
    def bb_count(self) -> int:
        """Number of distinct executed basic blocks."""
        return len(self.executed_bbs)

    @property
    def memory_reads(self) -> List[MemoryTraceEntry]:
        return [entry for entry in self.memory_accesses if entry.kind == MEMORY_READ]

    @property
    def memory_writes(self) -> List[MemoryTraceEntry]:
        return [entry for entry in self.memory_accesses if entry.kind == MEMORY_WRITE]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "engine": self.engine,
            "target_sha256": self.target_sha256,
            "total_instructions": int(self.total_instructions),
            "executed_bbs": [int(address) for address in self.executed_bbs],
            "memory_accesses": [entry.to_dict() for entry in self.memory_accesses],
            "elapsed_sec": float(self.elapsed_sec),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DbiTraceResult":
        raw_bbs = data.get("executed_bbs") or []
        raw_accesses = data.get("memory_accesses") or []
        accesses: List[MemoryTraceEntry] = []
        for item in raw_accesses:
            if isinstance(item, MemoryTraceEntry):
                accesses.append(item)
            elif isinstance(item, dict):
                accesses.append(MemoryTraceEntry.from_dict(item))
        return cls(
            engine=str(data.get("engine", "")),
            target_sha256=str(data.get("target_sha256", "")),
            total_instructions=_coerce_int(data.get("total_instructions")),
            executed_bbs=[_coerce_int(address) for address in raw_bbs],
            memory_accesses=accesses,
            elapsed_sec=_coerce_float(data.get("elapsed_sec")),
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> "DbiTraceResult":
        return cls.from_dict(json.loads(payload))


def normalize_memory_kind(kind: str) -> str:
    """Normalise a memory access kind token to :data:`MEMORY_READ` / ``write``."""
    token = str(kind).strip().lower()
    if token in _READ_TOKENS:
        return MEMORY_READ
    if token in _WRITE_TOKENS:
        return MEMORY_WRITE
    raise DbiParseError(f"unknown memory access kind {kind!r}")


# ---------------------------------------------------------------------------
# Chunk 2 -- Pintool & DynamoRIO scaffolding generators
# ---------------------------------------------------------------------------

_PIN_RECORDS_DOC = {
    TOOL_BBTRACE: (
        " *   # toothpick-dbi v1 engine=pin tool=bbtrace pid=<pid>\n"
        " *   B <bb_addr_hex> <bb_size> <insn_count>\n"
        " *   E instructions=<n> bbs=<n> elapsed_sec=<f>"
    ),
    TOOL_INSCOUNT: (
        " *   # toothpick-dbi v1 engine=pin tool=inscount pid=<pid>\n"
        " *   Count <n>\n"
        " *   E instructions=<n> bbs=<n> elapsed_sec=<f>"
    ),
    TOOL_MEMTRACE: (
        " *   # toothpick-dbi v1 engine=pin tool=memtrace pid=<pid>\n"
        " *   B <bb_addr_hex> <bb_size> <insn_count>\n"
        " *   I <insn_addr_hex> <insn_size> <mnemonic> <operands>\n"
        " *   M <insn_addr_hex> <R|W> <mem_addr_hex> <size> <value|->\n"
        " *   E instructions=<n> bbs=<n> elapsed_sec=<f>"
    ),
}

_DR_RECORDS_DOC = {
    TOOL_BBTRACE: (
        " *   # toothpick-dbi v1 engine=dynamorio tool=bbtrace\n"
        " *   B <bb_addr_hex> <bb_size> <insn_count>\n"
        " *   E instructions=<n> bbs=<n> elapsed_sec=<f>"
    ),
    TOOL_INSCOUNT: (
        " *   # toothpick-dbi v1 engine=dynamorio tool=inscount\n"
        " *   Count <n>\n"
        " *   E instructions=<n> bbs=<n> elapsed_sec=<f>"
    ),
    TOOL_MEMTRACE: (
        " *   # toothpick-dbi v1 engine=dynamorio tool=memtrace\n"
        " *   B <bb_addr_hex> <bb_size> <insn_count>\n"
        " *   I <insn_addr_hex> <insn_size> <mnemonic>\n"
        " *   M <insn_addr_hex> <R|W> <mem_addr_hex> <size> <value|->\n"
        " *   E instructions=<n> bbs=<n> elapsed_sec=<f>"
    ),
}

# --- Intel PIN: shared prologue -------------------------------------------

_PIN_PROLOGUE = """/*
 * Toothpick DBI pintool template -- Intel PIN.
 *
 * engine : pin
 * tool   : @TOOL@
 * format : @FORMAT@ (read back by toothpick.dbi.TraceParser.parse_pin_log)
 *
 * Records written to the log:
@RECORDS@
 *
 * Build (canonical Pin layout, this directory holds Makefile + makefile.rules):
 *   Linux  : export PIN_ROOT=/opt/pin-3.30-98830-g1d7b601b3-gcc-linux && make
 *   Windows: set PIN_ROOT=C:\\pin-3.30-98830-g1d7b601b3-msvc-windows
 *            nmake /f Makefile
 * Run:
 *   %PIN_ROOT%\\pin.exe -t obj-intel64\\pintool_@TOOL@.dll -o trace_@TOOL@.log -- target.exe
 *
 * The tool is non-invasive: it never writes target memory, never terminates the
 * target, and -budget only stops further *log writes* after N instructions
 * (Risk R7 budget enforcement).
 */

#if defined(_MSC_VER)
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "pin.H"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <string>

namespace {

/* ------------------------------------------------------------------ knobs */

KNOB<std::string> KnobOutFile(KNOB_MODE_WRITEONCE, "pintool", "o", "pin_@TOOL@.log",
                              "toothpick @FORMAT@ trace log to create");
KNOB<UINT64> KnobBudget(KNOB_MODE_WRITEONCE, "pintool", "budget", "0",
                        "stop writing records after N instructions (0 = unlimited)");

/* ------------------------------------------------------------------ state */

FILE *g_log = NULL;
UINT64 g_instructions = 0;
UINT64 g_blocks = 0;
UINT64 g_budget = 0;
clock_t g_start = 0;

bool LoggingEnabled()
{
    if (g_log == NULL) {
        return false;
    }
    if (g_budget != 0 && g_instructions >= g_budget) {
        return false;
    }
    return true;
}

void WriteHeader(const char *tool)
{
    fprintf(g_log, "# toothpick-dbi v1 engine=pin tool=%s pid=%d\\n", tool, (int)PIN_GetPid());
    fflush(g_log);
}

double ElapsedSeconds()
{
    return (double)(clock() - g_start) / (double)CLOCKS_PER_SEC;
}

void WriteSummary()
{
    fprintf(g_log, "E instructions=%llu bbs=%llu elapsed_sec=%.6f\\n",
            (unsigned long long)g_instructions, (unsigned long long)g_blocks,
            ElapsedSeconds());
    fflush(g_log);
}
"""

# --- Intel PIN: per-tool hook sections ------------------------------------

_PIN_HOOKS = {
    TOOL_BBTRACE: """
/* -------------------------------------------------------- record writers */

void WriteBlock(ADDRINT address, UINT32 size, UINT32 insn_count)
{
    if (!LoggingEnabled()) {
        return;
    }
    g_blocks++;
    g_instructions += insn_count;
    fprintf(g_log, "B 0x%016llx %u %u\\n", (unsigned long long)address, (unsigned)size,
            (unsigned)insn_count);
    fflush(g_log);
}

/* ------------------------------------------------------------------ hooks */

/* Walk BBL_InsHead..BBL_InsTail (INS_Next alone would run into the next
   block once the tail is reached) so the block size and instruction count
   stay inside the basic block. */
void DescribeBlock(BBL block, ADDRINT *address, UINT32 *size, UINT32 *insn_count)
{
    INS first = BBL_InsHead(block);
    INS last = BBL_InsTail(block);
    UINT32 count = 0;
    ADDRINT start = INS_Address(first);
    ADDRINT end = INS_Address(last) + (ADDRINT)INS_Size(last);
    for (INS ins = first; INS_Valid(ins); ins = INS_Next(ins)) {
        count++;
        if (ins == last) {
            break;
        }
    }
    *address = start;
    *size = (UINT32)(end - start);
    *insn_count = count;
}

VOID OnBlock(BBL block, VOID *v)
{
    ADDRINT address = 0;
    UINT32 size = 0;
    UINT32 insn_count = 0;
    DescribeBlock(block, &address, &size, &insn_count);
    BBL_InsertCall(block, IPOINT_BEFORE, (AFUNPTR)WriteBlock,
                   IARG_ADDRINT, address,
                   IARG_UINT32, size,
                   IARG_UINT32, insn_count,
                   IARG_END);
}

void RegisterHooks()
{
    BBL_AddInstrumentFunction(OnBlock, 0);
}

void WriteToolSummary()
{
    fflush(g_log);
}
""",
    TOOL_INSCOUNT: """
/* -------------------------------------------------------- record writers */

void WriteCount(UINT64 count)
{
    /* Classic Pin inscount line, kept so legacy log consumers still work. */
    fprintf(g_log, "Count %llu\\n", (unsigned long long)count);
    fflush(g_log);
}

/* ------------------------------------------------------------------ hooks */

VOID RecordInstruction()
{
    if (g_budget != 0 && g_instructions >= g_budget) {
        return;
    }
    g_instructions++;
}

VOID RecordBlock()
{
    if (g_budget != 0 && g_instructions >= g_budget) {
        return;
    }
    g_blocks++;
}

VOID OnInstruction(INS ins, VOID *v)
{
    if (!INS_Valid(ins)) {
        return;
    }
    /* Predicated: conditionally executed instructions must not be counted. */
    INS_InsertPredicatedCall(ins, IPOINT_BEFORE, (AFUNPTR)RecordInstruction, IARG_END);
}

VOID OnBlock(BBL block, VOID *v)
{
    BBL_InsertCall(block, IPOINT_BEFORE, (AFUNPTR)RecordBlock, IARG_END);
}

void RegisterHooks()
{
    INS_AddInstrumentFunction(OnInstruction, 0);
    BBL_AddInstrumentFunction(OnBlock, 0);
}

void WriteToolSummary()
{
    WriteCount(g_instructions);
}
""",
    TOOL_MEMTRACE: """
/* -------------------------------------------------------- record writers */

void WriteInstruction(ADDRINT address, UINT32 size, const char *disassembly)
{
    const char *text = disassembly;
    const char *separator;
    if (!LoggingEnabled()) {
        free((void *)disassembly);
        return;
    }
    g_instructions++;
    if (text == NULL || text[0] == '\\0') {
        fprintf(g_log, "I 0x%016llx %u unknown\\n", (unsigned long long)address, (unsigned)size);
    } else {
        separator = strchr(text, ' ');
        if (separator == NULL) {
            fprintf(g_log, "I 0x%016llx %u %s\\n", (unsigned long long)address, (unsigned)size, text);
        } else {
            std::string mnemonic(text, (size_t)(separator - text));
            fprintf(g_log, "I 0x%016llx %u %s %s\\n", (unsigned long long)address, (unsigned)size,
                    mnemonic.c_str(), separator + 1);
        }
    }
    fflush(g_log);
    free((void *)disassembly);
}

void WriteMemory(ADDRINT instruction, const char *kind, ADDRINT address, UINT32 size)
{
    unsigned char buffer[64];
    char hex[2 * sizeof(buffer) + 3];
    size_t index;
    if (!LoggingEnabled()) {
        return;
    }
    if (size == 0 || size > sizeof(buffer) ||
        PIN_SafeCopy(buffer, (const VOID *)address, size) != size) {
        fprintf(g_log, "M 0x%016llx %s 0x%016llx %u -\\n", (unsigned long long)instruction, kind,
                (unsigned long long)address, (unsigned)size);
        fflush(g_log);
        return;
    }
    hex[0] = '0';
    hex[1] = 'x';
    for (index = 0; index < size; index++) {
        snprintf(hex + 2 + 2 * index, 3, "%02x", (unsigned)buffer[index]);
    }
    hex[2 + 2 * size] = '\\0';
    fprintf(g_log, "M 0x%016llx %s 0x%016llx %u %s\\n", (unsigned long long)instruction, kind,
            (unsigned long long)address, (unsigned)size, hex);
    fflush(g_log);
}

/* ------------------------------------------------------------------ hooks */

VOID OnInstruction(INS ins, VOID *v)
{
    if (!INS_Valid(ins)) {
        return;
    }
    /* The mnemonic/operand text is captured at instrumentation time and freed
       by WriteInstruction (standard Pin strdup idiom). */
    INS_InsertPredicatedCall(ins, IPOINT_BEFORE, (AFUNPTR)WriteInstruction,
                             IARG_INST_PTR, IARG_UINT32, INS_Size(ins),
                             IARG_PTR, strdup(INS_Disassemble(ins).c_str()), IARG_END);
    if (INS_IsMemoryRead(ins)) {
        /* Inserted before the load: PIN_SafeCopy observes the pre-read value. */
        INS_InsertPredicatedCall(ins, IPOINT_BEFORE, (AFUNPTR)WriteMemory,
                                 IARG_INST_PTR, IARG_PTR, "R",
                                 IARG_MEMORYREAD_EA, IARG_MEMORYREAD_SIZE, IARG_END);
        if (INS_HasMemoryRead2(ins)) {
            INS_InsertPredicatedCall(ins, IPOINT_BEFORE, (AFUNPTR)WriteMemory,
                                     IARG_INST_PTR, IARG_PTR, "R",
                                     IARG_MEMORYREAD2_EA, IARG_MEMORYREAD_SIZE, IARG_END);
        }
    }
    if (INS_IsMemoryWrite(ins)) {
        /* Inserted after the store: PIN_SafeCopy observes the stored value. */
        INS_InsertPredicatedCall(ins, IPOINT_AFTER, (AFUNPTR)WriteMemory,
                                 IARG_INST_PTR, IARG_PTR, "W",
                                 IARG_MEMORYWRITE_EA, IARG_MEMORYWRITE_SIZE, IARG_END);
    }
}

void RegisterHooks()
{
    INS_AddInstrumentFunction(OnInstruction, 0);
}

void WriteToolSummary()
{
    fflush(g_log);
}
""",
}

# --- Intel PIN: shared epilogue -------------------------------------------

_PIN_EPILOGUE = """
VOID Fini(INT32 code, VOID *v)
{
    if (g_log != NULL) {
        WriteToolSummary();
        WriteSummary();
        fclose(g_log);
        g_log = NULL;
    }
}

INT32 Usage()
{
    fprintf(stderr, "toothpick pintool template (tool=@TOOL@)\\n");
    fprintf(stderr, "  -o <file>       @FORMAT@ trace log to create\\n");
    fprintf(stderr, "  -budget <n>     stop writing records after n instructions\\n");
    return -1;
}

} /* namespace */

int main(int argc, CHAR *argv[])
{
    if (PIN_Init(argc, argv)) {
        return Usage();
    }
    PIN_InitSymbols();
    g_budget = KnobBudget.Value();
    g_log = fopen(KnobOutFile.Value().c_str(), "w");
    if (g_log == NULL) {
        fprintf(stderr, "toothpick: cannot open trace log '%s'\\n", KnobOutFile.Value().c_str());
        return -2;
    }
    g_start = clock();
    WriteHeader("@TOOL@");
    RegisterHooks();
    PIN_AddFiniFunction(Fini, 0);
    PIN_StartProgram();
    return 0;
}
"""

_PIN_MAKEFILE = """# Toothpick DBI pintool build config -- Intel Pin canonical layout (tool=@TOOL@).
#
#   Linux  : export PIN_ROOT=/opt/pin-3.30-98830-g1d7b601b3-gcc-linux && make
#   Windows: set PIN_ROOT=C:\\pin-3.30-98830-g1d7b601b3-msvc-windows
#            nmake /f Makefile
#
# PIN_ROOT can also be passed inline:  make PIN_ROOT=/path/to/pin

ifdef PIN_ROOT
CONFIG_ROOT := $(PIN_ROOT)/source/tools/Config
else
$(error PIN_ROOT is not set; point it at the Intel Pin kit root)
endif

TOOLS_ROOT ?= $(PIN_ROOT)/source/tools

include $(CONFIG_ROOT)/makefile.config
include makefile.rules
include $(TOOLS_ROOT)/Config/makefile.default.rules
"""

_PIN_MAKEFILE_RULES = """# Toothpick DBI pintool tool list (consumed by $(CONFIG_ROOT)/makefile.config).
TOOL_ROOTS := pintool_@TOOL@
"""

# --- DynamoRIO: shared prologue -------------------------------------------

_DR_PROLOGUE = """/*
 * Toothpick DBI DynamoRIO client template.
 *
 * engine : dynamorio
 * tool   : @TOOL@
 * format : @FORMAT@ (read back by toothpick.dbi.TraceParser.parse_dynamorio_log)
 *
 * Records written to the log:
@RECORDS@
 *
 * Build:
 *   cmake -B build -DDynamoRIO_DIR=<DynamoRIO>/cmake
 *   cmake --build build --config Release
 * Run:
 *   <DynamoRIO>/bin64/drrun.exe -c drclient_@TOOL@.dll -o trace_@TOOL@.log -- target.exe
 *
 * The client is non-invasive: it never writes target memory, never terminates
 * the target, and -budget only stops further *log writes* after N instructions
 * (Risk R7 budget enforcement).
 */

#include "dr_api.h"
#include "drmgr.h"
@INCLUDES@

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace {

/* ------------------------------------------------------------- client state */

static FILE *g_log = NULL;
static const char *g_out = "dr_@TOOL@.log";
static uint64_t g_instructions = 0;
static uint64_t g_blocks = 0;
static uint64_t g_budget = 0;
static uint64_t g_start_us = 0;

static bool LoggingEnabled(void)
{
    if (g_log == NULL) {
        return false;
    }
    if (g_budget != 0 && g_instructions >= g_budget) {
        return false;
    }
    return true;
}

static void WriteHeader(const char *tool)
{
    fprintf(g_log, "# toothpick-dbi v1 engine=dynamorio tool=%s\\n", tool);
    fflush(g_log);
}

static double ElapsedSeconds(void)
{
    return (double)(dr_get_microseconds() - g_start_us) / 1000000.0;
}

static void WriteSummary(void)
{
    fprintf(g_log, "E instructions=%llu bbs=%llu elapsed_sec=%.6f\\n",
            (unsigned long long)g_instructions, (unsigned long long)g_blocks,
            ElapsedSeconds());
    fflush(g_log);
}

static void WriteBlock(app_pc address, uint32 size, uint32 insn_count)
{
    if (!LoggingEnabled()) {
        return;
    }
    g_blocks++;
    g_instructions += insn_count;
    fprintf(g_log, "B 0x%016llx %u %u\\n", (unsigned long long)(ptr_uint_t)address,
            (unsigned)size, (unsigned)insn_count);
    fflush(g_log);
}
"""

# --- DynamoRIO: per-tool sections -----------------------------------------

_DR_INCLUDES = {
    TOOL_BBTRACE: "",
    TOOL_INSCOUNT: "",
    TOOL_MEMTRACE: '#include "drreg.h"\n#include "drutil.h"',
}

_DR_INSERT = {
    TOOL_BBTRACE: """
static dr_emit_flags_t
event_bb_insert(void *drcontext, void *tag, instrlist_t *bb, instr_t *instr,
                bool for_trace, bool translating, void *user_data)
{
    instr_t *head;

    if (instr == NULL) {
        /* drmgr may report the basic block start with instr == NULL; no record
           is emitted here because the head-instruction branch below already
           covers it (and stays correct if this callback shape changes). */
        return DR_EMIT_DEFAULT;
    }
    if (!instr_is_app(instr)) {
        return DR_EMIT_DEFAULT;
    }
    head = instrlist_first_app(bb);
    if (instr == head) {
        instr_t *walk;
        uint32 insn_count = 0;
        for (walk = head; walk != NULL; walk = instr_get_next_app(walk)) {
            insn_count++;
        }
        dr_insert_clean_call(drcontext, bb, instr, (void *)WriteBlock, false, 3,
                             OPND_CREATE_INTPTR((ptr_uint_t)dr_fragment_app_pc(tag)),
                             OPND_CREATE_INT32((int)dr_fragment_size(tag)),
                             OPND_CREATE_INT32((int)insn_count));
    }
    return DR_EMIT_DEFAULT;
}
""",
    TOOL_INSCOUNT: """
static void RecordInstruction(void)
{
    if (g_budget != 0 && g_instructions >= g_budget) {
        return;
    }
    g_instructions++;
}

static void WriteCount(uint64_t count)
{
    /* Classic inscount line, kept so legacy log consumers still work. */
    fprintf(g_log, "Count %llu\\n", (unsigned long long)count);
    fflush(g_log);
}

static void WriteToolSummary(void)
{
    WriteCount(g_instructions);
}

static dr_emit_flags_t
event_bb_insert(void *drcontext, void *tag, instrlist_t *bb, instr_t *instr,
                bool for_trace, bool translating, void *user_data)
{
    instr_t *head;

    if (instr == NULL) {
        return DR_EMIT_DEFAULT;
    }
    if (!instr_is_app(instr)) {
        return DR_EMIT_DEFAULT;
    }
    head = instrlist_first_app(bb);
    if (instr == head) {
        dr_insert_clean_call(drcontext, bb, instr, (void *)WriteBlock, false, 3,
                             OPND_CREATE_INTPTR((ptr_uint_t)dr_fragment_app_pc(tag)),
                             OPND_CREATE_INT32((int)dr_fragment_size(tag)),
                             OPND_CREATE_INT32(0));
    }
    dr_insert_clean_call(drcontext, bb, instr, (void *)RecordInstruction, false, 0);
    return DR_EMIT_DEFAULT;
}
""",
    TOOL_MEMTRACE: """
static void WriteInstruction(app_pc pc, uint32 size, const char *mnemonic)
{
    if (!LoggingEnabled()) {
        return;
    }
    g_instructions++;
    /* DR exposes the decode opcode name as the mnemonic; operand text would
       need a full disassembler and is intentionally left to a follow-up. */
    fprintf(g_log, "I 0x%016llx %u %s\\n", (unsigned long long)(ptr_uint_t)pc, (unsigned)size,
            (mnemonic != NULL && mnemonic[0] != '\\0') ? mnemonic : "unknown");
    fflush(g_log);
}

static void WriteMemoryRead(app_pc instruction, app_pc address, uint32 size)
{
    unsigned char buffer[64];
    char hex[2 * sizeof(buffer) + 3];
    uint32 index;
    size_t bytes = 0;

    if (!LoggingEnabled()) {
        return;
    }
    if (size > 0 && size <= sizeof(buffer) && dr_safe_read(address, size, buffer, &bytes) &&
        bytes == size) {
        hex[0] = '0';
        hex[1] = 'x';
        for (index = 0; index < size; index++) {
            snprintf(hex + 2 + 2 * index, 3, "%02x", (unsigned)buffer[index]);
        }
        hex[2 + 2 * size] = '\\0';
    } else {
        strcpy(hex, "-");
    }
    fprintf(g_log, "M 0x%016llx R 0x%016llx %u %s\\n", (unsigned long long)(ptr_uint_t)instruction,
            (unsigned long long)(ptr_uint_t)address, (unsigned)size, hex);
    fflush(g_log);
}

static void WriteMemoryWrite(app_pc instruction, app_pc address, uint32 size, uint64_t value)
{
    char hex[2 * sizeof(uint64_t) + 3];
    uint32 index;

    if (!LoggingEnabled()) {
        return;
    }
    hex[0] = '0';
    hex[1] = 'x';
    /* Store values larger than 64 bits (vector/FP) are reported width-capped. */
    for (index = 0; index < size && index < (uint32)sizeof(uint64_t); index++) {
        snprintf(hex + 2 + 2 * index, 3, "%02x", (unsigned)((value >> (8 * index)) & 0xffu));
    }
    hex[2 + 2 * index] = '\\0';
    fprintf(g_log, "M 0x%016llx W 0x%016llx %u %s\\n", (unsigned long long)(ptr_uint_t)instruction,
            (unsigned long long)(ptr_uint_t)address, (unsigned)size, hex);
    fflush(g_log);
}

static dr_emit_flags_t
event_bb_insert(void *drcontext, void *tag, instrlist_t *bb, instr_t *instr,
                bool for_trace, bool translating, void *user_data)
{
    instr_t *head;
    reg_id_t reg1;
    reg_id_t reg2;
    int i;

    if (instr == NULL) {
        return DR_EMIT_DEFAULT;
    }
    if (!instr_is_app(instr)) {
        return DR_EMIT_DEFAULT;
    }
    head = instrlist_first_app(bb);
    if (instr == head) {
        instr_t *walk;
        uint32 insn_count = 0;
        for (walk = head; walk != NULL; walk = instr_get_next_app(walk)) {
            insn_count++;
        }
        dr_insert_clean_call(drcontext, bb, instr, (void *)WriteBlock, false, 3,
                             OPND_CREATE_INTPTR((ptr_uint_t)dr_fragment_app_pc(tag)),
                             OPND_CREATE_INT32((int)dr_fragment_size(tag)),
                             OPND_CREATE_INT32((int)insn_count));
    }

    dr_insert_clean_call(drcontext, bb, instr, (void *)WriteInstruction, false, 3,
                         OPND_CREATE_INTPTR((ptr_uint_t)instr_get_app_pc(instr)),
                         OPND_CREATE_INT32((int)instr_length(drcontext, instr)),
                         OPND_CREATE_INTPTR((ptr_uint_t)decode_opcode_name(instr_get_opcode(instr))));

    if (!instr_reads_memory(instr) && !instr_writes_memory(instr)) {
        return DR_EMIT_DEFAULT;
    }
    if (drreg_reserve_register(drcontext, bb, instr, NULL, &reg1) != DRREG_SUCCESS) {
        return DR_EMIT_DEFAULT;
    }
    if (drreg_reserve_register(drcontext, bb, instr, NULL, &reg2) != DRREG_SUCCESS) {
        drreg_unreserve_register(drcontext, bb, instr, reg1);
        return DR_EMIT_DEFAULT;
    }

    for (i = 0; i < instr_num_srcs(instr); i++) {
        opnd_t opnd = instr_get_src(instr, i);
        int size;
        if (!opnd_is_memory_reference(opnd)) {
            continue;
        }
        size = drutil_opnd_mem_size_in_bytes(opnd, instr);
        if (size <= 0 || size > 64) {
            continue;
        }
        if (!drutil_insert_get_mem_addr(drcontext, bb, instr, opnd, reg1, reg2)) {
            continue;
        }
        dr_insert_clean_call(drcontext, bb, instr, (void *)WriteMemoryRead, false, 3,
                             OPND_CREATE_INTPTR((ptr_uint_t)instr_get_app_pc(instr)),
                             opnd_create_reg(reg1), OPND_CREATE_INT32(size));
    }

    for (i = 0; i < instr_num_dsts(instr); i++) {
        opnd_t dst = instr_get_dst(instr, i);
        opnd_t value;
        int size;
        if (!opnd_is_memory_reference(dst)) {
            continue;
        }
        size = drutil_opnd_mem_size_in_bytes(dst, instr);
        if (size <= 0 || size > 64) {
            continue;
        }
        /* Stores carry the written value as source operand 0 (memtrace_simple
           recipe); vector/FP sources are reported as a zero-width value. */
        value = instr_get_src(instr, 0);
        if (!opnd_is_immed_int(value) &&
            !(opnd_is_reg(value) && opnd_get_size(value) <= OPSZ_8)) {
            value = OPND_CREATE_INTPTR(0);
        }
        if (!drutil_insert_get_mem_addr(drcontext, bb, instr, dst, reg1, reg2)) {
            continue;
        }
        dr_insert_clean_call(drcontext, bb, instr, (void *)WriteMemoryWrite, false, 4,
                             OPND_CREATE_INTPTR((ptr_uint_t)instr_get_app_pc(instr)),
                             opnd_create_reg(reg1), OPND_CREATE_INT32(size), value);
    }

    drreg_unreserve_register(drcontext, bb, instr, reg1);
    drreg_unreserve_register(drcontext, bb, instr, reg2);
    return DR_EMIT_DEFAULT;
}
""",
}

_DR_INIT = {
    TOOL_BBTRACE: "",
    TOOL_INSCOUNT: "",
    TOOL_MEMTRACE: """    {
        drreg_options_t ops = {sizeof(ops), 3 /* scratch slots */, false};
        if (!drreg_init(&ops)) {
            dr_printf("toothpick: drreg init failed\\n");
            return;
        }
        if (!drutil_init()) {
            dr_printf("toothpick: drutil init failed\\n");
            drreg_exit();
            return;
        }
    }
""",
}

_DR_EXIT_EXTRA = {
    TOOL_BBTRACE: "",
    TOOL_INSCOUNT: "",
    TOOL_MEMTRACE: "    drutil_exit();\n    drreg_exit();\n",
}

_DR_EPILOGUE = """
@TOOL_SUMMARY@
} /* namespace */

static void event_exit(void)
{
    if (g_log != NULL) {
        @SUMMARY_CALL@
        WriteSummary();
        fclose(g_log);
        g_log = NULL;
    }
@EXIT_EXTRA@
    drmgr_exit();
}

DR_EXPORT void
dr_client_main(client_id_t id, int argc, const char *argv[])
{
    int i;

    dr_set_client_name("toothpick drclient_@TOOL@",
                       "https://github.com/walkthestars/Toothpick");
    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            g_out = argv[++i];
        } else if (strcmp(argv[i], "-budget") == 0 && i + 1 < argc) {
            g_budget = (uint64_t)strtoull(argv[++i], NULL, 10);
        }
    }
    g_log = fopen(g_out, "w");
    if (g_log == NULL) {
        dr_fprintf(STDERR, "toothpick: cannot open trace log '%s'\\n", g_out);
        return;
    }
    if (!drmgr_init()) {
        dr_fprintf(STDERR, "toothpick: drmgr init failed\\n");
        fclose(g_log);
        g_log = NULL;
        return;
    }
@INIT_EXTRA@    g_start_us = dr_get_microseconds();
    WriteHeader("@TOOL@");
    drmgr_register_bb_instrumentation_event(NULL, event_bb_insert, NULL);
    dr_register_exit_event(event_exit);
    dr_printf("toothpick drclient_@TOOL@: writing @FORMAT@ trace to %s\\n", g_out);
}
"""

_DR_TOOL_SUMMARY = {
    TOOL_BBTRACE: "static void WriteToolSummary(void)\n{\n    fflush(g_log);\n}",
    TOOL_INSCOUNT: "",  # WriteToolSummary/WriteCount live with the hooks above
    TOOL_MEMTRACE: "static void WriteToolSummary(void)\n{\n    fflush(g_log);\n}",
}

#: Call placed in ``event_exit()`` (@SUMMARY_CALL@) to flush the tool's counter.
_DR_SUMMARY_CALL = {
    TOOL_BBTRACE: "WriteToolSummary();",
    TOOL_INSCOUNT: "WriteToolSummary();",
    TOOL_MEMTRACE: "WriteToolSummary();",
}

_DR_CMAKE = """# Toothpick DBI DynamoRIO client build config (tool=@TOOL@).
#
#   cmake -B build -DDynamoRIO_DIR=<DynamoRIO>/cmake
#   cmake --build build --config Release
#
# The resulting drclient_@TOOL@.dll is loaded with:
#   <DynamoRIO>/bin64/drrun.exe -c drclient_@TOOL@.dll -o trace_@TOOL@.log -- target.exe

cmake_minimum_required(VERSION 3.10)
project(drclient_@TOOL@ CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (NOT DEFINED DynamoRIO_DIR)
    message(FATAL_ERROR "pass -DDynamoRIO_DIR=<DynamoRIO>/cmake")
endif()

find_package(DynamoRIO REQUIRED)

add_library(drclient_@TOOL@ SHARED drclient_@TOOL@.cpp)
configure_DynamoRIO_client(drclient_@TOOL@)
use_DynamoRIO_extension(drclient_@TOOL@ drmgr)
@EXTENSIONS@
set_target_properties(drclient_@TOOL@ PROPERTIES PREFIX "")
"""

_DR_CMAKE_EXTENSIONS = {
    TOOL_BBTRACE: "",
    TOOL_INSCOUNT: "",
    TOOL_MEMTRACE: (
        "use_DynamoRIO_extension(drclient_@TOOL@ drreg)\n"
        "use_DynamoRIO_extension(drclient_@TOOL@ drutil)"
    ),
}


def _render(template: str, tool_type: str, extra: Optional[Dict[str, str]] = None) -> str:
    """Single-pass ``@TOKEN@`` substitution (keeps C++ braces untouched)."""
    fields: Dict[str, str] = {
        "@TOOL@": tool_type,
        "@FORMAT@": DBI_LOG_FORMAT_VERSION,
    }
    if extra:
        fields.update(extra)
    rendered = template
    for key, value in fields.items():
        rendered = rendered.replace(key, value)
    return rendered


def _pintool_source(tool_type: str) -> str:
    records = _PIN_RECORDS_DOC[tool_type]
    prologue = _render(_PIN_PROLOGUE, tool_type, {"@RECORDS@": records})
    hooks = _PIN_HOOKS[tool_type]
    epilogue = _render(_PIN_EPILOGUE, tool_type)
    return f"{prologue}\n{hooks}\n{epilogue}"


def _dynamorio_source(tool_type: str) -> str:
    records = _DR_RECORDS_DOC[tool_type]
    prologue = _render(
        _DR_PROLOGUE,
        tool_type,
        {"@RECORDS@": records, "@INCLUDES@": _DR_INCLUDES[tool_type]},
    )
    body = _DR_INSERT[tool_type] + "\n" + _DR_TOOL_SUMMARY[tool_type] + "\n"
    epilogue = _render(
        _DR_EPILOGUE,
        tool_type,
        {
            "@TOOL_SUMMARY@": "",
            "@SUMMARY_CALL@": _DR_SUMMARY_CALL[tool_type],
            "@EXIT_EXTRA@": _DR_EXIT_EXTRA[tool_type],
            "@INIT_EXTRA@": _DR_INIT[tool_type],
        },
    )
    return f"{prologue}\n{body}\n{epilogue}"


class DbiToolGenerator:
    """Writes buildable Intel PIN / DynamoRIO instrumentation projects.

    Both generators are pure text emitters: no Pin, DynamoRIO, compiler or
    network access is required, so tool scaffolding is produced identically on
    any host (Risk R5).  Overwriting an existing project directory is
    intentional (regeneration is idempotent).
    """

    #: Tool types with a generator template.
    TOOL_TYPES = SUPPORTED_TOOL_TYPES

    @staticmethod
    def _out_dir(out_dir: PathLike) -> Path:
        target = _as_path(out_dir)
        if target.exists() and not target.is_dir():
            raise DbiToolGenerationError(f"out_dir exists and is not a directory: {target}")
        try:
            target.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise DbiToolGenerationError(f"cannot create out_dir {target}: {exc}") from exc
        return target

    @staticmethod
    def _validate_tool_type(tool_type: str) -> str:
        normalized = str(tool_type).strip().lower()
        if normalized not in SUPPORTED_TOOL_TYPES:
            supported = ", ".join(SUPPORTED_TOOL_TYPES)
            raise DbiToolGenerationError(
                f"unsupported DBI tool type {tool_type!r} (expected one of: {supported})"
            )
        return normalized

    @staticmethod
    def _write(target: Path, content: str) -> Path:
        try:
            target.write_text(content, encoding="utf-8", newline="\n")
        except OSError as exc:
            raise DbiToolGenerationError(f"cannot write {target}: {exc}") from exc
        return target

    @classmethod
    def generate_pintool(cls, tool_type: str, out_dir: PathLike) -> Path:
        """Generate an Intel PIN tool project; return the C++ source path.

        Writes ``pintool_<tool_type>.cpp`` plus the canonical Pin build config
        (``Makefile`` and ``makefile.rules``).
        """
        tool = cls._validate_tool_type(tool_type)
        directory = cls._out_dir(out_dir)
        source = directory / f"pintool_{tool}.cpp"
        cls._write(source, _pintool_source(tool))
        cls._write(directory / "Makefile", _render(_PIN_MAKEFILE, tool))
        cls._write(directory / "makefile.rules", _render(_PIN_MAKEFILE_RULES, tool))
        return source

    @classmethod
    def generate_dynamorio_client(cls, tool_type: str, out_dir: PathLike) -> Path:
        """Generate a DynamoRIO client project; return the C++ source path.

        Writes ``drclient_<tool_type>.cpp`` plus ``CMakeLists.txt`` wiring the
        ``drmgr`` (and, for ``memtrace``, ``drreg``/``drutil``) extensions.
        """
        tool = cls._validate_tool_type(tool_type)
        directory = cls._out_dir(out_dir)
        source = directory / f"drclient_{tool}.cpp"
        cls._write(source, _dynamorio_source(tool))
        cls._write(
            directory / "CMakeLists.txt",
            _render(_DR_CMAKE, tool, {"@EXTENSIONS@": _DR_CMAKE_EXTENSIONS[tool]}),
        )
        return source


# ---------------------------------------------------------------------------
# Chunk 3 -- Trace parsers (parsers only; Unicorn engine is stubbed below)
# ---------------------------------------------------------------------------

#: ``[tid=1234]`` / ``pin:`` / ``dr:`` prefixes emitted by engine loggers.
_RE_LOG_PREFIX = re.compile(
    r"^(?:\[(?:tid|thread|thr|pid|time)[^\]]*\]|(?:pin|pintool|dynamorio|dr|tdbi)\s*[:>]\s*)",
    re.IGNORECASE,
)
#: ``key=value`` metadata inside the banner / summary lines.
_RE_KEY_VALUE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(\S+)")
#: ``sha256: <64 hex>`` banner alternative.
_RE_SHA_LINE = re.compile(
    r"^(?:target_sha256|sha256|sha)\s*[:=]\s*([0-9a-fA-F]{64})$", re.IGNORECASE
)
#: DynamoRIO ``drcov`` style banners that carry no per-access detail.
_RE_DRCOV_MODULE = re.compile(r"^module\s*\[[^\]]*\]", re.IGNORECASE)
_RE_DRCOV_BB_TABLE = re.compile(r"^bb\s+table\s*[:=]", re.IGNORECASE)
_RE_BANNER = re.compile(r"toothpick-dbi", re.IGNORECASE)

_INSTRUCTION_TAGS = frozenset({"I", "INS", "INSTRUCTION"})
_BLOCK_TAGS = frozenset({"B", "BB", "BLOCK"})
_MEMORY_TAGS = frozenset({"M", "MEM", "MEMORY"})
_SUMMARY_TAGS = frozenset({"E", "END", "SUMMARY"})
_COUNT_TAGS = frozenset({"COUNT", "INSCOUNT"})
_SHA_TAGS = frozenset({"SHA256", "TARGET_SHA256", "SHA"})

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


@dataclass
class _ParsedTrace:
    """Working set of one parsed log (not part of the public schema)."""

    instructions: List[InstructionTraceEntry] = field(default_factory=list)
    executed_bbs: List[int] = field(default_factory=list)
    memory_accesses: List[MemoryTraceEntry] = field(default_factory=list)
    target_sha256: str = ""
    declared_instructions: Optional[int] = None
    bb_insn_total: int = 0
    elapsed_sec: float = 0.0
    saw_anything: bool = False
    unknown_lines: int = 0

    def add_block(self, address: int, insn_count: int) -> None:
        if address not in self.executed_bbs:
            self.executed_bbs.append(address)
        self.bb_insn_total += max(0, insn_count)

    def add_instructions(self, count: int) -> None:
        observed = max(0, count)
        if self.declared_instructions is None or observed > self.declared_instructions:
            self.declared_instructions = observed


def _parse_instruction_line(parts: List[str], parsed: _ParsedTrace) -> None:
    if len(parts) < 4:
        raise DbiParseError(
            "malformed instruction record (need: tag address size mnemonic [operands])"
        )
    parsed.instructions.append(
        InstructionTraceEntry(
            address=_parse_address(parts[1]),
            size=_parse_size(parts[2]),
            mnemonic=parts[3],
            operands=" ".join(parts[4:]),
        )
    )
    parsed.add_instructions(len(parsed.instructions))


def _parse_block_line(parts: List[str], parsed: _ParsedTrace) -> None:
    if len(parts) < 2:
        raise DbiParseError("malformed basic block record (need: tag address [size [count]])")
    address = _parse_address(parts[1])
    insn_count = 0
    if len(parts) > 2:
        size = _try_parse_size(parts[2])
        if size is None:
            raise DbiParseError(f"unparsable basic block size field {parts[2]!r}")
    if len(parts) > 3:
        parsed_count = _try_parse_size(parts[3])
        if parsed_count is not None:
            insn_count = parsed_count
    parsed.add_block(address, insn_count)
    if insn_count:
        parsed.add_instructions(insn_count)


def _parse_memory_line(parts: List[str], parsed: _ParsedTrace) -> None:
    if len(parts) < 5:
        raise DbiParseError(
            "malformed memory record (need: tag insn_addr R|W mem_addr size [value])"
        )
    size = _parse_size(parts[4])
    value = decode_memory_value(parts[5], size) if len(parts) > 5 else b""
    parsed.memory_accesses.append(
        MemoryTraceEntry(
            address=_parse_address(parts[1]),
            kind=normalize_memory_kind(parts[2]),
            mem_addr=_parse_address(parts[3]),
            size=size,
            value=value,
        )
    )


def _parse_summary_line(line: str, parsed: _ParsedTrace) -> None:
    for key, value in _RE_KEY_VALUE.findall(line):
        lowered = key.lower()
        if lowered in ("instructions", "total_instructions", "count", "inscount"):
            parsed.add_instructions(_parse_size(value))
        elif lowered in ("bbs", "blocks"):
            # A block count without addresses cannot reconstruct executed_bbs.
            continue
        elif lowered in ("elapsed", "elapsed_sec", "seconds"):
            parsed.elapsed_sec = _parse_float(value)
        elif lowered in ("sha256", "target_sha256"):
            if _SHA256_RE.match(value):
                parsed.target_sha256 = value.lower()


def _parse_count_line(tag: str, parts: List[str], parsed: _ParsedTrace) -> None:
    if tag == "INSCOUNT" and len(parts) > 1:
        parsed.add_instructions(_parse_size(parts[1]))
        return
    if len(parts) < 2:
        raise DbiParseError("malformed Count record (need: Count <n>)")
    parsed.add_instructions(_parse_size(parts[-1]))


def _parse_trace_log(log_path: PathLike, engine: str) -> _ParsedTrace:
    path = _as_path(log_path)
    if not path.exists():
        raise DbiParseError(f"trace log not found: {path}")
    if not path.is_dir() and path.is_file():
        pass
    elif path.is_dir():
        raise DbiParseError(f"trace log is a directory, not a file: {path}")
    try:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError as exc:
        raise DbiParseError(f"cannot read trace log {path}: {exc}") from exc

    parsed = _ParsedTrace()
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = _RE_LOG_PREFIX.sub("", raw.strip())
        if not line:
            continue
        if line.startswith("#"):
            if _RE_BANNER.search(line):
                parsed.saw_anything = True
                for key, value in _RE_KEY_VALUE.findall(line):
                    lowered = key.lower()
                    if lowered in ("sha256", "target_sha256") and _SHA256_RE.match(value):
                        parsed.target_sha256 = value.lower()
                    elif lowered in ("elapsed", "elapsed_sec", "seconds"):
                        try:
                            parsed.elapsed_sec = _parse_float(value)
                        except DbiParseError:
                            pass
            continue
        try:
            parts = line.split()
            tag = parts[0].upper().rstrip(":=")
            if tag in _INSTRUCTION_TAGS:
                _parse_instruction_line(parts, parsed)
                parsed.saw_anything = True
            elif tag in _BLOCK_TAGS:
                _parse_block_line(parts, parsed)
                parsed.saw_anything = True
            elif tag in _MEMORY_TAGS:
                _parse_memory_line(parts, parsed)
                parsed.saw_anything = True
            elif tag in _SUMMARY_TAGS:
                _parse_summary_line(line, parsed)
                parsed.saw_anything = True
            elif tag in _COUNT_TAGS:
                _parse_count_line(tag, parts, parsed)
                parsed.saw_anything = True
            elif tag in _SHA_TAGS:
                candidate = parts[-1]
                if _SHA256_RE.match(candidate):
                    parsed.target_sha256 = candidate.lower()
                parsed.saw_anything = True
            elif _RE_DRCOV_MODULE.match(line) or _RE_DRCOV_BB_TABLE.match(line):
                parsed.saw_anything = True
            else:
                match = _RE_SHA_LINE.match(line)
                if match:
                    parsed.target_sha256 = match.group(1).lower()
                    parsed.saw_anything = True
                else:
                    parsed.unknown_lines += 1
        except DbiParseError as exc:
            raise DbiParseError(f"{path}:{lineno}: {exc}") from None

    if not parsed.saw_anything:
        raise DbiParseError(f"no {DBI_LOG_FORMAT_VERSION} trace records found in {path}")
    return parsed


def _result_from(parsed: _ParsedTrace, engine: str) -> DbiTraceResult:
    observed = max(len(parsed.instructions), parsed.bb_insn_total)
    declared = parsed.declared_instructions
    if declared is None or declared < observed:
        total = observed
    else:
        total = declared
    return DbiTraceResult(
        engine=engine,
        target_sha256=parsed.target_sha256,
        total_instructions=total,
        executed_bbs=list(parsed.executed_bbs),
        memory_accesses=list(parsed.memory_accesses),
        elapsed_sec=parsed.elapsed_sec,
    )


class TraceParser:
    """Parsers for the ``tdbi/1`` text traces emitted by the generated tools.

    ``parse_pin_log`` / ``parse_dynamorio_log`` return the plan's
    :class:`DbiTraceResult`.  Because the plan's result schema carries executed
    basic blocks and memory accesses but no per-instruction list, the parsed
    :class:`InstructionTraceEntry` records are exposed through
    ``parse_pin_instructions`` / ``parse_dynamorio_instructions``.

    The ``engine`` of the returned result is the parser the caller used; the
    log banner's ``engine=`` token is informational only.
    """

    @staticmethod
    def parse_pin_log(log_path: PathLike) -> DbiTraceResult:
        """Parse an Intel PIN trace log into a :class:`DbiTraceResult`."""
        return _result_from(_parse_trace_log(log_path, ENGINE_PIN), ENGINE_PIN)

    @staticmethod
    def parse_dynamorio_log(log_path: PathLike) -> DbiTraceResult:
        """Parse a DynamoRIO client trace log into a :class:`DbiTraceResult`."""
        return _result_from(_parse_trace_log(log_path, ENGINE_DYNAMORIO), ENGINE_DYNAMORIO)

    @staticmethod
    def parse_pin_instructions(log_path: PathLike) -> List[InstructionTraceEntry]:
        """Return the per-instruction records of an Intel PIN trace log."""
        return list(_parse_trace_log(log_path, ENGINE_PIN).instructions)

    @staticmethod
    def parse_dynamorio_instructions(log_path: PathLike) -> List[InstructionTraceEntry]:
        """Return the per-instruction records of a DynamoRIO trace log."""
        return list(_parse_trace_log(log_path, ENGINE_DYNAMORIO).instructions)


# ---------------------------------------------------------------------------
# Chunk 3 (runner) -- Unicorn offline emulator
# ---------------------------------------------------------------------------

#: Page size for every Unicorn mapping.
_PAGE_SIZE = 0x1000
#: Load address used when a trace target is raw code bytes rather than a file.
_FLAT_CODE_BASE = 0x400000
#: Size of the synthetic stack mapped for each emulated run.
_STACK_SIZE = 0x10000
#: Default wall-clock guard for one emulated run, in seconds.
_TRACE_TIME_BUDGET_SEC = 10.0
#: Ceiling on how many target bytes one run maps into the emulator.
_MAX_TRACE_IMAGE_BYTES = 32 * 1024 * 1024
#: Ceiling on recorded memory accesses per run (docs/traps.md Trap 6, Risk R7).
_MAX_TRACE_MEMORY_ACCESSES = 100_000
#: Ceiling on per-instruction records kept on the tracer instance.
_MAX_TRACE_INSTRUCTION_RECORDS = 200_000
#: Bytes handed to Capstone per decode attempt.
_MAX_DECODE_BYTES = 16

#: Bytes that park mapped memory the target must never execute: ``hlt`` on x86,
#: ``b .`` (spin) on ARM.  Reaching a parked address ends the run cleanly.
_PARKING_CODE = {
    "x86_64": b"\xf4",
    "x86": b"\xf4",
    "arm64": b"\x00\x00\x00\x14",
    "arm": b"\xfe\xff\xff\xea",
}

#: ``UnicornDbiTracer.last_stop_reason`` values (one run, one reason).
STOP_COMPLETED = "completed"
STOP_RETURN_SENTINEL = "returned_to_sentinel"
STOP_FELL_OFF_CODE = "fell_off_code"
STOP_INSTRUCTION_BUDGET = "instruction_budget"
STOP_TIMEOUT = "timeout"
STOP_UNMAPPED_MEMORY = "unmapped_memory_access"
STOP_PROTECTED_MEMORY = "protected_memory_access"
STOP_INVALID_INSTRUCTION = "invalid_instruction"
STOP_CPU_EXCEPTION = "cpu_exception"


def _align_up(value: int, alignment: int) -> int:
    """Round ``value`` up to the next multiple of ``alignment``."""
    return (value + alignment - 1) // alignment * alignment


def _region_contains(regions: Sequence[Tuple[int, bytes]], address: int) -> bool:
    """True when ``address`` falls inside one of the sparse code regions."""
    return any(base <= address < base + len(data) for base, data in regions)


def _region_summary(regions: Sequence[Tuple[int, bytes]], limit: int = 4) -> str:
    """Compact ``0xbase+0xsize`` listing used in error messages."""
    parts = [f"0x{base:x}+0x{len(data):x}" for base, data in regions[:limit]]
    if len(regions) > limit:
        parts.append(f"...(+{len(regions) - limit} more)")
    return ", ".join(parts) or "<empty>"


def _encode_hook_value(value: int, size: int) -> bytes:
    """Render a Unicorn write-hook value as little-endian bytes."""
    if size <= 0 or value < 0:
        return b""
    try:
        return value.to_bytes(size, "little")
    except OverflowError:
        return b""


def _merge_pages(
    regions: Sequence[Tuple[int, bytes]], fill: bytes
) -> List[Tuple[int, bytes]]:
    """Project sparse regions onto page buffers and coalesce adjacent pages.

    ``fill`` (parking code) pads whatever a region does not cover, so a
    fall-through past the real code never decodes uninitialised memory.
    """
    if not fill:
        raise DbiError("parking-code fill must not be empty")
    pages: Dict[int, bytearray] = {}
    for base, data in regions:
        index = 0
        while index < len(data):
            position = base + index
            page_base = position // _PAGE_SIZE * _PAGE_SIZE
            offset = position - page_base
            take = min(_PAGE_SIZE - offset, len(data) - index)
            page = pages.get(page_base)
            if page is None:
                page = bytearray((fill * (_PAGE_SIZE // len(fill) + 1))[:_PAGE_SIZE])
                pages[page_base] = page
            page[offset : offset + take] = data[index : index + take]
            index += take
    merged: List[Tuple[int, bytes]] = []
    for page_base in sorted(pages):
        blob = bytes(pages[page_base])
        if merged and merged[-1][0] + len(merged[-1][1]) == page_base:
            merged[-1] = (merged[-1][0], merged[-1][1] + blob)
        else:
            merged.append((page_base, blob))
    return merged


def _import_emulation_engines() -> Tuple[Any, Any]:
    """Lazily import Unicorn and Capstone.

    The imports live *inside* the tracing path so that ``import toothpick.dbi``
    stays free of native dependencies (module contract kept from chunk 1).
    """
    try:
        import unicorn
    except ImportError as exc:  # pragma: no cover - host installation dependent
        raise DbiError(
            "Unicorn is required for emulated tracing; install the 'unicorn' "
            "package (declared in pyproject.toml)"
        ) from exc
    try:
        import capstone
    except ImportError as exc:  # pragma: no cover - host installation dependent
        raise DbiError(
            "Capstone is required for emulated tracing; install the 'capstone' "
            "package (declared in pyproject.toml)"
        ) from exc
    return unicorn, capstone


@dataclass(frozen=True)
class _EngineConfig:
    """Unicorn/Capstone parameters for one supported instruction set."""

    arch: str
    uc_arch: int
    uc_mode: int
    cs_arch: int
    cs_mode: int
    pointer_size: int
    sp_reg: int
    pc_reg: int
    parking_code: bytes
    lr_reg: Optional[int] = None


def _engine_config(unicorn: Any, capstone: Any, arch: Optional[str]) -> _EngineConfig:
    """Resolve ``arch`` into Unicorn/Capstone constants (lazy const imports)."""
    name = str(arch or "x86_64").strip().lower().replace("-", "_")
    if name in ("x86_64", "amd64", "x64", "intel64"):
        import unicorn.x86_const as x86_const

        return _EngineConfig(
            arch="x86_64",
            uc_arch=unicorn.UC_ARCH_X86,
            uc_mode=unicorn.UC_MODE_64,
            cs_arch=capstone.CS_ARCH_X86,
            cs_mode=capstone.CS_MODE_64,
            pointer_size=8,
            sp_reg=x86_const.UC_X86_REG_RSP,
            pc_reg=x86_const.UC_X86_REG_RIP,
            parking_code=_PARKING_CODE["x86_64"],
        )
    if name in ("x86", "x86_32", "i386", "i486", "i586", "i686"):
        import unicorn.x86_const as x86_const

        return _EngineConfig(
            arch="x86",
            uc_arch=unicorn.UC_ARCH_X86,
            uc_mode=unicorn.UC_MODE_32,
            cs_arch=capstone.CS_ARCH_X86,
            cs_mode=capstone.CS_MODE_32,
            pointer_size=4,
            sp_reg=x86_const.UC_X86_REG_ESP,
            pc_reg=x86_const.UC_X86_REG_EIP,
            parking_code=_PARKING_CODE["x86"],
        )
    if name in ("arm64", "aarch64"):
        import unicorn.arm64_const as arm64_const

        return _EngineConfig(
            arch="arm64",
            uc_arch=unicorn.UC_ARCH_ARM64,
            uc_mode=unicorn.UC_MODE_ARM,
            cs_arch=capstone.CS_ARCH_ARM64,
            cs_mode=capstone.CS_MODE_ARM,
            pointer_size=8,
            sp_reg=arm64_const.UC_ARM64_REG_SP,
            pc_reg=arm64_const.UC_ARM64_REG_PC,
            parking_code=_PARKING_CODE["arm64"],
            lr_reg=arm64_const.UC_ARM64_REG_LR,
        )
    if name in ("arm", "arm32", "aarch32"):
        import unicorn.arm_const as arm_const

        return _EngineConfig(
            arch="arm",
            uc_arch=unicorn.UC_ARCH_ARM,
            uc_mode=unicorn.UC_MODE_ARM,
            cs_arch=capstone.CS_ARCH_ARM,
            cs_mode=capstone.CS_MODE_ARM,
            pointer_size=4,
            sp_reg=arm_const.UC_ARM_REG_SP,
            pc_reg=arm_const.UC_ARM_REG_PC,
            parking_code=_PARKING_CODE["arm"],
            lr_reg=arm_const.UC_ARM_REG_LR,
        )
    raise DbiError(
        f"unsupported emulation architecture {arch!r}; supported values are "
        "'x86_64', 'x86', 'arm64' and 'arm'"
    )


@dataclass(frozen=True)
class _DecodedInsn:
    """Capstone facts about one instruction that a trace run needs."""

    size: int
    mnemonic: str
    operands: str
    terminates_block: bool
    target: Optional[int] = None


class _InsnDecoder:
    """Lazy Capstone decoder over the code regions of a single trace run.

    Only addresses the emulator actually executes are decoded, so a large image
    does not bound tracing cost; results are memoised per address.
    """

    def __init__(
        self,
        capstone: Any,
        config: _EngineConfig,
        regions: Sequence[Tuple[int, bytes]],
    ) -> None:
        self._capstone = capstone
        self._regions = list(regions)
        self._disassembler = capstone.Cs(config.cs_arch, config.cs_mode)
        self._disassembler.detail = True
        self._cache: Dict[int, Optional[_DecodedInsn]] = {}
        self._terminal_groups: Set[int] = {
            int(getattr(capstone, group_name))
            for group_name in ("CS_GRP_JUMP", "CS_GRP_CALL", "CS_GRP_RET", "CS_GRP_IRET")
            if hasattr(capstone, group_name)
        }
        self._immediate_operand = getattr(capstone, "CS_OP_IMM", None)

    def decode(self, address: int) -> Optional[_DecodedInsn]:
        """Return the decoded instruction at ``address`` (``None`` if unknown)."""
        if address not in self._cache:
            self._cache[address] = self._decode_uncached(address)
        return self._cache[address]

    def _slice(self, address: int) -> bytes:
        for base, data in self._regions:
            if base <= address < base + len(data):
                offset = address - base
                return data[offset : offset + _MAX_DECODE_BYTES]
        return b""

    def _decode_uncached(self, address: int) -> Optional[_DecodedInsn]:
        blob = self._slice(address)
        if not blob:
            return None
        try:
            insn = next(iter(self._disassembler.disasm(blob, address, count=1)), None)
        except Exception:  # pragma: no cover - capstone surface is defensive
            return None
        if insn is None:
            return None
        terminates = False
        target: Optional[int] = None
        try:
            groups = {int(group) for group in insn.groups}
            terminates = bool(groups & self._terminal_groups)
            if self._immediate_operand is not None:
                for operand in insn.operands:
                    if operand.type == self._immediate_operand:
                        target = int(operand.imm)
                        break
        except Exception:  # pragma: no cover - detail data is best effort
            pass
        return _DecodedInsn(
            size=int(insn.size),
            mnemonic=str(insn.mnemonic or ""),
            operands=str(insn.op_str or ""),
            terminates_block=terminates,
            target=target,
        )

@dataclass
class _TraceImage:
    """Sparse memory image (region base -> bytes) that a trace run emulates."""

    regions: List[Tuple[int, bytes]]
    entry: int
    sha256: str
    arch: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


def _detect_arch(parsed: Any) -> Optional[str]:
    """Map a ``lief`` machine type onto one of the supported ISA names."""
    try:
        machine = str(getattr(getattr(parsed, "header", None), "machine", ""))
    except Exception:  # pragma: no cover - lief header surface is defensive
        return None
    token = machine.upper()
    if "AMD64" in token or "X86_64" in token or "X64" in token:
        return "x86_64"
    if "I386" in token or "I686" in token:
        return "x86"
    if "AARCH64" in token or "ARM64" in token:
        return "arm64"
    if "ARM" in token:
        return "arm"
    return None


def _image_from_executable(data: bytes) -> Optional[_TraceImage]:
    """Build a sparse image from a PE/ELF/Mach-O blob via ``lief``.

    Returns ``None`` when ``lief`` is unavailable or does not recognise the
    bytes, in which case the caller falls back to a flat code blob.
    """
    try:
        import lief
    except ImportError:  # pragma: no cover - host installation dependent
        return None
    try:
        parsed = lief.parse(data)
    except Exception:
        return None
    if parsed is None:
        return None
    try:
        sections = list(parsed.sections)
        entry = int(parsed.entrypoint)
    except Exception:
        return None
    if not sections:
        return None
    # PE section virtual addresses are RVAs and need the image base; ELF and
    # Mach-O sections are already expressed as virtual addresses.
    pe_like = hasattr(parsed, "optional_header")
    try:
        image_base = int(getattr(parsed, "imagebase", 0) or 0) if pe_like else 0
    except Exception:
        image_base = 0
    regions: List[Tuple[int, bytes]] = []
    warnings: List[str] = []
    mapped = 0
    for section in sections:
        try:
            content = bytes(section.content)
            section_va = int(section.virtual_address)
        except Exception:
            continue
        if not content:
            continue
        if mapped + len(content) > _MAX_TRACE_IMAGE_BYTES:
            warnings.append(
                "skipped a section past the "
                f"{_MAX_TRACE_IMAGE_BYTES}-byte emulation mapping cap"
            )
            break
        regions.append((section_va + image_base, content))
        mapped += len(content)
    if not regions:
        return None
    return _TraceImage(
        regions=regions,
        entry=entry,
        sha256=hashlib.sha256(data).hexdigest(),
        arch=_detect_arch(parsed),
        warnings=warnings,
    )


def _load_trace_image(binary_path: Any) -> _TraceImage:
    """Accept raw code bytes or a file path and describe the image to emulate."""
    if isinstance(binary_path, (bytes, bytearray, memoryview)):
        data = bytes(binary_path)
        if not data:
            raise DbiError("cannot trace an empty in-memory code blob")
        return _TraceImage(
            regions=[(_FLAT_CODE_BASE, data)],
            entry=_FLAT_CODE_BASE,
            sha256=hashlib.sha256(data).hexdigest(),
        )
    path = _as_path(binary_path)
    if path.exists() and path.is_dir():
        raise DbiError(f"trace target is a directory, not a file: {path}")
    if not path.exists():
        raise DbiError(f"trace target not found: {path}")
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise DbiError(f"cannot read trace target {path}: {exc}") from exc
    if not data:
        raise DbiError(f"trace target is empty: {path}")
    image = _image_from_executable(data)
    if image is not None:
        return image
    return _TraceImage(
        regions=[(_FLAT_CODE_BASE, data)],
        entry=_FLAT_CODE_BASE,
        sha256=hashlib.sha256(data).hexdigest(),
        warnings=[
            f"{path} is not a recognised executable image; traced as a flat "
            f"code blob at 0x{_FLAT_CODE_BASE:x}"
        ],
    )


def _classify_uc_error(unicorn: Any, exc: Exception) -> str:
    """Translate a :class:`unicorn.UcError` into a ``STOP_*`` reason."""
    errno_value = getattr(exc, "errno", None)
    if errno_value is not None:
        for constant, reason in (
            ("UC_ERR_READ_UNMAPPED", STOP_UNMAPPED_MEMORY),
            ("UC_ERR_WRITE_UNMAPPED", STOP_UNMAPPED_MEMORY),
            ("UC_ERR_FETCH_UNMAPPED", STOP_UNMAPPED_MEMORY),
            ("UC_ERR_READ_PROT", STOP_PROTECTED_MEMORY),
            ("UC_ERR_WRITE_PROT", STOP_PROTECTED_MEMORY),
            ("UC_ERR_FETCH_PROT", STOP_PROTECTED_MEMORY),
            ("UC_ERR_INSN_INVALID", STOP_INVALID_INSTRUCTION),
            ("UC_ERR_EXCEPTION", STOP_CPU_EXCEPTION),
        ):
            if getattr(unicorn, constant, None) == errno_value:
                return reason
    return f"emulation_error:{errno_value}"


class _UnicornRunState:
    """Bookkeeping shared by the Unicorn hooks of a single trace run."""

    def __init__(
        self,
        budget: int,
        deadline: float,
        decoder: _InsnDecoder,
        regions: Sequence[Tuple[int, bytes]],
        sentinel_span: Tuple[int, int],
        mem_read: int,
        mem_write: int,
    ) -> None:
        self.budget = budget
        self.deadline = deadline
        self.decoder = decoder
        self.regions = list(regions)
        self.sentinel_start, self.sentinel_end = sentinel_span
        self.mem_read = mem_read
        self.mem_write = mem_write
        self.total_instructions = 0
        self.block_starts: List[int] = []
        self.accesses: List[MemoryTraceEntry] = []
        self.instructions: List[InstructionTraceEntry] = []
        self.leaders: Set[int] = set()
        self.current_insn = 0
        self.stop_reason = STOP_COMPLETED
        self.stop_detail = ""
        self.budget_exhausted = False
        self.timed_out = False
        self.warnings: List[str] = []
        self._blocks_seen: Set[int] = set()
        self._accesses_truncated = False
        self._instructions_truncated = False
        self._last_address: Optional[int] = None
        self._last_size = 0
        self._last_terminates = False

    def on_code(self, uc: Any, address: int, size: int, _user_data: Any = None) -> None:
        """``UC_HOOK_CODE``: budget/timeout guards plus execution recording."""
        if self.sentinel_start <= address < self.sentinel_end:
            self._stop(uc, STOP_RETURN_SENTINEL, f"0x{address:x}")
            return
        if not _region_contains(self.regions, address):
            self._stop(uc, STOP_FELL_OFF_CODE, f"0x{address:x}")
            return
        if self.total_instructions >= self.budget:
            self.budget_exhausted = True
            self._stop(uc, STOP_INSTRUCTION_BUDGET, f"{self.budget} instructions")
            return
        if time.monotonic() >= self.deadline:
            self.timed_out = True
            self._stop(uc, STOP_TIMEOUT)
            return
        info = self.decoder.decode(address)
        insn_size = info.size if info is not None else int(size)
        fallthrough = (
            self._last_address is not None
            and address == self._last_address + self._last_size
            and not self._last_terminates
        )
        if not fallthrough or address in self.leaders:
            self._begin_block(address)
        self.total_instructions += 1
        self._record_instruction(address, insn_size, info)
        if info is not None:
            if info.terminates_block:
                self.leaders.add(address + info.size)
            if info.target is not None:
                self.leaders.add(info.target)
        self.current_insn = address
        self._last_address = address
        self._last_size = insn_size
        self._last_terminates = info is not None and info.terminates_block

    def on_memory(
        self,
        uc: Any,
        access: int,
        address: int,
        size: int,
        value: int,
        _user_data: Any = None,
    ) -> None:
        """``UC_HOOK_MEM_READ|UC_HOOK_MEM_WRITE``: record one access."""
        if access == self.mem_read:
            kind = MEMORY_READ
        elif access == self.mem_write:
            kind = MEMORY_WRITE
        else:
            self.warnings.append(
                f"unclassified memory access {access!r} at 0x{address:x}"
            )
            return
        if len(self.accesses) >= _MAX_TRACE_MEMORY_ACCESSES:
            if not self._accesses_truncated:
                self._accesses_truncated = True
                self.warnings.append(
                    f"memory trace truncated at {_MAX_TRACE_MEMORY_ACCESSES} accesses"
                )
            return
        raw = _encode_hook_value(int(value), int(size))
        if kind == MEMORY_READ or not raw:
            try:
                raw = bytes(uc.mem_read(address, size))
            except Exception:
                raw = b""
        self.accesses.append(
            MemoryTraceEntry(
                address=self.current_insn,
                kind=kind,
                mem_addr=int(address),
                size=int(size),
                value=raw,
            )
        )

    def _stop(self, uc: Any, reason: str, detail: str = "") -> None:
        self.stop_reason = reason
        self.stop_detail = detail
        try:
            uc.emu_stop()
        except Exception:  # pragma: no cover - defensive, emu_stop returns None
            pass

    def _begin_block(self, address: int) -> None:
        if address not in self._blocks_seen:
            self._blocks_seen.add(address)
            self.block_starts.append(address)

    def _record_instruction(
        self, address: int, size: int, info: Optional[_DecodedInsn]
    ) -> None:
        if len(self.instructions) >= _MAX_TRACE_INSTRUCTION_RECORDS:
            if not self._instructions_truncated:
                self._instructions_truncated = True
                self.warnings.append(
                    "per-instruction trace truncated at "
                    f"{_MAX_TRACE_INSTRUCTION_RECORDS} entries"
                )
            return
        self.instructions.append(
            InstructionTraceEntry(
                address=address,
                size=size,
                mnemonic=info.mnemonic if info is not None else "",
                operands=info.operands if info is not None else "",
            )
        )


class UnicornDbiTracer:
    """Offline Unicorn tracer: bounded emulation with code/memory hooks.

    :meth:`trace` emulates a raw machine-code blob (``bytes``) or an executable
    file under Unicorn and records every executed instruction (``UC_HOOK_CODE``)
    and every memory read/write (``UC_HOOK_MEM_READ`` / ``UC_HOOK_MEM_WRITE``)
    into a :class:`DbiTraceResult` with ``engine == "unicorn"``.

    Bounds (Risk R2 / R7).  A run stops at the first of:

    * the instruction budget, ``min(max_instructions, self.max_instructions)``;
    * the ``timeout_sec`` wall-clock guard, enforced both by Unicorn's own
      microsecond timeout and re-checked inside the code hook;
    * a return that lands on the synthetic sentinel page, so a stray ``ret``
      leaves the emulated code instead of walking the synthetic stack;
    * execution running past the last mapped code byte;
    * an unmapped/protected access, an undecodable instruction or a CPU
      exception, reported as a stop reason instead of a crash.

    Nothing runs on the host: each call builds a fresh Unicorn instance with its
    own code, stack and sentinel mappings, so the target can never touch the
    operator's process (Risk R2).

    Memory layout.  Raw ``bytes`` load at ``0x400000`` and default to
    ``x86_64`` unless ``arch=`` says otherwise.  File targets are mapped per
    section via ``lief``; files ``lief`` does not recognise are traced as a flat
    blob.  The stack is synthetic and pre-filled with the sentinel return
    address, so stack contents in the trace reflect that synthetic layout.

    ``last_stop_reason``, ``last_stop_detail``, ``last_stop_address``,
    ``last_instructions``, ``last_registers`` and ``last_warnings`` carry run
    detail that the :class:`DbiTraceResult` schema has no field for.
    """

    ENGINE = ENGINE_UNICORN

    def __init__(self, max_instructions: int = DEFAULT_MAX_INSTRUCTIONS) -> None:
        if int(max_instructions) <= 0:
            raise DbiError(f"max_instructions must be positive, got {max_instructions!r}")
        if int(max_instructions) > MAX_TRACE_INSTRUCTIONS:
            raise DbiError(
                f"max_instructions {max_instructions!r} exceeds the "
                f"{MAX_TRACE_INSTRUCTIONS} budget ceiling (docs/traps.md Trap 6)"
            )
        self.max_instructions = int(max_instructions)
        self.last_stop_reason: Optional[str] = None
        self.last_stop_detail: str = ""
        self.last_stop_address: Optional[int] = None
        self.last_instructions: List[InstructionTraceEntry] = []
        self.last_registers: Dict[str, int] = {}
        self.last_warnings: List[str] = []

    def trace(
        self,
        binary_path: Union[PathLike, bytes, bytearray, memoryview],
        max_instructions: int = DEFAULT_MAX_INSTRUCTIONS,
        *,
        arch: Optional[str] = None,
        entry: Optional[int] = None,
        timeout_sec: Optional[float] = _TRACE_TIME_BUDGET_SEC,
    ) -> DbiTraceResult:
        """Emulate ``binary_path`` under Unicorn and return its trace.

        Args:
            binary_path: raw machine-code ``bytes``, or a path to an executable.
            max_instructions: per-call instruction budget; the run uses the
                tighter of this and the tracer's own budget.
            arch: ``"x86_64"`` (the default for raw bytes), ``"x86"``,
                ``"arm64"`` or ``"arm"``; file targets auto-detect when possible.
            entry: entry-address override; defaults to the image entry point, or
                to ``0x400000`` for a raw code blob.
            timeout_sec: wall-clock guard for the whole run.

        Returns:
            DbiTraceResult: ``engine == "unicorn"`` with the executed
            instruction count, basic-block start addresses and memory accesses.

        Raises:
            DbiError: unreadable/empty target, unsupported architecture, invalid
                budget or timeout, an entry point outside the mapped code, or a
                run that executed no instruction at all.
        """
        budget = self._resolve_budget(max_instructions)
        wall_budget = float(
            _TRACE_TIME_BUDGET_SEC if timeout_sec is None else timeout_sec
        )
        if wall_budget <= 0:
            raise DbiError(f"timeout_sec must be positive, got {timeout_sec!r}")

        unicorn, capstone = _import_emulation_engines()
        image = _load_trace_image(binary_path)
        config = _engine_config(unicorn, capstone, arch or image.arch)
        regions = list(image.regions)
        start_address = int(image.entry if entry is None else entry)
        if not _region_contains(regions, start_address):
            raise DbiError(
                f"entry point 0x{start_address:x} is not inside any mapped code "
                f"region ({_region_summary(regions)})"
            )

        image_end = max(base + len(data) for base, data in regions)
        stack_base = _align_up(image_end + _PAGE_SIZE, _PAGE_SIZE)
        sentinel_base = stack_base + _STACK_SIZE + _PAGE_SIZE

        uc = unicorn.Uc(config.uc_arch, config.uc_mode)
        for page_base, blob in _merge_pages(regions, config.parking_code):
            uc.mem_map(page_base, len(blob))
            uc.mem_write(page_base, blob)
        uc.mem_map(stack_base, _STACK_SIZE)
        uc.mem_map(sentinel_base, _PAGE_SIZE)
        sentinel_word = sentinel_base.to_bytes(config.pointer_size, "little")
        uc.mem_write(
            stack_base, sentinel_word * (_STACK_SIZE // config.pointer_size)
        )
        uc.mem_write(
            sentinel_base,
            (config.parking_code * (_PAGE_SIZE // len(config.parking_code) + 1))[
                :_PAGE_SIZE
            ],
        )
        stack_pointer = stack_base + _STACK_SIZE // 2
        stack_pointer -= stack_pointer % 16
        uc.reg_write(config.sp_reg, stack_pointer)
        if config.lr_reg is not None:
            uc.reg_write(config.lr_reg, sentinel_base)

        state = _UnicornRunState(
            budget=budget,
            deadline=time.monotonic() + wall_budget,
            decoder=_InsnDecoder(capstone, config, regions),
            regions=regions,
            sentinel_span=(sentinel_base, sentinel_base + _PAGE_SIZE),
            mem_read=unicorn.UC_MEM_READ,
            mem_write=unicorn.UC_MEM_WRITE,
        )
        uc.hook_add(unicorn.UC_HOOK_CODE, state.on_code)
        uc.hook_add(
            unicorn.UC_HOOK_MEM_READ | unicorn.UC_HOOK_MEM_WRITE, state.on_memory
        )

        started = time.monotonic()
        try:
            uc.emu_start(
                start_address,
                0,
                timeout=int(wall_budget * 1_000_000),
                count=budget,
            )
        except unicorn.UcError as exc:
            state.stop_reason = _classify_uc_error(unicorn, exc)
            state.stop_detail = str(exc)
        elapsed_sec = time.monotonic() - started
        if state.stop_reason == STOP_COMPLETED and state.total_instructions >= budget:
            state.budget_exhausted = True
            state.stop_reason = STOP_INSTRUCTION_BUDGET
            state.stop_detail = f"{budget} instructions"

        registers = self._read_registers(uc, config)
        self.last_stop_reason = state.stop_reason
        self.last_stop_detail = state.stop_detail
        self.last_stop_address = registers.get("pc")
        self.last_instructions = list(state.instructions)
        self.last_registers = registers
        self.last_warnings = list(image.warnings) + list(state.warnings)
        if state.total_instructions == 0:
            detail = state.stop_reason
            if state.stop_detail:
                detail = f"{detail}: {state.stop_detail}"
            raise DbiError(
                f"emulation of {self._describe_target(binary_path)} executed no "
                f"instruction (stop reason: {detail})"
            )
        return DbiTraceResult(
            engine=ENGINE_UNICORN,
            target_sha256=image.sha256,
            total_instructions=state.total_instructions,
            executed_bbs=list(state.block_starts),
            memory_accesses=list(state.accesses),
            elapsed_sec=elapsed_sec,
        )

    def _resolve_budget(self, max_instructions: int) -> int:
        """Validate the per-call budget and clamp it against the instance one."""
        try:
            requested = int(max_instructions)
        except (TypeError, ValueError) as exc:
            raise DbiError(
                f"max_instructions must be an integer, got {max_instructions!r}"
            ) from exc
        if requested <= 0:
            raise DbiError(f"max_instructions must be positive, got {max_instructions!r}")
        if requested > MAX_TRACE_INSTRUCTIONS:
            raise DbiError(
                f"max_instructions {requested} exceeds the {MAX_TRACE_INSTRUCTIONS} "
                "budget ceiling (docs/traps.md Trap 6)"
            )
        return min(requested, self.max_instructions)

    @staticmethod
    def _read_register(uc: Any, register: Optional[int]) -> Optional[int]:
        if register is None:
            return None
        try:
            return int(uc.reg_read(register))
        except Exception:
            return None

    def _read_registers(self, uc: Any, config: _EngineConfig) -> Dict[str, int]:
        """Snapshot the stack and instruction pointers after the run."""
        registers: Dict[str, int] = {}
        for name, register in (("sp", config.sp_reg), ("pc", config.pc_reg)):
            value = self._read_register(uc, register)
            if value is not None:
                registers[name] = value
        return registers

    @staticmethod
    def _describe_target(binary_path: Any) -> str:
        if isinstance(binary_path, (bytes, bytearray, memoryview)):
            return f"{len(binary_path)}-byte in-memory code blob"
        return str(_as_path(binary_path))
