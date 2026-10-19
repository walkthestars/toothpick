"""Toothpick profile ingestion (P1, Chunk 1): sampled stacks -> per-function hotness.

Binding contract: ``docs/units/UNIT-P1-PROFILE-PLAN.md`` section 1 (unified
importer, attribution, hotness normalization) and section 2 (interfaces),
governed by ``docs/NEXT-TRACK.md`` P1 and ``docs/design-document.md`` sections 1
and 19 (receipts-or-refusal, never emit unobserved offsets).

Three sampled-stack text formats collapse into one row shape --
:class:`ProfileSample` -- and one table -- ``func_profile``, keyed
``(binary_sha256, func_addr)``.  The module owns four invariants:

* **Epistemic (never fabricate).**  A sample is attributed to an existing
  ``funcs`` row by exact address first, then by exact ``name`` match.  Anything
  else is counted ``unattributed``: this module never inserts a ``funcs`` row
  for a symbol the analysis has not observed, and unattributed samples are
  reported, never silently redistributed into hotness.
* **Strict CSV (no ``eval``).**  Columns are located by header name, every row
  is validated, and a malformed row increments ``rejected`` instead of raising,
  so one bad line in a million-row profile cannot lose the import.
* **Determinism.**  Parser output is sorted by address, aggregation happens on
  integer keys only, and list-returning queries are ordered
  ``hotness DESC, func_addr ASC`` (Risk R1).
* **Bounded responses.**  :func:`hottest` defaults to :data:`DEFAULT_TOP_N` (50)
  and clamps to :data:`MAX_TOP_N` (1000); an unbounded list is unreachable.

Windows traps (``docs/traps.md`` 4/7): every connection opened here is closed in
a ``finally`` (a leaked WAL handle surfaces as ``PermissionError: [WinError 32]``
on teardown), and nothing in this module installs ``faulthandler`` -- the suite
keeps ``-p no:faulthandler`` (``pyproject.toml`` ``addopts``).

Table creation: :func:`ensure_profile_table` is idempotent and callable with a
:class:`~toothpick.storage.Database`, a raw :class:`sqlite3.Connection`, or a db
path.  :func:`import_samples` ensures the table on its write path, so an
existing store needs no migration step; wiring ``func_profile`` into
``Database._SCHEMA`` (the other half of plan section 1.2) is a storage-module
change scheduled with the Chunk 2 CLI work.  ``create=True`` on a path that does
not exist yet also creates the minimal *empty* ``funcs`` table attribution reads
against, and a caller-supplied connection is never re-configured: its
``row_factory``/``isolation_level`` stay the caller's and result rows are read
positionally.

J2 formats (``docs/units/UNIT-J2-JFR-PLAN.md`` Chunk 2a): :class:`ProfileFormat`
gains ``ASYNC``/``JFR``, and :func:`parse_profile_text`/:func:`parse_profile_file`
route those -- and AUTO-detect async-profiler collapsed / ``jfr print --json``
input -- to :mod:`toothpick.profile_jfr` through a *function-local* import
(``profile_jfr`` imports this module's row shape at module scope, so a top-level
import here would close a cycle).  The parsing work itself lives in that module.
"""

from __future__ import annotations

import csv
import io
import re
import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from toothpick.storage import Database

__all__ = [
    "DEFAULT_TOP_N",
    "MAX_TOP_N",
    "FUNC_PROFILE_DDL",
    "FUNC_PROFILE_INDEX_DDL",
    "MINIMAL_FUNCS_DDL",
    "ProfileError",
    "ProfileFormat",
    "ProfileSample",
    "detect_format",
    "ensure_profile_table",
    "get_profile",
    "hottest",
    "import_samples",
    "parse_etw_csv",
    "parse_perf_csv",
    "parse_profile_file",
    "parse_profile_text",
    "parse_vtune_csv",
    "resolve_top_n",
]

#: Default top-K budget for :func:`hottest` (mirrors ``query.DEFAULT_LIMIT``).
DEFAULT_TOP_N = 50

#: Hard ceiling for :func:`hottest`: an explicit ``top_n`` is clamped to this
#: (Risk R1 -- never an unbounded list).
MAX_TOP_N = 1_000

#: ``func_profile`` DDL (plan section 1.2).  One row per observed function of
#: one binary; ``hotness`` is the normalized sample share of the last import.
FUNC_PROFILE_DDL = """
CREATE TABLE IF NOT EXISTS func_profile (
    binary_sha256 TEXT    NOT NULL,
    func_addr     INTEGER NOT NULL,
    sample_count  INTEGER NOT NULL DEFAULT 0,
    alloc_bytes   INTEGER NOT NULL DEFAULT 0,
    hotness       REAL    NOT NULL DEFAULT 0.0,
    updated_at    TEXT    NOT NULL DEFAULT '',
    PRIMARY KEY (binary_sha256, func_addr)
)
"""

#: Hotness read path is ``WHERE binary_sha256 = ? ORDER BY hotness DESC,
#: func_addr ASC``; the index matches that key order exactly.
FUNC_PROFILE_INDEX_DDL = (
    "CREATE INDEX IF NOT EXISTS idx_func_profile_hot "
    "ON func_profile (binary_sha256, hotness DESC, func_addr ASC)"
)

#: Minimal ``funcs`` DDL for the ``create=True`` bootstrap path of
#: :func:`import_samples`.  A brand-new store has no ``funcs`` table, and the
#: attribution query needs one to read; only the columns attribution touches are
#: declared.  The table is created *empty* -- it observes nothing, so no ``funcs``
#: row is ever fabricated (epistemic rule, plan section 1.2).
MINIMAL_FUNCS_DDL = """
CREATE TABLE IF NOT EXISTS funcs (
    binary_sha256 TEXT    NOT NULL,
    func_addr     INTEGER NOT NULL,
    name          TEXT    NOT NULL DEFAULT '',
    PRIMARY KEY (binary_sha256, func_addr)
)
"""

#: The ``(func_addr, name)`` pair read by attribution and by name decoration
#: (one source of truth so the two readers cannot drift).
_FUNCS_PAIR_SQL = "SELECT func_addr, name FROM funcs WHERE binary_sha256 = ?"

#: Hotness is answered from ``func_profile`` *alone* -- names are decoration
#: looked up afterwards, and only when a ``funcs`` table exists -- so a store
#: without ``funcs`` returns a bounded list instead of "no such table: funcs".
#: Order is the Risk R1 key order.
_HOTTEST_SQL = """
SELECT func_addr   AS func_addr,
       sample_count AS sample_count,
       alloc_bytes  AS alloc_bytes,
       hotness      AS hotness
FROM func_profile
WHERE binary_sha256 = ?
ORDER BY hotness DESC, func_addr ASC
LIMIT ?
"""

_GET_PROFILE_SQL = """
SELECT sample_count, alloc_bytes, hotness
FROM func_profile
WHERE binary_sha256 = ? AND func_addr = ?
"""

PathLike = Union[str, Path]
TargetLike = Union[Database, sqlite3.Connection, PathLike]


class ProfileError(Exception):
    """Raised for unreadable profiles, unknown formats, and bad arguments."""


class ProfileFormat(str, Enum):
    """Supported sampled-stack text formats (plan section 2).

    ``VTUNE``/``ETW``/``PERF`` are the P1 CSV/delimited formats.  ``ASYNC`` and
    ``JFR`` are the J2 family (async-profiler collapsed stacks and
    ``jfr print --json``), parsed by :mod:`toothpick.profile_jfr` and routed by
    :func:`parse_profile_text`.  ``AUTO`` stays last and stays the
    :meth:`coerce` default for ``None`` -- AUTO means "sniff", never "unknown".
    """

    VTUNE = "vtune"
    ETW = "etw"
    PERF = "perf"
    ASYNC = "async"
    JFR = "jfr"
    AUTO = "auto"

    @classmethod
    def coerce(cls, value: Union["ProfileFormat", str, None]) -> "ProfileFormat":
        if value is None:
            return cls.AUTO
        if isinstance(value, cls):
            return value
        text = str(value).strip().lower()
        for member in cls:
            if member.value == text:
                return member
        raise ProfileError(
            f"unknown profile format {value!r}; expected one of "
            f"{[m.value for m in cls]}"
        )


@dataclass
class ProfileSample:
    """One parsed profile row: a symbol/address plus its sampled weight.

    ``func_addr`` is ``None`` when the format (or the row) carries no address,
    in which case attribution falls back to the exact ``symbol`` name.
    ``alloc_bytes`` is ``0`` for formats without an allocation column.
    """

    symbol: str = ""
    func_addr: Optional[int] = None
    samples: int = 0
    alloc_bytes: int = 0

    def __post_init__(self) -> None:
        self.symbol = "" if self.symbol is None else str(self.symbol).strip()
        if self.func_addr is not None:
            if isinstance(self.func_addr, bool):
                raise ProfileError(f"invalid func_addr {self.func_addr!r}")
            if isinstance(self.func_addr, str):
                # Hex text ("0x1000", "140001000h") is an address, not a name:
                # parse it with the module's base-0 fallback so callers may pass
                # the spelling a profiler export uses.
                parsed = _parse_address_text(self.func_addr)
                if parsed is None:
                    raise ProfileError(f"invalid func_addr {self.func_addr!r}")
                addr = parsed
            else:
                try:
                    addr = int(self.func_addr)
                except (TypeError, ValueError) as exc:
                    raise ProfileError(
                        f"invalid func_addr {self.func_addr!r}"
                    ) from exc
            if addr < 0:
                raise ProfileError(f"func_addr must be non-negative, got {addr}")
            self.func_addr = addr
        try:
            self.samples = int(self.samples)
            self.alloc_bytes = int(self.alloc_bytes)
        except (TypeError, ValueError) as exc:
            raise ProfileError(f"non-numeric sample weight: {self!r}") from exc
        if self.samples < 0:
            raise ProfileError(f"samples must be non-negative, got {self.samples}")
        if self.alloc_bytes < 0:
            raise ProfileError(
                f"alloc_bytes must be non-negative, got {self.alloc_bytes}"
            )

    def as_dict(self) -> Dict[str, Any]:
        """JSON-ready form (plan trap 14: lists/plain scalars, never tuples)."""
        return {
            "symbol": self.symbol,
            "func_addr": self.func_addr,
            "addr_hex": _hex_addr(self.func_addr) if self.func_addr is not None else None,
            "samples": self.samples,
            "alloc_bytes": self.alloc_bytes,
        }


# --------------------------------------------------------------------------- #
# small parsers / coercions
# --------------------------------------------------------------------------- #

_HEX_DIGITS = "abcdefABCDEF"
_MULTIPLIERS = {"k": 1_000, "m": 1_000_000, "g": 1_000_000_000, "t": 1_000_000_000_000}
_DELIMITERS = (",", ";", "\t", "|")


class _RowError(ValueError):
    """One malformed CSV row (counted, never fatal to the import)."""


def _hex_addr(addr: int) -> str:
    return f"0x{int(addr):x}"


def _normalize_key(name: Any) -> str:
    """``"CPU Time"`` -> ``"cpu_time"``; stable header matching key."""
    text = re.sub(r"[^0-9a-z]+", "_", str(name or "").strip().lower())
    return text.strip("_")


def _sniff_delimiter(line: str) -> str:
    """Pick the most frequent candidate delimiter in the header line."""
    best_count, best_delim = 0, ","
    for delim in _DELIMITERS:
        count = line.count(delim)
        if count > best_count:
            best_count, best_delim = count, delim
    return best_delim


def _parse_address_text(text: Any) -> Optional[int]:
    """``"0x140001000"``/``"140001000h"``/``"sub_1400"`` -> int or ``None``.

    Unparsable text yields ``None`` (the row then falls back to name
    attribution) rather than an invented address -- never an unobserved offset.
    """
    body = "" if text is None else str(text).strip().strip("`").replace("_", "")
    if not body or body.startswith("-"):
        return None
    lowered = body.lower()
    if lowered.startswith("0x"):
        base, digits = 16, body[2:]
    elif lowered.endswith("h") and len(body) > 1:
        base, digits = 16, body[:-1]
    elif any(c in _HEX_DIGITS for c in body):
        base, digits = 16, body
    else:
        base, digits = 10, body
    if not digits:
        return None
    try:
        value = int(digits, base)
    except ValueError:
        return None
    return None if value < 0 else value


def _parse_count(text: Any) -> Optional[int]:
    """Parse ``"1200"``, ``"1.2K"``, ``"3M"`` -> int; ``None`` when malformed."""
    body = "" if text is None else str(text).strip().replace("_", "")
    if not body:
        return None
    multiplier = 1
    if body[-1].lower() in _MULTIPLIERS:
        multiplier = _MULTIPLIERS[body[-1].lower()]
        body = body[:-1]
    try:
        value = float(body)
    except ValueError:
        return None
    if value < 0:
        return None
    return int(round(value * multiplier))


def _parse_seconds(text: Any) -> Optional[float]:
    """Parse ``"1.234s"``/``"250ms"``/``"1.5"`` (implicit s) -> float seconds."""
    body = "" if text is None else str(text).strip()
    if not body:
        return None
    scale = 1.0
    lowered = body.lower()
    for suffix, factor in (("ms", 0.001), ("us", 1e-6), ("ns", 1e-9), ("s", 1.0)):
        if lowered.endswith(suffix):
            scale = factor
            body = body[: -len(suffix)]
            break
    try:
        value = float(body.strip())
    except ValueError:
        return None
    if value < 0:
        return None
    return value * scale


def _cpu_time_to_samples(seconds: float) -> int:
    """CPU-time-only VTune exports become a pseudo-count in milliseconds."""
    return max(0, int(round(seconds * 1000.0)))


def _require_samples(text: Any) -> int:
    value = _parse_count(text)
    if value is None:
        raise _RowError(f"invalid sample count {text!r}")
    return value


def _optional_alloc(text: Any) -> int:
    body = "" if text is None else str(text).strip()
    if not body:
        return 0
    value = _parse_count(body)
    if value is None:
        raise _RowError(f"invalid allocation size {text!r}")
    return value


# --------------------------------------------------------------------------- #
# header spec + row loop
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _Spec:
    """Column-name alternatives for one format."""

    name: str
    symbol_keys: Tuple[str, ...] = ()
    address_keys: Tuple[str, ...] = ()
    samples_keys: Tuple[str, ...] = ()
    time_keys: Tuple[str, ...] = ()
    alloc_keys: Tuple[str, ...] = ()
    symbol_or_address: bool = False
    time_to_ms: bool = False

    def required_groups(self) -> Tuple[Tuple[str, ...], ...]:
        first = self.symbol_keys
        if self.symbol_or_address:
            first = self.symbol_keys + self.address_keys
        return (first, self.samples_keys + self.time_keys)


_VTUNE_SPEC = _Spec(
    name="vtune",
    symbol_keys=("function", "function_name", "symbol", "func", "name"),
    samples_keys=("samples", "sample_count", "samples_count", "event_count"),
    address_keys=(
        "function_address",
        "address",
        "start_address",
        "func_addr",
        "module_offset",
        "ip",
    ),
    time_keys=("cpu_time", "cpu_time_s", "cputime", "time"),
    alloc_keys=(
        "alloc_bytes",
        "allocation_bytes",
        "bytes_allocated",
        "total_alloc_bytes",
        "alloc",
    ),
    time_to_ms=True,
)

_ETW_SPEC = _Spec(
    name="etw",
    symbol_keys=("symbol", "stack", "stack_symbol", "function", "image_function", "name"),
    samples_keys=("count", "weight", "events", "samples", "sample_count"),
    address_keys=("address", "stack_address", "func_addr", "module_offset", "ip"),
    alloc_keys=("alloc", "alloc_bytes", "allocation", "total_bytes", "bytes", "size"),
)

_PERF_SPEC = _Spec(
    name="perf",
    symbol_keys=("symbol", "sym", "function", "name", "comm_symbol"),
    address_keys=("address", "ip", "addr", "func_addr", "symbol_address", "start_address"),
    samples_keys=("samples", "sample", "sample_count", "period", "count", "event_count"),
    alloc_keys=("alloc", "alloc_bytes", "allocation_bytes", "bytes", "size", "size_bytes"),
    symbol_or_address=True,
)


def _first_index(header_norm: Sequence[str], keys: Sequence[str]) -> Optional[int]:
    for key in keys:
        if key in header_norm:
            return header_norm.index(key)
    return None


def _cell(row: Sequence[str], index: Optional[int]) -> str:
    if index is None or index >= len(row):
        return ""
    return row[index]


def _header_and_rows(
    text: str,
) -> Tuple[List[str], List[List[str]], str]:
    """Split profile text into (header, data rows, delimiter).

    Blank lines and ``#`` comment lines are dropped; the first other line is the
    header.  Deliberately strict: no ``eval``, no type guessing beyond the
    documented number formats.
    """
    normalized = str(text).replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    start = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            start = index
            break
    if start is None:
        raise ProfileError("profile text contains no header row")
    header_line = lines[start]
    delimiter = _sniff_delimiter(header_line)
    reader = csv.reader(io.StringIO("\n".join(lines[start:])), delimiter=delimiter)
    parsed = [row for row in reader]
    if not parsed:
        raise ProfileError("profile text contains no header row")
    header = [cell.strip() for cell in parsed[0]]
    return header, parsed[1:], delimiter


def _parse_with_spec(text: str, spec: _Spec) -> Tuple[List[ProfileSample], Dict[str, Any]]:
    """Shared strict row loop -> (address-sorted samples, diagnostics)."""
    header, rows, delimiter = _header_and_rows(text)
    header_norm = [_normalize_key(cell) for cell in header]
    if not any(header_norm):
        raise ProfileError(f"{spec.name}: empty header row")

    indices = {
        "symbol": _first_index(header_norm, spec.symbol_keys),
        "address": _first_index(header_norm, spec.address_keys),
        "samples": _first_index(header_norm, spec.samples_keys),
        "time": _first_index(header_norm, spec.time_keys),
        "alloc": _first_index(header_norm, spec.alloc_keys),
    }
    for group in spec.required_groups():
        if _first_index(header_norm, group) is None:
            raise ProfileError(
                f"{spec.name}: header lacks a required column "
                f"(one of {list(group)}); got {header}"
            )

    samples: List[ProfileSample] = []
    rejected = 0
    skipped = 0
    for row in rows:
        if not row or not any(cell.strip() for cell in row):
            skipped += 1
            continue
        if row[0].lstrip().startswith("#"):
            skipped += 1
            continue
        if len(row) != len(header):
            rejected += 1
            continue
        symbol = _cell(row, indices["symbol"])
        addr_text = _cell(row, indices["address"])
        addr = _parse_address_text(addr_text) if indices["address"] is not None else None
        if not symbol and addr is None:
            rejected += 1
            continue
        try:
            count_text = _cell(row, indices["samples"])
            if indices["samples"] is not None and count_text.strip():
                weight = _require_samples(count_text)
            else:
                seconds = _parse_seconds(_cell(row, indices["time"]))
                if seconds is None:
                    raise _RowError(f"invalid sample count {count_text!r}")
                weight = _cpu_time_to_samples(seconds) if spec.time_to_ms else int(seconds)
            alloc = _optional_alloc(_cell(row, indices["alloc"]))
        except _RowError:
            rejected += 1
            continue
        try:
            samples.append(
                ProfileSample(
                    symbol=symbol,
                    func_addr=addr,
                    samples=weight,
                    alloc_bytes=alloc,
                )
            )
        except ProfileError:
            rejected += 1

    samples.sort(key=_sort_key)
    diagnostics: Dict[str, Any] = {
        "format": spec.name,
        "header": list(header),
        "delimiter": delimiter,
        "rows": len(samples),
        "rejected": rejected,
        "skipped": skipped,
    }
    return samples, diagnostics


def _sort_key(sample: ProfileSample) -> Tuple[int, int, str, int, int]:
    """Deterministic address sort: addressed rows first, then address, symbol."""
    if sample.func_addr is None:
        return (1, 0, sample.symbol, sample.samples, sample.alloc_bytes)
    return (0, sample.func_addr, sample.symbol, sample.samples, sample.alloc_bytes)


# --------------------------------------------------------------------------- #
# public parsers
# --------------------------------------------------------------------------- #


def _read_text(source: Union[str, Path]) -> str:
    if isinstance(source, Path):
        try:
            return source.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise ProfileError(f"cannot read profile file {source}: {exc}") from exc
    return str(source)


def parse_vtune_csv(text: Union[str, Path]) -> List[ProfileSample]:
    """Parse a VTune hotspots CSV (header: ``Function`` + ``CPU Time``/``Samples``)."""
    samples, _diag = _parse_with_spec(_read_text(text), _VTUNE_SPEC)
    return samples


def parse_etw_csv(text: Union[str, Path]) -> List[ProfileSample]:
    """Parse an ETW/WPA CSV (header: ``Stack``/``Symbol`` + ``Count``)."""
    samples, _diag = _parse_with_spec(_read_text(text), _ETW_SPEC)
    return samples


def parse_perf_csv(text: Union[str, Path]) -> List[ProfileSample]:
    """Parse a perf-script CSV (header: ``symbol``/``address`` + ``samples``)."""
    samples, _diag = _parse_with_spec(_read_text(text), _PERF_SPEC)
    return samples


#: P1 CSV/delimited column specs.  The J2 family (``ASYNC``/``JFR``) is *not*
#: here: its parsers live in :mod:`toothpick.profile_jfr` and are reached by the
#: routing in :func:`parse_profile_text`, never through a ``_Spec``.
_PARSERS = {
    ProfileFormat.VTUNE: (_VTUNE_SPEC,),
    ProfileFormat.ETW: (_ETW_SPEC,),
    ProfileFormat.PERF: (_PERF_SPEC,),
}

#: J2 formats served by :mod:`toothpick.profile_jfr` (plan section 1.2).
_JFR_FORMATS = frozenset({ProfileFormat.ASYNC, ProfileFormat.JFR})


def _jfr_parsers() -> Tuple[Any, Any]:
    """Deferred access to the ``profile_jfr`` parsers (import-cycle guard).

    ``profile_jfr`` imports :class:`ProfileSample`, :class:`ProfileFormat` and the
    shared helpers from *this* module at module scope, so importing it at module
    scope here would close the cycle.  The import therefore happens at call time
    -- the deferred-import pattern ``cli.py`` already uses.  Nothing else in this
    module depends on the J2 module, so a store with no J2 profile never pays for
    it.
    """
    from toothpick.profile_jfr import parse_async_collapsed, parse_jfr_json

    return parse_async_collapsed, parse_jfr_json


def _parse_jfr_family(
    text: str, fmt: ProfileFormat
) -> Tuple[List[ProfileSample], Dict[str, Any]]:
    """Route explicit ``ASYNC``/``JFR`` text to its ``profile_jfr`` parser."""
    parse_async_collapsed, parse_jfr_json = _jfr_parsers()
    if fmt is ProfileFormat.ASYNC:
        return parse_async_collapsed(text)
    return parse_jfr_json(text)


def _sniff_jfr_family(text: str) -> Optional[ProfileFormat]:
    """``profile_jfr.detect_format(text)`` as a :class:`ProfileFormat`, or ``None``.

    Lazy import for the same cycle reason as :func:`_jfr_parsers`.  ``None`` means
    "no J2 signature", so the CSV header sniff still decides; it never means
    "reject", and it never overrides an explicit format.
    """
    from toothpick.profile_jfr import ASYNC_FORMAT, JFR_FORMAT, detect_format

    detected = detect_format(text)
    if detected == ASYNC_FORMAT:
        return ProfileFormat.ASYNC
    if detected == JFR_FORMAT:
        return ProfileFormat.JFR
    return None


def _sniff_csv_format(text: str) -> ProfileFormat:
    """P1 fallback sniff: the first non-comment line is a CSV header row."""
    header, _rows, _delim = _header_and_rows(text)
    return detect_format(header)


def detect_format(header: Sequence[str]) -> ProfileFormat:
    """Sniff the format from a header row; raise when nothing matches.

    Priority (documented, deterministic): VTune, then ETW, then perf.  Ambiguous
    headers are benign because the three row loops agree on
    symbol/address/samples semantics.
    """
    names = {_normalize_key(cell) for cell in header if str(cell or "").strip()}

    def has(*keys: str) -> bool:
        return any(key in names for key in keys)

    if has("function", "function_name", "func") and has(
        "cpu_time", "cpu_time_s", "time", "samples", "sample_count"
    ):
        return ProfileFormat.VTUNE
    if has("stack", "symbol", "image_function") and has("count", "weight", "events"):
        return ProfileFormat.ETW
    if (
        has("symbol", "sym", "function", "name")
        or has("address", "ip", "addr", "func_addr")
    ) and has("samples", "sample", "period", "count", "event_count"):
        return ProfileFormat.PERF
    raise ProfileError(
        f"cannot determine profile format from header {list(header)!r}; "
        "pass format='vtune'|'etw'|'perf'|'async'|'jfr'"
    )


def parse_profile_text(
    text: str, format: Union[ProfileFormat, str, None] = "auto"
) -> Tuple[List[ProfileSample], Dict[str, Any]]:
    """Parse profile *text* -> ``(samples, diagnostics)``.

    ``format`` is ``"auto"``, a :class:`ProfileFormat`, or its value string
    (``"vtune"``/``"etw"``/``"perf"``/``"async"``/``"jfr"``); anything else raises
    :class:`ProfileError` through :meth:`ProfileFormat.coerce` (whose ``None`` ->
    ``AUTO`` default is unchanged).

    **AUTO** asks the J2 sniff first (:func:`_sniff_jfr_family`, i.e.
    ``profile_jfr.detect_format``) and only then sniffs a CSV header row, so an
    async-profiler collapsed file or a ``jfr print --json`` document is detected
    and imported instead of being rejected for having no header row.  Text with
    neither signature keeps the P1 refusal.

    **Explicit** ``async``/``jfr`` route to
    ``profile_jfr.parse_async_collapsed`` / ``parse_jfr_json``, imported inside
    the routing helper: no top-level dependency, no import cycle.

    CSV diagnostics carry ``format``, ``header``, ``delimiter``, ``rows``,
    ``rejected`` and ``skipped``; the J2 formats carry their own ``profile_jfr``
    diagnostics (``format``, ``rows``, ``rejected`` plus the format counters).
    Malformed rows are counted, never raised.
    """
    fmt = ProfileFormat.coerce(format)
    if fmt is ProfileFormat.AUTO:
        fmt = _sniff_jfr_family(text) or _sniff_csv_format(text)
    if fmt in _JFR_FORMATS:
        return _parse_jfr_family(text, fmt)
    spec = _PARSERS[fmt][0]
    return _parse_with_spec(text, spec)


def parse_profile_file(
    path: PathLike, format: Union[ProfileFormat, str, None] = "auto"
) -> Tuple[List[ProfileSample], Dict[str, Any]]:
    """Read ``path`` and parse it -> ``(samples, diagnostics)``.

    ``format`` has the same meaning -- and the same J2 routing (AUTO sniff, or
    explicit ``async``/``jfr``) -- as :func:`parse_profile_text`, which does the
    parsing; the resolved path and file size are added to the diagnostics.
    """
    target = Path(path)
    try:
        text = target.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ProfileError(f"cannot read profile file {target}: {exc}") from exc
    diagnostics_path = dict(_scaffold_diagnostics(target))
    samples, diagnostics = parse_profile_text(text, format=format)
    diagnostics.update(diagnostics_path)
    return samples, diagnostics


def _scaffold_diagnostics(path: Path) -> Dict[str, Any]:
    return {"path": str(path), "size_bytes": _safe_size(path)}


def _safe_size(path: Path) -> int:
    try:
        return int(path.stat().st_size)
    except OSError:  # pragma: no cover - path disappeared mid-read
        return 0


# --------------------------------------------------------------------------- #
# storage: DDL + helpers
# --------------------------------------------------------------------------- #


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _create_profile_table(conn: sqlite3.Connection) -> None:
    conn.execute(FUNC_PROFILE_DDL)
    conn.execute(FUNC_PROFILE_INDEX_DDL)


def _table_exists(conn: sqlite3.Connection, name: str = "func_profile") -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
    ).fetchone()
    return row is not None


def _is_fresh_path(target: TargetLike) -> bool:
    """True when ``target`` is a path that does not name a db file yet.

    Answers the "brand-new store?" question, so it must be asked *before* a
    write connection creates the file.
    """
    if isinstance(target, (Database, sqlite3.Connection)):
        return False
    return not Path(target).exists()


def _column(row: Any, index: int, name: str) -> Any:
    """One column of a result row, for ``sqlite3.Row`` and tuple rows alike.

    A caller-supplied :class:`sqlite3.Connection` is never re-configured -- its
    ``row_factory``/``isolation_level`` belong to the caller -- so its rows are
    plain tuples: queries read by position and keep the SQL aliases only as
    documentation.
    """
    try:
        return row[index]
    except (IndexError, KeyError, TypeError):
        return row[name]


@contextmanager
def _read_connection(target: TargetLike) -> Iterator[sqlite3.Connection]:
    """Yield a read connection; anything opened here is closed in ``finally``."""
    if isinstance(target, Database):
        with target.connection() as conn:
            yield conn
        return
    if isinstance(target, sqlite3.Connection):
        yield target
        return
    path = Path(target)
    if not path.exists():
        raise ProfileError(f"database not found: {path}")
    conn = sqlite3.connect(str(path), isolation_level=None)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 10000")
        yield conn
    except sqlite3.Error as exc:
        raise ProfileError(f"profile read failed for {path}: {exc}") from exc
    finally:
        conn.close()


@contextmanager
def _write_connection(
    target: TargetLike, *, create: bool
) -> Iterator[sqlite3.Connection]:
    """Yield a write connection inside a transaction; always closed/released."""
    if isinstance(target, Database):
        with target.transaction() as conn:
            yield conn
        return
    if isinstance(target, sqlite3.Connection):
        conn = target
        # Never touch the caller's ``row_factory`` or ``isolation_level``: only
        # the transaction boundary is ours.  In autocommit mode we open one
        # explicitly; in legacy mode SQLite's transaction control owns it (and a
        # nested call inside the caller's own transaction joins that one), so we
        # commit/roll back if -- and only if -- a transaction is open.
        if conn.isolation_level is None and not conn.in_transaction:
            conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        else:
            if conn.in_transaction:
                conn.execute("COMMIT")
        return
    path = Path(target)
    if not path.exists() and not create:
        raise ProfileError(f"database not found: {path}")
    if str(path.parent) not in ("", "."):
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), isolation_level=None)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 10000")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        else:
            conn.execute("COMMIT")
    except sqlite3.Error as exc:
        raise ProfileError(f"profile write failed for {path}: {exc}") from exc
    finally:
        conn.close()


def ensure_profile_table(target: TargetLike) -> None:
    """Create ``func_profile`` + its hotness index if absent (idempotent).

    Accepts a :class:`Database`, a raw :class:`sqlite3.Connection`, or a db path
    (a path is created when missing -- "ensure" is explicit about that).
    """
    with _write_connection(target, create=True) as conn:
        _create_profile_table(conn)


# --------------------------------------------------------------------------- #
# attribution + hotness
# --------------------------------------------------------------------------- #


def _require_sha(sha: Any) -> str:
    text = "" if sha is None else str(sha).strip()
    if not text:
        raise ProfileError("binary_sha256 is required")
    return text


def _coerce_sample(obj: Any) -> ProfileSample:
    if isinstance(obj, ProfileSample):
        return obj
    if isinstance(obj, Mapping):
        addr = obj.get("func_addr", obj.get("address"))
        symbol = obj.get("symbol", obj.get("name", "")) or ""
        weight = obj.get("samples", obj.get("sample_count", 0)) or 0
        alloc = obj.get("alloc_bytes", obj.get("alloc", 0)) or 0
        return ProfileSample(
            symbol=symbol,
            func_addr=_parse_address_text(addr) if isinstance(addr, str) else addr,
            samples=weight,
            alloc_bytes=alloc,
        )
    raise ProfileError(
        f"unsupported sample input {type(obj).__name__}; expected ProfileSample"
    )


def _coerce_samples(samples: Any) -> Tuple[List[ProfileSample], int]:
    """Normalize accepted inputs -> (rows, parser-rejected count)."""
    if (
        isinstance(samples, tuple)
        and len(samples) == 2
        and isinstance(samples[1], Mapping)
    ):
        rows, diagnostics = samples
        return [_coerce_sample(row) for row in rows], int(
            diagnostics.get("rejected", 0) or 0
        )
    if isinstance(samples, Path):
        rows, diagnostics = parse_profile_file(samples)
        return rows, int(diagnostics.get("rejected", 0) or 0)
    if isinstance(samples, str):
        if "\n" in samples or "\r" in samples:
            rows, diagnostics = parse_profile_text(samples)
        else:
            rows, diagnostics = parse_profile_file(samples)
        return rows, int(diagnostics.get("rejected", 0) or 0)
    if samples is None:
        raise ProfileError(
            "samples must be a sequence of ProfileSample, a profile path, or CSV text"
        )
    try:
        items = list(samples)
    except TypeError as exc:
        raise ProfileError(
            "samples must be a sequence of ProfileSample, a profile path, or CSV text"
        ) from exc
    return [_coerce_sample(item) for item in items], 0


def import_samples(
    db_or_path: TargetLike,
    sha: Any,
    samples: Any,
    *,
    create: bool = False,
    rejected: Optional[int] = None,
) -> Dict[str, Any]:
    """Attribute profile rows onto known ``funcs`` and store per-function hotness.

    ``samples`` may be a list of :class:`ProfileSample` (or mappings), the
    ``(samples, diagnostics)`` tuple from :func:`parse_profile_file`, a profile
    path, or raw CSV text.

    Attribution order per plan section 1.2: exact ``func_addr`` against ``funcs``
    first, then exact ``name``.  Rows matching neither are counted
    ``unattributed`` -- **no ``funcs`` row is ever fabricated**, and unattributed
    samples are never folded back into the hotness denominator.

    ``create=True`` on a path that does not exist yet bootstraps a fresh store:
    the ``func_profile`` table *and* the empty ``funcs`` table attribution reads
    against (creating an empty table observes nothing).  An existing database
    without a ``funcs`` table is a foreign store -- refused, never guessed at.

    Import semantics are *replace*, not accumulate: the previous ``func_profile``
    rows for ``sha`` are deleted in the same transaction, so hotness always sums
    to 1.0 over the attributed set of the import that wrote it (0.0 when
    nothing was attributed).
    """
    binary = _require_sha(sha)
    rows, parse_rejected = _coerce_samples(samples)
    if rejected is not None:
        parse_rejected = int(rejected)

    # Ask "brand-new file?" before the write connection creates it.
    bootstrap_schema = create and _is_fresh_path(db_or_path)

    with _write_connection(db_or_path, create=create) as conn:
        _create_profile_table(conn)
        if not _table_exists(conn, "funcs"):
            if not bootstrap_schema:
                raise ProfileError(
                    "database has no funcs table to attribute against; import "
                    "into a toothpick store, or pass create=True for a fresh path"
                )
            conn.execute(MINIMAL_FUNCS_DDL)

        func_rows = conn.execute(_FUNCS_PAIR_SQL, (binary,)).fetchall()
        known_addrs = set()
        name_to_addr: Dict[str, int] = {}
        for row in func_rows:
            addr = int(_column(row, 0, "func_addr"))
            known_addrs.add(addr)
            name = (_column(row, 1, "name") or "").strip()
            if name and (name not in name_to_addr or addr < name_to_addr[name]):
                name_to_addr[name] = addr

        aggregated: Dict[int, List[int]] = {}
        imported_rows = 0
        unattributed_rows = 0
        unattributed_samples = 0
        total_samples = 0
        for sample in rows:
            total_samples += sample.samples
            target: Optional[int] = None
            if sample.func_addr is not None and sample.func_addr in known_addrs:
                target = sample.func_addr
            elif sample.symbol and sample.symbol in name_to_addr:
                target = name_to_addr[sample.symbol]
            if target is None:
                unattributed_rows += 1
                unattributed_samples += sample.samples
                continue
            imported_rows += 1
            slot = aggregated.setdefault(target, [0, 0])
            slot[0] += sample.samples
            slot[1] += sample.alloc_bytes

        attributed_samples = sum(count for count, _alloc in aggregated.values())
        now = _utc_now()
        conn.execute("DELETE FROM func_profile WHERE binary_sha256 = ?", (binary,))
        if aggregated:
            payload = []
            for addr in sorted(aggregated):
                count, alloc = aggregated[addr]
                hotness = (count / attributed_samples) if attributed_samples else 0.0
                payload.append((binary, addr, count, alloc, hotness, now))
            conn.executemany(
                "INSERT INTO func_profile (binary_sha256, func_addr, sample_count, "
                "alloc_bytes, hotness, updated_at) VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT (binary_sha256, func_addr) DO UPDATE SET "
                "sample_count = excluded.sample_count, "
                "alloc_bytes = excluded.alloc_bytes, "
                "hotness = excluded.hotness, "
                "updated_at = excluded.updated_at",
                payload,
            )

    return {
        "binary_sha256": binary,
        "imported": imported_rows,
        "unattributed": unattributed_rows,
        "rejected": parse_rejected,
        "total_samples": total_samples,
        "attributed_samples": attributed_samples,
        "unattributed_samples": unattributed_samples,
        "functions": len(aggregated),
    }


def resolve_top_n(top_n: Optional[int] = None) -> int:
    """Coerce ``top_n``: default :data:`DEFAULT_TOP_N`, clamp to :data:`MAX_TOP_N`."""
    if top_n is None:
        return DEFAULT_TOP_N
    if isinstance(top_n, bool):
        raise ProfileError(f"top_n must be an integer, got {top_n!r}")
    try:
        value = int(top_n)
    except (TypeError, ValueError) as exc:
        raise ProfileError(f"top_n must be an integer, got {top_n!r}") from exc
    if value <= 0:
        raise ProfileError(f"top_n must be positive, got {value}")
    return min(value, MAX_TOP_N)


def hottest(
    db_or_path: TargetLike, sha: Any, top_n: Optional[int] = DEFAULT_TOP_N
) -> List[Dict[str, Any]]:
    """Hottest functions of ``sha``, bounded and deterministically ordered.

    Ordered ``hotness DESC, func_addr ASC`` (Risk R1) and capped by
    :func:`resolve_top_n`.  ``func_profile`` is queried on its own and names are
    decoration looked up afterwards, only when a ``funcs`` table exists, so a
    store without ``funcs`` still answers a bounded list instead of raising
    "no such table: funcs".  A store without a ``func_profile`` table answers
    ``[]`` -- reads never create state.
    """
    budget = resolve_top_n(top_n)
    binary = _require_sha(sha)
    with _read_connection(db_or_path) as conn:
        if not _table_exists(conn):
            return []
        rows = conn.execute(_HOTTEST_SQL, (binary, budget)).fetchall()
        names: Dict[int, str] = {}
        if rows and _table_exists(conn, "funcs"):
            for name_row in conn.execute(_FUNCS_PAIR_SQL, (binary,)).fetchall():
                names[int(_column(name_row, 0, "func_addr"))] = (
                    _column(name_row, 1, "name") or ""
                )
    items: List[Dict[str, Any]] = []
    for row in rows[:budget]:
        addr = int(_column(row, 0, "func_addr"))
        items.append(
            {
                "func_addr": addr,
                "addr_hex": _hex_addr(addr),
                "name": names.get(addr, ""),
                "sample_count": int(_column(row, 1, "sample_count")),
                "alloc_bytes": int(_column(row, 2, "alloc_bytes")),
                "hotness": float(_column(row, 3, "hotness")),
            }
        )
    return items


def get_profile(
    db_or_path: TargetLike, sha: Any, func_addr: Union[int, str]
) -> Optional[Dict[str, Any]]:
    """Profile row for one function -> ``{sample_count, alloc_bytes, hotness}``.

    ``None`` when the function has no profile row, when no profile table exists
    (callers extend ``get_function`` output only when this is non-``None``), or
    when ``func_addr`` is text that parses to no address: unparsable text is not
    an observed offset, so there is nothing to look up and nothing to guess.
    A boolean or negative ``func_addr`` is a programming error and raises.
    """
    binary = _require_sha(sha)
    if isinstance(func_addr, bool):
        raise ProfileError(f"invalid func_addr {func_addr!r}")
    addr: Optional[int]
    if isinstance(func_addr, str):
        addr = _parse_address_text(func_addr)
    elif isinstance(func_addr, int):
        if func_addr < 0:
            raise ProfileError(f"func_addr must be non-negative, got {func_addr}")
        addr = func_addr
    else:
        raise ProfileError(f"invalid func_addr {func_addr!r}")
    with _read_connection(db_or_path) as conn:
        if not _table_exists(conn):
            return None
        if addr is None:
            return None
        row = conn.execute(_GET_PROFILE_SQL, (binary, addr)).fetchone()
    if row is None:
        return None
    return {
        "sample_count": int(_column(row, 0, "sample_count")),
        "alloc_bytes": int(_column(row, 1, "alloc_bytes")),
        "hotness": float(_column(row, 2, "hotness")),
    }
