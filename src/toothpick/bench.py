"""Toothpick A/B bench harness (P4, Chunk 1): wall-clock runs -> gated verdicts.

Binding contract: ``docs/units/UNIT-P4-BENCH-PLAN.md`` section 1 (module --
:func:`measure`, :func:`compare`, :func:`record_claim`, :class:`BenchError`, the
three defaults) and section 2 (interfaces), governed by ``docs/NEXT-TRACK.md``
P4 ("Before/after wall-clock measurement with statistical gating; speed claims
receipted") and the deck postmortem rule that a speed claim carries the
``UNVERIFIED (stability)`` label.

Three properties are load-bearing:

* **No shell, ever.**  ``cmd`` is an argv list handed to :func:`subprocess.run`
  with ``shell=False``; a bare string command is *refused* rather than split, so
  a command that reads like shell text can never acquire shell meaning here
  (Design section 19: refuse rather than guess intent).
* **A crashing target is never benchmarked.**  Warmup runs and measured runs
  both check the exit status: a non-zero return raises :class:`BenchError`
  instead of folding a fast failure into the timings -- a target that dies in
  3 ms is not a speedup.
* **Noise is never a win.**  :func:`compare` claims ``faster=True`` only when
  the median ratio clears ``min_effect`` **and** the two inter-quartile bands
  are disjoint.  Overlapping spreads yield ``faster=False`` with an
  ``inconclusive`` reason: there is no code path here that claims a win over
  noise.

Warmup runs are executed but never timed (cold-page/JIT/cache warm paths are
exactly what warmup is for); only ``reps`` samples reach ``times_ms``.  Every
statistic is derived from ``times_ms`` by pure stdlib arithmetic with a
deterministic linear-interpolation quantile, so a recorded run replays to the
same numbers on any interpreter (no locale-, version- or platform-dependent
statistics helper decides a verdict).

Receipts: :func:`record_claim` writes through the existing
``Database.add_receipt`` (Chunk 2's CLI reuses it), with the ``func_addr=-1``
sentinel for "not attached to a function" and an explicit ``func_addr: null``
preserved in the JSON payload (``docs/traps.md`` 23: negative *column* value,
explicit ``None`` in the payload), ``phase="bench"``, and the binding
``verifier_status="UNVERIFIED (stability)"`` label -- a wall-clock delta is
evidence about a schedule, never a grounding claim about bytes.  The receipt id
comes from :func:`toothpick.storage.make_receipt_id`, so replaying the same
claim updates the row instead of appending a duplicate.

Windows traps (``docs/traps.md`` 7/11): a ``Database`` this module opens (path
targets) is closed in a ``finally`` -- a leaked WAL handle surfaces as
``PermissionError: [WinError 32]`` on teardown -- a caller-supplied database or
connection is used as-is and never closed here, and nothing in this module
installs ``faulthandler`` (the suite keeps ``-p no:faulthandler``).

Out of scope (stated): profiler-guided attribution (P1 owns hotness),
perf-counter depth and any claim beyond wall-clock means/medians, and the
Chunk 2 CLI/``tests/test_bench.py`` surface.
"""

from __future__ import annotations

import json
import math
import sqlite3
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from toothpick.storage import Database, PathLike, StorageError, make_receipt_id

__all__ = [
    "BENCH_PHASE",
    "DEFAULT_MIN_EFFECT",
    "DEFAULT_REPS",
    "DEFAULT_TIMEOUT",
    "DEFAULT_WARMUP",
    "FUNC_ADDR_NONE_SENTINEL",
    "OVERLAP_RULE_IQR",
    "SUPPORTED_OVERLAP_RULES",
    "VERIFIER_STATUS_STABILITY",
    "BenchError",
    "RunResult",
    "Verdict",
    "compare",
    "measure",
    "record_claim",
]

#: Default measured-repetition count (plan section 1: ``reps=11``).
DEFAULT_REPS = 11

#: Default discarded warmup-run count (plan section 1: ``warmup=2``).
DEFAULT_WARMUP = 2

#: Default per-run wall-clock timeout in seconds (plan section 1: ``timeout=300``).
DEFAULT_TIMEOUT = 300.0

#: Default minimum relative improvement that can count as a win (plan section 1).
DEFAULT_MIN_EFFECT = 0.05

#: The only supported noise-rejection rule (plan section 1: ``overlap_rule="iqr"``).
OVERLAP_RULE_IQR = "iqr"
SUPPORTED_OVERLAP_RULES: Tuple[str, ...] = (OVERLAP_RULE_IQR,)

#: Receipt phase for every bench claim.
BENCH_PHASE = "bench"

#: ``func_addr`` sentinel: a bench claim is not attached to any function
#: (``docs/traps.md`` 23 -- negative sentinel in the SQLite column).
FUNC_ADDR_NONE_SENTINEL = -1

#: Binding postmortem label for transformed-bytecode speed claims
#: (``docs/NEXT-TRACK.md`` binding constraints).
VERIFIER_STATUS_STABILITY = "UNVERIFIED (stability)"

_STDERR_SNIPPET_CHARS = 240

_STAT_FIELDS: Tuple[str, ...] = (
    "median_ms",
    "mean_ms",
    "stdev_ms",
    "min_ms",
    "max_ms",
)

_BAND_KEYS: Tuple[str, ...] = ("before_iqr", "after_iqr")

# Mirrors ``toothpick.storage._RECEIPT_UPSERT`` for raw-connection targets: the
# only difference between the two write paths is who owns the handle.
_RECEIPT_UPSERT_SQL = """
INSERT INTO receipts (receipt_id, binary_sha256, func_addr, phase, timestamp, payload_json)
VALUES (?, ?, ?, ?, ?, ?)
ON CONFLICT (receipt_id) DO UPDATE SET
    phase        = excluded.phase,
    timestamp    = excluded.timestamp,
    payload_json = excluded.payload_json
"""


class BenchError(Exception):
    """Raised for an invalid bench request or a target that cannot be measured."""


# --------------------------------------------------------------------------- #
# argument validation (strict: refuse, never silently coerce)
# --------------------------------------------------------------------------- #


def _coerce_argv(argv: Any, *, name: str = "cmd") -> List[str]:
    """Normalize an argv container to ``list[str]``; empty strings are refused."""
    if isinstance(argv, (str, bytes, bytearray)):
        raise BenchError(
            f"{name} must be a list of argument strings, not a single string"
        )
    if argv is None:
        return []
    try:
        items = list(argv)
    except TypeError as exc:
        raise BenchError(
            f"{name} must be an iterable of argument strings, got {type(argv).__name__}"
        ) from exc
    result: List[str] = []
    for index, item in enumerate(items):
        if not isinstance(item, str):
            raise BenchError(f"{name}[{index}] must be a string, got {type(item).__name__}")
        if not item:
            raise BenchError(f"{name}[{index}] must not be empty")
        result.append(item)
    return result


def _require_cmd(cmd: Any) -> List[str]:
    """Validate an executable argv: non-empty, all strings, never a shell string."""
    if cmd is None:
        raise BenchError("cmd must be a non-empty list of argument strings")
    if isinstance(cmd, (str, bytes, bytearray)):
        raise BenchError(
            "cmd must be an argv list of strings, not a shell string "
            "(this harness runs shell=False; pass ['prog', 'arg', ...])"
        )
    argv = _coerce_argv(cmd, name="cmd")
    if not argv:
        raise BenchError("cmd must not be empty")
    return argv


def _coerce_float(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BenchError(f"{name} must be a number, got {value!r}")
    number = float(value)
    if math.isnan(number) or math.isinf(number):
        raise BenchError(f"{name} must be finite, got {value!r}")
    return number


def _require_non_negative_float(name: str, value: Any) -> float:
    number = _coerce_float(name, value)
    if number < 0:
        raise BenchError(f"{name} must be non-negative, got {number}")
    return number


def _require_count(name: str, value: Any, default: int, *, minimum: int) -> int:
    if value is None:
        value = default
    if isinstance(value, bool) or not isinstance(value, int):
        raise BenchError(f"{name} must be an integer, got {value!r}")
    if value < minimum:
        raise BenchError(f"{name} must be >= {minimum}, got {value}")
    return value


def _require_timeout(timeout: Any) -> float:
    if timeout is None:
        return DEFAULT_TIMEOUT
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise BenchError(f"timeout must be a positive number of seconds, got {timeout!r}")
    value = float(timeout)
    if math.isnan(value) or math.isinf(value) or value <= 0:
        raise BenchError(f"timeout must be a positive number of seconds, got {timeout!r}")
    return value


def _require_min_effect(min_effect: Any) -> float:
    if isinstance(min_effect, bool) or not isinstance(min_effect, (int, float)):
        raise BenchError(f"min_effect must be a number in [0, 1), got {min_effect!r}")
    value = float(min_effect)
    if math.isnan(value) or not 0.0 <= value < 1.0:
        raise BenchError(f"min_effect must be in [0, 1), got {min_effect!r}")
    return value


def _require_overlap_rule(overlap_rule: Any) -> str:
    if overlap_rule is None:
        return OVERLAP_RULE_IQR
    if not isinstance(overlap_rule, str) or overlap_rule.strip().lower() != OVERLAP_RULE_IQR:
        raise BenchError(
            "overlap_rule must be one of "
            + ", ".join(repr(rule) for rule in SUPPORTED_OVERLAP_RULES)
            + f", got {overlap_rule!r}"
        )
    return OVERLAP_RULE_IQR


def _require_sha(sha: Any) -> str:
    text = "" if sha is None else str(sha).strip()
    if not text:
        raise BenchError("binary sha is required to receipt a bench claim")
    return text


def _require_label(label: Any) -> str:
    if not isinstance(label, str) or not label.strip():
        raise BenchError(f"label must be a non-empty string, got {label!r}")
    return label.strip()


# --------------------------------------------------------------------------- #
# deterministic statistics (pure functions of the sample list)
# --------------------------------------------------------------------------- #


def _coerce_times(times: Any, *, name: str = "times_ms") -> List[float]:
    if times is None:
        return []
    if isinstance(times, (str, bytes, bytearray, Mapping)):
        raise BenchError(f"{name} must be a sequence of numbers, got {type(times).__name__}")
    try:
        items = list(times)
    except TypeError as exc:
        raise BenchError(
            f"{name} must be a sequence of numbers, got {type(times).__name__}"
        ) from exc
    return [
        _require_non_negative_float(f"{name}[{index}]", item)
        for index, item in enumerate(items)
    ]


def _quantile(ordered: Sequence[float], fraction: float) -> float:
    """Deterministic quantile of an ascending sample list (linear interpolation).

    ``fraction`` is clamped by the caller; ``ordered`` must be non-empty and
    ascending.  A single sample is its own quantile.  This is the inclusive
    linear method, implemented here so a verdict never depends on an
    interpreter's statistics helper version.
    """
    count = len(ordered)
    if count == 1:
        return float(ordered[0])
    position = fraction * (count - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _iqr_bounds(times: Sequence[float]) -> Tuple[float, float]:
    """``(Q1, Q3)`` of ``times`` -- the band :func:`compare` gates on."""
    ordered = sorted(float(value) for value in times)
    if not ordered:
        raise BenchError("IQR bands require at least one timing sample")
    return _quantile(ordered, 0.25), _quantile(ordered, 0.75)


def _bands_overlap(
    before: Sequence[float], after: Sequence[float]
) -> bool:
    """Closed bands touch == overlap: only strict separation proves a delta."""
    return before[0] <= after[1] and after[0] <= before[1]


def _stats(times: Sequence[float]) -> Dict[str, float]:
    """Summary statistics of one measured run (sample stdev; 0.0 for one sample)."""
    ordered = sorted(float(value) for value in times)
    count = len(ordered)
    if not count:
        raise BenchError("times_ms must contain at least one sample")
    mean = math.fsum(ordered) / count
    if count > 1:
        variance = math.fsum((value - mean) ** 2 for value in ordered) / (count - 1)
        stdev = math.sqrt(variance)
    else:
        stdev = 0.0
    return {
        "median_ms": _quantile(ordered, 0.5),
        "mean_ms": mean,
        "stdev_ms": stdev,
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


# --------------------------------------------------------------------------- #
# result records
# --------------------------------------------------------------------------- #


@dataclass
class RunResult:
    """One measured command: the argv, the sampling plan, and the sample stats.

    ``cmd``/``reps``/``warmup``/``times_ms`` are the inputs; the five statistics
    are *derived*.  Left unset they are computed from ``times_ms`` in
    :meth:`__post_init__`; supplied values are validated and trusted, so a
    fixture can pin a synthetic timer without running a process.  :meth:`as_dict`
    is JSON-ready plain scalars (``docs/traps.md`` 14: lists, never tuples).
    """

    cmd: List[str] = field(default_factory=list)
    reps: Optional[int] = None
    warmup: int = 0
    times_ms: List[float] = field(default_factory=list)
    median_ms: Optional[float] = None
    mean_ms: Optional[float] = None
    stdev_ms: Optional[float] = None
    min_ms: Optional[float] = None
    max_ms: Optional[float] = None

    def __post_init__(self) -> None:
        self.cmd = _coerce_argv(self.cmd, name="cmd")
        self.times_ms = _coerce_times(self.times_ms)
        if self.reps is None:
            self.reps = len(self.times_ms)
        if isinstance(self.reps, bool) or not isinstance(self.reps, int) or self.reps < 1:
            raise BenchError(f"reps must be a positive integer, got {self.reps!r}")
        if (
            isinstance(self.warmup, bool)
            or not isinstance(self.warmup, int)
            or self.warmup < 0
        ):
            raise BenchError(f"warmup must be a non-negative integer, got {self.warmup!r}")
        computed = _stats(self.times_ms) if self.times_ms else None
        for name in _STAT_FIELDS:
            supplied = getattr(self, name)
            if supplied is None:
                if computed is None:
                    raise BenchError(
                        "RunResult requires times_ms or explicit "
                        + "/".join(_STAT_FIELDS)
                    )
                setattr(self, name, computed[name])
            else:
                setattr(self, name, _require_non_negative_float(name, supplied))

    def iqr_bounds(self) -> Tuple[float, float]:
        """``(Q1, Q3)`` of this run's samples (the gate :func:`compare` applies)."""
        return _iqr_bounds(self.times_ms)

    def stat(self, name: str) -> float:
        """One derived statistic as ``float`` (always set by :meth:`__post_init__`)."""
        if name not in _STAT_FIELDS:
            raise BenchError(f"unknown RunResult statistic {name!r}")
        value = getattr(self, name)
        if value is None:  # pragma: no cover - __post_init__ fills every field
            raise BenchError(f"RunResult.{name} is unset")
        return float(value)

    @property
    def rep_count(self) -> int:
        """Measured-run count as ``int`` (always set by :meth:`__post_init__`)."""
        reps = self.reps
        if reps is None:  # pragma: no cover - __post_init__ fills every field
            raise BenchError("RunResult.reps is unset")
        return int(reps)

    def as_dict(self) -> Dict[str, Any]:
        """JSON-ready form: plain scalars and lists only (no tuples, no objects)."""
        return {
            "cmd": [str(part) for part in self.cmd],
            "reps": self.rep_count,
            "warmup": int(self.warmup),
            "times_ms": [float(value) for value in self.times_ms],
            "median_ms": self.stat("median_ms"),
            "mean_ms": self.stat("mean_ms"),
            "stdev_ms": self.stat("stdev_ms"),
            "min_ms": self.stat("min_ms"),
            "max_ms": self.stat("max_ms"),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RunResult":
        """Rebuild a run from :meth:`as_dict` output (Chunk 2 loads JSON files)."""
        if not isinstance(data, Mapping):
            raise BenchError(f"expected a RunResult mapping, got {type(data).__name__}")
        try:
            return cls(
                cmd=list(data.get("cmd") or []),
                reps=data.get("reps"),
                warmup=data.get("warmup") or 0,
                times_ms=list(data.get("times_ms") or []),
                median_ms=data.get("median_ms"),
                mean_ms=data.get("mean_ms"),
                stdev_ms=data.get("stdev_ms"),
                min_ms=data.get("min_ms"),
                max_ms=data.get("max_ms"),
            )
        except BenchError:
            raise
        except (TypeError, ValueError) as exc:
            raise BenchError(f"malformed RunResult mapping: {exc}") from exc


def _coerce_bands(bands: Any) -> Dict[str, List[float]]:
    if not isinstance(bands, Mapping):
        raise BenchError(f"bands must be a mapping, got {type(bands).__name__}")
    result: Dict[str, List[float]] = {}
    for key in _BAND_KEYS:
        raw = bands.get(key)
        if isinstance(raw, (str, bytes, bytearray)) or raw is None:
            raise BenchError(f"bands[{key!r}] must be a two-number sequence")
        try:
            low_raw, high_raw = raw[0], raw[1]
        except (TypeError, IndexError, KeyError) as exc:
            raise BenchError(f"bands[{key!r}] must be a two-number sequence") from exc
        low = _require_non_negative_float(f"bands[{key!r}][0]", low_raw)
        high = _require_non_negative_float(f"bands[{key!r}][1]", high_raw)
        if low > high:
            raise BenchError(f"bands[{key!r}] must be ascending, got [{low}, {high}]")
        result[key] = [low, high]
    return result


@dataclass
class Verdict:
    """The gated answer to "did ``after`` beat ``before``?".

    ``faster`` is ``True`` only for a clean win: the median ratio cleared
    ``min_effect`` **and** the IQR bands are disjoint.  ``reason`` is a
    human-readable explanation whose prefix names the case (``faster:`` /
    ``inconclusive:`` / ``not faster:``).
    """

    faster: bool
    ratio: float
    delta_ms: float
    reason: str
    bands: Dict[str, List[float]]
    min_effect: float = DEFAULT_MIN_EFFECT

    def __post_init__(self) -> None:
        if not isinstance(self.faster, bool):
            raise BenchError(f"faster must be a bool, got {self.faster!r}")
        self.ratio = _require_non_negative_float("ratio", self.ratio)
        self.delta_ms = _coerce_float("delta_ms", self.delta_ms)
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise BenchError("reason must be a non-empty string")
        self.bands = _coerce_bands(self.bands)
        self.min_effect = _require_min_effect(self.min_effect)

    @property
    def overlap(self) -> bool:
        """``True`` when the two IQR bands are not strictly separated."""
        bands = self.bands
        return _bands_overlap(bands["before_iqr"], bands["after_iqr"])

    def as_dict(self) -> Dict[str, Any]:
        """JSON-ready verdict (receipt payload body; plain scalars only)."""
        return {
            "faster": bool(self.faster),
            "ratio": float(self.ratio),
            "delta_ms": float(self.delta_ms),
            "reason": str(self.reason),
            "bands": {key: [float(value) for value in self.bands[key]] for key in _BAND_KEYS},
            "min_effect": float(self.min_effect),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Verdict":
        """Rebuild a verdict from :meth:`as_dict` output (CLI ``bench compare`` JSON)."""
        if not isinstance(data, Mapping):
            raise BenchError(f"expected a Verdict mapping, got {type(data).__name__}")
        missing = [
            name
            for name in ("faster", "ratio", "delta_ms", "reason", "bands")
            if name not in data
        ]
        if missing:
            raise BenchError("verdict mapping missing " + ", ".join(missing))
        try:
            return cls(
                faster=data["faster"],
                ratio=data["ratio"],
                delta_ms=data["delta_ms"],
                reason=data["reason"],
                bands=data["bands"],
                min_effect=data.get("min_effect", DEFAULT_MIN_EFFECT),
            )
        except BenchError:
            raise
        except (TypeError, ValueError) as exc:
            raise BenchError(f"malformed Verdict mapping: {exc}") from exc


def _coerce_run(name: str, value: Any) -> RunResult:
    if isinstance(value, RunResult):
        return value
    if isinstance(value, Mapping):
        try:
            return RunResult.from_dict(value)
        except BenchError as exc:
            raise BenchError(f"{name} is not a valid RunResult: {exc}") from exc
    raise BenchError(
        f"{name} must be a RunResult or its as_dict() mapping, got {type(value).__name__}"
    )


def _coerce_verdict(name: str, value: Any) -> Verdict:
    if isinstance(value, Verdict):
        return value
    if isinstance(value, Mapping):
        try:
            return Verdict.from_dict(value)
        except BenchError as exc:
            raise BenchError(f"{name} is not a valid Verdict: {exc}") from exc
    raise BenchError(
        f"{name} must be a Verdict or its as_dict() mapping, got {type(value).__name__}"
    )


# --------------------------------------------------------------------------- #
# measurement
# --------------------------------------------------------------------------- #


def _describe(argv: Sequence[str]) -> str:
    return " ".join(argv)


def _stderr_snippet(stderr: Optional[bytes]) -> str:
    if not stderr:
        return ""
    text = stderr.decode("utf-8", errors="replace").strip()
    if not text:
        return ""
    if len(text) > _STDERR_SNIPPET_CHARS:
        text = text[:_STDERR_SNIPPET_CHARS] + "..."
    return f" (stderr: {text})"


def _run_once(argv: Sequence[str], timeout: float) -> float:
    """Run one command, return its wall-clock duration in milliseconds.

    Output is captured (a chatty target must not flood the harness) and the exit
    status is checked before the timing is allowed to become a sample.
    """
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            list(argv),
            shell=False,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise BenchError(
            f"command exceeded the {timeout:g}s timeout: {_describe(argv)}"
        ) from exc
    except OSError as exc:
        raise BenchError(
            f"command could not be started: {_describe(argv)} ({exc})"
        ) from exc
    except ValueError as exc:
        raise BenchError(
            f"command could not be started: {_describe(argv)} ({exc})"
        ) from exc
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if completed.returncode != 0:
        raise BenchError(
            f"command exited {completed.returncode} after {elapsed_ms:.3f} ms; "
            f"refusing to benchmark a non-zero exit: {_describe(argv)}"
            f"{_stderr_snippet(completed.stderr)}"
        )
    return elapsed_ms


def measure(
    cmd: Sequence[str],
    reps: int = DEFAULT_REPS,
    warmup: int = DEFAULT_WARMUP,
    timeout: float = DEFAULT_TIMEOUT,
) -> RunResult:
    """Time ``cmd`` ``reps`` times, after ``warmup`` discarded runs.

    Wall-clock is taken with :func:`time.perf_counter` around
    :func:`subprocess.run` (``shell=False``).  Warmup runs are executed -- so a
    cold target warms up -- but never timed.  An empty ``cmd``, a non-positive
    ``reps``, a negative ``warmup``, a non-positive ``timeout`` or a non-zero
    exit (including an unrunnable argv) raises :class:`BenchError`: a harness
    that timings a crashing target reports a meaningless "speedup".
    """
    argv = _require_cmd(cmd)
    counted_reps = _require_count("reps", reps, DEFAULT_REPS, minimum=1)
    counted_warmup = _require_count("warmup", warmup, DEFAULT_WARMUP, minimum=0)
    limit = _require_timeout(timeout)

    for _ in range(counted_warmup):
        _run_once(argv, limit)
    times = [_run_once(argv, limit) for _ in range(counted_reps)]
    return RunResult(
        cmd=list(argv),
        reps=counted_reps,
        warmup=counted_warmup,
        times_ms=times,
    )


# --------------------------------------------------------------------------- #
# comparison (pure function of the two runs)
# --------------------------------------------------------------------------- #


def compare(
    before: RunResult,
    after: RunResult,
    min_effect: float = DEFAULT_MIN_EFFECT,
    overlap_rule: str = OVERLAP_RULE_IQR,
) -> Verdict:
    """Gate ``after`` against ``before``: a win needs effect *and* separation.

    ``ratio = median(after) / median(before)``; ``faster`` is ``True`` iff
    ``ratio <= 1 - min_effect`` **and** the IQR bands do not overlap.  When the
    bands overlap the answer is ``faster=False`` with an ``inconclusive`` reason
    even if the medians moved -- overlapping spreads are exactly the noise case
    this harness exists to refuse.
    """
    before_run = _coerce_run("before", before)
    after_run = _coerce_run("after", after)
    effect = _require_min_effect(min_effect)
    _require_overlap_rule(overlap_rule)
    if not before_run.times_ms or not after_run.times_ms:
        raise BenchError("compare requires runs with at least one timing sample each")

    before_median = before_run.stat("median_ms")
    after_median = after_run.stat("median_ms")
    if before_median <= 0.0:
        raise BenchError(
            f"before median must be positive to form a ratio, got {before_median}"
        )

    ratio = after_median / before_median
    delta_ms = after_median - before_median
    threshold = 1.0 - effect
    before_band = list(before_run.iqr_bounds())
    after_band = list(after_run.iqr_bounds())
    overlaps = _bands_overlap(before_band, after_band)
    faster = bool(ratio <= threshold and not overlaps)

    if faster:
        reason = (
            f"faster: ratio {ratio:.4f} <= {threshold:.4f} and IQR bands separate"
        )
    elif overlaps:
        reason = (
            "inconclusive: IQR bands overlap "
            f"(before [{before_band[0]:.4f}, {before_band[1]:.4f}], "
            f"after [{after_band[0]:.4f}, {after_band[1]:.4f}])"
        )
    else:
        reason = f"not faster: ratio {ratio:.4f} > threshold {threshold:.4f}"

    return Verdict(
        faster=faster,
        ratio=ratio,
        delta_ms=delta_ms,
        reason=reason,
        bands={"before_iqr": before_band, "after_iqr": after_band},
        min_effect=effect,
    )


# --------------------------------------------------------------------------- #
# receipting
# --------------------------------------------------------------------------- #


def _open_receipt_target(target: Any) -> Tuple[Any, bool]:
    """Return ``(sink, owned)``; ``owned`` means this module must close it."""
    if isinstance(target, Database):
        if target.closed:
            raise BenchError("database is closed; pass an open Database or a path")
        return target, False
    if isinstance(target, sqlite3.Connection):
        return target, False
    if isinstance(target, (str, Path)):
        if not str(target).strip():
            raise BenchError("database path must not be empty")
        try:
            return Database(target, create=True), True
        except StorageError as exc:
            raise BenchError(f"cannot open database {target}: {exc}") from exc
    raise BenchError(
        "target must be a toothpick Database, a sqlite3 connection, or a db path, "
        f"got {type(target).__name__}"
    )


def _add_bench_receipt(sink: Any, binary_sha256: str, payload: Dict[str, Any]) -> str:
    if isinstance(sink, Database):
        try:
            return sink.add_receipt(
                binary_sha256, FUNC_ADDR_NONE_SENTINEL, BENCH_PHASE, payload
            )
        except StorageError as exc:
            raise BenchError(f"cannot write bench receipt: {exc}") from exc
    payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    receipt_id = make_receipt_id(
        binary_sha256, FUNC_ADDR_NONE_SENTINEL, BENCH_PHASE, payload_json
    )
    try:
        sink.execute(
            _RECEIPT_UPSERT_SQL,
            (
                receipt_id,
                binary_sha256,
                FUNC_ADDR_NONE_SENTINEL,
                BENCH_PHASE,
                _utc_now(),
                payload_json,
            ),
        )
        sink.commit()
    except sqlite3.Error as exc:
        raise BenchError(f"cannot write bench receipt: {exc}") from exc
    return receipt_id


def record_claim(
    target: Union[Database, sqlite3.Connection, PathLike],
    sha: Any,
    label: str,
    verdict: Union[Verdict, Mapping[str, Any]],
) -> str:
    """Persist one bench claim as a receipt and return its receipt id.

    The row is ``phase="bench"`` at ``func_addr=-1`` -- "not attached to any
    function" (``docs/traps.md`` 23) -- with ``func_addr: null`` preserved in the
    payload, and it always carries
    ``verifier_status="UNVERIFIED (stability)"``: a wall-clock delta is not
    grounding evidence, and the label says so where a reader will find it.

    ``target`` may be an open :class:`~toothpick.storage.Database`, a
    :class:`sqlite3.Connection`, or a db path (a path is opened ``create=True``
    and closed here in a ``finally`` -- trap 7).  A caller-supplied handle is
    never closed by this function.  The receipt id is deterministic
    (:func:`~toothpick.storage.make_receipt_id`), so re-recording the identical
    claim updates that row instead of appending a duplicate.
    """
    binary_sha256 = _require_sha(sha)
    claim_label = _require_label(label)
    body = _coerce_verdict("verdict", verdict).as_dict()

    payload: Dict[str, Any] = {
        "label": claim_label,
        "func_addr": None,
        "verifier_status": VERIFIER_STATUS_STABILITY,
    }
    payload.update(body)

    sink, owned = _open_receipt_target(target)
    try:
        return _add_bench_receipt(sink, binary_sha256, payload)
    finally:
        if owned:
            sink.close()
