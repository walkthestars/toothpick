#!/usr/bin/env python3
"""Toothpick Coverage Floor Gate.

Runs the verification battery under coverage instrumentation, then enforces the
two-level floor agreed for the new-track modules:

1. ``TOTAL`` coverage of the measured tree must be >= 80%.
2. Every new-track module listed in :data:`FLOOR_MODULES` must be >= 75%.

``src/toothpick/mcp_server.py`` is *grandfathered*: its percentage is reported
but never fails the gate, because its uncovered lines sit in the pre-existing
SDK-path gap rather than in missing new-track tests.

Stdlib only.  The coverage tool is driven through ``sys.executable -m coverage``
so this gate needs no dependency of its own and behaves identically whether it
is launched as ``uv run python scripts/coverage_gate.py`` (the CI step) or as a
plain ``python scripts/coverage_gate.py``.
"""

from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

TOTAL_FLOOR = 80.0
MODULE_FLOOR = 75.0
SOURCE_DIR = "src/toothpick"

#: New-track modules that must each clear :data:`MODULE_FLOOR`.
FLOOR_MODULES: Tuple[str, ...] = (
    "jvm.py",
    "jvm_parse.py",
    "rtti_parse.py",
    "rtti_store.py",
    "rtti.py",
    "profile.py",
    "profile_jfr.py",
    "hook.py",
    "bench.py",
    "lua_mod.py",
)

#: Modules exempt from :data:`MODULE_FLOOR`, reported only.
GRANDFATHERED_MODULES: Tuple[str, ...] = ("mcp_server.py",)

GRANDFATHER_NOTE = (
    "GRANDFATHERED at any level: its uncovered lines are the pre-existing "
    "SDK-path gap (pyghidra / Ghidra and IDA bindings are only importable and "
    "reachable when a live SDK is present), which predates this track -- it is "
    "reported and counted in TOTAL, but no floor is enforced for it."
)

#: A coverage "Cover" cell: ``85%`` or ``85.5%``.
_PERCENT_CELL = re.compile(r"^\d{1,3}(?:\.\d+)?%$")

_REPORT_CMDS: List[Tuple[str, List[str]]] = [
    ("run", ["-m", "coverage", "run", "-m", "pytest", "-p", "no:faulthandler", "-q"]),
    ("report", ["-m", "coverage", "report"]),
]


def _run(cmd: List[str], cwd: Path) -> Tuple[int, str]:
    """Execute one coverage sub-command and return (returncode, output)."""
    res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return res.returncode, (res.stdout + "\n" + res.stderr).strip()


def _parse_report(report: str) -> Tuple[Dict[str, float], Optional[float]]:
    """Parse ``coverage report`` rows into ({path: percent}, TOTAL percent).

    Layout-agnostic on purpose: the percentage is located as the first
    ``NN%``-shaped cell that has at least ``Stmts`` and ``Miss`` to its left, so
    optional ``Branch``/``BrPart`` and ``Missing`` columns do not shift the
    reading.  Header, separator and warning lines carry no such cell and drop
    out.
    """
    rows: Dict[str, float] = {}
    total: Optional[float] = None
    for raw in report.splitlines():
        tokens = raw.split()
        percent_index: Optional[int] = None
        for index, token in enumerate(tokens):
            if _PERCENT_CELL.match(token):
                percent_index = index
                break
        # Need at least: <name> <stmts> <miss> <percent> ...
        if percent_index is None or percent_index < 3:
            continue
        percent = float(tokens[percent_index][:-1])
        name = tokens[0].replace("\\", "/")
        if name == "TOTAL":
            total = percent
        else:
            rows[name] = percent
    return rows, total


def _lookup(rows: Dict[str, float], module: str) -> Optional[float]:
    """Return the measured percentage for *module*, or ``None`` if absent."""
    wanted = f"{SOURCE_DIR}/{module}"
    if wanted in rows:
        return rows[wanted]
    for name, percent in rows.items():
        if name == module or name.endswith(f"/{module}"):
            return percent
    return None


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent

    print("======================================================================")
    print("TOOTHPICK COVERAGE FLOOR GATE")
    print(f"Working Directory: {repo_root}")
    print(
        f"Floors: TOTAL >= {TOTAL_FLOOR:.0f}%, "
        f"new-track modules >= {MODULE_FLOOR:.0f}% each"
    )
    print("======================================================================")

    outputs: Dict[str, str] = {}
    for label, argv in _REPORT_CMDS:
        cmd = [sys.executable, *argv]
        print(f"\n[coverage-gate] >>> coverage {label} ({' '.join(cmd)})")
        returncode, output = _run(cmd, repo_root)
        outputs[label] = output
        if returncode != 0:
            print("----------------------------------------------------------------------")
            print(output)
            print("----------------------------------------------------------------------")
            print(
                f"[coverage-gate] FAIL: 'coverage {label}' exited with code "
                f"{returncode}"
            )
            return 1
        if label == "report":
            print(output)

    rows, total = _parse_report(outputs["report"])
    if not rows or total is None:
        print("[coverage-gate] FAIL: could not parse a TOTAL row from 'coverage report'")
        print("----------------------------------------------------------------------")
        print(outputs["report"])
        print("----------------------------------------------------------------------")
        return 1

    violations: List[str] = []
    if total < TOTAL_FLOOR:
        violations.append(f"TOTAL {total:.2f}% < {TOTAL_FLOOR:.0f}%")

    print("----------------------------------------------------------------------")
    print("NEW-TRACK MODULE FLOOR CHECK")
    print("----------------------------------------------------------------------")
    for module in FLOOR_MODULES:
        percent = _lookup(rows, module)
        if percent is None:
            violations.append(f"{SOURCE_DIR}/{module}: not measured at all")
            print(f"  [MISS] {module:<16} not present in coverage report")
            continue
        if percent < MODULE_FLOOR:
            violations.append(f"{SOURCE_DIR}/{module}: {percent:.2f}% < {MODULE_FLOOR:.0f}%")
            print(f"  [FAIL] {module:<16} {percent:>6.2f}%  (< {MODULE_FLOOR:.0f}%)")
        else:
            print(f"  [ OK ] {module:<16} {percent:>6.2f}%")

    print("----------------------------------------------------------------------")
    print(f"  [TOTAL] {total:.2f}% (floor {TOTAL_FLOOR:.0f}%)")
    for module in GRANDFATHERED_MODULES:
        percent = _lookup(rows, module)
        shown = f"{percent:.2f}%" if percent is not None else "not measured"
        print(f"  [NOTE] {module:<16} {shown:>8}  {GRANDFATHER_NOTE}")
    print("----------------------------------------------------------------------")

    if violations:
        print("[coverage-gate] FAIL: coverage floor violated")
        for violation in violations:
            print(f"  - {violation}")
        return 1

    print(
        f"[coverage-gate] PASS: TOTAL {total:.2f}% >= {TOTAL_FLOOR:.0f}% and all "
        f"{len(FLOOR_MODULES)} new-track modules >= {MODULE_FLOOR:.0f}%"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
