#!/usr/bin/env python3
"""Toothpick CI Verification Runner.

Single-command CI gate executing the complete verification harness:
1. Code Linting (ruff check .)
2. Distribution Build (uv build)
3. Pytest Verification Suite (uv run pytest -p no:faulthandler)
4. Coverage Floor (new-track modules >= 75%, total >= 80%)
5. Dependency Licensing & Hygiene Audit
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import time
from typing import List, Tuple


def _run_step(name: str, cmd: List[str], cwd: Path) -> Tuple[bool, float, str]:
    """Execute a single CI step with timing and stdout capture."""
    print(f"\n[CI] >>> Running: {name} ({' '.join(cmd)})")
    t0 = time.perf_counter()
    res = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - t0

    output = (res.stdout + "\n" + res.stderr).strip()
    if res.returncode == 0:
        print(f"[CI] [PASS] {name} in {elapsed:.2f}s")
        return True, elapsed, output
    else:
        print(f"[CI] [FAIL] {name} exited with code {res.returncode} in {elapsed:.2f}s")
        print("----------------------------------------------------------------------")
        print(output)
        print("----------------------------------------------------------------------")
        return False, elapsed, output


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    t_start = time.perf_counter()

    print("======================================================================")
    print("TOOTHPICK CONTINUOUS INTEGRATION HARNESS")
    print(f"Working Directory: {repo_root}")
    print("======================================================================")

    steps = [
        ("Linting & Code Hygiene", ["uv", "run", "ruff", "check", "."]),
        ("Distribution Packaging Build", ["uv", "build"]),
        ("Test Battery & Verification", ["uv", "run", "pytest", "-p", "no:faulthandler"]),
        (
            "Coverage Floor (new-track modules >=75, total >=80)",
            ["uv", "run", "python", "scripts/coverage_gate.py"],
        ),
        ("License Conformance Audit", ["uv", "run", "pytest", "tests/test_license_audit.py"]),
    ]

    all_passed = True
    results: List[Tuple[str, bool, float]] = []

    for name, cmd in steps:
        ok, elapsed, _ = _run_step(name, cmd, repo_root)
        results.append((name, ok, elapsed))
        if not ok:
            all_passed = False
            break

    total_time = time.perf_counter() - t_start

    print("\n======================================================================")
    print("CI SUMMARY")
    print("======================================================================")
    for name, ok, elapsed in results:
        status_str = "PASS [OK]" if ok else "FAIL [XX]"
        print(f"{status_str:<12} | {name:<35} | {elapsed:.2f}s")

    print("----------------------------------------------------------------------")
    print(f"Total Duration: {total_time:.2f}s")

    if all_passed:
        print("[CI] ALL GATES PASSED CLEANLY.")
        return 0
    else:
        print("[CI] CI FAILED. Please resolve errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
