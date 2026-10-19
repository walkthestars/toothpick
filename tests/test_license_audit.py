"""Automated dependency license conformance and hygiene audit.

Governed by design-document.md §15 (Third-Party Dependency & Licensing Policy)
and execution-plan.md §4 (CI/Verification specification).

Enforces:
1. 100% of dependencies (direct and transitive) use permissive OSI-approved licenses
   (MIT, Apache-2.0, BSD, PSF, Public Domain, wxWindows Library Licence).
2. No GPL, AGPL, LGPL, MPL, CDDL, EPL, or other copyleft/commercial taints.
3. No bundled proprietary tools, IDA/Ghidra binaries, or closed SDK blobs under src/, docs/, or scripts/.
"""

from __future__ import annotations

import importlib.metadata as im
from pathlib import Path
import re
from typing import List, Set

# Permissive license tokens
PERMISSIVE_PATTERNS = [
    re.compile(r"\bmit(-0)?\b", re.IGNORECASE),
    re.compile(r"\bapache\b", re.IGNORECASE),
    re.compile(r"\bbsd\b", re.IGNORECASE),
    re.compile(r"\bpsf(-2\.0)?\b", re.IGNORECASE),
    re.compile(r"python software foundation", re.IGNORECASE),
    re.compile(r"public\s*domain", re.IGNORECASE),
    re.compile(r"wxwindows", re.IGNORECASE),
    re.compile(r"unlicense", re.IGNORECASE),
    re.compile(r"cc0", re.IGNORECASE),
]

# Forbidden copyleft / proprietary / restrictive patterns
FORBIDDEN_PATTERNS = [
    re.compile(r"\b(a|l)?gpl\b", re.IGNORECASE),
    re.compile(r"gnu\s+(general|affero|lesser)", re.IGNORECASE),
    re.compile(r"\bmpl\b", re.IGNORECASE),
    re.compile(r"mozilla\s+public", re.IGNORECASE),
    re.compile(r"\bcddl\b", re.IGNORECASE),
    re.compile(r"\bepl\b", re.IGNORECASE),
    re.compile(r"eclipse\s+public", re.IGNORECASE),
    re.compile(r"proprietary", re.IGNORECASE),
    re.compile(r"commercial", re.IGNORECASE),
    re.compile(r"sspl", re.IGNORECASE),
]


def _extract_license_text(dist: im.Distribution) -> str:
    """Extract combined license text, expression, and classifiers from distribution metadata."""
    meta = dist.metadata
    parts: List[str] = []
    expr = meta.get("License-Expression")
    if expr:
        parts.append(str(expr))
    lic = meta.get("License")
    if lic:
        parts.append(str(lic))
    classifiers = meta.get_all("Classifier") or []
    for c in classifiers:
        if "License" in c:
            parts.append(str(c))
    return " | ".join(parts)


def test_dependency_licenses_are_strictly_permissive() -> None:
    """Assert that every installed dependency in the project environment has a permissive license."""
    distributions = list(im.distributions())
    assert len(distributions) > 0, "No installed distributions detected in environment"

    violations: List[str] = []
    for dist in distributions:
        name = dist.metadata.get("Name")
        if not name or name.lower() == "toothpick":
            continue

        lic_str = _extract_license_text(dist)
        if not lic_str:
            violations.append(f"{name}: has no license metadata found")
            continue

        # wxWindows Library Licence is an OSI-approved permissive exception to LGPL
        is_wxwindows = "wxwindows" in lic_str.lower()

        # Check for forbidden copyleft / restrictive licenses
        if not is_wxwindows:
            forbidden_matches = [p.pattern for p in FORBIDDEN_PATTERNS if p.search(lic_str)]
            if forbidden_matches:
                violations.append(
                    f"{name}: contains forbidden license pattern ({forbidden_matches}) in: '{lic_str}'"
                )
                continue

        # Check that at least one permissive pattern matches
        is_permissive = any(p.search(lic_str) for p in PERMISSIVE_PATTERNS)
        if not is_permissive:
            violations.append(
                f"{name}: no permissive license pattern matched in: '{lic_str}'"
            )

    assert not violations, "License audit violations found:\n" + "\n".join(violations)


def test_no_bundled_proprietary_tools_or_binaries() -> None:
    """Assert that the repository contains no forbidden binary blobs, compiled DLLs/EXEs,

    or proprietary tool drops under src/, docs/, or scripts/.
    """
    repo_root = Path(__file__).resolve().parent.parent
    scan_dirs = [repo_root / "src", repo_root / "docs", repo_root / "scripts"]

    forbidden_extensions = {".exe", ".dll", ".so", ".dylib", ".sys"}
    found_binaries: Set[str] = set()

    for s_dir in scan_dirs:
        if not s_dir.exists():
            continue
        for path in s_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in forbidden_extensions:
                found_binaries.add(str(path.relative_to(repo_root)))

    assert not found_binaries, (
        f"Found forbidden precompiled binaries in source tree: {found_binaries}"
    )
