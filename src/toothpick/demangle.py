"""
Toothpick full MSVC demangling via ``undname.exe`` (UNIT-SEM5 Chunk 2).

Binding contract: ``docs/units/UNIT-SEM5-PLAN.md`` chunk 2 (undname demangling)
plus chunk 1 of the same unit for the surrounding discipline.  Design anchors:
``docs/design-document.md`` section 1 ("no fabricated structs -- every class row
traces to parsed bytes") and section 19 (a label is only ever what was proven,
never an invented link).

Two entry points:

* :func:`full_demangle` -- the *full* path.  The leading ``.`` of an RTTI string
  is stripped and ``undname.exe`` (Visual Studio's C++ Name Undecorator) is run
  over the result under a hard timeout.  The tool's ``is :-`` output line is
  parsed into the readable name.  Every way this can fail yields ``None``:
  no ``undname.exe`` (no ``%VCToolsInstallDir%`` and no well-known BuildTools /
  Community install), a non-Windows host, a timeout, a launch error, a
  non-zero exit, unparsable output, or output that merely echoes the input back.
* :func:`demangle_best` -- the full path, else the existing strip fallback
  :func:`toothpick.rtti_parse.demangle_msvc_type` (*imported*, never duplicated),
  returned with the same ``(name, demangled)`` shape so it is a drop-in
  replacement for the strip-only helper.

Epistemic rules enforced here:

* ``undname.exe`` echoes its argument when it cannot decode it.  Returned text
  must therefore never equal the symbol that was submitted; an echo is treated
  as *not decoded* and ``None`` is returned instead of presenting raw decorated
  bytes as a name.
* The name is taken verbatim from the ``is :-`` line, with one normalisation:
  the `` `RTTI Type Descriptor'`` tail of the descriptor symbol form is
  removed, because that label describes *why* the symbol exists and not the
  type.  The MSVC type keyword (``class`` / ``struct`` / ``enum``) is kept, so
  ``.?AVCGameApplication@@`` yields ``class CGameApplication``.
* Nothing is guessed: no name is synthesised from the mangling grammar, and no
  result is produced from a transcript that does not contain an ``is :-`` line.

Pure and offline: the only side effect is one sandboxed subprocess running a
local Microsoft tool over a string argument.  No network access, no shell
(``shell`` is never enabled; the argv list is passed straight to
:func:`subprocess.run`), no file written, and the module reads nothing but
directory listings during executable discovery.

Trap hygiene: this module never imports ``faulthandler`` (Trap 11 -- suites run
under ``-p no:faulthandler``) and it opens no SQLite handle, so the Trap 7
``[WinError 32]`` class of leak cannot originate here.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

from toothpick.rtti_parse import demangle_msvc_type

__all__ = [
    "DEMANGLE_TIMEOUT_SECONDS",
    "RTTI_DESCRIPTOR_SUFFIX",
    "TOOTHPICK_UNDNAME_ENV",
    "VCTOOLS_ENV",
    "DemangleError",
    "demangle_best",
    "find_undname",
    "full_demangle",
]

#: Hard wall-clock budget for one ``undname.exe`` invocation, in seconds.
DEMANGLE_TIMEOUT_SECONDS = 30.0

#: Environment variable holding ``...\VC\Tools\MSVC\<version>\`` (set by a
#: Developer Command Prompt); the primary hint for executable discovery.
VCTOOLS_ENV = "VCToolsInstallDir"

#: Explicit override for the executable path (mirrors ``TOOTHPICK_CDB`` in
#: :mod:`toothpick.windbg`: an operator who names the tool wins over discovery).
TOOTHPICK_UNDNAME_ENV = "TOOTHPICK_UNDNAME"

#: The trailing label ``undname.exe`` appends to a ``??_R0...@8`` RTTI
#: descriptor symbol; stripped so the type name itself is what is returned.
RTTI_DESCRIPTOR_SUFFIX = "`RTTI Type Descriptor'"

#: Executable basename looked for underneath every candidate directory.
UNDNAME_EXE = "undname.exe"

#: The marker introducing the decoded payload of one ``undname.exe`` result.
_IS_LINE_MARKER = "is :-"

#: Canonical produced-symbol shape of an RTTI type descriptor (`.?AVFoo@@`
#: lives in the image as `??_R0?AVFoo@@@8`); ``undname.exe`` only expands the
#: class name from the *symbol*, not from the bare type string.
_DESCRIPTOR_SYMBOL_PREFIX = "??_R0"
_DESCRIPTOR_SYMBOL_SUFFIX = "@8"

#: MSVC type-string prefixes that denote an RTTI type descriptor body.
_RTTI_TYPE_NAME = re.compile(r"^\?A[VUW]")

#: Host/target subdirectories under ``<tools>\bin``, most-preferred first.
_HOST_SUBDIRS: Tuple[Tuple[str, str], ...] = (
    ("Hostx64", "x64"),
    ("Hostx64", "x86"),
    ("Hostx86", "x64"),
    ("Hostx86", "x86"),
)

#: Visual Studio editions scanned, in a fixed order (BuildTools is the
#: headless CI install; Community/Professional/Enterprise are the desktop ones).
_VS_EDITIONS: Tuple[str, ...] = ("BuildTools", "Community", "Professional", "Enterprise")

#: Visual Studio release years scanned, newest first.
_VS_YEARS: Tuple[str, ...] = ("2022", "2019", "2017")

#: An MSVC toolset directory name is one or more dotted numeric components.
_VERSION_NAME = re.compile(r"^\d+(?:\.\d+)*$")


class DemangleError(ValueError):
    """Raised for a caller argument that no MSVC mangling could ever satisfy."""


def _is_windows() -> bool:
    """True on a Windows host -- the only host ``undname.exe`` is shipped for."""
    return os.name == "nt"


def _resolve_timeout(value: float) -> float:
    """Validate a caller-supplied timeout budget (seconds, strictly positive)."""
    try:
        budget = float(value)
    except (TypeError, ValueError) as exc:
        raise DemangleError(f"timeout must be a number, got {value!r}") from exc
    if not budget > 0:
        raise DemangleError(f"timeout must be positive, got {value!r}")
    return budget


def _version_key(name: str) -> Tuple[int, ...]:
    """Numeric sort key for an MSVC toolset directory name (``14.44.35207``)."""
    try:
        return tuple(int(part) for part in name.split("."))
    except ValueError:  # pragma: no cover - guarded by _VERSION_NAME
        return ()


def _msvc_version_dirs(tools_root: Path) -> List[Path]:
    """Toolset directories under ``<...>\\VC\\Tools\\MSVC``, newest first."""
    try:
        entries = [
            entry
            for entry in tools_root.iterdir()
            if entry.is_dir() and _VERSION_NAME.match(entry.name)
        ]
    except OSError:
        return []
    return sorted(entries, key=lambda entry: _version_key(entry.name), reverse=True)


def _vctools_candidates() -> List[Path]:
    """Candidate executables from ``%VCToolsInstallDir%`` (empty when unset)."""
    raw = os.environ.get(VCTOOLS_ENV, "").strip()
    if not raw:
        return []
    root = Path(raw)
    return [
        root / "bin" / host / target / UNDNAME_EXE for host, target in _HOST_SUBDIRS
    ]


def _well_known_candidates() -> List[Path]:
    """Candidate executables under the conventional Visual Studio install roots."""
    out: List[Path] = []
    for variable in ("ProgramFiles(x86)", "ProgramFiles"):
        base = os.environ.get(variable, "").strip()
        if not base:
            continue
        root = Path(base) / "Microsoft Visual Studio"
        for year in _VS_YEARS:
            for edition in _VS_EDITIONS:
                tools = root / year / edition / "VC" / "Tools" / "MSVC"
                for version in _msvc_version_dirs(tools):
                    for host, target in _HOST_SUBDIRS:
                        out.append(version / "bin" / host / target / UNDNAME_EXE)
    return out


def find_undname() -> Optional[str]:
    """Locate ``undname.exe``, or ``None`` when no candidate exists on this host.

    Discovery order is deterministic: the :data:`TOOTHPICK_UNDNAME_ENV` override,
    then ``%VCToolsInstallDir%\\bin\\Host<host>\\<target>\\undname.exe`` for each
    host/target pair, then the well-known
    ``%ProgramFiles(x86)%`` / ``%ProgramFiles%`` ``Microsoft Visual Studio``
    layouts (years 2022/2019/2017, editions BuildTools/Community/Professional/
    Enterprise, newest toolset first).  The function is pure discovery -- it does
    not care about the host platform; :func:`full_demangle` owns that gate.
    """
    override = os.environ.get(TOOTHPICK_UNDNAME_ENV, "").strip()
    if override:
        candidate = Path(override)
        if candidate.is_file():
            return str(candidate)
    for candidate in _vctools_candidates() + _well_known_candidates():
        try:
            if candidate.is_file():
                return str(candidate)
        except OSError:  # pragma: no cover - unreadable path component
            continue
    return None


def _symbol_forms(symbol: str) -> List[str]:
    """Argument forms to try, best first, for one dot-stripped symbol.

    An RTTI type body (``?AVCGameApplication@@``) is not a complete decorated
    symbol, so ``undname.exe`` cannot expand it on its own; the descriptor symbol
    it was emitted from (``??_R0?AVCGameApplication@@@8``) is tried first, and the
    bare symbol is the fallback.
    """
    forms: List[str] = []
    if _RTTI_TYPE_NAME.match(symbol):
        forms.append(f"{_DESCRIPTOR_SYMBOL_PREFIX}{symbol}{_DESCRIPTOR_SYMBOL_SUFFIX}")
    forms.append(symbol)
    return forms


def _parse_undname_output(stdout: str) -> Optional[str]:
    """Extract the decoded name from one ``undname.exe`` transcript.

    The first ``is :-`` line is authoritative (the tool prints exactly one per
    argument).  Its payload is optionally quote-wrapped; the RTTI descriptor
    label is then removed.  ``None`` means "no decodable payload", never an
    empty name.
    """
    if not isinstance(stdout, str):
        return None
    for line in stdout.splitlines():
        index = line.find(_IS_LINE_MARKER)
        if index < 0:
            continue
        payload = line[index + len(_IS_LINE_MARKER) :].strip()
        if len(payload) >= 2 and payload.startswith('"') and payload.endswith('"'):
            payload = payload[1:-1].strip()
        if payload.endswith(RTTI_DESCRIPTOR_SUFFIX):
            payload = payload[: -len(RTTI_DESCRIPTOR_SUFFIX)].strip()
        return payload or None
    return None


def _run_undname(executable: str, symbol: str, timeout: float) -> Optional[str]:
    """Run ``undname.exe`` over one symbol argument under *timeout*.

    Returns the captured standard output, or ``None`` when the tool timed out,
    could not be launched, or exited non-zero.  ``shell`` is never enabled and
    standard input is closed, so the child cannot block on a prompt.
    """
    creationflags = 0
    if _is_windows() and hasattr(subprocess, "CREATE_NO_WINDOW"):
        creationflags = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
    try:
        completed = subprocess.run(
            [executable, symbol],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
            creationflags=creationflags,
        )
    except subprocess.TimeoutExpired:
        return None
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode:
        return None
    return completed.stdout or ""


def full_demangle(
    mangled: str,
    *,
    executable: Optional[str] = None,
    timeout: float = DEMANGLE_TIMEOUT_SECONDS,
) -> Optional[str]:
    """Demangle *mangled* with ``undname.exe``, or return ``None``.

    The leading ``.`` of an RTTI string is stripped, the symbol is handed to
    ``undname.exe`` under a 30-second default timeout, and the ``is :-`` output
    line is parsed into the readable name (``.?AVCGameApplication@@`` ->
    ``class CGameApplication``, templates expanded: ``.?AV?$vector@...`` ->
    ``class vector<class Foo,class std::allocator<class Foo> >``).

    ``None`` -- never a guess and never an exception -- for every failure: a
    non-string is the caller's bug and raises :class:`DemangleError`, but an
    empty/degenerate string, a non-Windows host without an explicit
    ``executable``, a missing ``undname.exe``, a timeout, a launch error, a
    non-zero exit, missing ``is :-`` output, or an echoed-back argument all
    yield ``None``.

    ``executable`` overrides discovery and is honoured on any platform (that is
    what keeps the synthetic battery in ``tests/test_demangle.py`` deterministic
    off Windows); ``timeout`` is the per-invocation budget in seconds.  Callers
    demangling many names in one pass should resolve the executable once and
    pass it in.
    """
    if not isinstance(mangled, str):
        raise DemangleError(f"full_demangle expects str, got {type(mangled).__name__}")
    symbol = mangled.strip()
    if symbol.startswith("."):
        symbol = symbol[1:].strip()
    if not symbol:
        return None

    if executable is None:
        if not _is_windows():
            return None
        executable = find_undname()
        if not executable:
            return None

    budget = _resolve_timeout(timeout)
    for form in _symbol_forms(symbol):
        try:
            transcript = _run_undname(executable, form, budget)
        except (subprocess.TimeoutExpired, OSError, subprocess.SubprocessError):
            return None
        if transcript is None:
            continue
        name = _parse_undname_output(transcript)
        if name is None or name == form:
            # No payload, or the tool echoed the argument undecoded: not a name.
            continue
        return name
    return None


def demangle_best(
    mangled: str,
    *,
    executable: Optional[str] = None,
    timeout: float = DEMANGLE_TIMEOUT_SECONDS,
) -> Tuple[str, bool]:
    """Best available name for *mangled*: full demangle, else the strip fallback.

    Returns ``(name, demangled)`` -- the same shape as
    :func:`toothpick.rtti_parse.demangle_msvc_type`, which supplies the fallback
    here (imported, not re-implemented), so this is a drop-in replacement for the
    strip-only helper.  When :func:`full_demangle` produces a name the flag is
    ``True``; otherwise the ``(raw, False)``/``(stripped, True)`` verdict of the
    fallback is returned unchanged and the raw string is never dressed up as a
    recovered name.
    """
    full = full_demangle(mangled, executable=executable, timeout=timeout)
    if full:
        return full, True
    return demangle_msvc_type(mangled)
