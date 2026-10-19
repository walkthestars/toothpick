"""Tests for UNIT-SEM5 Chunk 2 (`docs/units/UNIT-SEM5-PLAN.md`): undname demangling.

Binding contract: chunk 2 of the unit plan (`src/toothpick/demangle.py`,
`full_demangle` via `undname.exe` with `%VCToolsInstallDir%` plus well-known
BuildTools/Community discovery, 30-second timeout, Windows-only, `None` when the
tool is absent or failing, the `is :-` output line parsed as the name; and
`demangle_best` = full path else the existing `rtti_parse.demangle_msvc_type`
strip fallback, imported and never duplicated).  Design SS1 / SS19: a returned
name is only ever what `undname.exe` actually printed -- never a name synthesised
from the mangling grammar.

Battery:

1. `_parse_undname_output` over captured-shape transcripts (quoted payload,
   unquoted payload, `` `RTTI Type Descriptor'`` tail, blank payload, no
   `is :-` line at all, first-marker-wins, non-string).
2. `_symbol_forms` -- the `??_R0<type>@8` descriptor symbol is tried before the
   bare `?AV...`/`?AU...`/`?AW4...` body (the bare body is not a complete
   decorated symbol and `undname.exe` only echoes it).
3. Discovery (`find_undname`): `TOOTHPICK_UNDNAME` override wins,
   `%VCToolsInstallDir%` is honoured, the well-known Program Files layout is
   scanned newest toolset first, and nothing installed yields `None`.
4. Invocation without any Microsoft tool: a synthetic shim executable (a `.bat`
   calling `sys.executable` on Windows, a shebang script elsewhere) is passed as
   `executable=`, so `class CGameApplication`, `struct ...`, `enum ...` and the
   `std::vector` template expansion are asserted deterministically on any host.
5. Failure honesty: missing executable, launch error, non-zero exit, timeout
   (both injected `TimeoutExpired` and a real hanging shim), `is :- ""`,
   markerless garbage, and an echoed-back argument all yield `None`.
6. `demangle_best` fallback: the full path wins when it works, otherwise the
   result is byte-for-byte the `toothpick.rtti_parse.demangle_msvc_type` verdict
   (the helper is imported -- identity is asserted -- not re-implemented).
7. Hygiene: no `faulthandler`, no network module, no `shell=True` in the module
   source, and a 30-second default timeout.

No real binary, no real `undname.exe` and no network are required: tests that
need the Microsoft tool are skipped when it is not installed.  The suite runs
under `-p no:faulthandler` (pyproject) and opens no SQLite handle, so no
`[WinError 32]` teardown leak can originate here.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from toothpick import demangle
from toothpick import rtti_parse
from toothpick.demangle import (
    DEMANGLE_TIMEOUT_SECONDS,
    RTTI_DESCRIPTOR_SUFFIX,
    TOOTHPICK_UNDNAME_ENV,
    VCTOOLS_ENV,
    DemangleError,
    demangle_best,
    find_undname,
    full_demangle,
)
from toothpick.rtti_parse import demangle_msvc_type

# The real Microsoft tool, only when a Windows host has Visual Studio installed.
REAL_UNDNAME = find_undname() if os.name == "nt" else None
_needs_real_undname = pytest.mark.skipif(
    REAL_UNDNAME is None,
    reason="undname.exe is not installed on this host (synthetic battery still runs)",
)

# --- synthetic undname.exe stand-in ----------------------------------------- #

_SHIM_SOURCE = '''"""Synthetic undname.exe stand-in: no Microsoft tool is invoked."""
import sys
import time

MODE = "__MODE__"

BANNER = (
    "Microsoft (R) C++ Name Undecorator\\n"
    "Copyright (C) Microsoft Corporation. All rights reserved.\\n"
    "\\n"
)

SUFFIX = " `RTTI Type Descriptor'"

NAMES = {
    "??_R0?AVCGameApplication@@@8": "class CGameApplication",
    "??_R0?AUCGameApplication@@@8": "struct CGameApplication",
    "??_R0?AW4EColor@@@8": "enum EColor",
    "??_R0?AV?$vector@VFoo@@V?$allocator@VFoo@@@std@@@std@@@8": (
        "class std::vector<class Foo,class std::allocator<class Foo> >"
    ),
}


def transcript(argument):
    """The output text this shim prints for *argument*."""
    if MODE == "garbage":
        return BANNER + 'Undecoration of :- "%s"\\n' % argument
    if MODE == "blank":
        return BANNER + 'is :- ""\\n'
    if MODE == "echo":
        return BANNER + 'is :- "%s"\\n' % argument
    name = NAMES.get(argument)
    if MODE == "descriptor" and name is None:
        # A bare RTTI type body is not a complete symbol: echo it back.
        name = argument
    if name is None:
        name = "class CGameApplication"
    if argument.startswith("??_R0"):
        name = name + SUFFIX
    return BANNER + 'Undecoration of :- "%s"\\nis :- "%s"\\n\\n' % (argument, name)


def main():
    if MODE == "hang":
        time.sleep(1.5)
        return 0
    if MODE == "exit3":
        sys.stdout.write(BANNER)
        return 3
    sys.stdout.write(transcript(sys.argv[1] if len(sys.argv) > 1 else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''

_PROBE_SYMBOL = "?AVProbe@@"
_PROBED: dict = {}


@pytest.fixture(scope="module")
def shim_factory(tmp_path_factory):
    """Build (once per mode) an executable shim standing in for undname.exe."""
    root = tmp_path_factory.mktemp("undname-shim")
    cache: dict = {}

    def build(mode: str) -> str:
        if mode in cache:
            return cache[mode]
        handler = root / ("handler_%s.py" % mode)
        handler.write_text(_SHIM_SOURCE.replace("__MODE__", mode), encoding="utf-8")
        if os.name == "nt":
            shim = root / ("shim_%s.bat" % mode)
            shim.write_text(
                '@echo off\r\n"%s" "%s" %%*\r\n' % (sys.executable, handler),
                encoding="utf-8",
            )
        else:
            shim = root / ("shim_%s" % mode)
            shim.write_text(
                '#!/bin/sh\nexec "%s" "%s" "$@"\n' % (sys.executable, handler),
                encoding="utf-8",
            )
            shim.chmod(0o755)
        cache[mode] = str(shim)
        return cache[mode]

    return build


def _shim(shim_factory, mode: str) -> str:
    """Path to the shim for *mode*, skipped if the host cannot execute it."""
    path = shim_factory(mode)
    if mode == "hang":
        return path  # probing it would sleep; the timeout test owns it
    if not _PROBED.get(mode):
        try:
            probe = subprocess.run(
                [path, _PROBE_SYMBOL],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover
            pytest.skip("host cannot execute the synthetic undname shim: %s" % exc)
        if not probe.stdout:  # pragma: no cover - defensive host guard
            pytest.skip("synthetic undname shim produced no transcript")
        _PROBED[mode] = True
    return path


# --- 1. transcript parsing --------------------------------------------------- #

REAL_TRANSCRIPT_CLASS = (
    "Microsoft (R) C++ Name Undecorator\n"
    "Copyright (C) Microsoft Corporation. All rights reserved.\n"
    "\n"
    'Undecoration of :- "??_R0?AVCGameApplication@@@8"\n'
    'is :- "class CGameApplication `RTTI Type Descriptor\'"\n'
    "\n"
)


def test_parse_extracts_quoted_payload():
    transcript = 'Undecoration of :- "x"\nis :- "class CGameApplication"\n\n'
    assert demangle._parse_undname_output(transcript) == "class CGameApplication"


def test_parse_strips_rtti_descriptor_suffix():
    assert demangle._parse_undname_output(REAL_TRANSCRIPT_CLASS) == "class CGameApplication"


def test_parse_keeps_template_expansion_verbatim():
    transcript = (
        'is :- "class std::vector<class Foo,class std::allocator<class Foo> > '
        "`RTTI Type Descriptor'\"\n"
    )
    assert (
        demangle._parse_undname_output(transcript)
        == "class std::vector<class Foo,class std::allocator<class Foo> >"
    )


def test_parse_accepts_unquoted_payload():
    assert demangle._parse_undname_output("is :- class Foo\n") == "class Foo"


def test_parse_uses_first_marker_only():
    transcript = 'is :- "class First"\nis :- "class Second"\n'
    assert demangle._parse_undname_output(transcript) == "class First"


def test_parse_returns_none_without_is_line():
    assert demangle._parse_undname_output("Microsoft (R) C++ Name Undecorator\n") is None
    assert demangle._parse_undname_output('Undecoration of :- "x"\n') is None


def test_parse_returns_none_for_blank_payload():
    assert demangle._parse_undname_output('is :- ""\n') is None
    assert demangle._parse_undname_output("is :-    \n") is None


def test_parse_returns_none_for_non_string():
    assert demangle._parse_undname_output(None) is None
    assert demangle._parse_undname_output(b'is :- "class Foo"') is None


def test_descriptor_suffix_constant_matches_the_tool_verbatim():
    assert "`RTTI Type Descriptor'" == RTTI_DESCRIPTOR_SUFFIX


# --- 2. argument forms ------------------------------------------------------ #

@pytest.mark.parametrize(
    "symbol",
    ["?AVCGameApplication@@", "?AUCGameApplication@@", "?AW4EColor@@"],
)
def test_symbol_forms_prefer_descriptor_symbol(symbol):
    assert demangle._symbol_forms(symbol) == ["??_R0" + symbol + "@8", symbol]


def test_symbol_forms_leave_non_rtti_symbol_alone():
    assert demangle._symbol_forms("?garbage") == ["?garbage"]
    assert demangle._symbol_forms("rawbytes") == ["rawbytes"]


# --- 3. executable discovery ------------------------------------------------ #

def test_find_undname_honours_override_env(tmp_path, monkeypatch):
    fake = tmp_path / "undname.exe"
    fake.write_text("", encoding="utf-8")
    monkeypatch.setenv(TOOTHPICK_UNDNAME_ENV, str(fake))
    found = find_undname()
    assert found == str(fake)


def test_find_undname_ignores_missing_override(tmp_path, monkeypatch):
    root = tmp_path / "msvc"
    exe = root / "bin" / "Hostx64" / "x64" / "undname.exe"
    exe.parent.mkdir(parents=True)
    exe.write_text("", encoding="utf-8")
    monkeypatch.setenv(TOOTHPICK_UNDNAME_ENV, str(tmp_path / "absent.exe"))
    monkeypatch.setenv(VCTOOLS_ENV, str(root))
    assert find_undname() == str(exe)


def test_find_undname_returns_none_when_nothing_is_installed(tmp_path, monkeypatch):
    for variable in (TOOTHPICK_UNDNAME_ENV, VCTOOLS_ENV, "ProgramFiles(x86)", "ProgramFiles"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("ProgramFiles", str(tmp_path / "empty-program-files"))
    assert find_undname() is None


def test_find_undname_scans_well_known_layout_newest_first(tmp_path, monkeypatch):
    for variable in (TOOTHPICK_UNDNAME_ENV, VCTOOLS_ENV, "ProgramFiles(x86)"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("ProgramFiles", str(tmp_path))
    tools = tmp_path / "Microsoft Visual Studio" / "2022" / "BuildTools" / "VC" / "Tools" / "MSVC"
    newer = tools / "14.44.35207" / "bin" / "Hostx64" / "x64" / "undname.exe"
    older = tools / "14.30.0" / "bin" / "Hostx64" / "x64" / "undname.exe"
    for exe in (newer, older):
        exe.parent.mkdir(parents=True)
        exe.write_text("", encoding="utf-8")
    assert find_undname() == str(newer)


# --- 4. invocation through a synthetic tool --------------------------------- #

def test_full_demangle_parses_is_line_without_a_real_tool(shim_factory):
    exe = _shim(shim_factory, "descriptor")
    assert full_demangle(".?AVCGameApplication@@", executable=exe) == "class CGameApplication"


@pytest.mark.parametrize(
    ("mangled", "expected"),
    [
        (".?AVCGameApplication@@", "class CGameApplication"),
        (".?AUCGameApplication@@", "struct CGameApplication"),
        (".?AW4EColor@@", "enum EColor"),
    ],
)
def test_full_demangle_covers_av_au_aw_forms(shim_factory, mangled, expected):
    exe = _shim(shim_factory, "descriptor")
    assert full_demangle(mangled, executable=exe) == expected


def test_full_demangle_expands_templates(shim_factory):
    exe = _shim(shim_factory, "descriptor")
    symbol = ".?AV?$vector@VFoo@@V?$allocator@VFoo@@@std@@@std@@"
    assert full_demangle(symbol, executable=exe) == (
        "class std::vector<class Foo,class std::allocator<class Foo> >"
    )


def test_full_demangle_honours_explicit_executable_off_windows(shim_factory, monkeypatch):
    exe = _shim(shim_factory, "descriptor")
    monkeypatch.setattr(demangle, "_is_windows", lambda: False)
    assert full_demangle(".?AVCGameApplication@@", executable=exe) == "class CGameApplication"


def test_full_demangle_uses_discovered_executable(shim_factory, monkeypatch):
    exe = _shim(shim_factory, "descriptor")
    monkeypatch.setattr(demangle, "_is_windows", lambda: True)
    monkeypatch.setattr(demangle, "find_undname", lambda: exe)
    assert full_demangle(".?AVCGameApplication@@") == "class CGameApplication"


def test_run_undname_returns_transcript(shim_factory):
    exe = _shim(shim_factory, "descriptor")
    transcript = demangle._run_undname(exe, "?AVCGameApplication@@", 30.0)
    assert isinstance(transcript, str)
    assert "is :-" in transcript


# --- 5. failure honesty ----------------------------------------------------- #

def test_full_demangle_rejects_echoed_argument(shim_factory):
    exe = _shim(shim_factory, "echo")
    assert full_demangle(".?AVCGameApplication@@", executable=exe) is None


def test_full_demangle_rejects_markerless_transcript(shim_factory):
    exe = _shim(shim_factory, "garbage")
    assert full_demangle(".?AVCGameApplication@@", executable=exe) is None


def test_full_demangle_rejects_blank_payload(shim_factory):
    exe = _shim(shim_factory, "blank")
    assert full_demangle(".?AVCGameApplication@@", executable=exe) is None


def test_full_demangle_returns_none_on_nonzero_exit(shim_factory):
    exe = _shim(shim_factory, "exit3")
    assert demangle._run_undname(exe, "?AVCGameApplication@@", 30.0) is None
    assert full_demangle(".?AVCGameApplication@@", executable=exe) is None


def test_full_demangle_returns_none_for_missing_executable(tmp_path):
    missing = str(tmp_path / "no-such-undname.exe")
    assert demangle._run_undname(missing, "?AVFoo@@", 30.0) is None
    assert full_demangle(".?AVCGameApplication@@", executable=missing) is None


def test_full_demangle_times_out_on_a_hanging_tool(shim_factory):
    exe = _shim(shim_factory, "hang")
    assert demangle._run_undname(exe, "rawsymbol", 0.3) is None
    assert full_demangle("rawsymbol", executable=exe, timeout=0.3) is None


def test_timeout_expiry_is_absorbed(monkeypatch):
    def _expire(argv, **kwargs):
        raise subprocess.TimeoutExpired(cmd=argv, timeout=kwargs.get("timeout"))

    monkeypatch.setattr(demangle.subprocess, "run", _expire)
    assert demangle._run_undname("undname.exe", "?AVFoo@@", 30.0) is None
    assert full_demangle(".?AVCGameApplication@@", executable="undname.exe") is None


def test_launch_error_is_absorbed(monkeypatch):
    def _refuse(argv, **kwargs):
        raise OSError("not a valid Win32 application")

    monkeypatch.setattr(demangle.subprocess, "run", _refuse)
    assert demangle.full_demangle(".?AVFoo@@", executable="undname.exe") is None


def test_non_windows_host_short_circuits_without_subprocess(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("subprocess must not run off Windows")

    monkeypatch.setattr(demangle, "_is_windows", lambda: False)
    monkeypatch.setattr(demangle.subprocess, "run", _boom)
    assert full_demangle(".?AVCGameApplication@@") is None


def test_missing_exe_on_windows_returns_none(tmp_path, monkeypatch):
    for variable in (TOOTHPICK_UNDNAME_ENV, VCTOOLS_ENV, "ProgramFiles(x86)", "ProgramFiles"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("ProgramFiles", str(tmp_path / "nothing-here"))
    monkeypatch.setattr(demangle, "_is_windows", lambda: True)
    assert find_undname() is None
    assert full_demangle(".?AVCGameApplication@@") is None
    assert demangle_best(".?AVCGameApplication@@") == ("CGameApplication", True)


@pytest.mark.parametrize("mangled", ["", ".", "   ", "\t\n"])
def test_full_demangle_returns_none_for_degenerate_input(mangled):
    assert full_demangle(mangled, executable="undname.exe") is None


def test_full_demangle_rejects_non_string():
    with pytest.raises(DemangleError):
        full_demangle(None)  # type: ignore[arg-type]
    with pytest.raises(DemangleError):
        full_demangle(123)  # type: ignore[arg-type]


# --- 6. subprocess contract ------------------------------------------------- #

class _Recorder:
    """Stand-in for ``subprocess.run`` capturing argv and keyword arguments."""

    def __init__(self, stdout: str = "", returncode: int = 0) -> None:
        self.stdout = stdout
        self.returncode = returncode
        self.calls: list = []

    def __call__(self, argv, **kwargs):
        self.calls.append((list(argv), kwargs))
        return subprocess.CompletedProcess(argv, self.returncode, self.stdout, "")


def test_subprocess_contract_is_sandboxed(monkeypatch):
    recorder = _Recorder('is :- "class CGameApplication"\n')
    monkeypatch.setattr(demangle.subprocess, "run", recorder)
    assert full_demangle(".?AVCGameApplication@@", executable="undname.exe") == (
        "class CGameApplication"
    )
    argv, kwargs = recorder.calls[0]
    assert argv == ["undname.exe", "??_R0?AVCGameApplication@@@8"]
    assert kwargs["timeout"] == DEMANGLE_TIMEOUT_SECONDS
    assert kwargs["timeout"] == 30.0
    assert "shell" not in kwargs
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert kwargs["check"] is False
    assert kwargs["text"] is True


def test_leading_dot_is_stripped_before_invocation(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr(demangle.subprocess, "run", recorder)
    assert full_demangle(".?AUFoo@@", executable="undname.exe") is None
    assert [argv[1] for argv, _ in recorder.calls] == ["??_R0?AUFoo@@@8", "?AUFoo@@"]


def test_descriptor_form_is_tried_first_and_bare_form_only_as_fallback(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr(demangle.subprocess, "run", recorder)
    assert full_demangle(".?AVCGameApplication@@", executable="undname.exe") is None
    assert len(recorder.calls) == 2
    assert recorder.calls[0][0][1] == "??_R0?AVCGameApplication@@@8"
    assert recorder.calls[1][0][1] == "?AVCGameApplication@@"


def test_custom_timeout_is_passed_through(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr(demangle.subprocess, "run", recorder)
    full_demangle(".?AVFoo@@", executable="undname.exe", timeout=7)
    assert recorder.calls[0][1]["timeout"] == 7.0


@pytest.mark.parametrize("bad", [0, -1, "later", None, float("nan")])
def test_bad_timeout_is_a_caller_error(bad):
    with pytest.raises(DemangleError):
        demangle._resolve_timeout(bad)
    with pytest.raises(DemangleError):
        full_demangle(".?AVFoo@@", executable="undname.exe", timeout=bad)


def test_good_timeout_values_are_accepted():
    assert demangle._resolve_timeout(5) == 5.0
    assert demangle._resolve_timeout("5") == 5.0
    assert demangle._resolve_timeout(0.5) == 0.5


# --- 7. demangle_best fallback ---------------------------------------------- #

def test_demangle_best_prefers_the_full_path(shim_factory):
    exe = _shim(shim_factory, "descriptor")
    assert demangle_best(".?AVCGameApplication@@", executable=exe) == (
        "class CGameApplication",
        True,
    )


@pytest.mark.parametrize(
    "mangled",
    [
        ".?AVCGameApplication@@",
        ".?AUPlainStruct@@",
        ".?AV?$vector@VFoo@@V?$allocator@VFoo@@@std@@@std@@",
        "?AVCGameApplication@@",
        "raw-bytes",
        "",
    ],
)
def test_demangle_best_falls_back_to_the_strip_helper(tmp_path, mangled):
    missing = str(tmp_path / "absent-undname.exe")
    assert demangle_best(mangled, executable=missing) == demangle_msvc_type(mangled)


def test_demangle_best_falls_back_when_the_tool_only_echoes(shim_factory):
    exe = _shim(shim_factory, "echo")
    assert demangle_best(".?AVCGameApplication@@", executable=exe) == (
        demangle_msvc_type(".?AVCGameApplication@@")
    )
    assert demangle_best(".?AVCGameApplication@@", executable=exe) == ("CGameApplication", True)


def test_strip_helper_is_imported_not_duplicated():
    assert demangle.demangle_msvc_type is rtti_parse.demangle_msvc_type
    source = Path(demangle.__file__).read_text(encoding="utf-8")
    assert "from toothpick.rtti_parse import demangle_msvc_type" in source


def test_demangle_best_propagates_argument_error():
    with pytest.raises(DemangleError):
        demangle_best(42)  # type: ignore[arg-type]


# --- 8. hygiene ------------------------------------------------------------- #

def test_default_timeout_is_thirty_seconds():
    assert DEMANGLE_TIMEOUT_SECONDS == 30.0


def test_module_is_offline_shell_free_and_faulthandler_free():
    source = Path(demangle.__file__).read_text(encoding="utf-8")
    assert re.search(r"^(?:import|from)\s+faulthandler", source, re.M) is None
    assert re.search(r"^(?:import|from)\s+(?:socket|ssl|urllib|http|requests|httpx)\b", source, re.M) is None
    assert re.search(r"shell\s*=\s*True", source) is None
    assert "is :-" in source


def test_public_api_is_importable():
    for name in demangle.__all__:
        assert getattr(demangle, name) is not None
    for name in ("full_demangle", "demangle_best", "find_undname", "DemangleError"):
        assert name in demangle.__all__


# --- 9. the real tool, when installed --------------------------------------- #

@_needs_real_undname
def test_real_undname_demangles_an_av_symbol():
    assert full_demangle(".?AVCGameApplication@@") == "class CGameApplication"


@_needs_real_undname
def test_real_undname_expands_a_template_symbol():
    symbol = ".?AV?$vector@VFoo@@V?$allocator@VFoo@@@std@@@std@@"
    name = full_demangle(symbol)
    assert name is not None
    assert "vector" in name
    assert "class Foo" in name


@_needs_real_undname
def test_real_undname_matches_demangle_best():
    assert demangle_best(".?AVCGameApplication@@") == ("class CGameApplication", True)


@_needs_real_undname
def test_real_undname_garbage_never_round_trips_as_a_name():
    assert full_demangle("??_R0?AWFoo@@@8") is None
