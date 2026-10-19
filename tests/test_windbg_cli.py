"""Tests for UNIT-W15 Chunk 3 (CLI integration) + plan Test 6.

Validates the ``toothpick cdb`` subcommand wired in :mod:`toothpick.cli`:

1. ``cdb --help`` exits 0 and documents ``--pid`` and ``--dump``.
2. ``cdb --dump <minidump>`` emits structured JSON carrying ``exception_code``
   and ``faulting_ip`` (plan Test 6), with ``--out`` writing the same payload.
3. ``cdb`` with neither ``--pid`` nor ``--dump`` exits nonzero.
4. ``cdb --pid`` drives :meth:`CdbEngine.attach_non_invasive` and persists a
   ``phase="cdb_trace"`` receipt via :meth:`Database.add_receipt`.

The synthetic ``MDMP`` dump is packed in-test with :mod:`struct` (magic
``0x504d444d``) using the exact ``MINIDUMP_*`` layouts consumed by
:class:`toothpick.windbg.MinidumpReader`, so nothing here needs a Windows host,
a real ``cdb.exe``, or the Windows SDK (plan Risk R5).
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path
from typing import Sequence

import pytest
from click.testing import CliRunner

from toothpick.cli import cli
from toothpick.storage import Database
from toothpick.windbg import CDB_PHASE, MDMP_MAGIC, MinidumpReader

# ---------------------------------------------------------------------------
# Minidump fixture (stream types per minidumpapiset.h)
# ---------------------------------------------------------------------------

STREAM_THREAD_LIST = 3
STREAM_MODULE_LIST = 4
STREAM_EXCEPTION = 6
STREAM_SYSTEM_INFO = 7

STREAM_ORDER = (
    (STREAM_EXCEPTION, "ExceptionStream"),
    (STREAM_THREAD_LIST, "ThreadListStream"),
    (STREAM_MODULE_LIST, "ModuleListStream"),
    (STREAM_SYSTEM_INFO, "SystemInfoStream"),
)

_MINIDUMP_VERSION = 0x0000A793
_MINIDUMP_TIMESTAMP = 0x65B7C3A1
_PROCESSOR_ARCHITECTURE_AMD64 = 9

EXCEPTION_CODE = 0xC0000005
FAULTING_IP = 0x7FF61A5C1234
FAULTING_CDB_MODULE = "game"
#: The minidump reader takes the module name from `Path(path).name`; the
#: `MINIDUMP_STRING` length field counts the UTF-16 NUL terminator, so the name
#: read back from a dump keeps a trailing NUL and is normalised for comparison.
FAULTING_DUMP_MODULE = "game.exe"


def _module_name(raw: str) -> str:
    """Normalise a module name read from a minidump (strip the UTF-16 NUL)."""
    return raw.rstrip("\x00")
_THREAD_ID = 0x1A2B
_EXCEPTION_PARAMETERS = (0x7FF61A5B0000, 1)
_CONTEXT_SIZE = 0x300

#: (name, base of image, size of image, image path)
_MODULE_FIXTURE = (
    ("game", 0x7FF61A5B0000, 0x20000, r"C:\Games\game.exe"),
    ("ntdll", 0x7FF83A200000, 0x1C1000, r"C:\Windows\SYSTEM32\ntdll.dll"),
)

_HEADER_STRUCT = struct.Struct("<4sIIIIIQ")
_DIRECTORY_STRUCT = struct.Struct("<III")
_EXCEPTION_STRUCT = struct.Struct("<IIIIQQII15QII")
_SYSTEM_INFO_STRUCT = struct.Struct("<HHHBBIIIIIHH24s")
_THREAD_LIST_HEAD_STRUCT = struct.Struct("<I")
_THREAD_STRUCT = struct.Struct("<IIIIQQIIII")
_MODULE_LIST_HEAD_STRUCT = struct.Struct("<I")
_MODULE_STRUCT = struct.Struct("<QIIII52sIIIIQQ")

_PYTHON = sys.executable


def _minidump_string(text: str) -> bytes:
    """A ``MINIDUMP_STRING``: ULONG32 byte length followed by UTF-16LE text."""
    encoded = text.encode("utf-16-le") + b"\x00\x00"
    return struct.pack("<I", len(encoded)) + encoded


def _context_blob() -> bytes:
    """An x64 ``CONTEXT`` record (EFlags 0x44, Rax 0x78 .. R15 0xF0, Rip 0xF8)."""
    blob = bytearray(_CONTEXT_SIZE)
    struct.pack_into("<I", blob, 0x44, 0x202)  # eflags
    ordered = (
        0x42,  # rax
        0x7FF61A5B1000,  # rcx
        0x1,  # rdx
        0x0,  # rbx
        0x8A3F4FF8B0,  # rsp
        0x8A3F4FF9A0,  # rbp
        0x24A1F3A0000,  # rsi
        0x0,  # rdi
    ) + (0,) * 8  # r8 .. r15
    for index, value in enumerate(ordered):
        struct.pack_into("<Q", blob, 0x78 + index * 8, value)
    struct.pack_into("<Q", blob, 0xF8, FAULTING_IP)  # rip
    return bytes(blob)


def build_minidump(
    exception_code: int = EXCEPTION_CODE,
    exception_parameters: Sequence[int] = _EXCEPTION_PARAMETERS,
) -> bytes:
    """Build a synthetic ``MDMP`` dump (exception/thread/module/sysinfo streams)."""
    context = _context_blob()
    parameters = tuple(int(value) for value in exception_parameters)[:15]
    padded_parameters = parameters + (0,) * (15 - len(parameters))

    exception_size = _EXCEPTION_STRUCT.size
    thread_list_size = _THREAD_LIST_HEAD_STRUCT.size + _THREAD_STRUCT.size
    module_list_size = (
        _MODULE_LIST_HEAD_STRUCT.size + _MODULE_STRUCT.size * len(_MODULE_FIXTURE)
    )
    system_info_size = _SYSTEM_INFO_STRUCT.size

    directory_rva = _HEADER_STRUCT.size
    exception_rva = directory_rva + _DIRECTORY_STRUCT.size * len(STREAM_ORDER)
    thread_list_rva = exception_rva + exception_size
    module_list_rva = thread_list_rva + thread_list_size
    system_info_rva = module_list_rva + module_list_size
    context_rva = system_info_rva + system_info_size
    strings_rva = context_rva + len(context)

    strings = [_minidump_string(path) for _, _, _, path in _MODULE_FIXTURE]
    name_rvas = []
    cursor = strings_rva
    for blob in strings:
        name_rvas.append(cursor)
        cursor += len(blob)

    exception_stream = _EXCEPTION_STRUCT.pack(
        _THREAD_ID,
        0,
        exception_code,
        0,  # exception flags
        FAULTING_IP,  # nested MINIDUMP_EXCEPTION pointer (unread by the reader)
        FAULTING_IP,  # ExceptionAddress
        len(parameters),
        0,
        *padded_parameters,
        len(context),
        context_rva,
    )

    thread_list = _THREAD_LIST_HEAD_STRUCT.pack(1) + _THREAD_STRUCT.pack(
        _THREAD_ID,
        1,
        2,
        15,
        0x24A1F3B0000,  # teb
        0x8A3F4FF0000,  # stack start of memory range
        0x21000,  # stack size
        0,  # stack rva
        len(context),
        context_rva,
    )

    module_list = _MODULE_LIST_HEAD_STRUCT.pack(len(_MODULE_FIXTURE))
    for (_, base_addr, size_of_image, _), name_rva in zip(_MODULE_FIXTURE, name_rvas):
        module_list += _MODULE_STRUCT.pack(
            base_addr,
            size_of_image,
            0,  # checksum
            0,  # time date stamp
            name_rva,
            b"\x00" * 52,  # VS_FIXEDFILEINFO
            0,
            0,  # cv record
            0,
            0,  # misc record
            0,
            0,  # reserved0 / reserved1
        )

    system_info = _SYSTEM_INFO_STRUCT.pack(
        _PROCESSOR_ARCHITECTURE_AMD64,
        6,
        0x3A09,
        8,
        1,  # product type: workstation
        10,
        0,
        19041,
        2,  # platform id: windows
        0,  # csd version rva
        0,
        0,
        b"\x00" * 24,  # cpu union
    )

    header = _HEADER_STRUCT.pack(
        b"MDMP",
        _MINIDUMP_VERSION,
        len(STREAM_ORDER),
        directory_rva,
        0,
        _MINIDUMP_TIMESTAMP,
        0,
    )

    directory = b"".join(
        _DIRECTORY_STRUCT.pack(stream_type, size, rva)
        for stream_type, size, rva in (
            (STREAM_EXCEPTION, exception_size, exception_rva),
            (STREAM_THREAD_LIST, thread_list_size, thread_list_rva),
            (STREAM_MODULE_LIST, module_list_size, module_list_rva),
            (STREAM_SYSTEM_INFO, system_info_size, system_info_rva),
        )
    )

    return (
        header
        + directory
        + exception_stream
        + thread_list
        + module_list
        + system_info
        + context
        + b"".join(strings)
    )


# ---------------------------------------------------------------------------
# Captured-style CDB transcripts replayed by the fake cdb.exe
# ---------------------------------------------------------------------------

REGISTER_DUMP = """\
rax=0000000000000042 rbx=0000000000000000 rcx=00007ff61a5b1000 rdx=0000000000000001
rsi=0000024a1f3a0000 rdi=0000000000000000 rbp=0000008a3f4ff9a0 rsp=0000008a3f4ff8b0
rip=00007ff61a5c1234
iopl=0         nv up ei pl nz na po nc
cs=0010  ss=0018  ds=002b  es=002b  fs=0053  gs=002b             efl=00000202
"""

CALL_STACK_TEXT = r""" # Child-SP          RetAddr               Call Site
00 0000008a`3f4ff8b0 00007ff6`1a5c1234     game!CrashRoutine+0x2f [C:\src\game.cpp @ 137]
01 0000008a`3f4ff940 00007ff8`3a212345     game!main+0x42
"""

MODULE_LIST_TEXT = r"""start             end                 module name
00007ff6`1a5b0000 00007ff6`1a5d0000   game       (deferred)
00007ff8`3a200000 00007ff8`3a3c1000   ntdll      (deferred)
"""

_FAKE_CDB_SCRIPT = (
    "import sys\n"
    "\n"
    'if "-pv" not in sys.argv:\n'
    '    sys.stderr.write("fake cdb: expected a non-invasive -pv attach\\n")\n'
    "    sys.exit(3)\n"
    "\n"
    'sys.stdout.write("""' + REGISTER_DUMP + '""")\n'
    'sys.stdout.write("""' + CALL_STACK_TEXT + '""")\n'
    'sys.stdout.write("""' + MODULE_LIST_TEXT + '""")\n'
    "sys.exit(0)\n"
)


@pytest.fixture
def fake_cdb(tmp_path: Path) -> Path:
    """A stand-in ``cdb.exe`` that replays captured-style r/kp/lm transcripts."""
    script = tmp_path / "fake_cdb.py"
    script.write_text(_FAKE_CDB_SCRIPT, encoding="utf-8")
    return script


@pytest.fixture
def dump_file(tmp_path: Path) -> Path:
    """A synthetic on-disk ``.dmp`` built with :mod:`struct` (MDMP magic)."""
    path = tmp_path / "synthetic_crash.dmp"
    path.write_bytes(build_minidump())
    return path


# ---------------------------------------------------------------------------
# 1. --help
# ---------------------------------------------------------------------------


def test_cdb_help_lists_pid_and_dump_options() -> None:
    """``cdb --help`` succeeds and documents both target modes."""
    runner = CliRunner()
    result = runner.invoke(cli, ["cdb", "--help"])
    assert result.exit_code == 0, f"cdb --help failed: {result.output}"
    assert "--pid" in result.output
    assert "--dump" in result.output
    assert "--cmd" in result.output
    assert "--out" in result.output
    assert "--verify-addr" in result.output


# ---------------------------------------------------------------------------
# 2. --dump structured JSON (plan Test 6)
# ---------------------------------------------------------------------------


def test_cdb_dump_emits_structured_json_with_exception_code_and_faulting_ip(
    dump_file: Path, tmp_path: Path
) -> None:
    """Plan Test 6: ``cdb --dump`` emits structured JSON for a real MDMP file."""
    # Sanity-check the fixture against the production parser before exercising
    # the CLI, so a malformed build is reported as a fixture bug, not a CLI bug.
    parsed = MinidumpReader(dump_file).read_header()
    assert parsed.signature == MDMP_MAGIC
    assert parsed.magic_hex == "0x504d444d"

    runner = CliRunner()
    result = runner.invoke(cli, ["cdb", "--dump", str(dump_file)])
    assert result.exit_code == 0, f"cdb --dump failed: {result.output}"

    payload = json.loads(result.output)
    assert payload["exception_code"] == EXCEPTION_CODE
    assert payload["exception_code_hex"] == "0xc0000005"
    assert payload["exception_name"] == "STATUS_ACCESS_VIOLATION"
    assert payload["faulting_ip"] == FAULTING_IP
    assert payload["faulting_ip_hex"] == f"0x{FAULTING_IP:x}"
    assert payload["crash"]["faulting_ip"] == FAULTING_IP
    assert _module_name(payload["crash"]["faulting_module"]) == FAULTING_DUMP_MODULE
    assert payload["crash"]["exception_name"] == "STATUS_ACCESS_VIOLATION"
    assert payload["mode"] == "crash_dump"
    assert payload["phase"] == CDB_PHASE

    # --out writes the identical report (which is still emitted on stdout).
    out_path = tmp_path / "crash_analysis.json"
    out_result = runner.invoke(
        cli, ["cdb", "--dump", str(dump_file), "--out", str(out_path)]
    )
    assert out_result.exit_code == 0, f"cdb --dump --out failed: {out_result.output}"
    assert out_path.is_file()
    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written["exception_code"] == EXCEPTION_CODE
    assert written["faulting_ip"] == FAULTING_IP


def test_cdb_dump_verify_addr_reports_module_resolution(dump_file: Path) -> None:
    """``--verify-addr`` resolves against the modules of the crash dump."""
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["cdb", "--dump", str(dump_file), "--verify-addr", f"0x{FAULTING_IP:x}"],
    )
    assert result.exit_code == 0, f"cdb --dump --verify-addr failed: {result.output}"
    payload = json.loads(result.output)
    verification = payload["address_verification"]
    assert verification["resolved"] is True
    assert _module_name(verification["module"]["name"]) == FAULTING_DUMP_MODULE
    assert verification["addr"] == FAULTING_IP


def test_cdb_dump_missing_file_exits_nonzero(tmp_path: Path) -> None:
    """A non-existent dump path fails cleanly instead of raising."""
    runner = CliRunner()
    result = runner.invoke(cli, ["cdb", "--dump", str(tmp_path / "nope.dmp")])
    assert result.exit_code != 0
    assert "not found" in result.output or "Error" in result.output


# ---------------------------------------------------------------------------
# 3. Neither --pid nor --dump
# ---------------------------------------------------------------------------


def test_cdb_requires_pid_or_dump() -> None:
    """Neither ``--pid`` nor ``--dump`` is a usage error (nonzero exit)."""
    runner = CliRunner()
    result = runner.invoke(cli, ["cdb"])
    assert result.exit_code != 0
    assert "--pid" in result.output or "Error" in result.output


def test_cdb_pid_and_dump_are_mutually_exclusive(dump_file: Path) -> None:
    """Both target modes at once is rejected rather than silently preferred."""
    runner = CliRunner()
    result = runner.invoke(
        cli, ["cdb", "--pid", "1234", "--dump", str(dump_file)]
    )
    assert result.exit_code != 0
    assert "Mutually exclusive" in result.output


# ---------------------------------------------------------------------------
# 4. --pid non-invasive attach + cdb_trace receipt
# ---------------------------------------------------------------------------


def _install_fake_cdb(monkeypatch: pytest.MonkeyPatch, fake_cdb: Path) -> None:
    """Force every ``CdbEngine`` to launch the fake CDB script."""
    import toothpick.windbg as windbg_module

    real_init = windbg_module.CdbEngine.__init__

    def _init(self, cdb_path=None, timeout=30.0, symbol_path=None, extra_env=None):
        real_init(
            self,
            cdb_path=[_PYTHON, str(fake_cdb)],
            timeout=timeout,
        )

    monkeypatch.setattr(windbg_module.CdbEngine, "__init__", _init)


def test_cdb_pid_attaches_non_invasively_and_writes_cdb_trace_receipt(
    fake_cdb: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--pid`` calls attach_non_invasive and persists a cdb_trace receipt."""
    _install_fake_cdb(monkeypatch, fake_cdb)

    db_path = tmp_path / "cdb.db"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "cdb",
            "--pid",
            "4242",
            "--cmd",
            "r",
            "--cmd",
            "kp",
            "--cmd",
            "q",
            "--verify-addr",
            f"0x{FAULTING_IP:x}",
            "--db",
            str(db_path),
            "--sha",
            "a" * 64,
        ],
    )
    assert result.exit_code == 0, f"cdb --pid failed: {result.output}"

    payload = json.loads(result.output)
    assert payload["mode"] == "non_invasive_attach"
    assert payload["pid"] == 4242
    assert payload["commands"] == ["r", "kp", "q"]
    assert payload["registers"]["rip"] == FAULTING_IP
    assert payload["stack_trace"], "expected parsed CDB call-stack frames"
    assert payload["address_verification"]["resolved"] is True
    assert payload["address_verification"]["module"]["name"] == FAULTING_CDB_MODULE
    assert payload["receipt"]["phase"] == CDB_PHASE
    assert payload["binary_sha256"] == "a" * 64

    # The receipt lands in toothpick.db under phase="cdb_trace".
    with Database(db_path, create=False) as db:
        receipts = db.get_receipts("a" * 64)
    assert len(receipts) == 1
    receipt = receipts[0]
    assert receipt["phase"] == CDB_PHASE
    assert receipt["func_addr"] == FAULTING_IP
    assert receipt["payload"]["phase"] == CDB_PHASE
    assert receipt["payload"]["verifier_status"] == "GROUNDED"
    assert receipt["payload"]["ground_type"] == "CDB_LIVE_ATTACH"
    assert receipt["payload"]["session"]["pid"] == 4242
    assert receipt["payload"]["address_verification"]["resolved"] is True
    assert receipt["receipt_id"] == payload["receipt_id"]


def test_cdb_dump_writes_cdb_trace_receipt_with_faulting_ip(tmp_path: Path) -> None:
    """``--dump --db`` stores the crash receipt with func_addr=faulting_ip."""
    dump_path = tmp_path / "crash.dmp"
    dump_path.write_bytes(build_minidump())
    db_path = tmp_path / "crash.db"

    runner = CliRunner()
    result = runner.invoke(
        cli, ["cdb", "--dump", str(dump_path), "--db", str(db_path)]
    )
    assert result.exit_code == 0, f"cdb --dump --db failed: {result.output}"
    payload = json.loads(result.output)
    assert payload["receipt"]["phase"] == CDB_PHASE
    assert payload["receipt"]["func_addr"] == FAULTING_IP

    with Database(db_path, create=False) as db:
        receipts = db.get_receipts("", func_addr=FAULTING_IP)
    assert len(receipts) == 1
    assert receipts[0]["phase"] == CDB_PHASE
    assert receipts[0]["payload"]["crash"]["exception_code"] == EXCEPTION_CODE
