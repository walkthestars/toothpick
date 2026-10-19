"""
Tests for UNIT-W15 Chunk 4 (`docs/units/UNIT-W15-WINDBG-PLAN.md`): the frozen
public API of `src/toothpick/windbg.py`.

Battery:

1. `MinidumpReader.read_header` / `read_directory` over a synthetic but
   structurally exact `MDMP` minidump (32-byte `MINIDUMP_HEADER`, 12-byte
   `MINIDUMP_DIRECTORY` entries, 168-byte `MINIDUMP_EXCEPTION_STREAM`,
   48-byte `MINIDUMP_THREAD`, 108-byte `MINIDUMP_MODULE`, 56-byte
   `MINIDUMP_SYSTEM_INFO`).
2. Access violation (`0xc0000005`) decoding through `parse_exception` and the
   exception's `exception_name`.
3. CDB `r` register text parsing (`parse_cdb_registers`,
   `CdbRegisters.from_cdb_text`).
4. CDB `kp` call stack parsing into `CdbStackFrame` records.
5. CDB `lm` module list parsing into `CdbModule` records.
6. `CrashReport.to_dict` / `to_json` / `receipt_payload` grounding shape
   (`phase="cdb_trace"`, `verifier_status="GROUNDED"`).

No live `cdb.exe` and no Windows SDK are required: the minidump is generated
in-process and every CDB transcript is a captured-style text blob, so the whole
battery is deterministic on any host (plan Risk R5).
"""

from __future__ import annotations

import json
import struct

import pytest

from toothpick.windbg import (
    CDB_PHASE,
    EXCEPTION_NAMES,
    MDMP_MAGIC,
    CdbEngine,
    CdbError,
    CdbModule,
    CdbRegisters,
    CdbStackFrame,
    CrashReport,
    MinidumpReader,
    parse_cdb_call_stack,
    parse_cdb_modules,
    parse_cdb_registers,
    parse_x64_context,
)

# ---------------------------------------------------------------------------
# Minidump fixture constants (MINIDUMP stream types; winnt.h / minidumpapiset.h)
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

_EXCEPTION_ADDRESS = 0x7FF61A5C1234
_CONTEXT_RIP = 0x7FF61A5C1234
_CONTEXT_EFLAGS = 0x202
_CONTEXT_RAX = 0x42
_CONTEXT_RCX = 0x7FF61A5B1000
_CONTEXT_RSP = 0x8A3F4FF8B0
_CONTEXT_SIZE = 0x300

#: (name, base of image, size of image, image path)
_MODULE_FIXTURE = (
    ("game", 0x7FF61A5B0000, 0x20000, r"C:\Games\game.exe"),
    ("ntdll", 0x7FF83A200000, 0x1C1000, r"C:\Windows\SYSTEM32\ntdll.dll"),
    ("KERNEL32", 0x7FF83B100000, 0x1FB000, r"C:\Windows\System32\KERNEL32.DLL"),
)

# Layout-exact packers for the structures above (4-byte packing, little endian).
_HEADER_STRUCT = struct.Struct("<4sIIIIIQ")
_DIRECTORY_STRUCT = struct.Struct("<III")
_EXCEPTION_STRUCT = struct.Struct("<IIIIQQII15QII")
_SYSTEM_INFO_STRUCT = struct.Struct("<HHHBBIIIIIHH24s")
_THREAD_LIST_HEAD_STRUCT = struct.Struct("<I")
_THREAD_STRUCT = struct.Struct("<IIIIQQIIII")
_MODULE_LIST_HEAD_STRUCT = struct.Struct("<I")
_MODULE_STRUCT = struct.Struct("<QIIII52sIIIIQQ")


def _minidump_string(text: str) -> bytes:
    """A `MINIDUMP_STRING`: ULONG32 byte length followed by UTF-16LE text."""
    encoded = text.encode("utf-16-le") + b"\x00\x00"
    return struct.pack("<I", len(encoded)) + encoded


def _context_blob() -> bytes:
    """An x64 `CONTEXT` record (EFlags 0x44, Rax 0x78 .. R15 0xF0, Rip 0xF8)."""
    blob = bytearray(_CONTEXT_SIZE)
    struct.pack_into("<I", blob, 0x44, _CONTEXT_EFLAGS)
    ordered = (
        _CONTEXT_RAX,
        _CONTEXT_RCX,
        0x1,  # rdx
        0x0,  # rbx
        _CONTEXT_RSP,
        0x8A3F4FF9A0,  # rbp
        0x24A1F3A0000,  # rsi
        0x0,  # rdi
    ) + (0,) * 8  # r8 .. r15
    for index, value in enumerate(ordered):
        struct.pack_into("<Q", blob, 0x78 + index * 8, value)
    struct.pack_into("<Q", blob, 0xF8, _CONTEXT_RIP)
    return bytes(blob)


def _build_minidump(
    exception_code: int = 0xC0000005,
    exception_parameters: tuple = (0x7FF61A5B0000, 1),
    thread_id: int = 0x1A2B,
) -> bytes:
    """Build a synthetic `MDMP` dump with exception/thread/module/sysinfo streams."""
    context = _context_blob()
    parameters = tuple(int(value) for value in exception_parameters)[:15]
    padded_parameters = parameters + (0,) * (15 - len(parameters))

    exception_size = _EXCEPTION_STRUCT.size
    thread_list_size = _THREAD_LIST_HEAD_STRUCT.size + _THREAD_STRUCT.size
    module_list_size = _MODULE_LIST_HEAD_STRUCT.size + _MODULE_STRUCT.size * len(_MODULE_FIXTURE)
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
        thread_id,
        0,
        exception_code,
        0,  # exception flags
        _EXCEPTION_ADDRESS,  # nested MINIDUMP_EXCEPTION pointer (unused)
        _EXCEPTION_ADDRESS,  # ExceptionAddress
        len(parameters),
        0,
        *padded_parameters,
        len(context),
        context_rva,
    )

    thread_list = _THREAD_LIST_HEAD_STRUCT.pack(1) + _THREAD_STRUCT.pack(
        thread_id,
        1,
        2,
        15,
        0x24A1F3B0000,  # teb
        0x8A3F4FF0000,  # stack start of memory range
        0x21000,  # stack size
        0,  # stack rva (not needed by the reader)
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
# CDB text fixtures (captured-style console transcripts)
# ---------------------------------------------------------------------------

REGISTER_DUMP = """\
rax=0000000000000042 rbx=0000000000000000 rcx=00007ff61a5b1000 rdx=0000000000000001
rsi=0000024a1f3a0000 rdi=0000000000000000 rbp=0000008a3f4ff9a0 rsp=0000008a3f4ff8b0
r8=0000000000000000  r9=0000000000000000  r10=0000000000000000 r11=0000000000000000
r12=0000000000000000 r13=0000000000000000 r14=0000000000000000 r15=0000000000000000
rip=00007ff61a5c1234
iopl=0         nv up ei pl nz na po nc
cs=0010  ss=0018  ds=002b  es=002b  fs=0053  gs=002b             efl=00000202
"""

CALL_STACK_TEXT = r""" # Child-SP          RetAddr               Call Site
00 0000008a`3f4ff8b0 00007ff6`1a5c1234     game!CrashRoutine+0x2f [C:\src\game.cpp @ 137]
01 0000008a`3f4ff940 00007ff8`3a212345     game!main+0x42
02 0000008a`3f4ff9d0 00007ff8`3b1c9abc     KERNEL32!BaseThreadInitThunk+0x14
03 0000008a`3f4ffa00 00000000`00000000     ntdll!RtlUserThreadStart+0x21
Unable to load image C:\Games\missing.dll, Win32 error 0n2
"""

MODULE_LIST_TEXT = r"""start             end                 module name
00007ff6`1a5b0000 00007ff6`1a5d0000   game       (deferred)
00007ff8`3a200000 00007ff8`3a3c1000   ntdll      (deferred)
00007ff8`3b100000 00007ff8`3b2fb000   KERNEL32   C:\Windows\System32\KERNEL32.DLL
Unloaded modules:
"""


# ---------------------------------------------------------------------------
# 1. Minidump header + directory
# ---------------------------------------------------------------------------


def test_minidump_reader_parses_synthetic_mdmp_header_and_directory(tmp_path):
    data = _build_minidump()
    reader = MinidumpReader(data)

    header = reader.read_header()
    assert header.signature == MDMP_MAGIC
    assert header.magic_hex == "0x504d444d"
    assert header.is_valid
    assert header.version == _MINIDUMP_VERSION
    assert (header.version_major, header.version_minor) == (0x0000, 0xA793)
    assert header.number_of_streams == len(STREAM_ORDER)
    assert header.stream_directory_rva == _HEADER_STRUCT.size
    assert header.stream_directory_size == _DIRECTORY_STRUCT.size * len(STREAM_ORDER)
    assert header.time_date_stamp == _MINIDUMP_TIMESTAMP
    assert header.to_dict()["signature_ascii"] == "MDMP"

    directory = reader.read_directory()
    assert [entry.stream_type for entry in directory] == [s for s, _ in STREAM_ORDER]
    assert [entry.stream_name for entry in directory] == [n for _, n in STREAM_ORDER]
    assert all(entry.rva >= header.stream_directory_rva for entry in directory)
    assert all(entry.end_rva == entry.rva + entry.data_size for entry in directory)
    assert reader.directory_summary() == [entry.to_dict() for entry in directory]

    # The same bytes parse identically when they arrive as a .dmp file on disk.
    dump_path = tmp_path / "synthetic_crash.dmp"
    dump_path.write_bytes(data)
    from_path = MinidumpReader(dump_path)
    assert from_path.path == dump_path
    assert from_path.read_header().to_dict() == header.to_dict()
    assert [entry.to_dict() for entry in from_path.read_directory()] == [
        entry.to_dict() for entry in directory
    ]

    # Truncated or non-MDMP containers are CdbError, never a struct/IndexError.
    with pytest.raises(CdbError):
        MinidumpReader(b"NOPE" + data[4:]).read_header()
    with pytest.raises(CdbError):
        MinidumpReader(data[:16])
    with pytest.raises(CdbError):
        MinidumpReader(tmp_path / "does_not_exist.dmp")


# ---------------------------------------------------------------------------
# 2. Access violation decoding
# ---------------------------------------------------------------------------


def test_access_violation_exception_decoding():
    reader = MinidumpReader(_build_minidump())
    exception = reader.parse_exception()
    assert exception is not None
    assert exception.thread_id == 0x1A2B
    assert exception.exception_code == 0xC0000005
    assert exception.to_dict()["exception_code_hex"] == "0xc0000005"
    assert EXCEPTION_NAMES[0xC0000005] == "STATUS_ACCESS_VIOLATION"
    assert exception.exception_name == "STATUS_ACCESS_VIOLATION"
    assert exception.exception_address == _EXCEPTION_ADDRESS
    assert exception.parameters == [0x7FF61A5B0000, 1]
    assert exception.faulting_address == 0x7FF61A5B0000
    assert exception.faulting_operation == "write"

    # The crashing thread's CONTEXT decodes through the x64 context parser.
    registers = reader.parse_registers()
    assert registers == parse_x64_context(_context_blob())
    assert registers.rip == _CONTEXT_RIP
    assert registers.rsp == _CONTEXT_RSP
    assert registers.rcx == _CONTEXT_RCX
    assert registers.eflags == _CONTEXT_EFLAGS

    # No ExceptionInformation parameters -> no dereferenced faulting address,
    # and an unknown code still decodes to a stable UNKNOWN_EXCEPTION_* name.
    bare = MinidumpReader(_build_minidump(exception_parameters=())).parse_exception()
    assert bare.faulting_address is None
    assert bare.faulting_operation is None
    exotic = MinidumpReader(_build_minidump(exception_code=0xDEAD0001)).parse_exception()
    assert exotic.exception_name == "UNKNOWN_EXCEPTION_0xDEAD0001"
    assert exotic.exception_name not in EXCEPTION_NAMES.values()


# ---------------------------------------------------------------------------
# 3. Register text parsing
# ---------------------------------------------------------------------------


def test_parse_cdb_registers_and_from_cdb_text():
    registers = parse_cdb_registers(REGISTER_DUMP)
    assert isinstance(registers, CdbRegisters)
    assert registers == CdbRegisters.from_cdb_text(REGISTER_DUMP)

    assert registers.rax == 0x42
    assert registers.rcx == 0x7FF61A5B1000
    assert registers.rdx == 0x1
    assert registers.rsi == 0x24A1F3A0000
    assert registers.rsp == 0x8A3F4FF8B0
    assert registers.rbp == 0x8A3F4FF9A0
    assert registers.r15 == 0x0
    assert registers.rip == 0x7FF61A5C1234
    assert registers.eflags == 0x202
    assert registers.get("rcx") == 0x7FF61A5B1000
    assert registers.get("cs") == 0  # segment/iopl mnemonics are ignored

    payload = registers.to_dict()
    assert payload["rax"] == 0x42
    assert payload["rax_hex"] == "0x42"
    assert payload["r15_hex"] == "0x0"
    assert json.loads(registers.to_json()) == payload

    # 32-bit aliases canonicalise and non-register lines are tolerated.
    aliased = parse_cdb_registers("eip=00001234 nv up ei pl\ncs=0010\n")
    assert aliased.rip == 0x1234
    assert aliased.get("eip") == 0x1234
    assert parse_cdb_registers("") == CdbRegisters()


# ---------------------------------------------------------------------------
# 4. Call stack parsing
# ---------------------------------------------------------------------------


def test_parse_cdb_call_stack_returns_cdb_stack_frames():
    frames = parse_cdb_call_stack(CALL_STACK_TEXT)
    assert len(frames) == 4
    assert all(isinstance(frame, CdbStackFrame) for frame in frames)
    assert [frame.frame_num for frame in frames] == [0, 1, 2, 3]
    assert frames[0].child_sp == 0x8A3F4FF8B0
    assert frames[0].ret_addr == 0x7FF61A5C1234
    assert frames[0].call_site.startswith("game!CrashRoutine+0x2f")
    assert "[C:\\src\\game.cpp @ 137]" in frames[0].call_site
    assert frames[2].call_site == "KERNEL32!BaseThreadInitThunk+0x14"
    assert frames[3].ret_addr == 0x0
    assert frames[0].to_dict()["child_sp_hex"] == "0x8a3f4ff8b0"

    # Header, annotation and unparsable lines are dropped, not misparsed.
    assert frames == CdbEngine.parse_call_stack(CALL_STACK_TEXT)
    assert parse_cdb_call_stack("# Child-SP          RetAddr               Call Site\n") == []
    assert parse_cdb_call_stack("Unable to load image missing.dll\n") == []
    assert parse_cdb_call_stack("") == []

    # Out-of-order listings come back sorted by frame number.
    shuffled = "02 0000008a`3f4ff9d0 0000008a`3f4ffa00 game!c\n00 0000008a`3f4ff8b0 0000008a`3f4ff940 game!a\n"
    assert [frame.frame_num for frame in parse_cdb_call_stack(shuffled)] == [0, 2]


# ---------------------------------------------------------------------------
# 5. Module list parsing
# ---------------------------------------------------------------------------


def test_parse_cdb_modules_returns_cdb_modules():
    modules = parse_cdb_modules(MODULE_LIST_TEXT)
    assert len(modules) == 3
    assert all(isinstance(module, CdbModule) for module in modules)
    assert [module.name for module in modules] == ["game", "ntdll", "KERNEL32"]

    game, ntdll, kernel32 = modules
    assert game.base_addr == 0x7FF61A5B0000
    assert game.end_addr == 0x7FF61A5D0000
    assert game.size == 0x20000
    assert game.path == ""
    assert ntdll.path == ""
    assert kernel32.path == r"C:\Windows\System32\KERNEL32.DLL"
    assert ntdll.to_dict()["base_addr_hex"] == "0x7ff83a200000"

    # Range containment, and the same table through the engine facade.
    assert game.contains(0x7FF61A5B1000)
    assert not game.contains(game.end_addr)
    assert not ntdll.contains(0x7FF61A5B1000)
    assert modules == CdbEngine.parse_modules(MODULE_LIST_TEXT)
    hit, owner = CdbEngine.verify_address(0x7FF61A5B1000, modules)
    assert hit
    assert owner is not None and owner.name == "game"
    assert CdbEngine.verify_address(0x1, modules) == (False, None)

    # The column header and the unloaded-modules section are not entries.
    assert parse_cdb_modules("start             end                 module name\n") == []
    assert parse_cdb_modules(MODULE_LIST_TEXT).__len__() == len(parse_cdb_modules(MODULE_LIST_TEXT))
    assert parse_cdb_modules("") == []


# ---------------------------------------------------------------------------
# 6. CrashReport serialization + grounded receipt payload
# ---------------------------------------------------------------------------


def test_crash_report_to_dict_to_json_and_receipt_payload():
    report = CrashReport(
        exception_code=0xC0000005,
        exception_name=EXCEPTION_NAMES[0xC0000005],
        faulting_ip=_EXCEPTION_ADDRESS,
        faulting_address=0x7FF61A5B0000,
        registers=parse_cdb_registers(REGISTER_DUMP),
        stack_trace=parse_cdb_call_stack(CALL_STACK_TEXT),
        modules=parse_cdb_modules(MODULE_LIST_TEXT),
        parameters=[0x7FF61A5B0000, 1],
        faulting_operation="write",
        thread_id=0x1A2B,
        source="minidump",
        dump_path=r"C:\dumps\game_crash.dmp",
    )

    payload = report.to_dict()
    assert payload["exception_code"] == 0xC0000005
    assert payload["exception_code_hex"] == "0xc0000005"
    assert payload["exception_name"] == "STATUS_ACCESS_VIOLATION"
    assert payload["faulting_ip"] == _EXCEPTION_ADDRESS
    assert payload["faulting_ip_hex"] == "0x7ff61a5c1234"
    assert payload["faulting_address"] == 0x7FF61A5B0000
    assert payload["faulting_address_hex"] == "0x7ff61a5b0000"
    assert payload["faulting_operation"] == "write"
    assert payload["faulting_module"] == "game"
    assert payload["faulting_module_offset"] == "0x11234"
    assert payload["thread_id"] == 0x1A2B
    assert payload["parameters"] == [0x7FF61A5B0000, 1]
    assert payload["registers"] == report.registers.to_dict()
    assert payload["registers"]["rax_hex"] == "0x42"
    assert len(payload["stack_trace"]) == 4
    assert payload["stack_depth"] == 4
    assert payload["module_count"] == 3
    assert [module["name"] for module in payload["modules"]] == ["game", "ntdll", "KERNEL32"]
    assert payload["threads"] == []
    assert payload["system_info"] == {}
    assert payload["has_exception_stream"] is True
    assert payload["source"] == "minidump"
    assert payload["dump_path"] == r"C:\dumps\game_crash.dmp"

    assert json.loads(report.to_json()) == payload
    assert report.faulting_module() is report.modules[0]

    receipt = report.receipt_payload()
    assert receipt["phase"] == CDB_PHASE == "cdb_trace"
    assert receipt["verifier_status"] == "GROUNDED"
    assert receipt["ground_type"] == "CDB_CRASH_DUMP"
    assert receipt["crash"] == payload

    live = report.receipt_payload(ground_type="CDB_LIVE_ATTACH")
    assert live["ground_type"] == "CDB_LIVE_ATTACH"
    assert live["phase"] == "cdb_trace"
    assert live["verifier_status"] == "GROUNDED"
