import struct
from pathlib import Path
from click.testing import CliRunner
from toothpick.cli import cli
from toothpick.static_engine import StaticEngine
from toothpick.storage import Database, EdgeKind


def _build_minimal_pe_with_code(code_bytes: bytes, machine: int = 0x8664) -> bytes:
    dos_header = bytearray(b"MZ" + b"\x00" * 58)
    dos_header += struct.pack("<I", 0x40)
    pe_sig = b"PE\x00\x00"
    file_header = struct.pack("<HHIIIHH", machine, 1, 0, 0, 0, 0, 0x0022)
    optional_size = 0xF0 if machine == 0x8664 else 0xE0
    magic = 0x20B if machine == 0x8664 else 0x10B
    optional_header = bytearray(optional_size)
    struct.pack_into("<H", optional_header, 0, magic)
    struct.pack_into("<I", optional_header, 16, 0x1000)  # EntryPoint RVA
    if machine == 0x8664:
        struct.pack_into("<Q", optional_header, 24, 0x140000000)  # ImageBase
    else:
        struct.pack_into("<I", optional_header, 28, 0x400000)
    struct.pack_into("<I", optional_header, 32, 0x1000)  # SectionAlignment
    struct.pack_into("<I", optional_header, 36, 0x200)   # FileAlignment
    struct.pack_into("<I", optional_header, 56, 0x3000)  # SizeOfImage

    # Section Header (.text)
    section_header = bytearray(40)
    section_header[0:5] = b".text"
    struct.pack_into("<IIII", section_header, 8, max(len(code_bytes), 0x1000), 0x1000, len(code_bytes), 0x200)
    struct.pack_into("<I", section_header, 36, 0x60000020)  # CODE | EXECUTE | READ

    headers = bytes(dos_header) + pe_sig + file_header + bytes(optional_header) + bytes(section_header)
    padding = b"\x00" * (0x200 - len(headers))
    return headers + padding + code_bytes


def test_static_x64_disassembly_and_calls(tmp_path: Path) -> None:
    # A function at 0x1000 that calls 0x1020 and returns
    # 0x1000: 48 83 ec 28         sub rsp, 0x28
    # 0x1004: e8 17 00 00 00      call 0x1020 (rel 0x17 from next ip 0x1009)
    # 0x1009: 48 83 c4 28         add rsp, 0x28
    # 0x100d: c3                  ret
    # ...
    # 0x1020: 31 c0               xor eax, eax
    # 0x1022: c3                  ret
    code = (
        b"\x48\x83\xec\x28"
        b"\xe8\x17\x00\x00\x00"
        b"\x48\x83\xc4\x28"
        b"\xc3"
        + b"\x90" * 18
        + b"\x31\xc0\xc3"
    )
    target = tmp_path / "target_x64.exe"
    target.write_bytes(_build_minimal_pe_with_code(code, machine=0x8664))
    db_path = tmp_path / "toothpick.db"

    engine = StaticEngine(db_path=db_path)
    result = engine.analyze_binary(target)

    assert result["function_count"] >= 2
    assert result["edge_count"] >= 1

    db = Database(db_path)
    sha = result["binary_sha256"]
    callees = db.get_callees(sha, 0x140001000)
    assert 0x140001020 in callees


def test_static_arm64_raw_disassembly(tmp_path: Path) -> None:
    # ARM64:
    # sub sp, sp, #0x10 (d10043ff)
    # bl #+16           (94000004) -> calls offset +16
    # add sp, sp, #0x10 (910043ff)
    # ret               (d65f03c0)
    # nop               (d503201f)
    # ret               (d65f03c0)
    code = (
        b"\xff\x43\x00\xd1"
        b"\x04\x00\x00\x94"
        b"\xff\x43\x00\x91"
        b"\xc0\x03\x5f\xd6"
        b"\x1f\x20\x03\xd5"
        b"\xc0\x03\x5f\xd6"
    )
    target = tmp_path / "raw_arm64.bin"
    target.write_bytes(code)
    db_path = tmp_path / "toothpick.db"

    engine = StaticEngine(db_path=db_path)
    result = engine.analyze_binary(target, is_raw=True, raw_arch="arm64", raw_base=0x80000000)

    assert result["function_count"] >= 1
    assert result["arch"] == "arm64"


def test_static_1000_func_stress_replay(tmp_path: Path) -> None:
    # Generate 1,000 synthetic functions in a raw binary
    # Each function: xor eax, eax (31 c0) ; ret (c3) + 13 bytes padding = 16 bytes each
    one_func = b"\x31\xc0\xc3" + b"\x90" * 13
    total_funcs = 1000
    blob = one_func * total_funcs

    target = tmp_path / "stress_1k.bin"
    target.write_bytes(blob)
    db_path = tmp_path / "toothpick.db"

    engine = StaticEngine(db_path=db_path)
    result1 = engine.analyze_binary(target, is_raw=True, raw_arch="x86_64", raw_base=0x10000)
    assert result1["function_count"] == 1000

    db = Database(db_path)
    sha = result1["binary_sha256"]
    assert db.count_functions(sha) == 1000

    # Test idempotent kill-and-replay re-analysis
    result2 = engine.analyze_binary(target, is_raw=True, raw_arch="x86_64", raw_base=0x10000)
    assert result2["function_count"] == 1000
    assert db.count_functions(sha) == 1000


def test_static_cli_command(tmp_path: Path) -> None:
    code = b"\x31\xc0\xc3"
    target = tmp_path / "cli_test.exe"
    target.write_bytes(_build_minimal_pe_with_code(code))
    db_path = tmp_path / "cli.db"

    runner = CliRunner()
    res = runner.invoke(cli, ["static", str(target), "--db", str(db_path)])
    assert res.exit_code == 0
    assert '"function_count":' in res.output
    assert db_path.exists()


def test_static_if_else_worklist_and_cfg_edges(tmp_path: Path) -> None:
    # 0x1000: 83 f9 00         cmp ecx, 0
    # 0x1003: 74 07            jz +7 (to 0x100c: else block)
    # 0x1005: b8 01 00 00 00   mov eax, 1 (then block)
    # 0x100a: eb 05            jmp +5 (to 0x1011: merge)
    # 0x100c: b8 02 00 00 00   mov eax, 2 (else block)
    # 0x1011: c3               ret
    code = (
        b"\x83\xf9\x00"
        b"\x74\x07"
        b"\xb8\x01\x00\x00\x00"
        b"\xeb\x05"
        b"\xb8\x02\x00\x00\x00"
        b"\xc3"
    )
    target = tmp_path / "if_else.exe"
    target.write_bytes(_build_minimal_pe_with_code(code, machine=0x8664))
    db_path = tmp_path / "if_else.db"

    engine = StaticEngine(db_path=db_path)
    result = engine.analyze_binary(target)

    # Must NOT be prematurely truncated by the 'jmp +5'
    assert result["function_count"] == 1
    assert result["instruction_count"] == 6

    db = Database(db_path)
    sha = result["binary_sha256"]
    fn = db.get_function(sha, 0x140001000)
    assert fn is not None
    assert fn.size == len(code)

    # Check CFG edges in database: must connect jump instruction to target
    edges = db.get_edges(sha)
    cfg_edges = [e for e in edges if e.kind == EdgeKind.CFG.value]
    # jz to else (0x140001003 -> 0x14000100c)
    assert any(e.src_addr == 0x140001003 and e.dst_addr == 0x14000100C for e in cfg_edges)
    # jz fallthrough (0x140001003 -> 0x140001005)
    assert any(e.src_addr == 0x140001003 and e.dst_addr == 0x140001005 for e in cfg_edges)
    # jmp to merge (0x14000100a -> 0x140001011)
    assert any(e.src_addr == 0x14000100A and e.dst_addr == 0x140001011 for e in cfg_edges)


def test_static_ssa_reaching_definitions_kill(tmp_path: Path) -> None:
    # 0: mov ecx, 10   (def ecx#0)
    # 1: mov eax, ecx  (read ecx#0 -> use of def 0)
    # 2: mov ecx, 20   (def ecx#2, kills def 0)
    # 3: add eax, ecx  (read ecx#2 -> use of def 2, NOT def 0!)
    # 4: ret
    code = (
        b"\xb9\x0a\x00\x00\x00"  # mov ecx, 10
        b"\x89\xc8"              # mov eax, ecx
        b"\xb9\x14\x00\x00\x00"  # mov ecx, 20
        b"\x01\xc8"              # add eax, ecx
        b"\xc3"                  # ret
    )
    target = tmp_path / "ssa_kill.exe"
    target.write_bytes(_build_minimal_pe_with_code(code, machine=0x8664))
    db_path = tmp_path / "ssa_kill.db"

    engine = StaticEngine(db_path=db_path)
    result = engine.analyze_binary(target)

    db = Database(db_path)
    sha = result["binary_sha256"]
    facts = db.get_ssa_facts(sha, 0x140001000)

    # Definition of ecx at def_point 0 must only have use_point 1, NOT 3
    def_0 = [f for f in facts if f["def_point"] == 0 and f["var_name"] in ("rcx", "ecx")]
    assert len(def_0) >= 1
    assert def_0[0]["use_points"] == [1]

    # Definition of ecx at def_point 2 must have use_point 3
    def_2 = [f for f in facts if f["def_point"] == 2 and f["var_name"] in ("rcx", "ecx")]
    assert len(def_2) >= 1
    assert def_2[0]["use_points"] == [3]


def test_static_playstation_self_ingest(tmp_path: Path) -> None:
    # Build minimal ELF64 (x86_64) payload
    ehdr = bytearray(64)
    ehdr[0:4] = b"\x7fELF"
    ehdr[4] = 2  # 64-bit
    ehdr[5] = 1  # little-endian
    ehdr[6] = 1  # version
    struct.pack_into("<HHIQQQIHHHHHH", ehdr, 16, 2, 0x3E, 1, 0x400000, 64, 0, 0, 64, 56, 1, 0, 0, 0)

    phdr = bytearray(56)
    struct.pack_into("<IIQQQQQQ", phdr, 0, 1, 5, 0x100, 0x400000, 0x400000, 16, 16, 0x1000)

    code = b"\x31\xc0\xc3" + b"\x90" * 13
    elf_bytes = bytes(ehdr) + bytes(phdr) + b"\x00" * (0x100 - len(ehdr) - len(phdr)) + code

    # PlayStation SELF container wraps ELF with SCE magic
    self_bytes = b"SCE\x00" + b"\x00" * 28 + elf_bytes
    target = tmp_path / "game.self"
    target.write_bytes(self_bytes)
    db_path = tmp_path / "self.db"

    engine = StaticEngine(db_path=db_path)
    result = engine.analyze_binary(target)

    assert result["container"] == "self"
    assert result["arch"] == "x86_64"
    assert result["function_count"] >= 1
    assert result["instruction_count"] >= 1


def test_static_gamecube_dol_ingest(tmp_path: Path) -> None:
    # PPC code: li r3, 0 (0x38600000) ; blr (0x4e800020)
    ppc_code = b"\x38\x60\x00\x00\x4e\x80\x00\x20"

    dol_header = bytearray(0x100)
    struct.pack_into(">I", dol_header, 0, 0x100)            # text_offset[0]
    struct.pack_into(">I", dol_header, 0x48, 0x80003100)    # text_address[0]
    struct.pack_into(">I", dol_header, 0x90, len(ppc_code)) # text_size[0]
    struct.pack_into(">I", dol_header, 0xE0, 0x80003100)    # entry point

    target = tmp_path / "boot.dol"
    target.write_bytes(bytes(dol_header) + ppc_code)
    db_path = tmp_path / "dol.db"

    engine = StaticEngine(db_path=db_path)
    result = engine.analyze_binary(target)

    assert result["container"] == "dol"
    assert result["arch"] == "ppc"
    assert result["function_count"] >= 1
    assert result["instruction_count"] == 2


def test_static_nso_lz4_decompression(tmp_path: Path) -> None:
    import lz4.block

    # ARM64 code: mov x0, #0 (d2800000) ; ret (d65f03c0)
    arm64_code = b"\x00\x00\x80\xd2\xc0\x03\x5f\xd6"
    compressed = lz4.block.compress(arm64_code, store_size=False)

    nso_header = bytearray(0x100)
    nso_header[0:4] = b"NSO0"
    struct.pack_into("<I", nso_header, 0x0C, 1)                  # flags: bit 0 = compressed .text
    struct.pack_into("<I", nso_header, 0x14, len(arm64_code))    # text decompressed size
    struct.pack_into("<I", nso_header, 0x40, 0x100)               # text file offset
    struct.pack_into("<I", nso_header, 0x48, len(compressed))    # text file size

    target = tmp_path / "main.nso"
    target.write_bytes(bytes(nso_header) + compressed)
    db_path = tmp_path / "nso.db"

    engine = StaticEngine(db_path=db_path)
    result = engine.analyze_binary(target)

    assert result["container"] == "nso"
    assert result["arch"] == "arm64"
    assert result["function_count"] >= 1
    assert result["instruction_count"] == 2
    assert any("NSO .text successfully decompressed with LZ4" in n for n in result["notes"])
