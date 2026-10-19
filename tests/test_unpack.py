from pathlib import Path
import pytest
from toothpick.triage import PackedAbortError
from toothpick.unpack import (
    UnpackMethod,
    UnpackStatus,
    _build_unpacked_pe,
    _nrv2_decode,
    emulate_tailjump_oep,
    parse_pe,
    unpack_binary,
)


def _build_minimal_pe(extra_bytes: bytes = b"") -> bytes:
    import struct
    dos_header = bytearray(b"MZ" + b"\x00" * 58)
    dos_header += struct.pack("<I", 0x40)
    pe_sig = b"PE\x00\x00"
    file_header = struct.pack("<HHIIIHH", 0x8664, 1, 0, 0, 0, 0, 0x0002)
    return bytes(dos_header) + pe_sig + file_header + extra_bytes


def test_unpack_already_unpacked(tmp_path: Path) -> None:
    pe = tmp_path / "plain.exe"
    pe.write_bytes(_build_minimal_pe())
    report = unpack_binary(pe)
    assert report.status == UnpackStatus.ALREADY_UNPACKED


def test_unpack_vm_protected_aborts(tmp_path: Path) -> None:
    pe_vmp = tmp_path / "protected.exe"
    pe_vmp.write_bytes(_build_minimal_pe(b".vmp0\x00\x00\x00VMProtect code"))
    with pytest.raises(PackedAbortError, match="PACKED_ABORT\\(VM_PROTECTED\\)"):
        unpack_binary(pe_vmp)


def test_unpack_unicorn_micro_emulation_tailjump() -> None:
    # A small x86_64 packer stub:
    # 0x1000: XOR loop deobfuscator
    # 0x1007: JMP 0x2000 (tail jump to OEP)
    # Machine code:
    #   31 c0                   xor eax, eax
    #   ff c0                   inc eax
    #   48 b8 00 20 00 00 00 00 00 00   mov rax, 0x2000
    #   ff e0                   jmp rax
    stub_code = b"\x31\xc0\xff\xc0\x48\xb8\x00\x20\x00\x00\x00\x00\x00\x00\xff\xe0"
    oep = emulate_tailjump_oep(
        code=stub_code,
        load_addr=0x1000,
        stub_start=0x1000,
        stub_size=0x100,
        max_instructions=100,
    )
    assert oep == 0x2000


def test_unpack_simple_xor_payload(tmp_path: Path) -> None:
    # An XOR-encoded payload file with single-byte key 0x5A
    plain_content = b"MZ" + b"\x00" * 64 + b"UNPACKED_PAYLOAD_OK"
    key = 0x5A
    obfuscated = bytes([b ^ key for b in plain_content])
    
    target = tmp_path / "xor_packed.bin"
    target.write_bytes(obfuscated)
    out_file = tmp_path / "xor_unpacked.bin"

    report = unpack_binary(target, output_path=out_file, xor_key=key)
    assert report.status == UnpackStatus.UNPACKED
    assert report.method == UnpackMethod.STATIC_XOR
    assert out_file.exists()
    assert out_file.read_bytes() == plain_content
    # PLAN-W02 wire contract: only "static"/"unicorn_tailjump" may be emitted.
    assert report.method_wire == "static"
    assert report.to_dict()["method"] == "static"
    assert report.method_detail == "static_xor"


IMAGE_BASE = 0x140000000
UPX_PAYLOAD_RVA = 0x1000
UPX_STUB_RVA = 0x2000


def _build_upx_packed_pe(payload: bytes) -> bytes:
    """Synthetic UPX-packed PE32+ image for static/stub recovery tests.

    The packer section (``UPX1``) mirrors UPX's own layout: the compressed
    bitstream first, then the decompression stub, then the 32-byte ``UPX!``
    pack header that the static unpacker keys on.
    """
    import struct
    import zlib

    compressed = zlib.compress(payload)
    # mov rax, <OEP in UPX0> ; jmp rax   -> packer stub ending in a tail jump.
    stub = b"\x48\xb8" + struct.pack("<Q", IMAGE_BASE + UPX_PAYLOAD_RVA) + b"\xff\xe0"
    pack_header = bytearray(32)
    pack_header[0:4] = b"UPX!"
    pack_header[4] = 13  # version
    pack_header[5] = 1   # format
    pack_header[6] = 1   # method (raw zlib stream)
    pack_header[7] = 5   # level
    struct.pack_into("<I", pack_header, 16, len(payload))
    struct.pack_into("<I", pack_header, 20, len(compressed))
    struct.pack_into("<I", pack_header, 24, len(payload))
    section_raw = compressed + stub + bytes(pack_header)

    e_lfanew = 0x40
    optional_size = 0xF0
    raw_ptr = 0x200
    image = bytearray(raw_ptr + len(section_raw))

    image[0:2] = b"MZ"
    struct.pack_into("<I", image, 0x3C, e_lfanew)
    image[e_lfanew : e_lfanew + 4] = b"PE\x00\x00"
    struct.pack_into(
        "<HHIIIHH", image, e_lfanew + 4, 0x8664, 2, 0, 0, 0, optional_size, 0x0022
    )

    optional = e_lfanew + 24
    struct.pack_into("<H", image, optional, 0x20B)
    struct.pack_into("<I", image, optional + 16, UPX_STUB_RVA + len(compressed))
    struct.pack_into("<Q", image, optional + 24, IMAGE_BASE)
    struct.pack_into("<I", image, optional + 32, 0x1000)  # section alignment
    struct.pack_into("<I", image, optional + 36, 0x200)   # file alignment
    struct.pack_into("<I", image, optional + 56, 0x3000)  # size of image

    section_table = optional + optional_size
    struct.pack_into(
        "<8sIIIIII", image, section_table, b"UPX0", 0x1000, UPX_PAYLOAD_RVA, 0, 0, 0, 0
    )
    struct.pack_into("<I", image, section_table + 36, 0xE0000020)
    second = section_table + 40
    struct.pack_into(
        "<8sIIIIII", image, second, b"UPX1", 0x1000, UPX_STUB_RVA, len(section_raw), raw_ptr, 0, 0
    )
    struct.pack_into("<I", image, second + 36, 0xE0000020)

    image[raw_ptr : raw_ptr + len(section_raw)] = section_raw
    return bytes(image)


def test_unpack_upx_static_recovery(tmp_path: Path) -> None:
    payload = b"MZ" + bytes(range(256)) * 4
    target = tmp_path / "packed_upx.exe"
    target.write_bytes(_build_upx_packed_pe(payload))
    out_file = tmp_path / "unpacked_upx.exe"

    report = unpack_binary(target, output_path=out_file, emulate_stub=False)

    assert report.status == UnpackStatus.UNPACKED
    assert report.packer == "upx"
    assert report.method == UnpackMethod.STATIC
    assert report.method_wire == "static"
    assert report.codec == "zlib"
    assert report.payload_size == len(payload)
    assert out_file.exists()

    unpacked_bytes = out_file.read_bytes()
    unpacked = parse_pe(unpacked_bytes)
    assert unpacked is not None
    assert unpacked.entry_point_rva == UPX_PAYLOAD_RVA
    upx0 = next(s for s in unpacked.sections if s.name == "UPX0")
    upx1 = next(s for s in unpacked.sections if s.name == "UPX1")
    assert upx1.raw_size == 0
    assert unpacked_bytes[upx0.raw_ptr : upx0.raw_ptr + len(payload)] == payload


def test_unpack_upx_tailjump_stub_recovery(tmp_path: Path) -> None:
    payload = b"MZ" + bytes(range(256)) * 4
    target = tmp_path / "packed_upx_emul.exe"
    target.write_bytes(_build_upx_packed_pe(payload))
    out_file = tmp_path / "unpacked_upx_emul.exe"

    report = unpack_binary(target, output_path=out_file)

    assert report.status == UnpackStatus.UNPACKED
    assert report.method == UnpackMethod.UNICORN_TAILJUMP
    assert report.method_wire == "unicorn_tailjump"
    assert report.oep == IMAGE_BASE + UPX_PAYLOAD_RVA
    assert report.tailjumps
    assert report.instruction_count >= 1
    assert unpacked_bytes_ok(out_file.read_bytes(), payload)


def unpacked_bytes_ok(unpacked_bytes: bytes, payload: bytes) -> bool:
    unpacked = parse_pe(unpacked_bytes)
    if unpacked is None:
        return False
    upx0 = next((s for s in unpacked.sections if s.name == "UPX0"), None)
    if upx0 is None:
        return False
    return unpacked_bytes[upx0.raw_ptr : upx0.raw_ptr + len(payload)] == payload


def test_unpack_cli_command(tmp_path: Path) -> None:
    from click.testing import CliRunner
    from toothpick.cli import cli

    pe = tmp_path / "plain_cli.exe"
    pe.write_bytes(_build_minimal_pe())
    runner = CliRunner()
    result = runner.invoke(cli, ["unpack", str(pe)])
    assert result.exit_code == 0
    assert '"status": "already_unpacked"' in result.output


def test_unpack_nrv2_variants_decoding() -> None:
    class NRV2Encoder:
        def __init__(self) -> None:
            self.output = bytearray()
            self.ctrl_bits: list[int] = []
            self.pending_bytes = bytearray()

        def add_bit(self, b: int) -> None:
            self.ctrl_bits.append(b & 1)
            if len(self.ctrl_bits) == 8:
                self._flush_ctrl()

        def add_byte(self, b: int) -> None:
            self.pending_bytes.append(b)

        def _flush_ctrl(self) -> None:
            val = 0
            for b in self.ctrl_bits:
                val = (val << 1) | b
            self.output.append(val)
            self.output.extend(self.pending_bytes)
            self.ctrl_bits = []
            self.pending_bytes = bytearray()

        def finish(self) -> bytes:
            if self.ctrl_bits or self.pending_bytes:
                while len(self.ctrl_bits) < 8:
                    self.ctrl_bits.append(0)
                self._flush_ctrl()
            return bytes(self.output)

    # 1. NRV2B
    enc_b = NRV2Encoder()
    for ch in b"XY":
        enc_b.add_bit(1)
        enc_b.add_byte(ch)
    enc_b.add_bit(0)
    enc_b.add_bit(1)
    enc_b.add_bit(0)
    enc_b.add_byte(2)  # offset 2
    enc_b.add_bit(1)
    enc_b.add_bit(0)  # len 3
    enc_b.add_bit(0)
    enc_b.add_bit(1)
    enc_b.add_bit(0)
    enc_b.add_byte(0)  # end marker
    dec_b = _nrv2_decode(enc_b.finish(), "nrv2b")
    assert dec_b == b"XYXYX"

    # 2. NRV2D with m_off == 2 branch
    enc_d = NRV2Encoder()
    for ch in b"HELLO":
        enc_d.add_bit(1)
        enc_d.add_byte(ch)
    enc_d.add_bit(0)
    enc_d.add_bit(0)
    enc_d.add_bit(0)  # m_off == 2
    for i in range(7, -1, -1):
        enc_d.add_bit((4 >> i) & 1)  # offset 4 + 1 = 5
    enc_d.add_bit(1)
    enc_d.add_bit(0)  # len 3
    enc_d.add_bit(0)
    enc_d.add_bit(1)
    enc_d.add_bit(0)
    enc_d.add_byte(0)  # end marker
    dec_d = _nrv2_decode(enc_d.finish(), "nrv2d")
    assert dec_d == b"HELLOHEL"

    # 3. NRV2E with m_off == 2 branch
    enc_e = NRV2Encoder()
    for ch in b"WORLD":
        enc_e.add_bit(1)
        enc_e.add_byte(ch)
    enc_e.add_bit(0)
    enc_e.add_bit(0)
    enc_e.add_bit(0)  # m_off == 2
    for i in range(7, -1, -1):
        enc_e.add_bit((4 >> i) & 1)  # offset 4 + 1 = 5
    enc_e.add_bit(1)
    enc_e.add_bit(0)  # len 3
    enc_e.add_bit(0)
    enc_e.add_bit(1)
    enc_e.add_bit(0)
    enc_e.add_byte(0)  # end marker
    dec_e = _nrv2_decode(enc_e.finish(), "nrv2e")
    assert dec_e == b"WORLDWOR"


def test_unpack_teb_peb_emulation() -> None:
    # x86_64 stub: reads PEB from gs:[0x60], verifies non-null, tail jumps to 0x2000
    stub_64 = (
        b"\x65\x48\x8b\x04\x25\x60\x00\x00\x00"
        b"\x48\x85\xc0"
        b"\x48\xb8\x00\x20\x00\x00\x00\x00\x00\x00"
        b"\xff\xe0"
    )
    oep_64 = emulate_tailjump_oep(
        code=stub_64,
        load_addr=0x1000,
        stub_start=0x1000,
        stub_size=0x100,
        arch="x86_64",
    )
    assert oep_64 == 0x2000

    # x86 (32-bit) stub: reads PEB from fs:[0x30], verifies non-null, tail jumps to 0x3000
    stub_32 = (
        b"\x64\xa1\x30\x00\x00\x00"
        b"\x85\xc0"
        b"\xb8\x00\x30\x00\x00"
        b"\xff\xe0"
    )
    oep_32 = emulate_tailjump_oep(
        code=stub_32,
        load_addr=0x1000,
        stub_start=0x1000,
        stub_size=0x100,
        arch="x86",
    )
    assert oep_32 == 0x3000


def test_unpack_rebuilt_pe_lief_validation_and_data_dir_cleanup() -> None:
    import struct
    import lief

    payload = b"MZ" + bytes(range(256)) * 4
    raw_pe = bytearray(_build_upx_packed_pe(payload))
    struct.pack_into("<I", raw_pe, 0xC4, 16)
    struct.pack_into("<II", raw_pe, 0xD0, 0x2500, 0x100)

    pe_img = parse_pe(bytes(raw_pe))
    rebuilt = _build_unpacked_pe(bytes(raw_pe), pe_img, {pe_img.sections[1].index: b""}, 0x1000)

    rva, sz = struct.unpack_from("<II", rebuilt, 0xD0)
    assert rva == 0 and sz == 0

    parsed = lief.parse(list(rebuilt))
    assert parsed is not None
    assert parsed.header.machine == lief.PE.Header.MACHINE_TYPES.AMD64
    assert parsed.optional_header.addressof_entrypoint == 0x1000
