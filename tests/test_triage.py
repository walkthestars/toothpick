import struct
from pathlib import Path
import pytest
from toothpick.triage import (
    Architecture,
    ContainerType,
    DynamicCapability,
    EngineType,
    PackedAbortError,
    PackerAction,
    PackerStatus,
    UnknownFormatError,
    triage_binary,
)


def _build_minimal_pe(machine: int = 0x8664, extra_bytes: bytes = b"") -> bytes:
    """Builds a minimal valid PE header stub."""
    # DOS Header
    dos_header = bytearray(b"MZ" + b"\x00" * 58)
    e_lfanew = 0x40
    dos_header += struct.pack("<I", e_lfanew)
    # NT Headers
    pe_sig = b"PE\x00\x00"
    file_header = struct.pack("<HHIIIHH", machine, 1, 0, 0, 0, 0, 0x0002)
    return bytes(dos_header) + pe_sig + file_header + extra_bytes


@pytest.fixture
def fixtures_dir(tmp_path: Path) -> Path:
    fdir = tmp_path / "fixtures"
    fdir.mkdir()

    # 1. PE Native x64
    pe_native = fdir / "pe_native_x64.exe"
    pe_native.write_bytes(_build_minimal_pe(0x8664))

    # 2. PE Unity IL2CPP
    pe_unity = fdir / "pe_unity_il2cpp.exe"
    pe_unity.write_bytes(_build_minimal_pe(0x8664, b"GameAssembly.dll\x00il2cpp_init\x00"))

    # 3. PE Unreal Engine
    pe_ue = fdir / "pe_ue.exe"
    pe_ue.write_bytes(_build_minimal_pe(0x8664, b"UnrealEngine\x00FPlatformMemory\x00"))

    # 4. PE Godot
    pe_godot = fdir / "pe_godot.exe"
    pe_godot.write_bytes(_build_minimal_pe(0x8664, b"godot_engine\x00GDScript\x00"))

    # 5. NSO Nintendo Switch
    nso_switch = fdir / "main.nso"
    nso_switch.write_bytes(b"NSO0" + b"\x00" * 100)

    # 6. PE UPX packed
    pe_upx = fdir / "pe_upx.exe"
    pe_upx.write_bytes(_build_minimal_pe(0x8664, b"UPX0\x00\x00\x00\x00UPX1\x00\x00\x00\x00UPX!"))

    # 7. PE VMProtect abort
    pe_vmp = fdir / "pe_vmp.exe"
    pe_vmp.write_bytes(_build_minimal_pe(0x8664, b".vmp0\x00\x00\x00VMProtect begin"))

    return fdir


def test_triage_pe_native_x64(fixtures_dir: Path) -> None:
    report = triage_binary(fixtures_dir / "pe_native_x64.exe")
    assert report.container == ContainerType.PE
    assert report.arch == Architecture.X86_64
    assert report.engine == EngineType.NATIVE
    assert report.protection.status == PackerStatus.UNPACKED
    assert report.protection.action == PackerAction.PROCEED
    assert report.dynamic_capability == DynamicCapability.NATIVE_FRIDA


def test_triage_pe_unity_il2cpp(fixtures_dir: Path) -> None:
    report = triage_binary(fixtures_dir / "pe_unity_il2cpp.exe")
    assert report.container == ContainerType.PE
    assert report.arch == Architecture.X86_64
    assert report.engine == EngineType.UNITY_IL2CPP
    assert report.protection.status == PackerStatus.UNPACKED


def test_triage_pe_ue(fixtures_dir: Path) -> None:
    report = triage_binary(fixtures_dir / "pe_ue.exe")
    assert report.container == ContainerType.PE
    assert report.engine == EngineType.UNREAL


def test_triage_pe_godot(fixtures_dir: Path) -> None:
    report = triage_binary(fixtures_dir / "pe_godot.exe")
    assert report.container == ContainerType.PE
    assert report.engine == EngineType.GODOT


def test_triage_nso_switch(fixtures_dir: Path) -> None:
    report = triage_binary(fixtures_dir / "main.nso")
    assert report.container == ContainerType.NSO
    assert report.arch == Architecture.ARM64
    assert report.dynamic_capability == DynamicCapability.EMULATOR_BRIDGED


def test_triage_upx_packed(fixtures_dir: Path) -> None:
    report = triage_binary(fixtures_dir / "pe_upx.exe")
    assert report.container == ContainerType.PE
    assert report.protection.status == PackerStatus.PACKED_SIMPLE
    assert report.protection.packer == "upx"
    assert report.protection.action == PackerAction.UNPACK_REQUIRED


def test_triage_vmprotect_abort(fixtures_dir: Path) -> None:
    with pytest.raises(PackedAbortError, match="PACKED_ABORT\\(VM_PROTECTED\\)"):
        triage_binary(fixtures_dir / "pe_vmp.exe")


def test_triage_raw_blob(tmp_path: Path) -> None:
    raw_file = tmp_path / "firmware.bin"
    raw_file.write_bytes(b"\x12\x34\x56\x78" * 32)
    report = triage_binary(raw_file, is_raw=True, raw_arch="arm64", raw_base=0x80000000)
    assert report.container == ContainerType.RAW
    assert report.arch == Architecture.ARM64
    assert report.dynamic_capability == DynamicCapability.STATIC_ONLY


def test_triage_unknown_format_abort(tmp_path: Path) -> None:
    unknown_file = tmp_path / "unknown.dat"
    unknown_file.write_bytes(b"\x99\x88\x77\x66" * 8)
    with pytest.raises(UnknownFormatError, match="UNKNOWN_FORMAT_NEED_HINT"):
        triage_binary(unknown_file)


def test_triage_static_only_flag(fixtures_dir: Path) -> None:
    report = triage_binary(fixtures_dir / "pe_native_x64.exe", static_only=True)
    assert report.dynamic_capability == DynamicCapability.STATIC_ONLY


def test_triage_playstation_self(tmp_path: Path) -> None:
    self_file = tmp_path / "game.self"
    self_file.write_bytes(b"SCE\x00" + b"\x00" * 60)
    report = triage_binary(self_file)
    assert report.container == ContainerType.SELF
    assert report.arch == Architecture.X86_64
    assert report.dynamic_capability == DynamicCapability.EMULATOR_BRIDGED


def test_triage_gamecube_dol(tmp_path: Path) -> None:
    dol_file = tmp_path / "boot.dol"
    # Build minimal valid DOL header:
    # offset 0: text[0] file offset = 0x100
    # offset 0xE0: entry point = 0x80003100
    dol_header = bytearray(0x100)
    struct.pack_into(">I", dol_header, 0, 0x100)       # text_offset[0]
    struct.pack_into(">I", dol_header, 0x48, 0x80003100) # text_address[0]
    struct.pack_into(">I", dol_header, 0x90, 0x40)       # text_size[0]
    struct.pack_into(">I", dol_header, 0xE0, 0x80003100) # entry point
    dol_file.write_bytes(bytes(dol_header) + b"\x00" * 0x40)

    report = triage_binary(dol_file)
    assert report.container == ContainerType.DOL
    assert report.arch == Architecture.PPC
    assert report.dynamic_capability == DynamicCapability.EMULATOR_BRIDGED
