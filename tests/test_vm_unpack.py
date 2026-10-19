"""W19 VM-unpacker tests: entropy, protector triage, OEP hunt, PE rebuild (chunks).

All fixtures are synthetic: random-vs-uniform blobs, marker-carrying files,
and a hand-built minimal PE64 — no protected binaries, no SDK.
"""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path

from click.testing import CliRunner

from toothpick.cli import cli
from toothpick.unpack import (
    OepHunter,
    PeReconstructor,
    VmProtectionType,
    compute_shannon_entropy,
    detect_vm_protection,
    parse_pe,
)

_IMAGE_BASE = 0x140000000


def _minimal_pe64(
    code: bytes = b"\x55\x48\x89\xe5\xc3",
    marker_section: bytes = b".vmp0\x00\x00\x00",
    marker_blob: bytes = b"VMProtect",
) -> bytes:
    """Minimal PE64: DOS + COFF + optional + .text (prologue) + marker section."""
    dos = bytearray(0x40)
    dos[0:2] = b"MZ"
    struct.pack_into("<I", dos, 0x3C, 0x40)
    coff = struct.pack("<HHIIIHH", 0x8664, 2, 0, 0, 0, 0xF0, 0x22)
    opt = bytearray(0xF0)
    struct.pack_into("<H", opt, 0, 0x20B)  # PE32+
    struct.pack_into("<I", opt, 16, 0x1000)  # entry RVA
    struct.pack_into("<Q", opt, 24, _IMAGE_BASE)
    struct.pack_into("<I", opt, 32, 0x1000)  # section alignment
    struct.pack_into("<I", opt, 36, 0x200)  # file alignment
    struct.pack_into("<I", opt, 56, 0x3000)  # size of image
    headers = bytes(dos) + b"PE\x00\x00" + coff + bytes(opt)
    text_off = 0x200
    vmp_off = 0x400
    sec_text = b".text\x00\x00\x00" + struct.pack(
        "<IIIIIIHHI", 0x1000, 0x1000, 0x200, text_off, 0, 0, 0, 0, 0x60000020
    )
    sec_vmp = bytes(marker_section).ljust(8, b"\x00") + struct.pack(
        "<IIIIIIHHI", 0x1000, 0x2000, 0x200, vmp_off, 0, 0, 0, 0, 0x40000040
    )
    image = bytearray(max(vmp_off + 0x200, len(headers) + 80))
    image[0 : len(headers)] = headers
    image[len(headers) : len(headers) + 80] = sec_text + sec_vmp
    image[text_off : text_off + len(code)] = code
    image[vmp_off : vmp_off + len(marker_blob)] = marker_blob
    return bytes(image)


def test_shannon_entropy_uniform_vs_random() -> None:
    """Uniform blobs score ~0; random blobs score high."""
    assert compute_shannon_entropy(b"\x00" * 4096) == 0.0
    assert compute_shannon_entropy(b"") == 0.0
    random_blob = os.urandom(8192)
    assert compute_shannon_entropy(random_blob) > 7.0
    assert compute_shannon_entropy(b"AB" * 2048) < 2.0


def test_vm_protector_detection_fixtures(tmp_path: Path) -> None:
    """Section names and marker strings identify the protector family."""
    assert detect_vm_protection(b"MZ" + b"\x00" * 64) is None
    vmp = detect_vm_protection(_minimal_pe64())
    assert vmp is not None
    assert vmp.protector == VmProtectionType.VMPROTECT
    assert vmp.confidence > 0
    assert vmp.to_dict()["protector"] == "vmprotect"

    themida = detect_vm_protection(
        _minimal_pe64(marker_section=b".themida", marker_blob=b"Themida")
    )
    assert themida is not None
    assert themida.protector == VmProtectionType.THEMIDA

    denuvo = detect_vm_protection(
        _minimal_pe64(marker_section=b".denuvo\x00", marker_blob=b"Denuvo")
    )
    assert denuvo is not None
    assert denuvo.protector == VmProtectionType.DENUVO


def test_oep_hunter_discovers_prologue(tmp_path: Path) -> None:
    """The x86_64 prologue at .text start is reported as the OEP."""
    blob = _minimal_pe64()
    pe = parse_pe(blob)
    assert pe is not None
    hunter = OepHunter()
    found = hunter.hunt_oep(blob, pe)
    assert found == _IMAGE_BASE + 0x1000
    candidates = hunter.scan(blob, pe)
    assert candidates, "expected at least one OEP candidate"
    assert candidates[0].address == _IMAGE_BASE + 0x1000


def test_pe_dump_reconstruction_round_trip(tmp_path: Path) -> None:
    """A rebuilt dump re-parses with headers, sections, and entry intact."""
    blob = _minimal_pe64()
    pe = parse_pe(blob)
    assert pe is not None
    rebuilt = PeReconstructor.reconstruct_dump(blob, pe.image_base, 0x1000)
    assert rebuilt[:2] == b"MZ"
    fixed = parse_pe(rebuilt)
    assert fixed is not None
    assert fixed.entry_point_rva == 0x1000
    assert len(fixed.sections) == len(pe.sections)
    assert fixed.size_of_image > 0


def test_unpack_detect_vm_cli_json(tmp_path: Path) -> None:
    """``unpack --detect-vm`` triages without PACKED_ABORT and emits JSON."""
    target = tmp_path / "protected.exe"
    target.write_bytes(_minimal_pe64())
    runner = CliRunner()
    result = runner.invoke(cli, ["unpack", str(target), "--detect-vm"])
    assert result.exit_code == 0, f"unpack --detect-vm failed: {result.output}"
    payload = json.loads(result.output)
    assert payload["command"] == "unpack"
    assert payload["vm_detection"]["protector"] == "vmprotect"
    assert payload["vm_protected"] is True

    helped = runner.invoke(cli, ["unpack", "--help"])
    assert helped.exit_code == 0
    assert "--detect-vm" in helped.output
    assert "--hunt-oep" in helped.output
    assert "--reconstruct-pe" in helped.output
