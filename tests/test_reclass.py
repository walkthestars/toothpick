"""W18 ReClass automation tests: inference, exporters, CLI (UNIT-W18 chunks).

All fixtures are synthetic byte buffers — no game, no ReClass.NET install.
"""

from __future__ import annotations

import json
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

from click.testing import CliRunner

from toothpick.cli import cli
from toothpick.reclass import (
    StructLayout,
    StructSynthesizer,
)

_BASE = 0x140500000
_MODULE_RANGES = [(0x7FF600000000, 0x7FF600100000)]


def _mixed_buffer() -> bytes:
    """vtable ptr, float, int32, pointer, string, zero tail (48 bytes)."""
    blob = bytearray(48)
    struct.pack_into("<Q", blob, 0x00, 0x7FF600001000)  # vtable candidate
    struct.pack_into("<f", blob, 0x08, 1.5)  # float
    struct.pack_into("<i", blob, 0x0C, 1337)  # int32
    struct.pack_into("<Q", blob, 0x10, 0x7FF600002000)  # pointer
    blob[0x18 : 0x18 + 7] = b"Player\x00"  # ASCII string
    return bytes(blob)


def _synthesize(name: str = "PlayerEntity") -> StructLayout:
    synth = StructSynthesizer(module_ranges=_MODULE_RANGES, base_address=_BASE)
    return synth.synthesize_from_memory(_mixed_buffer(), base_addr=_BASE, name=name)


def test_field_type_inference_on_mixed_buffer() -> None:
    """vtable, float, pointer, and int fields are recovered with full coverage."""
    layout = _synthesize()
    assert layout.total_size == 48
    assert layout.fields, "expected inferred fields"
    layout.validate()
    covered = sum(field.size for field in layout.fields)
    assert covered == 48, f"every byte must be accounted for, got {covered}"
    by_offset = {field.offset: field for field in layout.fields}
    assert by_offset[0].field_type == "vtable"
    assert by_offset[0].size == 8
    kinds = {field.field_type for field in layout.fields}
    assert "float" in kinds
    assert "pointer" in kinds


def test_reclass_xml_valid_schema() -> None:
    """The XML parses and carries the ReClass.NET Class/Node shape."""
    layout = _synthesize()
    root = ET.fromstring(layout.to_reclass_xml())
    assert root.tag == "RCNetProject"
    cls = root.find("./Classes/Class")
    assert cls is not None
    assert cls.get("Name") == "PlayerEntity"
    nodes = cls.findall("./Nodes/Node")
    assert len(nodes) == len(layout.fields)
    assert all(node.get("Name") and node.get("Type") for node in nodes)


def test_cpp_header_offsets_and_size_gate() -> None:
    """C++ output has packing, offset comments, and a static_assert."""
    layout = _synthesize()
    header = layout.to_cpp()
    assert "#pragma pack(push, 1)" in header
    assert "struct PlayerEntity" in header
    assert "/* 0x0000 */" in header
    assert "static_assert(sizeof(PlayerEntity) == 0x30" in header


def test_csharp_explicit_layout() -> None:
    """C# output uses explicit layout with per-field offsets."""
    layout = _synthesize()
    source = layout.to_csharp()
    assert "LayoutKind.Explicit" in source
    assert "public struct PlayerEntity" in source
    assert "[FieldOffset(0x0000)]" in source


def test_struct_cli_end_to_end_on_buffer_file(tmp_path: Path) -> None:
    """``struct --file`` synthesizes, emits JSON, and ingests a types row."""
    dump = tmp_path / "entity.bin"
    dump.write_bytes(_mixed_buffer())
    db_path = tmp_path / "reclass.db"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "struct",
            "--file",
            str(dump),
            "--base",
            f"0x{_BASE:x}",
            "--size",
            "0x30",
            "--name",
            "PlayerEntity",
            "--format",
            "json",
            "--db",
            str(db_path),
            "--sha",
            "c" * 64,
        ],
    )
    assert result.exit_code == 0, f"struct failed: {result.output}"
    payload = json.loads(result.output)
    assert payload["command"] == "struct"
    assert payload["layout"]["total_size"] == 48
    assert payload["receipt"]["phase"] == "struct_synth"

    helped = runner.invoke(cli, ["struct", "--help"])
    assert helped.exit_code == 0
    for flag in ("--file", "--size", "--name", "--format", "--out"):
        assert flag in helped.output
