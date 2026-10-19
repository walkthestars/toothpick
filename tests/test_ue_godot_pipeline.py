"""
Tests for UNIT-W13: Unreal Engine & Godot Mod Pipelines (toothpick.mod_builder).

Validates:
1. UE4SS Lua mod scaffolding: enabled.txt, scripts/main.lua, boot tokens, hooks, grounding status.
2. UE4SS Lua hook variants: pre, post, replace.
3. Unreal native C++ DLL scaffolding: CMakeLists.txt, dllmain.cpp, UnrealSDK.h.
4. Godot GDExtension C++ scaffolding: .gdextension, register types/cpp, CMakeLists.txt.
5. Godot GDScript override scaffolding: override.cfg, mods/mod_entry.gd.
6. Godot PCK format packaging, parsing, and MD5 checksum integrity.
7. Godot PCK file extraction: single file and extract_all.
8. Epistemic Grounding Gate refusal on unverified Unreal/Godot targets (Design §16 #1).
9. Epistemic Grounding Gate acceptance on confirmed GROUNDED dynamic receipts.
10. Grounding gate bypass via --allow-unverified.
11. Symbol parsing without single-token duplication and receipt address hygiene.
12. CLI integration: toothpick build --engine unreal/godot and toothpick pck list/extract.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from toothpick.cli import cli
from toothpick.mod_builder import (
    GodotModConfig,
    GodotModGenerator,
    GodotPckReader,
    ModBuilder,
    PckParseError,
    UngroundedTargetError,
    UnrealModConfig,
    UnrealModGenerator,
)
from toothpick.storage import Database

TEST_SHA = "c" * 64
TEST_ADDR = 0x140500120


@pytest.fixture
def test_db(tmp_path: Path) -> Database:
    """Isolated SQLite database fixture."""
    db_file = tmp_path / "test_ue_godot.db"
    db = Database(db_file, create=True)
    yield db
    db.close()


# ---------------------------------------------------------------------------
# Unreal Engine Pipeline Tests
# ---------------------------------------------------------------------------


def test_ue4ss_lua_scaffolding(tmp_path: Path) -> None:
    """Verify UE4SS Lua mod structure, boot token, and grounding status."""
    out_dir = tmp_path / "ue_mod"
    config = UnrealModConfig(
        plugin_name="SanityMod",
        target_addr=TEST_ADDR,
        target_class="ABETPlayerState",
        target_function="SetSanity",
        hook_type="post",
        mod_type="ue4ss_lua",
        boot_token="test_token_12345",
        binary_sha256=TEST_SHA,
    )

    manifest_file, source_file = UnrealModGenerator.generate(
        config=config, out_dir=out_dir, is_grounded=True
    )

    assert manifest_file.is_file()
    assert manifest_file.name == "enabled.txt"
    assert manifest_file.read_text(encoding="utf-8").strip() == "1"

    assert source_file.is_file()
    assert source_file.name == "main.lua"
    lua_code = source_file.read_text(encoding="utf-8")

    assert "test_token_12345" in lua_code
    assert "[Toothpick] Boot verification token:" in lua_code
    assert "ABETPlayerState:SetSanity" in lua_code
    assert "GROUNDED (Design §16 #1)" in lua_code
    assert "RegisterHook" in lua_code


def test_ue4ss_lua_hook_types(tmp_path: Path) -> None:
    """Verify pre, post, and replace hook variants for UE4SS Lua."""
    for hook_type, expected_str in [
        ("pre", "Pre-hook fired"),
        ("post", "Post-hook fired"),
        ("replace", "return true"),
    ]:
        out_dir = tmp_path / f"ue_{hook_type}"
        config = UnrealModConfig(
            plugin_name="HookTestMod",
            target_addr=TEST_ADDR,
            target_class="Pawn",
            target_function="TakeDamage",
            hook_type=hook_type,
            mod_type="ue4ss_lua",
            binary_sha256=TEST_SHA,
        )
        _, source_file = UnrealModGenerator.generate(config, out_dir=out_dir)
        content = source_file.read_text(encoding="utf-8")
        assert expected_str in content


def test_unreal_native_cpp_scaffolding(tmp_path: Path) -> None:
    """Verify Unreal native C++ DLL scaffolding files and boot token."""
    out_dir = tmp_path / "ue_cpp"
    config = UnrealModConfig(
        plugin_name="NativeMod",
        target_addr=TEST_ADDR,
        target_class="PlayerController",
        target_function="ClientTravel",
        hook_type="post",
        mod_type="native_cpp",
        boot_token="cpp_token_999",
        binary_sha256=TEST_SHA,
    )

    manifest_file, source_file = UnrealModGenerator.generate(
        config=config, out_dir=out_dir, is_grounded=False
    )

    assert manifest_file.name == "CMakeLists.txt"
    assert manifest_file.is_file()
    cmake_text = manifest_file.read_text(encoding="utf-8")
    assert "project(NativeMod LANGUAGES CXX)" in cmake_text

    assert source_file.name == "dllmain.cpp"
    assert source_file.is_file()
    cpp_text = source_file.read_text(encoding="utf-8")
    assert "cpp_token_999" in cpp_text
    assert "UNVERIFIED (Design §16 #1)" in cpp_text
    assert "DllMain" in cpp_text

    sdk_header = out_dir / "include" / "UnrealSDK.h"
    assert sdk_header.is_file()
    assert "struct UFunction" in sdk_header.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Godot Engine Pipeline Tests
# ---------------------------------------------------------------------------


def test_godot_gdextension_scaffolding(tmp_path: Path) -> None:
    """Verify Godot 4 GDExtension C++ project structure and configuration."""
    out_dir = tmp_path / "godot_gde"
    config = GodotModConfig(
        plugin_name="SpeedMod",
        godot_version=4,
        mod_type="gdextension",
        target_node="Player",
        target_method="move_and_slide",
        target_addr=TEST_ADDR,
        boot_token="gde_token_777",
        binary_sha256=TEST_SHA,
    )

    manifest_file, source_file = GodotModGenerator.generate(
        config=config, out_dir=out_dir, is_grounded=True
    )

    assert manifest_file.name == "SpeedMod.gdextension"
    assert manifest_file.is_file()
    gde_text = manifest_file.read_text(encoding="utf-8")
    assert "entry_symbol = \"speedmod_library_init\"" in gde_text
    assert "compatibility_minimum = \"4.1\"" in gde_text

    assert source_file.name == "SpeedMod.cpp"
    assert source_file.is_file()
    cpp_text = source_file.read_text(encoding="utf-8")
    assert "gde_token_777" in cpp_text
    assert "speedmod_library_init" in cpp_text

    cmake_file = out_dir / "CMakeLists.txt"
    assert cmake_file.is_file()


def test_godot_gdscript_scaffolding(tmp_path: Path) -> None:
    """Verify Godot GDScript override project structure and boot token."""
    out_dir = tmp_path / "godot_gds"
    config = GodotModConfig(
        plugin_name="ChaosMod",
        godot_version=4,
        mod_type="gdscript",
        target_node="GameManager",
        target_method="_process",
        target_addr=None,
        boot_token="gd_token_555",
        binary_sha256=TEST_SHA,
    )

    manifest_file, source_file = GodotModGenerator.generate(
        config=config, out_dir=out_dir, is_grounded=False
    )

    assert manifest_file.name == "override.cfg"
    assert manifest_file.is_file()
    cfg_text = manifest_file.read_text(encoding="utf-8")
    assert "autoload/ToothpickMod=\"*res://mods/ChaosMod/mod_entry.gd\"" in cfg_text

    assert source_file.name == "mod_entry.gd"
    assert source_file.is_file()
    gd_text = source_file.read_text(encoding="utf-8")
    assert "gd_token_555" in gd_text
    assert "func _process(delta = 0.0):" in gd_text
    assert "UNVERIFIED (Design §16 #1)" in gd_text


# ---------------------------------------------------------------------------
# Godot PCK Package Parser & Synthesizer Tests
# ---------------------------------------------------------------------------


def test_godot_pck_roundtrip(tmp_path: Path) -> None:
    """Verify synthetic Godot .pck container creation and parsing."""
    pck_path = tmp_path / "test_package.pck"
    files = {
        "res://scripts/main.gd": b"extends Node\nfunc _ready(): pass\n",
        "res://config.json": b'{"difficulty": "hard", "fps": 60}',
        "res://assets/icon.png": b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR",
    }

    created = GodotPckReader.create_pck(
        files=files, out_path=pck_path, godot_version=(4, 2, 0), format_version=2
    )
    assert created == pck_path
    assert pck_path.is_file()

    reader = GodotPckReader(pck_path)
    entries = reader.parse()

    assert len(entries) == 3
    assert reader.format_version == 2
    assert reader.version_major == 4
    assert reader.version_minor == 2
    assert reader.version_patch == 0

    entry_map = {e.path: e for e in entries}
    for path, expected_bytes in files.items():
        assert path in entry_map
        assert entry_map[path].size == len(expected_bytes)
        assert len(entry_map[path].md5) == 16


def test_godot_pck_extraction(tmp_path: Path) -> None:
    """Verify extracting individual files and all files from a .pck container."""
    pck_path = tmp_path / "game.pck"
    files = {
        "res://scene.tscn": b"[gd_scene load_steps=1 format=3]\n",
        "res://data/items.csv": b"id,name,value\n1,sword,10\n",
    }
    GodotPckReader.create_pck(files=files, out_path=pck_path)

    reader = GodotPckReader(pck_path)

    # 1. Extract single file
    single_out = tmp_path / "extracted_single.tscn"
    extracted = reader.extract_file("res://scene.tscn", single_out)
    assert extracted.is_file()
    assert extracted.read_bytes() == files["res://scene.tscn"]

    # 2. Extract all files
    all_out_dir = tmp_path / "extracted_all"
    all_files = reader.extract_all(all_out_dir)
    assert len(all_files) == 2

    extracted_scene = all_out_dir / "scene.tscn"
    extracted_csv = all_out_dir / "data" / "items.csv"
    assert extracted_scene.is_file()
    assert extracted_scene.read_bytes() == files["res://scene.tscn"]
    assert extracted_csv.is_file()
    assert extracted_csv.read_bytes() == files["res://data/items.csv"]

    # 3. Nonexistent file raises PckParseError
    with pytest.raises(PckParseError):
        reader.extract_file("res://missing.dat", tmp_path / "missing.dat")


# ---------------------------------------------------------------------------
# Epistemic Grounding Gate Integration Tests
# ---------------------------------------------------------------------------


def test_grounding_gate_unreal(test_db: Database, tmp_path: Path) -> None:
    """Verify Epistemic Grounding Gate refusal, bypass, and acceptance on Unreal targets."""
    out_dir = tmp_path / "ue_grounding"

    with ModBuilder(db=test_db, binary_sha256=TEST_SHA) as builder:
        # 1. Refusal: unverified target address raises UngroundedTargetError
        with pytest.raises(UngroundedTargetError):
            builder.build(
                target=TEST_ADDR,
                out_dir=out_dir / "refused",
                engine="unreal",
                mod_type="ue4ss_lua",
            )

        # 2. Bypass: --allow-unverified succeeds with is_grounded=False
        art_bypassed = builder.build(
            target=TEST_ADDR,
            out_dir=out_dir / "bypassed",
            engine="unreal",
            mod_type="ue4ss_lua",
            allow_unverified=True,
        )
        assert not art_bypassed.is_grounded
        assert art_bypassed.engine == "unreal"

        # 3. Acceptance: target with confirmed GROUNDED receipt succeeds
        test_db.add_receipt(
            binary_sha256=TEST_SHA,
            func_addr=TEST_ADDR,
            phase="ground",
            payload={"verifier_status": "GROUNDED", "target_addr": TEST_ADDR},
        )

        art_grounded = builder.build(
            target=TEST_ADDR,
            out_dir=out_dir / "grounded",
            engine="unreal",
            mod_type="ue4ss_lua",
        )
        assert art_grounded.is_grounded
        assert art_grounded.receipt_id is not None


def test_grounding_gate_godot(test_db: Database, tmp_path: Path) -> None:
    """Verify Epistemic Grounding Gate refusal, bypass, and acceptance on Godot targets."""
    out_dir = tmp_path / "godot_grounding"

    with ModBuilder(db=test_db, binary_sha256=TEST_SHA) as builder:
        # 1. Refusal on symbolic target without dynamic verification
        with pytest.raises(UngroundedTargetError):
            builder.build(
                target="Player._physics_process",
                out_dir=out_dir / "refused",
                engine="godot",
                mod_type="gdscript",
            )

        # 2. Bypass succeeds
        art = builder.build(
            target="Player._physics_process",
            out_dir=out_dir / "bypassed",
            engine="godot",
            mod_type="gdscript",
            allow_unverified=True,
        )
        assert not art.is_grounded
        assert art.engine == "godot"
        assert art.mod_type == "gdscript"


def test_symbol_splitting_and_receipt_discipline(test_db: Database, tmp_path: Path) -> None:
    """Verify single-token symbols do not duplicate class/function, and no -1 receipt addr."""
    with ModBuilder(db=test_db, binary_sha256=TEST_SHA) as builder:
        # Single-token target without separator
        _, cls_name, func_name = builder._resolve_target_any("SoloFunction", TEST_SHA)
        assert cls_name == "Global"
        assert func_name == "SoloFunction"

        # Building ungrounded symbolic target with allow_unverified
        out_dir = tmp_path / "symbolic_mod"
        art = builder.build(
            target="SoloFunction",
            out_dir=out_dir,
            engine="unreal",
            mod_type="ue4ss_lua",
            allow_unverified=True,
        )
        # target_addr is None, so receipt_id should be None (no invalid -1 address stored)
        assert art.target_addr is None
        assert art.receipt_id is None


# ---------------------------------------------------------------------------
# CLI Integration Tests
# ---------------------------------------------------------------------------


def test_cli_build_unreal_and_godot(tmp_path: Path) -> None:
    """Verify toothpick build CLI with --engine unreal and --engine godot."""
    runner = CliRunner()

    # Unreal build
    ue_out = tmp_path / "cli_ue"
    res_ue = runner.invoke(
        cli,
        [
            "build",
            "0x140500120",
            "--engine",
            "unreal",
            "--mod-type",
            "ue4ss_lua",
            "--allow-unverified",
            "--out",
            str(ue_out),
            "--sha",
            TEST_SHA,
        ],
    )
    assert res_ue.exit_code == 0
    ue_data = json.loads(res_ue.output)
    assert ue_data["engine"] == "unreal"
    assert ue_data["mod_type"] == "ue4ss_lua"
    assert (ue_out / "scripts" / "main.lua").is_file()

    # Godot build
    godot_out = tmp_path / "cli_godot"
    res_godot = runner.invoke(
        cli,
        [
            "build",
            "GameNode._process",
            "--engine",
            "godot",
            "--mod-type",
            "gdscript",
            "--allow-unverified",
            "--out",
            str(godot_out),
            "--sha",
            TEST_SHA,
        ],
    )
    assert res_godot.exit_code == 0
    godot_data = json.loads(res_godot.output)
    assert godot_data["engine"] == "godot"
    assert (godot_out / "override.cfg").is_file()


def test_cli_pck_list_and_extract(tmp_path: Path) -> None:
    """Verify toothpick pck list and toothpick pck extract CLI subcommands."""
    pck_path = tmp_path / "sample.pck"
    files = {
        "res://init.gd": b"print('init')",
        "res://textures/bg.png": b"fake_png_data",
    }
    GodotPckReader.create_pck(files=files, out_path=pck_path)

    runner = CliRunner()

    # List
    res_list = runner.invoke(cli, ["pck", "list", str(pck_path)])
    assert res_list.exit_code == 0
    list_data = json.loads(res_list.output)
    assert list_data["file_count"] == 2
    paths = [e["path"] for e in list_data["entries"]]
    assert "res://init.gd" in paths

    # Extract all
    extract_dir = tmp_path / "cli_extracted"
    res_extract = runner.invoke(
        cli, ["pck", "extract", str(pck_path), "--out", str(extract_dir)]
    )
    assert res_extract.exit_code == 0
    extract_data = json.loads(res_extract.output)
    assert extract_data["extracted_count"] == 2
    assert (extract_dir / "init.gd").is_file()
    assert (extract_dir / "textures" / "bg.png").is_file()
