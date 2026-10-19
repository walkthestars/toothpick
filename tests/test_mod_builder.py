"""
Tests for UNIT-W08: Unity IL2CPP Mod Pipeline (toothpick.mod_builder).

Validates:
1. Ingestion of IL2CPP metadata JSON dumps into types and funcs tables.
2. Ingestion of binary global-metadata.dat headers (magic 0xFAB11BAF).
3. BepInEx C# project generation with Harmony patches and boot verification tokens.
4. Epistemic Grounding Gate refusal on unverified target addresses (Design §16 #1).
5. Epistemic Grounding Gate acceptance when target has a GROUNDED receipt.
6. --allow-unverified override bypass.
7. Real compilation of generated C# project using dotnet build to produce .dll.
8. CLI commands toothpick ingest-metadata and toothpick build.
9. Ambiguous SHA handling across multi-binary databases (Trap 20).
10. Proper SQLite handle cleanup on teardown (Trap 7).
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest
from click.testing import CliRunner

from toothpick.cli import cli
from toothpick.mod_builder import (
    Il2CppMetadataReader,
    MetadataParseError,
    ModBuilder,
    ModConfig,
    ModGenerator,
    UngroundedTargetError,
)
from toothpick.storage import Database, FunctionRecord

TEST_SHA = "a" * 64
TEST_SHA_2 = "b" * 64
TEST_ADDR = 0x140012340


@pytest.fixture
def test_db(tmp_path: Path) -> Database:
    """Fixture providing an isolated SQLite database."""
    db_file = tmp_path / "test.db"
    db = Database(db_file, create=True)
    yield db
    db.close()


def test_ingest_metadata_json(test_db: Database, tmp_path: Path) -> None:
    """Test ingesting structured IL2CPP JSON dumps into types and funcs."""
    json_data = {
        "types": [
            {
                "name": "PlayerController",
                "type_id": "PlayerController",
                "kind": "class",
                "size_bytes": 128,
                "fields": [
                    {"name": "speed", "type": "System.Single", "offset": 32},
                    {"name": "health", "type": "System.Int32", "offset": 36},
                ],
                "methods": [
                    {
                        "name": "Update",
                        "addr": TEST_ADDR,
                        "signature": "void Update()",
                    },
                    {
                        "name": "TakeDamage",
                        "addr": TEST_ADDR + 0x40,
                        "signature": "void TakeDamage(int amount)",
                    },
                ],
            }
        ]
    }
    json_file = tmp_path / "metadata.json"
    json_file.write_text(json.dumps(json_data), encoding="utf-8")

    res = Il2CppMetadataReader.ingest(json_file, test_db, binary_sha256=TEST_SHA)
    assert res["types_ingested"] == 1
    assert res["funcs_ingested"] == 2

    # Verify stored in types
    t = test_db.get_type("PlayerController", binary_sha256=TEST_SHA)
    assert t is not None
    assert t.name == "PlayerController"
    assert t.layout is not None
    assert len(t.layout["fields"]) == 2
    assert len(t.layout["methods"]) == 2

    # Verify stored in funcs
    f = test_db.get_function(TEST_SHA, TEST_ADDR)
    assert f is not None
    assert f.name == "PlayerController.Update"


def test_ingest_metadata_script_json(test_db: Database) -> None:
    """Test ingesting Il2CppDumper style script.json output."""
    script_json = {
        "ScriptMethod": [
            {
                "Address": TEST_ADDR,
                "Name": "PlayerController$$Update",
                "Signature": "void PlayerController_Update (PlayerController_o *__this, const MethodInfo* method);",
                "TypeSignature": "iii",
            }
        ]
    }
    res = Il2CppMetadataReader.ingest(script_json, test_db, binary_sha256=TEST_SHA)
    assert res["types_ingested"] == 1
    assert res["funcs_ingested"] == 1

    f = test_db.get_function(TEST_SHA, TEST_ADDR)
    assert f is not None
    assert f.name == "PlayerController$$Update"


def test_ingest_metadata_binary(test_db: Database, tmp_path: Path) -> None:
    """Test ingesting binary global-metadata.dat header."""
    # Magic 0xFAB11BAF, version 29, followed by dummy header offsets
    magic = 0xFAB11BAF
    version = 29
    header = struct.pack("<Iiiiiiii", magic, version, 0x100, 10, 0x200, 20, 0x300, 30)
    meta_file = tmp_path / "global-metadata.dat"
    meta_file.write_bytes(header)

    res = Il2CppMetadataReader.ingest(meta_file, test_db, binary_sha256=TEST_SHA)
    assert res["types_ingested"] == 1

    t = test_db.get_type("__il2cpp_global_metadata__", binary_sha256=TEST_SHA)
    assert t is not None
    assert t.kind == "metadata"
    assert t.layout["version"] == 29
    assert t.layout["magic"] == "0xFAB11BAF"


def test_ingest_metadata_invalid_magic(test_db: Database, tmp_path: Path) -> None:
    """Test that invalid binary metadata magic raises MetadataParseError."""
    bad_header = struct.pack("<II", 0xDEADBEEF, 29)
    meta_file = tmp_path / "bad-metadata.dat"
    meta_file.write_bytes(bad_header)

    with pytest.raises(MetadataParseError, match="Invalid global-metadata.dat magic"):
        Il2CppMetadataReader.ingest(meta_file, test_db, binary_sha256=TEST_SHA)


def test_mod_generator_scaffolding(tmp_path: Path) -> None:
    """Test BepInEx C# scaffolding generation and Harmony syntax."""
    out_dir = tmp_path / "mod_project"
    config = ModConfig(
        plugin_name="SpeedMod",
        target_addr=TEST_ADDR,
        target_class="PlayerController",
        target_method="Update",
        hook_type="postfix",
        mod_logic="Console.WriteLine(\"Speed injected!\");",
    )

    csproj_path, source_path = ModGenerator.generate(config, out_dir, is_grounded=True)
    assert csproj_path.is_file()
    assert source_path.is_file()

    cs_code = source_path.read_text(encoding="utf-8")
    assert "class SpeedModPlugin : BaseUnityPlugin" in cs_code
    assert "[BepInPlugin(PluginInfo.PLUGIN_GUID, PluginInfo.PLUGIN_NAME, PluginInfo.PLUGIN_VERSION)]" in cs_code
    assert f"[Toothpick] Boot verification token: {config.boot_token}" in cs_code
    assert "AccessTools.Method(\"PlayerController:Update\")" in cs_code
    assert "[HarmonyPostfix]" in cs_code
    assert "Speed injected!" in cs_code
    assert "GROUNDED" in cs_code


def test_mod_generator_hook_types(tmp_path: Path) -> None:
    """Test prefix and replace hook type generation."""
    out_dir = tmp_path / "prefix_proj"
    config_prefix = ModConfig(
        plugin_name="PrefixMod",
        target_addr=TEST_ADDR,
        target_class="PlayerController",
        target_method="Update",
        hook_type="prefix",
    )
    _, src_prefix = ModGenerator.generate(config_prefix, out_dir)
    assert "[HarmonyPrefix]" in src_prefix.read_text(encoding="utf-8")
    assert "return true;" in src_prefix.read_text(encoding="utf-8")

    out_dir_rep = tmp_path / "replace_proj"
    config_replace = ModConfig(
        plugin_name="ReplaceMod",
        target_addr=TEST_ADDR,
        target_class="PlayerController",
        target_method="Update",
        hook_type="replace",
    )
    _, src_replace = ModGenerator.generate(config_replace, out_dir_rep)
    assert "[HarmonyPrefix]" in src_replace.read_text(encoding="utf-8")
    assert "return false;" in src_replace.read_text(encoding="utf-8")


def test_grounding_gate_refusal_on_ungrounded_target(test_db: Database, tmp_path: Path) -> None:
    """Test that building against an unverified address raises UngroundedTargetError (Design §16 #1)."""
    # Put a function in DB, but NO receipts
    test_db.upsert_function(
        FunctionRecord(
            binary_sha256=TEST_SHA,
            func_addr=TEST_ADDR,
            name="PlayerController.Update",
        ),
        write_receipt=False,
    )

    builder = ModBuilder(db=test_db, binary_sha256=TEST_SHA)
    out_dir = tmp_path / "refused_mod"

    with pytest.raises(UngroundedTargetError, match="has no confirmed GROUNDED dynamic receipt"):
        builder.build(
            target=TEST_ADDR,
            out_dir=out_dir,
            allow_unverified=False,
        )

    # Also test with method name string
    with pytest.raises(UngroundedTargetError):
        builder.build(
            target="PlayerController.Update",
            out_dir=out_dir,
            allow_unverified=False,
        )


def test_grounding_gate_acceptance_with_grounded_receipt(test_db: Database, tmp_path: Path) -> None:
    """Test that building succeeds when target address has a GROUNDED receipt."""
    test_db.upsert_function(
        FunctionRecord(
            binary_sha256=TEST_SHA,
            func_addr=TEST_ADDR,
            name="PlayerController.Update",
        ),
        write_receipt=False,
    )
    # Add a GROUNDED dynamic receipt
    test_db.add_receipt(
        binary_sha256=TEST_SHA,
        func_addr=TEST_ADDR,
        phase="ground",
        payload={"ground_type": "FRIDA_LIVE", "verifier_status": "GROUNDED"},
    )

    out_dir = tmp_path / "grounded_mod"
    with ModBuilder(db=test_db, binary_sha256=TEST_SHA) as builder:
        artifact = builder.build(
            target=TEST_ADDR,
            out_dir=out_dir,
            allow_unverified=False,
        )
        assert artifact.is_grounded is True
        assert artifact.receipt_id is not None
        assert artifact.source_file.is_file()

        # Check mod_build receipt in database
        build_receipts = [
            r for r in test_db.get_receipts(TEST_SHA, TEST_ADDR)
            if r["phase"] == "mod_build"
        ]
        assert len(build_receipts) == 1
        assert build_receipts[0]["payload"]["verifier_status"] == "GROUNDED"


def test_grounding_gate_override_allow_unverified(test_db: Database, tmp_path: Path) -> None:
    """Test that --allow-unverified overrides the grounding gate cleanly."""
    test_db.upsert_function(
        FunctionRecord(
            binary_sha256=TEST_SHA,
            func_addr=TEST_ADDR,
            name="PlayerController.Update",
        ),
        write_receipt=False,
    )

    out_dir = tmp_path / "unverified_mod"
    with ModBuilder(db=test_db, binary_sha256=TEST_SHA) as builder:
        artifact = builder.build(
            target=TEST_ADDR,
            out_dir=out_dir,
            allow_unverified=True,
        )
        assert artifact.is_grounded is False
        assert artifact.source_file.is_file()

        build_receipts = [
            r for r in test_db.get_receipts(TEST_SHA, TEST_ADDR)
            if r["phase"] == "mod_build"
        ]
        assert len(build_receipts) == 1
        assert build_receipts[0]["payload"]["verifier_status"] == "UNGROUNDED"
        assert build_receipts[0]["payload"]["allow_unverified"] is True


def test_dotnet_compilation_e2e(test_db: Database, tmp_path: Path) -> None:
    """Test end-to-end compilation using dotnet build to produce a valid .dll."""
    test_db.upsert_function(
        FunctionRecord(
            binary_sha256=TEST_SHA,
            func_addr=TEST_ADDR,
            name="PlayerController.Update",
        ),
        write_receipt=False,
    )
    test_db.add_receipt(
        binary_sha256=TEST_SHA,
        func_addr=TEST_ADDR,
        phase="ground",
        payload={"verifier_status": "GROUNDED"},
    )

    out_dir = tmp_path / "compiled_mod"
    with ModBuilder(db=test_db, binary_sha256=TEST_SHA) as builder:
        artifact = builder.build(
            target="PlayerController.Update",
            out_dir=out_dir,
            name="PlayerSpeedPlugin",
            compile=True,
        )
        assert artifact.dll_path is not None
        assert artifact.dll_path.is_file()
        assert artifact.dll_path.suffix == ".dll"
        assert artifact.dll_path.stat().st_size > 0


def test_cli_ingest_metadata(tmp_path: Path) -> None:
    """Test CLI subcommand 'toothpick ingest-metadata'."""
    runner = CliRunner()
    db_file = tmp_path / "cli_test.db"

    meta_json = {
        "types": [
            {
                "name": "EnemyAI",
                "methods": [
                    {"name": "Think", "addr": 0x140020000}
                ]
            }
        ]
    }
    meta_path = tmp_path / "enemy_meta.json"
    meta_path.write_text(json.dumps(meta_json), encoding="utf-8")

    res = runner.invoke(
        cli,
        [
            "ingest-metadata",
            str(meta_path),
            "--db",
            str(db_file),
            "--sha",
            TEST_SHA,
        ],
    )
    assert res.exit_code == 0, res.output
    data = json.loads(res.output)
    assert data["status"] == "ok"
    assert data["types_ingested"] == 1
    assert data["funcs_ingested"] == 1


def test_cli_build_grounding_refusal(tmp_path: Path) -> None:
    """Test CLI build refusal with exit code 2 when target is ungrounded."""
    runner = CliRunner()
    db_file = tmp_path / "cli_refusal.db"

    # Ingest metadata but do not ground
    with Database(db_file, create=True) as db:
        db.upsert_function(
            FunctionRecord(binary_sha256=TEST_SHA, func_addr=TEST_ADDR, name="GameManager.Init"),
            write_receipt=False,
        )

    out_dir = tmp_path / "cli_refused_mod"
    res = runner.invoke(
        cli,
        [
            "build",
            hex(TEST_ADDR),
            "--db",
            str(db_file),
            "--sha",
            TEST_SHA,
            "--out",
            str(out_dir),
        ],
    )
    assert res.exit_code == 2
    assert "Ungrounded Target" in res.output


def test_cli_build_success(tmp_path: Path) -> None:
    """Test CLI build success with grounded receipt and compilation."""
    runner = CliRunner()
    db_file = tmp_path / "cli_success.db"

    with Database(db_file, create=True) as db:
        db.upsert_function(
            FunctionRecord(binary_sha256=TEST_SHA, func_addr=TEST_ADDR, name="GameManager.Init"),
            write_receipt=False,
        )
        db.add_receipt(
            binary_sha256=TEST_SHA,
            func_addr=TEST_ADDR,
            phase="ground",
            payload={"verifier_status": "GROUNDED"},
        )

    out_dir = tmp_path / "cli_success_mod"
    res = runner.invoke(
        cli,
        [
            "build",
            "GameManager.Init",
            "--db",
            str(db_file),
            "--sha",
            TEST_SHA,
            "--out",
            str(out_dir),
            "--name",
            "GameManagerMod",
            "--compile",
        ],
    )
    assert res.exit_code == 0, res.output
    data = json.loads(res.output)
    assert data["status"] == "success"
    assert data["plugin_name"] == "GameManagerMod"
    assert data["is_grounded"] is True
    assert data["dll_path"] is not None
    assert Path(data["dll_path"]).is_file()


def test_ambiguous_sha_detection(test_db: Database, tmp_path: Path) -> None:
    """Test that ambiguous SHA inference raises ModBuildError (Trap 20)."""
    # Insert functions for two different binaries
    test_db.upsert_function(
        FunctionRecord(binary_sha256=TEST_SHA, func_addr=TEST_ADDR, name="FuncA"),
        write_receipt=False,
    )
    test_db.upsert_function(
        FunctionRecord(binary_sha256=TEST_SHA_2, func_addr=TEST_ADDR, name="FuncB"),
        write_receipt=False,
    )

    with ModBuilder(db=test_db) as builder:
        with pytest.raises(Exception, match="Cannot infer binary SHA-256"):
            builder.build(target=TEST_ADDR, out_dir=tmp_path / "ambig")


def test_sqlite_handle_cleanup_trap_7(tmp_path: Path) -> None:
    """Verify that ModBuilder cleanly closes database connection and leaves no locks (Trap 7)."""
    db_file = tmp_path / "trap7.db"
    db = Database(db_file, create=True)
    db.upsert_function(
        FunctionRecord(binary_sha256=TEST_SHA, func_addr=TEST_ADDR, name="Foo.Bar"),
        write_receipt=False,
    )
    db.add_receipt(TEST_SHA, TEST_ADDR, "ground", payload={"verifier_status": "GROUNDED"})
    db.close()

    builder = ModBuilder(db=db_file)
    builder.build(target=TEST_ADDR, out_dir=tmp_path / "mod7", allow_unverified=True)
    builder.close()

    # Deleting db_file must succeed without Windows file lock errors
    db_file.unlink()
    assert not db_file.exists()
