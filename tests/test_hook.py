"""
Tests for UNIT-P2 (Chunk 1 + Chunk 2b): toothpick.hook native hook artifact.

Validates (Chunk 1 contracts + Chunk 2b battery):
1. ``PATCH_LEN_REL32`` / ``PATCH_LEN_ABS64`` constants match the plan (5 / 12).
2. ``HookTarget`` strict construction (addr/arch/prologue/symbol validation,
   arch normalization) and its JSON-serializable view (Trap 14: lists, not tuples).
3. ``validate_prologue`` Capstone acceptance on real x64/x86 prologues, with both
   instruction-boundary patch lengths reported and the conservative one chosen.
4. ``validate_prologue`` refusals: relative-branch/return first instruction, short
   buffers, non-bytes input, empty buffer, unknown architecture.
5. ``validate_prologue`` purity/determinism (same input, same verdict).
6. ``HookGate.is_grounded`` via ``Database.get_receipts`` + ``verifier_status``
   ``GROUNDED``, including the negative paths (no receipt, REFUTED, no db,
   non-callable db, unparseable address).
7. ``HookGate.enforce`` raising ``toothpick.mod_builder.UngroundedTargetError``
   (imported, never redefined) and honouring ``allow_unverified``.
8. ``generate_hook_project`` file set, MinHook call sites, grounding banner,
   boot token, CMake ``add_subdirectory``/``add_library SHARED`` and README
   manual-mapping note.
9. Determinism (byte-identical regeneration) and Trap 7 handle hygiene (no
   dangling handles on the generated tree: it can be re-written and removed).
10. ``grounded=False`` stamping ``UNVERIFIED (allows-unverified)`` in the C++,
    the generated README and the returned metadata.
11. Receipts-or-refusal: a prologue that cannot be validated aborts generation.
12. Trap 11: no ``faulthandler`` import/installation in the module.

CLI round-trip (``toothpick hook build``) is Chunk 2a and lands in a follow-up
drive; it is deliberately not exercised here.
"""

from __future__ import annotations

import inspect
import json
import shutil
from pathlib import Path

import pytest

from toothpick import hook
from toothpick.hook import (
    PATCH_LEN_ABS64,
    PATCH_LEN_REL32,
    HookError,
    HookGate,
    HookTarget,
    UngroundedTargetError,
    generate_hook_project,
    validate_prologue,
)
from toothpick.mod_builder import UngroundedTargetError as ModBuilderUngroundedTargetError
from toothpick.storage import Database

TEST_SHA = "c" * 64
TEST_ADDR = 0x140012340

#: 12 real x64 bytes: mov [rsp+8],rbx / mov [rsp+0x10],rsi / push rdi / push rbp
PROLOGUE_X64_VALID = bytes.fromhex("48895c240848897424105755")
#: 6 real x86 bytes: push ebp / mov ebp,esp / sub esp,0x10
PROLOGUE_X86_VALID = bytes.fromhex("558bec83ec10")
#: Relative JMP (E9 rel32) -- relocation-unsafe to copy into a trampoline.
PROLOGUE_X64_RELATIVE_JMP = bytes.fromhex("e901000000")


@pytest.fixture
def test_db(tmp_path: Path) -> Database:
    """Isolated SQLite database, always closed (Trap 7)."""
    db = Database(tmp_path / "hook_test.db", create=True)
    try:
        yield db
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Constants and data contract
# ---------------------------------------------------------------------------


def test_patch_length_constants_match_the_plan() -> None:
    """rel32 JMP overwrite is 5 bytes; x64 absolute JMP is 12 bytes."""
    assert PATCH_LEN_REL32 == 5
    assert PATCH_LEN_ABS64 == 12


def test_hook_target_normalizes_and_exposes_stable_views() -> None:
    target = HookTarget(
        func_addr=TEST_ADDR,
        symbol="  PlayerController::Update  ",
        arch=" X64 ",
        prologue_bytes=bytearray(PROLOGUE_X64_VALID),
    )
    assert target.arch == "x64"
    assert target.symbol == "PlayerController::Update"
    assert target.prologue_bytes == PROLOGUE_X64_VALID
    assert isinstance(target.prologue_bytes, bytes)
    assert target.func_addr_hex == "0x140012340"
    assert target.symbol_display == "PlayerController::Update"


def test_hook_target_defaults_fall_back_to_the_address() -> None:
    target = HookTarget(func_addr=0x1000)
    assert target.arch == "x64"
    assert target.prologue_bytes == b""
    assert target.symbol_display == "0x1000"
    assert target.to_dict()["prologue_supplied"] is False
    json.dumps(target.to_dict())  # Trap 14: JSON-serializable


@pytest.mark.parametrize(
    "kwargs",
    [
        {"func_addr": -1},
        {"func_addr": True},
        {"func_addr": "0x1000"},
        {"func_addr": 0x1000, "arch": "arm64"},
        {"func_addr": 0x1000, "prologue_bytes": "558bec"},
        {"func_addr": 0x1000, "symbol": 123},
    ],
)
def test_hook_target_is_strict(kwargs: dict) -> None:
    with pytest.raises(HookError):
        HookTarget(**kwargs)


# ---------------------------------------------------------------------------
# validate_prologue -- acceptance
# ---------------------------------------------------------------------------


def test_validate_prologue_accepts_x64_prologue_and_reports_both_lengths() -> None:
    result = validate_prologue(PROLOGUE_X64_VALID, "x64")
    assert result["ok"] is True
    assert result["reason"] is None
    assert result["arch"] == "x64"
    assert result["decoded_len"] == 12
    assert result["patch_len_rel32"] == 5
    assert result["patch_len_abs64"] == 12
    # Conservative choice for x64: the 12-byte absolute jump.
    assert result["patch_len"] == PATCH_LEN_ABS64
    assert isinstance(result["insns"], list)
    assert [insn["mnemonic"] for insn in result["insns"]] == [
        "mov",
        "mov",
        "push",
        "push",
    ]
    assert result["insns"][0]["bytes"] == "48895c2408"


def test_validate_prologue_accepts_x86_prologue_and_chooses_rel32() -> None:
    result = validate_prologue(PROLOGUE_X86_VALID, "x86")
    assert result["ok"] is True
    # x86 only needs the rel32 jump; the boundary lands at 6 bytes (1+2+3).
    assert result["patch_len"] == result["patch_len_rel32"] == 6
    assert result["patch_len"] >= PATCH_LEN_REL32
    assert result["patch_len_abs64"] is None
    assert result["decoded_len"] == 6


def test_validate_prologue_accepts_memoryview_and_bytearray() -> None:
    assert validate_prologue(bytearray(PROLOGUE_X64_VALID), "x64")["ok"] is True
    assert validate_prologue(memoryview(PROLOGUE_X64_VALID), "x64")["ok"] is True


def test_validate_prologue_result_is_json_serializable() -> None:
    payload = validate_prologue(PROLOGUE_X64_VALID, "x64")
    decoded = json.loads(json.dumps(payload))
    assert isinstance(decoded["insns"], list)


# ---------------------------------------------------------------------------
# validate_prologue -- refusals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        PROLOGUE_X64_RELATIVE_JMP,  # E9 rel32 jmp
        bytes.fromhex("eb05"),  # EB rel8 jmp
        bytes.fromhex("c3"),  # ret
        bytes.fromhex("e800000000"),  # call rel32
    ],
)
def test_validate_prologue_refuses_relocation_unsafe_first_instruction(
    raw: bytes,
) -> None:
    result = validate_prologue(raw, "x64")
    assert result["ok"] is False
    assert result["patch_len"] is None
    assert "relocation-unsafe" in result["reason"]


@pytest.mark.parametrize(
    ("raw", "arch", "needle"),
    [
        (b"\x90" * 7, "x64", "too short"),
        (bytes.fromhex("558bec"), "x86", "too short"),
        (b"\x90" * 11, "x64", "too short"),
    ],
)
def test_validate_prologue_refuses_short_buffer(
    raw: bytes, arch: str, needle: str
) -> None:
    result = validate_prologue(raw, arch)
    assert result["ok"] is False
    assert result["patch_len"] is None
    assert needle in result["reason"]


@pytest.mark.parametrize(
    ("raw", "arch", "needle"),
    [
        ("48895c24", "x64", "bytes-like"),
        (None, "x64", "bytes-like"),
        (b"", "x64", "empty prologue buffer"),
        (PROLOGUE_X64_VALID, "arm64", "unknown arch"),
        (PROLOGUE_X64_VALID, "", "unknown arch"),
    ],
)
def test_validate_prologue_refuses_bad_input(raw: object, arch: str, needle: str) -> None:
    result = validate_prologue(raw, arch)
    assert result["ok"] is False
    assert result["patch_len"] is None
    assert needle in result["reason"]


def test_validate_prologue_is_pure_and_deterministic() -> None:
    raw = bytearray(PROLOGUE_X64_VALID)
    first = validate_prologue(raw, "x64")
    second = validate_prologue(raw, "x64")
    assert first == second
    # The caller's buffer is not mutated (pure/offline).
    assert raw == bytearray(PROLOGUE_X64_VALID)
    refused = validate_prologue(raw, "arm64")
    assert refused == validate_prologue(raw, "arm64")
    assert refused["ok"] is False


# ---------------------------------------------------------------------------
# HookGate -- Epistemic Grounding Gate
# ---------------------------------------------------------------------------


def _ground(db: Database, addr: int = TEST_ADDR, status: str = "GROUNDED") -> None:
    db.add_receipt(
        binary_sha256=TEST_SHA,
        func_addr=addr,
        phase="ground",
        payload={"ground_type": "FRIDA_LIVE", "verifier_status": status},
    )


def test_gate_is_grounded_with_grounded_receipt(test_db: Database) -> None:
    _ground(test_db)
    assert HookGate.is_grounded(test_db, TEST_SHA, TEST_ADDR) is True
    # Address may be supplied as hex text (CLI-friendly).
    assert HookGate.is_grounded(test_db, TEST_SHA, "0x140012340") is True


def test_gate_is_not_grounded_without_matching_receipt(test_db: Database) -> None:
    assert HookGate.is_grounded(test_db, TEST_SHA, TEST_ADDR) is False
    _ground(test_db, addr=TEST_ADDR + 0x40)
    assert HookGate.is_grounded(test_db, TEST_SHA, TEST_ADDR) is False
    assert HookGate.is_grounded(test_db, "d" * 64, TEST_ADDR) is False


def test_gate_is_not_grounded_for_refuted_receipt(test_db: Database) -> None:
    _ground(test_db, status="REFUTED")
    assert HookGate.is_grounded(test_db, TEST_SHA, TEST_ADDR) is False


@pytest.mark.parametrize("bad_db", [None, object(), "not-a-db", 7])
def test_gate_is_not_grounded_without_a_usable_database(bad_db: object) -> None:
    assert HookGate.is_grounded(bad_db, TEST_SHA, TEST_ADDR) is False


@pytest.mark.parametrize("bad_addr", ["not-an-address", None, True, -1])
def test_gate_is_not_grounded_for_unusable_address(
    test_db: Database, bad_addr: object
) -> None:
    assert HookGate.is_grounded(test_db, TEST_SHA, bad_addr) is False


def test_gate_enforce_accepts_grounded_target(test_db: Database) -> None:
    _ground(test_db)
    target = HookTarget(func_addr=TEST_ADDR, symbol="Update")
    assert HookGate.enforce(test_db, TEST_SHA, target) is True
    assert HookGate.enforce(test_db, TEST_SHA, TEST_ADDR) is True


def test_gate_enforce_raises_mod_builder_error_when_ungrounded(
    test_db: Database,
) -> None:
    # One error taxonomy: the class is imported, never redefined.
    assert UngroundedTargetError is ModBuilderUngroundedTargetError
    assert hook.UngroundedTargetError is ModBuilderUngroundedTargetError

    target = HookTarget(func_addr=TEST_ADDR, symbol="PlayerController::Update")
    with pytest.raises(UngroundedTargetError, match="no confirmed GROUNDED"):
        HookGate.enforce(test_db, TEST_SHA, target)
    with pytest.raises(UngroundedTargetError):
        HookGate.enforce(test_db, TEST_SHA, "0x140012340")
    # Nothing was written by a refused enforcement.
    assert test_db.get_receipts(TEST_SHA, TEST_ADDR) == []


def test_gate_enforce_override_returns_false_but_does_not_claim_grounding(
    test_db: Database,
) -> None:
    target = HookTarget(func_addr=TEST_ADDR)
    assert HookGate.enforce(test_db, TEST_SHA, target, allow_unverified=True) is False
    # A grounded target still reports True even with the override flag set.
    _ground(test_db)
    assert HookGate.enforce(test_db, TEST_SHA, target, allow_unverified=True) is True


# ---------------------------------------------------------------------------
# generate_hook_project -- structure, determinism, receipts-or-refusal
# ---------------------------------------------------------------------------


def _generate(tmp_path: Path, grounded: bool = True) -> tuple[dict, Path]:
    out_dir = tmp_path / "hooks" / "myhook"
    result = generate_hook_project(
        [
            HookTarget(
                func_addr=TEST_ADDR,
                symbol="PlayerController::Update",
                prologue_bytes=PROLOGUE_X64_VALID,
            )
        ],
        out_dir,
        name="myhook",
        grounded=grounded,
        minhook_dir="external/minhook",
    )
    return result, out_dir


def test_generate_hook_project_emits_three_files_and_minhook_calls(
    tmp_path: Path,
) -> None:
    result, out_dir = _generate(tmp_path)

    assert result["status"] == "success"
    assert result["name"] == "myhook"
    assert result["arch"] == "x64"
    assert result["grounded"] is True
    assert result["grounding_status"] == "GROUNDED"
    assert len(result["boot_token"]) == 32
    assert sorted(p.name for p in out_dir.iterdir()) == [
        "CMakeLists.txt",
        "README.md",
        "dllmain.cpp",
    ]
    assert result["files"] == sorted(str(p) for p in out_dir.iterdir())

    cpp = (out_dir / "dllmain.cpp").read_text(encoding="utf-8")
    assert "MH_Initialize" in cpp
    assert "MH_CreateHook" in cpp
    assert "MH_EnableHook" in cpp
    assert "BOOT_TOKEN" in cpp
    assert result["boot_token"] in cpp
    assert "GROUNDED" in cpp
    assert "UNVERIFIED (allows-unverified)" not in cpp
    assert "0x140012340" in cpp
    assert "__declspec(dllexport)" in cpp
    assert "ToothpickInit" in cpp

    cmake = (out_dir / "CMakeLists.txt").read_text(encoding="utf-8")
    assert "add_subdirectory" in cmake
    assert "add_library(myhook SHARED" in cmake
    assert "minhook" in cmake
    assert "dllmain.cpp" in cmake

    readme = (out_dir / "README.md").read_text(encoding="utf-8")
    assert "Manual mapping" in readme
    assert "ToothpickInit" in readme
    assert result["boot_token"] in readme
    assert "UNVERIFIED (allows-unverified)" not in readme


def test_generate_hook_project_is_deterministic_and_leaves_no_locked_handles(
    tmp_path: Path,
) -> None:
    first_result, first_dir = _generate(tmp_path)
    first_bytes = {
        path.name: path.read_bytes() for path in sorted(first_dir.iterdir())
    }

    # Regenerating over the same tree must work (no dangling handles, Trap 7).
    second_result, second_dir = _generate(tmp_path)
    second_bytes = {
        path.name: path.read_bytes() for path in sorted(second_dir.iterdir())
    }

    assert first_bytes == second_bytes
    assert first_result["boot_token"] == second_result["boot_token"]

    other_dir = tmp_path / "second"
    generate_hook_project(
        HookTarget(
            func_addr=TEST_ADDR,
            symbol="PlayerController::Update",
            prologue_bytes=PROLOGUE_X64_VALID,
        ),
        other_dir,
        name="myhook",
        minhook_dir="external/minhook",
    )
    assert {p.name: p.read_bytes() for p in other_dir.iterdir()} == first_bytes

    # Py3.8+ on Windows raises PermissionError here if a handle leaked.
    shutil.rmtree(tmp_path / "hooks")


def test_generate_hook_project_marks_unverified_without_claiming_grounding(
    tmp_path: Path,
) -> None:
    result, out_dir = _generate(tmp_path, grounded=False)

    assert result["grounded"] is False
    assert result["grounding_status"] == "UNVERIFIED (allows-unverified)"
    cpp = (out_dir / "dllmain.cpp").read_text(encoding="utf-8")
    readme = (out_dir / "README.md").read_text(encoding="utf-8")
    assert "UNVERIFIED (allows-unverified)" in cpp
    assert "UNVERIFIED (allows-unverified)" in readme
    # The unverified artifact still carries the boot-token banner.
    assert result["boot_token"] in cpp
    # Different grounding claim => different boot token.
    grounded_result, _ = _generate(tmp_path / "again", grounded=True)
    assert grounded_result["boot_token"] != result["boot_token"]


def test_generate_hook_project_reports_unvalidated_prologue(tmp_path: Path) -> None:
    out_dir = tmp_path / "no_prologue"
    result = generate_hook_project(
        [HookTarget(func_addr=TEST_ADDR, symbol="Update")],
        out_dir,
        name="noprologue",
    )
    record = result["targets"][0]
    assert record["patch_len"] is None
    assert record["patch_len_rel32"] is None
    assert record["patch_len_abs64"] is None
    assert "UNVALIDATED" in record["prologue_status"]
    cpp = (out_dir / "dllmain.cpp").read_text(encoding="utf-8")
    assert "minimum trampoline patch length: unvalidated" in cpp
    readme = (out_dir / "README.md").read_text(encoding="utf-8")
    assert "UNVALIDATED" in readme


def test_generate_hook_project_metadata_is_json_serializable(tmp_path: Path) -> None:
    result, _ = _generate(tmp_path)
    decoded = json.loads(json.dumps(result))
    assert isinstance(decoded["targets"], list)
    assert len(decoded["targets"]) == 1
    assert decoded["targets"][0]["arch"] == "x64"
    assert decoded["targets"][0]["patch_len"] == PATCH_LEN_ABS64


def test_generate_hook_project_refuses_an_unvalidatable_prologue(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "refused"
    with pytest.raises(HookError, match="prologue"):
        generate_hook_project(
            [
                HookTarget(
                    func_addr=TEST_ADDR,
                    symbol="Update",
                    prologue_bytes=PROLOGUE_X64_RELATIVE_JMP,
                )
            ],
            out_dir,
            name="refused",
        )
    # Receipts-or-refusal: nothing was emitted for a target it cannot back.
    assert not out_dir.exists()


@pytest.mark.parametrize(
    ("targets", "name"),
    [
        ([], "empty"),
        ([HookTarget(func_addr=0x1000, arch="x86"), HookTarget(func_addr=0x2000)], "mix"),
        ("0x140001000", "stringtarget"),
        ([HookTarget(func_addr=0x1000)], "   "),
    ],
)
def test_generate_hook_project_rejects_bad_inputs(
    tmp_path: Path, targets: object, name: str
) -> None:
    with pytest.raises(HookError):
        generate_hook_project(targets, tmp_path / "bad", name=name)


def test_generate_hook_project_accepts_a_single_bare_address(tmp_path: Path) -> None:
    out_dir = tmp_path / "bare"
    result = generate_hook_project(0x140001000, out_dir)
    assert result["name"] == "toothpick_hook"
    assert result["targets"][0]["func_addr"] == 0x140001000
    assert (out_dir / "dllmain.cpp").is_file()


# ---------------------------------------------------------------------------
# Trap 11 discipline
# ---------------------------------------------------------------------------


def test_module_does_not_import_or_install_faulthandler() -> None:
    """Native-engine suites run under ``-p no:faulthandler``: no hooks installed."""
    source = inspect.getsource(hook)
    assert "import faulthandler" not in source
    assert "faulthandler.enable" not in source
    assert "faulthandler.register" not in source
    assert "sys.excepthook" not in source
