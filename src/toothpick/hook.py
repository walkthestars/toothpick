"""
Toothpick P2: generic native hook artifact (injected DLL + MinHook trampolines).

Implements **Chunk 1** of ``docs/units/UNIT-P2-HOOK-PLAN.md`` -- data contracts,
Capstone prologue validation, the Epistemic Grounding Gate, and the deterministic
MinHook project generator.  The Chunk 2b battery lives in ``tests/test_hook.py``
(CLI wiring, Chunk 2a, is still a follow-up drive).

Contents
--------
* :class:`HookTarget` -- strict dataclass for one trampoline target
  (``func_addr``, optional ``symbol``, ``arch`` in ``{x86, x64}``, optional
  caller-supplied ``prologue_bytes``; prologue bytes are never invented here).
* :func:`validate_prologue` -- pure/offline/deterministic Capstone decode of the
  caller-supplied bytes.  It computes the minimum trampoline patch length at an
  instruction boundary (``PATCH_LEN_REL32 = 5`` for a rel32 ``JMP``;
  ``PATCH_LEN_ABS64 = 12`` for the x64 absolute-jump sequence ``48 B8``
  ``mov rax, imm64`` + ``FF E0`` ``jmp rax``), reports both
  boundaries, and chooses the conservative one (x64 => 12, x86 => 5).  It
  refuses (``ok: False``) on a relocation-unsafe first instruction, on a buffer
  that decodes to fewer bytes than required, and on an unknown architecture.
* :class:`HookGate` -- ``verifier_status == "GROUNDED"`` lookups through
  ``Database.get_receipts`` (the same semantics as
  :meth:`toothpick.mod_builder.ModBuilder.is_target_grounded`) plus
  :meth:`HookGate.enforce`, which raises
  :class:`toothpick.mod_builder.UngroundedTargetError` (imported, never
  redefined -- one error taxonomy, Design §16 #1).
* :func:`generate_hook_project` -- writes ``dllmain.cpp`` (MinHook
  ``MH_Initialize`` / ``MH_CreateHook`` / ``MH_EnableHook`` per target, boot
  token, grounding banner), ``CMakeLists.txt`` (``add_subdirectory`` MinHook +
  ``add_library(... SHARED)``) and ``README.md`` (build + injection steps plus
  the manual-mapping note).  Deterministic file order and bytes;
  ``grounded=False`` stamps ``UNVERIFIED (allows-unverified)`` and never claims
  grounding.

Discipline
----------
* **Receipts-or-refusal**: no offset is emitted without a ``GROUNDED`` receipt
  unless the operator explicitly passes ``allow_unverified=True`` (gate) or
  ``grounded=False`` (generator) -- and the artifact then says so in the C++
  banner, in the generated ``README.md`` and in the returned metadata.
* **Trap 7 (handle hygiene)**: every file is written with an explicit encoding
  and ``newline="\\n"`` via :meth:`pathlib.Path.write_text`, which closes the
  handle immediately -- no dangling file objects, no half-flushed artifacts and
  no Windows ``WinError 32`` locks on the generated tree.
* **Trap 11 (``faulthandler``)**: this module imports no ``faulthandler`` and
  installs no exception hooks, so native-engine suites keep running under
  ``-p no:faulthandler``.
* **Pure/offline**: Capstone only.  No process is attached, no memory is read,
  no network is touched, and nothing is submitted to an LLM.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import capstone

from toothpick.mod_builder import UngroundedTargetError

__all__ = [
    "HookError",
    "PATCH_LEN_REL32",
    "PATCH_LEN_ABS64",
    "HookTarget",
    "validate_prologue",
    "HookGate",
    "generate_hook_project",
    "UngroundedTargetError",
]

#: Minimum overwrite length for a 5-byte relative ``JMP`` (x86 and x64).
PATCH_LEN_REL32 = 5

#: Minimum overwrite length for a 12-byte absolute ``JMP`` sequence: ``48 B8``
#: (``mov rax, imm64``, 10 bytes) followed by ``FF E0`` (``jmp rax``, 2 bytes).
PATCH_LEN_ABS64 = 12

_GENERATED_FILES = ("dllmain.cpp", "CMakeLists.txt", "README.md")

_ARCH_MODES: Dict[str, Tuple[int, int]] = {
    "x86": (capstone.CS_ARCH_X86, capstone.CS_MODE_32),
    "x64": (capstone.CS_ARCH_X86, capstone.CS_MODE_64),
}

_SUPPORTED_ARCHES = ("x86", "x64")

PathLike = Union[str, Path]


class HookError(Exception):
    """Base exception for native hook artifact failures."""


# ---------------------------------------------------------------------------
# Input coercion helpers
# ---------------------------------------------------------------------------


def _coerce_bytes(value: Any) -> Optional[bytes]:
    """Return ``value`` as ``bytes`` when it is bytes-like, else ``None``."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    return None


def _normalize_arch(arch: Any) -> str:
    """Normalize an architecture label to a lowercase string (no validation)."""
    if isinstance(arch, (bytes, bytearray)):
        arch = bytes(arch).decode("ascii", "replace")
    return str(arch).strip().lower()


def _normalize_addr(addr: Any) -> Optional[int]:
    """Coerce an address to ``int``; ``None`` when it is unusable."""
    if isinstance(addr, bool):
        return None
    if isinstance(addr, int):
        return addr
    if isinstance(addr, str):
        try:
            return int(addr.strip(), 0)
        except ValueError:
            return None
    return None


def _sanitize_cpp_identifier(name: str, default: str = "Target") -> str:
    """Sanitize ``name`` into a valid C/C++ identifier (deterministic)."""
    cleaned = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in str(name or ""))
    cleaned = cleaned.strip("_")
    if not cleaned:
        return default
    if cleaned[0].isdigit():
        cleaned = f"M_{cleaned}"
    return cleaned


def _c_string_literal(text: Any) -> str:
    """Render ``text`` as a single-line double-quoted C string literal."""
    raw = str(text)
    escaped = (
        raw.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\r", " ")
        .replace("\n", " ")
        .replace("\t", " ")
    )
    return f'"{escaped}"'


# ---------------------------------------------------------------------------
# Chunk 1 -- data contract
# ---------------------------------------------------------------------------


@dataclass
class HookTarget:
    """One trampoline target for a generated native hook DLL.

    ``prologue_bytes`` is **caller-supplied** and optional: when it is empty the
    trampoline patch length is deferred and the target is reported as
    ``UNVALIDATED`` (nothing is invented, Design §16 #1).  Construction is
    strict -- a malformed target raises :class:`HookError` instead of producing
    an artifact that silently claims something it does not know.
    """

    func_addr: int
    symbol: str = ""
    arch: str = "x64"
    prologue_bytes: bytes = b""

    def __post_init__(self) -> None:
        if isinstance(self.func_addr, bool) or not isinstance(self.func_addr, int):
            raise HookError(
                f"func_addr must be an int, got {type(self.func_addr).__name__}"
            )
        if self.func_addr < 0:
            raise HookError(f"func_addr must be non-negative, got 0x{self.func_addr:x}")
        if self.symbol is None or not isinstance(self.symbol, str):
            raise HookError(f"symbol must be a str, got {type(self.symbol).__name__}")
        self.symbol = self.symbol.strip()

        arch = _normalize_arch(self.arch)
        if arch not in _SUPPORTED_ARCHES:
            raise HookError(
                f"arch must be one of {_SUPPORTED_ARCHES}, got {self.arch!r}"
            )
        self.arch = arch

        prologue = _coerce_bytes(self.prologue_bytes)
        if prologue is None:
            raise HookError(
                "prologue_bytes must be bytes-like (bytes/bytearray/memoryview), "
                f"got {type(self.prologue_bytes).__name__}"
            )
        self.prologue_bytes = prologue

    @property
    def func_addr_hex(self) -> str:
        """Canonical ``0x``-prefixed address text (deterministic)."""
        return f"0x{self.func_addr:x}"

    @property
    def symbol_display(self) -> str:
        """Symbol for humans, falling back to the address."""
        return self.symbol or self.func_addr_hex

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable view with a stable key order."""
        return {
            "func_addr": self.func_addr,
            "func_addr_hex": self.func_addr_hex,
            "symbol": self.symbol,
            "arch": self.arch,
            "prologue_bytes_hex": self.prologue_bytes.hex(),
            "prologue_supplied": bool(self.prologue_bytes),
        }


# ---------------------------------------------------------------------------
# Chunk 1 -- Capstone prologue validation
# ---------------------------------------------------------------------------


def _decode(prologue_bytes: bytes, arch: str) -> List[Any]:
    """Decode up to the end of the buffer with Capstone (detail on)."""
    arch_const, mode = _ARCH_MODES[arch]
    md = capstone.Cs(arch_const, mode)
    md.detail = True
    return list(md.disasm(prologue_bytes, 0))


def _boundary(decoded: Sequence[Any], required: int) -> Optional[int]:
    """Smallest instruction-boundary length ``>= required`` (``None`` if short)."""
    if required <= 0:
        return 0
    accumulated = 0
    for insn in decoded:
        accumulated += int(insn.size)
        if accumulated >= required:
            return accumulated
    return None


def _unsafe_first_insn(insn: Any) -> Optional[str]:
    """Describe a relocation-unsafe control transfer, or return ``None``.

    A copied prologue that *begins* with a relative branch/call (or a return)
    cannot be relocated blindly by the trampoline, so validation refuses it.
    Only the first instruction is judged -- MinHook relocates the remainder.

    The predicate over-approximates deliberately: every control-transfer group
    is refused, including register-indirect forms (``ff e0`` ``jmp rax``) whose
    copy would in principle be position-independent.  A refused target costs a
    re-run; a silently mis-relocated trampoline costs a crash, so refusal is the
    conservative side.
    """
    mnemonic = str(getattr(insn, "mnemonic", "") or "").lower()
    groups = set(getattr(insn, "groups", []) or [])
    checks = (
        ("jump", getattr(capstone, "CS_GRP_JUMP", None)),
        ("call", getattr(capstone, "CS_GRP_CALL", None)),
        ("ret", getattr(capstone, "CS_GRP_RET", None)),
        ("iret", getattr(capstone, "CS_GRP_IRET", None)),
    )
    for label, code in checks:
        if code is not None and code in groups:
            return f"{label} ({mnemonic})"
    if mnemonic.startswith(("j", "call", "loop", "ret")):
        return f"branch ({mnemonic})"
    return None


def validate_prologue(prologue_bytes: Any, arch: str = "x64") -> Dict[str, Any]:
    """Validate caller-supplied prologue bytes for trampoline patching.

    Returns ``{ok, arch, patch_len, patch_len_rel32, patch_len_abs64,
    decoded_len, insns, reason}``.  ``reason`` is ``None`` only when ``ok`` is
    ``True``.  The function never raises and never touches the network or a live
    process: the same input always produces the same verdict (pure/offline/
    deterministic).

    Refusal cases (``ok: False``):
      1. ``prologue_bytes`` is not bytes-like, or the buffer is empty;
      2. ``arch`` is not ``x86``/``x64``;
      3. the bytes do not decode at all;
      4. the first instruction is a relocation-unsafe relative branch/call
         (``jmp``/``call``/``jcc``/``loop``) or a return;
      5. the buffer decodes to fewer bytes than the required patch length at an
         instruction boundary (``PATCH_LEN_ABS64`` for x64, ``PATCH_LEN_REL32``
         for x86).
    """
    arch_norm = _normalize_arch(arch)
    result: Dict[str, Any] = {
        "ok": False,
        "arch": arch_norm,
        "patch_len": None,
        "patch_len_rel32": None,
        "patch_len_abs64": None,
        "decoded_len": 0,
        "insns": [],
        "reason": None,
    }

    data = _coerce_bytes(prologue_bytes)
    if data is None:
        result["reason"] = (
            "prologue_bytes must be bytes-like (bytes/bytearray/memoryview), got "
            f"{type(prologue_bytes).__name__}"
        )
        return result
    if arch_norm not in _SUPPORTED_ARCHES:
        result["reason"] = f"unknown arch {arch_norm!r}; expected 'x86' or 'x64'"
        return result
    if not data:
        result["reason"] = "empty prologue buffer; nothing to validate (UNVALIDATED)"
        return result

    decoded = _decode(data, arch_norm)
    if not decoded:
        result["reason"] = (
            f"prologue does not decode as {arch_norm}: {data[:8].hex()}"
        )
        return result

    result["insns"] = [
        {
            "offset": int(insn.address),
            "size": int(insn.size),
            "mnemonic": str(insn.mnemonic),
            "op_str": str(insn.op_str),
            "bytes": bytes(insn.bytes).hex(),
        }
        for insn in decoded
    ]
    result["decoded_len"] = sum(int(insn.size) for insn in decoded)
    result["patch_len_rel32"] = _boundary(decoded, PATCH_LEN_REL32)
    result["patch_len_abs64"] = _boundary(decoded, PATCH_LEN_ABS64)

    unsafe = _unsafe_first_insn(decoded[0])
    if unsafe is not None:
        result["reason"] = (
            "first instruction is relocation-unsafe to copy "
            f"({unsafe} {decoded[0].op_str}); reject a prologue that starts with "
            "a relative branch/call (or return)"
        )
        return result

    # Conservative choice: x64 needs the 12-byte absolute jump; x86 needs rel32.
    required = PATCH_LEN_ABS64 if arch_norm == "x64" else PATCH_LEN_REL32
    patch_len = (
        result["patch_len_abs64"] if arch_norm == "x64" else result["patch_len_rel32"]
    )
    result["patch_len"] = patch_len
    if patch_len is None:
        result["reason"] = (
            f"prologue too short: {result['decoded_len']} byte(s) in "
            f"{len(decoded)} instruction(s), need >= {required} bytes at an "
            f"instruction boundary for {arch_norm}"
        )
        return result

    result["ok"] = True
    return result


# ---------------------------------------------------------------------------
# Chunk 1 -- Epistemic Grounding Gate
# ---------------------------------------------------------------------------


class HookGate:
    """Grounding gate for native hook targets (Design §16 #1, §19).

    Semantics mirror :meth:`toothpick.mod_builder.ModBuilder.is_target_grounded`:
    a target is grounded only when ``Database.get_receipts(sha, addr)`` yields a
    receipt whose ``verifier_status`` is ``GROUNDED`` -- either in the payload or
    on the receipt row.
    """

    @staticmethod
    def is_grounded(db: Any, binary_sha256: str, func_addr: Any) -> bool:
        """``True`` when a confirmed ``GROUNDED`` receipt backs this address."""
        if db is None:
            return False
        addr = _normalize_addr(func_addr)
        if addr is None or addr < 0:
            return False
        get_receipts = getattr(db, "get_receipts", None)
        if not callable(get_receipts):
            return False
        try:
            receipts = get_receipts(binary_sha256, addr)
        except Exception:  # pragma: no cover - defensive: treat as ungrounded
            return False
        if not isinstance(receipts, (list, tuple)):
            return False
        for receipt in receipts:
            if not isinstance(receipt, dict):
                continue
            payload = receipt.get("payload") or {}
            status = None
            if isinstance(payload, dict):
                status = payload.get("verifier_status")
            status = status or receipt.get("verifier_status")
            if status == "GROUNDED":
                return True
        return False

    @classmethod
    def enforce(
        cls,
        db: Any,
        binary_sha256: str,
        target: Union[HookTarget, int],
        allow_unverified: bool = False,
    ) -> bool:
        """Return ``True`` (grounded) / ``False`` (explicit override) or raise.

        Raises :class:`toothpick.mod_builder.UngroundedTargetError` when the
        target has no confirmed ``GROUNDED`` receipt and the caller did not pass
        ``allow_unverified=True``.
        """
        if isinstance(target, HookTarget):
            func_addr = target.func_addr
            symbol = target.symbol_display
        else:
            func_addr = _normalize_addr(target)
            symbol = f"0x{func_addr:x}" if func_addr is not None else "<unresolved>"

        grounded = func_addr is not None and cls.is_grounded(
            db, binary_sha256, func_addr
        )
        if grounded or allow_unverified:
            return grounded

        addr_text = f"0x{func_addr:x}" if func_addr is not None else "<unresolved>"
        raise UngroundedTargetError(
            f"Target address {addr_text} ({symbol}) has no confirmed GROUNDED "
            "dynamic receipt in database; pass --allow-unverified to override "
            "(Design §16 #1)"
        )


# ---------------------------------------------------------------------------
# Chunk 1 -- deterministic MinHook project generator
# ---------------------------------------------------------------------------


def _normalize_targets(targets: Any) -> List[HookTarget]:
    """Coerce ``targets`` into a list of :class:`HookTarget` (strict)."""
    if isinstance(targets, HookTarget) or isinstance(targets, int):
        items: Iterable[Any] = [targets]
    elif isinstance(targets, (str, bytes, bytearray)):
        raise HookError(
            "targets must be HookTarget instances or integer addresses, not a string"
        )
    else:
        try:
            items = list(targets)
        except TypeError as exc:
            raise HookError(
                f"targets must be an iterable of HookTarget, got {type(targets).__name__}"
            ) from exc

    normalized: List[HookTarget] = []
    for item in items:
        if isinstance(item, HookTarget):
            normalized.append(item)
        elif isinstance(item, bool) or not isinstance(item, int):
            raise HookError(
                f"unsupported target entry: {type(item).__name__}; expected HookTarget or int"
            )
        else:
            normalized.append(HookTarget(func_addr=item))
    return normalized


def _boot_token(
    name: str,
    arch: str,
    grounded: bool,
    minhook_dir: str,
    records: Sequence[Dict[str, Any]],
) -> str:
    """Deterministic 32-hex-char boot-verification token."""
    payload = {
        "name": name,
        "arch": arch,
        "grounded": bool(grounded),
        "minhook_dir": minhook_dir,
        "targets": [
            {
                "func_addr": f"0x{record['func_addr']:x}",
                "symbol": record["symbol"],
                "arch": record["arch"],
                "prologue": record["prologue_hex"],
                "patch_len": record["patch_len"],
            }
            for record in records
        ],
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


def _write_text(path: Path, content: str) -> None:
    """Write UTF-8 text with explicit ``\\n`` newlines, closing the handle.

    Trap 7: ``Path.write_text`` opens and closes the file inside the call, so no
    handle is left dangling (no ``WinError 32`` on the generated tree).
    """
    path.write_text(content, encoding="utf-8", newline="\n")


_DLLMAIN_TEMPLATE = '''// ==============================================================================
// Toothpick native hook artifact: @@NAME@@ (@@ARCH@@)
// Generated by toothpick.hook.generate_hook_project -- do not hand-edit.
// Spec: docs/units/UNIT-P2-HOOK-PLAN.md Chunk 1 (MinHook trampoline pattern).
//
// Grounding status: @@GROUNDING@@ (Design §16 #1: receipts-or-refusal)
// Boot token: @@BOOT_TOKEN@@
// Targets: @@TARGET_COUNT@@
// MinHook directory: @@MINHOOK_DIR@@
//
// This DLL ships no injector and no evasion machinery.  It is a user-mode
// singleplayer-modding artifact: inject it into a process you own, on your own
// machine, or load it via a manual mapper that calls the exported
// ToothpickInit() entry point (see README.md).
// ==============================================================================

#include <windows.h>
#include <cstdint>
#include <cstdio>

#include <MinHook.h>

namespace {

const char* BOOT_TOKEN = @@BOOT_TOKEN_LITERAL@@;
const char* HOOK_NAME = @@NAME_LITERAL@@;
const char* GROUNDING_STATUS = @@GROUNDING_LITERAL@@;
const uintptr_t TARGET_COUNT = @@TARGET_COUNT@@;

volatile LONG g_init_state = 0;  // 0 = idle, 1 = hooks installed / in flight.

@@TARGET_BLOCKS@@
}  // namespace

// Idempotent entry point.  Safe to call twice (for example when an injector
// also loads the DLL through the Windows loader): the InterlockedCompareExchange
// gate installs hooks exactly once.
extern "C" __declspec(dllexport) BOOL WINAPI ToothpickInit(void) {
    if (InterlockedCompareExchange(&g_init_state, 1, 0) != 0) {
        return TRUE;
    }

    std::printf("[Toothpick] Boot verification token: %s\\n", BOOT_TOKEN);
    std::printf("[Toothpick] Hook artifact: %s (%llu target(s))\\n", HOOK_NAME,
                static_cast<unsigned long long>(TARGET_COUNT));
    std::printf("[Toothpick] Grounding status: %s\\n", GROUNDING_STATUS);

    if (MH_Initialize() != MH_OK) {
        std::printf("[Toothpick] MH_Initialize failed\\n");
        InterlockedExchange(&g_init_state, 0);
        return FALSE;
    }

    int failures = 0;
@@INSTALL_CALLS@@

    if (failures != 0) {
        std::printf("[Toothpick] %d hook(s) failed to install\\n", failures);
        return FALSE;
    }
    std::printf("[Toothpick] All hooks enabled\\n");
    return TRUE;
}

BOOL APIENTRY DllMain(HMODULE hModule, DWORD ul_reason_for_call, LPVOID lpReserved) {
    (void)lpReserved;
    switch (ul_reason_for_call) {
    case DLL_PROCESS_ATTACH:
        DisableThreadLibraryCalls(hModule);
        ToothpickInit();
        break;
    case DLL_PROCESS_DETACH:
        // Hooks are intentionally left installed: MH_DisableHook during process
        // teardown can race other threads (MinHook documentation).
        break;
    }
    return TRUE;
}
'''


_CMAKE_TEMPLATE = '''cmake_minimum_required(VERSION 3.20)

project(@@NAME@@ LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# MinHook is vendored by the operator -- nothing third-party is bundled here.
# Clone https://github.com/TsudaKageyu/minhook into @@MINHOOK_DIR@@ (the default)
# or configure with -DMINHOOK_DIR=<path-to-minhook>.
set(MINHOOK_DIR "@@MINHOOK_DIR@@" CACHE PATH "MinHook checkout (CMakeLists.txt + include/MinHook.h)")
if(NOT EXISTS "${MINHOOK_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "MinHook not found at ${MINHOOK_DIR}; clone it there or pass -DMINHOOK_DIR=<path>")
endif()

add_subdirectory("${MINHOOK_DIR}" "${CMAKE_BINARY_DIR}/minhook")

add_library(@@NAME@@ SHARED
    dllmain.cpp
)

target_include_directories(@@NAME@@ PRIVATE "${MINHOOK_DIR}/include")

# MinHook's own CMake target is named `minhook`.
target_link_libraries(@@NAME@@ PRIVATE minhook)

if(MSVC)
    target_compile_options(@@NAME@@ PRIVATE /W4)
endif()
'''


def _render_target_block(record: Dict[str, Any], index: int, total: int) -> str:
    """Render the C++ declaration/detour/install block for one target."""
    symbol = record["c_symbol"]
    display = record["symbol_display"]
    addr_hex = record["func_addr_hex"]
    prologue_state = record["prologue_status"]
    patch_len = "unvalidated" if record["patch_len"] is None else str(record["patch_len"])
    return f'''// --- target {index + 1}/{total}: {display} ({addr_hex}, {record["arch"]}) ---
// prologue: {prologue_state}; minimum trampoline patch length: {patch_len}
typedef void (*HookFn_{symbol})(void);
static HookFn_{symbol} g_original_{symbol} = nullptr;
static LPVOID g_target_{symbol} =
    reinterpret_cast<LPVOID>(static_cast<uintptr_t>({addr_hex}ULL));

// Signature-agnostic detour stub: no parameters are declared, so reading
// arguments requires re-declaring this prototype from the decompiled function.
static void Detour_{symbol}(void) {{
    std::printf("[Toothpick] hook fired: %s (%s)\\n",
                {_c_string_literal(display)}, {_c_string_literal(addr_hex)});
    if (g_original_{symbol} != nullptr) {{
        g_original_{symbol}();
    }}
}}

static MH_STATUS Install_{symbol}(void) {{
    MH_STATUS status = MH_CreateHook(
        g_target_{symbol},
        reinterpret_cast<LPVOID>(&Detour_{symbol}),
        reinterpret_cast<LPVOID*>(&g_original_{symbol}));
    if (status != MH_OK) {{
        std::printf("[Toothpick] MH_CreateHook failed for %s: %s\\n",
                    {_c_string_literal(display)}, MH_StatusToString(status));
        return status;
    }}
    status = MH_EnableHook(g_target_{symbol});
    if (status != MH_OK) {{
        std::printf("[Toothpick] MH_EnableHook failed for %s: %s\\n",
                    {_c_string_literal(display)}, MH_StatusToString(status));
    }}
    return status;
}}
'''


def _render_install_calls(records: Sequence[Dict[str, Any]]) -> str:
    """Render the per-target ``CreateHook``/``EnableHook`` invocation sequence."""
    lines = []
    for record in records:
        symbol = record["c_symbol"]
        lines.append(
            f"    if (Install_{symbol}() != MH_OK) {{\n"
            f"        ++failures;\n"
            f"    }}"
        )
    if not lines:
        return "    // no targets were requested"
    return "\n".join(lines)


def _render_dllmain(
    name: str,
    arch: str,
    boot_token: str,
    grounding_status: str,
    minhook_dir: str,
    records: Sequence[Dict[str, Any]],
) -> str:
    blocks = "\n".join(
        _render_target_block(record, index, len(records))
        for index, record in enumerate(records)
    )
    return (
        _DLLMAIN_TEMPLATE.replace("@@NAME@@", name)
        .replace("@@ARCH@@", arch)
        .replace("@@GROUNDING@@", grounding_status)
        .replace("@@GROUNDING_LITERAL@@", _c_string_literal(grounding_status))
        .replace("@@BOOT_TOKEN@@", boot_token)
        .replace("@@BOOT_TOKEN_LITERAL@@", _c_string_literal(boot_token))
        .replace("@@NAME_LITERAL@@", _c_string_literal(name))
        .replace("@@TARGET_COUNT@@", str(len(records)))
        .replace("@@MINHOOK_DIR@@", minhook_dir)
        .replace("@@TARGET_BLOCKS@@", blocks.rstrip("\n"))
        .replace("@@INSTALL_CALLS@@", _render_install_calls(records))
    )


def _render_readme(
    name: str,
    arch: str,
    boot_token: str,
    grounding_status: str,
    minhook_dir: str,
    grounded: bool,
    records: Sequence[Dict[str, Any]],
) -> str:
    rows = "\n".join(
        "| {idx} | {symbol} | `{addr}` | {arch} | {prologue} | {patch} | {grounding} |".format(
            idx=index + 1,
            symbol=record["symbol_display"],
            addr=record["func_addr_hex"],
            arch=record["arch"],
            prologue=record["prologue_status"],
            patch="unvalidated" if record["patch_len"] is None else record["patch_len"],
            grounding=record["grounding"],
        )
        for index, record in enumerate(records)
    )
    unverified_note = (
        ""
        if grounded
        else (
            "\n> **UNVERIFIED (allows-unverified)**: this project was generated "
            "without a confirmed `GROUNDED` dynamic receipt for every target. "
            "Treat every address below as a hypothesis until a live-read receipt "
            "exists in `toothpick.db` (Design §16 #1).\n"
        )
    )
    return f'''# {name} — Toothpick native hook artifact ({arch})

Generated by `toothpick.hook.generate_hook_project` (spec:
`docs/units/UNIT-P2-HOOK-PLAN.md` Chunk 1). MinHook trampoline pattern, derived
from the W13 `native_cpp` scaffolding in `src/toothpick/mod_builder.py`.

- **Grounding status**: `{grounding_status}`
- **Boot token**: `{boot_token}`
- **Architecture**: `{arch}` (one DLL cannot hook both x86 and x64 — generate one
  project per architecture)
- **MinHook directory**: `{minhook_dir}`
{unverified_note}
## Targets

| # | Symbol | Address | Arch | Prologue | Patch len | Grounding |
|---|--------|---------|------|----------|-----------|-----------|
{rows}

`UNVALIDATED` means no `prologue_bytes` were supplied by the caller, so the
minimum trampoline patch length was **not** computed. Nothing was invented:
obtain a real prologue dump (e.g. `toothpick query func <addr> --db
toothpick.db`) and regenerate with `HookTarget(prologue_bytes=...)`.

## Build

```bash
# 1. Vendor MinHook (not bundled — no third-party binaries in toothpick):
git clone https://github.com/TsudaKageyu/minhook {minhook_dir}

# 2. Configure and build (MSVC x64 developer prompt):
cmake -B build -S . -A x64 -DMINHOOK_DIR={minhook_dir}
cmake --build build --config Release
# -> build/Release/{name}.dll  (x86: -A Win32)
```

If `cmake`/MSVC is unavailable, the project is still structurally complete:
`dllmain.cpp`, `CMakeLists.txt` and this README are the whole artifact, and the
generated sources are deterministic.

## Inject

`LoadLibrary`-based injection (any standard injector, or your own harness):

```bash
# The process must be singleplayer and yours. Snapshot the VM first.
injector.exe --pid <pid> --dll build/Release/{name}.dll
```

On attach the DLL prints, to the target's stdout/debug console:

```
[Toothpick] Boot verification token: {boot_token}
[Toothpick] Grounding status: {grounding_status}
[Toothpick] All hooks enabled
```

Compare that token with the value above to confirm the artifact under test is
the artifact that was generated.

## Manual mapping

When the DLL is **manually mapped** rather than loaded by the Windows loader,
the loader never invokes `DllMain`, so no hook is installed automatically:

1. Map the image with your mapper of choice (toothpick ships none).
2. Call the exported entry point explicitly:
   `ToothpickInit` (`extern "C" __declspec(dllexport) BOOL WINAPI ToothpickInit(void)`).
   It is idempotent, so calling it after a loader-based `DllMain` is harmless.
3. Manual mapping changes import resolution: MinHook's `MH_Initialize` allocates
   trampoline memory and patches pages with `VirtualProtect`; if the mapper
   remaps sections with stricter protections, prepare trampolines *before*
   tightening them, or the hook install fails (`MH_ERROR_MEMORY_ALLOC` /
   `MH_ERROR_MEMORY_PROTECT` are printed by the generated code).

No injector, no driver and no evasion logic is generated here — the choice of
loader mechanism is the operator's (NEXT-TRACK P2 open question 2).

## Scope and anti-scope

- Singleplayer, own-machine, user-mode targets only. No kernel components.
- Multiplayer/anti-cheat integration is out of scope and refused by the
  grounding gate path (`allow_unverified` overrides are logged, not hidden).
- VM-protected targets (`PACKED_ABORT(VM_PROTECTED)`) never reach this generator:
  only the protector stub is visible to static analysis.
- EULA surface: modding clauses vary per title — verify before shipping a mod.

## Regenerating

The emitted bytes are a pure function of `(targets, name, grounded, minhook_dir)`.
Re-running the generator with the same inputs produces byte-identical files; the
boot token is derived from that same canonical input, not from a clock or a
random source.
'''


def generate_hook_project(
    targets: Any,
    out_dir: PathLike,
    name: str = "toothpick_hook",
    grounded: bool = True,
    minhook_dir: str = "external/minhook",
) -> Dict[str, Any]:
    """Emit a deterministic, buildable MinHook hook project.

    ``targets`` accepts :class:`HookTarget` instances (or bare integer
    addresses, which become ``HookTarget`` defaults), a single target, or a
    sequence of them.  Exactly three files are written — ``dllmain.cpp``,
    ``CMakeLists.txt``, ``README.md`` — in a fixed order and with byte-identical
    content for identical inputs.

    ``grounded=False`` stamps ``UNVERIFIED (allows-unverified)`` through the C++
    banner, the generated README and the returned metadata; it never claims
    grounding.  A supplied ``prologue_bytes`` that fails
    :func:`validate_prologue` raises :class:`HookError` (receipts-or-refusal:
    the generator will not emit a patch length it cannot back).

    Returns ``{status, name, arch, grounded, grounding_status, boot_token,
    minhook_dir, project_dir, files, targets}``.
    """
    target_list = _normalize_targets(targets)
    if not target_list:
        raise HookError("at least one HookTarget is required")

    arch = target_list[0].arch
    for target in target_list[1:]:
        if target.arch != arch:
            raise HookError(
                f"mixed architectures in one artifact: {arch} and {target.arch}; "
                "a single DLL cannot hook both x86 and x64 (split the drives)"
            )

    if not isinstance(name, str) or not name.strip():
        raise HookError(f"name must be a non-empty str, got {name!r}")
    c_name = _sanitize_cpp_identifier(name.strip(), default="ToothpickHook")
    if not isinstance(minhook_dir, str) or not minhook_dir.strip():
        raise HookError(f"minhook_dir must be a non-empty str, got {minhook_dir!r}")
    minhook_dir_norm = minhook_dir.strip().replace("\\", "/")

    if not isinstance(grounded, bool):
        raise HookError(
            "grounded must be a bool (True = the operator asserts a confirmed "
            f"GROUNDED receipt for every target); got {type(grounded).__name__} -- "
            "refusing to stamp a grounding claim from a truthy value"
        )
    grounding_status = "GROUNDED" if grounded else "UNVERIFIED (allows-unverified)"

    records: List[Dict[str, Any]] = []
    for index, target in enumerate(target_list):
        validation = (
            validate_prologue(target.prologue_bytes, target.arch)
            if target.prologue_bytes
            else None
        )
        if validation is not None and not validation["ok"]:
            raise HookError(
                f"target {target.func_addr_hex} ({target.symbol_display}) prologue "
                f"refused: {validation['reason']}"
            )
        base = _sanitize_cpp_identifier(target.symbol or target.func_addr_hex)
        records.append(
            {
                "index": index,
                "func_addr": target.func_addr,
                "func_addr_hex": target.func_addr_hex,
                "symbol": target.symbol,
                "symbol_display": target.symbol_display,
                "c_symbol": f"{base}_{index}",
                "arch": target.arch,
                "prologue_hex": target.prologue_bytes.hex(),
                "prologue_status": (
                    "UNVALIDATED (no prologue_bytes supplied)"
                    if validation is None
                    else f"VALIDATED ({validation['decoded_len']} byte(s), "
                    f"{len(validation['insns'])} instruction(s))"
                ),
                "patch_len": None if validation is None else validation["patch_len"],
                "patch_len_rel32": (
                    None if validation is None else validation["patch_len_rel32"]
                ),
                "patch_len_abs64": (
                    None if validation is None else validation["patch_len_abs64"]
                ),
                "grounding": grounding_status,
            }
        )

    boot_token = _boot_token(c_name, arch, grounded, minhook_dir_norm, records)

    project_dir = Path(out_dir)
    project_dir.mkdir(parents=True, exist_ok=True)

    contents = {
        "dllmain.cpp": _render_dllmain(
            c_name, arch, boot_token, grounding_status, minhook_dir_norm, records
        ),
        "CMakeLists.txt": _CMAKE_TEMPLATE.replace("@@NAME@@", c_name).replace(
            "@@MINHOOK_DIR@@", minhook_dir_norm
        ),
        "README.md": _render_readme(
            c_name,
            arch,
            boot_token,
            grounding_status,
            minhook_dir_norm,
            bool(grounded),
            records,
        ),
    }

    files: List[str] = []
    for file_name in _GENERATED_FILES:  # fixed emission order
        path = project_dir / file_name
        _write_text(path, contents[file_name])
        files.append(str(path))

    return {
        "status": "success",
        "name": c_name,
        "arch": arch,
        "grounded": bool(grounded),
        "grounding_status": grounding_status,
        "boot_token": boot_token,
        "minhook_dir": minhook_dir_norm,
        "project_dir": str(project_dir),
        "files": sorted(files),
        "targets": [dict(record, functional_target=None) for record in records],
    }
