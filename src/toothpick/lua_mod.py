"""
Toothpick J3: Lua mod scaffolding, Workshop packaging, and the class-file patch
descriptor.

Implements **Chunk 1** of ``docs/units/UNIT-J3-LUA-PLAN.md`` -- the data
contracts, the deterministic Project-Zomboid-class Lua mod generator, the
Workshop-style packager, and the (descriptive only) patch descriptor.  Chunk 2
(CLI ``lua build`` / ``lua pack`` / ``lua patch-emit`` plus the
``tests/test_lua_mod.py`` battery) lands in follow-up drives.

Contents
--------
* :class:`LuaModConfig` -- strict dataclass (``name``, ``mod_id``, ``version``,
  ``target_class``, ``target_method``, ``target_desc``, ``author``) with
  identifier validation (``[A-Za-z0-9_.-]+``), semver-ish version validation,
  JVM class/method/descriptor validation, and a refusal of ``.``/``..`` in
  ``mod_id`` so a mod id can never escape its output directory.
* :func:`generate_lua_mod` -- writes ``media/lua/client/<mod_id>/main.lua``
  (hook stub naming the JVM target as ``Class.method:desc`` plus the boot-token
  print), ``mod.info`` (``name``/``id``/``version``/``description``) and
  ``README.md`` (install steps, boot verification, Workshop upload steps).
  Byte-identical for identical inputs; ``grounded=False`` stamps a loud
  ``UNVERIFIED (allows-unverified)`` banner into ``main.lua`` and ``README.md``
  and never claims grounding.
* :func:`package_workshop` -- packs a generated project into
  ``<mod_id>.zip`` (deterministic entries: fixed order, fixed metadata, no
  timestamps) plus ``<mod_id>.workshop.json``, a manifest carrying
  ``mod_id``/``version``/``files`` (per-file ``path``/``size``/``sha256``) and a
  manifest-level ``sha256`` over the canonical file list.  Steam API uploads are
  explicitly out of scope: the manifest is the hand-off.
* :func:`emit_patch_descriptor` / :func:`parse_patch_descriptor` -- the
  ``{class, method, desc, op, lua_hook}`` patch vocabulary
  (:data:`PATCH_OPS` = ``prepend`` | ``append`` | ``replace``) with strict
  schema validation (required keys, unknown keys refused, op vocabulary,
  ``()V``-shaped JVM descriptor syntax, class/method name syntax) and
  round-trip equality.  A descriptor *describes* edits for external bytecode
  tools; this module never reads or rewrites a class file.
* :class:`LuaModError` -- the single error type for this module.

Discipline
----------
* **Receipts-or-refusal**: the generator never invents a grounding claim.  It
  takes an explicit ``grounded: bool`` (a bare truthy value is refused) and
  writes the resulting status into ``main.lua``, ``README.md`` and the returned
  metadata (``mod.info`` stays a plain metadata file).  Enforcement itself (the
  ``HookGate``-style receipt check on the method ordinal) lives in the CLI and
  reuses ``toothpick.hook.HookGate`` -- no new taxonomy is introduced here.
* **Pure/offline/deterministic**: stdlib only (``hashlib``, ``json``, ``re``,
  ``zipfile``).  No network, no process, no clock and no randomness: the boot
  token is a digest of the canonical config, so the same inputs always produce
  the same bytes.
* **Trap 7 (handle hygiene)**: every file is written through
  :meth:`pathlib.Path.write_text` with an explicit encoding and
  ``newline="\\n"``, and the zip is written inside a ``with`` block, so no
  handle is left dangling (no Windows ``WinError 32`` on the generated tree).
* **Trap 11 (``faulthandler``)**: this module imports no ``faulthandler`` and
  installs no exception hooks, so native-engine suites keep running under
  ``-p no:faulthandler``.
* **Trap 14 (plain data)**: every returned structure is JSON-plain -- ``dict``,
  ``list``, ``str``, ``int``, ``bool`` only.  No ``Path`` objects, no tuples,
  no sets leak into results, so ``json.dumps`` on any return value works.
"""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

__all__ = [
    "LuaModError",
    "LuaModConfig",
    "PATCH_OPS",
    "PATCH_KEYS",
    "DESCRIPTOR_SCHEMA",
    "DESCRIPTOR_SCHEMA_VERSION",
    "is_valid_identifier",
    "is_valid_method_descriptor",
    "validate_patches",
    "generate_lua_mod",
    "package_workshop",
    "emit_patch_descriptor",
    "parse_patch_descriptor",
]

PathLike = Union[str, Path]

#: The patch-descriptor operation vocabulary (UNIT-J3-LUA-PLAN.md section 2).
PATCH_OPS = frozenset({"prepend", "append", "replace"})

#: The exact key set of one patch entry, in canonical emission order.
PATCH_KEYS = ("class", "method", "desc", "op", "lua_hook")

#: Marker written into (and required from) every descriptor document.
DESCRIPTOR_SCHEMA = "toothpick.patch_descriptor"

#: Descriptor document schema revision.
DESCRIPTOR_SCHEMA_VERSION = 1

#: Zomboid console line the receipt-checking harness greps for (boot token).
BOOT_TOKEN_PREFIX = "[Toothpick] Boot verification token: "

#: Grounded / explicit-override status strings (never a third, vaguer value).
GROUNDING_STATUS_GROUNDED = "GROUNDED"
GROUNDING_STATUS_UNVERIFIED = "UNVERIFIED (allows-unverified)"

#: Project-relative emission order of the three generated files.
_GENERATED_RELATIVE_FILES = ("main.lua", "mod.info", "README.md")

#: Fixed ZIP metadata: identical inputs must produce identical zip bytes.
_ZIP_DATE_TIME = (1980, 1, 1, 0, 0, 0)
_ZIP_EXTERNAL_ATTR = 0o644 << 16

# --------------------------------------------------------------------------- #
# Validation grammar
# --------------------------------------------------------------------------- #

#: Identifiers (mod id, ...): ``[A-Za-z0-9_.-]+`` per the unit plan.
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9_.\-]+")

#: Semver-ish version: ``1``, ``1.0``, ``1.0.0``, ``1.0.0.0``, ``1.0.0-beta.1``.
_VERSION_RE = re.compile(r"\d+(?:\.\d+){0,3}(?:[-+][0-9A-Za-z.\-]+)?")

#: JVM binary class name, internal (``com/foo/Bar``) or dotted (``com.foo.Bar``).
_CLASS_RE = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*(?:[./][A-Za-z_$][A-Za-z0-9_$]*)*")

#: JVM method name, including the pseudo-names ``<init>`` / ``<clinit>``.
_METHOD_NAME_RE = re.compile(r"[A-Za-z_$<][A-Za-z0-9_$<>]*")

#: JVM field-type grammar (base, object, or array of either).
_BASE_TYPE = r"[BCDFIJSZ]"
_OBJECT_TYPE = r"L[A-Za-z0-9_$./\-]+;"
_FIELD_TYPE = rf"(?:{_BASE_TYPE}|{_OBJECT_TYPE}|\[+(?:{_BASE_TYPE}|{_OBJECT_TYPE}))"

#: JVM method descriptor: ``(params)return`` -- the ``()V``-shaped grammar.
_METHOD_DESC_RE = re.compile(rf"\({_FIELD_TYPE}*\)(?:V|{_FIELD_TYPE})")


class LuaModError(Exception):
    """Base exception for Lua mod scaffolding, packaging and descriptor errors."""


# --------------------------------------------------------------------------- #
# Coercion / validation helpers
# --------------------------------------------------------------------------- #


def _require_text(
    value: Any,
    field: str,
    *,
    allow_empty: bool = False,
) -> str:
    """Return ``value`` as a stripped single-line ``str``, or raise.

    Single-line is not cosmetic: the generated ``mod.info`` is a
    ``key=value`` text file, so a newline in a field would silently forge a
    second key.  The CR/LF check runs on the *raw* value, before stripping, so a
    trailing newline is a refusal rather than something ``strip`` quietly hides.
    """
    if not isinstance(value, str):
        raise LuaModError(f"{field} must be a str, got {type(value).__name__}")
    if "\n" in value or "\r" in value:
        raise LuaModError(f"{field} must be a single line (no CR/LF), got {value!r}")
    text = value.strip()
    if not allow_empty and not text:
        raise LuaModError(f"{field} must be a non-empty string")
    return text


def is_valid_identifier(value: Any) -> bool:
    """``True`` when ``value`` is a path-safe identifier (``[A-Za-z0-9_.-]+``).

    A bare dot matches the character class but is *not* a valid identifier here,
    and neither is any string containing ``..``: both would make the identifier
    usable as a path component pointing outside the output tree.
    """
    if not isinstance(value, str):
        return False
    if not _IDENTIFIER_RE.fullmatch(value):
        return False
    return value not in (".", "..") and ".." not in value


def _require_identifier(value: Any, field: str) -> str:
    """Validate an identifier (``[A-Za-z0-9_.-]+``, never a path escape)."""
    text = _require_text(value, field)
    if ".." in text:
        raise LuaModError(
            f"{field} must not contain '..' (identifier would be usable as a path "
            f"escape), got {value!r}"
        )
    if text in (".", ".."):
        raise LuaModError(f"{field} must not be '.' or '..', got {value!r}")
    if not _IDENTIFIER_RE.fullmatch(text):
        raise LuaModError(
            f"{field} must match [A-Za-z0-9_.-]+, got {value!r}"
        )
    return text


def _require_version(value: Any, field: str) -> str:
    """Validate a semver-ish version string (``1.0.0``, ``0.1``, ``1.0.0-beta.1``)."""
    text = _require_text(value, field)
    if not _VERSION_RE.fullmatch(text):
        raise LuaModError(
            f"{field} must be semver-ish (e.g. '1.0.0' or '1.0.0-beta.1'), got {value!r}"
        )
    return text


def _require_class_name(value: Any, field: str) -> str:
    """Validate a JVM binary class name (internal ``a/b/C`` or dotted ``a.b.C``)."""
    text = _require_text(value, field)
    if not _CLASS_RE.fullmatch(text):
        raise LuaModError(
            f"{field} must be a JVM binary class name (internal 'com/foo/Bar' or "
            f"dotted 'com.foo.Bar'), got {value!r}"
        )
    return text


def _require_method_name(value: Any, field: str) -> str:
    """Validate a JVM method name (``tick``, ``<init>``, ...)."""
    text = _require_text(value, field)
    if not _METHOD_NAME_RE.fullmatch(text):
        raise LuaModError(f"{field} must be a JVM method name, got {value!r}")
    return text


def is_valid_method_descriptor(value: Any) -> bool:
    """``True`` when ``value`` is a well-formed JVM method descriptor (``()V``)."""
    if not isinstance(value, str) or not value:
        return False
    if value != value.strip():
        return False
    return _METHOD_DESC_RE.fullmatch(value) is not None


def _require_method_descriptor(value: Any, field: str) -> str:
    """Validate a JVM method descriptor, e.g. ``()V`` or ``(IZLjava/lang/String;)V``."""
    if not isinstance(value, str):
        raise LuaModError(f"{field} must be a str, got {type(value).__name__}")
    if not is_valid_method_descriptor(value):
        raise LuaModError(
            f"{field} must be a JVM method descriptor of the form '(params)return' "
            f"(e.g. '()V'), got {value!r}"
        )
    return value


def _canonical_json(payload: Any) -> str:
    """Deterministic JSON text for hashing (sorted keys, fixed separators)."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _canonical_sha256(payload: Any) -> str:
    """SHA-256 hex digest of :func:`_canonical_json`."""
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _write_text(path: Path, content: str) -> None:
    """Write UTF-8 text with explicit ``\\n`` newlines, closing the handle.

    Trap 7: ``Path.write_text`` opens and closes the file inside the call, so no
    handle is left dangling (no ``WinError 32`` on the generated tree).
    """
    path.write_text(content, encoding="utf-8", newline="\n")


def _resolve_dir(value: Any, field: str, *, must_exist: bool = False) -> Path:
    """Coerce ``value`` to a directory :class:`Path` (creating it if needed)."""
    try:
        path = Path(value)
    except TypeError as exc:
        raise LuaModError(f"{field} must be a path-like value, got {value!r}") from exc
    if path.exists() and not path.is_dir():
        raise LuaModError(f"{field} exists and is not a directory: {path}")
    if must_exist and not path.is_dir():
        raise LuaModError(f"{field} is not an existing directory: {path}")
    return path


def _lua_string(text: Any) -> str:
    """Render ``text`` as a single-line double-quoted Lua string literal."""
    raw = str(text)
    escaped = (
        raw.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\r", " ")
        .replace("\n", " ")
    )
    return f'"{escaped}"'


# --------------------------------------------------------------------------- #
# Chunk 1 -- configuration contract
# --------------------------------------------------------------------------- #


@dataclass
class LuaModConfig:
    """Strict configuration for one scaffolded Lua mod.

    Construction validates rather than sanitizes: a bad identifier, version,
    class name, method name or descriptor raises :class:`LuaModError` instead of
    producing an artifact that claims something it does not know.
    """

    name: str
    mod_id: str
    version: str
    target_class: str
    target_method: str
    target_desc: str
    author: str = ""

    def __post_init__(self) -> None:
        self.name = _require_text(self.name, "name")
        self.mod_id = _require_identifier(self.mod_id, "mod_id")
        self.version = _require_version(self.version, "version")
        self.target_class = _require_class_name(self.target_class, "target_class")
        self.target_method = _require_method_name(self.target_method, "target_method")
        self.target_desc = _require_method_descriptor(self.target_desc, "target_desc")
        self.author = _require_text(self.author, "author", allow_empty=True)

    @property
    def target_symbol(self) -> str:
        """The JVM target as ``Class.method:desc`` (the CLI's display form)."""
        return f"{self.target_class}.{self.target_method}:{self.target_desc}"

    @property
    def main_lua_relative(self) -> str:
        """Project-relative ``main.lua`` path (``media/lua/client/<mod_id>/main.lua``)."""
        return f"media/lua/client/{self.mod_id}/main.lua"

    @property
    def description(self) -> str:
        """Single-line ``mod.info`` description (deterministic, ASCII-safe)."""
        return (
            f"{self.name} v{self.version} -- target {self.target_symbol} "
            "(generated by Toothpick)"
        )

    def to_dict(self) -> Dict[str, Any]:
        """JSON-plain view with a stable key order (Trap 14)."""
        return {
            "name": self.name,
            "mod_id": self.mod_id,
            "version": self.version,
            "target_class": self.target_class,
            "target_method": self.target_method,
            "target_desc": self.target_desc,
            "target_symbol": self.target_symbol,
            "author": self.author,
        }


# --------------------------------------------------------------------------- #
# Chunk 1 -- deterministic Lua mod scaffolding
# --------------------------------------------------------------------------- #


def _boot_token(config: LuaModConfig, grounded: bool) -> str:
    """Deterministic 32-hex-char boot-verification token.

    Derived from the canonical configuration -- never from a clock or a random
    source -- so two identical generations are byte-identical and the token
    identifies the artifact under test (Design §16 #1).
    """
    payload = {
        "generator": "toothpick.lua_mod",
        "config": config.to_dict(),
        "grounded": bool(grounded),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()[:32]


_MAIN_LUA_TEMPLATE = '''@@HEADER@@local MOD_ID = @@MOD_ID_LITERAL@@
local MOD_NAME = @@NAME_LITERAL@@
local MOD_VERSION = @@VERSION_LITERAL@@
local BOOT_TOKEN = @@BOOT_TOKEN_LITERAL@@
local TARGET_CLASS = @@TARGET_CLASS_LITERAL@@
local TARGET_METHOD = @@TARGET_METHOD_LITERAL@@
local TARGET_DESC = @@TARGET_DESC_LITERAL@@
local TARGET_SYMBOL = @@TARGET_LITERAL@@
local GROUNDING_STATUS = @@GROUNDING_LITERAL@@

-- Boot token print: the receipt-checking harness greps for this exact line.
print("@@BOOT_TOKEN_PREFIX@@" .. BOOT_TOKEN)
print("[Toothpick] Lua mod loaded: " .. MOD_ID .. " v" .. MOD_VERSION)
print("[Toothpick] Grounding status: " .. GROUNDING_STATUS)
print("[Toothpick] Target JVM method (descriptor only, not applied here): " .. TARGET_SYMBOL)

local ToothpickLuaMod = {}

-- Hook stub.  Scaffolding only: it names the intended hook point and does
-- nothing else.  A class-file edit is *described* by the patch descriptor
-- (`toothpick lua patch-emit`) and applied by an external bytecode tool; this
-- module never rewrites class files (UNIT-J3 scope: no Lua/JVM binding depth).
function ToothpickLuaMod.onHook()
    print("[Toothpick] hook stub reached for " .. TARGET_SYMBOL)
    return true
end

function ToothpickLuaMod.onGameStart()
    print("[Toothpick] " .. MOD_NAME .. " v" .. MOD_VERSION .. " ready")
    ToothpickLuaMod.onHook()
end

-- Zomboid entry point: register once per Lua load when the Events table exists.
-- A headless/no-client host leaves Events nil -- never crash the mod for it.
if Events and Events.OnGameStart then
    Events.OnGameStart.Add(ToothpickLuaMod.onGameStart)
end

return ToothpickLuaMod
'''

_MOD_INFO_TEMPLATE = '''name=@@NAME@@
id=@@MOD_ID@@
version=@@VERSION@@
description=@@DESCRIPTION@@
'''

_README_TEMPLATE = '''# @@NAME@@ (`@@MOD_ID@@`) v@@VERSION@@

Toothpick-generated Project Zomboid Lua mod scaffolding.  Generated by
`toothpick.lua_mod.generate_lua_mod`; spec: `docs/units/UNIT-J3-LUA-PLAN.md`
Chunk 1.

- **Mod id**: `@@MOD_ID@@`
- **Version**: `@@VERSION@@`
- **Author**: @@AUTHOR@@
- **Grounding status**: `@@GROUNDING_STATUS@@`
- **Boot token**: `@@BOOT_TOKEN@@`
- **Target JVM method**: `@@TARGET@@` (descriptor only -- this mod does not
  apply, describe-with-authority or execute any class-file edit)
@@UNVERIFIED_NOTE@@
## Generated files

| Path | Purpose |
|------|---------|
| `@@MAIN_LUA@@` | Lua entry point: boot-token print + hook stub |
| `mod.info` | Zomboid mod metadata (`name`/`id`/`version`/`description`) |
| `README.md` | This file |

Re-running the generator with the same configuration reproduces these files
byte-for-byte; the boot token is a digest of the configuration, not a timestamp.

## Install (Project Zomboid)

1. Copy (or symlink) this whole folder into your Zomboid mods directory as
   `<mod_id>`:
   - Windows: `%USERPROFILE%\\Zomboid\\Mods\\@@MOD_ID@@`
   - Linux: `~/Zomboid/Mods/@@MOD_ID@@`
2. Keep the layout intact -- `mod.info` must sit at the folder root and the Lua
   file inside `media/lua/client/@@MOD_ID@@/`.
3. Launch the game, open **Mods** and enable **@@NAME@@**.
4. Verify the boot: the console must print exactly

   ```
   @@BOOT_TOKEN_PREFIX@@@@BOOT_TOKEN@@
   ```

   The boot token identifies the artifact under test; if it differs from
   `@@BOOT_TOKEN@@` above, the wrong folder is loaded.

## Workshop upload (Steam)

Steam API upload calls are out of scope for this module: the manifest is the
hand-off to the Workshop tooling.

1. Package the project:

   ```bash
   toothpick lua pack . --out dist/
   ```

   -> `dist/@@MOD_ID@@.zip` plus `dist/@@MOD_ID@@.workshop.json`.
2. Check `@@MOD_ID@@.workshop.json`: every entry under `files` carries its own
   `sha256`, and the manifest-level `sha256` covers the canonical file list.
   Attach that manifest (and the digest) to the upload record so the published
   bytes stay attributable.
3. Upload the zip through the Project Zomboid Workshop upload flow (the in-game
   uploader or the Steam Workshop item editor), then set the item's title to
   `@@NAME@@` and its description to this project's `mod.info` description.
4. Keep the zip, the manifest and this README together: the manifest records
   what was uploaded.

## Describing a class-file edit (patch descriptor)

This mod is scaffolding.  A real JVM-side edit is *described* as data and
applied by an external bytecode tool:

```json
{
  "class": "@@TARGET_CLASS@@",
  "method": "@@TARGET_METHOD@@",
  "desc": "@@TARGET_DESC@@",
  "op": "replace",
  "lua_hook": "ToothpickLuaMod.onHook"
}
```

`op` is one of `prepend`, `append`, `replace`; `desc` must be a JVM method
descriptor (`()V`-shaped); the round-trip through
`emit_patch_descriptor`/`parse_patch_descriptor` is byte-stable.

## Scope and anti-scope

- Scaffolding and packaging only: no Lua runtime depth, no JVM binding, no
  class-file rewriting, no Steam API calls, no network access.
- Singleplayer, own-machine, own-copy modding.  EULA surface varies per title:
  verify before publishing.
- Grounding enforcement (the receipt check on the method ordinal) happens at the
  CLI through `toothpick.hook.HookGate`, not in this folder.
'''


def _render_main_lua(config: LuaModConfig, boot_token: str, grounded: bool) -> str:
    """Render ``media/lua/client/<mod_id>/main.lua`` deterministically."""
    grounding_status = (
        GROUNDING_STATUS_GROUNDED if grounded else GROUNDING_STATUS_UNVERIFIED
    )
    header_lines = [
        "-- =============================================================================",
        "-- Toothpick Lua mod (Project Zomboid class): "
        f"{config.name} v{config.version}",
        f"-- Mod id: {config.mod_id}",
        f"-- Target JVM method: {config.target_symbol} (descriptor only)",
        f"-- Grounding status: {grounding_status} (Design §16 #1: receipts-or-refusal)",
        f"-- Boot token: {boot_token}",
        "-- Generated by toothpick.lua_mod.generate_lua_mod -- do not hand-edit.",
        "-- =============================================================================",
    ]
    if not grounded:
        header_lines.append(
            "-- UNVERIFIED (allows-unverified): no confirmed GROUNDED receipt backs "
            "this target."
        )
        header_lines.append(
            "-- Treat the target, the hook stub and the patch vocabulary below as a "
            "hypothesis"
        )
        header_lines.append(
            "-- until a live receipt exists in toothpick.db (Design §16 #1)."
        )
    header = "\n".join(header_lines) + "\n\n"
    return (
        _MAIN_LUA_TEMPLATE.replace("@@HEADER@@", header)
        .replace("@@BOOT_TOKEN_PREFIX@@", BOOT_TOKEN_PREFIX)
        .replace("@@MOD_ID_LITERAL@@", _lua_string(config.mod_id))
        .replace("@@NAME_LITERAL@@", _lua_string(config.name))
        .replace("@@VERSION_LITERAL@@", _lua_string(config.version))
        .replace("@@BOOT_TOKEN_LITERAL@@", _lua_string(boot_token))
        .replace("@@TARGET_CLASS_LITERAL@@", _lua_string(config.target_class))
        .replace("@@TARGET_METHOD_LITERAL@@", _lua_string(config.target_method))
        .replace("@@TARGET_DESC_LITERAL@@", _lua_string(config.target_desc))
        .replace("@@TARGET_LITERAL@@", _lua_string(config.target_symbol))
        .replace("@@GROUNDING_LITERAL@@", _lua_string(grounding_status))
    )


def _render_mod_info(config: LuaModConfig) -> str:
    """Render ``mod.info`` (``name``/``id``/``version``/``description``)."""
    return (
        _MOD_INFO_TEMPLATE.replace("@@NAME@@", config.name)
        .replace("@@MOD_ID@@", config.mod_id)
        .replace("@@VERSION@@", config.version)
        .replace("@@DESCRIPTION@@", config.description)
    )


def _render_readme(config: LuaModConfig, boot_token: str, grounded: bool) -> str:
    """Render ``README.md`` (install + Workshop steps + grounding banner)."""
    grounding_status = (
        GROUNDING_STATUS_GROUNDED if grounded else GROUNDING_STATUS_UNVERIFIED
    )
    if grounded:
        unverified_note = ""
    else:
        unverified_note = (
            "\n> **UNVERIFIED (allows-unverified)**: this project was generated "
            "without a confirmed `GROUNDED` dynamic receipt for the target method "
            "ordinal. Treat every target, hook stub and patch example below as a "
            "hypothesis until a live receipt exists in `toothpick.db` "
            "(Design §16 #1).\n"
        )
    author = config.author or "(unspecified)"
    return (
        _README_TEMPLATE.replace("@@UNVERIFIED_NOTE@@", unverified_note)
        .replace("@@NAME@@", config.name)
        .replace("@@MOD_ID@@", config.mod_id)
        .replace("@@VERSION@@", config.version)
        .replace("@@AUTHOR@@", author)
        .replace("@@GROUNDING_STATUS@@", grounding_status)
        .replace("@@BOOT_TOKEN_PREFIX@@", BOOT_TOKEN_PREFIX)
        .replace("@@BOOT_TOKEN@@", boot_token)
        .replace("@@TARGET@@", config.target_symbol)
        .replace("@@TARGET_CLASS@@", config.target_class)
        .replace("@@TARGET_METHOD@@", config.target_method)
        .replace("@@TARGET_DESC@@", config.target_desc)
        .replace("@@MAIN_LUA@@", config.main_lua_relative)
    )


def generate_lua_mod(
    config: LuaModConfig,
    out_dir: PathLike,
    grounded: bool = True,
) -> Dict[str, Any]:
    """Emit a deterministic Project Zomboid Lua mod project.

    Writes exactly three files, in a fixed order and with byte-identical content
    for identical inputs: ``media/lua/client/<mod_id>/main.lua`` (boot-token
    print plus the hook stub that names ``Class.method:desc``), ``mod.info``
    (``name``/``id``/``version``/``description``) and ``README.md`` (install
    steps, boot verification, Workshop upload steps, patch-descriptor example).

    ``grounded`` must be a real ``bool``: ``False`` stamps
    ``UNVERIFIED (allows-unverified)`` into ``main.lua`` and ``README.md`` and
    reports it in the returned metadata; ``mod.info`` stays a plain
    ``name``/``id``/``version``/``description`` file either way.  It never claims
    grounding.  A truthy non-bool (``1``, ``"yes"``) is refused, because a
    grounding claim is a receipt: only a real ``bool`` may produce it.

    Returns ``{status, project_dir, files, mod_id, name, version, grounded,
    grounding_status, boot_token, target, main_lua}`` (JSON-plain, Trap 14).
    """
    if not isinstance(config, LuaModConfig):
        raise LuaModError(
            f"config must be a LuaModConfig, got {type(config).__name__}"
        )
    if not isinstance(grounded, bool):
        raise LuaModError(
            "grounded must be a bool (True = a confirmed GROUNDED receipt backs "
            f"this target); got {type(grounded).__name__} -- refusing to stamp a "
            "grounding claim from a truthy value"
        )

    project_dir = _resolve_dir(out_dir, "out_dir")
    project_dir.mkdir(parents=True, exist_ok=True)

    boot_token = _boot_token(config, grounded)
    grounding_status = (
        GROUNDING_STATUS_GROUNDED if grounded else GROUNDING_STATUS_UNVERIFIED
    )

    contents = {
        "main.lua": _render_main_lua(config, boot_token, grounded),
        "mod.info": _render_mod_info(config),
        "README.md": _render_readme(config, boot_token, grounded),
    }

    files: List[str] = []
    for name in _GENERATED_RELATIVE_FILES:
        relative = config.main_lua_relative if name == "main.lua" else name
        path = project_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_text(path, contents[name])
        files.append(str(path))

    return {
        "status": "success",
        "project_dir": str(project_dir),
        "files": sorted(files),
        "mod_id": config.mod_id,
        "name": config.name,
        "version": config.version,
        "grounded": bool(grounded),
        "grounding_status": grounding_status,
        "boot_token": boot_token,
        "target": config.target_symbol,
        "main_lua": str(project_dir / config.main_lua_relative),
    }


# --------------------------------------------------------------------------- #
# Chunk 1 -- Workshop packaging (zip + per-file-sha256 manifest)
# --------------------------------------------------------------------------- #


def _read_mod_info(project_dir: Path) -> Tuple[str, str]:
    """Read ``(mod_id, version)`` from ``mod.info``, falling back to the folder name.

    The id is validated again here (identifier grammar, no ``..``) because it
    becomes a filename stem: a hand-edited ``mod.info`` must not be able to steer
    the zip/manifest out of the output directory.
    """
    mod_id = ""
    version = ""
    info = project_dir / "mod.info"
    if info.is_file():
        for line in info.read_text(encoding="utf-8").splitlines():
            entry = line.strip()
            if not entry or entry.startswith("#") or "=" not in entry:
                continue
            key, _, value = entry.partition("=")
            key = key.strip().lower()
            value = value.strip()
            if key == "id" and value:
                mod_id = value
            elif key == "version" and value:
                version = value
    if not mod_id:
        mod_id = project_dir.name
    mod_id = _require_identifier(mod_id, "mod_id (from mod.info)")
    if version:
        version = _require_version(version, "version (from mod.info)")
    return mod_id, version


def package_workshop(
    project_dir: PathLike,
    out_path: Optional[PathLike] = None,
) -> Dict[str, Any]:
    """Package a generated mod project for Workshop distribution.

    ``project_dir`` is an existing mod project (the tree
    :func:`generate_lua_mod` wrote).  ``out_path`` names the **output
    directory** and defaults to the project's parent; it is created when
    missing.

    Writes ``<mod_id>.zip`` (deterministic entries: sorted project-relative
    POSIX paths, fixed 1980-01-01 metadata, no timestamps or host data) and
    ``<mod_id>.workshop.json``::

        {
          "mod_id": "...",
          "version": "...",
          "files": [{"path": "...", "size": 123, "sha256": "..."}, ...],
          "sha256": "<digest over the canonical file list>"
        }

    Any previously written ``<mod_id>.zip`` / ``<mod_id>.workshop.json`` inside
    the project tree is excluded from the file list, so packaging into the
    project directory is stable on re-runs.  Steam API uploads are out of scope
    (manifest only).

    Returns ``{status, zip_path, manifest_path, manifest, zip_sha256}``
    (JSON-plain, Trap 14).
    """
    project = _resolve_dir(project_dir, "project_dir", must_exist=True)
    out_dir = (
        Path(out_path) if out_path is not None else project.parent
    )
    out_dir = _resolve_dir(out_dir, "out_path")
    out_dir.mkdir(parents=True, exist_ok=True)

    mod_id, version = _read_mod_info(project)
    zip_name = f"{mod_id}.zip"
    manifest_name = f"{mod_id}.workshop.json"

    entries: List[Dict[str, Any]] = []
    for path in sorted(
        project.rglob("*"), key=lambda item: item.relative_to(project).as_posix()
    ):
        if path.is_dir():
            continue
        relative = path.relative_to(project).as_posix()
        if relative in (zip_name, manifest_name):
            # Idempotence: packaging into the project directory must not fold a
            # previous package into the next manifest/zip.
            continue
        data = path.read_bytes()
        entries.append(
            {
                "path": relative,
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )

    manifest: Dict[str, Any] = {
        "mod_id": mod_id,
        "version": version,
        "files": entries,
        "sha256": _canonical_sha256(
            {"mod_id": mod_id, "version": version, "files": entries}
        ),
    }

    zip_path = out_dir / zip_name
    manifest_path = out_dir / manifest_name

    with zipfile.ZipFile(
        zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for entry in entries:
            info = zipfile.ZipInfo(entry["path"], date_time=_ZIP_DATE_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = _ZIP_EXTERNAL_ATTR
            archive.writestr(info, (project / entry["path"]).read_bytes())

    _write_text(
        manifest_path,
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
    )

    return {
        "status": "success",
        "zip_path": str(zip_path),
        "manifest_path": str(manifest_path),
        "manifest": manifest,
        "zip_sha256": hashlib.sha256(zip_path.read_bytes()).hexdigest(),
    }


# --------------------------------------------------------------------------- #
# Chunk 1 -- patch descriptor (describes edits; never applies them)
# --------------------------------------------------------------------------- #


def _normalize_patch(patch: Any, index: int) -> Dict[str, str]:
    """Validate one patch entry and return it in canonical key order."""
    where = f"patch[{index}]"
    if not isinstance(patch, Mapping):
        raise LuaModError(f"{where} must be an object, got {type(patch).__name__}")
    keys = list(patch.keys())
    if any(not isinstance(key, str) for key in keys):
        raise LuaModError(f"{where} has a non-string key: {keys!r}")
    unknown = sorted(set(keys) - set(PATCH_KEYS))
    if unknown:
        raise LuaModError(
            f"{where} has unknown key(s) {unknown}; allowed keys: {list(PATCH_KEYS)}"
        )
    missing = [key for key in PATCH_KEYS if key not in patch]
    if missing:
        raise LuaModError(
            f"{where} is missing required key(s) {missing}; required keys: "
            f"{list(PATCH_KEYS)}"
        )

    op = patch["op"]
    if not isinstance(op, str) or op not in PATCH_OPS:
        raise LuaModError(
            f"{where}.op must be one of {sorted(PATCH_OPS)}, got {op!r}"
        )
    hook = patch["lua_hook"]
    if not isinstance(hook, str) or not hook.strip():
        raise LuaModError(
            f"{where}.lua_hook must be a non-empty string, got {hook!r}"
        )

    return {
        "class": _require_class_name(patch["class"], f"{where}.class"),
        "method": _require_method_name(patch["method"], f"{where}.method"),
        "desc": _require_method_descriptor(patch["desc"], f"{where}.desc"),
        "op": op,
        "lua_hook": hook,
    }


def validate_patches(patches: Any) -> List[Dict[str, str]]:
    """Validate a patch list against the descriptor schema (strict).

    Refuses a non-list, an empty list, and any entry that is not an object with
    exactly the keys :data:`PATCH_KEYS`, in-vocabulary ``op``, well-formed JVM
    class/method/descriptor syntax and a non-empty ``lua_hook``.  Returns a new
    list of fresh dicts in canonical key order (JSON-plain, Trap 14).
    """
    if isinstance(patches, (str, bytes, bytearray)) or isinstance(patches, Mapping):
        raise LuaModError(
            "patches must be a list of patch objects, got "
            f"{type(patches).__name__}"
        )
    try:
        items = list(patches)
    except TypeError as exc:
        raise LuaModError(
            f"patches must be a list of patch objects, got {type(patches).__name__}"
        ) from exc
    if not items:
        raise LuaModError("patches must contain at least one patch description")
    return [_normalize_patch(patch, index) for index, patch in enumerate(items)]


def emit_patch_descriptor(patches: Any, out_path: PathLike) -> Dict[str, Any]:
    """Write a class-file patch descriptor for external bytecode tooling.

    ``patches`` is a list of ``{class, method, desc, op, lua_hook}`` objects
    (:data:`PATCH_OPS` = ``prepend`` | ``append`` | ``replace``); the schema is
    validated before anything is written.  The emitted JSON is deterministic
    (fixed key order, two-space indent, ``\\n`` newlines, ASCII escapes).

    The descriptor *describes* edits: this module never opens, parses or rewrites
    a class file, and applying the descriptor is the external tool's job.

    Returns ``{status, out_path, count, patches, schema, schema_version,
    sha256}`` (JSON-plain, Trap 14).
    """
    normalized = validate_patches(patches)

    try:
        path = Path(out_path)
    except TypeError as exc:
        raise LuaModError(f"out_path must be a path-like value, got {out_path!r}") from exc
    if path.is_dir():
        raise LuaModError(f"out_path must be a file path, not a directory: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema": DESCRIPTOR_SCHEMA,
        "schema_version": DESCRIPTOR_SCHEMA_VERSION,
        "tool": "toothpick.lua_mod.emit_patch_descriptor",
        "note": (
            "Descriptive only: these edits are applied by an external bytecode "
            "tool. toothpick.lua_mod never rewrites class files."
        ),
        "patch_count": len(normalized),
        "patches": normalized,
    }
    _write_text(path, json.dumps(payload, indent=2, ensure_ascii=True) + "\n")

    return {
        "status": "success",
        "out_path": str(path),
        "count": len(normalized),
        "patches": normalized,
        "schema": DESCRIPTOR_SCHEMA,
        "schema_version": DESCRIPTOR_SCHEMA_VERSION,
        "sha256": _canonical_sha256(normalized),
    }


def parse_patch_descriptor(path: PathLike) -> List[Dict[str, str]]:
    """Read and schema-validate a patch descriptor written by this module.

    Round-trip guarantee: ``parse_patch_descriptor(emit_patch_descriptor(p,
    out)["out_path"])`` equals :func:`validate_patches` ``(p)``, entry for entry
    and key for key.

    Refusals (all :class:`LuaModError`): a missing/unreadable path, malformed
    JSON, a root that is not an object with a ``patches`` list, a foreign
    ``schema`` marker, a ``patch_count`` that disagrees with the list length, and
    every per-entry violation :func:`validate_patches` refuses.
    """
    try:
        descriptor_path = Path(path)
    except TypeError as exc:
        raise LuaModError(f"path must be a path-like value, got {path!r}") from exc
    if descriptor_path.is_dir():
        raise LuaModError(f"patch descriptor path is a directory: {descriptor_path}")
    if not descriptor_path.is_file():
        raise LuaModError(f"patch descriptor not found: {descriptor_path}")

    try:
        raw_text = descriptor_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise LuaModError(f"cannot read patch descriptor {descriptor_path}: {exc}") from exc
    try:
        document = json.loads(raw_text)
    except ValueError as exc:
        raise LuaModError(
            f"patch descriptor {descriptor_path} is not valid JSON: {exc}"
        ) from exc

    if not isinstance(document, dict):
        raise LuaModError(
            f"patch descriptor {descriptor_path} root must be a JSON object with a "
            f"'patches' list, got {type(document).__name__}"
        )
    schema = document.get("schema")
    if schema is not None and schema != DESCRIPTOR_SCHEMA:
        raise LuaModError(
            f"patch descriptor {descriptor_path} declares schema {schema!r}; expected "
            f"{DESCRIPTOR_SCHEMA!r}"
        )
    if "patches" not in document:
        raise LuaModError(
            f"patch descriptor {descriptor_path} is missing the required 'patches' key"
        )

    declared_count = document.get("patch_count")
    patches = document["patches"]
    if declared_count is not None and not isinstance(patches, list):
        raise LuaModError(
            f"patch descriptor {descriptor_path} 'patches' must be a list, got "
            f"{type(patches).__name__}"
        )
    if isinstance(declared_count, bool) or (
        declared_count is not None and not isinstance(declared_count, int)
    ):
        raise LuaModError(
            f"patch descriptor {descriptor_path} 'patch_count' must be an integer, got "
            f"{declared_count!r}"
        )
    if isinstance(patches, list) and declared_count is not None:
        if declared_count != len(patches):
            raise LuaModError(
                f"patch descriptor {descriptor_path} declares patch_count="
                f"{declared_count} but carries {len(patches)} patch(es)"
            )

    return validate_patches(patches)
