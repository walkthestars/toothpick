"""
Toothpick Mod Pipelines: Unity IL2CPP (unit W08), Unreal Engine & Godot (unit W13).

Implements the W08 and W13 mod generation contracts per UNIT-W08-UNITY-MOD-PLAN.md,
UNIT-W13-UE-GODOT-PLAN.md, and design-document.md §1, §6, §9, §11, §12, §16 #1:
- Unity IL2CPP: ingestion of binary global-metadata.dat headers and structured JSON
  dumps, scaffolding of compile-ready BepInEx 5/6 C# plugins with Harmony patches,
  and optional `dotnet build` compilation into a .dll assembly.
- Unreal Engine 4/5: UE4SS Lua mod scaffolding (`scripts/main.lua`, `enabled.txt`)
  with pre/post/replace UFunction hooks, and native C++ DLL scaffolding
  (`dllmain.cpp`, `UnrealSDK.h`, `CMakeLists.txt`) for MinHook/VMT injection.
- Godot Engine 3/4: `.pck` container parsing/extraction/synthesis (GDPC format) plus
  GDExtension and GDScript override mod scaffolding (`.gdextension`, `override.cfg`).
- Epistemic Grounding Gate (§16 #1): refuses to emit a grounded mod artifact unless
  the target offset has a confirmed live receipt (verifier_status="GROUNDED") in
  toothpick.db, unless explicitly overridden with --allow-unverified.
- Boot verification token logging and per-engine mod build receipt recording
  (phase="mod_build" | "mod_build_unreal" | "mod_build_godot").
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import struct
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4

from toothpick.query import parse_address
from toothpick.storage import Database, FunctionRecord, TypeRecord

__all__ = [
    "ModBuildError",
    "UngroundedTargetError",
    "MetadataParseError",
    "CompilationError",
    "PckParseError",
    "ModConfig",
    "ModArtifact",
    "Il2CppMetadataReader",
    "ModGenerator",
    "ModCompiler",
    "UnrealModConfig",
    "UnrealModGenerator",
    "GodotModConfig",
    "GodotModGenerator",
    "PckEntry",
    "GodotPckReader",
    "ModBuilder",
]

PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ModBuildError(Exception):
    """Base exception for mod builder failures."""


class UngroundedTargetError(ModBuildError):
    """Raised when target hook offset lacks a confirmed dynamic receipt (Design §16 #1)."""


class MetadataParseError(ModBuildError):
    """Raised when IL2CPP metadata parsing fails."""


class CompilationError(ModBuildError):
    """Raised when dotnet compilation of mod artifact fails."""


class PckParseError(ModBuildError):
    """Raised when a Godot ``.pck`` container cannot be parsed or synthesized."""


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


def _sanitize_identifier(name: str) -> str:
    """Sanitize string into a valid C# identifier."""
    cleaned = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"M_{cleaned}"
    return cleaned


def _sanitize_symbol(name: str, default: str = "Target") -> str:
    """Sanitize string into a valid symbol identifier with a fallback default."""
    cleaned = re.sub(r"[^a-zA-Z0-9_]", "_", name or "").strip("_")
    if not cleaned:
        return default
    if cleaned[0].isdigit():
        cleaned = f"M_{cleaned}"
    return cleaned


def normalize_pck_path(path_in_pck: str) -> str:
    """Strip Godot virtual prefixes (``res://``, ``user://``) and leading separators.

    Uses explicit *prefix* removal rather than :meth:`str.lstrip`, because
    ``lstrip`` treats its argument as a character set and would corrupt names
    such as ``res://scene.tscn`` -> ``cene.tscn``.
    """
    raw = (path_in_pck or "").strip()
    for prefix in ("res://", "user://"):
        if raw.lower().startswith(prefix):
            raw = raw[len(prefix):]
            break
    return raw.lstrip("/\\")


def safe_relative_pck_path(path_in_pck: str) -> Optional[Path]:
    """Map a PCK internal path to a relative on-disk path, or ``None`` if unsafe.

    Refuses absolute paths, drive-qualified paths, and any ``..`` traversal
    segment so extraction can never escape the destination directory.
    """
    raw = normalize_pck_path(path_in_pck)
    if not raw:
        return None
    candidate = Path(raw)
    if candidate.is_absolute() or candidate.drive:
        return None
    parts = [part for part in candidate.parts if part not in ("", ".")]
    if not parts or any(part == ".." for part in parts):
        return None
    return Path(*parts)


@dataclass
class ModConfig:
    """Configuration for generating a BepInEx mod plugin."""

    plugin_name: str = "ToothpickMod"
    plugin_id: str = ""
    version: str = "1.0.0"
    target_addr: Optional[int] = None
    target_class: str = "TargetClass"
    target_method: str = "TargetMethod"
    hook_type: str = "postfix"  # "prefix" | "postfix" | "replace"
    boot_token: str = field(default_factory=lambda: uuid4().hex)
    mod_logic: Optional[str] = None
    binary_sha256: str = ""
    author: str = "Toothpick"

    def __post_init__(self) -> None:
        self.plugin_name = _sanitize_identifier(self.plugin_name)
        if not self.plugin_id:
            self.plugin_id = f"com.toothpick.mod.{self.plugin_name.lower()}"
        if not self.boot_token:
            self.boot_token = uuid4().hex
        self.hook_type = self.hook_type.lower()
        if self.hook_type not in ("prefix", "postfix", "replace"):
            raise ModBuildError(
                f"Invalid hook_type: {self.hook_type!r}; expected 'prefix', 'postfix', or 'replace'"
            )


@dataclass
class ModArtifact:
    """Result of mod project generation and optional compilation.

    Polymorphic across engines: ``engine`` records which backend emitted the
    artifact, ``files`` enumerates every generated file, and ``manifest_file``
    is the engine-specific build manifest (``.csproj`` for BepInEx, ``enabled.txt``
    for UE4SS, ``CMakeLists.txt``/``override.cfg`` for Godot).
    """

    project_dir: Path
    source_file: Path
    csproj_file: Optional[Path] = None
    dll_path: Optional[Path] = None
    is_grounded: bool = False
    receipt_id: Optional[str] = None
    boot_token: str = ""
    config: Optional[Any] = None
    engine: str = "unity"
    mod_type: str = "bepinex_csharp"
    files: List[Path] = field(default_factory=list)
    manifest_file: Optional[Path] = None
    target_addr: Optional[int] = None
    target_symbol: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert into a JSON-serializable dictionary with a stable key order."""
        cfg = self.config
        return {
            "status": "success",
            "engine": self.engine,
            "mod_type": self.mod_type,
            "plugin_name": getattr(cfg, "plugin_name", "") if cfg is not None else "",
            "plugin_id": getattr(cfg, "plugin_id", "") if cfg is not None else "",
            "version": getattr(cfg, "version", "") if cfg is not None else "",
            "target_addr": self.target_addr,
            "target_addr_hex": (
                f"0x{self.target_addr:x}" if self.target_addr is not None else None
            ),
            "target_symbol": self.target_symbol,
            "target_class": getattr(cfg, "target_class", "") if cfg is not None else "",
            "target_method": getattr(cfg, "target_method", "") if cfg is not None else "",
            "target_function": (
                getattr(cfg, "target_function", "") if cfg is not None else ""
            ),
            "hook_type": getattr(cfg, "hook_type", "") if cfg is not None else "",
            "is_grounded": self.is_grounded,
            "receipt_id": self.receipt_id,
            "boot_token": self.boot_token,
            "project_dir": str(self.project_dir),
            "source_file": str(self.source_file),
            "csproj_file": str(self.csproj_file) if self.csproj_file else None,
            "manifest_file": str(self.manifest_file) if self.manifest_file else None,
            "files": [str(path) for path in self.files],
            "dll_path": str(self.dll_path) if self.dll_path else None,
        }


# ---------------------------------------------------------------------------
# IL2CPP Metadata Reader
# ---------------------------------------------------------------------------


class Il2CppMetadataReader:
    """Parses Unity IL2CPP type metadata and ingests it into toothpick.db."""

    IL2CPP_METADATA_MAGIC = 0xFAB11BAF

    @classmethod
    def read_binary(
        cls,
        data_or_path: Union[bytes, PathLike],
        binary_sha256: str = "",
    ) -> Tuple[List[TypeRecord], List[FunctionRecord]]:
        """Parse global-metadata.dat binary header and structures."""
        if isinstance(data_or_path, (str, Path)):
            path = Path(data_or_path)
            if not path.is_file():
                raise MetadataParseError(f"File not found: {path}")
            data = path.read_bytes()
        else:
            data = bytes(data_or_path)

        if len(data) < 8:
            raise MetadataParseError(
                f"Binary global-metadata.dat too short ({len(data)} bytes, need >= 8)"
            )

        magic = struct.unpack("<I", data[:4])[0]
        if magic != cls.IL2CPP_METADATA_MAGIC:
            raise MetadataParseError(
                f"Invalid global-metadata.dat magic: 0x{magic:08X}; expected 0x{cls.IL2CPP_METADATA_MAGIC:08X}"
            )

        version = struct.unpack("<i", data[4:8])[0]
        layout = {
            "magic": f"0x{cls.IL2CPP_METADATA_MAGIC:08X}",
            "version": version,
            "total_bytes": len(data),
        }

        # Header fields across IL2CPP versions (v24-v31 commonly hold 32-bit offset/size pairs)
        if len(data) >= 32:
            try:
                string_literal_offset, string_literal_count = struct.unpack("<ii", data[8:16])
                string_literal_data_offset, string_literal_data_count = struct.unpack("<ii", data[16:24])
                string_offset, string_count = struct.unpack("<ii", data[24:32])
                layout["string_literal_offset"] = string_literal_offset
                layout["string_literal_count"] = string_literal_count
                layout["string_offset"] = string_offset
                layout["string_count"] = string_count
            except struct.error:
                pass

        types: List[TypeRecord] = [
            TypeRecord(
                type_id="__il2cpp_global_metadata__",
                name="GlobalMetadata",
                kind="metadata",
                size_bytes=len(data),
                layout=layout,
                lattice_rank=1,
                binary_sha256=binary_sha256,
            )
        ]
        return types, []

    @classmethod
    def read_json(
        cls,
        data_or_path: Union[str, PathLike, dict, list],
        binary_sha256: str = "",
    ) -> Tuple[List[TypeRecord], List[FunctionRecord]]:
        """Parse structured IL2CPP JSON dumps into TypeRecords and FunctionRecords."""
        raw_obj: Any
        if isinstance(data_or_path, (str, Path)):
            path = Path(data_or_path)
            if path.is_file():
                try:
                    raw_obj = json.loads(path.read_text(encoding="utf-8"))
                except Exception as exc:
                    raise MetadataParseError(f"Failed to parse JSON file {path}: {exc}") from exc
            else:
                # Could be a raw JSON string
                try:
                    raw_obj = json.loads(str(data_or_path))
                except Exception as exc:
                    raise MetadataParseError(f"Failed to parse JSON string: {exc}") from exc
        else:
            raw_obj = data_or_path

        type_records: List[TypeRecord] = []
        func_records: List[FunctionRecord] = []

        # Case 1: Il2CppDumper script.json output {"ScriptMethod": [...], ...}
        if isinstance(raw_obj, dict) and "ScriptMethod" in raw_obj:
            methods = raw_obj["ScriptMethod"]
            for m in methods:
                addr_raw = m.get("Address")
                name = m.get("Name", "")
                sig = m.get("Signature", "")
                if addr_raw is not None:
                    try:
                        addr = int(addr_raw) if not isinstance(addr_raw, str) else parse_address(addr_raw)
                    except Exception as exc:
                        raise MetadataParseError(
                            f"Invalid method address {addr_raw!r} for method {name!r}: {exc}"
                        ) from exc

                    # Class and method name parsing (e.g. "PlayerController$$Update")
                    cls_name = "UnknownClass"
                    meth_name = name
                    if "$$" in name:
                        cls_name, meth_name = name.split("$$", 1)
                    elif "." in name:
                        cls_name, meth_name = name.rsplit(".", 1)

                    func_records.append(
                        FunctionRecord(
                            binary_sha256=binary_sha256,
                            func_addr=addr,
                            name=name,
                            prototype=sig,
                        )
                    )
                    type_records.append(
                        TypeRecord(
                            type_id=f"il2cpp_{cls_name}",
                            name=cls_name,
                            kind="class",
                            layout={
                                "methods": [
                                    {
                                        "name": meth_name,
                                        "addr": addr,
                                        "signature": sig,
                                    }
                                ]
                            },
                            binary_sha256=binary_sha256,
                        )
                    )
            return cls._consolidate_types(type_records), func_records

        # Case 2: Structured dictionary {"types": [...]} or list of types [{...}, ...]
        types_list: List[Dict[str, Any]] = []
        if isinstance(raw_obj, dict):
            if "types" in raw_obj and isinstance(raw_obj["types"], list):
                types_list = raw_obj["types"]
            elif "classes" in raw_obj and isinstance(raw_obj["classes"], list):
                types_list = raw_obj["classes"]
            else:
                # Key-value map of class_name -> type_spec
                for k, v in raw_obj.items():
                    if isinstance(v, dict):
                        entry = dict(v)
                        entry.setdefault("name", k)
                        types_list.append(entry)
        elif isinstance(raw_obj, list):
            types_list = [item for item in raw_obj if isinstance(item, dict)]
        else:
            raise MetadataParseError(f"Unexpected JSON metadata structure: {type(raw_obj)}")

        for t in types_list:
            name = t.get("name", "UnnamedType")
            type_id = t.get("type_id") or f"type_{_sanitize_identifier(name)}"
            kind = t.get("kind", "class")
            size = int(t.get("size_bytes") or t.get("size") or 0)
            fields = t.get("fields", [])
            methods = t.get("methods", [])

            # Extract any function records from methods with addresses
            for m in methods:
                if isinstance(m, dict):
                    m_addr = m.get("addr") or m.get("address")
                    if m_addr is not None:
                        try:
                            addr_int = int(m_addr) if not isinstance(m_addr, str) else parse_address(m_addr)
                        except Exception as exc:
                            raise MetadataParseError(
                                f"Invalid method address {m_addr!r} in {name}.{m.get('name', 'unknown')}: {exc}"
                            ) from exc

                        m_name = m.get("name", "Method")
                        func_name = f"{name}.{m_name}"
                        func_records.append(
                            FunctionRecord(
                                binary_sha256=binary_sha256,
                                func_addr=addr_int,
                                name=func_name,
                                prototype=m.get("signature") or m.get("return_type", ""),
                            )
                        )

            type_records.append(
                TypeRecord(
                    type_id=type_id,
                    name=name,
                    kind=kind,
                    size_bytes=size,
                    layout={"fields": fields, "methods": methods},
                    lattice_rank=t.get("lattice_rank", 1),
                    binary_sha256=binary_sha256,
                )
            )

        return cls._consolidate_types(type_records), func_records

    @staticmethod
    def _consolidate_types(types: List[TypeRecord]) -> List[TypeRecord]:
        """Merge method/field lists for types with the same type_id and binary_sha256."""
        merged: Dict[Tuple[str, str], TypeRecord] = {}
        for t in types:
            key = (t.binary_sha256, t.type_id)
            if key not in merged:
                merged[key] = t
            else:
                existing = merged[key]
                cur_layout = existing.layout or {}
                new_layout = t.layout or {}
                combined_methods = list(cur_layout.get("methods", [])) + list(new_layout.get("methods", []))
                combined_fields = list(cur_layout.get("fields", [])) + list(new_layout.get("fields", []))
                existing.layout = {"methods": combined_methods, "fields": combined_fields}
        return list(merged.values())

    @classmethod
    def ingest(
        cls,
        source: Union[bytes, str, PathLike, dict, list],
        db: Database,
        binary_sha256: str = "",
    ) -> Dict[str, int]:
        """Ingest metadata into SQLite database types and funcs tables."""
        is_binary = False
        if isinstance(source, bytes):
            is_binary = True
        elif isinstance(source, (str, Path)):
            path = Path(source)
            if path.is_file():
                if path.suffix.lower() in (".dat", ".bin") or "global-metadata" in path.name.lower():
                    is_binary = True
                else:
                    try:
                        with path.open("rb") as f:
                            head = f.read(4)
                            if len(head) == 4 and struct.unpack("<I", head)[0] == cls.IL2CPP_METADATA_MAGIC:
                                is_binary = True
                    except Exception:
                        pass

        sha = binary_sha256
        if not sha:
            rows = db._conn.execute(
                "SELECT DISTINCT binary_sha256 FROM funcs WHERE binary_sha256 != '' "
                "UNION SELECT DISTINCT binary_sha256 FROM types WHERE binary_sha256 != ''"
            ).fetchall()
            shas = [r[0] for r in rows if r[0]]
            if len(shas) == 1:
                sha = shas[0]

        if is_binary:
            types, funcs = cls.read_binary(source, binary_sha256=sha)
        else:
            types, funcs = cls.read_json(source, binary_sha256=sha)

        types_count = 0
        for t in types:
            db.upsert_type(t)
            types_count += 1

        funcs_count = 0
        for f in funcs:
            if not f.binary_sha256 and sha:
                f.binary_sha256 = sha
            if not f.binary_sha256:
                raise MetadataParseError(
                    "Cannot ingest recovered function without binary_sha256 attribution; specify --sha"
                )
            db.upsert_function(f, write_receipt=False)
            funcs_count += 1

        return {
            "types_ingested": types_count,
            "funcs_ingested": funcs_count,
        }


# ---------------------------------------------------------------------------
# Mod Generator (BepInEx 5/6 + Harmony)
# ---------------------------------------------------------------------------


class ModGenerator:
    """Emits clean, compile-ready BepInEx C# plugin projects."""

    CS_PROJ_TEMPLATE = """<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <TargetFramework>netstandard2.1</TargetFramework>
    <Nullable>enable</Nullable>
    <LangVersion>latest</LangVersion>
    <AssemblyName>{assembly_name}</AssemblyName>
    <RootNamespace>{root_namespace}</RootNamespace>
    <Version>{version}</Version>
  </PropertyGroup>
</Project>
"""

    CS_PLUGIN_TEMPLATE = """// Generated by Toothpick Reverse Engineering Engine
// Target: {target_class}.{target_method} (Address: {target_addr_hex})
// Epistemic Grounding Status: {grounded_status}

using System;
using System.Reflection;
using BepInEx;
using HarmonyLib;

#if !BEPINEX_REAL
// Self-contained offline stubs for BepInEx and HarmonyLib to allow immediate compilation
// via dotnet build without external NuGet or network dependencies.
namespace BepInEx
{{
    [AttributeUsage(AttributeTargets.Class, Inherited = false)]
    public class BepInPlugin : Attribute
    {{
        public BepInPlugin(string guid, string name, string version) {{}}
    }}

    public class BaseUnityPlugin
    {{
        public virtual void Awake() {{}}
        public virtual void OnDestroy() {{}}
    }}
}}

namespace HarmonyLib
{{
    [AttributeUsage(AttributeTargets.Class | AttributeTargets.Method, Inherited = false)]
    public class HarmonyPatch : Attribute
    {{
        public HarmonyPatch() {{}}
        public HarmonyPatch(string methodName) {{}}
        public HarmonyPatch(Type declaringType, string methodName) {{}}
    }}

    [AttributeUsage(AttributeTargets.Method, Inherited = false)]
    public class HarmonyTargetMethod : Attribute {{}}

    [AttributeUsage(AttributeTargets.Method, Inherited = false)]
    public class HarmonyPrefix : Attribute {{}}

    [AttributeUsage(AttributeTargets.Method, Inherited = false)]
    public class HarmonyPostfix : Attribute {{}}

    public class Harmony
    {{
        public Harmony(string id) {{}}
        public void PatchAll(Type type) {{}}
        public void UnpatchSelf() {{}}
    }}

    public static class AccessTools
    {{
        public static MethodBase? Method(string typeAndMethodName) => null;
        public static MethodBase? Method(Type type, string name) => null;
        public static Type? TypeByName(string name) => null;
    }}
}}
#endif

namespace Toothpick.Mods
{{
    public static class PluginInfo
    {{
        public const string PLUGIN_GUID = "{plugin_id}";
        public const string PLUGIN_NAME = "{plugin_name}";
        public const string PLUGIN_VERSION = "{version}";
        public const string BOOT_TOKEN = "{boot_token}";
    }}

    [BepInPlugin(PluginInfo.PLUGIN_GUID, PluginInfo.PLUGIN_NAME, PluginInfo.PLUGIN_VERSION)]
    public class {safe_plugin_name}Plugin : BaseUnityPlugin
    {{
        public const string BOOT_TOKEN = PluginInfo.BOOT_TOKEN;
        public const long TARGET_ADDR = {target_addr_dec}L;

        private Harmony? _harmony;

        public override void Awake()
        {{
            base.Awake();
            // Boot verification token required by Toothpick pipeline (§16 #1)
            Console.WriteLine($"[Toothpick] Boot verification token: {boot_token}");
            Console.WriteLine($"[Toothpick] Loaded {{PluginInfo.PLUGIN_NAME}} v{{PluginInfo.PLUGIN_VERSION}}");
            
            try
            {{
                _harmony = new Harmony(PluginInfo.PLUGIN_GUID);
                _harmony.PatchAll(typeof(Patch_{safe_method_name}));
                Console.WriteLine($"[Toothpick] Harmony patches applied for {{PluginInfo.PLUGIN_NAME}}");
            }}
            catch (Exception ex)
            {{
                Console.WriteLine($"[Toothpick] Error applying Harmony patches: {{ex.Message}}");
            }}
        }}

        public override void OnDestroy()
        {{
            base.OnDestroy();
            _harmony?.UnpatchSelf();
        }}
    }}

    [HarmonyPatch]
    public static class Patch_{safe_method_name}
    {{
        [HarmonyTargetMethod]
        public static MethodBase? CalculateTargetMethod()
        {{
            return AccessTools.Method("{target_class}:{target_method}");
        }}

{hook_implementation}
    }}
}}
"""

    @classmethod
    def generate(cls, config: ModConfig, out_dir: PathLike, is_grounded: bool = False) -> Tuple[Path, Path]:
        """Generate .csproj and Plugin.cs files in out_dir."""
        project_dir = Path(out_dir)
        project_dir.mkdir(parents=True, exist_ok=True)

        csproj_path = project_dir / f"{config.plugin_name}.csproj"
        source_path = project_dir / "Plugin.cs"

        csproj_content = cls.CS_PROJ_TEMPLATE.format(
            assembly_name=config.plugin_name,
            root_namespace=config.plugin_name,
            version=config.version,
        )
        csproj_path.write_text(csproj_content, encoding="utf-8")

        safe_method = _sanitize_identifier(config.target_method)
        target_addr_hex = f"0x{config.target_addr:x}" if config.target_addr is not None else "0x0"
        target_addr_dec = config.target_addr if config.target_addr is not None else 0

        # Construct hook implementation based on hook_type
        logic_body = config.mod_logic or f'Console.WriteLine("[Toothpick] Hook executed on {config.target_class}.{config.target_method}");'
        
        if config.hook_type == "prefix":
            hook_impl = f"""        [HarmonyPrefix]
        public static bool Prefix()
        {{
            {logic_body}
            return true; // continue original method
        }}"""
        elif config.hook_type == "replace":
            hook_impl = f"""        [HarmonyPrefix]
        public static bool Prefix()
        {{
            {logic_body}
            return false; // skip original method
        }}"""
        else:  # postfix
            hook_impl = f"""        [HarmonyPostfix]
        public static void Postfix()
        {{
            {logic_body}
        }}"""

        source_content = cls.CS_PLUGIN_TEMPLATE.format(
            target_class=config.target_class,
            target_method=config.target_method,
            target_addr_hex=target_addr_hex,
            target_addr_dec=target_addr_dec,
            grounded_status="GROUNDED" if is_grounded else "UNVERIFIED",
            plugin_id=config.plugin_id,
            plugin_name=config.plugin_name,
            safe_plugin_name=config.plugin_name,
            safe_method_name=safe_method,
            version=config.version,
            boot_token=config.boot_token,
            hook_implementation=hook_impl,
        )
        source_path.write_text(source_content, encoding="utf-8")

        return csproj_path, source_path


# ---------------------------------------------------------------------------
# Mod Compiler
# ---------------------------------------------------------------------------


class ModCompiler:
    """Invokes `dotnet build` to compile C# BepInEx plugins into binary .dll assemblies."""

    @classmethod
    def compile(
        cls,
        project_dir: PathLike,
        output_dir: Optional[PathLike] = None,
        configuration: str = "Release",
    ) -> Path:
        """Compile C# project at project_dir and return Path to emitted .dll."""
        pdir = Path(project_dir)
        if not pdir.is_dir():
            raise CompilationError(f"Project directory does not exist: {pdir}")

        dotnet_bin = shutil.which("dotnet")
        if dotnet_bin is None:
            raise CompilationError(
                "The 'dotnet' SDK is not available in PATH. Please install .NET SDK "
                "or run without --compile to emit source scaffolding only."
            )

        bin_dir = Path(output_dir) if output_dir is not None else pdir / "bin" / configuration
        bin_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            dotnet_bin,
            "build",
            str(pdir),
            "-c",
            configuration,
            "-o",
            str(bin_dir),
            "--nologo",
        ]

        try:
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=60.0,
            )
        except Exception as exc:
            raise CompilationError(f"Failed to execute dotnet build: {exc}") from exc

        if res.returncode != 0:
            err_msg = res.stderr.strip() or res.stdout.strip()
            raise CompilationError(f"dotnet build failed with exit code {res.returncode}:\n{err_msg}")

        # Look for produced .dll assembly
        dll_candidates = list(bin_dir.glob("*.dll"))
        if not dll_candidates:
            # Fallback search anywhere in bin_dir subtree
            dll_candidates = list(pdir.glob("bin/**/*.dll"))

        if not dll_candidates:
            raise CompilationError(f"Compilation succeeded but no .dll assembly was found in {bin_dir}")

        # Return the newest or matching dll
        return dll_candidates[0]


# ---------------------------------------------------------------------------
# Unreal Engine Mod Pipeline (W13)
# ---------------------------------------------------------------------------


@dataclass
class UnrealModConfig:
    """Configuration for generating an Unreal Engine UE4SS Lua or native C++ mod."""

    plugin_name: str = "ToothpickUnrealMod"
    target_addr: Optional[int] = None
    target_class: str = "TargetClass"
    target_function: str = "TargetFunction"
    hook_type: str = "post"  # "pre" | "post" | "replace"
    mod_type: str = "ue4ss_lua"  # "ue4ss_lua" | "native_cpp"
    boot_token: str = field(default_factory=lambda: uuid4().hex)
    mod_logic: Optional[str] = None
    binary_sha256: str = ""
    author: str = "Toothpick"
    version: str = "1.0.0"

    def __post_init__(self) -> None:
        self.plugin_name = _sanitize_identifier(self.plugin_name)
        if not self.boot_token:
            self.boot_token = uuid4().hex
        self.hook_type = self.hook_type.lower()
        if self.hook_type == "prefix":
            self.hook_type = "pre"
        elif self.hook_type == "postfix":
            self.hook_type = "post"
        if self.hook_type not in ("pre", "post", "replace"):
            raise ModBuildError(
                f"Invalid hook_type: {self.hook_type!r}; expected 'pre', 'post', or 'replace'"
            )
        self.mod_type = self.mod_type.lower()
        if self.mod_type not in ("ue4ss_lua", "native_cpp"):
            raise ModBuildError(
                f"Invalid mod_type: {self.mod_type!r}; expected 'ue4ss_lua' or 'native_cpp'"
            )

    @property
    def ue4ss_target_path(self) -> str:
        return f"{self.target_class}:{self.target_function}"


class UnrealModGenerator:
    """Generates UE4SS Lua or native C++ mod projects for Unreal Engine 4/5."""

    UE4SS_LUA_TEMPLATE = """-- ==============================================================================
-- Toothpick Unreal Engine Mod: {plugin_name} v{version}
-- Target: {target_class}:{target_function} ({target_addr_hex})
-- Grounding Status: {grounded_status} (Design §16 #1)
-- Boot Token: {boot_token}
-- ==============================================================================

local BOOT_TOKEN = "{boot_token}"
print("[Toothpick] Boot verification token: " .. BOOT_TOKEN)
print("[Toothpick] Initializing UE4SS mod: {plugin_name}")

local target_symbol = "{target_class}:{target_function}"
{hook_code}
"""

    CPP_CMAKElISTS_TEMPLATE = """cmake_minimum_required(VERSION 3.20)
project({plugin_name} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library({plugin_name} SHARED
    src/dllmain.cpp
)

target_include_directories({plugin_name} PRIVATE
    include
)
"""

    CPP_DLLMAIN_TEMPLATE = """// ==============================================================================
// Toothpick Unreal Engine Native C++ Mod: {plugin_name} v{version}
// Target: {target_class}:{target_function} ({target_addr_hex})
// Grounding Status: {grounded_status} (Design §16 #1)
// Boot Token: {boot_token}
// ==============================================================================

#include <windows.h>
#include <iostream>
#include "../include/UnrealSDK.h"

namespace {{
const char* BOOT_TOKEN = "{boot_token}";
const char* PLUGIN_NAME = "{plugin_name}";
const uintptr_t TARGET_ADDR = {target_addr_hex};
}}

void OnAttach() {{
    std::cout << "[Toothpick] Boot verification token: " << BOOT_TOKEN << std::endl;
    std::cout << "[Toothpick] Initializing Unreal native C++ mod: " << PLUGIN_NAME << std::endl;
{hook_code}
}}

BOOL APIENTRY DllMain(HMODULE hModule, DWORD ul_reason_for_call, LPVOID lpReserved) {{
    switch (ul_reason_for_call) {{
    case DLL_PROCESS_ATTACH:
        DisableThreadLibraryCalls(hModule);
        OnAttach();
        break;
    case DLL_PROCESS_DETACH:
        break;
    }}
    return TRUE;
}}
"""

    CPP_HEADER_TEMPLATE = """#pragma once
#include <cstdint>
#include <windows.h>

// Minimal Unreal SDK stub for standalone compilation
namespace UnrealSDK {
    struct UObject {
        void* VTable;
        int32_t ObjectFlags;
        int32_t InternalIndex;
        void* ClassPrivate;
        void* NamePrivate;
        UObject* OuterPrivate;
    };

    struct UFunction : UObject {
        uint32_t FunctionFlags;
        uint16_t NumParms;
        uint16_t ParmsSize;
        uint16_t ReturnValueOffset;
        uint16_t RPCId;
        uint16_t RPCResponseId;
        void* FirstPropertyToInit;
        UFunction* EventGraphFunction;
        int32_t EventGraphCallOffset;
        void* Func;
    };
}
"""

    @classmethod
    def generate(
        cls, config: UnrealModConfig, out_dir: PathLike, is_grounded: bool = False
    ) -> Tuple[Path, Path]:
        """Generate Unreal Engine mod project in out_dir."""
        project_dir = Path(out_dir)
        project_dir.mkdir(parents=True, exist_ok=True)
        target_addr_hex = f"0x{config.target_addr:x}" if config.target_addr is not None else "0x0"
        grounded_status = "GROUNDED" if is_grounded else "UNVERIFIED"

        if config.mod_type == "ue4ss_lua":
            scripts_dir = project_dir / "scripts"
            scripts_dir.mkdir(parents=True, exist_ok=True)
            manifest_file = project_dir / "enabled.txt"
            manifest_file.write_text("1\n", encoding="utf-8")

            source_file = scripts_dir / "main.lua"
            logic_body = (
                config.mod_logic
                or f'print("[Toothpick] Hook executed on {config.target_class}:{config.target_function}")'
            )

            if config.hook_type == "pre":
                hook_code = f"""RegisterHook(target_symbol, function(self, ...)
    print("[Toothpick] Pre-hook fired on " .. target_symbol)
    {logic_body}
end)"""
            elif config.hook_type == "replace":
                hook_code = f"""RegisterHook(target_symbol, function(self, ...)
    print("[Toothpick] Replace-hook fired on " .. target_symbol)
    {logic_body}
    return true
end)"""
            else:  # post
                hook_code = f"""RegisterHook(target_symbol, function(self, ...)
    print("[Toothpick] Post-hook fired on " .. target_symbol)
    {logic_body}
end)"""

            source_content = cls.UE4SS_LUA_TEMPLATE.format(
                plugin_name=config.plugin_name,
                version=config.version,
                target_class=config.target_class,
                target_function=config.target_function,
                target_addr_hex=target_addr_hex,
                grounded_status=grounded_status,
                boot_token=config.boot_token,
                hook_code=hook_code,
            )
            source_file.write_text(source_content, encoding="utf-8")
            return manifest_file, source_file

        else:  # native_cpp
            src_dir = project_dir / "src"
            include_dir = project_dir / "include"
            src_dir.mkdir(parents=True, exist_ok=True)
            include_dir.mkdir(parents=True, exist_ok=True)

            manifest_file = project_dir / "CMakeLists.txt"
            manifest_file.write_text(
                cls.CPP_CMAKElISTS_TEMPLATE.format(plugin_name=config.plugin_name),
                encoding="utf-8",
            )

            header_file = include_dir / "UnrealSDK.h"
            header_file.write_text(cls.CPP_HEADER_TEMPLATE, encoding="utf-8")

            source_file = src_dir / "dllmain.cpp"
            logic_body = (
                config.mod_logic
                or f'std::cout << "[Toothpick] Hook executed on {config.target_class}:{config.target_function}" << std::endl;'
            )
            hook_code = f"    // Hook logic for {config.target_class}:{config.target_function}\n    {logic_body}"

            source_content = cls.CPP_DLLMAIN_TEMPLATE.format(
                plugin_name=config.plugin_name,
                version=config.version,
                target_class=config.target_class,
                target_function=config.target_function,
                target_addr_hex=target_addr_hex,
                grounded_status=grounded_status,
                boot_token=config.boot_token,
                hook_code=hook_code,
            )
            source_file.write_text(source_content, encoding="utf-8")
            return manifest_file, source_file

    @classmethod
    def compile_project(cls, project_dir: PathLike) -> Optional[Path]:
        pdir = Path(project_dir)
        cmake_bin = shutil.which("cmake")
        if cmake_bin is None:
            return None
        cmakelists = pdir / "CMakeLists.txt"
        if not cmakelists.exists():
            return None
        build_dir = pdir / "build"
        build_dir.mkdir(parents=True, exist_ok=True)
        try:
            res = subprocess.run(
                [cmake_bin, "-B", str(build_dir), "-S", str(pdir)],
                capture_output=True,
                check=False,
                timeout=30.0,
            )
            if res.returncode == 0:
                subprocess.run(
                    [cmake_bin, "--build", str(build_dir), "--config", "Release"],
                    capture_output=True,
                    check=False,
                    timeout=60.0,
                )
                dlls = list(build_dir.glob("**/*.dll"))
                if dlls:
                    return dlls[0]
        except Exception:
            pass
        return None


# ---------------------------------------------------------------------------
# Godot Engine Mod Pipeline (W13)
# ---------------------------------------------------------------------------


@dataclass
class GodotModConfig:
    """Configuration for generating a Godot Engine GDExtension or GDScript mod."""

    plugin_name: str = "ToothpickGodotMod"
    godot_version: int = 4  # 3 or 4
    mod_type: str = "gdextension"  # "gdextension" | "gdscript"
    target_node: str = "Root"
    target_method: str = "_process"
    target_addr: Optional[int] = None
    boot_token: str = field(default_factory=lambda: uuid4().hex)
    mod_logic: Optional[str] = None
    binary_sha256: str = ""
    author: str = "Toothpick"
    version: str = "1.0.0"

    def __post_init__(self) -> None:
        self.plugin_name = _sanitize_identifier(self.plugin_name)
        if not self.boot_token:
            self.boot_token = uuid4().hex
        self.mod_type = self.mod_type.lower()
        if self.mod_type not in ("gdextension", "gdscript"):
            raise ModBuildError(
                f"Invalid mod_type: {self.mod_type!r}; expected 'gdextension' or 'gdscript'"
            )
        if self.godot_version not in (3, 4):
            raise ModBuildError(
                f"Unsupported godot_version: {self.godot_version}; expected 3 or 4"
            )


class GodotModGenerator:
    """Generates Godot 4 GDExtension C++ or GDScript override mod projects."""

    GDEXTENSION_TEMPLATE = """[configuration]
entry_symbol = "{entry_symbol}"
compatibility_minimum = "4.1"

[libraries]
windows.debug.x86_64 = "bin/{plugin_name}.dll"
windows.release.x86_64 = "bin/{plugin_name}.dll"
linux.debug.x86_64 = "bin/lib{plugin_name}.so"
linux.release.x86_64 = "bin/lib{plugin_name}.so"
"""

    GDSCRIPT_OVERRIDE_CFG = """[application]
config/name="{plugin_name}"
autoload/ToothpickMod="*res://mods/{plugin_name}/mod_entry.gd"
"""

    GDSCRIPT_ENTRY_TEMPLATE = """# ==============================================================================
# Toothpick Godot GDScript Mod: {plugin_name} v{version}
# Target: {target_node}.{target_method} ({target_addr_hex})
# Grounding Status: {grounded_status} (Design §16 #1)
# Boot Token: {boot_token}
# ==============================================================================

extends Node

const BOOT_TOKEN = "{boot_token}"

func _ready():
    print("[Toothpick] Boot verification token: " + BOOT_TOKEN)
    print("[Toothpick] Loaded Godot mod: {plugin_name}")

func {target_method}(delta = 0.0):
    {logic_body}
"""

    GDEXTENSION_CPP_TEMPLATE = """// ==============================================================================
// Toothpick Godot GDExtension C++ Mod: {plugin_name} v{version}
// Target: {target_node}.{target_method} ({target_addr_hex})
// Grounding Status: {grounded_status} (Design §16 #1)
// Boot Token: {boot_token}
// ==============================================================================

#include <iostream>
#include <string>

namespace {{
const char* BOOT_TOKEN = "{boot_token}";
const char* PLUGIN_NAME = "{plugin_name}";
}}

extern "C" {{
void {entry_symbol}() {{
    std::cout << "[Toothpick] Boot verification token: " << BOOT_TOKEN << std::endl;
    std::cout << "[Toothpick] Initializing GDExtension mod: " << PLUGIN_NAME << std::endl;
}}
}}
"""

    GDEXTENSION_CMAKELISTS = """cmake_minimum_required(VERSION 3.20)
project({plugin_name} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library({plugin_name} SHARED
    src/{plugin_name}.cpp
)
"""

    @classmethod
    def generate(
        cls, config: GodotModConfig, out_dir: PathLike, is_grounded: bool = False
    ) -> Tuple[Path, Path]:
        project_dir = Path(out_dir)
        project_dir.mkdir(parents=True, exist_ok=True)
        target_addr_hex = f"0x{config.target_addr:x}" if config.target_addr is not None else "0x0"
        grounded_status = "GROUNDED" if is_grounded else "UNVERIFIED"

        if config.mod_type == "gdextension":
            manifest_file = project_dir / f"{config.plugin_name}.gdextension"
            entry_symbol = f"{config.plugin_name.lower()}_library_init"
            manifest_file.write_text(
                cls.GDEXTENSION_TEMPLATE.format(
                    entry_symbol=entry_symbol,
                    plugin_name=config.plugin_name,
                ),
                encoding="utf-8",
            )

            src_dir = project_dir / "src"
            src_dir.mkdir(parents=True, exist_ok=True)

            cmake_file = project_dir / "CMakeLists.txt"
            cmake_file.write_text(
                cls.GDEXTENSION_CMAKELISTS.format(plugin_name=config.plugin_name),
                encoding="utf-8",
            )

            source_file = src_dir / f"{config.plugin_name}.cpp"
            source_content = cls.GDEXTENSION_CPP_TEMPLATE.format(
                plugin_name=config.plugin_name,
                version=config.version,
                target_node=config.target_node,
                target_method=config.target_method,
                target_addr_hex=target_addr_hex,
                grounded_status=grounded_status,
                boot_token=config.boot_token,
                entry_symbol=entry_symbol,
            )
            source_file.write_text(source_content, encoding="utf-8")
            return manifest_file, source_file

        else:  # gdscript
            manifest_file = project_dir / "override.cfg"
            manifest_file.write_text(
                cls.GDSCRIPT_OVERRIDE_CFG.format(plugin_name=config.plugin_name),
                encoding="utf-8",
            )

            mod_dir = project_dir / "mods" / config.plugin_name
            mod_dir.mkdir(parents=True, exist_ok=True)
            source_file = mod_dir / "mod_entry.gd"

            logic_body = (
                config.mod_logic
                or f'print("[Toothpick] Hook executed on {config.target_node}.{config.target_method}")'
            )
            source_content = cls.GDSCRIPT_ENTRY_TEMPLATE.format(
                plugin_name=config.plugin_name,
                version=config.version,
                target_node=config.target_node,
                target_method=config.target_method,
                target_addr_hex=target_addr_hex,
                grounded_status=grounded_status,
                boot_token=config.boot_token,
                logic_body=logic_body,
            )
            source_file.write_text(source_content, encoding="utf-8")
            return manifest_file, source_file

    @classmethod
    def compile_project(cls, project_dir: PathLike) -> Optional[Path]:
        pdir = Path(project_dir)
        cmake_bin = shutil.which("cmake")
        if cmake_bin is None:
            return None
        cmakelists = pdir / "CMakeLists.txt"
        if not cmakelists.exists():
            return None
        build_dir = pdir / "build"
        build_dir.mkdir(parents=True, exist_ok=True)
        try:
            res = subprocess.run(
                [cmake_bin, "-B", str(build_dir), "-S", str(pdir)],
                capture_output=True,
                check=False,
                timeout=30.0,
            )
            if res.returncode == 0:
                subprocess.run(
                    [cmake_bin, "--build", str(build_dir), "--config", "Release"],
                    capture_output=True,
                    check=False,
                    timeout=60.0,
                )
                dlls = list(build_dir.glob("**/*.dll"))
                if dlls:
                    return dlls[0]
        except Exception:
            pass
        return None


# ---------------------------------------------------------------------------
# Godot PCK Package Parser & Synthesizer (W13)
# ---------------------------------------------------------------------------


@dataclass
class PckEntry:
    """Directory entry inside a Godot .pck container package."""

    path: str
    offset: int
    size: int
    md5: bytes = field(default_factory=bytes)
    flags: int = 0


class GodotPckReader:
    """Parser, extractor, and writer for Godot Engine .pck container packages.

    Supports format version 1 (Godot 3.x) and format version 2 (Godot 4.x)
    packages with the 'GDPC' magic header (0x43504447).
    """

    MAGIC = b"GDPC"  # 0x43504447 in little-endian

    def __init__(self, pck_path: PathLike) -> None:
        self.pck_path = Path(pck_path)
        self._entries: Optional[List[PckEntry]] = None
        self.format_version: int = 0
        self.version_major: int = 0
        self.version_minor: int = 0
        self.version_patch: int = 0
        self.flags: int = 0
        self.file_base_offset: int = 0

    def parse(self) -> List[PckEntry]:
        """Parse the PCK header and directory index."""
        if not self.pck_path.is_file():
            raise PckParseError(f"PCK file not found: {self.pck_path}")

        data = self.pck_path.read_bytes()
        if len(data) < 16:
            raise PckParseError(f"File too small for PCK header: {len(data)} bytes")

        magic = data[:4]
        if magic != self.MAGIC:
            raise PckParseError(
                f"Invalid PCK magic: {magic!r}; expected {self.MAGIC!r} (GDPC)"
            )

        offset = 4
        (
            self.format_version,
            self.version_major,
            self.version_minor,
            self.version_patch,
        ) = struct.unpack_from("<IIII", data, offset)
        offset += 16

        if self.format_version >= 2:
            self.flags, self.file_base_offset = struct.unpack_from("<IQ", data, offset)
            offset += 12

        # 16 reserved uint32s = 64 bytes
        offset += 64
        if offset + 4 > len(data):
            raise PckParseError("Unexpected EOF reading reserved header fields")

        (file_count,) = struct.unpack_from("<I", data, offset)
        offset += 4

        entries: List[PckEntry] = []
        for _ in range(file_count):
            if offset + 4 > len(data):
                raise PckParseError("Unexpected EOF reading file entry path length")
            (path_len,) = struct.unpack_from("<I", data, offset)
            offset += 4

            # Path is padded to 4 bytes boundary
            padded_len = ((path_len + 3) // 4) * 4
            if offset + padded_len > len(data):
                raise PckParseError("Unexpected EOF reading file entry path")
            raw_path = data[offset : offset + path_len]
            file_path = raw_path.decode("utf-8", errors="replace").rstrip("\x00")
            offset += padded_len

            if offset + 16 > len(data):
                raise PckParseError("Unexpected EOF reading file offset and size")
            file_offset, file_size = struct.unpack_from("<QQ", data, offset)
            offset += 16

            md5 = b""
            if offset + 16 <= len(data):
                md5 = data[offset : offset + 16]
                offset += 16

            flags = 0
            if self.format_version >= 2:
                if offset + 4 <= len(data):
                    (flags,) = struct.unpack_from("<I", data, offset)
                    offset += 4

            entries.append(
                PckEntry(
                    path=file_path,
                    offset=file_offset,
                    size=file_size,
                    md5=md5,
                    flags=flags,
                )
            )

        self._entries = entries
        return entries

    @property
    def entries(self) -> List[PckEntry]:
        if self._entries is None:
            return self.parse()
        return self._entries

    def extract_file(self, path_in_pck: str, out_path: PathLike) -> Path:
        """Extract a single file from the PCK by its internal path."""
        entries = self.entries
        wanted = normalize_pck_path(path_in_pck)
        match = None
        for e in entries:
            if e.path == path_in_pck or normalize_pck_path(e.path) == wanted:
                match = e
                break

        if match is None:
            raise PckParseError(f"File {path_in_pck!r} not found in PCK {self.pck_path}")

        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        data = self.pck_path.read_bytes()
        start = match.offset
        end = start + match.size
        if end > len(data):
            raise PckParseError(
                f"Entry offset+size exceeds file boundary: {end} > {len(data)}"
            )

        out.write_bytes(data[start:end])
        return out

    def extract_all(self, out_dir: PathLike) -> List[Path]:
        """Extract all files in the PCK, preserving sanitized virtual paths.

        Every destination is mapped through :func:`safe_relative_pck_path`, so a
        hostile ``..`` or absolute entry can never write outside ``out_dir``.
        """
        base = Path(out_dir)
        base.mkdir(parents=True, exist_ok=True)
        extracted: List[Path] = []
        for e in self.entries:
            rel_path = safe_relative_pck_path(e.path)
            if rel_path is None:
                raise PckParseError(
                    f"Refusing unsafe path in PCK directory index: {e.path!r}"
                )
            dest = base / rel_path
            self.extract_file(e.path, dest)
            extracted.append(dest)
        return sorted(extracted)

    @classmethod
    def create_pck(
        cls,
        files: Dict[str, bytes],
        out_path: PathLike,
        godot_version: Tuple[int, int, int] = (4, 0, 0),
        format_version: int = 2,
    ) -> Path:
        """Create a new Godot .pck file containing the specified virtual file dict."""
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        major, minor, patch = godot_version
        file_count = len(files)

        hdr_size = 4 + 16 + (12 if format_version >= 2 else 0) + 64 + 4
        dir_size = 0
        encoded_paths: Dict[str, Tuple[bytes, bytes]] = {}
        for path_str in files.keys():
            raw_path = path_str.encode("utf-8")
            path_len = len(raw_path)
            padded_len = ((path_len + 3) // 4) * 4
            padded_path = raw_path.ljust(padded_len, b"\x00")
            encoded_paths[path_str] = (raw_path, padded_path)
            entry_bytes = 4 + padded_len + 8 + 8 + 16 + (4 if format_version >= 2 else 0)
            dir_size += entry_bytes

        data_start_offset = hdr_size + dir_size

        dir_buf = bytearray()
        current_data_offset = data_start_offset

        for path_str, file_bytes in files.items():
            raw_path, padded_path = encoded_paths[path_str]
            size = len(file_bytes)
            md5 = hashlib.md5(file_bytes).digest()

            dir_buf.extend(struct.pack("<I", len(raw_path)))
            dir_buf.extend(padded_path)
            dir_buf.extend(struct.pack("<QQ", current_data_offset, size))
            dir_buf.extend(md5)
            if format_version >= 2:
                dir_buf.extend(struct.pack("<I", 0))

            current_data_offset += size

        with open(out, "wb") as f:
            f.write(cls.MAGIC)
            f.write(struct.pack("<IIII", format_version, major, minor, patch))
            if format_version >= 2:
                f.write(struct.pack("<IQ", 0, 0))
            f.write(b"\x00" * 64)
            f.write(struct.pack("<I", file_count))
            f.write(dir_buf)

            for file_bytes in files.values():
                f.write(file_bytes)

        return out


# ---------------------------------------------------------------------------
# Mod Builder (Grounding Gate & Coordinator)
# ---------------------------------------------------------------------------



class ModBuilder:
    """Coordinates target resolution, Epistemic Grounding Gate, generation, and compilation."""

    def __init__(
        self,
        db: Optional[Union[Database, PathLike]] = None,
        binary_sha256: Optional[str] = None,
    ) -> None:
        self.db: Optional[Database] = None
        self._owned_db = False
        if db is not None:
            if isinstance(db, Database):
                self.db = db
            else:
                self.db = Database(Path(db), create=False)
                self._owned_db = True
        self.binary_sha256 = binary_sha256

    def close(self) -> None:
        """Cleanly close database handles (Trap 7)."""
        if self._owned_db and self.db is not None:
            self.db.close()
            self.db = None

    def __enter__(self) -> ModBuilder:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _resolve_sha(self) -> str:
        """Resolve target binary SHA-256 ensuring unambiguous provenance (Trap 20)."""
        if self.binary_sha256:
            return self.binary_sha256
        if self.db is not None:
            rows = self.db._conn.execute(
                "SELECT DISTINCT binary_sha256 FROM funcs WHERE binary_sha256 != '' "
                "UNION SELECT DISTINCT binary_sha256 FROM types WHERE binary_sha256 != ''"
            ).fetchall()
            shas = [r[0] for r in rows if r[0]]
            if len(shas) == 1:
                return shas[0]
            if len(shas) > 1:
                raise ModBuildError(
                    "Cannot infer binary SHA-256: database contains "
                    f"{len(shas)} distinct binaries ({', '.join(sorted(shas))}); "
                    "pass an explicit --sha"
                )
        return ""

    def resolve_target(
        self,
        target: Union[int, str],
        binary_sha256: str = "",
    ) -> Tuple[int, str, str]:
        """Resolve a target address or class/method string into (addr, class_name, method_name)."""
        # If target is integer or numeric/hex string
        if isinstance(target, int):
            addr = target
            return self._find_name_for_addr(addr, binary_sha256)

        target_str = str(target).strip()
        if target_str.startswith("0x") or target_str.isdigit():
            try:
                addr = parse_address(target_str)
                return self._find_name_for_addr(addr, binary_sha256)
            except Exception:
                pass

        # String target: ClassName.MethodName or ClassName$$MethodName
        cls_name = "TargetClass"
        meth_name = target_str
        if "$$" in target_str:
            cls_name, meth_name = target_str.split("$$", 1)
        elif "." in target_str:
            cls_name, meth_name = target_str.rsplit(".", 1)

        # Look in types table
        if self.db is not None:
            types = self.db.get_types(binary_sha256 if binary_sha256 else None)
            for t in types:
                if t.name.lower() == cls_name.lower():
                    cls_name = t.name
                    methods = (t.layout or {}).get("methods", [])
                    for m in methods:
                        if isinstance(m, dict) and m.get("name", "").lower() == meth_name.lower():
                            m_addr = m.get("addr") or m.get("address")
                            if m_addr is not None:
                                return int(m_addr), cls_name, m.get("name", meth_name)

            # Look in funcs table
            funcs = self.db.get_functions(binary_sha256) if binary_sha256 else []
            for f in funcs:
                if f.name.lower() == target_str.lower() or f.name.lower().endswith(f".{meth_name.lower()}"):
                    actual_cls = cls_name
                    actual_meth = meth_name
                    if "." in f.name:
                        actual_cls, actual_meth = f.name.rsplit(".", 1)
                    return f.func_addr, actual_cls, actual_meth

        raise ModBuildError(
            f"Cannot resolve target '{target}' to a known function address in database"
        )

    def _find_name_for_addr(self, addr: int, binary_sha256: str) -> Tuple[int, str, str]:
        """Find class and method names for a known address, or return sensible defaults."""
        if self.db is not None:
            # Check funcs table
            if binary_sha256:
                func = self.db.get_function(binary_sha256, addr)
                if func and func.name:
                    if "." in func.name:
                        c, m = func.name.rsplit(".", 1)
                        return addr, c, m
                    return addr, "TargetClass", func.name

            # Check types table
            types = self.db.get_types(binary_sha256 if binary_sha256 else None)
            for t in types:
                for m in (t.layout or {}).get("methods", []):
                    if isinstance(m, dict):
                        m_addr = m.get("addr") or m.get("address")
                        if m_addr == addr:
                            return addr, t.name, m.get("name", f"Sub_{addr:x}")

        return addr, "TargetClass", f"Sub_{addr:x}"

    def is_target_grounded(self, target_addr: int, binary_sha256: str) -> bool:
        """Check whether target_addr has a confirmed GROUNDED dynamic receipt."""
        if self.db is None:
            return False
        receipts = self.db.get_receipts(binary_sha256, target_addr)
        for r in receipts:
            payload = r.get("payload", {})
            status = payload.get("verifier_status") or r.get("verifier_status")
            if status == "GROUNDED":
                return True
        return False

    def _split_symbol(self, target: str) -> Tuple[str, str]:
        """Split a target symbol into ``(class_or_path, function_or_method)``.

        Handles Unreal script paths (``/Script/Engine.Actor:ReceiveBeginPlay``),
        Il2CppDumper names (``PlayerController$$Update``), and dotted names
        (``PlayerController.Update``).
        """
        text = (target or "").strip()
        for separator in (":", "$$", "."):
            if separator in text:
                head, _, tail = text.rpartition(separator)
                if head and tail:
                    return head, tail
        return "Global", (text or "TargetFunction")

    def _resolve_target_any(
        self, target: Union[int, str], binary_sha256: str
    ) -> Tuple[Optional[int], str, str]:
        """Resolve a target to ``(addr_or_None, class_or_node, function_or_method)``.

        Numeric targets and names present in the database resolve to a concrete
        address; symbolic targets without a recovered address stay unresolved
        (``None``), which the Epistemic Grounding Gate treats as ungrounded
        (Design §16 #1).
        """
        if isinstance(target, int):
            return self._find_name_for_addr(target, binary_sha256)

        text = str(target).strip()
        if text.startswith("0x") or text.isdigit():
            try:
                return self._find_name_for_addr(parse_address(text), binary_sha256)
            except Exception:
                pass

        try:
            return self.resolve_target(text, binary_sha256)
        except ModBuildError:
            cls_name, func_name = self._split_symbol(text)
            return None, cls_name, func_name

    def _enforce_grounding_gate(
        self,
        target_addr: Optional[int],
        target_class: str,
        target_function: str,
        binary_sha256: str,
        allow_unverified: bool,
    ) -> bool:
        """Refuse to emit a grounded mod artifact without a live receipt (§16 #1).

        Returns ``True`` when the target has a confirmed ``GROUNDED`` receipt and
        ``False`` when the caller explicitly overrode the gate with
        ``--allow-unverified``; raises :class:`UngroundedTargetError` otherwise.
        """
        is_grounded = target_addr is not None and self.is_target_grounded(
            target_addr, binary_sha256
        )
        if is_grounded or allow_unverified:
            return is_grounded
        addr_text = f"0x{target_addr:x}" if target_addr is not None else "<unresolved>"
        raise UngroundedTargetError(
            f"Target address {addr_text} ({target_class}.{target_function}) has no confirmed "
            "GROUNDED dynamic receipt in database; pass --allow-unverified to override (Design §16 #1)"
        )

    @staticmethod
    def _collect_files(project_dir: PathLike) -> List[Path]:
        """Every generated file in the project tree, deterministically ordered."""
        base = Path(project_dir)
        return sorted(path for path in base.rglob("*") if path.is_file())

    def _record_mod_receipt(
        self,
        phase: str,
        binary_sha256: str,
        target_addr: Optional[int],
        payload: Dict[str, Any],
    ) -> Optional[str]:
        """Persist an engine-scoped mod build receipt (no-op without a database or ungrounded)."""
        if self.db is None or target_addr is None:
            return None
        return self.db.add_receipt(
            binary_sha256=binary_sha256,
            func_addr=int(target_addr),
            phase=phase,
            payload=payload,
        )

    def _build_unity(
        self,
        target: Union[int, str],
        out_dir: PathLike,
        name: Optional[str],
        allow_unverified: bool,
        compile: bool,
        hook_type: str,
        mod_logic: Optional[str],
        binary_sha256: str,
        mod_type: Optional[str],
    ) -> ModArtifact:
        """Build a BepInEx C# mod project (unit W08 contract, unchanged behaviour)."""
        target_addr, target_class, target_method = self.resolve_target(target, binary_sha256)
        is_grounded = self._enforce_grounding_gate(
            target_addr, target_class, target_method, binary_sha256, allow_unverified
        )

        plugin_name = name or f"{target_class}Mod"
        config = ModConfig(
            plugin_name=plugin_name,
            target_addr=target_addr,
            target_class=target_class,
            target_method=target_method,
            hook_type=hook_type,
            mod_logic=mod_logic,
            binary_sha256=binary_sha256,
        )

        project_dir = Path(out_dir)
        csproj_file, source_file = ModGenerator.generate(
            config=config,
            out_dir=project_dir,
            is_grounded=is_grounded,
        )

        dll_path: Optional[Path] = None
        if compile:
            dll_path = ModCompiler.compile(project_dir=project_dir)

        resolved_mod_type = (mod_type or "bepinex_csharp").strip().lower()
        receipt_payload = {
            "engine": "unity",
            "mod_type": resolved_mod_type,
            "plugin_name": config.plugin_name,
            "plugin_id": config.plugin_id,
            "boot_token": config.boot_token,
            "target_class": target_class,
            "target_method": target_method,
            "verifier_status": "GROUNDED" if is_grounded else "UNGROUNDED",
            "allow_unverified": allow_unverified,
            "dll_path": str(dll_path) if dll_path else None,
        }
        receipt_id = self._record_mod_receipt(
            "mod_build", binary_sha256, target_addr, receipt_payload
        )

        return ModArtifact(
            project_dir=project_dir,
            source_file=source_file,
            csproj_file=csproj_file,
            dll_path=dll_path,
            is_grounded=is_grounded,
            receipt_id=receipt_id,
            boot_token=config.boot_token,
            config=config,
            engine="unity",
            mod_type=resolved_mod_type,
            files=self._collect_files(project_dir),
            manifest_file=csproj_file,
            target_addr=target_addr,
            target_symbol=f"{target_class}.{target_method}",
        )

    def _build_unreal(
        self,
        target: Union[int, str],
        out_dir: PathLike,
        name: Optional[str],
        allow_unverified: bool,
        compile: bool,
        hook_type: str,
        mod_logic: Optional[str],
        binary_sha256: str,
        mod_type: Optional[str],
    ) -> ModArtifact:
        """Build a UE4SS Lua or native C++ mod project for Unreal Engine 4/5 (W13)."""
        target_addr, target_class, target_function = self._resolve_target_any(
            target, binary_sha256
        )
        is_grounded = self._enforce_grounding_gate(
            target_addr, target_class, target_function, binary_sha256, allow_unverified
        )

        hook = (hook_type or "post").strip().lower()
        hook = {"postfix": "post", "prefix": "pre"}.get(hook, hook)
        plugin_name = name or f"{_sanitize_symbol(target_function, 'UnrealTarget')}Mod"
        config = UnrealModConfig(
            plugin_name=plugin_name,
            target_addr=target_addr,
            target_class=target_class,
            target_function=target_function,
            hook_type=hook,
            mod_type=(mod_type or "ue4ss_lua"),
            mod_logic=mod_logic,
            binary_sha256=binary_sha256,
        )

        project_dir = Path(out_dir)
        manifest_file, source_file = UnrealModGenerator.generate(
            config=config,
            out_dir=project_dir,
            is_grounded=is_grounded,
        )

        dll_path: Optional[Path] = None
        if compile:
            dll_path = UnrealModGenerator.compile_project(project_dir)

        receipt_payload = {
            "engine": "unreal",
            "mod_type": config.mod_type,
            "plugin_name": config.plugin_name,
            "boot_token": config.boot_token,
            "target_path": config.ue4ss_target_path,
            "target_addr": target_addr,
            "hook_type": config.hook_type,
            "verifier_status": "GROUNDED" if is_grounded else "UNGROUNDED",
            "allow_unverified": allow_unverified,
            "dll_path": str(dll_path) if dll_path else None,
        }
        receipt_id = self._record_mod_receipt(
            "mod_build_unreal", binary_sha256, target_addr, receipt_payload
        )

        return ModArtifact(
            project_dir=project_dir,
            source_file=source_file,
            csproj_file=None,
            dll_path=dll_path,
            is_grounded=is_grounded,
            receipt_id=receipt_id,
            boot_token=config.boot_token,
            config=config,
            engine="unreal",
            mod_type=config.mod_type,
            files=self._collect_files(project_dir),
            manifest_file=manifest_file,
            target_addr=target_addr,
            target_symbol=config.ue4ss_target_path,
        )

    def _build_godot(
        self,
        target: Union[int, str],
        out_dir: PathLike,
        name: Optional[str],
        allow_unverified: bool,
        compile: bool,
        mod_logic: Optional[str],
        binary_sha256: str,
        mod_type: Optional[str],
        godot_version: int,
    ) -> ModArtifact:
        """Build a Godot GDExtension C++ or GDScript override mod project (W13)."""
        target_addr, target_node, target_method = self._resolve_target_any(
            target, binary_sha256
        )
        is_grounded = self._enforce_grounding_gate(
            target_addr, target_node, target_method, binary_sha256, allow_unverified
        )

        plugin_name = name or f"{_sanitize_symbol(target_method, 'GodotTarget')}Mod"
        config = GodotModConfig(
            plugin_name=plugin_name,
            godot_version=godot_version,
            mod_type=(mod_type or "gdextension"),
            target_node=target_node,
            target_method=target_method,
            target_addr=target_addr,
            mod_logic=mod_logic,
            binary_sha256=binary_sha256,
        )

        project_dir = Path(out_dir)
        manifest_file, source_file = GodotModGenerator.generate(
            config=config,
            out_dir=project_dir,
            is_grounded=is_grounded,
        )

        library_path: Optional[Path] = None
        if compile:
            library_path = GodotModGenerator.compile_project(project_dir)

        receipt_payload = {
            "engine": "godot",
            "mod_type": config.mod_type,
            "plugin_name": config.plugin_name,
            "godot_version": config.godot_version,
            "boot_token": config.boot_token,
            "target_node": config.target_node,
            "target_method": config.target_method,
            "target_addr": target_addr,
            "verifier_status": "GROUNDED" if is_grounded else "UNGROUNDED",
            "allow_unverified": allow_unverified,
            "dll_path": str(library_path) if library_path else None,
        }
        receipt_id = self._record_mod_receipt(
            "mod_build_godot", binary_sha256, target_addr, receipt_payload
        )

        return ModArtifact(
            project_dir=project_dir,
            source_file=source_file,
            csproj_file=None,
            dll_path=library_path,
            is_grounded=is_grounded,
            receipt_id=receipt_id,
            boot_token=config.boot_token,
            config=config,
            engine="godot",
            mod_type=config.mod_type,
            files=self._collect_files(project_dir),
            manifest_file=manifest_file,
            target_addr=target_addr,
            target_symbol=f"{config.target_node}.{config.target_method}",
        )

    def build(
        self,
        target: Union[int, str],
        out_dir: PathLike,
        name: Optional[str] = None,
        allow_unverified: bool = False,
        compile: bool = False,
        hook_type: str = "postfix",
        mod_logic: Optional[str] = None,
        binary_sha256: Optional[str] = None,
        engine: str = "unity",
        mod_type: Optional[str] = None,
        godot_version: int = 4,
    ) -> ModArtifact:
        """Build a mod project, enforcing the Epistemic Grounding Gate (Design §16 #1).

        ``engine`` selects the pipeline backend: ``unity`` (BepInEx 5/6 C#, unit
        W08), ``unreal`` (UE4SS Lua or native C++ DLL for UE4/UE5, unit W13), or
        ``godot`` (GDExtension C++ or GDScript override, unit W13).  ``mod_type``
        selects the variant inside the engine (``bepinex_csharp`` | ``ue4ss_lua`` |
        ``native_cpp`` | ``gdextension`` | ``gdscript``).
        """
        sha = binary_sha256 or self._resolve_sha()
        engine_key = (engine or "unity").strip().lower()

        if engine_key in ("unity", "il2cpp", "bepinex", "mono"):
            return self._build_unity(
                target, out_dir, name, allow_unverified, compile, hook_type, mod_logic,
                sha, mod_type,
            )
        if engine_key in ("unreal", "unreal4", "unreal5", "ue", "ue4", "ue5"):
            return self._build_unreal(
                target, out_dir, name, allow_unverified, compile, hook_type, mod_logic,
                sha, mod_type,
            )
        if engine_key in ("godot", "godot3", "godot4"):
            return self._build_godot(
                target, out_dir, name, allow_unverified, compile, mod_logic,
                sha, mod_type, godot_version,
            )
        raise ModBuildError(
            f"Unknown engine {engine!r}; expected 'unity', 'unreal', or 'godot'"
        )
