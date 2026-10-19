"""
Toothpick Triage Module.
Classifies binary containers, target architectures, game engines, and packer protections.
"""

from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional
import struct
import json


class ContainerType(str, Enum):
    PE = "pe"
    ELF = "elf"
    NSO = "nso"
    NRO = "nro"
    SELF = "self"
    DOL = "dol"
    RAW = "raw"
    UNKNOWN = "unknown"


class Architecture(str, Enum):
    X86 = "x86"
    X86_64 = "x86_64"
    ARM64 = "arm64"
    PPC = "ppc"
    MIPS = "mips"
    UNKNOWN = "unknown"


class EngineType(str, Enum):
    UNITY_IL2CPP = "unity-il2cpp"
    UNITY_MONO = "unity-mono"
    UNREAL = "ue"
    GODOT = "godot"
    DOTNET = "dotnet"
    NATIVE = "native"
    UNSUPPORTED = "unsupported"


class PackerStatus(str, Enum):
    UNPACKED = "unpacked"
    PACKED_SIMPLE = "packed_simple"
    PACKED_VM = "packed_vm"
    UNKNOWN = "unknown"


class PackerAction(str, Enum):
    PROCEED = "proceed"
    UNPACK_REQUIRED = "unpack_required"
    ABORT = "abort"


class DynamicCapability(str, Enum):
    NATIVE_FRIDA = "native_frida"
    EMULATOR_BRIDGED = "emulator_bridged"
    STATIC_ONLY = "static_only"


@dataclass
class ProtectionReport:
    status: PackerStatus
    packer: str
    action: PackerAction


@dataclass
class TriageReport:
    target: str
    container: ContainerType
    arch: Architecture
    engine: EngineType
    protection: ProtectionReport
    dynamic_capability: DynamicCapability

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["container"] = self.container.value
        d["arch"] = self.arch.value
        d["engine"] = self.engine.value
        d["protection"]["status"] = self.protection.status.value
        d["protection"]["action"] = self.protection.action.value
        d["dynamic_capability"] = self.dynamic_capability.value
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class TriageError(Exception):
    """Base exception for triage failures."""
    pass


class PackedAbortError(TriageError):
    """Raised when target binary is protected by an unsupported hypervisor/VM packer."""
    pass


class UnknownFormatError(TriageError):
    """Raised when target format is unknown and no --raw mapping hint was provided."""
    pass


def triage_binary(
    target_path: Path,
    is_raw: bool = False,
    raw_arch: Optional[str] = None,
    raw_base: Optional[int] = None,
    static_only: bool = False,
) -> TriageReport:
    target_path = Path(target_path)
    if not target_path.exists():
        raise FileNotFoundError(f"Target file not found: {target_path}")

    data = target_path.read_bytes()
    if len(data) == 0:
        raise UnknownFormatError("UNKNOWN_FORMAT_NEED_HINT: Empty file")

    if is_raw:
        arch = Architecture(raw_arch) if raw_arch else Architecture.UNKNOWN
        return TriageReport(
            target=str(target_path),
            container=ContainerType.RAW,
            arch=arch,
            engine=EngineType.NATIVE,
            protection=ProtectionReport(
                status=PackerStatus.UNPACKED,
                packer="none",
                action=PackerAction.PROCEED,
            ),
            dynamic_capability=DynamicCapability.STATIC_ONLY,
        )

    # Magic inspection
    if data.startswith(b"MZ"):
        return _triage_pe(target_path, data, static_only)
    elif data.startswith(b"\x7fELF"):
        return _triage_elf(target_path, data, static_only)
    elif data.startswith(b"NSO0"):
        return TriageReport(
            target=str(target_path),
            container=ContainerType.NSO,
            arch=Architecture.ARM64,
            engine=EngineType.NATIVE,
            protection=ProtectionReport(
                status=PackerStatus.UNPACKED,
                packer="none",
                action=PackerAction.PROCEED,
            ),
            dynamic_capability=DynamicCapability.STATIC_ONLY if static_only else DynamicCapability.EMULATOR_BRIDGED,
        )
    elif len(data) > 0x14 and data[0x10:0x14] == b"NRO0":
        return TriageReport(
            target=str(target_path),
            container=ContainerType.NRO,
            arch=Architecture.ARM64,
            engine=EngineType.NATIVE,
            protection=ProtectionReport(
                status=PackerStatus.UNPACKED,
                packer="none",
                action=PackerAction.PROCEED,
            ),
            dynamic_capability=DynamicCapability.STATIC_ONLY if static_only else DynamicCapability.EMULATOR_BRIDGED,
        )
    elif data.startswith(b"SCE\x00"):
        return TriageReport(
            target=str(target_path),
            container=ContainerType.SELF,
            arch=Architecture.X86_64,
            engine=EngineType.NATIVE,
            protection=ProtectionReport(
                status=PackerStatus.UNPACKED,
                packer="none",
                action=PackerAction.PROCEED,
            ),
            dynamic_capability=DynamicCapability.STATIC_ONLY if static_only else DynamicCapability.EMULATOR_BRIDGED,
        )
    elif len(data) >= 0x100 and struct.unpack_from(">I", data, 0)[0] == 0x100 and (0x80000000 <= struct.unpack_from(">I", data, 0xE0)[0] < 0x81800000):
        return TriageReport(
            target=str(target_path),
            container=ContainerType.DOL,
            arch=Architecture.PPC,
            engine=EngineType.NATIVE,
            protection=ProtectionReport(
                status=PackerStatus.UNPACKED,
                packer="none",
                action=PackerAction.PROCEED,
            ),
            dynamic_capability=DynamicCapability.STATIC_ONLY if static_only else DynamicCapability.EMULATOR_BRIDGED,
        )

    raise UnknownFormatError("UNKNOWN_FORMAT_NEED_HINT")


def _triage_pe(target_path: Path, data: bytes, static_only: bool) -> TriageReport:
    # Basic PE parse
    arch = Architecture.UNKNOWN
    if len(data) >= 0x40:
        e_lfanew = struct.unpack_from("<I", data, 0x3C)[0]
        if e_lfanew + 24 <= len(data) and data[e_lfanew:e_lfanew+4] == b"PE\x00\x00":
            machine = struct.unpack_from("<H", data, e_lfanew + 4)[0]
            if machine == 0x8664:
                arch = Architecture.X86_64
            elif machine == 0x014C:
                arch = Architecture.X86
            elif machine == 0xAA64:
                arch = Architecture.ARM64

    # Packer / VM protection inspection
    if b".vmp0" in data or b".vmp1" in data or b"VMProtect" in data:
        raise PackedAbortError("PACKED_ABORT(VM_PROTECTED)")
    if b".themida" in data or b"Themida" in data:
        raise PackedAbortError("PACKED_ABORT(VM_PROTECTED)")

    packer_status = PackerStatus.UNPACKED
    packer_name = "none"
    packer_action = PackerAction.PROCEED

    if b"UPX0" in data or b"UPX1" in data or b"UPX!" in data:
        packer_status = PackerStatus.PACKED_SIMPLE
        packer_name = "upx"
        packer_action = PackerAction.UNPACK_REQUIRED
    elif b".pec2" in data or b"PECompact2" in data:
        packer_status = PackerStatus.PACKED_SIMPLE
        packer_name = "pecompact"
        packer_action = PackerAction.UNPACK_REQUIRED

    # Engine inspection
    engine = EngineType.NATIVE
    if b"GameAssembly.dll" in data or b"il2cpp" in data or b"UnityPlayer.dll" in data:
        engine = EngineType.UNITY_IL2CPP
    elif b"MonoBleedingEdge" in data or b"mono-2.0-bdwgc.dll" in data:
        engine = EngineType.UNITY_MONO
    elif b"UnrealEngine" in data or b"FPlatform" in data or b"UE4" in data or b"UE5" in data:
        engine = EngineType.UNREAL
    elif b"godot" in data or b"GDScript" in data:
        engine = EngineType.GODOT
    elif b"_CorExeMain" in data or b"mscoree.dll" in data:
        engine = EngineType.DOTNET

    dynamic_cap = DynamicCapability.STATIC_ONLY if static_only else DynamicCapability.NATIVE_FRIDA

    return TriageReport(
        target=str(target_path),
        container=ContainerType.PE,
        arch=arch,
        engine=engine,
        protection=ProtectionReport(
            status=packer_status,
            packer=packer_name,
            action=packer_action,
        ),
        dynamic_capability=dynamic_cap,
    )


def _triage_elf(target_path: Path, data: bytes, static_only: bool) -> TriageReport:
    arch = Architecture.UNKNOWN
    if len(data) >= 20:
        e_machine = struct.unpack_from("<H", data, 18)[0]
        if e_machine == 0x3E:
            arch = Architecture.X86_64
        elif e_machine == 0x03:
            arch = Architecture.X86
        elif e_machine == 0xB7:
            arch = Architecture.ARM64
        elif e_machine == 0x14:
            arch = Architecture.PPC
        elif e_machine == 0x08:
            arch = Architecture.MIPS

    return TriageReport(
        target=str(target_path),
        container=ContainerType.ELF,
        arch=arch,
        engine=EngineType.NATIVE,
        protection=ProtectionReport(
            status=PackerStatus.UNPACKED,
            packer="none",
            action=PackerAction.PROCEED,
        ),
        dynamic_capability=DynamicCapability.STATIC_ONLY if static_only else DynamicCapability.NATIVE_FRIDA,
    )
