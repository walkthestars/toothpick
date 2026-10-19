"""Parser fuzzing and hostile byte boundary testing battery.

Governed by design-document.md §6 (Target Binary Architecture Matrix)
and execution-plan.md §4 (CI/Verification specification).

Enforces strict robustness invariant:
All malformed, truncated, or hostile inputs across PE, ELF, NSO, NRO, SELF, DOL,
and IL2CPP metadata must either parse safely or raise structured exceptions
(TriageError, ContainerParseError, MetadataParseError, UnpackError).
Under no circumstance may unhandled exceptions (e.g. IndexError, struct.error,
MemoryError, ZeroDivisionError) crash the pipeline, nor may any parser hang.
"""

from __future__ import annotations

from pathlib import Path
import random
import struct
from typing import Callable, List

import pytest

from toothpick.mod_builder import Il2CppMetadataReader, MetadataParseError
from toothpick.static_engine import (
    ContainerParseError,
    StaticEngineError,
    load_container,
)
from toothpick.triage import (
    PackedAbortError,
    TriageError,
    UnknownFormatError,
    triage_binary,
)
from toothpick.unpack import UnpackError, unpack_binary

# Structured exceptions permitted when parsing hostile or malformed inputs
STRUCTURED_ERRORS = (
    TriageError,
    PackedAbortError,
    UnknownFormatError,
    ContainerParseError,
    StaticEngineError,
    MetadataParseError,
    UnpackError,
)


# ---------------------------------------------------------------------------
# Minimal valid fixtures for fuzz seeds
# ---------------------------------------------------------------------------


def _seed_pe() -> bytes:
    """Minimal valid 64-bit PE binary."""
    dos_header = bytearray(b"MZ" + b"\x00" * 58)
    dos_header += struct.pack("<I", 0x40)
    pe_sig = b"PE\x00\x00"
    # PE32+ requires SizeOfOptionalHeader = 0xF0; a spec-compliant loader locates the
    # section table at e_lfanew + 24 + SizeOfOptionalHeader (never assume 0).
    file_header = struct.pack("<HHIIIHH", 0x8664, 1, 0, 0, 0, 0xF0, 0x0022)
    optional_header = bytearray(0xF0)
    struct.pack_into("<H", optional_header, 0, 0x20B)  # PE32+
    struct.pack_into("<I", optional_header, 16, 0x1000)  # EntryPoint RVA
    struct.pack_into("<Q", optional_header, 24, 0x140000000)  # ImageBase
    struct.pack_into("<I", optional_header, 32, 0x1000)  # SectionAlignment
    struct.pack_into("<I", optional_header, 36, 0x200)  # FileAlignment
    struct.pack_into("<I", optional_header, 56, 0x3000)  # SizeOfImage
    # Section Header (.text)
    section_header = bytearray(40)
    section_header[0:5] = b".text"
    code = b"\x31\xc0\xc3"  # xor eax, eax; ret
    struct.pack_into("<IIII", section_header, 8, len(code), 0x1000, len(code), 0x200)
    struct.pack_into("<I", section_header, 36, 0x60000020)  # CODE | EXEC | READ
    headers = bytes(dos_header) + pe_sig + file_header + bytes(optional_header) + bytes(section_header)
    padding = b"\x00" * (0x200 - len(headers))
    return headers + padding + code


def _seed_elf() -> bytes:
    """Minimal valid 64-bit ELF binary."""
    # EI_MAG: 7f 45 4c 46, EI_CLASS: 2 (64-bit), EI_DATA: 1 (little-endian), EI_VERSION: 1
    ident = b"\x7fELF\x02\x01\x01\x00" + b"\x00" * 8
    # e_type: 2 (EXEC), e_machine: 0x3E (x86_64), e_version: 1
    # e_entry: 0x400078, e_phoff: 64, e_shoff: 0, e_flags: 0
    # e_ehsize: 64, e_phentsize: 56, e_phnum: 1, e_shentsize: 64, e_shnum: 0, e_shstrndx: 0
    ehdr = struct.pack(
        "<16sHHIQQQIHHHHHH",
        ident,
        2,
        0x3E,
        1,
        0x400078,
        64,
        0,
        0,
        64,
        56,
        1,
        64,
        0,
        0,
    )
    # Phdr: p_type: 1 (PT_LOAD), p_flags: 5 (PF_R | PF_X), p_offset: 0, p_vaddr: 0x400000
    # p_paddr: 0x400000, p_filesz: 123, p_memsz: 123, p_align: 0x1000
    code = b"\x31\xc0\xc3"
    total_size = len(ehdr) + 56 + len(code)
    phdr = struct.pack(
        "<IIQQQQQQ",
        1,
        5,
        0,
        0x400000,
        0x400000,
        total_size,
        total_size,
        0x1000,
    )
    return ehdr + phdr + code


def _seed_nso() -> bytes:
    """Minimal valid Nintendo Switch NSO container."""
    # NSO0 magic, version 0, flags 0 (uncompressed), code offset 0x100, size 0x10
    hdr = bytearray(0x100)
    hdr[0:4] = b"NSO0"
    struct.pack_into("<I", hdr, 0x0C, 0)  # flags (uncompressed)
    struct.pack_into("<I", hdr, 0x14, 0x10)  # text size
    struct.pack_into("<I", hdr, 0x40, 0x100)  # text file off
    struct.pack_into("<I", hdr, 0x48, 0x10)  # text file size
    code = b"\xc0\x03\x5f\xd6" * 4  # ARM64 ret * 4
    return bytes(hdr) + code


def _seed_dol() -> bytes:
    """Minimal valid GameCube/Wii DOL container."""
    hdr = bytearray(0x100)
    # text0 offset at 0x100, text0 vaddr 0x80004000, text0 size 0x20, entry 0x80004000
    struct.pack_into(">I", hdr, 0x00, 0x100)  # text0 offset
    struct.pack_into(">I", hdr, 0x48, 0x80004000)  # text0 vaddr
    struct.pack_into(">I", hdr, 0x90, 0x20)  # text0 size
    struct.pack_into(">I", hdr, 0xE0, 0x80004000)  # entry
    code = b"\x4e\x80\x00\x20" * 8  # PPC blr * 8
    return bytes(hdr) + code


def _seed_self() -> bytes:
    """Minimal PlayStation SELF wrapping minimal ELF."""
    elf_data = _seed_elf()
    sce_hdr = b"SCE\x00" + b"\x00" * 28
    return sce_hdr + elf_data


def _seed_il2cpp_metadata() -> bytes:
    """Minimal valid IL2CPP global-metadata.dat."""
    hdr = bytearray(64)
    struct.pack_into("<I", hdr, 0, 0xFAB11BAF)  # magic
    struct.pack_into("<i", hdr, 4, 29)  # version 29
    struct.pack_into("<ii", hdr, 8, 64, 0)  # string_literal
    struct.pack_into("<ii", hdr, 16, 64, 0)  # string_literal_data
    struct.pack_into("<ii", hdr, 24, 64, 0)  # string
    return bytes(hdr)


# ---------------------------------------------------------------------------
# Mutation Engine
# ---------------------------------------------------------------------------


class ByteMutator:
    """Deterministic bit-flip and boundary mutation generator."""

    def __init__(self, seed: int = 1337) -> None:
        self.rng = random.Random(seed)

    def mutate(self, data: bytes) -> bytes:
        """Apply random mutations to binary byte stream."""
        if not data:
            return b"\x00" * self.rng.randint(1, 128)

        buf = bytearray(data)
        mutation_kind = self.rng.choice([
            "bitflip",
            "byte_replace",
            "truncate",
            "expand",
            "zero_range",
            "boundary_int",
        ])

        if mutation_kind == "bitflip":
            idx = self.rng.randrange(len(buf))
            bit = 1 << self.rng.randrange(8)
            buf[idx] ^= bit
        elif mutation_kind == "byte_replace":
            idx = self.rng.randrange(len(buf))
            buf[idx] = self.rng.randrange(256)
        elif mutation_kind == "truncate":
            new_len = self.rng.randrange(1, len(buf))
            buf = buf[:new_len]
        elif mutation_kind == "expand":
            extra = self.rng.randbytes(self.rng.randint(1, 64))
            pos = self.rng.randrange(len(buf) + 1)
            buf = buf[:pos] + extra + buf[pos:]
        elif mutation_kind == "zero_range":
            start = self.rng.randrange(len(buf))
            span = min(self.rng.randint(1, 32), len(buf) - start)
            buf[start : start + span] = b"\x00" * span
        elif mutation_kind == "boundary_int":
            # inject boundary numbers: 0, -1, max_int, etc.
            idx = self.rng.randrange(max(1, len(buf) - 4))
            val = self.rng.choice([0, 0xFFFFFFFF, 0x7FFFFFFF, 0x80000000, 1])
            struct.pack_into("<I", buf, idx, val)

        return bytes(buf)


# ---------------------------------------------------------------------------
# Test Cases: Boundary & Malformed Inputs
# ---------------------------------------------------------------------------


def test_fuzz_pe_parser_boundaries(tmp_path: Path) -> None:
    """Test PE parser against truncated, out-of-bounds, and malformed header boundaries."""
    target = tmp_path / "fuzz_pe.bin"

    # Seed self-check: the valid-PE seed must be spec-compliant, i.e. the section table
    # must sit exactly where SizeOfOptionalHeader declares it, with no need for the
    # parser's magic-derived fallback path.
    seed = _seed_pe()
    assert struct.unpack_from("<H", seed, 0x40 + 20)[0] == 0xF0
    assert seed[0x40 + 24 + 0xF0 : 0x40 + 24 + 0xF0 + 5] == b".text"
    target.write_bytes(seed)
    seed_container = load_container(target)
    assert [s.name for s in seed_container.sections] == [".text"]
    assert not any("SizeOfOptionalHeader" in n for n in seed_container.notes)

    boundary_cases: List[bytes] = [
        b"",  # 0 bytes
        b"M",  # 1 byte
        b"MZ",  # 2 bytes
        b"MZ" + b"\x00" * 58,  # Truncated before e_lfanew
        b"MZ" + b"\x00" * 58 + struct.pack("<I", 0x1000),  # e_lfanew out of file
        b"MZ" + b"\x00" * 58 + struct.pack("<I", 0x40) + b"NE\x00\x00",  # Invalid sig NE
        b"MZ" + b"\x00" * 58 + struct.pack("<I", 0x40) + b"PE\x00\x00",  # Truncated file header
        # Invalid optional header magic (0x999)
        # NOTE: these two hostile cases intentionally declare SizeOfOptionalHeader=0 to
        # exercise the tolerance/fallback path, distinct from the valid seed above.
        # Invalid optional header magic (0x999)
        b"MZ" + b"\x00" * 58 + struct.pack("<I", 0x40) + b"PE\x00\x00" +
        struct.pack("<HHIIIHH", 0x8664, 1, 0, 0, 0, 0, 0x0022) +
        struct.pack("<H", 0x0999) + b"\x00" * 200,
        # Section count overflow (0xFFFF sections on tiny buffer)
        b"MZ" + b"\x00" * 58 + struct.pack("<I", 0x40) + b"PE\x00\x00" +
        struct.pack("<HHIIIHH", 0x8664, 0xFFFF, 0, 0, 0, 0, 0x0022) +
        struct.pack("<H", 0x20B) + b"\x00" * 200,
    ]

    for data in boundary_cases:
        target.write_bytes(data)
        try:
            load_container(target)
        except STRUCTURED_ERRORS:
            pass  # Expected structured rejection
        try:
            triage_binary(target)
        except STRUCTURED_ERRORS:
            pass  # Expected structured rejection


def test_fuzz_elf_parser_boundaries(tmp_path: Path) -> None:
    """Test ELF parser against truncated, out-of-bounds, and corrupted boundaries."""
    target = tmp_path / "fuzz_elf.bin"

    boundary_cases: List[bytes] = [
        b"\x7f",
        b"\x7fELF",
        b"\x7fELF\x02\x01\x01\x00",  # Truncated ident
        b"\x7fELF\x02\x01\x01\x00" + b"\x00" * 8,  # Truncated before ehdr fields
        # ELF header with out-of-bounds program header offset (phoff=0xFFFFFFFF)
        struct.pack(
            "<16sHHIQQQIHHHHHH",
            b"\x7fELF\x02\x01\x01\x00" + b"\x00" * 8,
            2,
            0x3E,
            1,
            0x400000,
            0xFFFFFFFF,
            0,
            0,
            64,
            56,
            1,
            64,
            0,
            0,
        ),
        # ELF with zero sections and zero segments
        struct.pack(
            "<16sHHIQQQIHHHHHH",
            b"\x7fELF\x02\x01\x01\x00" + b"\x00" * 8,
            2,
            0x3E,
            1,
            0x400000,
            0,
            0,
            0,
            64,
            56,
            0,
            64,
            0,
            0,
        ),
    ]

    for data in boundary_cases:
        target.write_bytes(data)
        try:
            load_container(target)
        except STRUCTURED_ERRORS:
            pass
        try:
            triage_binary(target)
        except STRUCTURED_ERRORS:
            pass


def test_fuzz_nso_container_boundaries(tmp_path: Path) -> None:
    """Test Switch NSO / NRO parser boundaries and corrupted compressed payloads."""
    target = tmp_path / "fuzz_nso.bin"

    boundary_cases: List[bytes] = [
        b"NSO0",
        b"NSO0" + b"\x00" * 20,
        # NSO with LZ4 flag set but corrupt/garbage compressed payload
        b"NSO0" + struct.pack("<III", 0, 0, 1) + b"\x00" * 0x30 + b"INVALID_GARBAGE_PAYLOAD" * 5,
        # NRO0 with invalid offsets
        b"\x00" * 0x10 + b"NRO0" + b"\x00" * 0x80,
    ]

    for data in boundary_cases:
        target.write_bytes(data)
        try:
            load_container(target)
        except STRUCTURED_ERRORS:
            pass
        try:
            triage_binary(target)
        except STRUCTURED_ERRORS:
            pass


def test_fuzz_self_container_boundaries(tmp_path: Path) -> None:
    """Test PlayStation SELF parser boundaries."""
    target = tmp_path / "fuzz_self.bin"

    boundary_cases: List[bytes] = [
        b"SCE\x00",
        b"SCE\x00" + b"\x00" * 128,  # SELF with no embedded ELF
        b"SCE\x00" + b"\x00" * 32 + b"\x7fELF_TRUNCATED",  # Truncated wrapped ELF
    ]

    for data in boundary_cases:
        target.write_bytes(data)
        try:
            load_container(target)
        except STRUCTURED_ERRORS:
            pass
        try:
            triage_binary(target)
        except STRUCTURED_ERRORS:
            pass


def test_fuzz_dol_container_boundaries(tmp_path: Path) -> None:
    """Test GameCube/Wii DOL container boundaries."""
    target = tmp_path / "fuzz_dol.bin"

    boundary_cases: List[bytes] = [
        # Under 0x100 bytes
        b"\x00" * 0x80,
        # 0x100 bytes of zeros -> no usable sections
        b"\x00" * 0x100,
        # DOL with text offset out of file range
        struct.pack(">I", 0x100) + b"\x00" * 0xDC + struct.pack(">I", 0x80004000) + b"\x00" * 0x10,
    ]

    for data in boundary_cases:
        target.write_bytes(data)
        try:
            load_container(target)
        except STRUCTURED_ERRORS:
            pass
        try:
            triage_binary(target)
        except STRUCTURED_ERRORS:
            pass


def test_fuzz_il2cpp_metadata_boundaries(tmp_path: Path) -> None:
    """Test IL2CPP global-metadata.dat boundaries."""
    target = tmp_path / "global-metadata.dat"

    boundary_cases: List[bytes] = [
        b"",
        b"\x00" * 4,
        b"NOT_A_MAGIC",
        struct.pack("<I", 0x12345678),  # Wrong magic
        # Correct magic, truncated before version
        struct.pack("<I", 0xFAB11BAF),
        # Correct magic and version, huge out-of-bounds offsets
        struct.pack("<Ii", 0xFAB11BAF, 29) + struct.pack("<iiiiii", 0x7FFFFFFF, 1000, 0x7FFFFFFF, 1000, 0x7FFFFFFF, 1000),
    ]

    for data in boundary_cases:
        target.write_bytes(data)
        try:
            Il2CppMetadataReader.read_binary(target)
        except MetadataParseError:
            pass  # Expected structured rejection


# ---------------------------------------------------------------------------
# Randomized Mutation Fuzzing Battery
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "seed_builder, target_ext",
    [
        (_seed_pe, "pe.exe"),
        (_seed_elf, "elf.bin"),
        (_seed_nso, "main.nso"),
        (_seed_dol, "game.dol"),
        (_seed_self, "eboot.self"),
        (_seed_il2cpp_metadata, "global-metadata.dat"),
    ],
)
def test_fuzz_mutation_engine_resilience(
    tmp_path: Path,
    seed_builder: Callable[[], bytes],
    target_ext: str,
) -> None:
    """Stress test parsers with 50 randomized mutations per binary architecture.

    Verifies zero unhandled exceptions and zero pipeline hangs.
    """
    mutator = ByteMutator(seed=42)
    base_data = seed_builder()
    target = tmp_path / f"mutated_{target_ext}"

    for i in range(50):
        mutated_data = mutator.mutate(base_data)
        target.write_bytes(mutated_data)

        # 1. Triage test
        try:
            triage_binary(target)
        except STRUCTURED_ERRORS:
            pass

        # 2. Container loader test
        try:
            load_container(target)
        except STRUCTURED_ERRORS:
            pass

        # 3. Unpack test
        try:
            unpack_binary(target)
        except STRUCTURED_ERRORS:
            pass

        # 4. If metadata format, test IL2CPP metadata reader
        if "metadata" in target_ext:
            try:
                Il2CppMetadataReader.read_binary(target)
            except STRUCTURED_ERRORS:
                pass
