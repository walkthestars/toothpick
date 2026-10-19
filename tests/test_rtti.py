"""P3 RTTI tests: MSVC vtable/COL/TD/CHD/BCD walks, demangle, bounds, store.

Binding contract: ``docs/units/UNIT-P3-RTTI-PLAN.md`` section 1 (RTTI module:
MSVC 32/64-bit ABI -- x86 absolute links and a 4-byte ``vftable[-1]`` slot, x64
RVA links with an 8-byte slot and a 16-byte type-descriptor header; a
bounds-checked reader that refuses rather than degrade; ``store``/``get_class``
on the ``types`` table with replace-per-sha semantics) and section 2 (interface
specification).  Risk R1 bounds emission (default 50 scanned vtables, ceiling
1000) and is asserted here together with its explicit cap reason.

The traps this suite closes on purpose:

* **Trap 7** -- every SQLite handle opened by a test (module, ``Database``,
  raw connection) is closed and the tmp db file is removed, so a leaked WAL
  handle cannot surface on Windows teardown as ``PermissionError: [WinError 32]``.
* **Trap 11** -- no ``faulthandler`` plugin anywhere; ``pyproject.toml`` pins
  ``-p no:faulthandler`` and the last test asserts it stays pinned.

Every image below is a *synthetic* byte buffer assembled with
:func:`struct.pack_into` at explicit ABI offsets: no real binary is read and no
address is invented, which is the epistemic rule the module is built on -- a
recovered class row is only ever what was placed in the buffer.
"""

from __future__ import annotations

import sqlite3
import struct
from pathlib import Path

import pytest

from toothpick.rtti import (
    DEFAULT_MAX_VTABLES,
    MAX_BASE_CLASSES,
    MAX_TYPE_NAME,
    MAX_VTABLES,
    BaseRef,
    ClassRecord,
    RttiArch,
    RttiError,
    ScanResult,
    class_type_id,
    demangle_msvc_type,
    get_class,
    read_base_class_descriptor,
    read_class_hierarchy_descriptor,
    read_complete_object_locator,
    read_type_descriptor,
    resolve_max_vtables,
    scan_vtables,
    store,
)
from toothpick.storage import Database, TypeRecord

X64_BASE = 0x140000000
X86_BASE = 0x00400000
IMAGE_SIZE = 0x1000

SHA_A = "a" * 64
SHA_B = "b" * 64

# --- MSVC structure field offsets (the ABI the images below are laid out in) --- #
COL_SIGNATURE = 0
COL_OFFSET = 4
COL_CD_OFFSET = 8
COL_TD_LINK = 12
COL_CHD_LINK = 16
COL_SELF_RVA = 20

CHD_SIGNATURE = 0
CHD_ATTRIBUTES = 4
CHD_NUM_BASES = 8
CHD_ARRAY_LINK = 12

BCD_TD_LINK = 0
BCD_NUM_CONTAINED = 4
BCD_MDISP = 8
BCD_PDISP = 12
BCD_VDISP = 16
BCD_ATTRIBUTES = 20
BCD_CHD_LINK = 24

TD_HEADER_X86 = 8
TD_HEADER_X64 = 16

# --- x64 minimal image: one class (Foo : Bar), every link an RVA --- #
X64_MIN_IMAGE_SIZE = 0x200
X64_MIN_SLOT = 0x08  # vftable[-1] lives at vtable - 8
X64_MIN_VTABLE = 0x10
X64_MIN_COL = 0x20
X64_MIN_CHD = 0x40
X64_MIN_ARRAY = 0x50
X64_MIN_BCD = 0x60
X64_MIN_TD = 0x80
X64_MIN_BASE_TD = 0xA0

# --- x64 rich image: Foo (Bar, Baz) plus Bar (Baz), two scanned vtables --- #
X64_CHD = 0x100
X64_CHD_ARRAY = 0x110
X64_BAR_BCD = 0x120
X64_BAZ_BCD = 0x140
X64_FOO_TD = 0x160
X64_BAR_CHD = 0x180
X64_BAR_TD = 0x190
X64_BAZ_TD = 0x1B0
X64_FOO_COL = 0x1E0
X64_BAR_COL = 0x200
X64_FOO_SLOT = 0x238
X64_FOO_VTABLE = 0x240
X64_BAR_SLOT = 0x258
X64_BAR_VTABLE = 0x260

# --- x86 rich image: the same hierarchy with absolute virtual addresses --- #
X86_CHD = 0x100
X86_ARRAY = 0x110
X86_BAR_BCD = 0x120
X86_BAZ_BCD = 0x140
X86_FOO_TD = 0x160
X86_BAR_TD = 0x190
X86_BAZ_TD = 0x1B0
X86_COL = 0x1E0
X86_SLOT = 0x23C  # vftable[-1] lives at vtable - 4
X86_VTABLE = 0x240


# --------------------------------------------------------------------------- #
# synthetic image builders (``struct.pack_into`` only)
# --------------------------------------------------------------------------- #


def _u32(image: bytearray, off: int, value: int) -> None:
    struct.pack_into("<I", image, off, value)


def _s32(image: bytearray, off: int, value: int) -> None:
    struct.pack_into("<i", image, off, value)


def _u64(image: bytearray, off: int, value: int) -> None:
    struct.pack_into("<Q", image, off, value)


def _cstr(image: bytearray, off: int, text: str) -> None:
    raw = text.encode("ascii") + b"\x00"
    struct.pack_into(f"<{len(raw)}s", image, off, raw)


def _td(image: bytearray, off: int, header: int, text: str) -> None:
    """TypeDescriptor: ``header`` bytes (pVFTable + spare) then the name."""
    _cstr(image, off + header, text)


def _chd(image: bytearray, off: int, num_bases: int, array_link: int) -> None:
    _u32(image, off + CHD_SIGNATURE, 0)
    _u32(image, off + CHD_ATTRIBUTES, 0)
    _u32(image, off + CHD_NUM_BASES, num_bases)
    _u32(image, off + CHD_ARRAY_LINK, array_link)


def _bcd(
    image: bytearray,
    off: int,
    td_link: int,
    *,
    mdisp: int,
    contained: int,
    chd_link: int,
) -> None:
    _u32(image, off + BCD_TD_LINK, td_link)
    _u32(image, off + BCD_NUM_CONTAINED, contained)
    _s32(image, off + BCD_MDISP, mdisp)
    _s32(image, off + BCD_PDISP, -1)
    _s32(image, off + BCD_VDISP, 0)
    _u32(image, off + BCD_ATTRIBUTES, 0)
    _u32(image, off + BCD_CHD_LINK, chd_link)


def _col(
    image: bytearray,
    off: int,
    *,
    signature: int,
    td_link: int,
    chd_link: int,
    self_rva: int | None = None,
) -> None:
    _u32(image, off + COL_SIGNATURE, signature)
    _u32(image, off + COL_OFFSET, 0)
    _u32(image, off + COL_CD_OFFSET, 0)
    _u32(image, off + COL_TD_LINK, td_link)
    _u32(image, off + COL_CHD_LINK, chd_link)
    if self_rva is not None:
        _u32(image, off + COL_SELF_RVA, self_rva)


def _x64_min_image() -> bytearray:
    """One x64 class (``Foo`` : ``Bar``); every structure link is an RVA."""
    image = bytearray(X64_MIN_IMAGE_SIZE)
    _u64(image, X64_MIN_SLOT, X64_BASE + X64_MIN_COL)  # vftable[-1]: absolute
    _col(
        image,
        X64_MIN_COL,
        signature=1,
        td_link=X64_MIN_TD,
        chd_link=X64_MIN_CHD,
        self_rva=X64_MIN_COL,
    )
    _chd(image, X64_MIN_CHD, 1, X64_MIN_ARRAY)
    _u32(image, X64_MIN_ARRAY, X64_MIN_BCD)
    _bcd(
        image,
        X64_MIN_BCD,
        X64_MIN_BASE_TD,
        mdisp=0,
        contained=1,
        chd_link=X64_MIN_CHD,
    )
    _td(image, X64_MIN_TD, TD_HEADER_X64, ".?AVFoo@@")
    _td(image, X64_MIN_BASE_TD, TD_HEADER_X64, ".?AUBar@@")
    return image


def _x64_image() -> bytearray:
    """Two x64 classes: ``Foo`` (bases ``Bar``, ``Baz``) and ``Bar`` (base ``Baz``)."""
    image = bytearray(IMAGE_SIZE)
    _chd(image, X64_CHD, 2, X64_CHD_ARRAY)
    _u32(image, X64_CHD_ARRAY, X64_BAR_BCD)
    _u32(image, X64_CHD_ARRAY + 4, X64_BAZ_BCD)
    _bcd(
        image,
        X64_BAR_BCD,
        X64_BAR_TD,
        mdisp=0,
        contained=1,
        chd_link=X64_CHD,
    )
    _bcd(
        image,
        X64_BAZ_BCD,
        X64_BAZ_TD,
        mdisp=8,
        contained=0,
        chd_link=X64_CHD,
    )
    _td(image, X64_FOO_TD, TD_HEADER_X64, ".?AVFoo@@")
    _td(image, X64_BAR_TD, TD_HEADER_X64, ".?AUBar@@")
    _td(image, X64_BAZ_TD, TD_HEADER_X64, ".?AUBaz@@")
    _col(
        image,
        X64_FOO_COL,
        signature=1,
        td_link=X64_FOO_TD,
        chd_link=X64_CHD,
        self_rva=X64_FOO_COL,
    )
    _chd(image, X64_BAR_CHD, 1, X64_CHD_ARRAY + 4)  # Bar's only base is Baz
    _col(
        image,
        X64_BAR_COL,
        signature=1,
        td_link=X64_BAR_TD,
        chd_link=X64_BAR_CHD,
        self_rva=X64_BAR_COL,
    )
    _u64(image, X64_FOO_SLOT, X64_BASE + X64_FOO_COL)
    _u64(image, X64_BAR_SLOT, X64_BASE + X64_BAR_COL)
    return image


def _x86_image() -> bytearray:
    """The same hierarchy on x86: every COL/CHD/BCD link is an absolute VA."""
    image = bytearray(IMAGE_SIZE)
    _chd(image, X86_CHD, 2, X86_BASE + X86_ARRAY)
    _u32(image, X86_ARRAY, X86_BASE + X86_BAR_BCD)
    _u32(image, X86_ARRAY + 4, X86_BASE + X86_BAZ_BCD)
    _bcd(
        image,
        X86_BAR_BCD,
        X86_BASE + X86_BAR_TD,
        mdisp=0,
        contained=1,
        chd_link=X86_BASE + X86_CHD,
    )
    _bcd(
        image,
        X86_BAZ_BCD,
        X86_BASE + X86_BAZ_TD,
        mdisp=8,
        contained=0,
        chd_link=X86_BASE + X86_CHD,
    )
    _td(image, X86_FOO_TD, TD_HEADER_X86, ".?AVFoo@@")
    _td(image, X86_BAR_TD, TD_HEADER_X86, ".?AUBar@@")
    _td(image, X86_BAZ_TD, TD_HEADER_X86, ".?AUBaz@@")
    _col(
        image,
        X86_COL,
        signature=0,
        td_link=X86_BASE + X86_FOO_TD,
        chd_link=X86_BASE + X86_CHD,
    )
    _u32(image, X86_SLOT, X86_BASE + X86_COL)
    return image


def _scan(image: bytes | bytearray, base: int, arch: str, vtable: int) -> ScanResult:
    """Shorthand: scan one candidate vtable of a synthetic image."""
    return scan_vtables(image, base, arch, [vtable])


# --------------------------------------------------------------------------- #
# full walks: x86 absolute links, x64 RVA links
# --------------------------------------------------------------------------- #


def test_x86_full_walk_recovers_name_bases_and_mdisp() -> None:
    image = _x86_image()
    vtable = X86_BASE + X86_VTABLE
    result = scan_vtables(image, X86_BASE, "x86", [vtable])

    assert isinstance(result, ScanResult)
    assert result.rejected == []
    assert len(result.classes) == 1
    record = result.classes[0]
    assert (record.mangled, record.name, record.demangled) == (".?AVFoo@@", "Foo", True)
    assert record.arch == "x86"
    assert record.vtable_addr == vtable
    assert record.col_addr == X86_BASE + X86_COL
    assert record.td_addr == X86_BASE + X86_FOO_TD
    assert record.chd_addr == X86_BASE + X86_CHD
    # BaseClassDescriptors are walked in array order; mdisp is signed evidence.
    assert [(b.mangled, b.name, b.mdisp) for b in record.bases] == [
        (".?AUBar@@", "Bar", 0),
        (".?AUBaz@@", "Baz", 8),
    ]
    # An explicit RttiArch spelling is the same call as the string form.
    assert scan_vtables(image, X86_BASE, RttiArch.X86, [vtable]) == result
    # The ABI is a caller contract, never inferred from bytes: the x86 image read
    # as x64 (8-byte slot, RVA links) must refuse instead of inventing a class.
    as_x64 = _scan(image, X86_BASE, "x64", vtable)
    assert as_x64.classes == []
    assert len(as_x64.rejected) == 1 and as_x64.rejected[0][0] == vtable
    assert RttiArch.coerce("AMD64") is RttiArch.X64
    assert RttiArch.coerce("i386") is RttiArch.X86
    with pytest.raises(RttiError):
        RttiArch.coerce("arm64")


def test_x64_rva_walk_recovers_two_classes_and_a_base_chain() -> None:
    image = _x64_image()
    foo_vtable = X64_BASE + X64_FOO_VTABLE
    bar_vtable = X64_BASE + X64_BAR_VTABLE
    result = scan_vtables(image, X64_BASE, "x64", [foo_vtable, bar_vtable])

    assert result.rejected == []
    assert [c.mangled for c in result.classes] == [".?AVFoo@@", ".?AUBar@@"]
    foo, bar = result.classes

    assert (foo.mangled, foo.name, foo.demangled) == (".?AVFoo@@", "Foo", True)
    assert foo.arch == "x64"
    assert (foo.vtable_addr, foo.col_addr) == (foo_vtable, X64_BASE + X64_FOO_COL)
    assert (foo.td_addr, foo.chd_addr) == (X64_BASE + X64_FOO_TD, X64_BASE + X64_CHD)
    assert [(b.mangled, b.name, b.mdisp) for b in foo.bases] == [
        (".?AUBar@@", "Bar", 0),
        (".?AUBaz@@", "Baz", 8),
    ]

    assert (bar.mangled, bar.name, bar.demangled) == (".?AUBar@@", "Bar", True)
    assert (bar.vtable_addr, bar.col_addr) == (bar_vtable, X64_BASE + X64_BAR_COL)
    assert (bar.td_addr, bar.chd_addr) == (X64_BASE + X64_BAR_TD, X64_BASE + X64_BAR_CHD)
    # Bar's own CHD reuses array entry [1], so its single base is Baz at mdisp 8.
    assert [(b.mangled, b.name, b.mdisp) for b in bar.bases] == [(".?AUBaz@@", "Baz", 8)]

    # x64 COL links are RVAs: the raw dword is an offset from the image base.
    locator = read_complete_object_locator(image, X64_BASE, "x64", X64_BASE + X64_FOO_COL)
    assert locator["self_rva"] == X64_FOO_COL
    assert locator["self"] == X64_BASE + X64_FOO_COL
    assert locator["signature"] == 1

    # The same bytes read as x86 (4-byte slot + absolute links) are refused.
    as_x86 = _scan(image, X64_BASE, "x86", foo_vtable)
    assert as_x86.classes == []
    assert len(as_x86.rejected) == 1 and as_x86.rejected[0][0] == foo_vtable


def test_structure_readers_return_evidence_dicts_in_bounds() -> None:
    image = _x64_min_image()
    locator = read_complete_object_locator(image, X64_BASE, "x64", X64_BASE + X64_MIN_COL)
    assert locator == {
        "col_addr": X64_BASE + X64_MIN_COL,
        "signature": 1,
        "offset": 0,
        "cd_offset": 0,
        "type_descriptor_link": X64_MIN_TD,
        "class_hierarchy_link": X64_MIN_CHD,
        "self_rva": X64_MIN_COL,
        "self": X64_BASE + X64_MIN_COL,
        "type_descriptor": X64_BASE + X64_MIN_TD,
        "class_hierarchy": X64_BASE + X64_MIN_CHD,
        "arch": "x64",
    }
    assert read_type_descriptor(image, X64_BASE, "x64", X64_BASE + X64_MIN_TD) == {
        "td_addr": X64_BASE + X64_MIN_TD,
        "mangled": ".?AVFoo@@",
        "name": "Foo",
        "demangled": True,
        "td_name_offset": 16,
        "arch": "x64",
    }
    chd = read_class_hierarchy_descriptor(image, X64_BASE, "x64", X64_BASE + X64_MIN_CHD)
    assert chd["num_base_classes"] == 1
    assert (chd["base_class_array_link"], chd["base_class_array"]) == (
        X64_MIN_ARRAY,
        X64_BASE + X64_MIN_ARRAY,
    )
    assert chd["signature"] == 0 and chd["attributes"] == 0
    bcd = read_base_class_descriptor(image, X64_BASE, "x64", X64_BASE + X64_MIN_BCD)
    assert (bcd["mangled"], bcd["name"], bcd["demangled"]) == (".?AUBar@@", "Bar", True)
    assert bcd["type_descriptor"] == X64_BASE + X64_MIN_BASE_TD
    assert (bcd["num_contained_bases"], bcd["mdisp"]) == (1, 0)
    assert (bcd["pdisp"], bcd["vdisp"]) == (-1, 0)
    assert bcd["class_hierarchy"] == X64_BASE + X64_MIN_CHD
    assert class_type_id(bcd["mangled"]) == "rtti:.?AUBar@@"


# --------------------------------------------------------------------------- #
# name recovery
# --------------------------------------------------------------------------- #


def test_demangle_av_au_ok_and_raw_fallback() -> None:
    assert demangle_msvc_type(".?AVFoo@@") == ("Foo", True)
    assert demangle_msvc_type(".?AUBar@@") == ("Bar", True)
    # Nested scope/template decorations stay verbatim; only the terminator goes.
    assert demangle_msvc_type(".?AV?$vec@VFoo@@@@") == ("?$vec@VFoo@@", True)
    for raw in (
        ".?AW4Colour@@",  # an encoding this parser does not claim to know
        "?AVFoo@@",  # missing the leading dot
        ".?AVFoo@",  # missing the terminator
        ".?AV@@",  # empty body
        "",
        "plain_name",
    ):
        assert demangle_msvc_type(raw) == (raw, False)
    with pytest.raises(RttiError):
        demangle_msvc_type(None)

    # End to end: an unrecognised descriptor is stored raw with demangled=False,
    # never presented as a guessed class name.
    image = _x64_min_image()
    _cstr(image, X64_MIN_TD + TD_HEADER_X64, ".?AW4Colour@@")
    result = _scan(image, X64_BASE, "x64", X64_BASE + X64_MIN_VTABLE)
    assert result.rejected == []
    record = result.classes[0]
    assert record.mangled == ".?AW4Colour@@"
    assert record.name == ".?AW4Colour@@"
    assert record.demangled is False


# --------------------------------------------------------------------------- #
# bounds, truncation, malformed descriptors
# --------------------------------------------------------------------------- #


def test_truncated_buffer_is_refused_without_partial_rows() -> None:
    image = _x64_min_image()
    vtable = X64_BASE + X64_MIN_VTABLE
    full = _scan(image, X64_BASE, "x64", vtable)
    assert len(full.classes) == 1 and full.rejected == []

    # Each cut length lands inside a different step of the walk (the vftable[-1]
    # slot, the 24-byte COL, the 16-byte TD header, the TD name), and every one
    # refuses the candidate while emitting no partial class row.  The needles are
    # quoted from the module's own bounds reports.
    for cut, needle in (
        (0x0C, "vftable[-1] CompleteObjectLocator pointer"),  # slot itself is cut off
        (0x2C, "CompleteObjectLocator"),  # 24-byte COL not fully present
        (0x7C, "COL.type_descriptor"),  # the TD is past the end of the buffer
        (0x8C, "TypeDescriptor header"),  # 16-byte TD header not fully present
        (0x90, "TypeDescriptor name"),  # the name starts exactly at the end
        (0x95, "unterminated string"),  # TD name has no NUL inside the buffer
    ):
        short = bytes(image[:cut])
        result = _scan(short, X64_BASE, "x64", vtable)
        assert result.classes == [], f"cut 0x{cut:x} emitted a partial row"
        assert len(result.rejected) == 1
        addr, reason = result.rejected[0]
        assert addr == vtable
        assert needle in reason, reason

    # A link that resolves inside the buffer but whose target runs past the end
    # is refused too: the structure is never half-read.
    chd_cut = _x64_min_image()
    _u32(chd_cut, X64_MIN_COL + COL_CHD_LINK, X64_MIN_IMAGE_SIZE - 8)
    result = _scan(chd_cut, X64_BASE, "x64", vtable)
    assert result.classes == []
    assert "ClassHierarchyDescriptor" in result.rejected[0][1]

    # The BCD is reached last (COL -> TD -> CHD -> array -> BCD): aiming the array
    # entry at the tail of the buffer makes the 28-byte descriptor illegible, so
    # no base row is emitted either.  On x64 a canonical (+12) array walk that
    # fails is retried at +24 (the HOI4-evidence fallback), so the refusal may be
    # reported one link earlier -- naming the fallback link that failed instead
    # of the descriptor behind it.  Both reasons name the structure that actually
    # failed, so either is accepted; the invariants (no partial row, exactly one
    # rejection, the candidate address) are unchanged.
    bcd_cut = _x64_min_image()
    _u32(bcd_cut, X64_MIN_ARRAY, X64_MIN_IMAGE_SIZE - 8)
    result = _scan(bcd_cut, X64_BASE, "x64", vtable)
    assert result.classes == []
    assert len(result.rejected) == 1
    addr, reason = result.rejected[0]
    assert addr == vtable
    assert "BaseClassDescriptor" in reason or "CHD.base_class_array (+24)" in reason, reason

    array_cut = _x64_min_image()
    _u32(array_cut, X64_MIN_CHD + CHD_ARRAY_LINK, X64_MIN_IMAGE_SIZE - 2)
    result = _scan(array_cut, X64_BASE, "x64", vtable)
    assert result.classes == []
    assert "CHD.base_class_array contents" in result.rejected[0][1]

    # A short buffer also refuses the individual readers -- no silent zeros.
    tiny = bytes(image[:0x10])
    outside = X64_BASE + 0x1000
    for reader in (
        read_complete_object_locator,
        read_type_descriptor,
        read_class_hierarchy_descriptor,
        read_base_class_descriptor,
    ):
        with pytest.raises(RttiError):
            reader(tiny, X64_BASE, "x64", outside)


def test_col_with_bad_signature_is_rejected_not_guessed() -> None:
    image = _x64_min_image()
    vtable = X64_BASE + X64_MIN_VTABLE
    _u32(image, X64_MIN_COL + COL_SIGNATURE, 2)

    result = _scan(image, X64_BASE, "x64", vtable)
    assert result.classes == []
    assert len(result.rejected) == 1
    addr, reason = result.rejected[0]
    assert addr == vtable
    assert "signature 0x2" in reason
    with pytest.raises(RttiError):
        read_complete_object_locator(image, X64_BASE, "x64", X64_BASE + X64_MIN_COL)

    # Signature 0 is an accepted (absolute-pointer-era) form: the signature is
    # recorded but never used to pick a resolution rule -- the arch decides.
    accepted = _x64_min_image()
    _u32(accepted, X64_MIN_COL + COL_SIGNATURE, 0)
    ok = _scan(accepted, X64_BASE, "x64", vtable)
    assert ok.rejected == []
    assert ok.classes[0].mangled == ".?AVFoo@@"
    assert ok.classes[0].col_addr == X64_BASE + X64_MIN_COL


def test_null_links_are_refused() -> None:
    vtable = X64_BASE + X64_MIN_VTABLE

    td_null = _x64_min_image()
    _u32(td_null, X64_MIN_COL + COL_TD_LINK, 0)
    result = _scan(td_null, X64_BASE, "x64", vtable)
    assert result.classes == []
    assert "null RVA link" in result.rejected[0][1]

    chd_null = _x64_min_image()
    _u32(chd_null, X64_MIN_COL + COL_CHD_LINK, 0)
    result = _scan(chd_null, X64_BASE, "x64", vtable)
    assert result.classes == []
    assert "null RVA link" in result.rejected[0][1]

    array_null = _x64_min_image()
    _u32(array_null, X64_MIN_CHD + CHD_ARRAY_LINK, 0)
    result = _scan(array_null, X64_BASE, "x64", vtable)
    assert result.classes == []
    assert "null RVA link" in result.rejected[0][1]

    # A zero link on x86 is the address 0, which is outside the supplied image:
    # a refusal too, just a different (honest) reason.
    x86 = _x86_image()
    _u32(x86, X86_COL + COL_TD_LINK, 0)
    result = _scan(x86, X86_BASE, "x86", X86_BASE + X86_VTABLE)
    assert result.classes == []
    assert "outside the supplied buffer" in result.rejected[0][1]


def test_descriptor_bounds_and_unterminated_names_are_refused() -> None:
    vtable = X64_BASE + X64_MIN_VTABLE

    over = _x64_min_image()
    _u32(over, X64_MIN_CHD + CHD_NUM_BASES, MAX_BASE_CLASSES + 1)
    result = _scan(over, X64_BASE, "x64", vtable)
    assert result.classes == []
    assert f"ceiling {MAX_BASE_CLASSES}" in result.rejected[0][1]

    empty = _x64_min_image()
    _u32(empty, X64_MIN_CHD + CHD_NUM_BASES, 0)
    result = _scan(empty, X64_BASE, "x64", vtable)
    assert result.classes == []
    assert "no base classes" in result.rejected[0][1]

    # A name with no NUL within MAX_TYPE_NAME bytes is refused, never truncated.
    blob = bytearray(b"A" * (TD_HEADER_X64 + MAX_TYPE_NAME))
    with pytest.raises(RttiError) as excinfo:
        read_type_descriptor(blob, 0, "x64", 0)
    assert "unterminated string" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# persistence: store / get_class / base rows / replace-per-sha
# --------------------------------------------------------------------------- #


def test_store_and_get_class_round_trip_exposes_queryable_base_rows(
    tmp_path: Path,
) -> None:
    image = _x64_image()
    result = scan_vtables(
        image,
        X64_BASE,
        "x64",
        [X64_BASE + X64_FOO_VTABLE, X64_BASE + X64_BAR_VTABLE],
    )
    assert [c.mangled for c in result.classes] == [".?AVFoo@@", ".?AUBar@@"]

    db_path = tmp_path / "rtti.db"
    out = store(db_path, SHA_A, result.classes, arch="x64")
    assert (out["classes"], out["bases"], out["updated"]) == (2, 1, 3)
    assert out["replaced"] == 0
    assert out["binary_sha256"] == SHA_A
    assert out["type_ids"] == ["rtti:.?AUBar@@", "rtti:.?AUBaz@@", "rtti:.?AVFoo@@"]

    foo = get_class(db_path, SHA_A, "Foo")
    assert foo is not None
    assert foo["type_id"] == "rtti:.?AVFoo@@"
    assert foo["name"] == "Foo" and foo["kind"] == "class"
    assert foo["provenance"] == "msvc-rtti" and foo["inferred"] is False
    assert foo["arch"] == "x64" and foo["lattice_rank"] == 3
    assert foo["vtable_addr"] == X64_BASE + X64_FOO_VTABLE
    assert foo["col_addr"] == X64_BASE + X64_FOO_COL
    assert foo["td_addr"] == X64_BASE + X64_FOO_TD
    assert foo["chd_addr"] == X64_BASE + X64_CHD
    assert foo["binary_sha256"] == SHA_A
    # A scanned parent (Bar) stays non-inferred; the referenced-only parent (Baz)
    # is a row of its own -- that is why the hierarchy is queryable.
    assert foo["bases"] == [
        {
            "mangled": ".?AUBar@@",
            "name": "Bar",
            "mdisp": 0,
            "inferred": False,
            "known": True,
        },
        {
            "mangled": ".?AUBaz@@",
            "name": "Baz",
            "mdisp": 8,
            "inferred": True,
            "known": True,
        },
    ]
    assert foo["base_type_ids"] == [
        class_type_id(".?AUBar@@"),
        class_type_id(".?AUBaz@@"),
    ]

    # Demangled name, mangled name and full type id resolve to the same row.
    assert get_class(db_path, SHA_A, ".?AVFoo@@") == foo
    assert get_class(db_path, SHA_A, "rtti:.?AVFoo@@") == foo
    assert get_class(db_path, SHA_A, "Nope") is None

    bar = get_class(db_path, SHA_A, "Bar")
    assert bar is not None
    assert bar["type_id"] == "rtti:.?AUBar@@"
    assert bar["vtable_addr"] == X64_BASE + X64_BAR_VTABLE
    assert bar["inferred"] is False and bar["referenced_by"] == []
    assert bar["bases"] == [
        {
            "mangled": ".?AUBaz@@",
            "name": "Baz",
            "mdisp": 8,
            "inferred": True,
            "known": True,
        }
    ]

    baz = get_class(db_path, SHA_A, "Baz")
    assert baz is not None
    assert baz["type_id"] == "rtti:.?AUBaz@@"
    assert baz["kind"] == "class" and baz["name"] == "Baz"
    assert baz["inferred"] is True and baz["mangled"] == ".?AUBaz@@"
    # Both scanned classes name Baz as a base; the join reports referrers in a
    # stable (sorted) order, so two scans of the same image compare equal.
    assert baz["referenced_by"] == sorted([".?AVFoo@@", ".?AUBar@@"]) == [
        ".?AUBar@@",
        ".?AVFoo@@",
    ]

    # The rows are real lattice rows: the storage layer sees them too.
    with Database(db_path) as db:
        rows = {t.type_id: t for t in db.get_types(SHA_A)}
        assert db.count_types(SHA_A) == 3
    assert sorted(rows) == ["rtti:.?AUBar@@", "rtti:.?AUBaz@@", "rtti:.?AVFoo@@"]
    assert rows["rtti:.?AVFoo@@"].kind == "class"
    assert rows["rtti:.?AVFoo@@"].layout["hierarchy"]["mdisp"] == {
        ".?AUBar@@": 0,
        ".?AUBaz@@": 8,
    }
    assert rows["rtti:.?AUBaz@@"].layout["inferred"] is True

    db_path.unlink()  # Trap 7: no handle outlives the test


def test_store_targets_database_path_and_connection_are_closed_trap7(
    tmp_path: Path,
) -> None:
    record = _scan(_x64_min_image(), X64_BASE, "x64", X64_BASE + X64_MIN_VTABLE).classes[0]
    assert isinstance(record, ClassRecord)

    # A Database target: the store owns its handle, the caller closes it.
    db_path = tmp_path / "handles" / "store.db"
    db = Database(db_path)
    try:
        assert store(db, SHA_A, [record])["updated"] == 2  # one class + one base
        assert get_class(db, SHA_A, "Foo")["name"] == "Foo"
    finally:
        db.close()
    assert db.closed is True
    db_path.unlink()  # PermissionError [WinError 32] if any handle leaked

    # A raw connection target: the caller's handle and state stay the caller's.
    conn_path = tmp_path / "handles" / "conn.db"
    conn = sqlite3.connect(str(conn_path), isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        assert store(conn, SHA_A, [record])["classes"] == 1
        assert get_class(conn, SHA_A, "Foo")["type_id"] == "rtti:.?AVFoo@@"
        assert conn.row_factory is sqlite3.Row
        assert conn.isolation_level is None
        assert conn.in_transaction is False
    finally:
        conn.close()
    conn_path.unlink()

    # A path target bootstraps the types table and releases its own handle.
    path_target = tmp_path / "handles" / "fresh.db"
    assert store(path_target, SHA_A, [record], create=True)["classes"] == 1
    assert get_class(path_target, SHA_A, "Foo")["mangled"] == ".?AVFoo@@"
    assert get_class(path_target, SHA_A, "Bar")["inferred"] is True
    path_target.unlink()


def test_store_replaces_per_sha_and_rescan_converges(tmp_path: Path) -> None:
    image = _x64_image()
    classes = scan_vtables(
        image,
        X64_BASE,
        "x64",
        [X64_BASE + X64_FOO_VTABLE, X64_BASE + X64_BAR_VTABLE],
    ).classes
    db_path = tmp_path / "replace.db"

    with Database(db_path) as db:
        # A non-RTTI lattice row the rtti namespace must never touch.
        db.upsert_type(
            TypeRecord(
                type_id="struct:Widget",
                name="Widget",
                kind="struct",
                binary_sha256=SHA_A,
            )
        )
        first = store(db, SHA_A, classes, arch="x64")
        assert (first["classes"], first["bases"], first["replaced"]) == (2, 1, 0)
        assert db.count_types(SHA_A) == 4

        before = {t.type_id: t.layout for t in db.get_types(SHA_A)}
        again = store(db, SHA_A, classes, arch="x64")
        assert again["replaced"] == 3  # every previous rtti row was reclaimed
        assert again["type_ids"] == first["type_ids"]
        assert db.count_types(SHA_A) == 4  # replaced, not accumulated
        after = {t.type_id: t.layout for t in db.get_types(SHA_A)}
        assert after == before  # a rescan converges to identical rows

        # Another binary's rows are untouched by this sha's replace.
        assert store(db, SHA_B, classes, arch="x64")["updated"] == 3
        assert db.count_types(SHA_B) == 3
        assert store(db, SHA_A, classes, arch="x64")["replaced"] == 3
        assert db.count_types(SHA_B) == 3 and db.count_types(SHA_A) == 4

        widget = db.get_type("struct:Widget", SHA_A)
        assert widget is not None and widget.kind == "struct"
        assert widget.layout is None

    # The same discipline through a db path: converges, keeps both shas apart.
    out_a = store(db_path, SHA_A, classes, arch="x64")
    out_b = store(db_path, SHA_A, classes, arch="x64")
    assert out_a["updated"] == out_b["updated"] == 3
    assert out_a["classes"] == out_b["classes"] == 2
    assert out_a["type_ids"] == out_b["type_ids"]
    assert get_class(db_path, SHA_B, "Foo")["binary_sha256"] == SHA_B
    assert get_class(db_path, SHA_A, "Foo")["binary_sha256"] == SHA_A

    db_path.unlink()


def test_store_and_get_class_refuse_bad_input(tmp_path: Path) -> None:
    image = _x64_min_image()
    record = scan_vtables(image, X64_BASE, "x64", [X64_BASE + X64_MIN_VTABLE]).classes[0]
    db_path = tmp_path / "refuse.db"
    store(db_path, SHA_A, [record], arch="x64")

    for bad in ("", None, 7, True, "   "):
        with pytest.raises(RttiError):
            store(db_path, bad, [record])
        with pytest.raises(RttiError):
            get_class(db_path, bad, "Foo")
    with pytest.raises(RttiError):
        store(db_path, SHA_A, [BaseRef(mangled=".?AUBar@@")])
    with pytest.raises(RttiError):
        store(db_path, SHA_A, [object()])
    with pytest.raises(RttiError):
        store(db_path, SHA_A, [ClassRecord(mangled="", name="broken")])
    with pytest.raises(RttiError):
        get_class(db_path, SHA_A, 7)
    with pytest.raises(RttiError):
        get_class(db_path, SHA_A, "   ")

    # A read against a store without a types table returns None, creates nothing.
    bare = tmp_path / "bare.db"
    sqlite3.connect(str(bare)).close()
    assert get_class(bare, SHA_A, "Foo") is None
    check = sqlite3.connect(str(bare))
    try:
        assert (
            check.execute("SELECT 1 FROM sqlite_master WHERE name = 'types'").fetchone()
            is None
        )
    finally:
        check.close()

    # A typo'd db path is refused, never silently created.
    missing = tmp_path / "absent.db"
    with pytest.raises(RttiError) as excinfo:
        get_class(missing, SHA_A, "Foo")
    assert "database not found" in str(excinfo.value)
    assert not missing.exists()

    db_path.unlink()
    bare.unlink()


# --------------------------------------------------------------------------- #
# R1: bounded emission
# --------------------------------------------------------------------------- #


def test_r1_cap_bounds_scan_and_reports_cap_reason() -> None:
    image = _x64_image()
    valid = [X64_BASE + X64_FOO_VTABLE, X64_BASE + X64_BAR_VTABLE]

    assert resolve_max_vtables() == DEFAULT_MAX_VTABLES == 50
    assert resolve_max_vtables(None) == DEFAULT_MAX_VTABLES
    assert resolve_max_vtables(2) == 2
    assert resolve_max_vtables(10_000) == MAX_VTABLES == 1000
    for bad in (0, -1, "many", True):
        with pytest.raises(RttiError):
            resolve_max_vtables(bad)

    # Exact duplicates are dropped before the cap is applied.
    deduped = scan_vtables(image, X64_BASE, "x64", valid + valid)
    assert len(deduped.classes) == 2 and deduped.rejected == []

    # A candidate beyond the budget is reported as *capped*, not silently cut.
    capped = scan_vtables(
        image, X64_BASE, "x64", valid + [X64_BASE + 0x900], max_vtables=2
    )
    assert len(capped.classes) == 2
    assert len(capped.rejected) == 1
    addr, reason = capped.rejected[0]
    assert addr == X64_BASE + 0x900
    assert "cap reached" in reason and "R1" in reason
    assert "2 of 3 candidates" in reason

    # The default budget (50) is enforced with the same explicit reason, and
    # nothing beyond it is parsed, let alone emitted as a class row.
    bogus = [X64_BASE + 0x1000 + 4 * i for i in range(60)]
    default = scan_vtables(image, X64_BASE, "x64", bogus)
    assert default.classes == []
    assert len(default.rejected) == 60
    capped_only = [(a, r) for a, r in default.rejected if "cap reached" in r]
    assert [a for a, _ in capped_only] == bogus[DEFAULT_MAX_VTABLES:]
    assert f"{DEFAULT_MAX_VTABLES} of 60 candidates" in capped_only[0][1]
    assert f"ceiling {MAX_VTABLES}" in capped_only[0][1]
    assert [a for a, r in default.rejected if "cap reached" not in r] == bogus[
        :DEFAULT_MAX_VTABLES
    ]

    # Over-large budgets are clamped, so a caller cannot ask for an unbounded walk.
    clamped = scan_vtables(image, X64_BASE, "x64", valid, max_vtables=10_000)
    assert len(clamped.classes) == 2 and clamped.rejected == []


# --------------------------------------------------------------------------- #
# trap 11: the suite keeps ``-p no:faulthandler``
# --------------------------------------------------------------------------- #


def test_suite_pins_no_faulthandler_addopt() -> None:
    config = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(
        encoding="utf-8"
    )
    assert "-p no:faulthandler" in config
