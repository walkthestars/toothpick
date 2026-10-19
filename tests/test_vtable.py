"""UNIT-VTABLE tests: vtable discovery (shape pass + COL confirmation).

Binding contract: ``docs/units/UNIT-VTABLE-PLAN.md`` (the POW-002 follow-up
unit).  The properties asserted here are the plan's three method steps and its
two policies:

* **shape pass** -- maximal runs of >= 3 pointer-sized slots whose every value
  resolves into a caller-supplied ``text_ranges`` executable interval and whose
  slots lie in a caller-supplied ``search_ranges`` interval, pointer-aligned;
* **confirmation** -- the value at ``candidate - pointer_size`` (the
  ``vftable[-1]`` slot) must decode as a validated COL through the certified
  :mod:`toothpick.rtti_parse` COL -> TD -> CHD walk, else the candidate is
  labelled ``shape-only`` and is **never promoted**;
* **bounds** -- every dereference bounds-checked (a bad image never crashes, a
  bad argument raises ``RttiError``), R1 caps (default 50, ceiling 1000), and a
  deterministic address order;
* **negative-result clause** -- a zero-confirmed catalogue is a *reported*
  result (shape-only rows labelled, criteria not loosened), so a test asserts
  exactly that, and that the summary carries the clause.

Every image below is a *synthetic* byte buffer assembled with
:func:`struct.pack_into` at explicit MSVC ABI offsets -- no real binary is read
and no address is invented.  A recovered identity is only ever what the test
itself placed in the buffer, which is the honesty rule the module is built on
(design §1 "no fabricated structs"); a shape-only row therefore carries an empty
name and the reader's refusal as its evidence.

Trap hygiene: no ``faulthandler`` plugin (Trap 11 -- ``pyproject.toml`` pins
``-p no:faulthandler``) and no SQLite handle at all, so Trap 7 cannot originate
here.
"""

from __future__ import annotations

import struct

import pytest

from toothpick.rtti_parse import RttiError
from toothpick.vtable import (
    CONFIDENCE_CONFIRMED,
    CONFIDENCE_SHAPE_ONLY,
    DEFAULT_MAX_VTABLES,
    MAX_ENTRIES,
    MAX_VTABLES,
    MIN_RUN_LENGTH,
    NEGATIVE_RESULT_NOTE,
    VtableCandidate,
    VtableScanResult,
    discover_vtables,
    entry_targets,
    hookable_addresses,
    summarize_discovery,
)

X64_BASE = 0x140000000
X86_BASE = 0x00400000

#: The one pointer value that counts as "executable" in these images.
X64_TEXT = X64_BASE + 0x1000
X86_TEXT = X86_BASE + 0x1000

#: An 8-byte executable window around the target: only the exact value counts,
#: so a run in a test is a run of *that* value and nothing else.
X64_TEXT_RANGE = (X64_TEXT, X64_TEXT + 1)
X86_TEXT_RANGE = (X86_TEXT, X86_TEXT + 1)

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

# --- x64 certified layout: class Foo : Bar, one COL -> TD -> CHD -> BCD walk -- #
X64_COL = 0x20
X64_CHD = 0x40
X64_ARRAY = 0x50
X64_BCD = 0x60
X64_TD = 0x80
X64_BASE_TD = 0xA0

# --- x86 certified layout: the same hierarchy with absolute VAs -------------- #
X86_COL = 0x20
X86_CHD = 0x40
X86_ARRAY = 0x50
X86_BCD = 0x60
X86_TD = 0x80
X86_BASE_TD = 0xA0


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
    contained: int,
    chd_link: int,
    mdisp: int = 0,
) -> None:
    _u32(image, off + BCD_TD_LINK, td_link)
    _u32(image, off + BCD_NUM_CONTAINED, contained)
    _s32(image, off + BCD_MDISP, mdisp)
    _s32(image, off + BCD_PDISP, -1)
    _s32(image, off + BCD_VDISP, 0)
    _u32(image, off + BCD_ATTRIBUTES, 0)
    _u32(image, off + BCD_CHD_LINK, chd_link)


def _x64_class_image(size: int = 0x400) -> bytearray:
    """The certified x64 class structures: ``Foo`` (base ``Bar``), links as RVAs.

    The image holds *no* shape run: each test places its own, so a candidate is
    always attributable to the bytes the test wrote.
    """
    image = bytearray(size)
    _col(image, X64_COL, signature=1, td_link=X64_TD, chd_link=X64_CHD, self_rva=X64_COL)
    _chd(image, X64_CHD, 1, X64_ARRAY)
    _u32(image, X64_ARRAY, X64_BCD)
    _bcd(image, X64_BCD, X64_BASE_TD, contained=1, chd_link=X64_CHD)
    _td(image, X64_TD, TD_HEADER_X64, ".?AVFoo@@")
    _td(image, X64_BASE_TD, TD_HEADER_X64, ".?AUBar@@")
    return image


def _x86_class_image(size: int = 0x400) -> bytearray:
    """The certified x86 class structures: every COL/CHD/BCD link an absolute VA."""
    image = bytearray(size)
    _col(
        image,
        X86_COL,
        signature=0,
        td_link=X86_BASE + X86_TD,
        chd_link=X86_BASE + X86_CHD,
    )
    _chd(image, X86_CHD, 1, X86_BASE + X86_ARRAY)
    _u32(image, X86_ARRAY, X86_BASE + X86_BCD)
    _bcd(
        image,
        X86_BCD,
        X86_BASE + X86_BASE_TD,
        contained=1,
        chd_link=X86_BASE + X86_CHD,
    )
    _td(image, X86_TD, TD_HEADER_X86, ".?AVFoo@@")
    _td(image, X86_BASE_TD, TD_HEADER_X86, ".?AUBar@@")
    return image


def _x64_text_run(image: bytearray, addr: int, slots: int, value: int = X64_TEXT) -> None:
    """A run of ``slots`` x64 text pointers starting at ``addr``."""
    for index in range(slots):
        _u64(image, addr + 8 * index, value)


def _x86_text_run(image: bytearray, addr: int, slots: int, value: int = X86_TEXT) -> None:
    """A run of ``slots`` x86 text pointers starting at ``addr``."""
    for index in range(slots):
        _u32(image, addr + 4 * index, value)


def _shape_only_runs(count: int, *, start: int = 0x1000, stride: int = 0x20) -> bytearray:
    """``count`` x64 shape runs with no COL link anywhere (all slots' -8 slot 0)."""
    size = start + count * stride + 0x40
    image = bytearray(size)
    for index in range(count):
        _x64_text_run(image, start + index * stride, 3)
    return image


class _Region:
    """A ``base``/``size`` region object, the shape ``toothpick.scanner`` uses."""

    def __init__(self, base: int, size: int) -> None:
        self.base = base
        self.size = size


def _x64_image_result(image: bytearray, *, max_vtables: int | None = None):
    return discover_vtables(
        bytes(image),
        X64_BASE,
        "x64",
        [X64_TEXT_RANGE],
        [(X64_BASE, X64_BASE + len(image))],
        max_vtables,
    )


# --------------------------------------------------------------------------- #
# Test 1-2: confirmed COL links (x64 RVA links, x86 absolute links)
# --------------------------------------------------------------------------- #


def test_x64_shape_run_with_validated_col_is_confirmed():
    """A 3-slot text run whose ``vftable[-1]`` is a validated COL is ``confirmed``."""
    image = _x64_class_image()
    _u64(image, 0xF8, X64_BASE + X64_COL)  # vftable[-1] -> COL
    _x64_text_run(image, 0x100, MIN_RUN_LENGTH)

    result = _x64_image_result(image)

    assert isinstance(result, VtableScanResult)
    assert result.confirmed_count == 1
    assert result.shape_only_count == 0
    assert result.rejected == []

    candidate = result.candidates[0]
    assert isinstance(candidate, VtableCandidate)
    assert candidate.addr == X64_BASE + 0x100
    assert candidate.confidence == CONFIDENCE_CONFIRMED
    assert candidate.hookable is True
    assert candidate.slot_count == MIN_RUN_LENGTH
    assert candidate.arch == "x64"
    # identity comes from the bytes the builder wrote -- nothing is invented
    assert candidate.mangled == ".?AVFoo@@"
    assert candidate.name == "Foo"
    assert candidate.demangled is True
    assert candidate.bases == (".?AUBar@@",)
    assert candidate.col_addr == X64_BASE + X64_COL
    assert candidate.col_pointer == X64_BASE + X64_COL
    assert candidate.td_addr == X64_BASE + X64_TD
    assert candidate.chd_addr == X64_BASE + X64_CHD
    assert candidate.reason == ""

    # only the confirmed row is ever offered as a hookable address
    assert hookable_addresses(result) == (X64_BASE + 0x100,)
    assert candidate.to_dict()["hookable"] is True


def test_x86_shape_run_uses_four_byte_slots_and_absolute_links():
    """x86: 4-byte slots, absolute COL pointer in ``vftable[-1]``, same confidence."""
    image = _x86_class_image()
    _u32(image, 0xFC, X86_BASE + X86_COL)  # vftable[-1] -> COL (absolute VA)
    _x86_text_run(image, 0x100, 4)

    result = discover_vtables(
        bytes(image),
        X86_BASE,
        "x86",
        [X86_TEXT_RANGE],
        [(X86_BASE, X86_BASE + len(image))],
    )

    assert result.confirmed_count == 1
    candidate = result.confirmed[0]
    assert candidate.addr == X86_BASE + 0x100
    assert candidate.slot_count == 4
    assert candidate.arch == "x86"
    assert candidate.confidence == CONFIDENCE_CONFIRMED
    assert candidate.name == "Foo"
    assert candidate.mangled == ".?AVFoo@@"
    assert candidate.col_addr == X86_BASE + X86_COL
    assert hookable_addresses(result) == (X86_BASE + 0x100,)


# --------------------------------------------------------------------------- #
# Test 3: shape-only labeling (catalogued, clearly labelled, never promoted)
# --------------------------------------------------------------------------- #


def test_shape_run_without_col_is_shape_only_and_never_promoted():
    """A text run whose ``-ptr`` slot is not a COL is ``shape-only`` with evidence."""
    image = bytearray(0x400)
    _x64_text_run(image, 0x100, 4)  # slot at 0xF8 is zero -> no COL

    result = _x64_image_result(image)

    assert result.confirmed_count == 0
    assert result.shape_only_count == 1
    candidate = result.shape_only[0]
    assert candidate.confidence == CONFIDENCE_SHAPE_ONLY
    assert candidate.hookable is False
    assert candidate.slot_count == 4
    # a shape-only row is catalogued with the reader's refusal, and carries no
    # invented identity
    assert candidate.reason
    assert "outside the supplied buffer" in candidate.reason
    assert candidate.name == ""
    assert candidate.mangled == ""
    assert candidate.bases == ()
    assert candidate.col_addr == 0
    assert hookable_addresses(result) == ()
    assert candidate.to_dict()["hookable"] is False


# --------------------------------------------------------------------------- #
# Test 4: empty and garbage images
# --------------------------------------------------------------------------- #


def test_empty_garbage_and_non_pointer_images_yield_empty_catalogue():
    """Empty/garbage bytes are an empty catalogue, never a fabricated candidate."""
    empty = discover_vtables(
        b"", X64_BASE, "x64", [X64_TEXT_RANGE], [(X64_BASE, X64_BASE + 0x100)]
    )
    assert empty.candidates == []
    assert empty.rejected == []
    assert empty.confirmed_count == 0

    garbage = bytearray(0x400)
    garbage[:] = bytes((0xA5 + index) & 0xFF for index in range(len(garbage)))
    assert _x64_image_result(garbage).candidates == []

    # 8-byte values that never land in text_ranges are not vtable shapes
    non_text = bytearray(0x400)
    for index in range(6):
        _u64(non_text, 0x100 + 8 * index, X64_BASE + 0x9000)  # outside the window
    assert _x64_image_result(non_text).candidates == []

    # no text range (or no search range) at all: empty, honestly
    image = _x64_class_image()
    _x64_text_run(image, 0x100, MIN_RUN_LENGTH)
    assert (
        discover_vtables(
            bytes(image), X64_BASE, "x64", None, [(X64_BASE, X64_BASE + 0x400)]
        ).candidates
        == []
    )
    assert discover_vtables(bytes(image), X64_BASE, "x64", [X64_TEXT_RANGE], []).candidates == []


# --------------------------------------------------------------------------- #
# Test 5: truncation / out-of-buffer links are refused, never crashed
# --------------------------------------------------------------------------- #


def test_truncated_slot_and_out_of_buffer_col_pointer_are_refused():
    """A read past the buffer end is clamped/refused; no exception escapes."""
    # a 2-slot run (one slot short of the minimum) yields nothing at all
    short = bytearray(0x120)
    _x64_text_run(short, 0xF0, MIN_RUN_LENGTH - 1)
    assert _x64_image_result(short).candidates == []

    # a search range that runs past the buffer is clamped to it, not believed
    image = bytearray(0x200)
    _x64_text_run(image, 0x100, 3)
    clamped = discover_vtables(
        bytes(image),
        X64_BASE,
        "x64",
        [X64_TEXT_RANGE],
        [(X64_BASE, X64_BASE + 0x4000)],
    )
    exact = discover_vtables(
        bytes(image), X64_BASE, "x64", [X64_TEXT_RANGE], [(X64_BASE, X64_BASE + 0x200)]
    )
    assert clamped.to_dict() == exact.to_dict()
    assert [item.addr for item in clamped.candidates] == [X64_BASE + 0x100]

    # a vftable[-1] slot pointing outside the image is shape-only evidence
    outside = bytearray(0x120)
    _u64(outside, 0xF8, X64_BASE + 0x99999)
    _x64_text_run(outside, 0x100, MIN_RUN_LENGTH)
    result = _x64_image_result(outside)
    assert result.confirmed_count == 0
    assert result.shape_only_count == 1
    assert "outside the supplied buffer" in result.shape_only[0].reason

    # a COL pointer inside the buffer but onto non-COL bytes: same refusal
    junk = _x64_class_image()
    junk[0:16] = b"\xff" * 16  # zero page: not a COL
    _u64(junk, 0xF8, X64_BASE)
    _x64_text_run(junk, 0x100, MIN_RUN_LENGTH)
    result = _x64_image_result(junk)
    assert result.confirmed_count == 0
    assert result.shape_only_count == 1
    assert result.shape_only[0].reason.startswith("CompleteObjectLocator")


# --------------------------------------------------------------------------- #
# Test 6: R1 caps (default 50, ceiling 1000) and cap reporting
# --------------------------------------------------------------------------- #


def test_r1_caps_default_ceiling_and_bad_budgets():
    """60 shape runs: default cap keeps 50 and reports 10; ceilings clamp, bad raise."""
    image = _shape_only_runs(DEFAULT_MAX_VTABLES + 10)
    result = _x64_image_result(image)

    assert len(result.candidates) == DEFAULT_MAX_VTABLES
    assert len(result.rejected) == 10
    assert result.candidates[0].addr == X64_BASE + 0x1000
    assert result.candidates[-1].addr == X64_BASE + 0x1000 + 49 * 0x20
    for addr, reason in result.rejected:
        assert reason.startswith("vtable cap reached")
        assert str(DEFAULT_MAX_VTABLES) in reason
    assert [addr for addr, _ in result.rejected] == [
        X64_BASE + 0x1000 + index * 0x20 for index in range(50, 60)
    ]

    # a small explicit cap is honoured, with the remainder reported
    small = _x64_image_result(image, max_vtables=3)
    assert len(small.candidates) == 3
    assert len(small.rejected) == 57

    # an over-large budget is clamped to the ceiling, never an error
    clamped = _x64_image_result(image, max_vtables=MAX_VTABLES * 10)
    assert len(clamped.candidates) == 60
    assert clamped.rejected == []

    # a non-positive or non-integer budget is a caller error
    for bad in (0, -1, True, "50", 3.5):
        with pytest.raises(RttiError):
            _x64_image_result(image, max_vtables=bad)


def test_cap_keeps_confirmed_links_before_shape_only_runs():
    """Under a cap a validated COL link is kept even when its address is higher."""
    image = _x64_class_image()
    _x64_text_run(image, 0x100, 3)  # shape-only (slot at 0xF8 is zero)
    _x64_text_run(image, 0x140, 3)  # shape-only
    _x64_text_run(image, 0x180, 3)  # shape-only
    _u64(image, 0x2F8, X64_BASE + X64_COL)  # confirmed, but the highest address
    _x64_text_run(image, 0x300, 5)

    result = _x64_image_result(image, max_vtables=1)

    assert [item.addr for item in result.candidates] == [X64_BASE + 0x300]
    assert result.candidates[0].confidence == CONFIDENCE_CONFIRMED
    assert len(result.rejected) == 3
    assert all("dropped shape-only candidate" in reason for _, reason in result.rejected)

    # with no cap the whole catalogue is present, still address-ordered
    full = _x64_image_result(image, max_vtables=MAX_VTABLES)
    assert [item.addr for item in full.candidates] == [
        X64_BASE + addr for addr in (0x100, 0x140, 0x180, 0x300)
    ]
    assert full.confirmed_count == 1


# --------------------------------------------------------------------------- #
# Test 7: determinism (address order + repeated calls)
# --------------------------------------------------------------------------- #


def test_discovery_is_deterministic_and_address_ordered():
    """Repeated discovery of one image is byte-identical and address-ascending."""
    image = _shape_only_runs(DEFAULT_MAX_VTABLES + 5)
    first = _x64_image_result(image)
    second = _x64_image_result(image)

    assert first.to_dict() == second.to_dict()
    addresses = [item.addr for item in first.candidates]
    assert addresses == sorted(addresses)
    assert addresses == [X64_BASE + 0x1000 + index * 0x20 for index in range(50)]

    # the scan order of the search ranges does not change the answer
    swapped = discover_vtables(
        bytes(image),
        X64_BASE,
        "x64",
        [X64_TEXT_RANGE],
        [(X64_BASE + 0x800, X64_BASE + len(image)), (X64_BASE, X64_BASE + 0x800)],
    )
    assert swapped.to_dict() == first.to_dict()


# --------------------------------------------------------------------------- #
# Test 8: shape-pass rules (run length, pointer alignment, text membership)
# --------------------------------------------------------------------------- #


def test_shape_pass_requires_three_aligned_slots_pointing_into_text():
    """Runs are >= 3 slots, pointer-aligned, and every slot must resolve to text."""
    two_slots = bytearray(0x400)
    _x64_text_run(two_slots, 0x100, 2)
    assert _x64_image_result(two_slots).candidates == []

    split = bytearray(0x400)  # a non-text value in the middle splits the run
    _x64_text_run(split, 0x100, 2)
    _u64(split, 0x110, X64_BASE + 0x9000)
    _x64_text_run(split, 0x118, 4)
    result = _x64_image_result(split)
    assert [item.addr for item in result.candidates] == [X64_BASE + 0x118]
    assert result.candidates[0].slot_count == 4

    # a search range starting mid-slot never yields a misaligned candidate
    misaligned = bytearray(0x400)
    _x64_text_run(misaligned, 0x104, 4)
    _x64_text_run(misaligned, 0x140, 3)
    shifted = discover_vtables(
        bytes(misaligned),
        X64_BASE,
        "x64",
        [X64_TEXT_RANGE],
        [(X64_BASE + 0x101, X64_BASE + 0x200)],
    )
    assert all(item.addr % 8 == 0 for item in shifted.candidates)
    assert X64_BASE + 0x104 not in [item.addr for item in shifted.candidates]
    assert X64_BASE + 0x140 in [item.addr for item in shifted.candidates]


# --------------------------------------------------------------------------- #
# Test 9: bounds-checked arguments (nothing guessed, nothing silently accepted)
# --------------------------------------------------------------------------- #


def test_bad_arguments_raise_rtti_error_and_out_of_image_ranges_stay_empty():
    """Malformed ranges/budgets/archtitectures raise; out-of-image ranges are empty."""
    image = _x64_class_image()
    _x64_text_run(image, 0x100, MIN_RUN_LENGTH)
    blob = bytes(image)

    with pytest.raises(RttiError):
        discover_vtables(blob, -1, "x64", [X64_TEXT_RANGE], [(X64_BASE, 0x400)])
    with pytest.raises(RttiError):
        discover_vtables("not bytes", X64_BASE, "x64", [X64_TEXT_RANGE], [(X64_BASE, 0x400)])
    with pytest.raises(RttiError):
        discover_vtables(blob, X64_BASE, "sparc", [X64_TEXT_RANGE], [(X64_BASE, 0x400)])
    # end before start, negative, and non-pair entries are all caller errors
    for bad_range in (
        [(X64_BASE + 0x10, X64_BASE + 0x04)],
        [(-1, 0x10)],
        ["0x140000000-0x140001000"],
        [(X64_BASE,)],
    ):
        with pytest.raises(RttiError):
            discover_vtables(blob, X64_BASE, "x64", bad_range, [(X64_BASE, 0x400)])
        with pytest.raises(RttiError):
            discover_vtables(blob, X64_BASE, "x64", [X64_TEXT_RANGE], bad_range)

    # ranges wholly outside the image are honest empty input, not an error
    far = discover_vtables(
        blob, X64_BASE, "x64", [X64_TEXT_RANGE], [(X64_BASE + 0x100000, X64_BASE + 0x200000)]
    )
    assert far.candidates == []
    assert far.rejected == []


def test_ranges_accept_pairs_region_objects_and_mappings():
    """A caller may spell a range as a pair, a ``base``/``size`` region, or a dict."""
    image = _x64_class_image()
    _u64(image, 0xF8, X64_BASE + X64_COL)
    _x64_text_run(image, 0x100, MIN_RUN_LENGTH)
    blob = bytes(image)

    as_pairs = discover_vtables(
        blob, X64_BASE, "x64", [X64_TEXT_RANGE], [(X64_BASE, X64_BASE + 0x400)]
    )
    as_region = discover_vtables(
        blob, X64_BASE, "x64", [_Region(*X64_TEXT_RANGE)], [_Region(X64_BASE, 0x400)]
    )
    as_mapping = discover_vtables(
        blob,
        X64_BASE,
        "x64",
        [{"start": X64_TEXT_RANGE[0], "end": X64_TEXT_RANGE[1]}],
        [{"base": X64_BASE, "size": 0x400}],
    )
    single_pair = discover_vtables(
        blob, X64_BASE, "x64", X64_TEXT_RANGE, (X64_BASE, X64_BASE + 0x400)
    )

    expected = as_pairs.to_dict()
    assert as_region.to_dict() == expected
    assert as_mapping.to_dict() == expected
    assert single_pair.to_dict() == expected
    assert as_pairs.confirmed_count == 1


# --------------------------------------------------------------------------- #
# Test 10: the plan's negative-result clause (a zero count is a report)
# --------------------------------------------------------------------------- #


def test_negative_result_is_reported_not_patched_around():
    """Zero COL-linked vtables stays a labelled catalogue plus the clause, not a loosening."""
    image = _shape_only_runs(6)
    result = _x64_image_result(image)

    assert result.confirmed_count == 0  # POW-002's HOI4 outcome, reproduced here
    assert result.shape_only_count == 6
    assert hookable_addresses(result) == ()
    labelled = [item for item in result.candidates if item.confidence == CONFIDENCE_SHAPE_ONLY]
    assert len(labelled) == 6
    assert all(item.reason and not item.name for item in labelled)

    summary = summarize_discovery(result)
    assert summary["confirmed"] == 0
    assert summary["shape_only"] == 6
    assert summary["catalogue_size"] == 6
    assert summary["confirmed_addresses"] == []
    assert summary["note"] == NEGATIVE_RESULT_NOTE
    assert result.to_dict()["note"] == NEGATIVE_RESULT_NOTE
    assert result.to_dict()["shape_only_count"] == 6

    # helpers refuse a foreign object instead of guessing an answer
    with pytest.raises(RttiError):
        hookable_addresses([1, 2, 3])  # type: ignore[arg-type]
    with pytest.raises(RttiError):
        summarize_discovery("not a result")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Tests 11-13 (UNIT-SEM5 chunk 1): bounded entry targets + confirmed-only helper
# --------------------------------------------------------------------------- #


#: A text window covering the target values the tests below place (so several
#: distinguishable slots can each resolve into text).
X64_TEXT_WINDOW = (X64_TEXT, X64_TEXT + 16)


def _x64_window_result(image: bytearray, *, max_vtables: int | None = None):
    """``discover_vtables`` over ``image`` with the wide text window."""
    return discover_vtables(
        bytes(image),
        X64_BASE,
        "x64",
        [X64_TEXT_WINDOW],
        [(X64_BASE, X64_BASE + len(image))],
        max_vtables,
    )


def test_confirmed_candidate_records_entry_targets_in_slot_order():
    """A confirmed row records the run's read values, in slot order, exactly."""
    image = _x64_class_image()
    _u64(image, 0xF8, X64_BASE + X64_COL)  # vftable[-1] -> validated COL
    for index in range(MIN_RUN_LENGTH):
        _u64(image, 0x100 + 8 * index, X64_TEXT + index)

    result = _x64_window_result(image)

    assert result.confirmed_count == 1
    candidate = result.confirmed[0]
    assert candidate.slot_count == MIN_RUN_LENGTH
    # the entries are the values the test itself wrote -- nothing is re-derived
    assert isinstance(candidate.entries, tuple)
    assert candidate.entries == (X64_TEXT, X64_TEXT + 1, X64_TEXT + 2)
    assert candidate.to_dict()["entries"] == [X64_TEXT, X64_TEXT + 1, X64_TEXT + 2]
    assert candidate.to_dict()["entries_hex"] == [
        f"0x{X64_TEXT + index:x}" for index in range(MIN_RUN_LENGTH)
    ]
    assert entry_targets(result) == (X64_TEXT, X64_TEXT + 1, X64_TEXT + 2)


def test_entries_are_bounded_to_the_first_32_slots():
    """A 40-slot run keeps exactly the first :data:`MAX_ENTRIES` targets."""
    assert MAX_ENTRIES == 32  # UNIT-SEM5 chunk 1 fixes the bound at 32 slots

    image = _x64_class_image()
    _u64(image, 0xF8, X64_BASE + X64_COL)
    _x64_text_run(image, 0x100, 40)

    result = _x64_image_result(image)

    assert result.confirmed_count == 1
    candidate = result.confirmed[0]
    assert candidate.slot_count == 40  # the whole run is still counted
    assert len(candidate.entries) == MAX_ENTRIES
    assert candidate.entries == (X64_TEXT,) * MAX_ENTRIES
    assert len(candidate.to_dict()["entries"]) == MAX_ENTRIES
    assert len(candidate.to_dict()["entries_hex"]) == MAX_ENTRIES
    # duplicates collapse, so the helper answers one target rather than 32
    assert entry_targets(result) == (X64_TEXT,)


def test_entry_targets_helper_is_confirmed_only_deduped_and_sorted():
    """Shape-only entries are recorded but never surface through ``entry_targets``."""
    image = _x64_class_image(0x600)
    _u64(image, 0xF8, X64_BASE + X64_COL)  # confirmed run #1 at 0x100
    _u64(image, 0x178, X64_BASE + X64_COL)  # confirmed run #2 at 0x180
    for addr, value in (
        (0x100, X64_TEXT + 1),
        (0x108, X64_TEXT),
        (0x110, X64_TEXT + 2),
        (0x180, X64_TEXT + 2),
        (0x188, X64_TEXT + 3),
        (0x190, X64_TEXT),
        (0x200, X64_TEXT + 8),  # shape-only run at 0x200: its -8 slot is zero
        (0x208, X64_TEXT + 9),
        (0x210, X64_TEXT + 10),
    ):
        _u64(image, addr, value)

    result = _x64_window_result(image)

    assert result.confirmed_count == 2
    assert result.shape_only_count == 1
    assert hookable_addresses(result) == (X64_BASE + 0x100, X64_BASE + 0x180)

    # the shape-only row keeps its evidence (recorded, labelled, never promoted)
    shape_only = result.shape_only[0]
    assert shape_only.addr == X64_BASE + 0x200
    assert shape_only.entries == (X64_TEXT + 8, X64_TEXT + 9, X64_TEXT + 10)
    assert shape_only.to_dict()["entries_hex"] == [
        f"0x{X64_TEXT + offset:x}" for offset in (8, 9, 10)
    ]

    # deduped (X64_TEXT + 2 twice, X64_TEXT twice), sorted, confirmed rows only
    assert entry_targets(result) == (
        X64_TEXT,
        X64_TEXT + 1,
        X64_TEXT + 2,
        X64_TEXT + 3,
    )
    for excluded in (X64_TEXT + 8, X64_TEXT + 9, X64_TEXT + 10):
        assert excluded not in entry_targets(result)

    # a hand-built row keeps constructor compatibility and an empty default
    hand_built = VtableCandidate(X64_BASE + 0x100, CONFIDENCE_SHAPE_ONLY, 3)
    assert hand_built.entries == ()
    assert hand_built.to_dict()["entries"] == []
    assert hand_built.to_dict()["entries_hex"] == []
    assert entry_targets(VtableScanResult([hand_built], [])) == ()

    # the helper refuses a foreign object instead of guessing an answer
    with pytest.raises(RttiError):
        entry_targets([1, 2, 3])  # type: ignore[arg-type]
