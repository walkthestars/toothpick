"""Tests for UNIT-SEM5 Chunk 3 (`docs/units/UNIT-SEM5-PLAN.md`): string-xref labeling.

Binding contract: chunk 3 of the unit plan (`src/toothpick/strxref.py`) -- a
RIP-relative ``lea`` scan over caller-supplied ``.text`` bytes against
caller-supplied string ranges yields per-caller string sets, and
``label_functions(funcs_sorted, xrefs)`` attributes each xref to the function
that contains it by binary search.  Pure / offline / deterministic, stdlib only.
Design SS1 / SS19: a label is only ever what was proven from the parsed bytes --
no string is repaired, no link is invented -- plus Risk R1 (bounded emission:
default 50 strings per caller / per function, ceiling 1000).

Battery:

1. Module surface: constants, ``__all__``, ``StrxrefError`` is a ``ValueError``,
   and the R1 cap helpers (``resolve_max_strings`` default / clamp / refusals).
2. Planted ``lea``: a synthetic image with a hand-assembled ``8D <modrm>
   <disp32>`` instruction (7-byte REX-prefixed and 6-byte REX-less) at a known
   address and printable strings at known targets, one caller and several
   callers, ascending keys, sorted lists.
3. Encoding breadth: every RIP-relative ModRM register form (``05 0D ... 3D``,
   the reg field masked out so ``rbx``/``rcx``/``rsi``/``rdi`` and the REX.R
   spellings all match), both instruction lengths (REX.W 7 bytes, REX-less
   6 bytes, REX without W 7 bytes), the rejections (mod != 00, wrong opcode),
   negative displacement, resynchronisation after garbage bytes before a real
   ``lea``, back-to-back instructions stepped past by their matched length, and
   a truncated instruction.
4. Reading honesty: non-printable bytes anywhere in the run, the printable
   window's own edges (0x20 / 0x7E), an unterminated run, a NUL just outside the
   supplied range, a target outside every range, a range past the buffer, and
   the ``min_len`` boundary.
5. Range spellings: pair / list of pairs / ``{start,end}`` / ``{base,size}`` /
   region-like objects / ``None`` / empty, plus clamping to the buffer and the
   split-at-every-boundary rule for overlapping supplied ranges.
6. Argument errors: non-bytes image, negative base, backwards range, malformed
   entries, bad ``min_len`` / ``max_strings`` -- all ``StrxrefError``.
7. ``label_functions``: boundary cases of ``func[i] <= caller < func[i + 1]``,
   the open-ended last function, ``UNATTRIBUTED`` below the first start, absent
   functions, dedupe, sorting, and the R1 caps.
8. Hygiene: determinism across runs and container types, no mutation of the
   caller's blob, no ``faulthandler``/network/SQLite/subprocess import, no
   ``shell=True``, and an RTTI-reader-free module.

No real binary is required: every image here is built byte-by-byte in the test.
"""

from __future__ import annotations

import re
import struct
from pathlib import Path

import pytest

from toothpick import strxref
from toothpick.strxref import (
    DEFAULT_MAX_STRINGS,
    LEA_LENGTH,
    LEA_SHORT_LENGTH,
    MAX_STRINGS,
    MIN_STRING_LEN,
    PRINTABLE_MAX,
    PRINTABLE_MIN,
    PROVENANCE,
    StrxrefError,
    UNATTRIBUTED,
    label_functions,
    resolve_max_strings,
    xref_strings,
)

# --- synthetic image layout ------------------------------------------------- #

BASE = 0x140001000
IMAGE_SIZE = 0x400
TEXT_START = BASE
TEXT_END = BASE + 0x100
STR_START = BASE + 0x100
STR_END = BASE + 0x200

#: All eight ModRM bytes with mod=00, r/m=101 (a RIP-relative operand), one per
#: ``reg`` field value 0..7.
RIP_MODRM_BYTES = (0x05, 0x0D, 0x15, 0x1D, 0x25, 0x2D, 0x35, 0x3D)

#: The destination registers a literal ``modrm == 0x05`` test can never match, as
#: ``(ModRM byte, register)``: the reg field sits in bits 5-3 and the 0xC7 mask
#: clears it, so all four spell the same mod=00 r/m=101 form.
REG_FIELD_MODRM_CASES = (
    (0x1D, "rbx"),
    (0x0D, "rcx"),
    (0x35, "rsi"),
    (0x3D, "rdi"),
)


class _Region:
    """A minimal region-like object (``base``/``size``), as a loader holds them."""

    def __init__(self, base: int, size: int) -> None:
        self.base = base
        self.size = size


class _Span:
    """A minimal span-like object (``start``/``end``)."""

    def __init__(self, start: int, end: int) -> None:
        self.start = start
        self.end = end


def _lea(
    at_addr: int,
    target: int,
    *,
    modrm: int = 0x05,
    rex: int | None = 0x48,
) -> bytes:
    """Hand-assemble a RIP-relative ``lea`` at *at_addr* resolving to *target*.

    ``rex=None`` emits the REX-less 6-byte ``8D /r disp32`` form; any other value
    emits that prefix byte, the opcode, the ModRM byte and a 32-bit
    displacement (7 bytes).
    """
    prefix = b"" if rex is None else bytes([rex])
    length = len(prefix) + 6
    displacement = target - (at_addr + length)
    assert -(2 ** 31) <= displacement < 2 ** 31
    return prefix + bytes([0x8D, modrm]) + struct.pack("<i", displacement)


def _place(blob: bytearray, at_addr: int, payload: bytes) -> None:
    offset = at_addr - BASE
    assert 0 <= offset and offset + len(payload) <= len(blob)
    blob[offset : offset + len(payload)] = payload


def _build(leas=(), strings=(), *, size: int = IMAGE_SIZE, filler: int = 0x00) -> bytes:
    """An image holding *leas* (``(addr, target)``) and *strings* (``(addr, text)``).

    ``text`` is a ``str`` (the builder appends the NUL) or the literal ``bytes``
    body to write, so malformed bodies can be planted verbatim.
    """
    blob = bytearray([filler]) * size
    for at_addr, text in strings:
        payload = text.encode("ascii") + b"\x00" if isinstance(text, str) else bytes(text)
        _place(blob, at_addr, payload)
    for at_addr, target in leas:
        _place(blob, at_addr, _lea(at_addr, target))
    return bytes(blob)


def _scan(data, text_range=(TEXT_START, TEXT_END), string_ranges=(STR_START, STR_END), **kwargs):
    return xref_strings(data, BASE, text_range, string_ranges, **kwargs)


# --- 1. module surface ------------------------------------------------------ #


def test_public_api_is_importable():
    for name in strxref.__all__:
        assert hasattr(strxref, name), name
    for name in (
        "xref_strings",
        "label_functions",
        "resolve_max_strings",
        "StrxrefError",
        "DEFAULT_MAX_STRINGS",
        "MAX_STRINGS",
        "UNATTRIBUTED",
    ):
        assert name in strxref.__all__
    assert UNATTRIBUTED is None


def test_documented_defaults():
    assert LEA_LENGTH == 7
    assert LEA_SHORT_LENGTH == 6
    assert MIN_STRING_LEN == 4
    assert PRINTABLE_MIN == 0x20
    assert PRINTABLE_MAX == 0x7E
    assert PROVENANCE == "rip-relative-lea"
    assert strxref.DISP_OFFSET == 3
    assert strxref.SHORT_DISP_OFFSET == 2
    assert strxref.DISP_SIZE == 4


def test_strxref_error_is_a_value_error():
    assert issubclass(StrxrefError, ValueError)


def test_resolve_max_strings_defaults_and_clamps():
    assert DEFAULT_MAX_STRINGS == 50
    assert MAX_STRINGS == 1000
    assert resolve_max_strings() == DEFAULT_MAX_STRINGS
    assert resolve_max_strings(None) == DEFAULT_MAX_STRINGS
    assert resolve_max_strings(1) == 1
    assert resolve_max_strings(MAX_STRINGS) == MAX_STRINGS
    assert resolve_max_strings(MAX_STRINGS + 1) == MAX_STRINGS
    assert resolve_max_strings(10 ** 9) == MAX_STRINGS


@pytest.mark.parametrize("bad", [0, -1, -1000, True, False, 3.5, "50", b"50", object()])
def test_resolve_max_strings_refuses_non_positive_or_non_integer(bad):
    with pytest.raises(StrxrefError):
        resolve_max_strings(bad)


# --- 2. planted lea --------------------------------------------------------- #


def test_planted_lea_yields_its_string():
    data = _build(leas=[(BASE + 0x10, STR_START)], strings=[(STR_START, "hello world")])
    assert _scan(data) == {BASE + 0x10: ["hello world"]}


def test_two_callers_get_their_own_strings_and_ascending_keys():
    data = _build(
        leas=[(BASE + 0x20, BASE + 0x110), (BASE + 0x10, STR_START)],
        strings=[(STR_START, "alpha"), (BASE + 0x110, "bravo")],
    )
    result = _scan(data)
    assert result == {BASE + 0x10: ["alpha"], BASE + 0x20: ["bravo"]}
    assert list(result) == sorted(result)


def test_two_callers_may_load_the_same_string():
    data = _build(
        leas=[(BASE + 0x10, STR_START), (BASE + 0x30, STR_START)],
        strings=[(STR_START, "shared text")],
    )
    assert _scan(data) == {BASE + 0x10: ["shared text"], BASE + 0x30: ["shared text"]}


def test_unrelated_bytes_and_distant_strings_do_not_bleed():
    data = _build(
        leas=[(BASE + 0x10, STR_START), (BASE + 0x40, BASE + 0x140)],
        strings=[(STR_START, "alpha"), (BASE + 0x140, "zeta")],
        filler=0xCC,
    )
    assert _scan(data) == {BASE + 0x10: ["alpha"], BASE + 0x40: ["zeta"]}


def test_min_len_boundary_is_inclusive():
    four = _build(leas=[(BASE + 0x10, STR_START)], strings=[(STR_START, "abcd")])
    assert _scan(four) == {BASE + 0x10: ["abcd"]}

    three = _build(leas=[(BASE + 0x10, STR_START)], strings=[(STR_START, "abc")])
    assert _scan(three) == {}
    assert _scan(three, min_len=3) == {BASE + 0x10: ["abc"]}


@pytest.mark.parametrize("bad", [0, -1, 2.5, "4", True])
def test_min_len_must_be_a_positive_integer(bad):
    data = _build(leas=[(BASE + 0x10, STR_START)], strings=[(STR_START, "hello")])
    with pytest.raises(StrxrefError):
        _scan(data, min_len=bad)


# --- 3. encoding breadth ---------------------------------------------------- #


@pytest.mark.parametrize("modrm", RIP_MODRM_BYTES)
def test_every_rip_relative_modrm_register_form_is_recognised(modrm):
    blob = bytearray(_build(strings=[(STR_START, "target")]))
    _place(blob, BASE + 0x10, _lea(BASE + 0x10, STR_START, modrm=modrm))
    assert _scan(bytes(blob)) == {BASE + 0x10: ["target"]}


@pytest.mark.parametrize(
    "modrm, register",
    REG_FIELD_MODRM_CASES,
    ids=[register for _, register in REG_FIELD_MODRM_CASES],
)
def test_reg_field_is_masked_out_so_rbx_rcx_rsi_rdi_match(modrm, register):
    """Regression: a literal ``modrm == 0x05`` comparison only ever matched rax.

    The RIP-relative form test is ``(modrm & 0xC7) == 0x05`` -- mod=00 and
    r/m=101 with the reg field (bits 5-3) masked away -- so ``rbx``/``rcx``/
    ``rsi``/``rdi`` are the same form, in both instruction lengths.
    """
    assert (modrm & strxref.RIP_MODRM_MASK) == strxref.RIP_MODRM_VALUE

    rex_w = bytearray(_build(strings=[(STR_START, "target")]))
    _place(rex_w, BASE + 0x10, _lea(BASE + 0x10, STR_START, modrm=modrm))
    assert _scan(bytes(rex_w)) == {BASE + 0x10: ["target"]}

    short = bytearray(_build(strings=[(STR_START, "target")]))
    _place(short, BASE + 0x10, _lea(BASE + 0x10, STR_START, modrm=modrm, rex=None))
    assert _scan(bytes(short)) == {BASE + 0x10: ["target"]}


@pytest.mark.parametrize("rex", [0x4C, 0x4D, 0x4E, 0x4F])
def test_rex_r_extended_destinations_are_masked_too(rex):
    """REX.R renames the destination (r8-r15); the operand form is unchanged."""
    blob = bytearray(_build(strings=[(STR_START, "target")]))
    _place(blob, BASE + 0x10, _lea(BASE + 0x10, STR_START, modrm=0x1D, rex=rex))
    assert _scan(bytes(blob)) == {BASE + 0x10: ["target"]}

@pytest.mark.parametrize("rex", [0x48, 0x49, 0x4C, 0x4F])
def test_rex_w_variants_are_accepted(rex):
    blob = bytearray(_build(strings=[(STR_START, "target")]))
    _place(blob, BASE + 0x10, _lea(BASE + 0x10, STR_START, rex=rex))
    assert _scan(bytes(blob)) == {BASE + 0x10: ["target"]}


@pytest.mark.parametrize("rex", [0x40, 0x41, 0x44, 0x47])
def test_rex_without_the_w_bit_is_a_32_bit_destination_lea(rex):
    """REX without W changes the destination width, not the addressing.

    ``REX 8D /r disp32`` still forms ``RIP + sign-extended disp32``; only the
    destination is 32-bit instead of 64-bit, so the string it references is the
    same evidence and the caller is still the instruction's first byte.
    """
    blob = bytearray(_build(strings=[(STR_START, "target")]))
    _place(blob, BASE + 0x10, _lea(BASE + 0x10, STR_START, rex=rex))
    assert _scan(bytes(blob)) == {BASE + 0x10: ["target"]}


def test_rex_less_lea_is_scanned_without_a_prefix():
    """The REX-less 32-bit form is ``8D /r disp32``: 6 bytes, disp32 at +2."""
    expected = {BASE + 0x10: ["target"]}
    blob = bytearray(_build(strings=[(STR_START, "target")]))
    _place(blob, BASE + 0x10, _lea(BASE + 0x10, STR_START, rex=None))
    assert len(_lea(BASE + 0x10, STR_START, rex=None)) == LEA_SHORT_LENGTH == 6
    assert strxref._match_lea(bytes(blob), BASE, BASE + 0x10) == (STR_START, 6)
    assert _scan(bytes(blob)) == expected


@pytest.mark.parametrize("modrm", RIP_MODRM_BYTES)
def test_every_rip_relative_modrm_register_form_is_recognised_without_rex(modrm):
    blob = bytearray(_build(strings=[(STR_START, "target")]))
    _place(blob, BASE + 0x10, _lea(BASE + 0x10, STR_START, modrm=modrm, rex=None))
    assert _scan(bytes(blob)) == {BASE + 0x10: ["target"]}


@pytest.mark.parametrize("modrm", [0x85, 0x45, 0xC5, 0xED, 0x06, 0x0E])
def test_rex_less_modrm_mod_bits_must_be_zero_too(modrm):
    blob = bytearray(_build(strings=[(STR_START, "target")]))
    _place(blob, BASE + 0x10, _lea(BASE + 0x10, STR_START, modrm=modrm, rex=None))
    assert _scan(bytes(blob)) == {}


def test_rex_less_negative_displacement_resolves_backwards():
    blob = bytearray(_build(strings=[(BASE + 0x110, "backwards")]))
    _place(blob, BASE + 0x200, _lea(BASE + 0x200, BASE + 0x110, rex=None))
    assert _scan(bytes(blob), text_range=(BASE + 0x200, BASE + 0x300)) == {
        BASE + 0x200: ["backwards"]
    }


def test_rex_less_lea_ending_exactly_at_the_buffer_end_is_scanned():
    size = 0x20
    blob = bytearray(size)
    lea_at = BASE + size - LEA_SHORT_LENGTH
    _place(blob, lea_at, _lea(lea_at, BASE + 0x10, rex=None))
    _place(blob, BASE + 0x10, b"edge\x00")
    result = _scan(bytes(blob), text_range=(BASE, BASE + size), string_ranges=(BASE, BASE + size))
    assert result == {lea_at: ["edge"]}


def test_both_instruction_lengths_are_stepped_past_exactly():
    """The scan advances by the matched length: 6-byte and 7-byte leas in a row."""
    blob = bytearray(_build(strings=[(STR_START, "alpha"), (BASE + 0x110, "bravo")]))
    _place(blob, BASE + 0x10, _lea(BASE + 0x10, STR_START, rex=None))
    _place(blob, BASE + 0x16, _lea(BASE + 0x16, BASE + 0x110))
    _place(blob, BASE + 0x1D, _lea(BASE + 0x1D, STR_START, rex=None))
    assert _scan(bytes(blob)) == {
        BASE + 0x10: ["alpha"],
        BASE + 0x16: ["bravo"],
        BASE + 0x1D: ["alpha"],
    }


@pytest.mark.parametrize("modrm", [0x85, 0x45, 0xC5, 0xED, 0x06, 0x0E])
def test_modrm_mod_bits_must_be_zero_for_the_rip_form(modrm):
    blob = bytearray(_build(strings=[(STR_START, "target")]))
    _place(blob, BASE + 0x10, _lea(BASE + 0x10, STR_START, modrm=modrm))
    assert _scan(bytes(blob)) == {}


def test_a_different_opcode_is_not_a_lea():
    blob = bytearray(_build(strings=[(STR_START, "target")]))
    _place(
        blob,
        BASE + 0x10,
        bytes([0x48, 0x8B, 0x05]) + struct.pack("<i", STR_START - (BASE + 0x17)),
    )
    assert _scan(bytes(blob)) == {}


def test_negative_displacement_resolves_backwards():
    data = _build(
        leas=[(BASE + 0x200, BASE + 0x110)],
        strings=[(BASE + 0x110, "backwards")],
        size=0x400,
    )
    result = _scan(data, text_range=(BASE + 0x200, BASE + 0x300))
    assert result == {BASE + 0x200: ["backwards"]}


def test_scan_resynchronises_after_garbage_and_a_partial_match():
    blob = bytearray(_build())
    _place(blob, BASE, b"\xCC" * 0x08)
    _place(blob, BASE + 0x08, b"\x48\x8D\x85\x11\x22\x33\x44\x55")  # mod != 00: no match
    _place(blob, BASE + 0x10, _lea(BASE + 0x10, STR_START))
    _place(blob, STR_START, b"recovered\x00")
    assert _scan(bytes(blob)) == {BASE + 0x10: ["recovered"]}


def test_truncated_instruction_bytes_are_never_read_past_the_blob():
    assert strxref._match_lea(b"", 0, 0) is None
    assert strxref._match_lea(bytes([0x48, 0x8D, 0x05, 0x00, 0x00, 0x00]), 0, 0) is None
    assert strxref._match_lea(bytes([0x48, 0x8D, 0x05]) + b"\x00" * 4, 0, 0) == (
        7,
        LEA_LENGTH,
    )
    # The REX-less form is 6 bytes: five bytes of it are not an instruction.
    assert strxref._match_lea(bytes([0x8D, 0x05, 0x00, 0x00, 0x00]), 0, 0) is None
    assert strxref._match_lea(bytes([0x8D, 0x05]) + b"\x00" * 4, 0, 0) == (6, LEA_SHORT_LENGTH)
    # A pattern that starts where only part of the instruction fits is ignored.
    tail = bytes([0x48, 0x8D, 0x05]) + b"\x00" * 4
    assert _scan(tail, text_range=(0, 7)) == {}


def test_a_lea_ending_exactly_at_the_buffer_end_is_scanned():
    size = 0x20
    blob = bytearray(size)
    lea_at = BASE + size - LEA_LENGTH
    _place(blob, lea_at, _lea(lea_at, BASE + 0x10))
    _place(blob, BASE + 0x10, b"edge\x00")
    result = _scan(bytes(blob), text_range=(BASE, BASE + size), string_ranges=(BASE, BASE + size))
    assert result == {lea_at: ["edge"]}


# --- 4. reading honesty ----------------------------------------------------- #


@pytest.mark.parametrize(
    "body",
    [
        b"abc\x01\x00",
        b"ab\x80cd\x00",
        b"ab\xffcd\x00",
        b"abc\x7f\x00",
        b"abc\x1f\x00",
        b"ab\x09cd\x00",
        b"\x00abcd\x00",
    ],
)
def test_non_printable_bytes_drop_the_string(body):
    blob = bytearray(_build(leas=[(BASE + 0x10, STR_START)]))
    _place(blob, STR_START, body)
    assert _scan(bytes(blob)) == {}


def test_printable_window_edges_are_accepted():
    blob = bytearray(_build(leas=[(BASE + 0x10, STR_START)]))
    _place(blob, STR_START, b"\x20abc\x7e\x00")
    assert _scan(bytes(blob)) == {BASE + 0x10: [" abc~"]}


def test_unterminated_run_inside_the_range_is_dropped():
    blob = bytearray(_build(leas=[(BASE + 0x10, STR_START)]))
    _place(blob, STR_START, b"abcd")  # no NUL before the range end
    assert _scan(bytes(blob), string_ranges=(STR_START, STR_START + 4)) == {}


def test_nul_just_outside_the_supplied_range_is_dropped():
    blob = bytearray(_build(leas=[(BASE + 0x10, STR_START)]))
    _place(blob, STR_START, b"abcdefgh\x00")  # the NUL sits at exactly range end
    assert _scan(bytes(blob), string_ranges=(STR_START, STR_START + 8)) == {}
    # One byte more of range and the very same bytes are evidence.
    assert _scan(bytes(blob), string_ranges=(STR_START, STR_START + 9)) == {
        BASE + 0x10: ["abcdefgh"]
    }


def test_target_at_the_range_end_is_not_inside_it():
    data = _build(leas=[(BASE + 0x10, STR_END)], strings=[(STR_END, "outside")])
    assert _scan(data, string_ranges=(STR_START, STR_END)) == {}


def test_target_before_the_range_start_is_not_inside_it():
    data = _build(leas=[(BASE + 0x10, STR_START - 1)], strings=[(STR_START, "inside")])
    assert _scan(data, string_ranges=(STR_START, STR_END)) == {}


def test_target_outside_every_range_yields_no_xref():
    data = _build(leas=[(BASE + 0x10, BASE + 0x180)], strings=[(BASE + 0x180, "orphan")])
    assert _scan(data, string_ranges=(STR_START, STR_START + 0x10)) == {}


def test_string_range_entirely_past_the_blob_yields_no_xref():
    far = BASE + 0x8000
    blob = bytearray(IMAGE_SIZE)
    _place(blob, BASE + 0x10, _lea(BASE + 0x10, far))
    assert _scan(bytes(blob), string_ranges=(far, far + 0x100)) == {}


def test_overlapping_ranges_split_at_every_boundary():
    data = _build(leas=[(BASE + 0x10, BASE + 0x109)], strings=[(BASE + 0x109, "abcd")])
    result = _scan(
        data,
        string_ranges=[(STR_START, STR_START + 0x10), (STR_START + 0x08, BASE + 0x120)],
    )
    assert result == {BASE + 0x10: ["abcd"]}


def test_a_string_cannot_run_across_two_supplied_ranges():
    data = _build(leas=[(BASE + 0x10, BASE + 0x10C)], strings=[(BASE + 0x10C, "abcdefgh")])
    # The wider range would let the string finish, but the narrower one ends
    # first and the terminator lies outside it: dropped, never repaired.
    assert (
        _scan(
            data,
            string_ranges=[(STR_START, STR_START + 0x10), (STR_START + 0x08, BASE + 0x120)],
        )
        == {}
    )


def test_string_running_past_every_supplied_range_is_dropped():
    data = _build(leas=[(BASE + 0x10, BASE + 0x118)], strings=[(BASE + 0x118, "toolongforrange")])
    assert _scan(data, string_ranges=(BASE + 0x118, BASE + 0x11A)) == {}


# --- 5. range spellings and clamping ---------------------------------------- #


def test_adjacent_string_ranges_both_work():
    data = _build(
        leas=[(BASE + 0x10, STR_START), (BASE + 0x20, BASE + 0x120)],
        strings=[(STR_START, "first"), (BASE + 0x120, "second")],
    )
    result = _scan(data, string_ranges=[(STR_START, BASE + 0x110), (BASE + 0x110, STR_END)])
    assert result == {BASE + 0x10: ["first"], BASE + 0x20: ["second"]}


def test_mapping_range_spellings_are_accepted():
    data = _build(leas=[(BASE + 0x10, STR_START)], strings=[(STR_START, "mapped")])
    assert _scan(data, text_range={"start": TEXT_START, "end": TEXT_END}) == {
        BASE + 0x10: ["mapped"]
    }
    as_base_size = xref_strings(
        data,
        BASE,
        {"base": TEXT_START, "size": TEXT_END - TEXT_START},
        {"base": STR_START, "size": STR_END - STR_START},
    )
    assert as_base_size == {BASE + 0x10: ["mapped"]}


def test_region_like_objects_are_accepted():
    data = _build(leas=[(BASE + 0x10, STR_START)], strings=[(STR_START, "region")])
    by_base_size = xref_strings(
        data,
        BASE,
        _Region(TEXT_START, TEXT_END - TEXT_START),
        _Region(STR_START, STR_END - STR_START),
    )
    assert by_base_size == {BASE + 0x10: ["region"]}

    by_span = xref_strings(data, BASE, _Span(TEXT_START, TEXT_END), _Span(STR_START, STR_END))
    assert by_span == {BASE + 0x10: ["region"]}


def test_region_like_objects_are_accepted_inside_a_sequence():
    data = _build(leas=[(BASE + 0x10, STR_START)], strings=[(STR_START, "listed")])
    result = xref_strings(
        data,
        BASE,
        [_Region(TEXT_START, TEXT_END - TEXT_START)],
        [_Span(STR_START, STR_END)],
    )
    assert result == {BASE + 0x10: ["listed"]}


def test_no_ranges_means_no_evidence():
    data = _build(leas=[(BASE + 0x10, STR_START)], strings=[(STR_START, "unused")])
    assert _scan(data, text_range=None) == {}
    assert _scan(data, text_range=[]) == {}
    assert _scan(data, string_ranges=None) == {}
    assert _scan(data, string_ranges=[]) == {}


def test_text_range_beyond_the_buffer_is_clamped_not_believed():
    data = _build(leas=[(BASE + 0x10, STR_START)], strings=[(STR_START, "clamped")])
    assert _scan(data, text_range=(TEXT_START, BASE + 0x10000)) == {BASE + 0x10: ["clamped"]}


def test_text_range_below_the_base_is_clamped():
    data = _build(leas=[(BASE + 0x10, STR_START)], strings=[(STR_START, "clamped")])
    assert _scan(data, text_range=(BASE - 0x1000, TEXT_END)) == {BASE + 0x10: ["clamped"]}


def test_two_text_ranges_are_scanned_independently():
    data = _build(
        leas=[(BASE + 0x10, STR_START), (BASE + 0x40, BASE + 0x110)],
        strings=[(STR_START, "alpha"), (BASE + 0x110, "bravo")],
    )
    result = _scan(data, text_range=[(TEXT_START, BASE + 0x12), (BASE + 0x14, TEXT_END)])
    # The ``lea`` at 0x10 sits only in the first range, which stops mid-instruction
    # at 0x12, so that instruction is never claimed; the one at 0x40 is scanned.
    assert BASE + 0x10 not in result
    assert result == {BASE + 0x40: ["bravo"]}


def test_bytearray_and_memoryview_images_are_accepted_and_not_mutated():
    data = bytearray(_build(leas=[(BASE + 0x10, STR_START)], strings=[(STR_START, "stable")]))
    snapshot = bytes(data)
    assert _scan(data) == {BASE + 0x10: ["stable"]}
    assert bytes(data) == snapshot
    assert _scan(memoryview(data)) == {BASE + 0x10: ["stable"]}
    assert bytes(data) == snapshot


# --- 6. argument errors ----------------------------------------------------- #


@pytest.mark.parametrize("bad", ["bytes please", 42, None, [1, 2]])
def test_non_bytes_like_image_is_a_caller_error(bad):
    with pytest.raises(StrxrefError):
        xref_strings(bad, BASE, (TEXT_START, TEXT_END), (STR_START, STR_END))


@pytest.mark.parametrize("bad", [-1, 1.5, True, "0x140001000"])
def test_bad_base_address_is_a_caller_error(bad):
    data = _build(leas=[(BASE + 0x10, STR_START)], strings=[(STR_START, "xxxx")])
    with pytest.raises(StrxrefError):
        xref_strings(data, bad, (TEXT_START, TEXT_END), (STR_START, STR_END))


def test_backwards_range_is_a_caller_error():
    data = _build(leas=[(BASE + 0x10, STR_START)], strings=[(STR_START, "xxxx")])
    with pytest.raises(StrxrefError):
        _scan(data, text_range=(TEXT_END, TEXT_START))
    with pytest.raises(StrxrefError):
        _scan(data, string_ranges=(STR_END, STR_START))
    with pytest.raises(StrxrefError):
        _scan(data, string_ranges={"start": STR_END, "end": STR_START})


def test_negative_range_address_is_a_caller_error():
    data = _build(leas=[(BASE + 0x10, STR_START)], strings=[(STR_START, "xxxx")])
    with pytest.raises(StrxrefError):
        _scan(data, text_range=(-1, TEXT_END))


@pytest.mark.parametrize(
    "bad",
    [
        "abc",
        b"abc",
        5,
        None,
        True,
        (1, 2, 3),
        {"base": 16, "size": -1},
        {"start": 10, "end": 5},
        {"nope": 1},
    ],
)
def test_malformed_range_entries_are_caller_errors(bad):
    data = _build(leas=[(BASE + 0x10, STR_START)], strings=[(STR_START, "xxxx")])
    with pytest.raises(StrxrefError):
        _scan(data, string_ranges=[bad])


def test_range_container_must_be_a_sequence():
    data = _build(leas=[(BASE + 0x10, STR_START)], strings=[(STR_START, "xxxx")])
    with pytest.raises(StrxrefError):
        _scan(data, string_ranges="not a range")
    with pytest.raises(StrxrefError):
        _scan(data, string_ranges=True)


# --- 7. label_functions ----------------------------------------------------- #


def test_attribution_uses_the_half_open_bounds():
    funcs = [BASE, BASE + 0x20, BASE + 0x40]
    xrefs = {
        BASE: ["at-first-start"],
        BASE + 0x1F: ["last-byte-of-first"],
        BASE + 0x20: ["at-second-start"],
        BASE + 0x3F: ["last-byte-of-second"],
        BASE + 0x40: ["at-third-start"],
        BASE + 0x1000: ["after-the-last-start"],
    }
    assert label_functions(funcs, xrefs) == {
        BASE: ["at-first-start", "last-byte-of-first"],
        BASE + 0x20: ["at-second-start", "last-byte-of-second"],
        BASE + 0x40: ["after-the-last-start", "at-third-start"],
    }


def test_callers_below_the_first_function_start_are_unattributed():
    result = label_functions([BASE + 0x20], {BASE: ["orphan"], BASE + 0x20: ["owned"]})
    assert result == {BASE + 0x20: ["owned"], UNATTRIBUTED: ["orphan"]}
    assert UNATTRIBUTED in result


def test_no_function_list_at_all_leaves_everything_unattributed():
    assert label_functions([], {BASE: ["aa"], BASE + 0x10: ["bb"]}) == {
        UNATTRIBUTED: ["aa", "bb"]
    }


def test_functions_without_xrefs_are_absent():
    result = label_functions([BASE, BASE + 0x20, BASE + 0x40], {BASE + 0x20: ["only"]})
    assert result == {BASE + 0x20: ["only"]}
    assert BASE not in result and BASE + 0x40 not in result


def test_no_xrefs_yields_no_rows():
    assert label_functions([BASE, BASE + 0x20], {}) == {}
    assert label_functions([], {}) == {}


def test_duplicate_strings_within_a_function_are_reported_once():
    xrefs = {BASE + 0x10: ["same"], BASE + 0x20: ["same"], BASE + 0x30: ["other"]}
    assert label_functions([BASE], xrefs) == {BASE: ["other", "same"]}


def test_duplicate_pairs_in_the_input_are_pruned():
    xrefs = {BASE + 0x10: ["same", "same"], BASE + 0x20: ["same"]}
    assert label_functions([BASE], xrefs) == {BASE: ["same"]}


def test_a_single_string_value_is_accepted_as_a_one_element_list():
    assert label_functions([BASE], {BASE + 0x4: "solo"}) == {BASE: ["solo"]}


def test_labelled_strings_are_sorted_for_reading():
    xrefs = {BASE + 0x30: ["zeta"], BASE + 0x10: ["alpha"], BASE + 0x20: ["mid"]}
    assert label_functions([BASE], xrefs) == {BASE: ["alpha", "mid", "zeta"]}


def test_cap_defaults_to_fifty_strings_per_function():
    xrefs = {BASE + 0x10 * index: [f"s{index:02d}"] for index in range(60)}
    result = label_functions([BASE], xrefs)
    assert len(result[BASE]) == DEFAULT_MAX_STRINGS
    assert result[BASE] == [f"s{index:02d}" for index in range(DEFAULT_MAX_STRINGS)]


def test_cap_keeps_the_earliest_call_sites_and_sorts_them():
    xrefs = {BASE + 0x10 * index: [f"s{index:02d}"] for index in range(60)}
    assert label_functions([BASE], xrefs, max_strings=10)[BASE] == [
        f"s{index:02d}" for index in range(10)
    ]


def test_cap_above_the_ceiling_is_clamped_not_honoured():
    xrefs = {BASE + 0x10 * index: [f"s{index:02d}"] for index in range(60)}
    result = label_functions([BASE], xrefs, max_strings=MAX_STRINGS + 5000)
    assert len(result[BASE]) == 60


def test_cap_applies_to_the_unattributed_bucket_too():
    xrefs = {BASE + index: [f"o{index:02d}"] for index in range(20)}
    result = label_functions([BASE + 0x1000], xrefs, max_strings=5)
    assert result[UNATTRIBUTED] == [f"o{index:02d}" for index in range(5)]


def test_cap_is_independent_per_function():
    xrefs = {BASE + index: [f"o{index:02d}"] for index in range(8)}
    xrefs.update({BASE + 0x100 + index: [f"t{index:02d}"] for index in range(8)})
    result = label_functions([BASE, BASE + 0x100], xrefs, max_strings=3)
    assert result[BASE] == [f"o{index:02d}" for index in range(3)]
    assert result[BASE + 0x100] == [f"t{index:02d}" for index in range(3)]


def test_xref_strings_accepts_a_cap_and_stays_a_plain_dict():
    data = _build(leas=[(BASE + 0x10, STR_START)], strings=[(STR_START, "capped")])
    result = _scan(data, max_strings=1)
    assert type(result) is dict
    assert result == {BASE + 0x10: ["capped"]}
    assert _scan(data, max_strings=MAX_STRINGS + 10) == {BASE + 0x10: ["capped"]}


def test_xref_strings_cap_bounds_strings_per_caller_not_callers():
    """R1 bounds the strings *kept per caller*, never the number of callers.

    Each caller here is one proven ``lea`` and so owns exactly one proven string,
    which means the cap can only ever cut *inside* a caller's own list.  However
    small the cap is made, every caller survives: dropping callers instead of
    trimming a caller's list would be a different semantic than the one R1 states.
    """
    leas = [(BASE + 0x10 * index, BASE + 0x100 + 0x10 * index) for index in range(1, 8)]
    strings = [(BASE + 0x100 + 0x10 * index, f"str{index:02d}") for index in range(1, 8)]
    expected = {BASE + 0x10 * index: [f"str{index:02d}"] for index in range(1, 8)}
    data = _build(leas=leas, strings=strings)

    for cap in (1, 3, DEFAULT_MAX_STRINGS):
        bounded = _scan(data, max_strings=cap)
        # Per-caller bounding: no caller's list may exceed the cap ...
        assert all(len(values) <= cap for values in bounded.values())
        # ... and no caller is dropped for want of budget: caller-count
        # truncation at three would have silently lost four of these seven.
        assert len(bounded) == len(expected) == 7
        assert bounded == expected
        assert list(bounded) == sorted(bounded)

    # A cap of one still leaves every caller, with its one in-cap string.
    capped_to_one = _scan(data, max_strings=1)
    assert capped_to_one == expected
    # The default (50 strings per caller) bounds nothing here either.
    assert _scan(data) == expected

    # The same non-truncation holds once the callers are attributed: one function
    # per caller is the tightest possible bucket set, and a cap of one still
    # leaves every function present with its one proven string.
    funcs = sorted(expected)
    labelled = label_functions(funcs, capped_to_one, max_strings=1)
    assert sorted(labelled) == funcs
    assert sorted(labelled, key=lambda key: -1 if key is None else key) == funcs


def test_negative_xref_caller_is_a_caller_error():
    with pytest.raises(StrxrefError):
        label_functions([BASE], {-1: ["aaaa"]})


@pytest.mark.parametrize(
    "xrefs",
    [
        ["not", "a", "mapping"],
        5,
        None,
        "nope",
        {BASE: b"bytes"},
        {BASE: 42},
        {BASE: ["ok", 7]},
        {"x": ["aaaa"]},
    ],
)
def test_malformed_xrefs_are_caller_errors(xrefs):
    with pytest.raises(StrxrefError):
        label_functions([BASE], xrefs)


@pytest.mark.parametrize("funcs", [[BASE + 0x20, BASE], [BASE, BASE], [-1], [1.5], "funcs", 7, None])
def test_malformed_function_lists_are_caller_errors(funcs):
    with pytest.raises(StrxrefError):
        label_functions(funcs, {BASE: ["aaaa"]})


@pytest.mark.parametrize("bad", [0, -3, 2.5, "50", True])
def test_bad_cap_is_a_caller_error(bad):
    with pytest.raises(StrxrefError):
        label_functions([BASE], {BASE: ["aaaa"]}, max_strings=bad)


def test_end_to_end_scan_then_attribute():
    data = _build(
        leas=[
            (BASE + 0x10, STR_START),
            (BASE + 0x28, BASE + 0x110),
            (BASE + 0x70, BASE + 0x110),
        ],
        strings=[(STR_START, "alpha"), (BASE + 0x110, "bravo")],
    )
    xrefs = _scan(data)
    assert xrefs == {BASE + 0x10: ["alpha"], BASE + 0x28: ["bravo"], BASE + 0x70: ["bravo"]}
    funcs = [BASE, BASE + 0x40, BASE + 0x80]
    assert label_functions(funcs, xrefs) == {BASE: ["alpha", "bravo"], BASE + 0x40: ["bravo"]}


# --- 8. determinism and hygiene --------------------------------------------- #


def test_two_runs_over_the_same_bytes_are_equal():
    data = _build(
        leas=[(BASE + 0x10, STR_START), (BASE + 0x40, BASE + 0x120)],
        strings=[(STR_START, "one1"), (BASE + 0x120, "two2")],
        filler=0xCC,
    )
    first = _scan(data)
    second = _scan(bytes(bytearray(data)))
    assert first == second == _scan(memoryview(bytearray(data)))
    assert list(first) == sorted(first)
    assert all(values == sorted(values) for values in first.values())


def test_label_functions_output_is_order_independent():
    xrefs = {BASE + 0x30: ["ccc"], BASE + 0x10: ["aaa"], BASE + 0x20: ["bbb"]}
    forward = label_functions([BASE], xrefs)
    shuffled = label_functions([BASE], dict(reversed(list(xrefs.items()))))
    assert forward == shuffled == {BASE: ["aaa", "bbb", "ccc"]}


def test_module_source_is_pure_offline_and_faulthandler_free():
    source = Path(strxref.__file__).read_text(encoding="utf-8")
    assert re.search(r"^(?:import|from)\s+faulthandler", source, re.M) is None
    assert re.search(r"^(?:import|from)\s+sqlite3", source, re.M) is None
    assert (
        re.search(r"^(?:import|from)\s+(?:socket|ssl|urllib|http|requests|httpx)\b", source, re.M)
        is None
    )
    assert re.search(r"^(?:import|from)\s+subprocess", source, re.M) is None
    assert re.search(r"shell\s*=\s*True", source) is None
    assert "rip-relative-lea" in source


def test_module_imports_no_rtti_reader():
    source = Path(strxref.__file__).read_text(encoding="utf-8")
    assert re.search(r"^(?:import|from)\s+toothpick\.rtti", source, re.M) is None
    assert re.search(r"^(?:import|from)\s+\S*rtti\S*\s+import", source, re.M) is None
    assert "RttiError(" not in source
    assert "class StrxrefError" in source
