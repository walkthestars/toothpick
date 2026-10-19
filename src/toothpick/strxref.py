"""
Toothpick string-xref labeling -- RIP-relative ``lea`` scan plus function
attribution (UNIT-SEM5 Chunk 3).

Binding contract: ``docs/units/UNIT-SEM5-PLAN.md`` chunk 3 ("String-xref
labeling": a RIP-relative ``lea`` scan over supplied ``.text`` bytes against
caller-supplied string ranges yields per-caller string sets, and
``label_functions(funcs_sorted, xrefs)`` attributes each xref to its containing
function by binary search, stdlib only).  Design anchors:
``docs/design-document.md`` section 1 ("no fabricated structs -- every row
traces to parsed bytes") and section 19 (a label is only ever what was proven,
never an invented link), plus Risk R1 (bounded emission: default 50 strings per
caller / per function, ceiling 1000).

Two entry points:

* :func:`xref_strings` -- one linear pass over the caller-supplied ``text_range``
  bytes of a caller-supplied image (``data`` at ``base_addr``).  Two x64
  encodings are recognised, both RIP-relative ``lea`` forms carrying a
  mod=00 r/m=101 ModRM byte: the 7-byte ``REX 8D /r disp32`` (REX.W selects a
  64-bit destination; a REX without W selects the 32-bit destination but forms
  the same RIP + disp32 address) and the 6-byte REX-less ``8D /r disp32``.
  The ModRM test masks the reg field (``(modrm & 0xC7) == 0x05``), so every
  destination register -- ``rax``/``rbx``/``rcx``/``rsi``/``rdi`` and their
  REX.R extensions -- matches.  The target is the next-instruction address
  plus the signed 32-bit displacement.  A target that lands inside one of the
  ``string_ranges`` is read as a NUL-terminated printable-ASCII string; the
  result maps that *caller* (the address of the ``lea`` itself) to the sorted
  list of strings it loads.
* :func:`label_functions` -- attributes those caller-keyed xrefs to the function
  that contains each caller, using :func:`bisect.bisect_right` over the
  caller-supplied, strictly ascending function starts: ``func[i] <= caller <
  func[i + 1]``.

Honesty rules enforced here (design sections 1 and 19):

* Nothing is invented.  A string is only ever the bytes actually present
  between a resolved target and its terminating NUL, bounded by the string range
  it was found in and by the supplied buffer; a target that is not inside a
  supplied string range is simply not an xref (no guessing, no re-anchoring).
* A run of bytes that is not printable ASCII, an unterminated run, or a run
  shorter than ``min_len`` is **dropped**, never repaired or trimmed into a
  name.  A caller that references two distinct strings gets both; the same
  string twice is reported once (the caller address is the evidence).
* A caller below the first supplied function start is outside every supplied
  function; it is reported under the :data:`UNATTRIBUTED` key rather than being
  blamed on the nearest function.
* Emission is bounded (Risk R1) in both directions: :data:`DEFAULT_MAX_STRINGS`
  = 50 per caller and per function, clamped to :data:`MAX_STRINGS` = 1000 by
  :func:`resolve_max_strings`.  Under a cap the *earliest evidence* survives
  (hit order: address order inside :func:`xref_strings`, caller-address order
  inside :func:`label_functions`) and the kept list is sorted for reading.

Pure, offline and deterministic: bytes in, dictionaries out.  No file is read,
no network, no subprocess, no SQLite handle, no clock, no randomness.  Malformed
bytes are never fatal (the scan is over an untrusted code blob and simply finds
nothing there); malformed *arguments* are caller errors and raise
:class:`StrxrefError`.  Every dereference is bounds-checked against the supplied
buffer: a range that pokes past it is clamped to it, never believed.

This module is deliberately ``RttiError``-free: it defines its own
:class:`StrxrefError` (a :class:`ValueError`, mirroring
:class:`toothpick.demangle.DemangleError`) and imports nothing from
:mod:`toothpick.rtti_parse` or any other RTTI reader.

Trap hygiene: this module never imports ``faulthandler`` (Trap 11 -- suites run
under ``-p no:faulthandler``) and opens no SQLite handle, so the Trap 7
``[WinError 32]`` class of leak cannot originate here.
"""

from __future__ import annotations

import bisect
import operator
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Dict, List, Optional, Tuple, TypeGuard, Union

__all__ = [
    "DEFAULT_MAX_STRINGS",
    "LEA_LENGTH",
    "LEA_SHORT_LENGTH",
    "MAX_STRINGS",
    "MIN_STRING_LEN",
    "PRINTABLE_MAX",
    "PRINTABLE_MIN",
    "PROVENANCE",
    "StrxrefError",
    "UNATTRIBUTED",
    "label_functions",
    "resolve_max_strings",
    "xref_strings",
]

#: Default cap on strings kept per caller and per function (Risk R1).
DEFAULT_MAX_STRINGS = 50

#: Hard ceiling for those caps; a larger budget is clamped, never honoured.
MAX_STRINGS = 1000

#: Smallest string worth reporting (plan default ``min_len=4``).
MIN_STRING_LEN = 4

#: Printable-ASCII window a string body must stay inside (space .. tilde).
PRINTABLE_MIN = 0x20
PRINTABLE_MAX = 0x7E

#: Provenance label for every row this module produces.
PROVENANCE = "rip-relative-lea"

#: Key under which :func:`label_functions` reports callers that fall below the
#: lowest supplied function start (outside every supplied function).
UNATTRIBUTED: Optional[int] = None

# --- x64 RIP-relative ``lea`` encoding ------------------------------------- #

#: Any REX prefix byte (0x40-0x4F): bits 4-6 are the REX.R/X/B extension and
#: bit 3 (W) selects a 64-bit destination.  A REX without W is a
#: 32-bit-destination ``lea``, but the address it forms is still RIP + disp32,
#: so every REX prefix belongs to the 7-byte form.
REX_MIN = 0x40
REX_MAX = 0x4F

#: The ``lea`` opcode.
LEA_OPCODE = 0x8D

#: ModRM test for mod=00, r/m=101 -- the RIP-relative form.  The 0xC7 mask
#: clears the reg field (bits 5-3), so any destination register matches.
RIP_MODRM_MASK = 0xC7
RIP_MODRM_VALUE = 0x05

#: Length of the REX-prefixed form: REX + opcode + ModRM + disp32.
LEA_LENGTH = 7

#: Length of the REX-less 32-bit form: opcode + ModRM + disp32.
LEA_SHORT_LENGTH = 6

#: Offset of the signed 32-bit displacement inside each form.
DISP_OFFSET = 3
SHORT_DISP_OFFSET = 2
DISP_SIZE = 4

#: The address pair shapes a caller may spell a range in (start/end model).
_RangeLike = Union[Tuple[int, int], Sequence[Any], Mapping[str, Any], Any]


class StrxrefError(ValueError):
    """Raised for a caller argument that cannot describe a scannable image.

    Malformed *bytes* never raise -- an untrusted ``.text`` blob yields an empty
    result, not an exception.  Only malformed arguments (a non-bytes image, a
    negative base, a backwards range, a non-integer cap) land here.
    """


def resolve_max_strings(value: Any = None) -> int:
    """Coerce an R1 string cap: default, else clamp to :data:`MAX_STRINGS`.

    ``None`` means "use the default" (:data:`DEFAULT_MAX_STRINGS`).  A bool, a
    float, a string or any other non-integer spelling is a caller error and
    raises (``"50"`` and ``3.5`` are refused, never guessed); a non-positive
    integer raises; an over-large integer is clamped to :data:`MAX_STRINGS`, so
    the answer is bounded either way.
    """
    if value is None:
        return DEFAULT_MAX_STRINGS
    budget = _coerce_int(value, "max_strings")
    if budget <= 0:
        raise StrxrefError(f"max_strings must be positive, got {budget}")
    return min(budget, MAX_STRINGS)


def _coerce_int(value: Any, what: str) -> int:
    """An integer, or a refusal (booleans are not integers here)."""
    if isinstance(value, bool):
        raise StrxrefError(f"{what} must be an integer, got {value!r}")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise StrxrefError(f"{what} must be an integer, got {value!r}") from exc


def _coerce_min_len(value: Any) -> int:
    """Validate ``min_len``: a positive integer (0 is not a string length)."""
    length = _coerce_int(value, "min_len")
    if length < 1:
        raise StrxrefError(f"min_len must be positive, got {length}")
    return length


def _coerce_blob(data: Union[bytes, bytearray, memoryview]) -> bytes:
    """The supplied image as ``bytes`` (a read-only copy; never mutated)."""
    if isinstance(data, (bytes, bytearray, memoryview)):
        return bytes(data)
    raise StrxrefError(f"data must be bytes-like, got {type(data).__name__}")


def _coerce_base(base_addr: Any) -> int:
    """Validate the image base address (a non-negative integer)."""
    base = _coerce_int(base_addr, "base_addr")
    if base < 0:
        raise StrxrefError(f"base_addr must be non-negative, got {base}")
    return base


def _is_int(value: Any) -> TypeGuard[int]:
    """True for an ``int`` that is not a ``bool`` (and narrows the type)."""
    return isinstance(value, int) and not isinstance(value, bool)

def _looks_like_pair(value: Any) -> bool:

    if isinstance(value, (str, bytes, bytearray, dict)):
        return False
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        return False
    return _is_int(value[0]) and _is_int(value[1])


def _check_pair(start: int, end: int, what: str) -> Tuple[int, int]:
    """Validate one ``[start, end)`` interval."""
    if start < 0 or end < 0:
        raise StrxrefError(f"{what} addresses must be non-negative, got [{start}, {end})")
    if end < start:
        raise StrxrefError(f"{what} range [{start:#x}, {end:#x}) ends before it starts")
    return start, end


def _coerce_range_item(item: Any, what: str) -> Tuple[int, int]:
    """One ``[start, end)`` interval, however the caller spelled it."""
    if isinstance(item, bool) or isinstance(item, (str, bytes, bytearray)):
        raise StrxrefError(
            f"{what} entries must be (start, end) address pairs, got {item!r}"
        )
    if isinstance(item, Mapping):
        if "base" in item and "size" in item:
            start = _coerce_int(item["base"], f"{what} base")
            size = _coerce_int(item["size"], f"{what} size")
            if size < 0:
                raise StrxrefError(f"{what} size must be non-negative, got {size}")
            return _check_pair(start, start + size, what)
        if "start" in item and "end" in item:
            return _check_pair(
                _coerce_int(item["start"], f"{what} start"),
                _coerce_int(item["end"], f"{what} end"),
                what,
            )
        raise StrxrefError(
            f"{what} entries must be (start, end) pairs or mappings with "
            f"base/size or start/end, got {item!r}"
        )
    # Region-like objects (``base``/``size`` or ``start``/``end``) are accepted,
    # so a caller never has to re-encode a range it already holds.
    base = getattr(item, "base", None)
    size = getattr(item, "size", None)
    if _is_int(base) and _is_int(size):
        if size < 0:
            raise StrxrefError(f"{what} size must be non-negative, got {size}")
        return _check_pair(base, base + size, what)
    start = getattr(item, "start", None)
    end = getattr(item, "end", None)
    if _is_int(start) and _is_int(end):
        return _check_pair(start, end, what)
    if _looks_like_pair(item):
        return _check_pair(item[0], item[1], what)
    try:
        seq = list(item)
    except TypeError as exc:
        raise StrxrefError(
            f"{what} entries must be (start, end) address pairs, got {item!r}"
        ) from exc
    if len(seq) != 2:
        raise StrxrefError(
            f"{what} entries must be (start, end) address pairs, got {item!r}"
        )
    return _check_pair(
        _coerce_int(seq[0], f"{what} start"), _coerce_int(seq[1], f"{what} end"), what
    )


def _coerce_ranges(value: Any, what: str) -> List[Tuple[int, int]]:
    """Coerce ``None``/a pair/a sequence of pairs into ``[(start, end), ...]``.

    ``None`` and an empty sequence mean "no ranges" and are honest inputs: no
    text range yields no xrefs, no string range yields no strings, rather than a
    guess.  Malformed entries are caller errors and raise.
    """
    if value is None:
        return []
    if isinstance(value, bool) or isinstance(value, (str, bytes, bytearray)):
        raise StrxrefError(
            f"{what} must be a sequence of (start, end) address pairs, got {value!r}"
        )
    if isinstance(value, Mapping):
        # A lone start/end or base/size mapping describes one range.
        return [_coerce_range_item(value, what)]
    if _looks_like_pair(value):
        return [_coerce_range_item(value, what)]
    try:
        items = list(value)
    except TypeError as exc:
        # A lone region-like object (``base``/``size`` or ``start``/``end``) is a
        # legitimate spelling of exactly one range.
        try:
            return [_coerce_range_item(value, what)]
        except StrxrefError:
            raise StrxrefError(
                f"{what} must be a sequence of (start, end) address pairs, got {value!r}"
            ) from exc
    return [_coerce_range_item(item, what) for item in items]


def _disjoint_cover(ranges: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Sorted, disjoint intervals covering *ranges* without welding them together.

    Overlapping or nested supplied ranges are split at every boundary that
    matters, so a string is still required to terminate inside *one* supplied
    range; a plain merge would wrongly let it run across two.
    """
    boundaries = sorted({point for start, end in ranges for point in (start, end)})
    cover: List[Tuple[int, int]] = []
    for low, high in zip(boundaries, boundaries[1:]):
        if low == high:
            continue
        for start, end in ranges:
            if start <= low and high <= end:
                cover.append((low, high))
                break
    return cover


class _StringIndex:
    """Membership plus end-of-range lookup over disjoint string intervals."""

    __slots__ = ("starts", "ends")

    def __init__(self, ranges: Sequence[Tuple[int, int]]) -> None:
        cover = _disjoint_cover(ranges)
        self.starts = [start for start, _ in cover]
        self.ends = [end for _, end in cover]

    def limit(self, addr: int) -> Optional[int]:
        """Exclusive upper bound of the string range holding *addr*, or ``None``."""
        index = bisect.bisect_right(self.starts, addr) - 1
        if index < 0 or addr >= self.ends[index]:
            return None
        return self.ends[index]


def _finish(values: List[str], cap: int) -> List[str]:
    """Dedupe in evidence order, keep the first *cap* hits, sort for reading."""
    kept: List[str] = []
    for value in values:
        if value in kept:
            continue
        kept.append(value)
        if len(kept) >= cap:
            break
    kept.sort()
    return kept


def _match_lea(blob: bytes, base: int, addr: int) -> Optional[Tuple[int, int]]:
    """``(target, length)`` of a RIP-relative ``lea`` at *addr*, else ``None``.

    Two encodings are recognised, both with the mod=00 r/m=101 ModRM byte and
    any ``reg`` field (the test is ``(modrm & 0xC7) == 0x05``, so ``rbx``,
    ``rcx``, ``rsi``, ``rdi`` and the REX.R spellings match too):

    * ``REX 8D /r disp32`` -- 7 bytes.  REX.W is the 64-bit destination; a REX
      without W is the 32-bit destination, and the address formed is still
      RIP + sign-extended disp32.
    * ``8D /r disp32`` -- 6 bytes, the REX-less 32-bit form: the same address
      arithmetic without a prefix byte.

    ``None`` means "these bytes are not that instruction" -- including a
    truncated tail at the end of the buffer -- and never raises.
    """
    offset = addr - base
    if offset < 0 or offset >= len(blob):
        return None
    if REX_MIN <= blob[offset] <= REX_MAX:
        opcode_at = offset + 1
        if offset + 3 > len(blob):
            return None
        length = LEA_LENGTH
        disp_offset = DISP_OFFSET
    else:
        opcode_at = offset
        if offset + 2 > len(blob):
            return None
        length = LEA_SHORT_LENGTH
        disp_offset = SHORT_DISP_OFFSET
    if blob[opcode_at] != LEA_OPCODE:
        return None
    if (blob[opcode_at + 1] & RIP_MODRM_MASK) != RIP_MODRM_VALUE:
        return None
    if offset + length > len(blob):
        return None
    displacement = int.from_bytes(
        blob[offset + disp_offset : offset + disp_offset + DISP_SIZE],
        "little",
        signed=True,
    )
    return addr + length + displacement, length


def _read_string(
    blob: bytes, base: int, target: int, limit: int, min_len: int
) -> Optional[str]:
    """NUL-terminated printable-ASCII string at *target*, or ``None``.

    *limit* is the exclusive end of the string range the target was found in; the
    read is additionally clamped to the supplied buffer.  Anything that is not a
    terminated run of printable ASCII at least ``min_len`` long is refused --
    never truncated, never repaired, never decoded lossily.
    """
    offset = target - base
    if offset < 0 or offset >= len(blob):
        return None
    stop = min(limit, base + len(blob)) - base
    end = offset
    while end < stop and blob[end] != 0:
        if not PRINTABLE_MIN <= blob[end] <= PRINTABLE_MAX:
            return None
        end += 1
    if end >= stop or blob[end] != 0:
        return None
    if end - offset < min_len:
        return None
    return blob[offset:end].decode("ascii")


def xref_strings(
    data: Union[bytes, bytearray, memoryview],
    base_addr: Any,
    text_range: _RangeLike,
    string_ranges: Any,
    min_len: int = MIN_STRING_LEN,
    *,
    max_strings: Any = DEFAULT_MAX_STRINGS,
) -> Dict[int, List[str]]:
    """Scan x64 RIP-relative ``lea`` instructions for string references.

    ``data`` is the image blob and ``base_addr`` its virtual address, so every
    address in ``text_range``/``string_ranges`` is an absolute VA.  ``text_range``
    is the code interval to scan (one pair, or a sequence of pairs), and each
    supplied interval is scanned independently; ``string_ranges`` are the
    intervals a resolved target must land in to be read as a string (a target
    must also *terminate* inside the interval it was found in).  ``min_len`` is
    the shortest string worth reporting (plan default 4).

    Returns ``{caller_addr: [string, ...]}`` -- the address of the ``lea`` itself
    (the caller loading the string) to its sorted, deduped string list.  Keys are
    ascending; a caller whose list exceeds ``max_strings`` keeps the earliest
    hits (address order) up to :func:`resolve_max_strings`' bound.

    Determinism: one forward pass, a fixed pattern, sorted output -- two runs
    over the same bytes return equal dictionaries.  Malformed arguments (a
    non-bytes image, a negative base, a backwards or malformed range, a bad
    ``min_len``/``max_strings``) raise :class:`StrxrefError`; malformed *bytes*
    simply yield fewer (or zero) xrefs.
    """
    blob = _coerce_blob(data)
    base = _coerce_base(base_addr)
    floor = base
    ceiling = base + len(blob)
    text_ranges = _coerce_ranges(text_range, "text_range")
    string_range_list = _coerce_ranges(string_ranges, "string_ranges")
    length_floor = _coerce_min_len(min_len)
    cap = resolve_max_strings(max_strings)
    index = _StringIndex(string_range_list)

    hits: Dict[int, List[str]] = {}
    for start, end in text_ranges:
        addr = max(start, floor)
        stop = min(end, ceiling)
        while addr < stop:
            matched = _match_lea(blob, base, addr)
            if matched is None:
                # Not an instruction we recognise: step one byte so a real ``lea``
                # that starts later in the stream is still reached.
                addr += 1
                continue
            target, length = matched
            cursor = addr + length
            if cursor > stop:
                # The instruction straddles the end of this supplied text range,
                # so it is not wholly inside the interval we were asked to scan.
                addr = cursor
                continue
            limit = index.limit(target)
            if limit is not None:
                text = _read_string(blob, base, target, limit, length_floor)
                if text is not None:
                    hits.setdefault(addr, []).append(text)
            # Advance past the whole instruction (6 or 7 bytes, as matched), so
            # its displacement bytes are never re-read as opcodes.
            addr = cursor

    return {caller: _finish(values, cap) for caller, values in sorted(hits.items())}


def _coerce_addrs(value: Any, what: str) -> List[int]:
    """Strictly ascending, non-negative function starts (a sorted func list)."""
    if isinstance(value, bool) or isinstance(value, (str, bytes, bytearray, Mapping)):
        raise StrxrefError(f"{what} must be a sequence of addresses, got {value!r}")
    try:
        items = list(value)
    except TypeError as exc:
        raise StrxrefError(
            f"{what} must be a sequence of addresses, got {value!r}"
        ) from exc
    addrs: List[int] = []
    previous: Optional[int] = None
    for item in items:
        addr = _coerce_int(item, what)
        if addr < 0:
            raise StrxrefError(f"{what} addresses must be non-negative, got {addr}")
        if previous is not None and addr <= previous:
            raise StrxrefError(
                f"{what} must be strictly ascending, got {previous:#x} then {addr:#x}"
            )
        addrs.append(addr)
        previous = addr
    return addrs


def _coerce_strings(value: Any, what: str) -> List[str]:
    """A single string or a sequence of strings, strictly typed."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, (bytes, bytearray)):
        raise StrxrefError(f"{what} must hold text strings, got {type(value).__name__}")
    try:
        items = list(value)
    except TypeError as exc:
        raise StrxrefError(
            f"{what} must hold text strings, got {type(value).__name__}"
        ) from exc
    out: List[str] = []
    for item in items:
        if not isinstance(item, str):
            raise StrxrefError(f"{what} must hold str, got {type(item).__name__}")
        out.append(item)
    return out


def _coerce_xrefs(xrefs: Any) -> List[Tuple[int, str]]:
    """Flatten a caller-keyed xref mapping into sorted, deduped pairs."""
    if not isinstance(xrefs, Mapping):
        raise StrxrefError(
            f"xrefs must be a mapping of caller address to strings, "
            f"got {type(xrefs).__name__}"
        )
    pairs: List[Tuple[int, str]] = []
    for caller, texts in xrefs.items():
        addr = _coerce_int(caller, "xrefs caller")
        if addr < 0:
            raise StrxrefError(f"xrefs caller addresses must be non-negative, got {addr}")
        for text in _coerce_strings(texts, f"xrefs[0x{addr:x}]"):
            pairs.append((addr, text))
    pairs.sort()
    deduped: List[Tuple[int, str]] = []
    for pair in pairs:
        if not deduped or deduped[-1] != pair:
            deduped.append(pair)
    return deduped


def label_functions(
    sorted_func_addrs: Iterable[int],
    xrefs: Mapping[int, Sequence[str]],
    *,
    max_strings: Any = DEFAULT_MAX_STRINGS,
) -> Dict[Optional[int], List[str]]:
    """Attribute each caller-keyed xref to the function that contains it.

    ``sorted_func_addrs`` must be strictly ascending (a sorted function-start
    list, the shape a symbol table or a disassembler index yields); ``xrefs`` is
    a mapping of caller address to strings, exactly what :func:`xref_strings`
    returns.  Attribution is a binary search over the function starts:
    ``func[i] <= caller < func[i + 1]``, with the last function open-ended.

    Returns ``{func_start: [string, ...]}``: functions that own no xref are
    absent (no invented rows), lists are deduped and sorted, and keys are
    ascending.  Callers below the lowest supplied function start sit in no
    supplied function and are reported under :data:`UNATTRIBUTED` (``None``)
    rather than being blamed on the nearest function; a caller with no function
    list at all is likewise unattributed.

    Emission is bounded like :func:`xref_strings` (Risk R1: default
    :data:`DEFAULT_MAX_STRINGS` = 50 strings per function, clamped to
    :data:`MAX_STRINGS` = 1000): under a cap the earliest call sites (ascending
    caller address) survive and the kept list is sorted for reading.

    Malformed arguments -- a non-mapping ``xrefs``, non-integer or negative
    addresses, a function list that is not strictly ascending, a bad
    ``max_strings`` -- raise :class:`StrxrefError`.  Function attribution is
    all this does: no symbol is read, no name is invented.
    """
    funcs = _coerce_addrs(sorted_func_addrs, "sorted_func_addrs")
    pairs = _coerce_xrefs(xrefs)
    cap = resolve_max_strings(max_strings)

    labelled: Dict[int, List[str]] = {}
    orphans: List[str] = []
    for caller, text in pairs:
        index = bisect.bisect_right(funcs, caller) - 1
        if index < 0:
            orphans.append(text)
            continue
        labelled.setdefault(funcs[index], []).append(text)

    result: Dict[Optional[int], List[str]] = {
        addr: _finish(values, cap) for addr, values in sorted(labelled.items())
    }
    if orphans:
        result[UNATTRIBUTED] = _finish(orphans, cap)
    return result
