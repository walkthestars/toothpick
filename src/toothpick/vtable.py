"""
Toothpick vtable discovery -- shape pass plus COL confirmation (unit VTABLE).

Binding contract: ``docs/units/UNIT-VTABLE-PLAN.md`` (the POW-002 follow-up
unit).  Design anchors: ``docs/design-document.md`` section 1 ("no fabricated
structs -- every class row traces to parsed bytes") and section 19 / Risk R1
(bounded emission: default 50 candidates, ceiling 1000).  Ground truth for the
drive is ``docs/field/hoi4/README.md`` Open Q2 (are ``vftable[-1]`` -> COL
references truly absent on HOI4, or is the scan missing them?).

Method, in the order the plan fixes it:

1. **Shape pass** -- a maximal run of >= 3 consecutive pointer-sized values
   (8-byte slots on x64, 4-byte slots on x86: the MSVC ABI slot size) whose
   every value resolves into a caller-supplied executable range
   (``text_ranges``) and whose slots lie inside a caller-supplied search range
   (``search_ranges``, the ``.rdata``/``.data``-class regions).  Runs start at
   an aligned address and each maximal run yields one candidate anchored at its
   first slot.
2. **Confirmation** -- the pointer at ``candidate - pointer_size`` (the
   ``vftable[-1]`` slot) must decode as a validated COL: the full COL -> TD ->
   CHD walk including the base-class array, executed by the certified readers
   in :mod:`toothpick.rtti_parse` (:func:`~toothpick.rtti_parse.scan_vtables`,
   so the field-certified x64 +24 base-array fallback is reused and not
   re-implemented).  A candidate whose slot parses is labelled
   :data:`CONFIDENCE_CONFIRMED` and carries the recovered class identity; every
   other candidate is labelled :data:`CONFIDENCE_SHAPE_ONLY`, carrying the
   reader's rejection reason as evidence, and is **never promoted** to a
   hookable address (:func:`hookable_addresses` returns confirmed rows only).
3. **Bounds** -- R1 caps (default :data:`DEFAULT_MAX_VTABLES` = 50, ceiling
   :data:`MAX_VTABLES` = 1000), deterministic address order, and every
   dereference bounds-checked against the caller-supplied buffer
   (:class:`~toothpick.rtti_parse.RttiError`).  A search range that pokes past
   the supplied buffer is clamped to it, never believed.

4. **Entry targets** (UNIT-SEM5 chunk 1, additive) -- each candidate records the
   first :data:`MAX_ENTRIES` values read for it during the shape pass
   (``entries``, slot order, bounded), and :func:`entry_targets` returns the
   deduped ascending targets of confirmed rows only.  The recorded values are the
   bytes actually read out of the caller's buffer -- nothing is synthesized.

Negative-result clause (plan): POW-002 found *zero* COL-linked vtables under
strict scans.  If the drive reproduces that on HOI4, the honest answer is the
catalogue this module returns -- a confirmed count of 0 plus the anchor
evidence, with shape-only rows clearly labelled -- rather than loosening
criteria until something matches.  Nothing here promotes a shape-only row,
invents a class name, or reads a file: the caller supplies ``data`` and
``base_addr``.

Trap hygiene: this module never imports ``faulthandler`` (Trap 11 -- suites run
under ``-p no:faulthandler``) and opens no SQLite handle at all (Trap 7).
"""

from __future__ import annotations

import bisect
import operator
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    TypeGuard,
    Union,
)

from toothpick.rtti_parse import (
    DEFAULT_MAX_VTABLES,
    MAX_VTABLES,
    RttiArch,
    RttiError,
    resolve_max_vtables,
    scan_vtables,
)

__all__ = [
    "CONFIDENCE_CONFIRMED",
    "CONFIDENCE_SHAPE_ONLY",
    "DEFAULT_MAX_VTABLES",
    "MAX_ENTRIES",
    "MAX_VTABLES",
    "MIN_RUN_LENGTH",
    "NEGATIVE_RESULT_NOTE",
    "PROVENANCE",
    "VtableCandidate",
    "VtableScanResult",
    "discover_vtables",
    "entry_targets",
    "hookable_addresses",
    "summarize_discovery",
]

#: Shortest shape run that counts as a vtable (plan step 1: "runs of >= 3").
MIN_RUN_LENGTH = 3

#: UNIT-SEM5 chunk 1: how many shape-pass target values a candidate records (the
#: first ``MAX_ENTRIES`` slots of its run, in slot order).  The bound keeps a
#: pathological run from growing an unbounded tuple.
MAX_ENTRIES = 32

#: Confidence label for a candidate whose ``vftable[-1]`` slot is a validated COL.
CONFIDENCE_CONFIRMED = "confirmed"

#: Confidence label for a shape run with no validated COL link (catalogued, never
#: promoted, never presented as a hookable address).
CONFIDENCE_SHAPE_ONLY = "shape-only"

#: Provenance string for catalogued candidates.
PROVENANCE = "vtable-shape+col"

#: The plan's negative-result clause, verbatim in spirit: reported, not patched
#: around.  :func:`summarize_discovery` carries it so a drive transcript can
#: never report a zero-result as a failure to look.
NEGATIVE_RESULT_NOTE = (
    "zero confirmed candidates is a reportable result: shape-only entries are "
    "catalogued, labelled and never promoted, and the discovery criteria are not "
    "loosened to manufacture a match"
)

_COL_POINTER_WHAT = "vftable[-1] CompleteObjectLocator pointer"
_SLOT_WHAT = "vtable shape slot"


# --------------------------------------------------------------------------- #
# Data contracts
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class VtableCandidate:
    """One discovered vtable candidate, with its confidence and evidence.

    ``addr`` is the first slot of the shape run (absolute virtual address);
    ``slot_count`` is how many consecutive pointer slots resolved into
    ``text_ranges``.  ``confidence`` is :data:`CONFIDENCE_CONFIRMED` only when
    the ``vftable[-1]`` slot decoded as a full COL -> TD -> CHD walk; otherwise
    :data:`CONFIDENCE_SHAPE_ONLY` and ``reason`` holds the reader's refusal.
    The recovered class fields stay empty for a shape-only row -- no name is
    ever invented for one.
    ``entries`` carries the first :data:`MAX_ENTRIES` target values read for this
    candidate during the shape pass, in slot order (empty for a hand-built row).
    It is evidence, not a link: :func:`entry_targets` reads it for confirmed rows
    only.
    """

    addr: int
    confidence: str
    slot_count: int
    arch: str = ""
    col_addr: int = 0
    col_pointer: int = 0
    mangled: str = ""
    name: str = ""
    demangled: bool = False
    bases: Tuple[str, ...] = ()
    td_addr: int = 0
    chd_addr: int = 0
    reason: str = ""

    entries: Tuple[int, ...] = ()

    @property
    def hookable(self) -> bool:
        """True only for a confirmed row (shape-only is never hookable)."""
        return self.confidence == CONFIDENCE_CONFIRMED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "addr": self.addr,
            "addr_hex": f"0x{self.addr:x}",
            "confidence": self.confidence,
            "hookable": self.hookable,
            "slot_count": self.slot_count,
            "entries": list(self.entries),
            "entries_hex": [f"0x{value:x}" for value in self.entries],
            "arch": self.arch,
            "col_addr": self.col_addr,
            "col_pointer": self.col_pointer,
            "mangled": self.mangled,
            "name": self.name,
            "demangled": self.demangled,
            "bases": list(self.bases),
            "td_addr": self.td_addr,
            "chd_addr": self.chd_addr,
            "reason": self.reason,
        }


class VtableScanResult(NamedTuple):
    """``(candidates, rejected)`` -- unpackable, and attribute-addressable.

    ``candidates`` is the bounded, address-ordered catalogue; ``rejected`` holds
    ``(address, reason)`` rows for candidates dropped by the R1 cap (a bounded
    answer is never presented as an exhaustive one).
    """

    candidates: List[VtableCandidate]
    rejected: List[Tuple[int, str]]

    @property
    def confirmed(self) -> List[VtableCandidate]:
        """Candidates whose ``vftable[-1]`` slot is a validated COL."""
        return [item for item in self.candidates if item.hookable]

    @property
    def shape_only(self) -> List[VtableCandidate]:
        """Candidates catalogued on shape alone (clearly labelled, never promoted)."""
        return [
            item for item in self.candidates if item.confidence == CONFIDENCE_SHAPE_ONLY
        ]

    @property
    def confirmed_count(self) -> int:
        return len(self.confirmed)

    @property
    def shape_only_count(self) -> int:
        return len(self.shape_only)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidates": [item.to_dict() for item in self.candidates],
            "confirmed_count": self.confirmed_count,
            "shape_only_count": self.shape_only_count,
            "rejected": [
                {"addr": addr, "addr_hex": f"0x{addr:x}", "reason": reason}
                for addr, reason in self.rejected
            ],
            "note": NEGATIVE_RESULT_NOTE,
        }


def hookable_addresses(result: VtableScanResult) -> Tuple[int, ...]:
    """Addresses of *confirmed* candidates only, ascending.

    This is the whole "never promoted" rule in one call: a shape-only row is
    catalogued evidence, not a hookable address, so it is excluded here.
    """
    if not isinstance(result, VtableScanResult):
        raise RttiError(
            f"hookable_addresses expects a VtableScanResult, got {type(result).__name__}"
        )
    return tuple(sorted(item.addr for item in result.confirmed))


def entry_targets(result: VtableScanResult) -> Tuple[int, ...]:
    """Deduped, ascending entry targets of *confirmed* candidates only.

    The targets are the values the shape pass read out of the caller's buffer for
    a confirmed candidate (bounded to the first :data:`MAX_ENTRIES` slots), so a
    caller can intersect them with static functions without a shape-only row's
    evidence ever being promoted to an address.  Duplicates collapse across
    candidates; the order is ascending and deterministic.
    """
    if not isinstance(result, VtableScanResult):
        raise RttiError(
            f"entry_targets expects a VtableScanResult, got {type(result).__name__}"
        )
    targets: set[int] = set()
    for item in result.confirmed:
        targets.update(item.entries)
    return tuple(sorted(targets))


def summarize_discovery(result: VtableScanResult) -> Dict[str, Any]:
    """Counts plus the plan's negative-result clause, for honest reporting."""
    if not isinstance(result, VtableScanResult):
        raise RttiError(
            f"summarize_discovery expects a VtableScanResult, got {type(result).__name__}"
        )
    return {
        "catalogue_size": len(result.candidates),
        "confirmed": result.confirmed_count,
        "shape_only": result.shape_only_count,
        "rejected": len(result.rejected),
        "confirmed_addresses": [f"0x{addr:x}" for addr in hookable_addresses(result)],
        "shape_only_addresses": [f"0x{item.addr:x}" for item in result.shape_only],
        "min_run_length": MIN_RUN_LENGTH,
        "provenance": PROVENANCE,
        "method": (
            "shape pass (maximal runs of >= 3 pointer slots resolving into "
            "text_ranges, inside search_ranges) + COL confirmation via the "
            "rtti_parse COL -> TD -> CHD walk"
        ),
        "note": NEGATIVE_RESULT_NOTE,
    }


# --------------------------------------------------------------------------- #
# Argument coercion (bounds-checked: a caller error raises, nothing is guessed)
# --------------------------------------------------------------------------- #


def _coerce_int(value: Any, what: str) -> int:
    """An integer address/offset, or a refusal (booleans are not integers here)."""
    if isinstance(value, bool):
        raise RttiError(f"{what} must be an integer, got {value!r}")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise RttiError(f"{what} must be an integer, got {value!r}") from exc


def _resolve_budget(value: Any, what: str = "max_vtables") -> int:
    """Validate the R1 cap strictly, then clamp it to the ceiling.

    ``None`` means "use the default".  A bool, a string, a float or any other
    non-integer spelling is a *caller error* and raises rather than being
    coerced (``"50"`` and ``3.5`` are refused, never guessed at); a
    non-positive integer raises; an over-large integer is clamped to
    :data:`MAX_VTABLES`.  Once the type is settled the clamp itself is
    delegated to :func:`~toothpick.rtti_parse.resolve_max_vtables`, so the
    ceiling lives in exactly one place.
    """
    if value is None:
        return resolve_max_vtables(None)
    budget = _coerce_int(value, what)
    if budget <= 0:
        raise RttiError(f"{what} must be positive, got {budget}")
    return resolve_max_vtables(budget)


def _coerce_image(
    data: Union[bytes, bytearray, memoryview], base_addr: Any
) -> Tuple[bytes, int]:
    """Validate the supplied image and its base address."""
    if isinstance(data, (bytes, bytearray, memoryview)):
        blob = bytes(data)
    else:
        raise RttiError(f"data must be bytes-like, got {type(data).__name__}")
    base = _coerce_int(base_addr, "base_addr")
    if base < 0:
        raise RttiError(f"base_addr must be non-negative, got {base}")
    return blob, base


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)
def _is_int(value: Any) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)

def _looks_like_pair(value: Any) -> bool:
    """True for a lone ``(start, end)`` pair of integers (not a list of them)."""
    if isinstance(value, (str, bytes, bytearray, dict)):
        return False
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        return False
    return _is_int(value[0]) and _is_int(value[1])


def _coerce_range_item(item: Any, what: str) -> Tuple[int, int]:
    """One ``[start, end)`` address interval, however the caller spelled it."""
    if isinstance(item, bool) or isinstance(item, (str, bytes, bytearray)):
        raise RttiError(
            f"{what} entries must be (start, end) address pairs, got {item!r}"
        )
    if isinstance(item, dict):
        if "base" in item and "size" in item:
            start = _coerce_int(item["base"], f"{what} base")
            size = _coerce_int(item["size"], f"{what} size")
            if size < 0:
                raise RttiError(f"{what} size must be non-negative, got {size}")
            return start, start + size
        if "start" in item and "end" in item:
            start = _coerce_int(item["start"], f"{what} start")
            end = _coerce_int(item["end"], f"{what} end")
            return _check_pair(start, end, what)
        raise RttiError(
            f"{what} entries must be (start, end) pairs or objects with base/size, "
            f"got {item!r}"
        )
    # ``toothpick.scanner.MemoryRegion``-style objects (base + size) and any
    # start/end-bearing region are accepted, so callers never re-encode ranges.
    base = getattr(item, "base", None)
    size = getattr(item, "size", None)
    if _is_int(base) and _is_int(size):
        if size < 0:
            raise RttiError(f"{what} size must be non-negative, got {size}")
        return base, base + size
    start = getattr(item, "start", None)
    end = getattr(item, "end", None)
    if _is_int(start) and _is_int(end):
        return _check_pair(start, end, what)
    if _looks_like_pair(item):
        return _check_pair(item[0], item[1], what)
    try:
        seq = list(item)
    except TypeError as exc:
        raise RttiError(
            f"{what} entries must be (start, end) address pairs, got {item!r}"
        ) from exc
    if len(seq) != 2:
        raise RttiError(
            f"{what} entries must be (start, end) address pairs, got {item!r}"
        )
    return _check_pair(
        _coerce_int(seq[0], f"{what} start"), _coerce_int(seq[1], f"{what} end"), what
    )


def _check_pair(start: int, end: int, what: str) -> Tuple[int, int]:
    if start < 0 or end < 0:
        raise RttiError(f"{what} addresses must be non-negative, got [{start}, {end})")
    if end < start:
        raise RttiError(f"{what} range [{start:#x}, {end:#x}) ends before it starts")
    return start, end


def _coerce_ranges(value: Any, what: str) -> List[Tuple[int, int]]:
    """Coerce ``None``/a pair/a sequence of pairs into ``[(start, end), ...]``.

    ``None`` and an empty sequence mean "no ranges" and are honest inputs: no
    executable or no searchable range yields an empty catalogue rather than a
    guess.  Malformed entries are caller errors and raise.
    """
    if value is None:
        return []
    if isinstance(value, bool) or isinstance(value, (str, bytes, bytearray)):
        raise RttiError(
            f"{what} must be a sequence of (start, end) address pairs, got {value!r}"
        )
    if _looks_like_pair(value):
        return [_coerce_range_item(value, what)]
    try:
        items = list(value)
    except TypeError as exc:
        raise RttiError(
            f"{what} must be a sequence of (start, end) address pairs, got {value!r}"
        ) from exc
    return [_coerce_range_item(item, what) for item in items]


def _merge_ranges(ranges: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Sorted, non-overlapping cover of ``ranges`` (deterministic order)."""
    merged: List[Tuple[int, int]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            prev_start, prev_end = merged[-1]
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


class _RangeIndex:
    """Half-open interval membership over a sorted, non-overlapping cover."""

    __slots__ = ("ranges", "starts")

    def __init__(self, ranges: Sequence[Tuple[int, int]]) -> None:
        self.ranges = list(ranges)
        self.starts = [start for start, _ in self.ranges]

    def contains(self, addr: int) -> bool:
        index = bisect.bisect_right(self.starts, addr) - 1
        return index >= 0 and addr < self.ranges[index][1]


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _read_pointer(blob: bytes, base: int, addr: int, size: int, what: str) -> int:
    """Bounds-checked pointer-sized little-endian read (never reads past ``blob``)."""
    offset = addr - base
    if offset < 0 or offset + size > len(blob):
        raise RttiError(
            f"{what}: address 0x{addr:x} (+{size} bytes) is outside the supplied "
            f"buffer [0x{base:x}, 0x{base + len(blob):x})"
        )
    return int.from_bytes(blob[offset : offset + size], "little")


# --------------------------------------------------------------------------- #
# Shape pass
# --------------------------------------------------------------------------- #


def _shape_runs(
    blob: bytes,
    base: int,
    pointer: int,
    start: int,
    stop: int,
    text: _RangeIndex,
) -> List[Tuple[int, int, Tuple[int, ...]]]:
    """Maximal runs of >= :data:`MIN_RUN_LENGTH` text pointers in ``[start, stop)``.

    Addresses advance on the ABI alignment (pointer size), so a misaligned run of
    values is never mistaken for a vtable.  Returns ``(run_start, slot_count,
    entries)``: ``entries`` holds the target values read for the run in slot order,
    capped at the first :data:`MAX_ENTRIES` slots so the tuple stays bounded.
    """
    runs: List[Tuple[int, int, Tuple[int, ...]]] = []
    addr = _align_up(start, pointer)
    run_start: Optional[int] = None
    entries: List[int] = []
    slots = 0
    while addr + pointer <= stop:
        value = _read_pointer(blob, base, addr, pointer, _SLOT_WHAT)
        if text.contains(value):
            if run_start is None:
                run_start = addr
                entries = []
                slots = 0
            slots += 1
            if len(entries) < MAX_ENTRIES:
                entries.append(value)
        else:
            if run_start is not None and slots >= MIN_RUN_LENGTH:
                runs.append((run_start, slots, tuple(entries)))
            run_start = None
            entries = []
            slots = 0
        addr += pointer
    if run_start is not None and slots >= MIN_RUN_LENGTH:
        runs.append((run_start, slots, tuple(entries)))
    return runs


def _classify(
    blob: bytes, base: int, engine: RttiArch, addr: int, slots: int,
    entries: Tuple[int, ...] = (),
) -> VtableCandidate:
    """Confirm ``addr`` against its ``vftable[-1]`` COL, or label it shape-only.

    ``entries`` are the target values the shape pass read for this run (bounded
    by :data:`MAX_ENTRIES`); they are recorded for shape-only rows too, since
    they are what was seen, while :func:`entry_targets` promotes confirmed rows
    only.
    """
    pointer = engine.pointer_size
    col_pointer = 0
    reason = ""
    try:
        col_pointer = _read_pointer(
            blob, base, addr - pointer, pointer, _COL_POINTER_WHAT
        )
    except RttiError as exc:
        reason = str(exc)
    if not reason:
        # The certified reader path: COL -> TD -> CHD plus the base-class array
        # walk (including the field-certified x64 +24 fallback).  A rejection is
        # evidence for a shape-only label, never a promoted row.
        scan = scan_vtables(blob, base, engine, [addr], max_vtables=1)
        if scan.classes:
            record = scan.classes[0]
            return VtableCandidate(
                addr=addr,
                confidence=CONFIDENCE_CONFIRMED,
                slot_count=slots,
                entries=entries,
                arch=engine.value,
                col_addr=record.col_addr,
                col_pointer=col_pointer,
                mangled=record.mangled,
                name=record.name,
                demangled=bool(record.demangled),
                bases=tuple(item.mangled for item in record.bases),
                td_addr=record.td_addr,
                chd_addr=record.chd_addr,
            )
        reason = (
            scan.rejected[0][1]
            if scan.rejected
            else "COL walk rejected the candidate without a reason"
        )
    return VtableCandidate(
        addr=addr,
        confidence=CONFIDENCE_SHAPE_ONLY,
        slot_count=slots,
        entries=entries,
        arch=engine.value,
        col_pointer=col_pointer,
        reason=reason,
    )


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def discover_vtables(
    data: Union[bytes, bytearray, memoryview],
    base_addr: int,
    arch: Any,
    text_ranges: Any,
    search_ranges: Any,
    max_vtables: Optional[int] = None,
) -> VtableScanResult:
    """Discover vtable candidates in ``data`` (unit VTABLE, plan step 1 + 2).

    ``data``/``base_addr`` describe the caller-supplied image; ``arch`` selects
    the MSVC ABI slot size and link rules (``'x64'``/``'x86'``); ``text_ranges``
    are the executable intervals a shape slot may point into; ``search_ranges``
    are the ``.rdata``/``.data``-class intervals whose aligned slots are examined
    (interval entries are ``(start, end)`` half-open pairs, ``base``/``size``
    region objects, or ``start``/``end`` mappings).

    Returns :class:`VtableScanResult`.  ``candidates`` is address-ordered and
    bounded by the R1 cap (``max_vtables``: default :data:`DEFAULT_MAX_VTABLES`
    = 50, clamped to :data:`MAX_VTABLES` = 1000); under a cap, confirmed rows are
    selected before shape-only ones (address-ordered within each class) and the
    returned catalogue is re-sorted by address, so the order is deterministic
    while a confirmed link is never dropped for a shape-only run.  Every dropped
    candidate is reported in ``rejected`` with an explicit cap reason.

    Ranges outside the supplied buffer are clamped to it (bounds-checked, never
    believed), a malformed range or a bad ``max_vtables`` raises
    :class:`~toothpick.rtti_parse.RttiError`, and no structure is parsed outside
    the buffer: the readers refuse and the candidate becomes ``shape-only``.

    Each candidate also carries the first :data:`MAX_ENTRIES` values read for it
    during the shape pass (``entries``); :func:`entry_targets` dedupes and sorts
    those targets across confirmed rows only.
    """
    engine = RttiArch.coerce(arch)
    budget = _resolve_budget(max_vtables)
    blob, base = _coerce_image(data, base_addr)
    executable = _merge_ranges(_coerce_ranges(text_ranges, "text_ranges"))
    searchable = _merge_ranges(_coerce_ranges(search_ranges, "search_ranges"))
    if not blob or not executable or not searchable:
        # Nothing executable to point at, nothing searchable to scan, or an
        # empty image: an empty catalogue is the honest answer.
        return VtableScanResult(candidates=[], rejected=[])

    pointer = engine.pointer_size
    image_end = base + len(blob)
    text = _RangeIndex(executable)

    runs: List[Tuple[int, int, Tuple[int, ...]]] = []
    for range_start, range_end in searchable:
        start = max(range_start, base)
        stop = min(range_end, image_end)
        if stop - start < pointer:
            continue
        runs.extend(_shape_runs(blob, base, pointer, start, stop, text))
    runs.sort()

    discovered: Dict[int, VtableCandidate] = {}
    for addr, slots, entries in runs:
        discovered[addr] = _classify(blob, base, engine, addr, slots, entries)

    ordered = sorted(discovered.values(), key=lambda item: item.addr)
    rejected: List[Tuple[int, str]] = []
    if len(ordered) > budget:
        selection = sorted(
            ordered, key=lambda item: (0 if item.hookable else 1, item.addr)
        )
        kept = {item.addr for item in selection[:budget]}
        for item in ordered:
            if item.addr in kept:
                continue
            rejected.append(
                (
                    item.addr,
                    f"vtable cap reached: {budget} of {len(ordered)} candidates kept "
                    f"under the R1 bound (ceiling {MAX_VTABLES}); dropped "
                    f"{item.confidence} candidate at 0x{item.addr:x}",
                )
            )
        ordered = [item for item in ordered if item.addr in kept]
    return VtableScanResult(candidates=ordered, rejected=rejected)
