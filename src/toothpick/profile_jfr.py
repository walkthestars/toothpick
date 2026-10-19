"""Toothpick J2 profile formats -- async-profiler collapsed + JFR JSON (P1 family 2).

Binding contract: ``docs/units/UNIT-J2-JFR-PLAN.md`` section 1 (Chunk 1 module) and
section 2 (interface specification); milestone J2 of ``docs/NEXT-TRACK.md``
("JFR/async-profiler import.  Second format for P1's importer.  Accept: synthetic
JFR attributes samples + allocation.").  Design anchors: ``docs/design-document.md``
sections 1 ("no fabricated structs -- every row traces to parsed bytes") and 19.

This is the second *format family* of the P1 profile importer.  It emits the very
same row shape as :mod:`toothpick.profile` -- :class:`~toothpick.profile.ProfileSample`,
**imported and never redefined** -- so every consumer that already handles a P1
parser output (:func:`toothpick.profile.import_samples`, ``func_profile.hotness``,
``hottest``) consumes these two parsers unchanged::

    parse_async_collapsed(text) -> (samples, diagnostics)   # async-profiler
    parse_jfr_json(text)        -> (samples, diagnostics)   # jfr print --json
    detect_format(text)         -> "async" | "jfr" | None   # CLI --format auto

The four invariants of the P1 importer are carried over unchanged:

* **Epistemic (never fabricate).**  A symbol is copied from the profile verbatim,
  and an address is emitted *only* when the profile carries one.  ``func+0x10`` is
  a symbol, not an address: the module never subtracts an offset to invent a
  function start, and it never invents a name for an address-only frame.
* **Counted, never fatal.**  A malformed collapsed line increments ``rejected``;
  an unknown JFR event type increments ``ignored``.  Neither aborts an import.
  Only structurally impossible input -- text that is not JSON, or JSON carrying no
  ``events`` anywhere -- raises :class:`~toothpick.profile.ProfileError`.
* **Deterministic.**  Rows are sorted by the P1 address key
  (:func:`toothpick.profile._sort_key`), JFR aggregation is keyed on
  ``(symbol, func_addr)`` with integer accumulators, and diagnostics dictionaries
  are built in a fixed key order.  No clock, no randomness, no network.
* **Trap 14.**  Diagnostics and sample views are JSON-plain: ``list``/``dict``
  plus str/int/bool/None scalars -- never tuples, never sets -- so
  ``json.dumps`` round-trips everything this module returns.

Formats, precisely:

* **async-profiler collapsed** -- ``frame;frame;frame <count> [<alloc>]``: the
  whitespace-separated tail is the sample count and, when present, the allocation
  column (``<count> <alloc>``, ``<count> <alloc>B``, or the attached
  ``<count>_<alloc>B`` spelling).  The **leaf frame** (last ``;``-separated frame)
  is the symbol, and its address is parsed when the frame is spelled ``0x...``
  (optionally bracketed, as async-profiler prints unresolved PCs).  The count
  column is the sample weight; the optional column is ``alloc_bytes``.  One row is
  emitted per accepted line: a collapsed file carries one line per *distinct
  stack*, so two stacks that share a leaf are two observations, and aggregation is
  the importer's concern -- exactly as in P1 ("the duplicate address keeps both
  observations", ``tests/test_profile.py``).
* **JFR JSON** (``jfr print --json`` projection; no binary ``.jfr`` decoder) --
  ``jdk.ExecutionSample`` contributes ``+1`` sample on its top frame;
  ``jdk.ObjectAllocationInNewTLAB`` / ``jdk.ObjectAllocationOutsideTLAB``
  contribute ``+1`` sample and ``alloc_bytes`` from ``tlabSize`` (falling back to
  ``allocationSize``).  Frames are taken top-down: ``_TOP_FRAME_INDEX`` is frame
  ``0`` of ``stackTrace.frames``, which is the order ``jfr print`` and the JFR
  binary format use (innermost frame first).  Every other event type increments
  ``ignored`` (and is counted per type in ``ignored_types``) -- a recording holds
  dozens of event types and importing one must not fail on the other forty.
  Rows are aggregated by ``(symbol, func_addr)``, so N samples of one hot method
  arrive as one row of ``samples=N``.
* **JVM naming.**  A JFR frame's symbol is ``com.example.Hot.hot`` -- the dotted
  class name plus the method name, i.e. exactly the spelling
  :func:`toothpick.jvm_parse._display_name` stamps on a J1 ``funcs`` row, so
  attribution by exact name can match a method the class-file ingest observed.

Chunk 2 wiring note: this module imports ``ProfileSample`` from ``toothpick.profile``
at module scope, so the routing ``UNIT-J2-JFR-PLAN.md`` section 1.2 adds to
``profile.py`` must import these parsers *inside* the routing functions (the
deferred-import pattern ``cli.py`` already uses) or the two modules form an
import cycle.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from toothpick.profile import (
    ProfileError,
    ProfileFormat,
    ProfileSample,
    _parse_address_text,
    _parse_count,
    _read_text,
    _sort_key,
)

__all__ = [
    "ASYNC_FORMAT",
    "JFR_FORMAT",
    "ProfileFormat",
    "detect_format",
    "parse_async_collapsed",
    "parse_jfr_json",
]

#: ``detect_format`` value for async-profiler collapsed stacks.
ASYNC_FORMAT = "async"

#: ``detect_format`` value for a ``jfr print --json`` event projection.
JFR_FORMAT = "jfr"

# --------------------------------------------------------------------------- #
# async-profiler collapsed
# --------------------------------------------------------------------------- #

#: A collapsed frame name is a single whitespace-free token; a numeric spelling of
#: this shape is an address, and an address-only leaf contributes ``func_addr``
#: with the verbatim frame as its symbol (the frame text is the evidence).
_ADDRESS_FRAME_RE = re.compile(r"^\[?\s*0[xX][0-9a-fA-F]+\s*\]?$")

#: One number token, in the spellings :func:`toothpick.profile._parse_count`
#: accepts (``42``, ``1.2K``, ``1_024``).
_NUMBER = r"[0-9][0-9._]*[kKmMgGtT]?"

#: Attached allocation spelling of the count column: ``<count>_<bytes>B``.
_INLINE_ALLOC_RE = re.compile(rf"^(?P<count>{_NUMBER})_(?P<size>{_NUMBER})[bB]$")

#: Second-column allocation spelling with a byte suffix: ``<bytes>B``.
_BYTES_COLUMN_RE = re.compile(rf"^(?P<size>{_NUMBER})[bB]$")


def parse_async_collapsed(
    text: Union[str, Path],
) -> Tuple[List[ProfileSample], Dict[str, Any]]:
    """Parse async-profiler collapsed text -> ``(samples, diagnostics)``.

    Grammar per line: ``frame;frame;frame <count> [<alloc>]``.  Blank lines,
    ``#`` comments and the ``--- Execution profile ---`` banner lines every
    async-profiler export starts with are skipped; anything else that does not fit
    the grammar (no count column, empty frame, non-numeric weight) increments
    ``rejected`` instead of raising, so one bad line cannot lose a 100k-line
    profile.

    Diagnostics: ``format``, ``lines`` (candidate lines examined), ``rows``,
    ``rejected``, ``skipped``, ``samples`` and ``alloc_bytes`` totals.
    """
    body = _read_text(text)
    normalized = body.replace("\r\n", "\n").replace("\r", "\n")

    samples: List[ProfileSample] = []
    rejected = 0
    skipped = 0
    candidates = 0
    total_samples = 0
    total_alloc = 0

    for raw in normalized.split("\n"):
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            skipped += 1
            continue
        candidates += 1
        parsed = _parse_collapsed_line(line)
        if parsed is None:
            rejected += 1
            continue
        frames, weight, alloc = parsed
        symbol, addr = _leaf_symbol_and_addr(frames[-1])
        try:
            samples.append(
                ProfileSample(
                    symbol=symbol,
                    func_addr=addr,
                    samples=weight,
                    alloc_bytes=alloc,
                )
            )
        except ProfileError:  # pragma: no cover - guarded by the value parsers
            rejected += 1
            continue
        total_samples += weight
        total_alloc += alloc

    samples.sort(key=_sort_key)
    diagnostics: Dict[str, Any] = {
        "format": ASYNC_FORMAT,
        "lines": candidates,
        "rows": len(samples),
        "rejected": rejected,
        "skipped": skipped,
        "samples": total_samples,
        "alloc_bytes": total_alloc,
    }
    return samples, diagnostics


def _parse_collapsed_line(line: str) -> Optional[Tuple[List[str], int, int]]:
    """``"a;B 42 1024"`` -> ``(["a", "B"], 42, 1024)``; ``None`` when malformed.

    The weight columns are read from the right: an attached ``<count>_<bytes>B``
    token is one column pair, a ``<bytes>B`` token pairs with the count before it,
    two adjacent plain numbers are ``(samples, alloc_bytes)``, and a lone number is
    the sample count.  Everything left of the weight columns is the stack.
    """
    tokens = line.split()
    if len(tokens) < 2:
        return None

    samples_count = 0
    alloc_bytes = 0
    stack_tokens = tokens
    last = tokens[-1]

    inline = _INLINE_ALLOC_RE.match(last)
    suffixed = None if inline is not None else _BYTES_COLUMN_RE.match(last)

    if inline is not None:
        count = _parse_count(inline.group("count"))
        size = _parse_count(inline.group("size"))
        if count is None or size is None:
            return None
        samples_count, alloc_bytes = count, size
        stack_tokens = tokens[:-1]
    elif suffixed is not None and len(tokens) >= 3:
        size = _parse_count(suffixed.group("size"))
        count = _parse_count(tokens[-2])
        if size is None or count is None:
            return None
        samples_count, alloc_bytes = count, size
        stack_tokens = tokens[:-2]
    else:
        count = _parse_count(last)
        if count is None:
            return None
        samples_count = count
        stack_tokens = tokens[:-1]
        if len(stack_tokens) >= 2:
            previous = _parse_count(stack_tokens[-1])
            if previous is not None:
                samples_count, alloc_bytes = previous, count
                stack_tokens = stack_tokens[:-1]

    frames = [frame.strip() for frame in " ".join(stack_tokens).split(";")]
    while frames and not frames[-1]:  # tolerate the trailing "frame;" spelling
        frames.pop()
    if not frames or any(not frame for frame in frames):
        return None
    return frames, samples_count, alloc_bytes


def _leaf_symbol_and_addr(frame: str) -> Tuple[str, Optional[int]]:
    """Leaf frame -> ``(symbol, func_addr)``; address only for a ``0x...`` frame.

    The symbol is the frame text verbatim (so ``func+0x10``, ``[unknown]`` and
    ``[0x7f...]`` all survive as evidence); ``func_addr`` is filled only when the
    frame *is* an address.
    """
    symbol = frame.strip()
    if _ADDRESS_FRAME_RE.match(symbol):
        addr = _parse_address_text(symbol.strip("[]"))
        if addr is not None:
            return symbol, addr
    return symbol, None


# --------------------------------------------------------------------------- #
# JFR JSON
# --------------------------------------------------------------------------- #

#: The only JFR event types this module attributes (plan section 1).
_EXECUTION_SAMPLE = "jdk.ExecutionSample"
_ALLOC_IN_NEW_TLAB = "jdk.ObjectAllocationInNewTLAB"
_ALLOC_OUTSIDE_TLAB = "jdk.ObjectAllocationOutsideTLAB"
_ALLOC_TYPES = frozenset({_ALLOC_IN_NEW_TLAB, _ALLOC_OUTSIDE_TLAB})
_KNOWN_TYPES = frozenset({_EXECUTION_SAMPLE}) | _ALLOC_TYPES

#: ``jfr print`` lists stack-trace frames top-down (innermost first); frame 0 is
#: therefore the top method the plan asks for.  Named rather than inlined so the
#: assumption is greppable.
_TOP_FRAME_INDEX = 0

#: Field spellings of ``jfr print --json`` plus the converter variants seen in the
#: wild; a value that is present but unusable is skipped, never guessed around.
_TYPE_KEYS = ("type", "eventType", "event_type")
_STACK_KEYS = ("stackTrace", "stacktrace", "stack_trace", "stack", "frames", "trace")
_METHOD_NAME_KEYS = ("name", "methodName", "method_name")
_CLASS_NAME_KEYS = ("className", "class_name")
_CLASS_KEYS = ("class", "clazz")
_ADDRESS_KEYS = (
    "address",
    "addr",
    "func_addr",
    "funcAddr",
    "startAddress",
    "start_address",
    "methodAddress",
    "codeAddress",
    "pc",
)

#: Allocation-size fields, plan order: ``tlabSize`` then ``allocationSize``.
_ALLOC_SIZE_KEYS = (
    "tlabSize",
    "tlab_size",
    "allocationSize",
    "allocation_size",
    "allocSize",
    "alloc_size",
)

#: JFR structural tags: a mapping like ``{"type": "Class"}`` carries no name, and
#: the tag must never masquerade as a class name.
_STRUCTURAL_TAGS = frozenset(
    {"class", "method", "frame", "stack", "field", "setting", "event", "value"}
)


def parse_jfr_json(
    text: Union[str, Path],
) -> Tuple[List[ProfileSample], Dict[str, Any]]:
    """Parse a ``jfr print --json`` projection -> ``(samples, diagnostics)``.

    Accepts an object with ``events`` (optionally nested under ``recording``), a
    bare array of events, or ``events`` grouped by type name.  Each event's payload
    is read from ``values`` (mapping or list-of-mappings) or from the event object
    itself, so both the JDK writer's nesting and a flattened converter output work.

    Diagnostics: ``format``, ``events``, ``rows``, ``rejected``, ``ignored``,
    ``ignored_types`` (per-type counts), ``sample_events``, ``alloc_events``,
    ``samples`` and ``alloc_bytes`` totals.
    """
    payload = _load_json(_read_text(text))
    events = _events_of(payload)

    aggregated: Dict[Tuple[str, Optional[int]], List[int]] = {}
    rejected = 0
    ignored = 0
    ignored_types: Dict[str, int] = {}
    sample_events = 0
    alloc_events = 0

    for event in events:
        if not isinstance(event, Mapping):
            rejected += 1
            continue
        type_name = _event_type_name(event)
        if not type_name:
            rejected += 1
            continue
        if type_name not in _KNOWN_TYPES:
            ignored += 1
            ignored_types[type_name] = ignored_types.get(type_name, 0) + 1
            continue

        body = _event_payload(event)
        frame = _top_frame(body)
        if frame is None:
            rejected += 1
            continue
        identified = _frame_symbol_and_addr(frame)
        if identified is None:
            rejected += 1
            continue
        symbol, addr = identified
        if not symbol and addr is None:
            rejected += 1
            continue

        slot = aggregated.setdefault((symbol, addr), [0, 0])
        slot[0] += 1
        if type_name in _ALLOC_TYPES:
            alloc_events += 1
            slot[1] += _allocation_bytes(body)
        else:
            sample_events += 1

    samples: List[ProfileSample] = []
    for (symbol, addr), (count, alloc) in aggregated.items():
        try:
            samples.append(
                ProfileSample(
                    symbol=symbol,
                    func_addr=addr,
                    samples=count,
                    alloc_bytes=alloc,
                )
            )
        except ProfileError:  # pragma: no cover - guarded by the value parsers
            rejected += 1
    samples.sort(key=_sort_key)

    diagnostics: Dict[str, Any] = {
        "format": JFR_FORMAT,
        "events": len(events),
        "rows": len(samples),
        "rejected": rejected,
        "ignored": ignored,
        "ignored_types": {name: ignored_types[name] for name in sorted(ignored_types)},
        "sample_events": sample_events,
        "alloc_events": alloc_events,
        "samples": sum(s.samples for s in samples),
        "alloc_bytes": sum(s.alloc_bytes for s in samples),
    }
    return samples, diagnostics


def _load_json(body: str) -> Any:
    """Parse event JSON, refusing anything that is not valid JSON."""
    data = body.lstrip("\ufeff").strip()
    if not data:
        raise ProfileError("JFR JSON is empty; expected 'jfr print --json' output")
    try:
        return json.loads(data)
    except ValueError as exc:
        raise ProfileError(
            f"JFR JSON is not valid JSON ({exc}); expected 'jfr print --json' output"
        ) from exc


def _events_of(payload: Any, depth: int = 0) -> List[Any]:
    """Locate the event list in a JFR JSON document (or refuse)."""
    if depth > 4:
        raise ProfileError("JFR JSON nests too deeply to find an event list")
    if isinstance(payload, Mapping):
        events = payload.get("events")
        if isinstance(events, (list, tuple)):
            return list(events)
        if isinstance(events, Mapping):
            # Events grouped by type: {"events": {"jdk.ExecutionSample": [...]}}.
            # The group key *is* the type name, so it is carried onto each item
            # that does not spell its own type -- no invented attribution.
            collected: List[Any] = []
            for type_name, items in events.items():
                if isinstance(items, (list, tuple)):
                    for item in items:
                        if isinstance(item, Mapping) and not _event_type_name(item):
                            item = dict(item)
                            item["type"] = str(type_name)
                        collected.append(item)
                else:
                    collected.append(items)
            return collected
        for key in ("recording", "jfr", "data", "content"):
            node = payload.get(key)
            if isinstance(node, (Mapping, list, tuple)):
                try:
                    return _events_of(node, depth + 1)
                except ProfileError:
                    continue
        # A single event object is legitimate input for a one-event projection.
        if _event_type_name(payload) or _top_frame(_event_payload(payload)) is not None:
            return [payload]
        raise ProfileError(
            "JFR JSON has no 'events' list; expected 'jfr print --json' output"
        )
    if isinstance(payload, (list, tuple)):
        return list(payload)
    raise ProfileError(
        "JFR JSON must be an object with 'events' or an array of events; "
        f"got {type(payload).__name__}"
    )


def _type_text(value: Any, depth: int = 0) -> Optional[str]:
    """Event-type name out of a string or a nested ``{"name": ...}`` tag object."""
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, Mapping) and depth < 3:
        for key in ("name", "eventType", "event_type", "type"):
            if key in value:
                name = _type_text(value[key], depth + 1)
                if name:
                    return name
    return None


def _event_type_name(event: Mapping[str, Any]) -> Optional[str]:
    """``type``/``eventType`` of an event object, in that order."""
    for key in _TYPE_KEYS:
        if key in event:
            name = _type_text(event[key])
            if name:
                return name
    return None


def _event_payload(event: Mapping[str, Any]) -> Dict[str, Any]:
    """Flatten ``values`` (mapping or list of mappings) onto the event object."""
    payload: Dict[str, Any] = dict(event)
    for key in ("values", "value"):
        values = event.get(key)
        if isinstance(values, Mapping):
            payload.update(values)
        elif isinstance(values, (list, tuple)):
            for item in values:
                if isinstance(item, Mapping):
                    payload.update(item)
    return payload


def _frames_of(node: Any, depth: int = 0) -> Optional[List[Any]]:
    """Frame array from a stack-trace node (dict-wrapped array, or the array)."""
    if depth > 4:
        return None
    if isinstance(node, Mapping):
        for key in _STACK_KEYS:
            if key in node:
                frames = _frames_of(node[key], depth + 1)
                if frames:
                    return frames
        return None
    if isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
        return list(node)
    return None


def _top_frame(payload: Mapping[str, Any]) -> Any:
    """Top (innermost) frame of a payload's stack trace, or ``None``."""
    for key in _STACK_KEYS:
        if key not in payload:
            continue
        frames = _frames_of(payload[key])
        if frames and len(frames) > _TOP_FRAME_INDEX:
            return frames[_TOP_FRAME_INDEX]
        if frames:
            return None
    return None


def _first_text(node: Any, keys: Sequence[str]) -> Optional[str]:
    """First non-blank string value among ``keys`` (mappings only, never coerced)."""
    if not isinstance(node, Mapping):
        return None
    for key in keys:
        value = node.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _class_text(value: Any) -> Optional[str]:
    """Class name out of ``"a/B"``, ``{"name": "a/B"}`` or ``{"type": "a/B"}``."""
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, Mapping):
        for key in ("name", "className", "class_name", "type"):
            text = _first_text(value, (key,))
            if text and text.lower() not in _STRUCTURAL_TAGS:
                return text
    return None


def _as_non_negative_int(value: Any) -> Optional[int]:
    """Exact non-negative integer from int/float/decimal/hex text; else ``None``.

    Never rounds a float and never coerces a bool: a profile field that is not a
    size or an address yields ``None`` (the row keeps symbol attribution only).
    """
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float):
        if value != value or value < 0:  # NaN / negative
            return None
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        parsed = _parse_count(text)
        if parsed is None:
            parsed = _parse_address_text(text)
        return parsed
    return None


def _frame_symbol_and_addr(frame: Any) -> Optional[Tuple[str, Optional[int]]]:
    """JFR frame -> ``(symbol, func_addr)``; ``None`` when the frame carries neither.

    The symbol is ``<dotted class>.<method>`` when both are present (the J1
    ``funcs``-row spelling, so name attribution can match), the method name alone
    when the class is absent, and the frame text itself when a converter emitted a
    bare string frame.
    """
    if isinstance(frame, str):
        text = frame.strip()
        if not text:
            return None
        addr = _parse_address_text(text) if _ADDRESS_FRAME_RE.match(text) else None
        return text, addr
    if not isinstance(frame, Mapping):
        return None

    method = frame.get("method")
    method_map = method if isinstance(method, Mapping) else {}

    if isinstance(method, str) and method.strip():
        symbol = method.strip()
    else:
        name = _first_text(method_map, _METHOD_NAME_KEYS) or _first_text(
            frame, _METHOD_NAME_KEYS
        )
        class_name = (
            _class_text(method_map.get("class"))
            or _class_text(frame.get("class"))
            or _first_text(method_map, _CLASS_NAME_KEYS)
            or _first_text(frame, _CLASS_NAME_KEYS)
        )
        for key in _CLASS_KEYS:
            if class_name is None and key in frame:
                class_name = _class_text(frame[key])
        parts: List[str] = []
        if class_name:
            parts.append(class_name.replace("/", "."))
        if name:
            parts.append(name)
        symbol = ".".join(parts)

    addr: Optional[int] = None
    for node in (frame, method_map):
        for key in _ADDRESS_KEYS:
            if key in node:
                addr = _as_non_negative_int(node[key])
                if addr is not None:
                    break
        if addr is not None:
            break

    if not symbol and addr is None:
        return None
    return symbol, addr


def _allocation_bytes(payload: Mapping[str, Any]) -> int:
    """Allocation weight of an event: ``tlabSize``, then ``allocationSize``."""
    for key in _ALLOC_SIZE_KEYS:
        if key in payload:
            value = _as_non_negative_int(payload[key])
            if value is not None:
                return value
    return 0


# --------------------------------------------------------------------------- #
# format sniffing
# --------------------------------------------------------------------------- #


def detect_format(text: Union[str, Path]) -> Optional[str]:
    """Sniff ``"jfr"`` / ``"async"`` from profile text, else ``None``.

    Evidence, never a guess: a JSON document that carries an ``events`` collection
    (or a bare array of ``jdk.*`` events) is ``"jfr"``; otherwise a non-JSON line
    that is a ``;``-separated stack **with a count column** is ``"async"``.  Text
    with neither signature returns ``None`` so the caller keeps its own refusal --
    single-frame collapsed files (no ``;``) are deliberately not claimed.

    This is the CLI ``--format auto`` assistance of plan section 1.3; it accepts a
    path for symmetry with the two parsers.
    """
    body = _read_text(text).lstrip("\ufeff")
    stripped = body.strip()
    if not stripped:
        return None
    if stripped[0] in "{[":
        try:
            payload = json.loads(stripped)
        except ValueError:
            return None
        return JFR_FORMAT if _looks_like_jfr(payload) else None
    return ASYNC_FORMAT if _looks_like_async(body) else None


def _looks_like_jfr(payload: Any, depth: int = 0) -> bool:
    """True when a JSON document carries an ``events`` collection or ``jdk.*`` events."""
    if depth > 4:
        return False
    if isinstance(payload, (list, tuple)):
        return any(
            isinstance(item, Mapping) and _is_jfr_type(_event_type_name(item))
            for item in payload
        )
    if isinstance(payload, Mapping):
        if isinstance(payload.get("events"), (list, tuple, Mapping)):
            return True
        for key in ("recording", "jfr", "data", "content"):
            node = payload.get(key)
            if isinstance(node, (Mapping, list, tuple)) and _looks_like_jfr(
                node, depth + 1
            ):
                return True
        return _is_jfr_type(_event_type_name(payload))
    return False


def _is_jfr_type(name: Optional[str]) -> bool:
    return bool(name) and str(name).startswith("jdk.")


def _looks_like_async(body: str) -> bool:
    """True when some non-comment line is a ``;``-stack with a count column."""
    normalized = body.replace("\r\n", "\n").replace("\r", "\n")
    for raw in normalized.split("\n"):
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        if ";" not in line:
            continue
        if _parse_collapsed_line(line) is not None:
            return True
    return False
