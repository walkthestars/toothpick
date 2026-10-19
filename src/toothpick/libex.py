"""
Toothpick library/vendor exclusion labelling -- namespace-prefix rules over
supplied ``(addr, name)`` pairs (UNIT-SEM5 Chunk 4).

Binding contract: ``docs/units/UNIT-SEM5-PLAN.md`` chunk 4 ("Library exclusion":
namespace-prefix rules over RTTI class names (``std::``, ``tbb::``,
``Concurrency::``, MFC ``CFile*`` ...) plus vendor string markers yield a
``{excluded, kept}`` partition of the supplied addrs/names **with reasons**;
nothing is ever deleted, only labelled).  Design anchors:
``docs/design-document.md`` section 1 (every row traces to a supplied fact --
here, to the caller's own name string, never to a guess) and section 19 (a label
is only ever what was proven), plus Risk R1 (bounded, counted emission).

One entry point does the work::

    partition_names(names, rules=None, *, max_names=DEFAULT_MAX_NAMES) -> Partition

``names`` is the caller's own evidence -- an iterable of ``(addr, name)`` pairs,
a mapping of ``addr -> name``, or a sequence of pair-like objects (objects with
``addr``/``name`` attributes, or mappings with ``addr``/``name`` keys).  Every
supplied pair is labelled exactly once:

* **excluded** -- a rule matched: the row carries the rule's ``reason`` and the
  rule itself (``Label.rule``, whose ``source`` says whether it came from the
  defaults or from the caller).
* **kept** -- no rule matched, or an explicit ``kept`` rule did.  A kept row that
  no rule matched carries :data:`KEEP_REASON` (``"no-rule-matched"``) and
  ``rule is None``; that is the default verdict for "everything else".

The rows are never mutated, never dropped and never reordered: ``Partition.rows``
holds them in the order supplied, every input either appears in ``rows`` or is
counted in ``Partition.skipped``, and the caller's containers are left untouched.
Nothing in this module deletes anything -- it prints labels over the caller's
evidence.

Default rules (:data:`DEFAULT_RULES`, first match wins) are the vendor/RTTI floor:

===============================  ==========================================
name shape                       reason
===============================  ==========================================
``std::…`` / ``std__…``          ``vendor-namespace:std``
``tbb::…`` / ``tbb_…`` / ``__TBB…``  ``vendor-namespace:tbb``
``Concurrency::…``               ``vendor-namespace:Concurrency``
``spdlog`` / ``spdlog::…`` / ``spdlog_…``   ``vendor-library:spdlog``
MFC ``CFile``/``CStdioFile``/``CMemFile``/…   ``mfc-class:CFile-family``
MFC ``CString…``                 ``mfc-class:CString-family``
MFC ``CException…``              ``mfc-class:CException-family``
MFC ``CArchive…``                ``mfc-class:CArchive-family``
``__imp_…`` / ``_imp__…`` / ``ILT+…`` / import-thunk wording
                                 ``import-marker``
``[thunk]`` / ``_thunk…`` / ``??_E…`` / ``??_G…``   ``thunk-marker``
``lua_…`` / ``luaL_…`` / ``luaopen_…`` / ``LUA_…``  ``stdlib-marker:lua``
``Py_…`` / ``PyObject…`` / ``_Py…`` / ``__pyx_…`` / ``pymalloc…`` / ``Python``
                                 ``stdlib-marker:python``
===============================  ==========================================

The last two default rules are *kind* rules: they match a **structural tag**
derived from the name (:func:`classify_kind`: one of :data:`KIND_TAGS` --
``import`` / ``thunk`` / ``vtable`` / ``rtti`` / ``plain``) rather than a literal
text prefix, which is how the ``__imp``/thunk markers of the plan are expressed.

Custom rules: ``rules`` is appended to the module's rule set and is consulted
*before* the defaults, so a caller can add exclusions **and** carve a name back
out of a default verdict (``Rule("prefix", "std::vector", ..., label=KEPT)``).
Within the supplied list the first match wins, exactly as in the defaults, and the
list may be spelled as :class:`Rule` objects, mappings
(``{"kind": …, "pattern": …, "reason": …}``) or ``(kind, pattern, reason[,
label])`` tuples.  Rule kinds (:data:`RULE_KINDS`): ``prefix`` (normalised name
starts with the pattern), ``contains`` (pattern appears in the normalised name),
``regex`` (``re.search`` over the normalised name) and ``kind`` (the derived
structural tag equals the pattern).  Matching is case-sensitive unless a rule
asks for ``ignore_case=True``.

Normalisation before matching is deliberately tiny and lossless: leading ``::``
scope qualifiers and a leading ``class``/``struct``/``enum``/``union``/
``interface`` keyword are stripped (``"class CFile"`` and ``"?AVCFile@@"``-style
decorated descriptors are still matched by the MFC rules, which accept an
optional ``.?AV``/``.?AU``/``.?AW4`` descriptor prefix).  The **reported** name is
always the caller's string, byte for byte.

Bounds (Risk R1, plan chunk 4 "R1 caps on input size"): ``max_names`` defaults to
:data:`DEFAULT_MAX_NAMES` (100 000) and is clamped to :data:`MAX_NAMES`
(1 000 000) by :func:`resolve_max_names`.  The cap bounds *work as well as
output*: entries past the cap are **counted** in ``Partition.skipped`` and are
not inspected at all (so they are not matched, not labelled, and a malformed
entry past the cap is not a caller error).  A shortfall is therefore visible in
the return value, never silently swallowed.

Honesty rules enforced here:

* A name is classified only by the caller's own string.  No symbol table is read,
  no mangled name is decoded, no vendor is inferred from an address, and no
  "probably from a library" guess is ever made -- an unlabelled-looking name that
  no rule names stays ``kept`` with :data:`KEEP_REASON`.
* The partition is exhaustive: ``len(rows) + skipped`` equals the number of
  supplied pairs, and each labelled row appears in ``excluded`` or ``kept``
  exactly once.  Nothing is deleted -- the caller keeps its own data.
* Malformed *arguments* are caller errors and raise :class:`LibexError` (a
  :class:`ValueError`): a non-iterable ``names``, a pair that is not an
  ``(addr, name)`` shape, a negative or non-integer address, a non-``str`` name, a
  bad rule (unknown kind, uncompilable regex, empty reason, unknown label), a bad
  ``max_names``.

Pure, offline and deterministic: strings in, labels out.  No file, no network, no
subprocess, no SQLite handle, no clock, no randomness; equal inputs return equal
results regardless of container type, and rule order -- not dict ordering -- is
what decides a verdict.

This module defines its own :class:`LibexError` and imports nothing from any RTTI
reader (`rtti_parse`/`rtti_store`/`rtti`); it never imports ``faulthandler``
(Trap 11 -- suites run under ``-p no:faulthandler``) and opens no SQLite handle
(so Trap 7's ``[WinError 32]`` leak class cannot originate here).
"""

from __future__ import annotations

import operator
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "DEFAULT_MAX_NAMES",
    "DEFAULT_RULES",
    "EXCLUDED",
    "KEEP_REASON",
    "KEPT",
    "KIND_TAGS",
    "Label",
    "LibexError",
    "MAX_NAMES",
    "PROVENANCE",
    "Partition",
    "RULE_KINDS",
    "Rule",
    "classify_kind",
    "default_rules",
    "label_name",
    "partition_names",
    "resolve_max_names",
]

#: Default input cap (Risk R1; plan chunk 4).
DEFAULT_MAX_NAMES = 100_000

#: Hard ceiling for that cap; a larger budget is clamped, never honoured.
MAX_NAMES = 1_000_000

#: The two verdicts this module can print, and nothing else.
EXCLUDED = "excluded"
KEPT = "kept"

#: Reason carried by a row that no rule matched (the "everything else" verdict).
KEEP_REASON = "no-rule-matched"

#: Provenance token for the rule family this module implements (plan chunk 4:
#: "namespace-prefix rules over RTTI class names + vendor string markers").
PROVENANCE = "namespace-prefix-rules"

#: Rule kinds accepted by :class:`Rule`.
RULE_KINDS = ("prefix", "contains", "regex", "kind")

#: Labels a rule may assign.
LABELS = (EXCLUDED, KEPT)

#: Rule provenances.
SOURCE_DEFAULT = "default"
SOURCE_CUSTOM = "custom"
SOURCES = (SOURCE_DEFAULT, SOURCE_CUSTOM)

#: Structural tags :func:`classify_kind` can return, in precedence order.
KIND_TAGS = ("import", "thunk", "vtable", "rtti", "plain")

#: ``undname`` may print the type kind in front of a class name (``class CFile``).
_KIND_KEYWORD = re.compile(r"^(?:class|struct|enum|union|interface)\s+")

#: Optional decorated-RTTI descriptor prefix (``.?AVCFile@@``), so the MFC rules
#: match both a demangled ``class CFile`` and a raw ``.?AVCFile@@`` descriptor.
_DECORATED = r"(?:\.\?AV|\.\?AU|\.\?AW4)?"

#: End-of-class-name boundary: another capital (``CFileException``), the
#: decorated-name ``@`` (``.?AVCFile@@``) or end of string.
_CLASS_END = r"(?:[A-Z@]|$)"

#: Import-thunk markers (``__imp`` family, MSVC ILT, IDA's ``j_`` trampolines are
#: thunks, not imports, and live with the thunk markers below).
_IMPORT_PREFIXES = ("__imp_", "__imp", "_imp__", "_imp_", "ILT+")
_IMPORT_WORDING = ("(import thunk)",)

#: Thunk / adjustor-thunk markers.
_THUNK_PREFIXES = ("??_E", "??_G", "_thunk", "__thunk")
_THUNK_WORDING = ("[thunk]",)

#: Vtable and type-descriptor markers (raw decorated RTTI spellings).
_VTABLE_PREFIXES = ("??_7", "_ZTV")
_RTTI_PREFIXES = (
    ".?AV",
    ".?AU",
    ".?AW",
    "??_R0",
    "??_R1",
    "??_R2",
    "??_R3",
    "??_R4",
    "._R",
    "_ZTI",
    "_ZTS",
)

#: Sentinel for "the caller did not spell this field".
_MISSING = object()


class LibexError(ValueError):
    """Raised for a caller argument that cannot describe a labellable pair.

    A *name* is never a caller error (any string can be labelled); only a
    malformed argument -- a non-iterable ``names``, a pair that is not an
    ``(addr, name)`` shape, a negative/non-integer address, a non-``str`` name, a
    bad rule or a bad ``max_names`` -- lands here.
    """


# --- small strict coercions ------------------------------------------------- #


def _coerce_int(value: Any, what: str) -> int:
    """An integer, or a refusal (booleans are not integers here)."""
    if isinstance(value, bool):
        raise LibexError(f"{what} must be an integer, got {value!r}")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise LibexError(f"{what} must be an integer, got {value!r}") from exc


def _coerce_addr(value: Any, what: str) -> int:
    """Validate one address: a non-negative integer."""
    addr = _coerce_int(value, what)
    if addr < 0:
        raise LibexError(f"{what} must be non-negative, got {addr}")
    return addr


@lru_cache(maxsize=256)
def _compile(pattern: str, ignore_case: bool) -> "re.Pattern[str]":
    """Compile a regex rule pattern, refusing a pattern that does not compile."""
    flags = re.IGNORECASE if ignore_case else 0
    try:
        return re.compile(pattern, flags)
    except re.error as exc:
        raise LibexError(f"regex rule pattern {pattern!r} does not compile: {exc}") from exc


def _match_candidate(name: str) -> str:
    """The normalised form a rule is matched against (the name itself is kept).

    Leading ``::`` scope qualifiers, surrounding whitespace and one leading
    ``class``/``struct``/``enum``/``union``/``interface`` keyword are dropped;
    nothing else is touched, so no evidence is invented.
    """
    candidate = name.strip()
    while candidate.startswith("::"):
        candidate = candidate[2:].lstrip()
    return _KIND_KEYWORD.sub("", candidate).strip()


def classify_kind(name: str) -> str:
    """The structural tag of *name*: one of :data:`KIND_TAGS`.

    Precedence is fixed and documented: ``import`` (``__imp`` family, MSVC
    ``ILT+``, "Import Thunk" wording) beats ``thunk`` (``[thunk]``, ``_thunk``,
    ``??_E``/``??_G`` adjustor thunks, IDA ``j_`` trampolines) beats ``vtable``
    (``??_7``, ``_ZTV``) beats ``rtti`` (``.?AV``/``.?AU``/``.?AW`` descriptors,
    ``??_R0``..``??_R4``, ``_ZTI``/``_ZTS``, ``type_info``); everything else is
    ``plain``.  The tag is derived from the supplied string alone.
    """
    if not isinstance(name, str):
        raise LibexError(f"name must be a str, got {type(name).__name__}")
    candidate = _match_candidate(name)
    folded = candidate.casefold()
    if candidate.startswith(_IMPORT_PREFIXES) or any(word in folded for word in _IMPORT_WORDING):
        return "import"
    if (
        candidate.startswith(_THUNK_PREFIXES)
        or candidate.startswith("j_")
        or any(word in folded for word in _THUNK_WORDING)
    ):
        return "thunk"
    if candidate.startswith(_VTABLE_PREFIXES):
        return "vtable"
    if candidate.startswith(_RTTI_PREFIXES) or "type_info" in folded:
        return "rtti"
    return "plain"


# --- rules ------------------------------------------------------------------ #


def _validate_rule_fields(
    kind: Any, pattern: Any, reason: Any, label: Any, ignore_case: Any, source: Any
) -> None:
    """Refuse a rule that could not label a name honestly."""
    if kind not in RULE_KINDS:
        raise LibexError(f"rule kind must be one of {RULE_KINDS}, got {kind!r}")
    if not isinstance(pattern, str):
        raise LibexError(f"rule pattern must be a str, got {type(pattern).__name__}")
    if kind == "kind":
        if pattern.casefold() not in KIND_TAGS:
            raise LibexError(
                f"kind rule pattern must be one of {KIND_TAGS}, got {pattern!r}"
            )
    elif not pattern:
        raise LibexError(f"{kind} rule pattern must not be empty")
    if not isinstance(reason, str) or not reason:
        raise LibexError(f"rule reason must be a non-empty str, got {reason!r}")
    if label not in LABELS:
        raise LibexError(f"rule label must be one of {LABELS}, got {label!r}")
    if not isinstance(ignore_case, bool):
        raise LibexError(f"rule ignore_case must be a bool, got {ignore_case!r}")
    if source not in SOURCES:
        raise LibexError(f"rule source must be one of {SOURCES}, got {source!r}")
    if kind == "regex":
        _compile(pattern, ignore_case)


@dataclass(frozen=True)
class Rule:
    """One exclusion/keep rule: ``kind`` + ``pattern`` -> ``label`` + ``reason``.

    ``kind`` is one of :data:`RULE_KINDS`, ``pattern`` its literal prefix /
    substring / regex / structural tag, ``reason`` the short stable token printed
    on every row the rule decides (e.g. ``"vendor-namespace:std"``), ``label`` the
    verdict it assigns (:data:`EXCLUDED` by default) and ``source`` where it came
    from (:data:`SOURCE_CUSTOM` for caller-supplied rules,
    :data:`SOURCE_DEFAULT` for :data:`DEFAULT_RULES`).  Constructing a rule that
    could not label a name (unknown kind, uncompilable regex, empty reason,
    unknown label) raises :class:`LibexError` immediately, never at match time.
    """

    kind: str
    pattern: str
    reason: str
    label: str = EXCLUDED
    ignore_case: bool = False
    source: str = SOURCE_CUSTOM

    def __post_init__(self) -> None:
        _validate_rule_fields(
            self.kind, self.pattern, self.reason, self.label, self.ignore_case, self.source
        )

    def matches(self, name: str) -> bool:
        """Whether this rule decides *name* (normalised, per the module rules)."""
        if not isinstance(name, str):
            raise LibexError(f"name must be a str, got {type(name).__name__}")
        if self.kind == "kind":
            return classify_kind(name) == self.pattern.casefold()
        candidate = _match_candidate(name)
        pattern = self.pattern
        if self.ignore_case:
            candidate = candidate.casefold()
            pattern = pattern.casefold()
        if self.kind == "prefix":
            return candidate.startswith(pattern)
        if self.kind == "contains":
            return pattern in candidate
        return _compile(self.pattern, self.ignore_case).search(candidate) is not None

    def describe(self) -> str:
        """``kind:pattern`` -- how this rule is spelled in a report."""
        return f"{self.kind}:{self.pattern}"

    def as_dict(self) -> Dict[str, Any]:
        """A JSON-friendly view of the rule (no compiled regex, no objects)."""
        return {
            "kind": self.kind,
            "pattern": self.pattern,
            "reason": self.reason,
            "label": self.label,
            "ignore_case": self.ignore_case,
            "source": self.source,
        }


def _build_default_rules() -> Tuple[Rule, ...]:
    """The built-in vendor/RTTI exclusion floor (first match wins)."""
    return (
        Rule("regex", r"^std(?:__|::|$)", "vendor-namespace:std", source=SOURCE_DEFAULT),
        Rule(
            "regex",
            r"^(?:tbb(?:::|_|$)|__TBB)",
            "vendor-namespace:tbb",
            source=SOURCE_DEFAULT,
        ),
        Rule(
            "regex",
            r"^Concurrency(?:::|$)",
            "vendor-namespace:Concurrency",
            source=SOURCE_DEFAULT,
        ),
        Rule(
            "regex",
            r"^spdlog(?:::|_|$|\.)",
            "vendor-library:spdlog",
            source=SOURCE_DEFAULT,
        ),
        Rule(
            "regex",
            r"^" + _DECORATED + r"CFile" + _CLASS_END,
            "mfc-class:CFile-family",
            source=SOURCE_DEFAULT,
        ),
        Rule(
            "regex",
            r"^"
            + _DECORATED
            + r"C(?:StdioFile|MemFile|SharedFile|SocketFile|OleStreamFile"
            + r"|InternetFile|HttpFile|GopherFile)"
            + _CLASS_END,
            "mfc-class:CFile-family",
            source=SOURCE_DEFAULT,
        ),
        Rule(
            "regex",
            r"^" + _DECORATED + r"CString" + _CLASS_END,
            "mfc-class:CString-family",
            source=SOURCE_DEFAULT,
        ),
        Rule(
            "regex",
            r"^" + _DECORATED + r"CException" + _CLASS_END,
            "mfc-class:CException-family",
            source=SOURCE_DEFAULT,
        ),
        Rule(
            "regex",
            r"^" + _DECORATED + r"CArchive" + _CLASS_END,
            "mfc-class:CArchive-family",
            source=SOURCE_DEFAULT,
        ),
        Rule("kind", "import", "import-marker", source=SOURCE_DEFAULT),
        Rule("kind", "thunk", "thunk-marker", source=SOURCE_DEFAULT),
        Rule(
            "regex",
            r"^(?:lua_|luaL_|luaopen_|LUA_)",
            "stdlib-marker:lua",
            source=SOURCE_DEFAULT,
        ),
        Rule(
            "regex",
            r"^(?:_?Py[A-Z_]|__pyx_|pymalloc|Python$)",
            "stdlib-marker:python",
            source=SOURCE_DEFAULT,
        ),
    )


#: The default (vendor/RTTI) rule floor; use :func:`default_rules` for the API.
DEFAULT_RULES: Tuple[Rule, ...] = _build_default_rules()


def default_rules() -> Tuple[Rule, ...]:
    """The built-in rule floor, as an immutable tuple (safe to share)."""
    return DEFAULT_RULES


def resolve_max_names(value: Any = None) -> int:
    """Coerce the R1 input cap: default, else clamp to :data:`MAX_NAMES`.

    ``None`` means "use the default" (:data:`DEFAULT_MAX_NAMES`).  A bool, a
    float, a string or any other non-integer spelling is a caller error and raises
    (``"50"`` and ``3.5`` are refused, never guessed); a non-positive integer
    raises; an over-large integer is clamped to :data:`MAX_NAMES`, so the answer
    is bounded either way.
    """
    if value is None:
        return DEFAULT_MAX_NAMES
    budget = _coerce_int(value, "max_names")
    if budget <= 0:
        raise LibexError(f"max_names must be positive, got {budget}")
    return min(budget, MAX_NAMES)


# --- labelling -------------------------------------------------------------- #


@dataclass(frozen=True)
class Label:
    """One labelled input row: the caller's pair, the verdict, and why.

    ``addr`` is the supplied address (``None`` only for
    :func:`label_name`, which labels a bare name), ``name`` the caller's string
    byte for byte, ``label`` either :data:`EXCLUDED` or :data:`KEPT`, ``reason``
    the deciding rule's stable reason token, and ``rule`` the deciding
    :class:`Rule` (``None`` for the default keep -- nothing was matched).
    """

    addr: Optional[int]
    name: str
    label: str
    reason: str
    rule: Optional[Rule] = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise LibexError(f"name must be a str, got {type(self.name).__name__}")
        if self.label not in LABELS:
            raise LibexError(f"label must be one of {LABELS}, got {self.label!r}")
        if not isinstance(self.reason, str) or not self.reason:
            raise LibexError(f"reason must be a non-empty str, got {self.reason!r}")
        if self.addr is not None:
            _coerce_addr(self.addr, "addr")
        if self.rule is not None and not isinstance(self.rule, Rule):
            raise LibexError(f"rule must be a Rule or None, got {type(self.rule).__name__}")

    @property
    def excluded(self) -> bool:
        """True when this row was excluded."""
        return self.label == EXCLUDED

    @property
    def kept(self) -> bool:
        """True when this row was kept."""
        return self.label == KEPT

    def as_dict(self) -> Dict[str, Any]:
        """A JSON-friendly view of the row (rule expanded, or ``None``)."""
        return {
            "addr": self.addr,
            "name": self.name,
            "label": self.label,
            "reason": self.reason,
            "rule": None if self.rule is None else self.rule.as_dict(),
        }

    def __getitem__(self, key: str) -> Any:
        if not isinstance(key, str):
            raise KeyError(key)
        return self.as_dict()[key]


@dataclass(frozen=True)
class Partition:
    """The labelled partition of the caller's pairs, plus the counted skips.

    ``rows`` holds one :class:`Label` per inspected pair, in the order supplied;
    ``excluded`` / ``kept`` are its two disjoint halves (same rows, filtered);
    ``skipped`` counts the pairs beyond the R1 cap that were never inspected, and
    ``max_names`` records the cap that was in force.  The partition is exhaustive:
    ``len(rows) + skipped`` is the number of supplied pairs, and nothing is ever
    removed from the caller's own data.  Built by :func:`partition_names` (or
    directly, from labels); mapping-style access (``part["excluded"]``) and
    ``as_dict()`` are provided for reporting.
    """

    rows: Tuple[Label, ...] = ()
    skipped: int = 0
    max_names: int = DEFAULT_MAX_NAMES

    def __post_init__(self) -> None:
        if not isinstance(self.skipped, int) or isinstance(self.skipped, bool) or self.skipped < 0:
            raise LibexError(f"skipped must be a non-negative int, got {self.skipped!r}")
        if (
            not isinstance(self.max_names, int)
            or isinstance(self.max_names, bool)
            or self.max_names < 1
        ):
            raise LibexError(f"max_names must be a positive int, got {self.max_names!r}")
        object.__setattr__(
            self, "_excluded", tuple(row for row in self.rows if row.label == EXCLUDED)
        )
        object.__setattr__(self, "_kept", tuple(row for row in self.rows if row.label == KEPT))

    @property
    def excluded(self) -> Tuple[Label, ...]:
        """The excluded rows, in the order supplied."""
        return self._excluded

    @property
    def kept(self) -> Tuple[Label, ...]:
        """The kept rows, in the order supplied."""
        return self._kept

    @property
    def total(self) -> int:
        """Labelled rows plus counted skips (every supplied pair)."""
        return len(self.rows) + self.skipped

    def excluded_names(self) -> Tuple[str, ...]:
        """The excluded names, in the order supplied."""
        return tuple(row.name for row in self._excluded)

    def kept_names(self) -> Tuple[str, ...]:
        """The kept names, in the order supplied."""
        return tuple(row.name for row in self._kept)

    def row_for(self, addr: int) -> Optional[Label]:
        """The first labelled row at *addr*, or ``None`` (duplicates: first wins)."""
        wanted = _coerce_addr(addr, "addr")
        for row in self.rows:
            if row.addr == wanted:
                return row
        return None

    def label_for(self, addr: int) -> Optional[str]:
        """The verdict at *addr* (:data:`EXCLUDED`/:data:`KEPT`), or ``None``."""
        row = self.row_for(addr)
        return None if row is None else row.label

    def as_dict(self) -> Dict[str, Any]:
        """A JSON-friendly report: both halves, the cap, and the skip count."""
        return {
            "excluded": [row.as_dict() for row in self._excluded],
            "kept": [row.as_dict() for row in self._kept],
            "skipped": self.skipped,
            "max_names": self.max_names,
            "total": self.total,
        }

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            if key == "rows":
                return self.rows
            if key == "excluded":
                return self._excluded
            if key == "kept":
                return self._kept
            if key == "skipped":
                return self.skipped
            if key == "max_names":
                return self.max_names
            if key == "total":
                return self.total
            raise KeyError(key)
        if isinstance(key, int) and not isinstance(key, bool):
            return self.rows[key]
        raise KeyError(key)

    def __iter__(self) -> Iterator[Label]:
        return iter(self.rows)

    def __len__(self) -> int:
        return len(self.rows)


def _coerce_rule(item: Any, index: int) -> Rule:
    """One supplied rule, however the caller spelled it."""
    what = f"rules[{index}]"
    if isinstance(item, Rule):
        return item
    if isinstance(item, Mapping):
        kind = item.get("kind", _MISSING)
        pattern = item.get("pattern", _MISSING)
        reason = item.get("reason", _MISSING)
        if kind is _MISSING or pattern is _MISSING or reason is _MISSING:
            raise LibexError(f"{what} must carry kind/pattern/reason, got {item!r}")
        return Rule(
            kind=kind,
            pattern=pattern,
            reason=reason,
            label=item.get("label", EXCLUDED),
            ignore_case=item.get("ignore_case", False),
            source=item.get("source", SOURCE_CUSTOM),
        )
    if isinstance(item, bool) or isinstance(item, (str, bytes, bytearray)):
        raise LibexError(
            f"{what} must be a Rule or a (kind, pattern, reason) tuple, got {item!r}"
        )
    try:
        seq = list(item)
    except TypeError as exc:
        raise LibexError(
            f"{what} must be a Rule or a (kind, pattern, reason) tuple, got {item!r}"
        ) from exc
    if len(seq) == 3:
        return Rule(kind=seq[0], pattern=seq[1], reason=seq[2])
    if len(seq) == 4:
        return Rule(kind=seq[0], pattern=seq[1], reason=seq[2], label=seq[3])
    raise LibexError(
        f"{what} must be a Rule or a (kind, pattern, reason[, label]) tuple, got {item!r}"
    )


def _coerce_rules(value: Any) -> Tuple[Rule, ...]:
    """Coerce ``None``/one rule/a sequence of rules into a tuple of rules."""
    if value is None:
        return ()
    if isinstance(value, Rule):
        return (value,)
    if isinstance(value, Mapping):
        return (_coerce_rule(value, 0),)
    if isinstance(value, bool) or isinstance(value, (str, bytes, bytearray)):
        raise LibexError(f"rules must be a sequence of Rule objects, got {value!r}")
    try:
        items = list(value)
    except TypeError as exc:
        raise LibexError(f"rules must be a sequence of Rule objects, got {value!r}") from exc
    return tuple(_coerce_rule(item, index) for index, item in enumerate(items))


def _rule_table(rules: Any) -> Tuple[Rule, ...]:
    """The effective rule order: caller rules first, then the default floor."""
    return _coerce_rules(rules) + DEFAULT_RULES


def _label_row(addr: Optional[int], name: str, table: Tuple[Rule, ...]) -> Label:
    """Apply the rule table to one pair (first match wins; else default keep)."""
    for rule in table:
        if rule.matches(name):
            return Label(addr=addr, name=name, label=rule.label, reason=rule.reason, rule=rule)
    return Label(addr=addr, name=name, label=KEPT, reason=KEEP_REASON, rule=None)


def label_name(name: str, rules: Any = None, addr: Optional[int] = None) -> Label:
    """Label a bare *name* (no address) by the same rules :func:`partition_names` uses.

    The caller-supplied ``rules`` are consulted before the defaults, exactly as in
    :func:`partition_names`, so this is the single-name form of the same verdict
    and is what makes per-name reasoning easy to report.  A non-``str`` ``name``,
    a negative ``addr`` or a malformed rule raises :class:`LibexError`.
    """
    if not isinstance(name, str):
        raise LibexError(f"name must be a str, got {type(name).__name__}")
    checked = None if addr is None else _coerce_addr(addr, "addr")
    return _label_row(checked, name, _rule_table(rules))


def _coerce_pair(item: Any, index: int) -> Tuple[int, str]:
    """One supplied ``(addr, name)`` pair, however the caller spelled it."""
    what = f"names[{index}]"
    if isinstance(item, Mapping):
        addr = item.get("addr", item.get("address", _MISSING))
        name = item.get("name", _MISSING)
        if addr is _MISSING or name is _MISSING:
            raise LibexError(
                f"{what} must be an (addr, name) pair or a mapping with addr/name, got {item!r}"
            )
    elif isinstance(item, bool) or isinstance(item, (str, bytes, bytearray)):
        raise LibexError(
            f"{what} must be an (addr, name) pair, got {type(item).__name__}"
        )
    else:
        addr = getattr(item, "addr", getattr(item, "address", _MISSING))
        name = getattr(item, "name", _MISSING)
        if addr is _MISSING or name is _MISSING:
            try:
                seq = list(item)
            except TypeError as exc:
                raise LibexError(
                    f"{what} must be an (addr, name) pair, got {item!r}"
                ) from exc
            if len(seq) != 2:
                raise LibexError(
                    f"{what} must be an (addr, name) pair, got {item!r}"
                )
            addr, name = seq[0], seq[1]
    if not isinstance(name, str):
        raise LibexError(f"{what} name must be a str, got {type(name).__name__}")
    return _coerce_addr(addr, f"{what} addr"), name


def _coerce_pairs(value: Any, budget: int) -> Tuple[List[Tuple[int, str]], int]:
    """The pairs worth inspecting, plus the count of entries past the cap.

    The cap bounds work as well as output: once *budget* pairs have been taken,
    every remaining entry is counted and left uninspected (so it is neither
    matched nor validated).  A non-iterable ``names`` is a caller error.
    """
    if isinstance(value, Mapping):
        items: List[Any] = list(value.items())
    elif isinstance(value, (str, bytes, bytearray)):
        raise LibexError(
            f"names must be an iterable of (addr, name) pairs, got {type(value).__name__}"
        )
    else:
        try:
            items = list(value)
        except TypeError as exc:
            raise LibexError(
                f"names must be an iterable of (addr, name) pairs, "
                f"got {type(value).__name__}"
            ) from exc
    pairs: List[Tuple[int, str]] = []
    skipped = 0
    for index, item in enumerate(items):
        if len(pairs) >= budget:
            skipped += 1
            continue
        pairs.append(_coerce_pair(item, index))
    return pairs, skipped


def partition_names(
    names: Any,
    rules: Any = None,
    *,
    max_names: Any = DEFAULT_MAX_NAMES,
) -> Partition:
    """Label every supplied ``(addr, name)`` pair as excluded or kept, with reasons.

    ``names`` is the caller's evidence: an iterable of ``(addr, name)`` pairs (two
    element sequences, mappings with ``addr``/``address`` + ``name`` keys, or
    objects carrying ``addr``/``address`` + ``name`` attributes) or a mapping of
    ``addr -> name``.  Each inspected pair gets exactly one :class:`Label`, in the
    order supplied; ``rules`` are caller rules consulted before
    :data:`DEFAULT_RULES` (first match wins -- see the module docstring), and
    ``max_names`` is the R1 input cap (:func:`resolve_max_names`: default
    :data:`DEFAULT_MAX_NAMES` = 100 000, clamped to :data:`MAX_NAMES` =
    1 000 000).

    Returns a :class:`Partition`: ``rows`` (every inspected pair labelled),
    ``excluded`` / ``kept`` (the two halves, filtered from ``rows``), ``skipped``
    (pairs beyond the cap, counted -- never inspected and never silently dropped)
    and ``max_names``.  Nothing is deleted or mutated: the caller's own data is
    left exactly as supplied.

    Raises :class:`LibexError` for a malformed argument (a non-iterable ``names``,
    a pair that is not an ``(addr, name)`` shape, a negative or non-integer
    address, a non-``str`` name, a bad rule, a bad ``max_names``).  Two runs over
    equal inputs return equal partitions.
    """
    budget = resolve_max_names(max_names)
    table = _rule_table(rules)
    pairs, skipped = _coerce_pairs(names, budget)
    rows = tuple(_label_row(addr, name, table) for addr, name in pairs)
    return Partition(rows=rows, skipped=skipped, max_names=budget)
