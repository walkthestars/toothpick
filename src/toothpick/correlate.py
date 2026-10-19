"""
Toothpick corpus correlation -- script keywords x mangled RTTI class cores
(UNIT-SEM5 Chunk 5).

Binding contract: ``docs/units/UNIT-SEM5-PLAN.md`` chunk 5 ("Corpus
intersection": HOI4/Stellaris ``common/`` script keywords intersected with RTTI
class names, fuzzed the way the field log proved -- ``AddToWarEffect``
corresponding to ``add_to_war``).  Design anchors:
``docs/design-document.md`` section 1 (every row traces to a supplied fact --
here to the caller's own markdown text and the caller's own mangled strings,
never to an inferred link) and section 19 (a label is only ever what was
proven), plus Risk R1 (bounded, counted emission).

Three pure functions do the work::

    harvest_keywords(markdown_text, *, strict=False)    -> List[str]
    camel_cores(mangled, *, keep_prefix=False)          -> List[str]
    correlate(keywords, mangled_names, *, max_items=...) -> Correlation

:func:`harvest_keywords` reads a markdown table of contents and returns its
``[name](#name)`` tokens -- the link *text* of every inline link whose
destination is a bare fragment (``#anchor``), trimmed of markdown decoration
(backticks, ``*`` emphasis), whitespace-collapsed, deduped by exact spelling and
kept in document order.  The scan is textual and deliberately shallow: no
markdown block parsing is done (a link written inside a fenced code block is
still harvested), reference-style links are not resolved, and an **image**'s alt
text is not a TOC token (a link preceded by ``!`` is skipped).  With
``strict=True`` the author's anchor must additionally be the fragment slug of
the link text (:func:`slugify`), i.e. a genuine ``[name](#name)`` TOC entry
rather than a cross-reference link to someone else's heading.

:func:`camel_cores` reads MSVC-mangled (or plain) class names and returns the
``C<Core>Effect|Trigger|Modifier|Scope`` tokens -- the tokens the plan's fuzzy
match keys on.  The string is split into identifier runs at every non-identifier
character (``?``, ``.``, ``$``, ``@``, ...); each run's leading decoration
marker (the MSVC pointer/qualifier codes ``P Q R S A B E`` plus a type code
``V``/``U``/``T``, repeated) is stripped, and the token regex is then applied to
what remains.  So ``?AVCAddToWarEffect@@``, ``.?AVCAddToWarEffect@@`` and the
template-nested ``?$vector@PEAVCAddToWarEffect@@V?$allocator@...@@@@@std@@`` all
yield ``AddToWarEffect``, and a plain ``AddToWarEffect`` does too.  Tokens are
deduped and returned in document order; the trailing suffix must *end* the
deduped and returned in document order; the trailing suffix must *end* the
identifier run (``CAddToWarEffectScope`` yields the single token
``AddToWarEffectScope``, a plural ``CAddToWarEffects`` yields nothing), and a
run whose ``@@`` terminator is followed by glued junk
(``?AVCAddToWarEffect@@s``) yields nothing either.  The
convention ``CAddToWarEffect``); ``keep_prefix=True`` returns the run's own
spelling.

:func:`correlate` intersects the two.  Keywords are matched to mangled names by
**canonical equality**, where ``canonical_key(word) = snake_case(word)`` with
every ``_`` removed -- so case and separator style do not matter (``add_to_war``,
``AddToWar`` and ``addToWar`` all canonicalise to ``addtowar``).  The
*candidate* canonicals of a mangled name are exactly these, in this order
(:data:`MATCH_RULES`):

1. ``prefix-strip-cv`` -- a leading ``C``/``V`` class-name prefix is stripped
   (:func:`strip_cv_prefix`), and the token is tried both with and without it.
2. ``suffix-variant-tries`` -- each candidate is tried whole and then with a
   trailing ``Effect``/``Trigger``/``Modifier``/``Scope`` removed, repeatedly
   (:func:`split_suffix`), so ``AddToWarEffectScope`` still reaches
   ``AddToWar``.
3. ``whole-name-plain-identifier`` -- a name that is a bare identifier (no
   mangling punctuation at all, e.g. ``AddToWar``) is tried whole as well, again
   with its ``C``/``V`` prefix stripped and its suffixes peeled.
4. ``canonical-equality`` -- a keyword matches when a candidate canonical equals
   its own canonical exactly.

Matching is deliberately **not** fuzzy beyond those rules: there is no substring
or edit-distance search, one mangled name may match several keywords, and a
keyword (or a name) that matches nothing is *reported* as unmatched rather than
guessed at.  Nothing here invents a link the caller's own strings do not state.

:func:`correlate` returns a :class:`Correlation`: ``pairs`` (the
``{keyword: (mangled, ...)}`` table in keyword input order), ``matches`` (that
table as a dict), ``unmatched_keywords``, ``unmatched_mangled``,
``skipped_keywords``/``skipped_mangled`` (R1-counted, never-inspected tail of
each supplied sequence) and ``max_items``.  ``as_dict()`` renders the whole
report JSON-friendly: ``{"matches": {keyword: [mangled, ...]},
"unmatched_keywords": [...], "unmatched_mangled": [...], ...}``.

Bounds (Risk R1): ``max_items`` defaults to :data:`DEFAULT_MAX_ITEMS`
(100 000) and is clamped to :data:`MAX_ITEMS` (1 000 000) by
:func:`resolve_max_items`, applied to *each* supplied sequence.  The cap bounds
work as well as output -- entries past it are counted and not inspected at all
(so a malformed entry past the cap is not a caller error), and the shortfall is
visible in the return value.

Honesty rules enforced here:

* A match is only ever between the caller's own keyword string and the caller's
  own mangled string.  No file, symbol table, RTTI reader or SQLite handle is
  consulted, and no "probably the same thing" guess is made.
* Both supplied sequences are left untouched and in place; duplicates are
  reported once, in first-appearance order.
* Malformed *arguments* are caller errors and raise :class:`CorrelateError` (a
  :class:`ValueError`): a non-``str`` markdown text/name, a bare string where a
  sequence of words is expected, a non-``str`` entry, a keyword with no
  canonical form at all (blank or punctuation-only), a bad ``max_items`` or
  ``keep_prefix``, a bad ``strict`` flag.

Pure, offline and deterministic: strings in, report out.  No clock, no
randomness, no dict-ordering dependence -- equal inputs return equal
correlations.

This module defines its own :class:`CorrelateError` and imports nothing from any
RTTI reader (`rtti_parse`/`rtti_store`/`rtti`) or from the other UNIT-SEM5
modules; it never imports ``faulthandler`` (Trap 11 -- suites run under
``-p no:faulthandler``) and opens no SQLite handle (so Trap 7's
``[WinError 32]`` leak class cannot originate here).
"""

from __future__ import annotations

import operator
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

__all__ = [
    "CORRELATE_PROVENANCE",
    "CorrelateError",
    "Correlation",
    "DEFAULT_MAX_ITEMS",
    "MATCH_RULES",
    "MAX_ITEMS",
    "SUFFIXES",
    "camel_core_bases",
    "camel_cores",
    "canonical_key",
    "correlate",
    "harvest_keywords",
    "resolve_max_items",
    "slugify",
    "snake_case",
    "split_suffix",
    "strip_cv_prefix",
]

#: Provenance token for the rule family this module implements (plan chunk 5:
#: "HOI4 + Stellaris ``common/`` script keywords x RTTI names, fuzzy").
CORRELATE_PROVENANCE = "script-keyword-x-rtti-core"

#: Default input cap per supplied sequence (Risk R1; plan chunk 5).
DEFAULT_MAX_ITEMS = 100_000

#: Hard ceiling for that cap; a larger budget is clamped, never honoured.
MAX_ITEMS = 1_000_000

#: The class-name suffixes a script keyword core is allowed to wear.  Order is
#: fixed and is the order :func:`split_suffix` tests them in (no two overlap).
SUFFIXES: Tuple[str, ...] = ("Effect", "Trigger", "Modifier", "Scope")

#: The documented match rules of :func:`correlate`, in the order they are tried.
#: Every rule is spelled out in the module docstring; this tuple is the
#: machine-readable statement of the same contract.
MATCH_RULES: Tuple[str, ...] = (
    "prefix-strip-cv",
    "suffix-variant-tries",
    "whole-name-plain-identifier",
    "canonical-equality",
)

#: Class-name prefixes stripped by :func:`strip_cv_prefix` (Hungarian ``C`` for a
#: class, MSVC's ``V`` class type code).
PREFIXES = "CV"

# --- markdown TOC harvesting ------------------------------------------------ #

#: An inline markdown link whose destination is a bare fragment.  The link text
#: may not contain brackets, and the destination may carry an optional title.
_LINK = re.compile(r"\[([^\[\]<>]*)\]\(\s*(#[^)\s\"']*)\s*(?:\"[^\"]*\"|'[^']*')?\s*\)")

#: Markdown decoration removed from a harvested name (backticks, ``*`` emphasis).
_EMPHASIS = re.compile(r"[`*]")

#: Whitespace runs inside a harvested name collapse to one space.
_WHITESPACE = re.compile(r"\s+")

#: Characters the module's slug rule keeps (everything else is dropped).
_SLUG_DROP = re.compile(r"[^0-9a-z_ -]")

#: Whitespace inside a name becomes the hyphen of a GitHub-style fragment slug.
_SLUG_SPACE = re.compile(r"\s+")

# --- identifier / core-token scanning --------------------------------------- #

#: One identifier run of a mangled name (runs are split at ``?``, ``.``, ``$``,
#: ``@`` and every other non-identifier character).
_RUN = re.compile(r"[A-Za-z0-9_]+")

#: Leading MSVC decoration marker: pointer/qualifier codes (``P Q R S A B E``)
#: plus one type code (``V`` class, ``U`` struct, ``T`` union), and only when a
#: capital follows -- so a real name like ``AddToWarEffect`` is never touched.
_DECORATION = re.compile(r"[PQRSABE]*[VUT](?=[A-Z])")

#: Upper bound on decoration strips per run (``?AUTC…`` peels twice, and a
#: handful of qualifier codes is all MSVC ever stacks).
_MAX_DECORATION_STRIPS = 8

#: ``C<Core>Effect|Trigger|Modifier|Scope`` -- the token the plan's fuzzy match
#: keys on.  The core must be followed by the suffix at the *end* of the run;
#: whether the run itself is closed honestly is judged against the original
#: string by :func:`_terminator_closes`, because the body this regex sees has
#: had every ``@`` split off it.
_CORE = re.compile(r"[CV]?([A-Z][A-Za-z0-9_]*?)(?:" + "|".join(SUFFIXES) + r")(?![A-Za-z0-9_])")

#: Trailing junk glued onto an ``@`` terminator: one or more ``@`` followed
#: directly by a lowercase identifier character.  MSVC spells type and scope
#: codes in capitals, so ``?AVCAddToWarEffect@@s`` is a mangled name with junk
#: after its ``@@`` terminator -- not a name that ends in ``Effect``.
_TERMINATOR_JUNK = re.compile(r"@+[a-z_]")

#: A bare, undecorated identifier (no ``?``/``.``/``$``/``@`` mangling at all).
_PLAIN_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# --- snake_case rules ------------------------------------------------------- #

#: Any run of non-identifier characters is one word separator.
_SEPARATOR = re.compile(r"[^A-Za-z0-9_]+")

#: ``lower``/digit -> ``Upper``: a new word.
_LOWER_UPPER = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

#: ``UPPER`` -> ``Upper``: the tail of an acronym run starts a new word
#: (``NSBGalaxy`` -> ``NSB_Galaxy``, ``HTMLParser`` -> ``HTML_Parser``).
_ACRONYM = re.compile(r"(?<=[A-Z])(?=[A-Z][a-z])")

#: Separator runs collapse to a single ``_``.
_UNDERSCORE_RUN = re.compile(r"_+")


class CorrelateError(ValueError):
    """Raised for a caller argument that cannot describe a correlation.

    A *word* is never a caller error -- any identifier can be canonicalised and
    any mangled string can be scanned (and simply yields no core token).  Only a
    malformed argument lands here: a non-``str`` markdown text, mangled name,
    keyword or token; a bare string where a sequence of words is expected; a
    keyword whose canonical form is empty; a bad ``max_items``, ``keep_prefix``
    or ``strict``.
    """


# --- small strict coercions ------------------------------------------------- #


def _coerce_int(value: Any, what: str) -> int:
    """An integer, or a refusal (booleans are not integers here)."""
    if isinstance(value, bool):
        raise CorrelateError(f"{what} must be an integer, got {value!r}")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise CorrelateError(f"{what} must be an integer, got {value!r}") from exc


def _coerce_text(value: Any, what: str) -> str:
    """A string, or a refusal (never a silent ``str()``)."""
    if not isinstance(value, str):
        raise CorrelateError(f"{what} must be a str, got {type(value).__name__}")
    return value


def _coerce_flag(value: Any, what: str) -> bool:
    """A boolean flag, or a refusal."""
    if not isinstance(value, bool):
        raise CorrelateError(f"{what} must be a bool, got {value!r}")
    return value


def resolve_max_items(value: Any = None) -> int:
    """Coerce the R1 input cap: default, else clamp to :data:`MAX_ITEMS`.

    ``None`` means "use the default" (:data:`DEFAULT_MAX_ITEMS`).  A bool, a
    float, a string or any other non-integer spelling is a caller error and
    raises (``"50"`` and ``3.5`` are refused, never guessed); a non-positive
    integer raises; an over-large integer is clamped to :data:`MAX_ITEMS`, so the
    answer is bounded either way.
    """
    if value is None:
        return DEFAULT_MAX_ITEMS
    budget = _coerce_int(value, "max_items")
    if budget <= 0:
        raise CorrelateError(f"max_items must be positive, got {budget}")
    return min(budget, MAX_ITEMS)


# --- snake_case / canonical form -------------------------------------------- #


def snake_case(camel: str) -> str:
    """The ``snake_case`` spelling of *camel*, by four documented, lossless steps.

    1. every run of non-identifier characters becomes one ``_`` (``-``, space,
       ``.``, ``?``, ``$``, ``@``, ... are all word separators);
    2. ``lower``/digit -> ``Upper`` gets an inserted ``_`` (``AddToWar`` ->
       ``Add_To_War``, ``Add2War`` -> ``Add2_War``);
    3. the tail of an acronym run gets an inserted ``_`` (``NSBGalaxy`` ->
       ``NSB_Galaxy``, ``HTMLParser`` -> ``HTML_Parser``, ``GetID`` ->
       ``Get_ID``);
    4. underscores collapse and the edges are trimmed, then the result is
       lowercased.

    So ``snake_case("AddToWarEffect") == "add_to_war_effect"``,
    ``snake_case("add_to_war") == "add_to_war"`` (already snake is a no-op) and
    ``snake_case("?AVCAddToWarEffect@@") == "avc_add_to_war_effect"``.  A blank
    or punctuation-only input yields ``""``; a non-``str`` raises
    :class:`CorrelateError`.
    """
    text = _coerce_text(camel, "camel")
    text = _SEPARATOR.sub("_", text)
    text = _LOWER_UPPER.sub("_", text)
    text = _ACRONYM.sub("_", text)
    text = _UNDERSCORE_RUN.sub("_", text)
    return text.strip("_").lower()


def canonical_key(word: str) -> str:
    """The matching form of *word*: :func:`snake_case` with every ``_`` removed.

    Case and separator style therefore do not matter -- ``add_to_war``,
    ``AddToWar``, ``addToWar`` and ``add-to-war`` all canonicalise to
    ``addtowar`` -- while the letters and digits themselves are compared exactly
    (no stemming, no plural folding, no substring search).  A blank or
    punctuation-only input yields ``""``.
    """
    return snake_case(word).replace("_", "")


def slugify(text: str) -> str:
    """The module's fragment-slug rule, used by strict TOC harvesting.

    Markdown decoration (backticks, ``*``) is dropped, the text is casefolded,
    every character outside ``0-9a-z``/``_``/``-``/space is dropped, whitespace
    runs become ``-`` and the edges are trimmed.  Thus ``"Add To War"`` ->
    ``"add-to-war"``, ``"AddToWar"`` -> ``"addtowar"`` and ``"add_to_war"`` ->
    ``"add_to_war"`` (underscores are kept: a script keyword anchor keeps its
    underscores).  A non-``str`` raises :class:`CorrelateError`.
    """
    folded = _EMPHASIS.sub("", _coerce_text(text, "text")).casefold()
    folded = _SLUG_DROP.sub("", folded)
    return _SLUG_SPACE.sub("-", folded.strip())


# --- TOC harvesting --------------------------------------------------------- #


def harvest_keywords(markdown_text: str, *, strict: bool = False) -> List[str]:
    """The ``[name](#name)`` TOC tokens of *markdown_text*, in document order.

    A token is the link *text* of an inline markdown link whose destination is a
    bare fragment (``#anchor``).  Backticks and ``*`` emphasis are stripped from
    the text, whitespace runs collapse to one space, and the caller's spelling is
    otherwise preserved byte for byte.  A link to a real destination
    (``other.md``, ``https://...``, ``other.md#anchor``), a fragment-only link
    with an empty anchor (``[name](#)``), an empty link text (``[](#anchor)``)
    and an image's alt text (``![name](#anchor)``) are not TOC tokens.  Names are
    deduped by exact spelling (first occurrence wins) and returned as a list.

    ``strict=True`` additionally requires the anchor to be the fragment slug of
    the link text (:func:`slugify`, case-insensitive), i.e. a genuine
    self-anchoring TOC entry rather than a cross-reference to somebody else's
    heading.  ``strict`` must be a bool, and *markdown_text* a ``str``
    (:class:`CorrelateError` otherwise); an empty text yields ``[]``.

    The scan is textual, not a markdown parse: a link written inside a fenced
    code block is still harvested, and reference-style links
    (``[name][ref]`` + ``[ref]: #name``) are not resolved.
    """
    text = _coerce_text(markdown_text, "markdown_text")
    only_self_anchored = _coerce_flag(strict, "strict")
    found: List[str] = []
    seen: Set[str] = set()
    for match in _LINK.finditer(text):
        if match.start() > 0 and text[match.start() - 1] == "!":
            continue  # an image's alt text, not a TOC token
        name = _clean_name(match.group(1))
        anchor = match.group(2)[1:]
        if not name or not anchor:
            continue
        if only_self_anchored and anchor.casefold() != slugify(name):
            continue
        if name in seen:
            continue
        seen.add(name)
        found.append(name)
    return found


def _clean_name(raw: str) -> str:
    """A link text as a keyword: decoration off, whitespace collapsed, trimmed."""
    return _WHITESPACE.sub(" ", _EMPHASIS.sub("", raw)).strip()


# --- mangled-name core tokens ----------------------------------------------- #


def strip_cv_prefix(token: str) -> str:
    """*token* without a single leading class-name prefix (``C`` or ``V``).

    ``"CAddToWarEffect"`` and ``"VAddToWarEffect"`` both give
    ``"AddToWarEffect"``; a token that starts with anything else -- or is a lone
    ``"C"`` -- is returned unchanged.  A non-``str`` raises
    :class:`CorrelateError`.
    """
    text = _coerce_text(token, "token")
    if len(text) > 1 and text[0] in PREFIXES:
        return text[1:]
    return text


def split_suffix(token: str) -> Tuple[str, Optional[str]]:
    """Split a trailing :data:`SUFFIXES` member off *token*.

    Returns ``(core, suffix)`` where ``suffix`` is one of ``Effect``,
    ``Trigger``, ``Modifier``, ``Scope`` (tested in that order) and ``core`` the
    non-empty remainder -- or ``(token, None)`` when the token wears no suffix.
    A bare suffix (``"Effect"``) is *not* split, because the core would be empty.
    Matching is case-sensitive, so ``"add_to_war_effect"`` is returned unchanged.
    A non-``str`` raises :class:`CorrelateError`.
    """
    text = _coerce_text(token, "token")
    for suffix in SUFFIXES:
        if len(text) > len(suffix) and text.endswith(suffix):
            return text[: -len(suffix)], suffix
    return text, None


def _strip_decoration(run: str) -> str:
    """*run* with its leading MSVC decoration marker(s) removed."""
    body = run
    for _ in range(_MAX_DECORATION_STRIPS):
        match = _DECORATION.match(body)
        if match is None or match.end() == 0:
            break
        body = body[match.end() :]
    return body


def _terminator_closes(text: str, end: int) -> bool:
    """True when the identifier run ending at *end* is closed honestly.

    Three shapes are honest: the run ends the string, the run is followed by a
    plain separator (``>``, a space, ``?``, ``$``, ...), or the run is closed by
    an ``@`` terminator/scope separator whose own successor is not glued junk.
    ``?AVCAddToWarEffect@@`` (end of string after ``@@``),
    ``?AVCAddToWarEffect@Scope@@`` (an uppercase scope code) and the
    template-nested ``…AddToWarEffect@@V?$allocator…`` (an uppercase type code)
    all pass; ``?AVCAddToWarEffect@@s`` fails, because a lowercase identifier
    character glued onto the terminator makes the run junk rather than the name
    it appears to spell (:data:`_TERMINATOR_JUNK`).
    """
    if end >= len(text) or text[end] != "@":
        return True
    return _TERMINATOR_JUNK.match(text, end) is None


def camel_cores(mangled: str, *, keep_prefix: bool = False) -> List[str]:
    """The ``C<Core>Effect|Trigger|Modifier|Scope`` tokens of *mangled*.

    The string is split into identifier runs at every non-identifier character;
    each run's leading MSVC decoration marker (pointer/qualifier codes ``P Q R S
    A B E`` plus one type code ``V``/``U``/``T``, repeatable) is stripped, and
    :data:`SUFFIXES`-terminated tokens are read off what remains.  A token must
    end its run *and* that run must be closed honestly: ``?AVCAddToWarEffect@@``,
    ``.?AVCAddToWarTrigger@@`` and the template-nested
    ``?$vector@PEAVCAddToWarEffect@@V?$allocator@…@@@@@std@@`` all yield their
    token, while ``CAddToWarEffects`` (plural -- the suffix no longer ends the
    run) and ``?AVCAddToWarEffect@@s`` (junk glued onto the ``@@`` terminator,
    see :func:`_terminator_closes`) yield nothing.

    Tokens are deduped and returned in document order.  By default the spelling
    returned is the **core token** with a single leading ``C``/``V`` class-name
    prefix removed (``"AddToWarEffect"``); ``keep_prefix=True`` returns the run's
    own spelling (``"CAddToWarEffect"``).  A non-``str``, or a non-bool
    ``keep_prefix``, raises :class:`CorrelateError`.
    """
    text = _coerce_text(mangled, "mangled")
    same_spelling = _coerce_flag(keep_prefix, "keep_prefix")
    tokens: List[str] = []
    seen: Set[str] = set()
    for run in _RUN.finditer(text):
        raw = run.group(0)
        body = _strip_decoration(raw)
        body_start = run.start() + len(raw) - len(body)
        for match in _CORE.finditer(body):
            # the body has no ``@`` left in it, so the terminator that closes the
            # run is judged against the original string
            if not _terminator_closes(text, body_start + match.end()):
                continue  # trailing junk after the terminator: not that name
            token = match.group(0) if same_spelling else strip_cv_prefix(match.group(0))
            if token and token not in seen:
                seen.add(token)
                tokens.append(token)
    return tokens


def camel_core_bases(mangled: str, *, keep_prefix: bool = False) -> List[str]:
    """The :func:`camel_cores` tokens with one trailing suffix removed.

    ``"AddToWar"`` is the base of ``"AddToWarEffect"``; a token that wears no
    suffix is absent from the result (it has no base to report), so
    ``camel_core_bases("CAddToWarEffect")`` is ``["AddToWar"]`` while
    ``camel_core_bases("CAddToWar")`` is ``[]``.  Deduped, document order; the
    ``keep_prefix`` flag is forwarded to :func:`camel_cores`.
    """
    bases: List[str] = []
    for token in camel_cores(mangled, keep_prefix=keep_prefix):
        core, suffix = split_suffix(token)
        if suffix is not None and core not in bases:
            bases.append(core)
    return bases


# --- correlation ------------------------------------------------------------ #


def _add_chain(base: str, add: Any) -> None:
    """Feed *base* and every suffix-peeled ancestor of it to *add*."""
    add(base)
    current = base
    core, suffix = split_suffix(current)
    while suffix is not None:
        add(core)
        current = core
        core, suffix = split_suffix(current)


def _variants(name: str) -> Tuple[str, ...]:
    """Every candidate spelling of *name* the documented match rules try.

    Rule order is :data:`MATCH_RULES`: each core token whole and prefix-stripped,
    each of those with its suffixes peeled one at a time, and -- for a name that
    is a bare identifier -- the name itself (again prefix-stripped and peeled).
    Returns an order-preserving tuple of unique spellings.
    """
    candidates: List[str] = []

    def add(value: str) -> None:
        if value and value not in candidates:
            candidates.append(value)

    for token in camel_cores(name):
        for base in (token, strip_cv_prefix(token)):
            _add_chain(base, add)
    if _PLAIN_IDENTIFIER.fullmatch(name) is not None:
        for base in (name, strip_cv_prefix(name)):
            _add_chain(base, add)
    return tuple(candidates)


def _coerce_words(value: Any, what: str, budget: int) -> Tuple[List[str], int]:
    """The words worth inspecting, plus the count of entries past the R1 cap.

    ``None`` means "nothing supplied" (an honest empty input).  The cap bounds
    work as well as output: once *budget* words have been taken, every remaining
    entry is counted and left uninspected (so it is neither scanned nor
    validated).  A bare string, a mapping or any other non-sequence is a caller
    error and raises :class:`CorrelateError`.
    """
    if value is None:
        return [], 0
    if isinstance(value, Mapping):
        raise CorrelateError(
            f"{what} must be a sequence of words, got {type(value).__name__}"
        )
    if isinstance(value, (str, bytes, bytearray)):
        raise CorrelateError(
            f"{what} must be a sequence of words, got {type(value).__name__}"
        )
    try:
        items = list(value)
    except TypeError as exc:
        raise CorrelateError(
            f"{what} must be a sequence of words, got {type(value).__name__}"
        ) from exc
    words: List[str] = []
    skipped = 0
    for item in items:
        if len(words) >= budget:
            skipped += 1
            continue
        words.append(_coerce_text(item, f"{what}[{len(words)}]"))
    return words, skipped


@dataclass(frozen=True)
class Correlation:
    """The result of :func:`correlate`: matches plus both unmatched sides.

    ``pairs`` holds ``(keyword, (mangled, ...))`` for every keyword that matched
    at least one name, in keyword input order; ``matches`` is the same table as a
    ``{keyword: (mangled, ...)}`` dict (the plan's literal ask) and
    ``matches_for(keyword)`` looks one spelling up (``()`` when it matched
    nothing).  ``unmatched_keywords`` and ``unmatched_mangled`` list the
    first-appearance spellings of everything that matched nothing -- never
    dropped, never guessed at.  ``skipped_keywords``/``skipped_mangled`` count
    the entries past the R1 ``max_items`` cap that were never inspected, and
    ``max_items`` records the cap in force.

    ``as_dict()`` renders the whole report JSON-friendly;
    ``result["matches"]`` / ``result["unmatched_keywords"]`` /
    ``result["unmatched_mangled"]`` and ``result[keyword]`` are supported, as is
    ``len(result)`` (matched keywords) and ``keyword in result``.  The
    dataclass is frozen and hashable (it holds only tuples and ints).
    """

    pairs: Tuple[Tuple[str, Tuple[str, ...]], ...] = ()
    unmatched_keywords: Tuple[str, ...] = ()
    unmatched_mangled: Tuple[str, ...] = ()
    skipped_keywords: int = 0
    skipped_mangled: int = 0
    max_items: int = DEFAULT_MAX_ITEMS

    def __post_init__(self) -> None:
        for word, names in self.pairs:
            if not isinstance(word, str):
                raise CorrelateError(f"pair keys must be str, got {type(word).__name__}")
            if not isinstance(names, tuple) or not all(
                isinstance(name, str) for name in names
            ):
                raise CorrelateError(f"pair values must be tuples of str, got {names!r}")
        for field in ("unmatched_keywords", "unmatched_mangled"):
            values = getattr(self, field)
            if not isinstance(values, tuple) or not all(
                isinstance(value, str) for value in values
            ):
                raise CorrelateError(f"{field} must be a tuple of str, got {values!r}")
        for field in ("skipped_keywords", "skipped_mangled"):
            count = getattr(self, field)
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise CorrelateError(f"{field} must be a non-negative int, got {count!r}")
        if (
            not isinstance(self.max_items, int)
            or isinstance(self.max_items, bool)
            or self.max_items < 1
        ):
            raise CorrelateError(f"max_items must be a positive int, got {self.max_items!r}")

    @property
    def matches(self) -> Dict[str, Tuple[str, ...]]:
        """The ``{keyword: (mangled, ...)}`` table, in keyword input order."""
        return dict(self.pairs)

    def matches_for(self, keyword: str) -> Tuple[str, ...]:
        """The mangled names matched by *keyword* (its exact spelling), else ``()``."""
        wanted = _coerce_text(keyword, "keyword")
        for word, names in self.pairs:
            if word == wanted:
                return names
        return ()

    def matched_keywords(self) -> Tuple[str, ...]:
        """Every keyword that matched at least one name, in input order."""
        return tuple(word for word, _ in self.pairs)

    def as_dict(self) -> Dict[str, Any]:
        """A JSON-friendly report: matches, both unmatched lists, caps and skips."""
        return {
            "matches": {word: list(names) for word, names in self.pairs},
            "unmatched_keywords": list(self.unmatched_keywords),
            "unmatched_mangled": list(self.unmatched_mangled),
            "skipped_keywords": self.skipped_keywords,
            "skipped_mangled": self.skipped_mangled,
            "max_items": self.max_items,
        }

    def __getitem__(self, key: Any) -> Any:
        if not isinstance(key, str):
            raise KeyError(key)
        if key == "matches":
            return self.matches
        if key == "unmatched_keywords":
            return self.unmatched_keywords
        if key == "unmatched_mangled":
            return self.unmatched_mangled
        if key == "skipped_keywords":
            return self.skipped_keywords
        if key == "skipped_mangled":
            return self.skipped_mangled
        if key == "max_items":
            return self.max_items
        if key == "pairs":
            return self.pairs
        for word, names in self.pairs:
            if word == key:
                return names
        raise KeyError(key)

    def __contains__(self, key: Any) -> bool:
        if not isinstance(key, str):
            return False
        if key in {
            "matches",
            "unmatched_keywords",
            "unmatched_mangled",
            "skipped_keywords",
            "skipped_mangled",
            "max_items",
            "pairs",
        }:
            return True
        return any(word == key for word, _ in self.pairs)

    def __len__(self) -> int:
        return len(self.pairs)


def correlate(
    keywords: Iterable[str],
    mangled_names: Iterable[str],
    *,
    max_items: Any = DEFAULT_MAX_ITEMS,
) -> Correlation:
    """Intersect script *keywords* with *mangled_names* by the documented rules.

    ``keywords`` (typically :func:`harvest_keywords` output) and
    ``mangled_names`` (typically mangled RTTI class names) are sequences of
    strings; ``None`` means "nothing supplied".  Each keyword is matched against
    each name by canonical equality over the candidate spellings of the name --
    the four rules of :data:`MATCH_RULES`, spelled out in the module docstring:
    strip a leading ``C``/``V`` prefix, try the token whole and with its
    ``Effect``/``Trigger``/``Modifier``/``Scope`` suffixes peeled repeatedly, try
    a bare identifier name whole, and compare the canonical forms
    (:func:`canonical_key`).  Matching is many-to-many: one name may match
    several keywords, and nothing beyond those rules is fuzzed -- no substring
    search, no guessed link.

    Returns a :class:`Correlation` carrying ``matches`` (``{keyword: (mangled,
    ...)}`` in keyword input order), ``unmatched_keywords``,
    ``unmatched_mangled``, the R1 skip counts and the ``max_items`` cap in force.
    Duplicate spellings are reported once (first occurrence wins) and the
    caller's own sequences are never mutated.

    ``max_items`` is the per-sequence input cap (Risk R1,
    :func:`resolve_max_items`: default :data:`DEFAULT_MAX_ITEMS` = 100 000,
    clamped to :data:`MAX_ITEMS` = 1 000 000); entries past it are counted in
    ``skipped_keywords``/``skipped_mangled`` and never inspected.

    Raises :class:`CorrelateError` for a malformed argument (a bare string or
    mapping where a sequence of words is expected, a non-``str`` entry, a
    keyword with no canonical form at all, a bad ``max_items``).  Two runs over
    equal inputs return equal correlations.
    """
    budget = resolve_max_items(max_items)
    words, skipped_keywords = _coerce_words(keywords, "keywords", budget)
    names, skipped_mangled = _coerce_words(mangled_names, "mangled_names", budget)

    ordered_words: List[str] = []
    word_keys: List[str] = []
    seen_words: Set[str] = set()
    for word in words:
        if word in seen_words:
            continue
        key = canonical_key(word)
        if not key:
            raise CorrelateError(
                f"keyword {word!r} has no canonical form (blank or punctuation-only)"
            )
        seen_words.add(word)
        ordered_words.append(word)
        word_keys.append(key)

    ordered_names: List[str] = []
    seen_names: Set[str] = set()
    for name in names:
        if name in seen_names:
            continue
        seen_names.add(name)
        ordered_names.append(name)

    by_key: Dict[str, Set[int]] = {}
    for index, name in enumerate(ordered_names):
        for variant in _variants(name):
            key = canonical_key(variant)
            if key:
                by_key.setdefault(key, set()).add(index)

    pairs: List[Tuple[str, Tuple[str, ...]]] = []
    unmatched_words: List[str] = []
    matched_names: Set[int] = set()
    for word, key in zip(ordered_words, word_keys):
        bucket = by_key.get(key)
        if not bucket:
            unmatched_words.append(word)
            continue
        indices = sorted(bucket)
        matched_names.update(indices)
        pairs.append((word, tuple(ordered_names[index] for index in indices)))

    unmatched_names = tuple(
        name for index, name in enumerate(ordered_names) if index not in matched_names
    )
    return Correlation(
        pairs=tuple(pairs),
        unmatched_keywords=tuple(unmatched_words),
        unmatched_mangled=unmatched_names,
        skipped_keywords=skipped_keywords,
        skipped_mangled=skipped_mangled,
        max_items=budget,
    )
