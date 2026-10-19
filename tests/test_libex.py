"""Tests for UNIT-SEM5 Chunk 4 (`docs/units/UNIT-SEM5-PLAN.md`): library exclusion.

Binding contract: chunk 4 of the unit plan (`src/toothpick/libex.py`) --
namespace-prefix rules over supplied RTTI/vendor class names (`std::`, `tbb::`,
`Concurrency::`, MFC `CFile*`/`CString`/`CException` families, `__imp`/thunk
markers, Lua/Python stdlib markers) produce an `{excluded, kept}` partition of the
supplied `(addr, name)` pairs **with reasons**; nothing is ever deleted, only
labelled.  Custom rules (prefix / regex / kind) are appended on top of the default
floor.  Pure / offline / deterministic, stdlib only.  Design SS1 / SS19: a label is
only ever what the caller's own string proves -- no symbol is read, no vendor is
inferred from an address -- plus Risk R1 (bounded input: default 100k, ceiling 1M,
with counted skips in both directions of the cap).

Battery:

1. Module surface: constants, `__all__`, `LibexError` is a `ValueError`, the
   default rule floor's shape and provenance, and the R1 cap helpers
   (`resolve_max_names` default / clamp / refusals).
2. Default-rule hits: every family named by the plan -- `std::`, `tbb::`,
   `Concurrency::`, `spdlog`, the MFC `CFile*`/`CString`/`CException`/`CArchive`
   families (demangled and raw `.?AV...@@` descriptor spellings), `__imp`/`ILT+`,
   `[thunk]`/`_thunk`/`??_E`, Lua and Python stdlib markers -- each with its exact
   reason token.
3. Default-rule misses: game-side names (`CGameApplication`, `Player`,
   `AddToWarEffect`, `numpy`, `CFilesystem`, `mystd::State`, `tbbThing`...) stay
   `kept` with `KEEP_REASON` and no rule -- the boundaries the regexes deliberately
   do not cross are asserted, so an over-eager rule cannot creep in silently.
4. Custom rules: prefix / contains / regex / kind exclusions, the `KEPT` override
   that carves a name back out of a default verdict, first-match-wins ordering,
   the four rule spellings (`Rule`, mapping, 3-tuple, 4-tuple), a lone `Rule`, and
   malformed rules as `LibexError`.
5. Reasons: a curated name -> reason table; every row's reason is non-empty and
   equal to its rule's reason; the default keep has `rule is None`; a kept rule
   keeps its own reason; `as_dict()` is JSON-friendly throughout.
6. Caps: default/clamp/refusal of `resolve_max_names`, `max_names` bounding the
   *work* (`skipped` counted, rows bounded, `total` conserved, a malformed entry
   past the cap not a caller error while one inside it is), and an empty input.
7. Partition bookkeeping: exhaustive (`rows + skipped == supplied`), order
   preserved, duplicates and duplicate addresses kept, lookups (`row_for`,
   `label_for`, `__getitem__`), and `Partition` validating its own invariants.
8. Input spellings: pairs, mappings, `addr -> name` mappings, pair-like objects,
   generators -- all the same verdict.
9. Hygiene: no `faulthandler`/`sqlite3`/network/`subprocess` import, no
   `shell=True`, no RTTI reader import, no mutation of the caller's containers, and
   determinism across runs and container types.

No real binary is required: every name here is a literal supplied in the test.
"""

from __future__ import annotations

import json
import re
from collections import namedtuple
from pathlib import Path

import pytest

from toothpick import libex
from toothpick.libex import (
    DEFAULT_MAX_NAMES,
    DEFAULT_RULES,
    EXCLUDED,
    KEEP_REASON,
    KEPT,
    KIND_TAGS,
    MAX_NAMES,
    PROVENANCE,
    RULE_KINDS,
    SOURCE_CUSTOM,
    SOURCE_DEFAULT,
    Label,
    LibexError,
    Partition,
    Rule,
    classify_kind,
    default_rules,
    label_name,
    partition_names,
    resolve_max_names,
)

BASE = 0x140001000


def _pairs(*names: str) -> list:
    """`(addr, name)` pairs at consecutive addresses (caller's own evidence)."""
    return [(BASE + 0x10 * index, name) for index, name in enumerate(names)]


def _label(names, rules=None, **kwargs):
    """Partition a tuple of names and return `{name: (label, reason)}`."""
    part = partition_names(_pairs(*names), rules, **kwargs)
    return {row.name: (row.label, row.reason) for row in part.rows}


# --- 1. module surface ------------------------------------------------------ #


def test_module_constants_and_all():
    assert DEFAULT_MAX_NAMES == 100_000
    assert MAX_NAMES == 1_000_000
    assert (EXCLUDED, KEPT) == ("excluded", "kept")
    assert KEEP_REASON == "no-rule-matched"
    assert PROVENANCE == "namespace-prefix-rules"
    assert RULE_KINDS == ("prefix", "contains", "regex", "kind")
    assert KIND_TAGS == ("import", "thunk", "vtable", "rtti", "plain")
    assert libex.LABELS == (EXCLUDED, KEPT)
    assert libex.SOURCES == (SOURCE_DEFAULT, SOURCE_CUSTOM)
    assert sorted(libex.__all__) == list(libex.__all__)
    assert len(set(libex.__all__)) == len(libex.__all__)
    for name in libex.__all__:
        assert hasattr(libex, name), name


def test_libex_error_is_a_value_error():
    assert issubclass(LibexError, ValueError)
    assert issubclass(libex.LibexError, ValueError)


def test_default_rule_floor_shape_and_provenance():
    rules = default_rules()
    assert rules is DEFAULT_RULES
    assert isinstance(rules, tuple) and len(rules) >= 8
    assert all(isinstance(rule, Rule) for rule in rules)
    assert all(rule.source == SOURCE_DEFAULT for rule in rules)
    assert all(rule.label == EXCLUDED for rule in rules)
    assert all(rule.reason for rule in rules)
    # Reasons are stable tokens, not prose: no whitespace, and unique per family.
    assert all(" " not in rule.reason for rule in rules)
    for family in (
        "vendor-namespace:std",
        "vendor-namespace:tbb",
        "vendor-namespace:Concurrency",
        "vendor-library:spdlog",
        "mfc-class:CFile-family",
        "mfc-class:CString-family",
        "mfc-class:CException-family",
        "import-marker",
        "thunk-marker",
        "stdlib-marker:lua",
        "stdlib-marker:python",
    ):
        assert family in {rule.reason for rule in rules}, family


def test_rule_is_frozen_and_reports_itself():
    rule = Rule("prefix", "CGame", "game-internal")
    assert rule.source == SOURCE_CUSTOM and rule.label == EXCLUDED
    assert rule.describe() == "prefix:CGame"
    assert rule.as_dict() == {
        "kind": "prefix",
        "pattern": "CGame",
        "reason": "game-internal",
        "label": "excluded",
        "ignore_case": False,
        "source": "custom",
    }
    assert json.loads(json.dumps(rule.as_dict())) == rule.as_dict()
    assert rule.matches("CGameApplication") and not rule.matches("Player")
    with pytest.raises(Exception):
        rule.pattern = "other"  # frozen dataclass: never mutated in place


@pytest.mark.parametrize("kind", RULE_KINDS)
def test_rule_matches_each_kind(kind):
    pattern = {"prefix": "CGame", "contains": "War", "regex": r"^C[A-Z]orld$", "kind": "vtable"}[
        kind
    ]
    rule = Rule(kind, pattern, f"custom-{kind}")
    positives = {
        "prefix": ["CGameApplication"],
        "contains": ["AddToWarEffect", "War"],
        "regex": ["CWorld"],
        "kind": ["??_7CWorld@@", "_ZTV4Base"],
    }[kind]
    negatives = {
        "prefix": ["GameApplication"],
        "contains": ["AddToPeaceEffect"],
        "regex": ["CWorld2"],
        "kind": ["player_update", "??_R0CWorld@@"],
    }[kind]
    for name in positives:
        assert rule.matches(name), (kind, name)
    for name in negatives:
        assert not rule.matches(name), (kind, name)


def test_rule_regex_uses_search_and_ignore_case():
    assert Rule("regex", r"War", "war-word").matches("AddToWarEffect")
    assert not Rule("regex", r"^War", "war-prefix").matches("AddToWarEffect")
    case_sensitive = Rule("prefix", "spdlog", "spdlog")
    insensitive = Rule("prefix", "SPDLOG", "spdlog", ignore_case=True)
    assert case_sensitive.matches("spdlog::logger")
    assert not case_sensitive.matches("Spdlog::Logger")
    assert insensitive.matches("Spdlog::Logger")


@pytest.mark.parametrize(
    "kind,pattern",
    [("nope", "x"), ("regex", "("), ("prefix", ""), ("kind", "gadget"), ("kind", "plain-ish")],
)
def test_bad_rule_kind_or_pattern_is_refused(kind, pattern):
    with pytest.raises(LibexError):
        Rule(kind, pattern, "why")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reason": ""},
        {"reason": None},
        {"reason": 7},
        {"label": "maybe"},
        {"label": None},
        {"ignore_case": "yes"},
        {"source": "imported"},
    ],
)
def test_bad_rule_fields_are_refused(kwargs):
    raw = {"kind": "prefix", "pattern": "CGame", "reason": "game-internal"}
    raw.update(kwargs)
    with pytest.raises(LibexError):
        Rule(**raw)


# --- 2. default-rule hits --------------------------------------------------- #


@pytest.mark.parametrize(
    "name,reason",
    [
        ("std::string", "vendor-namespace:std"),
        ("std::vector<int>", "vendor-namespace:std"),
        ("std__string", "vendor-namespace:std"),
        ("std", "vendor-namespace:std"),
        ("::std::map", "vendor-namespace:std"),
        ("tbb::task_group", "vendor-namespace:tbb"),
        ("tbb_task", "vendor-namespace:tbb"),
        ("tbb", "vendor-namespace:tbb"),
        ("__TBB_runtime_loader", "vendor-namespace:tbb"),
        ("Concurrency::task", "vendor-namespace:Concurrency"),
        ("Concurrency", "vendor-namespace:Concurrency"),
        ("spdlog::logger", "vendor-library:spdlog"),
        ("spdlog", "vendor-library:spdlog"),
        ("CFile", "mfc-class:CFile-family"),
        ("CFileException", "mfc-class:CFile-family"),
        ("CStdioFile", "mfc-class:CFile-family"),
        ("CMemFile", "mfc-class:CFile-family"),
        ("CSocketFile", "mfc-class:CFile-family"),
        ("CString", "mfc-class:CString-family"),
        ("CStringW", "mfc-class:CString-family"),
        ("CException", "mfc-class:CException-family"),
        ("CFileException::GetErrorMessage", "mfc-class:CFile-family"),
        ("CArchive", "mfc-class:CArchive-family"),
        ("__imp_?LoadLibrary@@YAXPEBD@Z", "import-marker"),
        ("__imp_GetProcAddress", "import-marker"),
        ("_imp__GetModuleHandleW@4", "import-marker"),
        ("ILT+120_?LoadLibrary@@YAXPEBD@Z", "import-marker"),
        ("?LoadLibrary@@YAXPEBD@Z (Import Thunk)", "import-marker"),
        ("[thunk]: ?f@CWorld@@UEAAXXZ", "thunk-marker"),
        ("_thunk_FUN_00401000", "thunk-marker"),
        ("??_ECWorld@@UEAAPEAXI@Z", "thunk-marker"),
        ("j_?LoadLibrary@@YAXPEBD@Z", "thunk-marker"),
        ("lua_pcall", "stdlib-marker:lua"),
        ("luaL_newstate", "stdlib-marker:lua"),
        ("luaopen_string", "stdlib-marker:lua"),
        ("LUA_VERSION", "stdlib-marker:lua"),
        ("Py_DecRef", "stdlib-marker:python"),
        ("Py_Initialize", "stdlib-marker:python"),
        ("PyObject_GetAttrString", "stdlib-marker:python"),
        ("_PyInterpreterState_Get", "stdlib-marker:python"),
        ("__pyx_module_total", "stdlib-marker:python"),
        ("pymalloc_alloc", "stdlib-marker:python"),
        ("Python", "stdlib-marker:python"),
    ],
)
def test_default_rule_hits(name, reason):
    part = partition_names(_pairs(name))
    (row,) = part.rows
    assert row.label == EXCLUDED, name
    assert row.reason == reason, (name, row.reason)
    assert row.rule is not None and isinstance(row.rule, Rule)
    assert row.rule.reason == reason
    assert row.rule.source == SOURCE_DEFAULT
    assert row.excluded and not row.kept
    assert part.excluded == (row,) and part.kept == ()
    assert part.excluded_names() == (name,)


@pytest.mark.parametrize(
    "name",
    [".?AVCFile@@", ".?AVCException@@", ".?AVCString@@", ".?AVCArchive@@", ".?AVCStdioFile@@"],
)
def test_default_hits_on_raw_decorated_rtti_descriptors(name):
    """A raw `.?AV<Class>@@` descriptor spells the same MFC class as `class CFile`."""
    part = partition_names(_pairs(name))
    (row,) = part.rows
    assert row.label == EXCLUDED
    assert row.reason.startswith("mfc-class:")


@pytest.mark.parametrize(
    "name",
    ["class CFile", "class CString", "struct CException", "CFile ", " ::CStdioFile"],
)
def test_default_hits_on_demangled_spellings(name):
    """`undname`-style prefixes and scope qualifiers do not defeat the rules."""
    part = partition_names(_pairs(name))
    (row,) = part.rows
    assert row.label == EXCLUDED
    assert row.reason.startswith("mfc-class:")
    assert row.name == name  # the reported name is the caller's string, untouched


# --- 3. default-rule misses ------------------------------------------------- #


@pytest.mark.parametrize(
    "name",
    [
        "CGameApplication",  # the game's own classes are never vendor
        "Player",
        "CWorld",
        "AddToWarEffect",
        "CModifier",
        "tbbThing",  # `tbb` family needs `::`, `_` or end
        "tbbx::task",
        "std_string_helper",  # `std` needs `::`, `__` or end -- no `std_`
        "mystd::State",
        "stdx",
        "Concurrencyy",
        "spdlogging",  # spdlog prefix rule is literal
        "CFilesystem",  # camel-case boundary: `CFile` + capital only
        "myCFile",
        "numpy",  # no Python-marker wording
        "Pyx",
        "luau_compile",  # not `lua*_`
        "type_info_lookup_fn",  # not a descriptor spelling
        "j_Player_update",  # `j_` is a thunk prefix -- proven thunk, not a keep
        "ILTX",
    ],
)
def test_default_rule_misses_are_kept_by_default(name):
    part = partition_names(_pairs(name))
    (row,) = part.rows
    if name in ("j_Player_update",):
        assert row.label == EXCLUDED and row.reason == "thunk-marker"
        return
    assert row.label == KEPT, name
    assert row.reason == KEEP_REASON
    assert row.rule is None  # nothing was matched: no invented rule
    assert row.kept and not row.excluded
    assert part.kept == (row,) and part.excluded == ()
    assert part.kept_names() == (name,)


def test_default_verdict_covers_every_supplied_pair():
    names = ["std::string", "CFile", "CGameApplication", "Player", "__imp_foo", "lua_pcall"]
    part = partition_names(_pairs(*names))
    assert part.excluded_names() == ("std::string", "CFile", "__imp_foo", "lua_pcall")
    assert part.kept_names() == ("CGameApplication", "Player")
    assert len(part.rows) == len(names)
    assert len(part.excluded) + len(part.kept) == len(part.rows)


# --- 4. custom rules -------------------------------------------------------- #


def test_custom_prefix_rule_excludes_and_reasons():
    part = partition_names(
        _pairs("CGameApplication", "Player", "std::string"),
        [Rule("prefix", "CGame", "game-internal")],
    )
    by_name = {row.name: row for row in part.rows}
    assert by_name["CGameApplication"].label == EXCLUDED
    assert by_name["CGameApplication"].reason == "game-internal"
    assert by_name["CGameApplication"].rule.source == SOURCE_CUSTOM
    assert by_name["Player"].label == KEPT
    assert by_name["std::string"].reason == "vendor-namespace:std"


def test_custom_kept_rule_carves_a_name_out_of_a_default_verdict():
    rules = [
        Rule("prefix", "std::vector", "std-inline-allowed", label=KEPT),
        Rule("regex", r"^C(?:File|StdioFile|MemFile)", "game-subclass-of-file", label=KEPT),
    ]
    part = partition_names(
        _pairs("std::vector<int>", "std::string", "CFile", "CStdioFile"), rules
    )
    by_name = {row.name: row for row in part.rows}
    assert (by_name["std::vector<int>"].label, by_name["std::vector<int>"].reason) == (
        KEPT,
        "std-inline-allowed",
    )
    assert by_name["std::vector<int>"].rule.label == KEPT
    assert by_name["std::vector<int>"].kept
    assert by_name["std::string"].label == EXCLUDED
    # The custom rule is consulted before the defaults, so a family carve-out applies
    # to every MFC CFile-family name -- the caller's rule order is what decides.
    assert by_name["CFile"].reason == "game-subclass-of-file"
    assert by_name["CStdioFile"].reason == "game-subclass-of-file"


def test_custom_regex_and_kind_rules():
    rules = [
        Rule("regex", r"^(?:Vector|Matrix)[0-9]?$", "math-type"),
        Rule("kind", "vtable", "vtable-row"),
        Rule("kind", "rtti", "rtti-row"),
    ]
    part = partition_names(
        _pairs("Vector3", "Matrix", "Vectorfield", "??_7CWorld@@", ".?AVCGame@@"), rules
    )
    by_name = {row.name: row for row in part.rows}
    assert (by_name["Vector3"].label, by_name["Vector3"].reason) == (EXCLUDED, "math-type")
    assert (by_name["Matrix"].label, by_name["Matrix"].reason) == (EXCLUDED, "math-type")
    assert by_name["Vectorfield"].label == KEPT
    assert by_name["??_7CWorld@@"].reason == "vtable-row"
    assert by_name[".?AVCGame@@"].reason == "rtti-row"


def test_custom_contains_rule_and_first_match_wins():
    rules = [
        Rule("contains", "War", "war-effect"),
        Rule("contains", "War", "never-reached"),
        Rule("prefix", "AddTo", "addto-shape"),
    ]
    part = partition_names(_pairs("AddToWarEffect", "AddToPeaceEffect", "Player"), rules)
    by_name = {row.name: row for row in part.rows}
    assert by_name["AddToWarEffect"].reason == "war-effect"
    assert by_name["AddToPeaceEffect"].reason == "addto-shape"
    assert by_name["Player"].label == KEPT


@pytest.mark.parametrize(
    "spelling",
    [
        Rule("prefix", "CGame", "game-internal"),
        {"kind": "prefix", "pattern": "CGame", "reason": "game-internal"},
        ("prefix", "CGame", "game-internal"),
        ["prefix", "CGame", "game-internal"],
        {"kind": "prefix", "pattern": "CGame", "reason": "game-internal", "label": KEPT},
        ("prefix", "CGame", "game-internal", KEPT),
    ],
)
def test_rule_spellings_are_equivalent(spelling):
    expected = KEPT if "label" in getattr(spelling, "keys", lambda: [])() else EXCLUDED
    if isinstance(spelling, (tuple, list)) and len(spelling) == 4:
        expected = KEPT
    part = partition_names(_pairs("CGameApplication"), [spelling])
    (row,) = part.rows
    assert row.label == expected
    assert row.reason == "game-internal"
    assert row.rule.source == SOURCE_CUSTOM


def test_a_lone_rule_object_is_accepted():
    part = partition_names(_pairs("CGameApplication", "Player"), Rule("prefix", "CGame", "g"))
    assert part.excluded_names() == ("CGameApplication",)
    assert part.kept_names() == ("Player",)


def test_empty_rules_list_is_the_default_floor():
    names = ("std::string", "CGameApplication")
    assert partition_names(_pairs(*names), []).as_dict() == partition_names(_pairs(*names)).as_dict()


@pytest.mark.parametrize(
    "rules",
    [
        "CGame",
        42,
        None.__class__,  # a type is not a rule sequence
        [{"kind": "prefix", "pattern": "CGame"}],  # no reason
        [{"kind": "prefix", "reason": "why"}],  # no pattern
        [("prefix", "CGame", "")],  # empty reason, refused at construction
        [("prefix", "CGame", "why", EXCLUDED, "extra")],  # too long
        [42],
        ["CGame"],
        lambda: Rule("prefix", "CGame", ""),  # empty reason: refused at construction
    ],
)
def test_malformed_rules_are_caller_errors(rules):
    if callable(rules) and not isinstance(rules, type):
        # Constructed here on purpose: the constructor itself must refuse it.
        with pytest.raises(LibexError):
            rules()
        return
    with pytest.raises(LibexError):
        partition_names(_pairs("Player"), rules)


def test_bad_regex_rule_is_refused_before_any_matching():
    with pytest.raises(LibexError):
        partition_names(_pairs("Player"), [("regex", "(", "broken")])


# --- 5. reasons ------------------------------------------------------------- #


@pytest.mark.parametrize(
    "name,reason",
    [
        ("std::vector<int>", "vendor-namespace:std"),
        ("Player", KEEP_REASON),
        ("CFile", "mfc-class:CFile-family"),
    ],
)
def test_label_name_reports_the_same_reason(name, reason):
    row = label_name(name)
    assert row.label == (KEPT if reason == KEEP_REASON else EXCLUDED)
    assert row.reason == reason
    assert row.addr is None


def test_every_row_reason_is_non_empty_and_traces_to_its_rule():
    names = (
        "std::string",
        "CFile",
        "__imp_foo",
        "lua_pcall",
        "Py_Initialize",
        "CGameApplication",
        "??_7CWorld@@",
        "AddToWarEffect",
    )
    rules = [Rule("contains", "War", "war-effect")]
    part = partition_names(_pairs(*names), rules)
    assert len(part.rows) == len(names)
    for row in part.rows:
        assert isinstance(row.reason, str) and row.reason
        if row.label == EXCLUDED:
            assert row.rule is not None
            assert row.rule.reason == row.reason
            assert row.rule.label == EXCLUDED
        else:
            assert row.rule is None or row.rule.label == KEPT
            if row.rule is None:
                assert row.reason == KEEP_REASON
    by_name = {row.name: row for row in part.rows}
    assert by_name["AddToWarEffect"].reason == "war-effect"
    assert by_name["CGameApplication"].reason == KEEP_REASON
    assert by_name["??_7CWorld@@"].label == KEPT  # the plan's vtable marker is custom


def test_row_as_dict_is_json_friendly_and_indexable():
    row = label_name("std::string", addr=BASE)
    payload = row.as_dict()
    assert payload["addr"] == BASE
    assert payload["name"] == "std::string"
    assert payload["label"] == EXCLUDED
    assert payload["reason"] == "vendor-namespace:std"
    assert payload["rule"]["kind"] == "regex"
    assert json.loads(json.dumps(payload)) == payload
    assert row["name"] == "std::string" and row["reason"] == payload["reason"]
    assert row["addr"] == BASE
    with pytest.raises(KeyError):
        row["nope"]


def test_bad_label_and_row_are_refused():
    with pytest.raises(LibexError):
        Label(addr=BASE, name="Player", label="maybe", reason="x")
    with pytest.raises(LibexError):
        Label(addr=BASE, name="Player", label=KEPT, reason="")
    with pytest.raises(LibexError):
        Label(addr=BASE, name="Player", label=KEPT, reason="x", rule="not-a-rule")
    with pytest.raises(LibexError):
        Label(addr=-1, name="Player", label=KEPT, reason="x")


# --- 6. R1 caps ------------------------------------------------------------- #


def test_resolve_max_names_defaults_to_the_r1_default():
    assert resolve_max_names() == DEFAULT_MAX_NAMES == 100_000
    assert resolve_max_names(None) == DEFAULT_MAX_NAMES
    assert resolve_max_names(MAX_NAMES) == MAX_NAMES


def test_resolve_max_names_clamps_to_the_ceiling():
    assert resolve_max_names(MAX_NAMES + 1) == MAX_NAMES
    assert resolve_max_names(10**12) == MAX_NAMES
    assert MAX_NAMES == 1_000_000


@pytest.mark.parametrize("bad", [0, -1, -1000, 2.5, "50", True, False, [5], object()])
def test_resolve_max_names_refuses_bad_budgets(bad):
    with pytest.raises(LibexError):
        resolve_max_names(bad)


def test_default_cap_does_not_skip_a_realistic_list():
    names = [(BASE + index, f"CGameType{index}") for index in range(2500)]
    part = partition_names(names)
    assert part.skipped == 0
    assert part.max_names == DEFAULT_MAX_NAMES
    assert part.total == len(names) == len(part.rows)
    assert len(part.kept) == len(names)


def test_cap_bounds_rows_and_counts_the_skips():
    names = _pairs("Player", "std::string", "AddToWarEffect", "CFile", "lua_pcall")
    part = partition_names(names, max_names=3)
    assert part.max_names == 3
    assert len(part.rows) == 3
    assert part.skipped == 2
    assert part.total == 5
    assert part.excluded_names() == ("std::string",)
    assert part.kept_names() == ("Player", "AddToWarEffect")
    part1 = partition_names(names, max_names=1)
    assert len(part1.rows) == 1 and part1.skipped == 4


def test_entries_past_the_cap_are_counted_not_inspected():
    good = _pairs("Player", "Player")
    malformed = [("not-an-address", 5), "nope", object()]
    part = partition_names(good + malformed, max_names=2)
    assert len(part.rows) == 2
    assert part.skipped == len(malformed)
    assert part.total == len(good) + len(malformed)
    # Inside the cap the same entries are caller errors, so the cap is a bound on
    # work, not a licence to accept garbage.
    for entry in malformed:
        with pytest.raises(LibexError):
            partition_names(good + [entry], max_names=DEFAULT_MAX_NAMES)


def test_max_names_above_the_ceiling_is_clamped():
    part = partition_names(_pairs("Player"), max_names=MAX_NAMES * 3)
    assert part.max_names == MAX_NAMES
    assert part.skipped == 0


@pytest.mark.parametrize("bad", [0, -5, "3", 1.5, True])
def test_bad_max_names_is_a_caller_error(bad):
    with pytest.raises(LibexError):
        partition_names(_pairs("Player"), max_names=bad)


# --- 7. partition bookkeeping ----------------------------------------------- #


def test_empty_inputs_yield_an_empty_partition():
    for empty in ([], (), {}, iter([])):
        part = partition_names(empty)
        assert part.rows == () and part.excluded == () and part.kept == ()
        assert part.skipped == 0 and part.total == 0
        assert part.max_names == DEFAULT_MAX_NAMES
        assert len(part) == 0 and list(part) == []
        assert part.excluded_names() == () and part.kept_names() == ()
        assert part.row_for(BASE) is None and part.label_for(BASE) is None
        assert part.as_dict() == {
            "excluded": [],
            "kept": [],
            "skipped": 0,
            "max_names": DEFAULT_MAX_NAMES,
            "total": 0,
        }


def test_partition_is_exhaustive_and_order_preserving():
    names = [
        (BASE + 0x40, "Player"),
        (BASE + 0x10, "std::string"),
        (BASE + 0x10, "CFile"),  # same address, distinct evidence
        (BASE + 0x20, "Player"),  # same name, distinct evidence
        (BASE + 0x30, "numpy"),
    ]
    part = partition_names(names)
    assert [(row.addr, row.name) for row in part.rows] == names
    assert len(part.excluded) + len(part.kept) == len(part.rows) == len(names)
    assert part.excluded_names() == ("std::string", "CFile")
    assert part.kept_names() == ("Player", "Player", "numpy")
    assert part.skipped == 0 and part.total == len(names)
    assert part.row_for(BASE + 0x10).name == "std::string"
    assert part.row_for(BASE + 0x10).label == EXCLUDED
    assert part.label_for(BASE + 0x20) == KEPT
    assert part.row_for(BASE + 0x99) is None
    assert [row.name for row in part.excluded] == ["std::string", "CFile"]


def test_partition_mapping_access_and_iteration():
    part = partition_names(_pairs("std::string", "Player"))
    assert part["rows"] is part.rows
    assert part["excluded"] == part.excluded
    assert part["kept"] == part.kept
    assert part["skipped"] == 0
    assert part["max_names"] == DEFAULT_MAX_NAMES
    assert part["total"] == 2
    assert part[0] is part.rows[0]
    assert [row.name for row in part] == ["std::string", "Player"]
    with pytest.raises(KeyError):
        part["nope"]
    with pytest.raises(KeyError):
        part[object()]


def test_partition_from_labels_and_invariant_refusals():
    rows = (label_name("Player", addr=BASE), label_name("CFile", addr=BASE + 0x10))
    part = Partition(rows=rows, skipped=1, max_names=2)
    assert part.total == 3
    assert part.excluded_names() == ("CFile",)
    assert part.kept_names() == ("Player",)
    with pytest.raises(LibexError):
        Partition(rows=rows, skipped=-1)
    with pytest.raises(LibexError):
        Partition(rows=rows, skipped="1")
    with pytest.raises(LibexError):
        Partition(rows=rows, max_names=0)


def test_partition_report_is_json_friendly():
    part = partition_names(_pairs("std::string", "Player"), [Rule("prefix", "Play", "game-play")])
    report = part.as_dict()
    assert [row["name"] for row in report["excluded"]] == ["std::string", "Player"]
    assert report["excluded"][1]["reason"] == "game-play"
    assert report["kept"] == []
    assert report["total"] == 2 and report["skipped"] == 0
    assert json.loads(json.dumps(report)) == report


# --- 8. input spellings ----------------------------------------------------- #


Pair = namedtuple("Pair", "addr name")


def test_supplied_pair_spellings_agree():
    names = ["std::string", "Player", "CFile", "AddToWarEffect"]
    pairs = _pairs(*names)
    mapping = dict(pairs)
    objects = [Pair(addr, name) for addr, name in pairs]
    mapping_rows = [{"addr": addr, "name": name} for addr, name in pairs]
    address_rows = [{"address": addr, "name": name} for addr, name in pairs]
    expected = [(EXCLUDED, "vendor-namespace:std"), (KEPT, KEEP_REASON), (EXCLUDED, "mfc-class:CFile-family"), (KEPT, KEEP_REASON)]
    for spelling in (pairs, tuple(pairs), mapping, objects, mapping_rows, address_rows, iter(pairs)):
        part = partition_names(spelling)
        assert [(row.label, row.reason) for row in part.rows] == expected, spelling


def test_mapping_input_uses_keys_as_addresses_in_order():
    part = partition_names({BASE: "Player", BASE + 0x10: "std::string"})
    assert [(row.addr, row.name) for row in part.rows] == [
        (BASE, "Player"),
        (BASE + 0x10, "std::string"),
    ]
    assert part.excluded_names() == ("std::string",)


@pytest.mark.parametrize(
    "names",
    [
        None,
        5,
        "Player",
        b"Player",
        [None],
        ["Player"],
        [(1,)],
        [(1, "Player", "extra")],
        [(-1, "Player")],
        [(1.5, "Player")],
        [(True, "Player")],
        [(BASE, b"Player")],
        [(BASE, None)],
        [{}],
        [{"addr": BASE}],
        [{"name": "Player"}],
        [("1", "Player")],
    ],
)
def test_malformed_names_are_caller_errors(names):
    with pytest.raises(LibexError):
        partition_names(names)


def test_a_pair_object_with_only_a_name_attribute_is_refused():
    class Blob:
        name = "Player"

    with pytest.raises(LibexError):
        partition_names([Blob()])


# --- 9. hygiene, purity, determinism ---------------------------------------- #


def test_supplied_containers_are_never_mutated_or_deleted():
    pairs = _pairs("std::string", "Player", "CFile")
    snapshot = list(pairs)
    rules = [Rule("prefix", "Play", "game-play")]
    rules_snapshot = list(rules)
    mapping = dict(pairs)
    mapping_snapshot = dict(mapping)
    names = ["std::string", "Player", "CFile"]
    names_snapshot = list(names)
    partition_names(pairs, rules)
    partition_names(mapping, rules)
    assert pairs == snapshot
    assert rules == rules_snapshot
    assert mapping == mapping_snapshot
    assert names == names_snapshot


def test_two_runs_over_equal_inputs_are_equal():
    names = _pairs("std::string", "Player", "CFile", "__imp_foo", "??_7CWorld@@")
    rules = [Rule("regex", r"^C[A-Z]orld$", "game-type")]
    first = partition_names(names, rules)
    second = partition_names(list(names), list(rules))
    assert first.as_dict() == second.as_dict()
    assert first.rows == second.rows


def test_verdicts_depend_on_rules_not_on_ordering_of_the_rules_mapping():
    names = _pairs("std::string", "Player")
    forward = partition_names(names, {"kind": "prefix", "pattern": "Play", "reason": "g"})
    assert forward.as_dict() == partition_names(
        names, dict(reversed(list({"kind": "prefix", "pattern": "Play", "reason": "g"}.items())))
    ).as_dict()


@pytest.mark.parametrize(
    "name,tag",
    [
        ("__imp_GetProcAddress", "import"),
        ("ILT+10_foo", "import"),
        ("[thunk]: ?f@CWorld@@UEAAXXZ", "thunk"),
        ("??_ECWorld@@UEAAPEAXI@Z", "thunk"),
        ("??_7CWorld@@", "vtable"),
        ("_ZTV4Base", "vtable"),
        (".?AVCWorld@@", "rtti"),
        ("??_R0CWorld@@", "rtti"),
        ("_ZTS4Base", "rtti"),
        ("CWorld", "plain"),
        ("Player", "plain"),
    ],
)
def test_classify_kind_tags(name, tag):
    assert classify_kind(name) == tag
    assert classify_kind(name) in KIND_TAGS


def test_classify_kind_rejects_a_non_string():
    with pytest.raises(LibexError):
        classify_kind(b"Player")


def test_module_source_is_pure_offline_and_faulthandler_free():
    source = Path(libex.__file__).read_text(encoding="utf-8")
    assert re.search(r"^(?:import|from)\s+faulthandler", source, re.M) is None
    assert re.search(r"^(?:import|from)\s+sqlite3", source, re.M) is None
    assert (
        re.search(r"^(?:import|from)\s+(?:socket|ssl|urllib|http|requests|httpx)\b", source, re.M)
        is None
    )
    assert re.search(r"^(?:import|from)\s+subprocess", source, re.M) is None
    assert re.search(r"shell\s*=\s*True", source) is None
    assert "namespace-prefix-rules" in source


def test_module_imports_no_rtti_reader_and_defines_its_own_error():
    source = Path(libex.__file__).read_text(encoding="utf-8")
    assert re.search(r"^(?:import|from)\s+toothpick\.rtti", source, re.M) is None
    assert re.search(r"^(?:import|from)\s+\S*rtti\S*\s+import", source, re.M) is None
    assert "RttiError(" not in source
    assert "class LibexError" in source
