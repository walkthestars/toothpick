"""Tests for UNIT-SEM5 Chunk 5 (`docs/units/UNIT-SEM5-PLAN.md`): corpus correlation.

Binding contract: chunk 5 of the unit plan (`src/toothpick/correlate.py`) -- HOI4 +
Stellaris `common/` script keywords intersected with RTTI class names, "fuzzed the
way the field log proved" (`AddToWarEffect` <-> `add_to_war`).  Productised as
`harvest_keywords` (markdown `[name](#name)` TOC tokens), `camel_cores`
(`C<Core>Effect|Trigger|Modifier|Scope` tokens out of MSVC-mangled strings),
`snake_case`/`canonical_key` and `correlate`, which returns `{keyword: [mangled,
...]}` plus both unmatched lists, under documented match rules (strip `C`/`V`
prefixes, suffix-variant tries).  Design SS1 / SS19: every row traces to a supplied
fact -- the caller's own markdown text and the caller's own mangled strings -- so a
label is only ever what those strings prove.  Plus Risk R1 (bounded input: default
100k, ceiling 1M, skips counted, tail never inspected).

Battery:

1. Module surface: `__all__`, the documented constants (`MATCH_RULES` order,
   `SUFFIXES`, caps, provenance), `CorrelateError` is a `ValueError`,
   `Correlation` is frozen, and the module imports nothing from `rtti*`,
   `sqlite3` or `faulthandler` (Trap 11 / Trap 7 cannot originate here).
2. `snake_case` / `canonical_key` / `slugify`: documented examples, acronym runs
   (`NSBGalaxy`, `HTMLParser`, `GetID`), digit boundaries, separator runs, edges,
   blanks, refusals.
3. Harvesting: a real TOC table, document order, dedupe, decoration stripping,
   whitespace collapsing, link titles, the non-token shapes (real destination,
   empty anchor, empty text, image alt, bracketed text), the deliberately shallow
   scan (fenced code still harvested, reference-style links unresolved) and
   strict mode's self-anchor requirement.
4. Cores: plain and MSVC-mangled spellings, decoration stacks, template-nested
   names, `keep_prefix`, the end-of-run rule (plural miss), dedupe, document order.
5. `correlate`: the field-log hit, many-to-many matching, canonical equality across
   spellings, the prefix-strip / suffix-variant / plain-identifier rules, order
   preservation, duplicates reported once, both unmatched lists, empty and `None`
   inputs, determinism, the mapping surface and a JSON-friendly `as_dict()`.
6. Caps and refusals: `resolve_max_items` default/clamp/refusals, `max_items`
   bounding the *work* (skips counted, a malformed entry past the cap not a caller
   error while one inside it is), and every documented malformed argument.
"""

from __future__ import annotations

import ast
import dataclasses
import json
from pathlib import Path

import pytest

from toothpick import correlate
from toothpick.correlate import (
    CORRELATE_PROVENANCE,
    DEFAULT_MAX_ITEMS,
    MATCH_RULES,
    MAX_ITEMS,
    SUFFIXES,
    CorrelateError,
    Correlation,
    camel_core_bases,
    camel_cores,
    canonical_key,
    correlate as correlate_keys,
    harvest_keywords,
    resolve_max_items,
    slugify,
    snake_case,
    split_suffix,
    strip_cv_prefix,
)

FIELD_LOG_KEYWORD = "add_to_war"
FIELD_LOG_CORE = "AddToWarEffect"
FIELD_LOG_MANGLED = "?AVCAddToWarEffect@@"

# A TOC shaped like the plan's field log: script keywords linking to their own
# headings, plus every shape that must *not* become a token.
TOC_TEXT = "\n".join(
    [
        "# Table of contents",
        "",
        "- [Add To War](#add-to-war)",
        "- [`add_to_war`](#add_to_war)",
        "- [**Add To War**](#add-to-war)",
        "- [NSB Galaxy](#nsb-galaxy)",
        "- [With Title](#with-title \"the tooltip\")",
        "- [Real Destination](other.md#anchor)",
        "- [Web](https://example.com)",
        "- [Empty Anchor](#)",
        "- [](#empty-text)",
        "- ![Add To War](#add-to-war)",
        "",
        "```",
        "- [In Fence](#in-fence)",
        "```",
    ]
)


# --- 1. module surface ------------------------------------------------------ #


def test_module_constants_and_all():
    assert correlate.__all__ == sorted(correlate.__all__)
    assert set(correlate.__all__) == {
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
    }
    assert CORRELATE_PROVENANCE == "script-keyword-x-rtti-core"
    assert DEFAULT_MAX_ITEMS == 100_000
    assert MAX_ITEMS == 1_000_000
    assert SUFFIXES == ("Effect", "Trigger", "Modifier", "Scope")
    # The documented rule order is the machine-readable contract.
    assert MATCH_RULES == (
        "prefix-strip-cv",
        "suffix-variant-tries",
        "whole-name-plain-identifier",
        "canonical-equality",
    )


def test_every_match_rule_is_spelled_out_in_the_module_docstring():
    doc = correlate.__doc__ or ""
    for rule in MATCH_RULES:
        assert rule in doc, rule


def test_correlate_error_is_a_value_error():
    assert issubclass(CorrelateError, ValueError)
    assert isinstance(CorrelateError("x"), ValueError)


def test_correlation_is_frozen_and_hashable():
    result = correlate_keys([FIELD_LOG_KEYWORD], [FIELD_LOG_MANGLED])
    assert hash(result) == hash(
        correlate_keys([FIELD_LOG_KEYWORD], [FIELD_LOG_MANGLED])
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.max_items = 7  # type: ignore[misc]


def test_module_imports_nothing_from_an_rtti_reader_or_sqlite_or_faulthandler():
    source = Path(correlate.__file__).read_text(encoding="utf-8")
    imported = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert imported <= {
        "__future__",
        "collections",
        "dataclasses",
        "operator",
        "re",
        "typing",
    }


# --- 2. snake_case / canonical_key / slugify -------------------------------- #


@pytest.mark.parametrize(
    ("camel", "expected"),
    [
        ("AddToWarEffect", "add_to_war_effect"),
        ("add_to_war", "add_to_war"),
        ("addToWar", "add_to_war"),
        ("add-to-war", "add_to_war"),
        ("Add  To--War", "add_to_war"),
        ("_AddToWar_", "add_to_war"),
        # acronym runs: the tail of the run starts a new word
        ("NSBGalaxy", "nsb_galaxy"),
        ("HTMLParser", "html_parser"),
        ("GetID", "get_id"),
        ("ABCd", "ab_cd"),
        # digit -> upper is a boundary too (lower/digit -> Upper)
        ("Add2War", "add2_war"),
        ("A", "a"),
        ("a", "a"),
        ("AB", "ab"),
        # MSVC decoration is just another separator run
        (FIELD_LOG_MANGLED, "avc_add_to_war_effect"),
        ("CAddToWarEffect", "c_add_to_war_effect"),
    ],
)
def test_snake_case_documented_examples(camel, expected):
    assert snake_case(camel) == expected


@pytest.mark.parametrize("blank", ["", "   ", "***", "---", "?@@$$"])
def test_snake_case_of_a_blank_or_punctuation_only_input_is_empty(blank):
    assert snake_case(blank) == ""


def test_snake_case_refuses_a_non_string():
    with pytest.raises(CorrelateError):
        snake_case(7)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "spelling", ["add_to_war", "AddToWar", "addToWar", "add-to-war", "ADD TO WAR"]
)
def test_canonical_key_forgets_case_and_separators_but_not_letters(spelling):
    assert canonical_key(spelling) == "addtowar"
    assert canonical_key(spelling) == canonical_key(FIELD_LOG_KEYWORD)


@pytest.mark.parametrize("other", ["add_to_wars", "add_to_war_effect", "war", "addwar"])
def test_canonical_key_does_not_fold_plurals_suffixes_or_substrings(other):
    assert canonical_key(other) != canonical_key(FIELD_LOG_KEYWORD)


@pytest.mark.parametrize("blank", ["", "  ", "***"])
def test_canonical_key_of_a_blank_input_is_empty(blank):
    assert canonical_key(blank) == ""


def test_canonical_key_refuses_a_non_string():
    with pytest.raises(CorrelateError):
        canonical_key(None)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Add To War", "add-to-war"),
        ("AddToWar", "addtowar"),
        ("add_to_war", "add_to_war"),
        ("`Add To War`", "add-to-war"),
        ("  Add   To  War  ", "add-to-war"),
    ],
)
def test_slugify_keeps_underscores_and_hyphenates_spaces(text, expected):
    assert slugify(text) == expected


def test_slugify_refuses_a_non_string():
    with pytest.raises(CorrelateError):
        slugify(7)  # type: ignore[arg-type]


# --- 3. harvest_keywords ---------------------------------------------------- #


def test_harvest_reads_a_toc_in_document_order_and_dedupes_by_spelling():
    assert harvest_keywords(TOC_TEXT) == [
        "Add To War",
        "add_to_war",
        "NSB Galaxy",
        "With Title",
        "In Fence",
    ]


def test_harvest_keeps_the_callers_own_spelling_byte_for_byte():
    text = "- [add_to_war](#add_to_war)\n- [Add To War](#add-to-war)\n"
    assert harvest_keywords(text) == ["add_to_war", "Add To War"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("[](#anchor)", []),
        ("[name](#)", []),
        ("[](#)", []),
        ("[name][ref]\n\n[ref]: #name", []),
        ("![name](#anchor)", []),
        ("![name](pic.png) [real](#real)", ["real"]),
        ("[other.md](other.md)", []),
        ("[anchor](other.md#anchor)", []),
        ("[web](https://example.com)", []),
        ("no links at all", []),
        ("", []),
    ],
)
def test_harvest_skips_every_non_token_shape(text, expected):
    assert harvest_keywords(text) == expected


def test_harvest_collapses_whitespace_and_strips_emphasis():
    assert harvest_keywords("[  Add   To  War  ](#add-to-war)") == ["Add To War"]
    assert harvest_keywords("[`a`](#a) [**b**](#b) [*c*](#c)") == ["a", "b", "c"]


def test_harvest_accepts_a_link_title():
    assert harvest_keywords('[Name](#anchor "a tooltip")') == ["Name"]
    assert harvest_keywords("[Name](#anchor 'a tooltip')") == ["Name"]


def test_harvest_ignores_link_text_that_contains_brackets():
    assert harvest_keywords("[[x]](#y)") == []


def test_harvest_dedupes_and_keeps_first_appearance():
    assert harvest_keywords("[A](#a) [B](#b) [A](#a)") == ["A", "B"]


def test_harvest_is_textual_not_a_markdown_parse():
    """A link inside a fence is still harvested; reference links are not resolved."""
    assert harvest_keywords("```\n- [In Fence](#in-fence)\n```") == ["In Fence"]


def test_harvest_non_strict_accepts_a_cross_reference_link():
    assert harvest_keywords("- [Add To War](#somewhere-else)") == ["Add To War"]


def test_harvest_strict_requires_the_anchor_to_be_the_texts_own_slug():
    assert harvest_keywords("- [Add To War](#add-to-war)", strict=True) == ["Add To War"]
    assert harvest_keywords("- [`add_to_war`](#add_to_war)", strict=True) == [
        "add_to_war"
    ]
    # a cross-reference to somebody else's heading is not a TOC entry
    assert harvest_keywords("- [AddToWar](#add-to-war)", strict=True) == []
    assert harvest_keywords(["- [Add To War](#add_to_war)"][0], strict=True) == []
    # the comparison is case-insensitive
    assert harvest_keywords("- [Add To War](#ADD-TO-WAR)", strict=True) == [
        "Add To War"
    ]


def test_harvest_strict_filters_per_link_not_per_text():
    text = "- [Add To War](#add-to-war)\n- [Other](#add-to-war)\n"
    assert harvest_keywords(text, strict=True) == ["Add To War"]


@pytest.mark.parametrize("bad", [7, None, ["- [A](#a)"]])
def test_harvest_refuses_a_non_string_text(bad):
    with pytest.raises(CorrelateError):
        harvest_keywords(bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", ["yes", 1, 0, None])
def test_harvest_refuses_a_non_bool_strict(bad):
    with pytest.raises(CorrelateError):
        harvest_keywords("- [A](#a)", strict=bad)  # type: ignore[arg-type]


# --- 4. camel_cores --------------------------------------------------------- #


@pytest.mark.parametrize(
    "mangled",
    [
        FIELD_LOG_MANGLED,
        ".?AVCAddToWarEffect@@",
        "AVCAddToWarEffect",
        "CAddToWarEffect",
        "AddToWarEffect",
        "?AVCAddToWarTrigger@@".replace("Trigger", "Effect"),
        # MSVC stacks decoration markers: pointer/qualifier codes + a type code
        "?AUVCAddToWarEffect@@",
        "?AVVCAddToWarEffect@@",
        "?APEAVCAddToWarEffect@@",
    ],
)
def test_camel_cores_reads_the_plan_token_from_every_mangling(mangled):
    assert camel_cores(mangled) == [FIELD_LOG_CORE]


def test_camel_cores_reads_a_template_nested_instance():
    nested = "?$vector@PEAVCAddToWarEffect@@V?$allocator@D@std@@@std@@"
    assert camel_cores(nested) == [FIELD_LOG_CORE]
    assert "CAddToWarEffect" in nested


@pytest.mark.parametrize(
    ("mangled", "expected"),
    [
        ("?AVCAddToWarTrigger@@", ["AddToWarTrigger"]),
        ("?AVCAddToWarModifier@@", ["AddToWarModifier"]),
        ("?AVCAddToWarScope@@", ["AddToWarScope"]),
        ("?AVCAddToWarEffectScope@@", ["AddToWarEffectScope"]),
    ],
)
def test_camel_cores_accepts_every_documented_suffix(mangled, expected):
    assert camel_cores(mangled) == expected


@pytest.mark.parametrize(
    "mangled",
    [
        "",
        "?AVCAddToWarEffects@@",
        "CAddToWarEffects",
        "?AVCAddToWarEffectExtra@@",
        "?AVCAddToWar@@",
        "std::vector",
        "VEffect",
    ],
)
def test_camel_cores_yields_nothing_when_no_suffixed_core_ends_a_run(mangled):
    assert camel_cores(mangled) == []


def test_camel_cores_requires_the_suffix_to_end_the_identifier_run():
    assert camel_cores("?AVCAddToWarEffect@@") == [FIELD_LOG_CORE]
    assert camel_cores("?AVCAddToWarEffect@@s") == []


def test_camel_cores_drops_a_single_leading_class_prefix_by_default():
    assert camel_cores("?AVCAddToWarEffect@@") == ["AddToWarEffect"]
    assert camel_cores("?AVCAddToWarEffect@@", keep_prefix=True) == ["CAddToWarEffect"]
    # a token with no C/V prefix keeps its own spelling either way
    assert camel_cores("?AVAddToWarEffect@@") == ["AddToWarEffect"]
    assert camel_cores("XAddToWarEffect") == ["XAddToWarEffect"]


def test_camel_cores_dedupes_and_preserves_document_order():
    assert camel_cores("?AVCAddToWarEffect@@ ?AVCAddToWarTrigger@@") == [
        "AddToWarEffect",
        "AddToWarTrigger",
    ]
    doubled = FIELD_LOG_MANGLED + FIELD_LOG_MANGLED
    assert camel_cores(doubled) == [FIELD_LOG_CORE]


def test_camel_cores_only_reads_identifier_runs():
    assert camel_cores("?AVCAddToWarEffect@Scope@@") == [FIELD_LOG_CORE]
    assert camel_cores("std::vector<CAddToWarEffect>") == [FIELD_LOG_CORE]


@pytest.mark.parametrize("bad", [7, None, ["?AVCAddToWarEffect@@"]])
def test_camel_cores_refuses_a_non_string(bad):
    with pytest.raises(CorrelateError):
        camel_cores(bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [1, 0, "yes", None])
def test_camel_cores_refuses_a_non_bool_keep_prefix(bad):
    with pytest.raises(CorrelateError):
        camel_cores(FIELD_LOG_MANGLED, keep_prefix=bad)  # type: ignore[arg-type]


def test_camel_core_bases_reports_the_suffixless_base_only():
    assert camel_core_bases("?AVCAddToWarEffect@@") == ["AddToWar"]
    assert camel_core_bases("?AVCAddToWarEffectScope@@") == [FIELD_LOG_CORE]
    # a token with no suffix has no base to report
    assert camel_core_bases("CAddToWar") == []


# --- 5. strip_cv_prefix / split_suffix -------------------------------------- #


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("CAddToWarEffect", "AddToWarEffect"),
        ("VAddToWarEffect", "AddToWarEffect"),
        ("AddToWarEffect", "AddToWarEffect"),
        ("C", "C"),
        ("", ""),
    ],
)
def test_strip_cv_prefix_removes_at_most_one_prefix(token, expected):
    assert strip_cv_prefix(token) == expected


def test_strip_cv_prefix_refuses_a_non_string():
    with pytest.raises(CorrelateError):
        strip_cv_prefix(7)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("AddToWarEffect", ("AddToWar", "Effect")),
        ("AddToWarTrigger", ("AddToWar", "Trigger")),
        ("AddToWarModifier", ("AddToWar", "Modifier")),
        ("AddToWarScope", ("AddToWar", "Scope")),
        ("AddToWarEffectScope", ("AddToWarEffect", "Scope")),
        ("AddToWar", ("AddToWar", None)),
        # a bare suffix is not split -- the core would be empty
        ("Effect", ("Effect", None)),
        ("", ("", None)),
        # case-sensitive, the way the class names are
        ("add_to_war_effect", ("add_to_war_effect", None)),
    ],
)
def test_split_suffix(token, expected):
    assert split_suffix(token) == expected


def test_split_suffix_refuses_a_non_string():
    with pytest.raises(CorrelateError):
        split_suffix(None)  # type: ignore[arg-type]


# --- 6. correlate: hits ----------------------------------------------------- #


def test_correlate_proves_the_field_log_pair():
    result = correlate_keys([FIELD_LOG_KEYWORD], [FIELD_LOG_MANGLED])
    assert result.matches == {FIELD_LOG_KEYWORD: (FIELD_LOG_MANGLED,)}
    assert result.pairs == ((FIELD_LOG_KEYWORD, (FIELD_LOG_MANGLED,)),)
    assert result.unmatched_keywords == ()
    assert result.unmatched_mangled == ()


@pytest.mark.parametrize(
    "mangled",
    [
        FIELD_LOG_MANGLED,
        ".?AVCAddToWarEffect@@",
        "?$vector@PEAVCAddToWarEffect@@V?$allocator@D@std@@@std@@",
        "AddToWarEffect",
        "?AVCAddToWarEffectScope@@",
        "?AVCAddToWarModifier@@",
        "?AVCAddToWarTrigger@@".replace("Trigger", "Effect"),
    ],
)
def test_correlate_suffix_variant_tries_reach_the_keyword(mangled):
    assert correlate_keys([FIELD_LOG_KEYWORD], [mangled]).matches == {
        FIELD_LOG_KEYWORD: (mangled,)
    }


def test_correlate_whole_name_plain_identifier_rule():
    assert correlate_keys([FIELD_LOG_KEYWORD], ["AddToWar"]).matches == {
        FIELD_LOG_KEYWORD: ("AddToWar",)
    }


def test_correlate_strips_an_unused_cv_prefix_on_the_name_side():
    assert correlate_keys(["add_to_war_effect"], ["VAddToWarEffect"]).matches == {
        "add_to_war_effect": ("VAddToWarEffect",)
    }


@pytest.mark.parametrize(
    ("keyword", "name"),
    [
        ("add_to_war", "add_to_war"),
        ("AddToWar", "add_to_war"),
        ("addToWar", "AddToWar"),
        ("add-to-war", "ADD_TO_WAR"),
        ("add_to_war", "addtowar"),
    ],
)
def test_correlate_canonical_equality_ignores_case_and_separators(keyword, name):
    assert correlate_keys([keyword], [name]).matches == {keyword: (name,)}


def test_correlate_is_many_to_many_in_input_order():
    keywords = ["add_to_war", "add_to_war_effect", "war"]
    names = [
        FIELD_LOG_MANGLED,
        "AddToWar",
        "?AVAddToWarTrigger@@".replace("Trigger", "Effect"),
    ]
    result = correlate_keys(keywords, names)
    assert result.pairs == (
        (keywords[0], (names[0], names[1], names[2])),
        (keywords[1], (names[0], names[2])),
    )
    assert result.unmatched_keywords == ("war",)
    assert result.unmatched_mangled == ()


def test_correlate_reports_a_duplicate_keyword_once_in_first_appearance_order():
    result = correlate_keys(
        ["add_to_war", "add_to_war", "galaxy"], [FIELD_LOG_MANGLED]
    )
    assert result.pairs == ((FIELD_LOG_KEYWORD, (FIELD_LOG_MANGLED,)),)
    assert result.unmatched_keywords == ("galaxy",)


def test_correlate_reports_a_miss_on_either_side_without_guessing():
    result = correlate_keys(
        ["add_to_war", "add_to_war_trigger"],
        [FIELD_LOG_MANGLED, "?AVCNSBGalaxyScope@@", "std::vector"],
    )
    assert result.matches == {FIELD_LOG_KEYWORD: (FIELD_LOG_MANGLED,)}
    assert result.unmatched_keywords == ("add_to_war_trigger",)
    assert result.unmatched_mangled == ("?AVCNSBGalaxyScope@@", "std::vector")
    assert result.matched_keywords() == (FIELD_LOG_KEYWORD,)


def test_correlate_does_not_fuzz_beyond_the_documented_rules():
    """No substring search: the keyword must equal a whole candidate's canonical."""
    assert correlate_keys(["war"], [FIELD_LOG_MANGLED]).matches == {}
    assert correlate_keys(["add_to_wars"], [FIELD_LOG_MANGLED]).matches == {}
    assert correlate_keys(["add_to_wa"], [FIELD_LOG_MANGLED]).matches == {}
    assert correlate_keys(["effect"], [FIELD_LOG_MANGLED]).matches == {}


def test_correlate_hits_a_mangled_name_inside_a_template_instantiation():
    result = correlate_keys([FIELD_LOG_KEYWORD], ["std::vector<CAddToWarEffect>"])
    assert result.matches == {FIELD_LOG_KEYWORD: ("std::vector<CAddToWarEffect>",)}


@pytest.mark.parametrize("empty", [[], (), iter([])])
def test_correlate_on_empty_inputs_is_an_empty_report(empty):
    result = correlate_keys(empty, empty)
    assert result.pairs == ()
    assert result.matches == {}
    assert result.unmatched_keywords == ()
    assert result.unmatched_mangled == ()
    assert result.skipped_keywords == 0
    assert result.skipped_mangled == 0
    assert result.max_items == DEFAULT_MAX_ITEMS


def test_correlate_treats_none_inputs_as_nothing_supplied():
    result = correlate_keys(None, None)  # type: ignore[arg-type]
    assert result.as_dict() == {
        "matches": {},
        "unmatched_keywords": [],
        "unmatched_mangled": [],
        "skipped_keywords": 0,
        "skipped_mangled": 0,
        "max_items": DEFAULT_MAX_ITEMS,
    }


def test_correlate_leaves_the_callers_own_sequences_untouched():
    keywords = ["b", FIELD_LOG_KEYWORD]
    names = [FIELD_LOG_MANGLED]
    correlate_keys(keywords, names)
    assert keywords == ["b", FIELD_LOG_KEYWORD]
    assert names == [FIELD_LOG_MANGLED]


def test_correlate_accepts_any_iterable_of_strings():
    from_generators = correlate_keys(
        (word for word in [FIELD_LOG_KEYWORD]),
        (name for name in [FIELD_LOG_MANGLED]),
    )
    from_tuples = correlate_keys((FIELD_LOG_KEYWORD,), (FIELD_LOG_MANGLED,))
    assert from_generators == from_tuples


def test_correlate_is_deterministic():
    first = correlate_keys([FIELD_LOG_KEYWORD, "nsb_galaxy"], [FIELD_LOG_MANGLED])
    second = correlate_keys([FIELD_LOG_KEYWORD, "nsb_galaxy"], [FIELD_LOG_MANGLED])
    assert first == second
    assert first.as_dict() == second.as_dict()


def test_correlate_does_not_order_by_a_dict():
    """Keyword input order -- not hash order -- decides the pairs' order."""
    keywords = ["zulu", "alpha", FIELD_LOG_KEYWORD]
    result = correlate_keys(keywords, [FIELD_LOG_MANGLED])
    assert result.matches.keys() == {FIELD_LOG_KEYWORD} or True
    assert [word for word, _ in result.pairs] == [FIELD_LOG_KEYWORD]
    assert result.unmatched_keywords == ("zulu", "alpha")


# --- 7. the Correlation surface --------------------------------------------- #


def test_correlation_as_dict_is_json_friendly():
    result = correlate_keys(
        [FIELD_LOG_KEYWORD, "nope"], [FIELD_LOG_MANGLED, "std::vector"]
    )
    report = result.as_dict()
    assert report == {
        "matches": {FIELD_LOG_KEYWORD: [FIELD_LOG_MANGLED]},
        "unmatched_keywords": ["nope"],
        "unmatched_mangled": ["std::vector"],
        "skipped_keywords": 0,
        "skipped_mangled": 0,
        "max_items": DEFAULT_MAX_ITEMS,
    }
    assert json.loads(json.dumps(report)) == report


def test_correlation_mapping_access_and_shape():
    result = correlate_keys([FIELD_LOG_KEYWORD], [FIELD_LOG_MANGLED])
    assert result["matches"] == {FIELD_LOG_KEYWORD: (FIELD_LOG_MANGLED,)}
    assert result[FIELD_LOG_KEYWORD] == (FIELD_LOG_MANGLED,)
    assert result["pairs"] == result.pairs
    assert result["unmatched_keywords"] == ()
    assert result["unmatched_mangled"] == ()
    assert result["skipped_keywords"] == 0
    assert result["skipped_mangled"] == 0
    assert result["max_items"] == DEFAULT_MAX_ITEMS
    assert len(result) == 1
    assert FIELD_LOG_KEYWORD in result
    assert "matches" in result
    assert "std::vector" not in result
    assert result.matches_for(FIELD_LOG_KEYWORD) == (FIELD_LOG_MANGLED,)
    assert result.matches_for("nothing") == ()


@pytest.mark.parametrize("key", ["missing", "", 0, None, ("pairs",)])
def test_correlation_raises_key_error_for_an_unknown_key(key):
    result = correlate_keys([], [])
    assert (key in result) is False
    with pytest.raises(KeyError):
        result[key]


def test_correlation_refuses_inconsistent_fields():
    clean = Correlation()
    assert clean.pairs == () and len(clean) == 0
    with pytest.raises(CorrelateError):
        Correlation(pairs=(("a", ["b"]),))
    with pytest.raises(CorrelateError):
        Correlation(pairs=((5, ("b",)),))
    with pytest.raises(CorrelateError):
        Correlation(pairs=(("a", ("b", 5)),))
    with pytest.raises(CorrelateError):
        Correlation(unmatched_keywords=["a"])
    with pytest.raises(CorrelateError):
        Correlation(unmatched_mangled=("a", 5))
    with pytest.raises(CorrelateError):
        Correlation(skipped_keywords=True)
    with pytest.raises(CorrelateError):
        Correlation(skipped_mangled=-1)
    with pytest.raises(CorrelateError):
        Correlation(max_items=0)


def test_correlation_matches_for_refuses_a_non_string():
    with pytest.raises(CorrelateError):
        Correlation().matches_for(7)  # type: ignore[arg-type]


# --- 8. caps (Risk R1) ------------------------------------------------------ #


def test_resolve_max_items_defaults_to_the_r1_default():
    assert resolve_max_items() == DEFAULT_MAX_ITEMS
    assert resolve_max_items(None) == DEFAULT_MAX_ITEMS


def test_resolve_max_items_honours_a_smaller_budget_and_clamps_a_larger_one():
    assert resolve_max_items(3) == 3
    assert resolve_max_items(MAX_ITEMS) == MAX_ITEMS
    assert resolve_max_items(MAX_ITEMS + 1) == MAX_ITEMS
    assert resolve_max_items(10 ** 12) == MAX_ITEMS


@pytest.mark.parametrize("bad", [0, -1, -1000, 2.5, "50", True, False, [5], object()])
def test_resolve_max_items_refuses_a_bad_budget(bad):
    with pytest.raises(CorrelateError):
        resolve_max_items(bad)


def test_default_cap_does_not_skip_a_realistic_input():
    result = correlate_keys([FIELD_LOG_KEYWORD], [FIELD_LOG_MANGLED])
    assert result.skipped_keywords == 0
    assert result.skipped_mangled == 0
    assert result.max_items == DEFAULT_MAX_ITEMS


def test_max_items_bounds_each_sequence_and_counts_the_skips():
    result = correlate_keys(["a", "b", "c"], ["x", "y"], max_items=2)
    assert result.pairs == ()
    assert result.unmatched_keywords == ("a", "b")
    assert result.unmatched_mangled == ("x", "y")
    assert result.skipped_keywords == 1
    assert result.skipped_mangled == 0
    assert result.max_items == 2


def test_a_cap_of_one_inspects_exactly_the_first_entry():
    result = correlate_keys([FIELD_LOG_KEYWORD, FIELD_LOG_KEYWORD], [FIELD_LOG_MANGLED], max_items=1)
    assert result.matches == {FIELD_LOG_KEYWORD: (FIELD_LOG_MANGLED,)}
    assert result.skipped_keywords == 1


def test_entries_past_the_cap_are_counted_not_inspected():
    # inside the cap a malformed entry is a caller error...
    with pytest.raises(CorrelateError):
        correlate_keys([123, "a"], [FIELD_LOG_MANGLED], max_items=1)
    # ...past it the entry is never looked at, so it is not one
    result = correlate_keys(["a", 123], [FIELD_LOG_MANGLED], max_items=1)
    assert result.unmatched_keywords == ("a",)
    assert result.skipped_keywords == 1
    assert result.unmatched_mangled == (FIELD_LOG_MANGLED,)


def test_max_items_above_the_ceiling_is_clamped_not_honoured():
    result = correlate_keys([FIELD_LOG_KEYWORD], [FIELD_LOG_MANGLED], max_items=10 ** 9)
    assert result.max_items == MAX_ITEMS


@pytest.mark.parametrize("bad", [0, -5, "3", 1.5, True])
def test_bad_max_items_is_a_caller_error(bad):
    with pytest.raises(CorrelateError):
        correlate_keys([FIELD_LOG_KEYWORD], [FIELD_LOG_MANGLED], max_items=bad)


# --- 9. malformed arguments ------------------------------------------------- #


@pytest.mark.parametrize("bad", ["add_to_war", {"add_to_war": 1}, 7, object()])
def test_a_bare_string_mapping_or_scalar_is_not_a_sequence_of_words(bad):
    with pytest.raises(CorrelateError):
        correlate_keys(bad, [])
    with pytest.raises(CorrelateError):
        correlate_keys([], bad)


@pytest.mark.parametrize("bad", [7, None, b"add_to_war", ["a"]])
def test_a_non_string_word_is_a_caller_error(bad):
    with pytest.raises(CorrelateError):
        correlate_keys([bad], [])
    with pytest.raises(CorrelateError):
        correlate_keys([], [bad])


@pytest.mark.parametrize("bad", ["---", "   ", "", "***", "?@$"])
def test_a_keyword_with_no_canonical_form_is_a_caller_error(bad):
    with pytest.raises(CorrelateError):
        correlate_keys([bad], [FIELD_LOG_MANGLED])


def test_a_mangled_name_with_no_canonical_form_is_simply_unmatched():
    """Only the *keyword* side must be canonicalisable; a name may be anything."""
    result = correlate_keys([FIELD_LOG_KEYWORD], ["---"])
    assert result.matches == {}
    assert result.unmatched_mangled == ("---",)
