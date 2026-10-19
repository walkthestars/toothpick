"""Rebase priors (CONS-01 acceptance 4): Diaphora candidates as scorer priors.

Failing-first TDD for the priors slice of src/toothpick/rebase.py plus the
``rebase --priors`` CLI flag. Contract: external candidate evidence (icepick
``diff-import`` documents: ``{candidates: [{old_ea, new_ea, score, old_name,
new_name, ...}]}``) may only ever *uplift* a pair toward the candidate score
-- the established one-directional pattern of the embedding uplift. It can
promote a name-confirmed pair to Tier-1 exact and rescue a renamed pair from
the NAME_MISMATCH refusal, but it can never depress a structural score, never
invent a match from nothing, and never move the threshold gates.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from toothpick.rebase import (
    MATCH_FLOOR,
    MATCH_TIER_EXACT_NAME,
    FunctionPriors,
    RebaseEngine,
    RebaseError,
    load_priors,
)
from toothpick.storage import Database, EdgeRecord, FunctionRecord

rebase_mod = pytest.importorskip("toothpick.rebase")

OLD_SHA = "c" * 64
NEW_SHA = "d" * 64


def _func(sha: str, addr: int, name: str, size: int = 100) -> FunctionRecord:
    return FunctionRecord(
        binary_sha256=sha,
        func_addr=addr,
        name=name,
        size=size,
        calling_convention="",
        prototype="",
        ir_pcode=None,
        decompiled_c=None,
    )


def _engine(db: Database) -> RebaseEngine:
    return RebaseEngine(db)


def _priors(entries) -> FunctionPriors:
    return load_priors({"candidates": list(entries)})


def _cand(old_ea, new_ea, score, old_name="", new_name="") -> dict:
    return {
        "old_ea": old_ea,
        "new_ea": new_ea,
        "score": score,
        "old_name": old_name,
        "new_name": new_name,
        "signals": {"name_equal": old_name == new_name},
    }


# -- Tier-1 boost ---------------------------------------------------------------


def test_name_equal_prior_promotes_to_exact_name(tmp_path: Path) -> None:
    db = Database(tmp_path / "tp.db", create=True)
    try:
        with _engine(db) as engine:
            old = _func(OLD_SHA, 0x1000, "Engine_Init", size=100)
            new = _func(NEW_SHA, 0x2000, "Engine_Init", size=40)
            plain = engine.compare_functions(old, new, [], [], {}, {})
            assert plain[1] != MATCH_TIER_EXACT_NAME  # size gate refuses

            priors = _priors([_cand(0x1000, 0x2000, 1.0, "Engine_Init", "Engine_Init")])
            boosted = engine.compare_functions(old, new, [], [], {}, {}, priors=priors)
            assert boosted[0] == 1.0
            assert boosted[1] == MATCH_TIER_EXACT_NAME
            assert boosted[2]["prior_score"] == 1.0
    finally:
        db.close()


# -- rename rescue -----------------------------------------------------------------


def test_strong_prior_rescues_renamed_pair(tmp_path: Path) -> None:
    db = Database(tmp_path / "tp.db", create=True)
    try:
        with _engine(db) as engine:
            old = _func(OLD_SHA, 0x1000, "RenderFrame", size=100)
            new = _func(NEW_SHA, 0x2000, "DrawFrame", size=60)
            refused = engine.compare_functions(old, new, [], [], {}, {})
            assert refused[0] == 0.0  # NAME_MISMATCH refusal stands alone

            priors = _priors([_cand(0x1000, 0x2000, 0.99, "RenderFrame", "DrawFrame")])
            rescued = engine.compare_functions(old, new, [], [], {}, {}, priors=priors)
            assert rescued[0] > refused[0]
            assert rescued[0] >= MATCH_FLOOR
            assert rescued[1] != "NAME_MISMATCH"
            assert rescued[2]["prior_score"] == 0.99
            assert rescued[2].get("prior_overrode_name_mismatch") is True
    finally:
        db.close()


def test_name_keyed_prior_matches_without_addr_overlap(tmp_path: Path) -> None:
    db = Database(tmp_path / "tp.db", create=True)
    try:
        with _engine(db) as engine:
            old = _func(OLD_SHA, 0x1000, "RenderFrame", size=100)
            new = _func(NEW_SHA, 0x9000, "DrawFrame", size=100)
            # Candidate addresses are image-based and need not equal DB addrs;
            # the name pair still rescues the match.
            priors = _priors([_cand(0x5000, 0x6000, 0.95, "RenderFrame", "DrawFrame")])
            scored = engine.compare_functions(old, new, [], [], {}, {}, priors=priors)
            assert scored[0] >= MATCH_FLOOR
    finally:
        db.close()


# -- non-override ----------------------------------------------------------------------


def test_weak_prior_never_depresses_or_invents(tmp_path: Path) -> None:
    db = Database(tmp_path / "tp.db", create=True)
    try:
        with _engine(db) as engine:
            old = _func(OLD_SHA, 0x1000, "Alpha", size=100)
            new = _func(NEW_SHA, 0x2000, "Alpha", size=100)
            plain = engine.compare_functions(old, new, [], [], {}, {})
            weak = _priors([_cand(0x1000, 0x2000, 0.1, "Alpha", "Alpha")])
            held = engine.compare_functions(old, new, [], [], {}, {}, priors=weak)
            assert held[0] == plain[0]  # low prior cannot sink a strong pair

            # Structurally divergent pair: different sizes, different call
            # shapes, disjoint register use. The weak prior must change nothing.
            junk_old = _func(OLD_SHA, 0x3000, "Zeta", size=10)
            junk_new = _func(NEW_SHA, 0x4000, "Omega", size=9000)
            old_edges = [
                EdgeRecord(OLD_SHA, 0x3000, 0x3001, "call"),
                EdgeRecord(OLD_SHA, 0x3000, 0x3002, "call"),
            ]
            thin = _priors([_cand(0x3000, 0x4000, 0.2, "Zeta", "Omega")])
            bare = engine.compare_functions(
                junk_old, junk_new, old_edges, [], {"rax": 3}, {"rcx": 5}
            )
            junk = engine.compare_functions(
                junk_old, junk_new, old_edges, [], {"rax": 3}, {"rcx": 5},
                priors=thin,
            )
            # The fall-through replaces the 0.0 refusal with the genuine
            # structural score -- but it stays below the floor, the status
            # gate does not move, and the evidence is on the record.
            assert bare[0] == 0.0
            assert junk[0] < MATCH_FLOOR  # weak prior cannot invent a match
            assert junk[2]["prior_score"] == 0.2
    finally:
        db.close()


# -- loader ------------------------------------------------------------------------------


def test_load_priors_keys_and_best_wins() -> None:
    priors = _priors(
        [
            _cand(0x1000, 0x2000, 0.70, "A", "B"),
            _cand(0x1000, 0x2000, 0.95, "A", "B"),  # duplicate: best wins
        ]
    )
    assert priors.lookup(0x1000, 0x2000, "A", "B") == 0.95
    assert priors.lookup(0x9999, 0x8888, "A", "B") == 0.95  # name fallback
    assert priors.lookup(0x1000, 0x2000, "X", "Y") == 0.95  # addr precedence
    assert priors.lookup(0x1111, 0x2222, "Q", "W") is None


def test_load_priors_refuses_malformed() -> None:
    with pytest.raises(RebaseError):
        load_priors({"no_candidates": []})
    with pytest.raises(RebaseError):
        load_priors({"candidates": [{"old_ea": 1, "new_ea": 2, "score": "high"}]})
    with pytest.raises(RebaseError):
        load_priors({"candidates": [{"old_ea": "0x10", "new_ea": 2, "score": 0.5}]})
    with pytest.raises(RebaseError):
        load_priors({"candidates": [{"old_ea": 1, "new_ea": 2, "score": 1.5}]})
    with pytest.raises(RebaseError):
        load_priors(["not", "a", "doc"])


# -- engine level ----------------------------------------------------------------------------


def test_rebase_with_priors_moves_statuses(tmp_path: Path) -> None:
    db = Database(tmp_path / "tp.db", create=True)
    try:
        db.upsert_function(_func(OLD_SHA, 0x1000, "RenderFrame", 100), write_receipt=False)
        db.upsert_function(_func(NEW_SHA, 0x2000, "DrawFrame", 100), write_receipt=False)
        with _engine(db) as engine:
            bare = engine.rebase(OLD_SHA, NEW_SHA)
            bare_addrs = set(bare.unmatched_old)
            assert 0x1000 in bare_addrs  # renamed pair unmatched alone

            priors = _priors([_cand(0x1000, 0x2000, 0.95, "RenderFrame", "DrawFrame")])
            guided = engine.rebase(OLD_SHA, NEW_SHA, priors=priors)
            assert 0x1000 not in set(guided.unmatched_old)
            match = next(m for m in guided.matches if m.old_addr == 0x1000)
            assert match.new_addr == 0x2000
            assert match.similarity >= MATCH_FLOOR
    finally:
        db.close()


# -- CLI -----------------------------------------------------------------------------------


def test_cli_priors_happy_path(tmp_path: Path) -> None:
    from click.testing import CliRunner

    from toothpick.cli import cli

    db_path = tmp_path / "tp.db"
    db = Database(db_path, create=True)
    try:
        db.upsert_function(_func(OLD_SHA, 0x1000, "RenderFrame", 100), write_receipt=False)
        db.upsert_function(_func(NEW_SHA, 0x2000, "DrawFrame", 100), write_receipt=False)
    finally:
        db.close()
    priors_path = tmp_path / "priors.json"
    priors_path.write_text(
        json.dumps({"candidates": [_cand(0x1000, 0x2000, 0.95, "RenderFrame", "DrawFrame")]}),
        encoding="utf-8",
    )
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["rebase", NEW_SHA, "--old-sha", OLD_SHA, "--db", str(db_path),
         "--priors", str(priors_path)],
    )
    assert result.exit_code == 0, result.output
    assert "REVIEW_REQUIRED" in result.output or "AUTO_PORTED" in result.output


def test_cli_priors_flag_refuses_bad_file(tmp_path: Path) -> None:
    from click.testing import CliRunner

    from toothpick.cli import cli

    runner = CliRunner()
    missing = tmp_path / "nope.json"
    result = runner.invoke(cli, ["rebase", "whatever", "--priors", str(missing)])
    # click's exists=True rejects the missing path as a usage error (exit 2),
    # same as every other file argument in this CLI.
    assert result.exit_code == 2

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"no_candidates": []}), encoding="utf-8")
    result = runner.invoke(cli, ["rebase", "whatever", "--priors", str(bad)])
    assert result.exit_code == 1
