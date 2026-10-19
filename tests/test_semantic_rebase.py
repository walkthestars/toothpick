"""W20 semantic rebase tests: embeddings, cosine tiers, persistence, CLI (chunks).

Uses the same synthetic-DB patterns as ``tests/test_rebase.py``; no target
binaries required.  ``sub_*`` names are synthetic on purpose so matching
falls through to the structural/semantic tiers.
"""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from toothpick.cli import cli
from toothpick.rebase import (
    MATCH_TIER_SEMANTIC_EMBEDDING,
    FunctionEmbedding,
    RebaseEngine,
    VectorSimilarityEngine,
)
from toothpick.storage import Database, EdgeKind, FunctionRecord, SsaFact

OLD_SHA = "a" * 64
NEW_SHA = "b" * 64


def _func(sha: str, addr: int, name: str, size: int = 64) -> FunctionRecord:
    return FunctionRecord(
        binary_sha256=sha,
        func_addr=addr,
        name=name,
        size=size,
        calling_convention="ms_x64",
        prototype=f"int {name}(int a)",
    )


def _fact(sha: str, addr: int, var: str, idx: int) -> SsaFact:
    return SsaFact(
        fact_id=f"{sha}:{addr:x}:{idx}:{var}",
        binary_sha256=sha,
        func_addr=addr,
        var_name=var,
        ssa_reg=f"{var}#{idx}",
        def_point=idx,
        use_points=[idx + 1],
    )


def _populate_pair(db: Database, extra_new_edge: bool = True) -> None:
    """Old/new function pair with near-identical shape at different addresses."""
    old, new = 0x14001000, 0x14006000
    db.upsert_function(_func(OLD_SHA, old, "sub_1000"), write_receipt=False)
    db.upsert_function(_func(NEW_SHA, new, "sub_2000"), write_receipt=False)
    for base, sha in ((old, OLD_SHA), (new, NEW_SHA)):
        db.add_edge(sha, base, base + 0x20, EdgeKind.CFG)
        db.add_edge(sha, base + 0x20, base + 0x40, EdgeKind.CFG)
        db.add_ssa_fact(_fact(sha, base, "rax", 0))
        db.add_ssa_fact(_fact(sha, base, "rcx", 1))
    if extra_new_edge:
        db.add_edge(NEW_SHA, new + 0x40, new + 0x60, EdgeKind.CFG)
        db.add_ssa_fact(_fact(NEW_SHA, new, "rdx", 2))


def test_embedding_serialization_round_trip() -> None:
    """to_bytes/from_bytes preserve the 64-float unit vector (256 bytes)."""
    engine = VectorSimilarityEngine()
    original = FunctionEmbedding(
        func_addr=0x1000, vector=engine.normalize([float(i) for i in range(64)])
    )
    blob = original.to_bytes()
    assert len(blob) == 256
    restored = FunctionEmbedding.from_bytes(blob, func_addr=0x1000)
    assert restored.func_addr == 0x1000
    assert len(restored.vector) == 64
    assert VectorSimilarityEngine.cosine_similarity(original.vector, restored.vector) > 0.999


def test_cosine_similarity_identical_vs_orthogonal() -> None:
    """Identical vectors score 1.0, orthogonal score 0.0, zero-norm is safe."""
    unit_x = [1.0] + [0.0] * 63
    unit_y = [0.0, 1.0] + [0.0] * 62
    assert VectorSimilarityEngine.cosine_similarity(unit_x, unit_x) == 1.0
    assert VectorSimilarityEngine.cosine_similarity(unit_x, unit_y) == 0.0
    assert VectorSimilarityEngine.cosine_similarity([0.0] * 64, unit_x) == 0.0


def test_relocated_function_matches_via_semantic_tier(tmp_path: Path) -> None:
    """A near-identical relocated pair lands on SEMANTIC_EMBEDDING, not exact."""
    db = Database(tmp_path / "sem.db", create=True)
    try:
        _populate_pair(db)
        with RebaseEngine(db=db) as engine:
            report = engine.rebase(old_sha256=OLD_SHA, new_target=NEW_SHA)
            tiers = {m.match_tier for m in report.matches}
            assert MATCH_TIER_SEMANTIC_EMBEDDING in tiers, (
                f"expected a SEMANTIC_EMBEDDING match, got {tiers}"
            )
            semantic = [
                m for m in report.matches
                if m.match_tier == MATCH_TIER_SEMANTIC_EMBEDDING
            ]
            assert semantic and all(m.similarity >= 0.85 for m in semantic)
    finally:
        db.close()


def test_embedding_blob_persisted_in_versions(tmp_path: Path) -> None:
    """The versions row carries a 256-byte embedding blob after rebase."""
    db = Database(tmp_path / "sem.db", create=True)
    try:
        _populate_pair(db)
        with RebaseEngine(db=db) as engine:
            engine.rebase(old_sha256=OLD_SHA, new_target=NEW_SHA)
        row = db._conn.execute(
            "SELECT embedding_blob FROM versions WHERE binary_sha256 = ? AND target_sha256 = ?",
            (OLD_SHA, NEW_SHA),
        ).fetchone()
        assert row is not None and row["embedding_blob"] is not None
        assert len(bytes(row["embedding_blob"])) == 256
    finally:
        db.close()


def test_rebase_semantic_cli_json(tmp_path: Path) -> None:
    """``rebase --semantic --dump-embeddings`` emits vectors in the report."""
    db_path = tmp_path / "sem.db"
    db = Database(db_path, create=True)
    try:
        _populate_pair(db)
    finally:
        db.close()
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "rebase", NEW_SHA,
            "--old-sha", OLD_SHA,
            "--db", str(db_path),
            "--semantic",
            "--dump-embeddings",
        ],
    )
    assert result.exit_code == 0, f"rebase --semantic failed: {result.output}"
    payload = json.loads(result.output)
    assert payload["semantic"] is True
    assert payload["embeddings"], "expected dumped embedding blobs"
    assert all(len(blob) == 512 for blob in payload["embeddings"].values())
