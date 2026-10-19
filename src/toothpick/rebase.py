"""
Toothpick Version Rebase Engine (unit W09).

Implements the W09 version rebase contract per UNIT-W09-REBASE-PLAN.md,
design-document.md §10, §11, §12, §16 #2, §19 (TBM-3):
- Multi-heuristic differential binary analysis (Diaphora / BinDiff pattern).
- Function matching across binary versions using exact symbol names, normalized
  bytecode/opcode profiles, CFG graph topology, instruction histograms, and call graphs.
- Epistemic threshold classification:
  * similarity >= 0.99: AUTO_PORTED (safe for autonomous mod porting)
  * 0.50 <= similarity < 0.99: REVIEW_REQUIRED (queued for human/agent review per RS-2)
  * similarity < 0.50: UNMATCHED
- Mod artifact rebase helper: rebase ModConfig offsets to new binary RVAs.
- Persistence to toothpick.db `versions` table and `receipts` table.

UNIT-W20 (Semantic Embedding Rebase Engine): extends the W09 engine with
64-dimensional unit-vector function embeddings (opcode histogram D00-D15, CFG
topology D16-D31, SSA def-use density D32-D47, Weisfeiler-Lehman graph
neighbourhood D48-D63), exact vector cosine similarity, a four-tier match
taxonomy (EXACT_NAME / EXACT_STRUCTURE / SEMANTIC_EMBEDDING /
GRAPH_SIMILARITY), and 256-byte IEEE-754 float32 vector persistence in
`versions.embedding_blob` (design §9, execution-plan.md W20, TBM-3).
"""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

from toothpick.mod_builder import ModConfig
from toothpick.static_engine import StaticEngine
from toothpick.storage import Database, EdgeRecord, FunctionRecord, VersionRecord

__all__ = [
    "RebaseError",
    "RebaseMatch",
    "RebaseReport",
    "RebaseEngine",
    "FunctionEmbedding",
    "VectorSimilarityEngine",
    "MATCH_TIERS",
    "MATCH_TIER_SEMANTIC_EMBEDDING",
    "SEMANTIC_EMBEDDING_THRESHOLD",
    "EMBEDDING_DIM",
    "EMBEDDING_BYTES",
    "DEFAULT_AUTO_PORT_THRESHOLD",
]

PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# Thresholds and match-tier enum (design-document.md §10, §16 #2, §19 TBM-3)
# ---------------------------------------------------------------------------

#: Default auto-port gate: similarity >= 0.99 ports without human review (RS-2).
DEFAULT_AUTO_PORT_THRESHOLD = 0.99

#: Matching floor: pairs scoring below this are never paired -> UNMATCHED.
MATCH_FLOOR = 0.50

#: Documented ``RebaseMatch.match_tier`` enum -- every emitted tier is a member.
MATCH_TIER_EXACT_NAME = "EXACT_NAME"
MATCH_TIER_EXACT_STRUCTURE = "EXACT_STRUCTURE"
MATCH_TIER_GRAPH_SIMILARITY = "GRAPH_SIMILARITY"
MATCH_TIER_NAME_MISMATCH = "NAME_MISMATCH"

#: W20 Tier 2 tag: semantic embedding match (design §9 four-tier taxonomy).
MATCH_TIER_SEMANTIC_EMBEDDING = "SEMANTIC_EMBEDDING"

MATCH_TIERS: Tuple[str, ...] = (
    MATCH_TIER_EXACT_NAME,
    MATCH_TIER_EXACT_STRUCTURE,
    MATCH_TIER_SEMANTIC_EMBEDDING,
    MATCH_TIER_GRAPH_SIMILARITY,
    MATCH_TIER_NAME_MISMATCH,
)

#: Provably-identical tiers: eligible for the Pass-1 exact-match sweep.
EXACT_MATCH_TIERS = (MATCH_TIER_EXACT_NAME, MATCH_TIER_EXACT_STRUCTURE)

#: W20 Tier 2 gate: embedding cosine >= 0.85 promotes a pair to SEMANTIC_EMBEDDING.
SEMANTIC_EMBEDDING_THRESHOLD = 0.85

#: Documented Tier 3 graph-similarity gate (design §10); the pairing floor
#: itself remains ``MATCH_FLOOR`` so W09 0.50-0.99 REVIEW_REQUIRED is preserved.
GRAPH_SIMILARITY_THRESHOLD = 0.70

#: Embedding width and serialized size (64 x IEEE-754 float32 = 256 bytes).
EMBEDDING_DIM = 64
EMBEDDING_BYTES = EMBEDDING_DIM * 4

#: Composite weight of the embedding cosine (dropped when no evidence exists).
EMBEDDING_SCORE_WEIGHT = 0.20

#: Weight of external candidate evidence (Diaphora ``to_rebase_candidates``
#: pairs) in the Tier-2 composite. A candidate score above the structural
#: composite uplifts it one-directionally toward the candidate score -- the
#: same bounded pattern as the embedding uplift. A prior can promote and can
#: rescue, but it can never depress a structural score and never moves the
#: threshold gates (CONS-01 acceptance 4).
PRIOR_SCORE_WEIGHT = 0.20


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class RebaseError(Exception):
    """Base exception for rebase failures."""


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class RebaseMatch:
    """Mapping of a single function between old and new binary versions."""

    old_addr: int
    new_addr: int
    old_name: str
    new_name: str
    similarity: float
    status: str  # "AUTO_PORTED" | "REVIEW_REQUIRED" | "UNMATCHED"
    match_tier: str  # one of MATCH_TIERS
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.match_tier not in MATCH_TIERS:
            raise ValueError(f"unknown match_tier {self.match_tier!r}; expected one of {MATCH_TIERS}")


    def to_dict(self) -> Dict[str, Any]:
        return {
            "old_addr": self.old_addr,
            "old_addr_hex": f"0x{self.old_addr:x}",
            "new_addr": self.new_addr,
            "new_addr_hex": f"0x{self.new_addr:x}",
            "old_name": self.old_name,
            "new_name": self.new_name,
            "similarity": round(self.similarity, 4),
            "status": self.status,
            "match_tier": self.match_tier,
            "details": self.details,
        }


@dataclass
class RebaseReport:
    """Full report of a version rebase operation."""

    old_sha256: str
    new_sha256: str
    total_old_funcs: int
    total_new_funcs: int
    auto_ported: int
    review_required: int
    unmatched: int
    auto_port_rate: float
    matches: List[RebaseMatch] = field(default_factory=list)
    offset_map: Dict[str, int] = field(default_factory=dict)
    unmatched_old: List[int] = field(default_factory=list)
    unmatched_new: List[int] = field(default_factory=list)
    threshold: float = DEFAULT_AUTO_PORT_THRESHOLD

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": "success",
            "old_sha256": self.old_sha256,
            "new_sha256": self.new_sha256,
            "total_old_funcs": self.total_old_funcs,
            "total_new_funcs": self.total_new_funcs,
            "auto_ported": self.auto_ported,
            "review_required": self.review_required,
            "unmatched": self.unmatched,
            "auto_port_rate": round(self.auto_port_rate, 4),
            "auto_port_threshold": round(self.threshold, 4),
            "offset_map": {k: f"0x{v:x}" if isinstance(v, int) else v for k, v in self.offset_map.items()},
            "matches": [m.to_dict() for m in self.matches],
            "unmatched_old": [f"0x{a:x}" for a in self.unmatched_old],
            "unmatched_new": [f"0x{a:x}" for a in self.unmatched_new],
        }


# ---------------------------------------------------------------------------
# Similarity Algorithms (Diaphora / BinDiff pattern)
# ---------------------------------------------------------------------------


def _cosine_similarity(vec1: Dict[str, int], vec2: Dict[str, int]) -> float:
    """Calculate cosine similarity between two feature count dictionaries."""
    if not vec1 and not vec2:
        return 1.0
    if not vec1 or not vec2:
        return 0.0

    all_keys = set(vec1.keys()) | set(vec2.keys())
    dot_product = sum(vec1.get(k, 0) * vec2.get(k, 0) for k in all_keys)
    norm1 = math.sqrt(sum(v * v for v in vec1.values()))
    norm2 = math.sqrt(sum(v * v for v in vec2.values()))

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot_product / (norm1 * norm2)


def _ratio(a: int, b: int) -> float:
    """Ratio between two non-negative numbers in [0.0, 1.0]."""
    if a == 0 and b == 0:
        return 1.0
    if a <= 0 or b <= 0:
        return 0.0
    return min(a, b) / max(a, b)


def _mnemonic_histogram(ir_pcode: Optional[str]) -> Dict[str, int]:
    """Instruction-mnemonic histogram from a ``funcs.ir_pcode`` IR snapshot.

    ``ir_pcode`` holds one ``0x<addr>  <mnemonic> <operands>`` line per decoded
    instruction (see ``StaticEngine._render_ir``).  Only the mnemonic token is
    kept: operand addresses and immediates legitimately move between binary
    revisions, so the mnemonic bag is the version-stable part of the
    instruction mix (Diaphora/BinDiff heuristic, design §10 Tier 2).
    """
    hist: Dict[str, int] = {}
    if not ir_pcode:
        return hist
    for line in ir_pcode.splitlines():
        text = line.strip()
        if not text or text.startswith(";"):
            continue
        parts = text.split()
        if len(parts) < 2:
            continue
        mnemonic = parts[1].lower().strip()
        if mnemonic:
            hist[mnemonic] = hist.get(mnemonic, 0) + 1
    return hist


def _is_synthetic_name(name: str) -> bool:
    """Check if function name is an auto-generated address placeholder."""
    if not name:
        return True
    low = name.lower()
    return (
        low.startswith("sub_")
        or low.startswith("func_")
        or low.startswith("m_")
        or low.startswith("0x")
        or low == "targetclass"
        or low == "targetmethod"
        or low == "unnamed"
    )


def _classify_status(
    similarity: float, threshold: float = DEFAULT_AUTO_PORT_THRESHOLD
) -> str:
    """Map a similarity score to its epistemic port status (§10, §16 #2, RS-2).

    ``similarity < MATCH_FLOOR`` -> ``"UNMATCHED"``; ``MATCH_FLOOR <= similarity <
    threshold`` -> ``"REVIEW_REQUIRED"``; ``similarity >= threshold`` ->
    ``"AUTO_PORTED"``.
    """
    if similarity < MATCH_FLOOR:
        return "UNMATCHED"
    return "AUTO_PORTED" if similarity >= threshold else "REVIEW_REQUIRED"



# ---------------------------------------------------------------------------
# W20: 64-Dimensional Function Embeddings (design §9, execution-plan.md W20)
# ---------------------------------------------------------------------------

#: Feature-block layout of the 64-D vector: 16 dims per evidence family
#: (UNIT-W20-SEMANTIC-REBASE-PLAN.md §1.1).
OPCODE_BLOCK_START = 0
TOPOLOGY_BLOCK_START = 16
SSA_BLOCK_START = 32
NEIGHBOURHOOD_BLOCK_START = 48
BLOCK_DIM = 16

#: Feature-hash bucket counts inside the register-distribution and
#: Weisfeiler-Lehman sub-blocks.
SSA_VAR_BUCKETS = 8
WL_BUCKETS = 12


def _edge_kind(edge: Any) -> str:
    """Normalized lowercase edge kind (``EdgeKind``, enum value, or plain str)."""
    kind = getattr(edge, "kind", "")
    value = getattr(kind, "value", kind)
    return str(value).lower()


@dataclass
class FunctionEmbedding:
    """A 64-dimensional unit-vector semantic embedding of one function (W20).

    ``vector`` is the L2-normalized feature vector built by
    :class:`VectorSimilarityEngine`: D00-D15 opcode/mnemonic distribution,
    D16-D31 CFG topology kernel, D32-D47 SSA def-use density, D48-D63
    Weisfeiler-Lehman graph neighbourhood (UNIT-W20-SEMANTIC-REBASE-PLAN.md §1.1).
    The vector serializes to exactly ``EMBEDDING_BYTES`` (64 x IEEE-754 float32
    = 256) little-endian bytes for direct storage in ``versions.embedding_blob``.
    """

    func_addr: int
    vector: List[float]

    def __post_init__(self) -> None:
        self.func_addr = int(self.func_addr)
        self.vector = [float(value) for value in self.vector]
        if len(self.vector) != EMBEDDING_DIM:
            raise ValueError(
                f"embedding vector must have {EMBEDDING_DIM} dimensions; got {len(self.vector)}"
            )
        if any(not math.isfinite(value) for value in self.vector):
            raise ValueError("embedding vector must contain only finite floats")

    def __len__(self) -> int:
        return len(self.vector)

    def to_bytes(self) -> bytes:
        """Pack the vector into a 256-byte little-endian IEEE-754 float32 blob."""
        return struct.pack(f"<{EMBEDDING_DIM}f", *self.vector)

    @classmethod
    def from_bytes(cls, blob: bytes, func_addr: int = 0) -> FunctionEmbedding:
        """Unpack a serialized blob, validating the exact 256-byte width.

        A blob of the wrong width is schema drift, never a valid embedding, so
        it raises ``ValueError`` instead of silently producing a truncated vector.
        """
        data = bytes(blob)
        if len(data) != EMBEDDING_BYTES:
            raise ValueError(
                f"embedding blob must be exactly {EMBEDDING_BYTES} bytes; got {len(data)}"
            )
        return cls(func_addr=func_addr, vector=list(struct.unpack(f"<{EMBEDDING_DIM}f", data)))

    def digest(self) -> str:
        """SHA-256 hex digest of the packed blob (integrity/dedup anchor)."""
        return hashlib.sha256(self.to_bytes()).hexdigest()

    def cosine_similarity(self, other: FunctionEmbedding) -> float:
        """Cosine similarity against another embedding vector."""
        return VectorSimilarityEngine.cosine_similarity(self.vector, other.vector)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "func_addr": self.func_addr,
            "func_addr_hex": f"0x{self.func_addr:x}",
            "dim": len(self.vector),
            "norm": round(VectorSimilarityEngine.l2_norm(self.vector), 6),
            "sha256": self.digest(),
        }


class VectorSimilarityEngine:
    """W20 semantic embedding engine: 64-D feature extraction + vector cosine.

    Coexists with the W09 ``_cosine_similarity`` scorer, which scores sparse
    feature-count dictionaries; this engine scores dense, L2-normalized
    64-float unit vectors (design §9 semantic-embedding rebase).
    """

    #: Number of float32 dimensions in one function embedding.
    dim = EMBEDDING_DIM

    def __init__(self) -> None:
        self._cache: Dict[Tuple[str, int], FunctionEmbedding] = {}

    # -- similarity --------------------------------------------------------

    @staticmethod
    def l2_norm(vector: Sequence[float]) -> float:
        """Euclidean (L2) norm ``sqrt(sum(v_i ** 2))`` of a vector."""
        return math.sqrt(sum(float(value) * float(value) for value in vector))

    @staticmethod
    def normalize(vector: Sequence[float]) -> List[float]:
        """L2-normalize ``vector``; a zero vector falls back to the uniform unit vector.

        The uniform fallback keeps ``norm == 1.0`` for functions with no
        observable evidence, so the "64-float unit vector" contract holds
        unconditionally (UNIT-W20-SEMANTIC-REBASE-PLAN.md §1.1).
        """
        norm = VectorSimilarityEngine.l2_norm(vector)
        if norm == 0.0:
            return [1.0 / math.sqrt(EMBEDDING_DIM)] * EMBEDDING_DIM
        return [float(value) / norm for value in vector]

    @staticmethod
    def cosine_similarity(vec1: Sequence[float], vec2: Sequence[float]) -> float:
        """Exact cosine similarity ``(u . v) / (||u|| ||v||)`` with zero-norm guards.

        Returns ``0.0`` when either vector has zero norm (no comparable
        evidence) and ``1.0`` for identical vectors; the result is clamped to
        [-1, 1] to absorb float rounding noise.
        """
        if len(vec1) != len(vec2):
            raise ValueError(
                f"embedding vectors must share a dimension; got {len(vec1)} and {len(vec2)}"
            )
        sum1 = sum(float(a) * float(a) for a in vec1)
        sum2 = sum(float(b) * float(b) for b in vec2)
        if sum1 == 0.0 or sum2 == 0.0:
            return 0.0
        if list(vec1) == list(vec2):
            return 1.0
        dot_product = sum(float(a) * float(b) for a, b in zip(vec1, vec2))
        # sqrt(sum1 * sum2) is the same denominator as ||u|| ||v||, but stays
        # exact for opposite vectors (cos == -1.0) under float rounding.
        return max(-1.0, min(1.0, dot_product / math.sqrt(sum1 * sum2)))

    # -- feature extraction ------------------------------------------------

    @staticmethod
    def _feature_bucket(label: str, buckets: int) -> int:
        """Deterministic feature-hash of ``label`` into ``[0, buckets)``."""
        digest = hashlib.sha256(label.encode("utf-8", "replace")).digest()
        return int.from_bytes(digest[:4], "little") % buckets

    @staticmethod
    def _fact_histogram(
        facts: Union[Mapping[str, int], Sequence[Any], None],
    ) -> Tuple[Dict[str, int], int, int]:
        """Normalize ``facts`` into ``(var_histogram, fact_count, use_count)``.

        Accepts the ``Dict[str, int]`` var histogram produced by
        ``RebaseEngine._get_facts_for_func`` as well as raw ``SsaFact`` records
        or ``dict`` rows (``var_name`` / ``use_points``).
        """
        histogram: Dict[str, int] = {}
        if facts is None:
            return histogram, 0, 0
        if isinstance(facts, Mapping):
            for var, count in facts.items():
                name = str(var)
                weight = int(count)
                if name and weight > 0:
                    histogram[name] = histogram.get(name, 0) + weight
            return histogram, sum(histogram.values()), 0
        fact_count = 0
        use_count = 0
        for fact in facts:
            if isinstance(fact, Mapping):
                name = str(fact.get("var_name") or "")
                uses = fact.get("use_points")
            else:
                name = str(getattr(fact, "var_name", "") or "")
                uses = getattr(fact, "use_points", None)
            fact_count += 1
            if hasattr(uses, "__len__"):
                use_count += len(uses)
            if name:
                histogram[name] = histogram.get(name, 0) + 1
        return histogram, fact_count, use_count

    @staticmethod
    def _normalize_block(vector: List[float], start: int) -> None:
        """L2-normalize the 16-dim block starting at ``start`` (in place)."""
        block = vector[start : start + BLOCK_DIM]
        norm = VectorSimilarityEngine.l2_norm(block)
        if norm == 0.0:
            return
        for offset, value in enumerate(block):
            vector[start + offset] = value / norm

    def _fill_opcode_block(self, vector: List[float], func: Any) -> None:
        """D00-D15: normalized opcode (mnemonic) distribution."""
        histogram = _mnemonic_histogram(getattr(func, "ir_pcode", None))
        total = sum(histogram.values())
        if total <= 0:
            return
        for mnemonic, count in sorted(histogram.items()):
            index = self._feature_bucket(mnemonic, BLOCK_DIM)
            vector[OPCODE_BLOCK_START + index] += count / total

    def _fill_topology_block(
        self, vector: List[float], func: Any, edges: Sequence[EdgeRecord]
    ) -> None:
        """D16-D31: CFG topology kernel (blocks, edges, cyclomatic, fanout, loops)."""
        if not edges:
            # No connectivity evidence at all: leave D16-D31 empty so a function
            # with nothing observable takes the uniform unit-vector fallback.
            return
        cfg_edges = [e for e in edges if _edge_kind(e) == "cfg"]
        call_count = sum(1 for e in edges if _edge_kind(e) == "call")
        xref_count = sum(1 for e in edges if _edge_kind(e) == "xref")
        nodes = {int(e.src_addr) for e in cfg_edges} | {int(e.dst_addr) for e in cfg_edges}
        blocks = max(len(nodes), 1)
        out_degree: Dict[int, int] = {}
        for edge in cfg_edges:
            src = int(edge.src_addr)
            out_degree[src] = out_degree.get(src, 0) + 1
        fanout = max(out_degree.values(), default=0)
        branches = sum(1 for degree in out_degree.values() if degree > 1)
        leaves = sum(1 for degree in out_degree.values() if degree == 0)
        back_edges = sum(1 for e in cfg_edges if int(e.dst_addr) <= int(e.src_addr))
        size = max(int(getattr(func, "size", 0) or 0), 0)
        features = (
            blocks,
            len(cfg_edges),
            call_count,
            xref_count,
            max(len(cfg_edges) - blocks + 1, 0),
            fanout,
            branches,
            leaves,
            back_edges,
            size,
            size / blocks,
            len(cfg_edges) / blocks,
            branches / blocks,
            leaves / blocks,
        )
        for offset, value in enumerate(features):
            index = TOPOLOGY_BLOCK_START + (offset % BLOCK_DIM)
            vector[index] += math.log1p(max(float(value), 0.0))

    def _fill_ssa_block(
        self,
        vector: List[float],
        func: Any,
        facts: Union[Mapping[str, int], Sequence[Any], None],
    ) -> None:
        """D32-D47: SSA def-use variable density and register-class distribution."""
        histogram, fact_count, use_count = self._fact_histogram(facts)
        total = sum(histogram.values())
        if total > 0:
            for var, count in sorted(histogram.items()):
                index = self._feature_bucket(var, SSA_VAR_BUCKETS)
                vector[SSA_BLOCK_START + index] += count / total
        size = max(int(getattr(func, "size", 0) or 0), 0)
        metrics = (
            fact_count,
            len(histogram),
            use_count,
            use_count / fact_count if fact_count else 0.0,
            fact_count / size if size else 0.0,
            use_count / size if size else 0.0,
            max(histogram.values(), default=0),
        )
        for offset, value in enumerate(metrics):
            index = SSA_BLOCK_START + SSA_VAR_BUCKETS + (offset % SSA_VAR_BUCKETS)
            vector[index] += math.log1p(max(float(value), 0.0))

    def _fill_neighbourhood_block(self, vector: List[float], edges: Sequence[EdgeRecord]) -> None:
        """D48-D63: Weisfeiler-Lehman topological graph neighbourhood features.

        One WL round relabels every node with the multiset of its neighbours'
        degrees; the refined label multiset is feature-hashed into D48-D59,
        while D60-D63 carry coarse graph metrics (nodes, edges, mean degree,
        edge density).
        """
        adjacency: Dict[int, Set[int]] = {}
        for edge in edges:
            src, dst = int(edge.src_addr), int(edge.dst_addr)
            adjacency.setdefault(src, set()).add(dst)
            adjacency.setdefault(dst, set()).add(src)
        if not adjacency:
            return
        degrees = {node: len(neighbours) for node, neighbours in adjacency.items()}
        labels = {
            node: hashlib.sha256(f"d{degrees[node]}".encode("utf-8")).hexdigest()
            for node in adjacency
        }
        refined: Dict[int, str] = {}
        for node in sorted(adjacency):
            signature = ",".join(sorted(labels[nb] for nb in adjacency[node]))
            payload = f"{labels[node]}|{signature}"
            refined[node] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        for node in sorted(refined):
            index = self._feature_bucket(refined[node], WL_BUCKETS)
            vector[NEIGHBOURHOOD_BLOCK_START + index] += 1.0
        node_count = len(adjacency)
        edge_count = sum(len(neighbours) for neighbours in adjacency.values()) // 2
        possible = node_count * (node_count - 1) / 2
        metrics = (
            node_count,
            edge_count,
            edge_count / node_count,
            edge_count / possible if possible > 0 else 0.0,
        )
        for offset, value in enumerate(metrics):
            index = NEIGHBOURHOOD_BLOCK_START + WL_BUCKETS + offset
            vector[index] += math.log1p(max(float(value), 0.0))

    def generate_embedding(
        self,
        func: Any,
        edges: Sequence[EdgeRecord],
        facts: Union[Mapping[str, int], Sequence[Any], None],
    ) -> FunctionEmbedding:
        """Build the L2-normalized 64-D unit vector for one function (W20).

        Blocks of 16 dims each: D00-D15 opcode histogram, D16-D31 CFG topology,
        D32-D47 SSA def-use density, D48-D63 Weisfeiler-Lehman neighbourhood.
        Every block is L2-normalized before the whole vector is, so each
        evidence family contributes equally; a function with no observable
        evidence falls back to the uniform unit vector.
        """
        edge_list = list(edges or ())
        vector = [0.0] * EMBEDDING_DIM
        self._fill_opcode_block(vector, func)
        self._fill_topology_block(vector, func, edge_list)
        self._fill_ssa_block(vector, func, facts)
        self._fill_neighbourhood_block(vector, edge_list)
        for start in (
            OPCODE_BLOCK_START,
            TOPOLOGY_BLOCK_START,
            SSA_BLOCK_START,
            NEIGHBOURHOOD_BLOCK_START,
        ):
            self._normalize_block(vector, start)
        return FunctionEmbedding(
            func_addr=int(getattr(func, "func_addr", 0) or 0),
            vector=self.normalize(vector),
        )

    def embedding_for(
        self,
        func: Any,
        edges: Sequence[EdgeRecord],
        facts: Union[Mapping[str, int], Sequence[Any], None],
    ) -> FunctionEmbedding:
        """Memoized :meth:`generate_embedding` keyed by ``(binary_sha256, addr)``."""
        key = (
            str(getattr(func, "binary_sha256", "")),
            int(getattr(func, "func_addr", 0) or 0),
        )
        cached = self._cache.get(key)
        if cached is None:
            cached = self.generate_embedding(func, edges, facts)
            self._cache[key] = cached
        return cached

    def clear_cache(self) -> None:
        """Drop cached embeddings (call between rebase passes over new data)."""
        self._cache.clear()


# Rebase Engine
# ---------------------------------------------------------------------------


class FunctionPriors:
    """External rebase evidence: Diaphora candidate pairs as score priors.

    Keyed by ``(old_addr, new_addr)`` first, then by lowercased
    ``(old_name, new_name)``: candidate addresses are image-based and need not
    equal database addresses, so the name pair is the portable key and the
    address pair the precise one. Best score wins per key; lookup returns
    ``None`` when no candidate covers the pair.
    """

    def __init__(self) -> None:
        self._by_addrs: Dict[Tuple[int, int], float] = {}
        self._by_names: Dict[Tuple[str, str], float] = {}

    def add(
        self,
        old_addr: int,
        new_addr: int,
        score: float,
        old_name: str = "",
        new_name: str = "",
    ) -> None:
        """Record one candidate pair (best score wins per key)."""
        addr_key = (int(old_addr), int(new_addr))
        current = self._by_addrs.get(addr_key)
        if current is None or float(score) > current:
            self._by_addrs[addr_key] = float(score)
        old_norm = str(old_name or "").strip().lower()
        new_norm = str(new_name or "").strip().lower()
        if old_norm and new_norm:
            name_key = (old_norm, new_norm)
            current = self._by_names.get(name_key)
            if current is None or float(score) > current:
                self._by_names[name_key] = float(score)

    def lookup(
        self,
        old_addr: int,
        new_addr: int,
        old_name: Any = "",
        new_name: Any = "",
    ) -> Optional[float]:
        """Candidate score for one pair, or ``None`` when uncovered."""
        hit = self._by_addrs.get((int(old_addr), int(new_addr)))
        if hit is not None:
            return hit
        old_norm = str(old_name or "").strip().lower()
        new_norm = str(new_name or "").strip().lower()
        if old_norm and new_norm:
            return self._by_names.get((old_norm, new_norm))
        return None

    def __len__(self) -> int:
        return len(self._by_addrs)


def load_priors(doc: Any) -> FunctionPriors:
    """Build :class:`FunctionPriors` from an icepick diff-import document.

    Expected shape: ``{"candidates": [{"old_ea", "new_ea", "score"
    [, "old_name", "new_name"]}]}`` -- exactly what ``icepick diff-import``
    emits. JSON transport only, no IDA. Malformed input is refused with
    :class:`RebaseError`.
    """
    if not isinstance(doc, Mapping):
        raise RebaseError(
            f"priors document must be a JSON object, got {type(doc).__name__}"
        )
    raw = doc.get("candidates")
    if not isinstance(raw, list):
        raise RebaseError("priors document needs a 'candidates' list")
    priors = FunctionPriors()
    for index, entry in enumerate(raw):
        path = f"candidates[{index}]"
        if not isinstance(entry, Mapping):
            raise RebaseError(f"{path}: expected an object")
        score = entry.get("score")
        if (isinstance(score, bool) or not isinstance(score, (int, float))
                or not 0.0 <= float(score) <= 1.0):
            raise RebaseError(f"{path}.score: expected a number in [0.0, 1.0]")
        eas = []
        for key in ("old_ea", "new_ea"):
            value = entry.get(key)
            if (isinstance(value, bool) or not isinstance(value, int)
                    or value < 0):
                raise RebaseError(
                    f"{path}.{key}: expected a non-negative integer address"
                )
            eas.append(int(value))
        old_name = entry.get("old_name", "")
        new_name = entry.get("new_name", "")
        for key, value in (("old_name", old_name), ("new_name", new_name)):
            if value is not None and not isinstance(value, str):
                raise RebaseError(f"{path}.{key}: expected a string or null")
        priors.add(
            eas[0], eas[1], float(score),
            old_name if isinstance(old_name, str) else "",
            new_name if isinstance(new_name, str) else "",
        )
    return priors


class RebaseEngine:
    """Engine for comparing and rebasing functions across binary versions."""

    def __init__(self, db: Union[Database, PathLike]) -> None:
        if isinstance(db, Database):
            self.db: Database = db
            self._owned_db = False
        else:
            self.db = Database(Path(db), create=False)
            self._owned_db = True
        self.embeddings = VectorSimilarityEngine()

    def close(self) -> None:
        """Close database handle if owned (Trap 7)."""
        if self._owned_db and self.db is not None:
            self.db.close()

    def __enter__(self) -> RebaseEngine:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _get_edges_for_func(self, binary_sha256: str, func_addr: int) -> List[EdgeRecord]:
        """Fetch all CFG, call, and xref edges associated with func_addr."""
        rows = self.db._conn.execute(
            "SELECT * FROM edges WHERE binary_sha256 = ? AND src_addr = ?",
            (binary_sha256, func_addr),
        ).fetchall()
        return [
            EdgeRecord(
                binary_sha256=r["binary_sha256"],
                src_addr=r["src_addr"],
                dst_addr=r["dst_addr"],
                kind=r["kind"],
            )
            for r in rows
        ]

    def _get_facts_for_func(self, binary_sha256: str, func_addr: int) -> Dict[str, int]:
        """Histogram of registers defined/used in SSA facts."""
        facts = self.db.get_ssa_facts(binary_sha256, func_addr)
        hist: Dict[str, int] = {}
        for f in facts:
            var = f.get("var_name", "")
            if var:
                hist[var] = hist.get(var, 0) + 1
        return hist

    def embeddings_for(self, binary_sha256: str) -> Dict[int, FunctionEmbedding]:
        """FunctionEmbedding per function of one binary (for --dump-embeddings)."""
        return {
            func.func_addr: self.embeddings.embedding_for(
                func,
                self._get_edges_for_func(binary_sha256, func.func_addr),
                self._get_facts_for_func(binary_sha256, func.func_addr),
            )
            for func in self.db.get_functions(binary_sha256)
        }

    def compare_functions(
        self,
        f_old: FunctionRecord,
        f_new: FunctionRecord,
        old_edges: Sequence[EdgeRecord],
        new_edges: Sequence[EdgeRecord],
        old_facts_hist: Dict[str, int],
        new_facts_hist: Dict[str, int],
        *,
        priors: Optional[FunctionPriors] = None,
    ) -> Tuple[float, str, Dict[str, Any]]:
        """Compute similarity score and matching tier between two functions.

        ``priors`` carries external candidate evidence (Diaphora pairs): a
        name-confirmed pair is promoted to Tier-1 exact, and any covered pair
        gets a bounded one-directional uplift toward the candidate score (the
        embedding-uplift pattern). Priors never depress, never invent, never
        move the gates.
        """
        prior: Optional[float] = (
            priors.lookup(
                f_old.func_addr, f_new.func_addr, f_old.name, f_new.name
            )
            if priors is not None
            else None
        )
        prior_overrode_mismatch = False
        # Tier 1: Exact meaningful symbol name match
        has_real_name = (
            bool(f_old.name)
            and not _is_synthetic_name(f_old.name)
            and bool(f_new.name)
            and not _is_synthetic_name(f_new.name)
        )
        if has_real_name:
            if f_old.name.lower() == f_new.name.lower():
                size_r = _ratio(f_old.size, f_new.size)
                if prior is not None:
                    # External diff evidence confirms the same function across
                    # versions: the size gate is waived, Tier-1 exact holds.
                    return (
                        1.0,
                        MATCH_TIER_EXACT_NAME,
                        {
                            "name_match": True,
                            "size_ratio": size_r,
                            "prior_score": prior,
                            "prior_boosted_tier1": True,
                        },
                    )
                if size_r >= 0.70:
                    return (
                        1.0,
                        MATCH_TIER_EXACT_NAME,
                        {"name_match": True, "size_ratio": size_r},
                    )
            elif prior is not None:
                # Renamed across versions per external evidence: fall through
                # to Tier-2 with uplift instead of refusing. Without a prior
                # the refusal below stands unchanged.
                prior_overrode_mismatch = True
            else:
                # Both functions have distinct non-synthetic symbol names that differ.
                # Symbol names are a strong anchor (Diaphora pattern): refuse the pair.
                # Score stays below MATCH_FLOOR so the pair is reported UNMATCHED with
                # the documented NAME_MISMATCH tier for diagnostics only.
                return (
                    0.0,
                    MATCH_TIER_NAME_MISMATCH,
                    {"name_old": f_old.name, "name_new": f_new.name},
                )

        # CFG topology metrics (basic blocks, edges, calls) -- design §10 Tier 2
        old_cfg = [e for e in old_edges if e.kind == "cfg"]
        new_cfg = [e for e in new_edges if e.kind == "cfg"]
        old_calls = [e for e in old_edges if e.kind == "call"]
        new_calls = [e for e in new_edges if e.kind == "call"]

        old_blocks = len(set(e.src_addr for e in old_cfg) | set(e.dst_addr for e in old_cfg)) or 1
        new_blocks = len(set(e.src_addr for e in new_cfg) | set(e.dst_addr for e in new_cfg)) or 1

        cfg_score = 0.5 * _ratio(old_blocks, new_blocks) + 0.5 * _ratio(len(old_cfg), len(new_cfg))
        call_score = _ratio(len(old_calls), len(new_calls))
        size_score = _ratio(f_old.size, f_new.size)
        reg_score = _cosine_similarity(old_facts_hist, new_facts_hist)

        # Instruction-mnemonic histogram cosine similarity (design §10 Tier 2).
        # Scored only when at least one side carries ``ir_pcode``: with no
        # disassembly evidence on either side the component is dropped and the
        # remaining weights are renormalised, so a fact-poor database is neither
        # credited nor penalised for mnemonics that were never observed.
        old_mnem = _mnemonic_histogram(f_old.ir_pcode)
        new_mnem = _mnemonic_histogram(f_new.ir_pcode)
        mnem_score: Optional[float] = (
            _cosine_similarity(old_mnem, new_mnem) if (old_mnem or new_mnem) else None
        )

        # Tier 1b: Exact structural identity (identical size, CFG, calls,
        # registers, and -- when both sides are disassembled -- mnemonic mix).
        if (
            f_old.size == f_new.size
            and old_blocks == new_blocks
            and len(old_cfg) == len(new_cfg)
            and len(old_calls) == len(new_calls)
            and reg_score >= 0.999
            and (mnem_score is None or mnem_score >= 0.999)
        ):
            exact_details: Dict[str, Any] = {
                "cfg_score": 1.0,
                "reg_score": 1.0,
                "call_score": 1.0,
                "size_score": 1.0,
            }
            if mnem_score is not None:
                exact_details["mnemonic_score"] = 1.0
            return (1.0, MATCH_TIER_EXACT_STRUCTURE, exact_details)

        # Tier 2: weighted composite similarity over the observable heuristics.
        # Weights: CFG 35%, register def-use 35%, calls 15%, size 15%, mnemonic
        # mix 15%; absent-evidence components are dropped and the surviving
        # weights renormalised so the score stays a weighted mean in [0, 1].
        weighted: List[Tuple[str, float, float]] = [
            ("cfg_score", cfg_score, 0.35),
            ("reg_score", reg_score, 0.35),
            ("call_score", call_score, 0.15),
            ("size_score", size_score, 0.15),
        ]
        if mnem_score is not None:
            weighted.append(("mnemonic_score", mnem_score, 0.15))
        total_weight = sum(weight for _, _, weight in weighted)
        composite = sum(score * weight for _, score, weight in weighted) / total_weight
        base_composite = composite

        details: Dict[str, Any] = {name: round(score, 4) for name, score, _ in weighted}
        details["weights"] = {
            name: round(weight / total_weight, 4) for name, _, weight in weighted
        }

        # Tier 2 (W20): exact 64-D embedding cosine.  A cosine at or above
        # SEMANTIC_EMBEDDING_THRESHOLD promotes the pair to the documented
        # MATCH_TIER_SEMANTIC_EMBEDDING tier, and the embedding evidence is
        # folded in as a bounded, one-directional uplift (EMBEDDING_SCORE_WEIGHT)
        # so a semantically near-identical pair scores higher without ever
        # depressing the W09 structural composite: the >= 0.99 auto-port gate
        # and the 0.50-0.99 REVIEW_REQUIRED band are preserved exactly (§16 #2,
        # RS-2).  Exact name/structure tiers outrank this tier and return above.
        old_embedding = self.embeddings.embedding_for(f_old, old_edges, old_facts_hist)
        new_embedding = self.embeddings.embedding_for(f_new, new_edges, new_facts_hist)
        embedding_cosine = VectorSimilarityEngine.cosine_similarity(
            old_embedding.vector, new_embedding.vector
        )
        semantic_match = embedding_cosine >= SEMANTIC_EMBEDDING_THRESHOLD
        if semantic_match:
            composite = min(
                1.0,
                base_composite + EMBEDDING_SCORE_WEIGHT * (embedding_cosine - base_composite),
            )
        details["embedding_cosine"] = round(embedding_cosine, 4)
        details["embedding_uplift"] = round(composite - base_composite, 4)
        details["embedding_sha256"] = new_embedding.digest()
        # Priors seed the Tier-2 weights: a candidate score above the
        # structural composite uplifts it one-directionally toward the
        # candidate score. A lower-or-equal prior changes nothing, so
        # structural evidence always wins disagreements.
        if prior is not None:
            pre_prior = composite
            if prior > composite:
                composite = min(
                    1.0,
                    composite + PRIOR_SCORE_WEIGHT * (prior - composite),
                )
            details["prior_score"] = prior
            details["prior_uplift"] = round(composite - pre_prior, 4)
            if prior_overrode_mismatch:
                details["prior_overrode_name_mismatch"] = True
        tier = MATCH_TIER_SEMANTIC_EMBEDDING if semantic_match else MATCH_TIER_GRAPH_SIMILARITY
        return composite, tier, details

    def _aggregate_embedding_blob(
        self,
        funcs: Sequence[FunctionRecord],
        edges_map: Mapping[int, Sequence[EdgeRecord]],
        facts_map: Mapping[int, Union[Mapping[str, int], Sequence[Any], None]],
    ) -> bytes:
        """Aggregate baseline embeddings into one 256-byte float32 blob (W20).

        Per-function unit vectors are averaged and re-normalized (a mean of
        unit vectors is not itself unit length); an evidence-free baseline
        falls back to the uniform unit vector, so the persisted blob always
        satisfies the 64-float unit-vector contract.
        """
        accumulator = [0.0] * EMBEDDING_DIM
        for func in funcs:
            embedding = self.embeddings.embedding_for(
                func,
                edges_map.get(func.func_addr, ()),
                facts_map.get(func.func_addr, {}),
            )
            for index, value in enumerate(embedding.vector):
                accumulator[index] += value
        aggregate = FunctionEmbedding(
            func_addr=0,
            vector=VectorSimilarityEngine.normalize(accumulator),
        )
        return aggregate.to_bytes()

    def rebase(
        self,
        old_sha256: str,
        new_target: Union[str, PathLike, Sequence[FunctionRecord]],
        new_sha256: Optional[str] = None,
        threshold: float = DEFAULT_AUTO_PORT_THRESHOLD,
        priors: Optional[FunctionPriors] = None,
    ) -> RebaseReport:
        """Execute version rebase diff between old_sha256 and new_target.

        ``threshold`` is the auto-port gate (default 0.99 per design §16 #2): pairs
        scoring ``>= threshold`` are ``"AUTO_PORTED"``, pairs in
        ``[MATCH_FLOOR, threshold)`` are ``"REVIEW_REQUIRED"`` (RS-2 review queue), and
        everything below ``MATCH_FLOOR`` stays ``"UNMATCHED"``.

        ``priors`` threads external candidate evidence through both scoring
        passes (see :meth:`compare_functions`); ``None`` is the legacy
        behavior exactly.
        """
        if not isinstance(threshold, (int, float)) or not 0.0 < float(threshold) <= 1.0:
            raise RebaseError(f"threshold must be a number in (0.0, 1.0]; got {threshold!r}")
        threshold = float(threshold)

        old_funcs = self.db.get_functions(old_sha256)
        if not old_funcs:
            raise RebaseError(f"No functions found in database for baseline SHA {old_sha256}")

        # Resolve new functions
        target_sha = new_sha256 or ""
        new_funcs: List[FunctionRecord] = []

        if isinstance(new_target, (str, Path)):
            path_or_sha = str(new_target)
            target_path = Path(path_or_sha)
            if target_path.is_file():
                # Analyze binary into database if not present
                engine = StaticEngine(db_path=self.db.db_path)
                result = engine.analyze(target_path, write_db=True)
                target_sha = result.binary_sha256
                new_funcs = self.db.get_functions(target_sha)
            else:
                # Treat as an existing binary SHA in the database
                target_sha = path_or_sha
                new_funcs = self.db.get_functions(target_sha)
        elif isinstance(new_target, Sequence):
            new_funcs = list(new_target)
            if not target_sha and new_funcs:
                target_sha = new_funcs[0].binary_sha256

        if not new_funcs:
            raise RebaseError(f"No functions found or extracted for target {new_target}")

        if not target_sha:
            target_sha = "target_rebase"

        # Pre-fetch edges and facts
        old_edges_map: Dict[int, List[EdgeRecord]] = {
            f.func_addr: self._get_edges_for_func(old_sha256, f.func_addr)
            for f in old_funcs
        }
        new_edges_map: Dict[int, List[EdgeRecord]] = {
            f.func_addr: self._get_edges_for_func(target_sha, f.func_addr)
            for f in new_funcs
        }
        old_facts_map: Dict[int, Dict[str, int]] = {
            f.func_addr: self._get_facts_for_func(old_sha256, f.func_addr)
            for f in old_funcs
        }
        new_facts_map: Dict[int, Dict[str, int]] = {
            f.func_addr: self._get_facts_for_func(target_sha, f.func_addr)
            for f in new_funcs
        }

        # Multi-tier candidate scoring
        matched_old: Set[int] = set()
        matched_new: Set[int] = set()
        matches: List[RebaseMatch] = []

        # Pass 1: Exact matches (Tier 1)
        for f_old in sorted(old_funcs, key=lambda x: x.func_addr):
            for f_new in sorted(new_funcs, key=lambda x: x.func_addr):
                if f_new.func_addr in matched_new:
                    continue
                score, tier, details = self.compare_functions(
                    f_old,
                    f_new,
                    old_edges_map[f_old.func_addr],
                    new_edges_map[f_new.func_addr],
                    old_facts_map[f_old.func_addr],
                    new_facts_map[f_new.func_addr],
                    priors=priors,
                )
                if tier in EXACT_MATCH_TIERS and score >= threshold:
                    matched_old.add(f_old.func_addr)
                    matched_new.add(f_new.func_addr)
                    matches.append(
                        RebaseMatch(
                            old_addr=f_old.func_addr,
                            new_addr=f_new.func_addr,
                            old_name=f_old.name,
                            new_name=f_new.name,
                            similarity=score,
                            status=_classify_status(score, threshold),
                            match_tier=tier,
                            details=details,
                        )
                    )
                    break

        # Pass 2: Bipartite greedy assignment over remaining functions
        unmatched_old_list = [f for f in old_funcs if f.func_addr not in matched_old]
        unmatched_new_list = [f for f in new_funcs if f.func_addr not in matched_new]

        candidates: List[Tuple[float, str, Dict[str, Any], FunctionRecord, FunctionRecord]] = []
        for f_old in unmatched_old_list:
            for f_new in unmatched_new_list:
                score, tier, details = self.compare_functions(
                    f_old,
                    f_new,
                    old_edges_map[f_old.func_addr],
                    new_edges_map[f_new.func_addr],
                    old_facts_map[f_old.func_addr],
                    new_facts_map[f_new.func_addr],
                    priors=priors,
                )
                if tier == MATCH_TIER_NAME_MISMATCH:
                    # Distinct real symbol names: the documented NAME_MISMATCH tier is
                    # diagnostic only (score 0.0, below MATCH_FLOOR) -- never pair.
                    continue
                if score >= MATCH_FLOOR:
                    candidates.append((score, tier, details, f_old, f_new))

        # Sort candidates by score descending, then by old_addr, new_addr for determinism (Trap 10)
        candidates.sort(key=lambda c: (-c[0], c[3].func_addr, c[4].func_addr))

        for score, tier, details, f_old, f_new in candidates:
            if f_old.func_addr in matched_old or f_new.func_addr in matched_new:
                continue
            matched_old.add(f_old.func_addr)
            matched_new.add(f_new.func_addr)
            status = _classify_status(score, threshold)
            matches.append(
                RebaseMatch(
                    old_addr=f_old.func_addr,
                    new_addr=f_new.func_addr,
                    old_name=f_old.name,
                    new_name=f_new.name,
                    similarity=score,
                    status=status,
                    match_tier=tier,
                    details=details,
                )
            )

        # Leftover unmatched functions
        unmatched_old_addrs = [f.func_addr for f in old_funcs if f.func_addr not in matched_old]
        unmatched_new_addrs = [f.func_addr for f in new_funcs if f.func_addr not in matched_new]

        auto_ported_count = sum(1 for m in matches if m.status == "AUTO_PORTED")
        review_required_count = sum(1 for m in matches if m.status == "REVIEW_REQUIRED")
        unmatched_count = len(unmatched_old_addrs)

        auto_port_rate = auto_ported_count / len(old_funcs) if old_funcs else 0.0

        # Build offset map for auto-ported functions
        offset_map = {
            f"0x{m.old_addr:x}": m.new_addr
            for m in matches
            if m.status == "AUTO_PORTED"
        }

        report = RebaseReport(
            old_sha256=old_sha256,
            new_sha256=target_sha,
            total_old_funcs=len(old_funcs),
            total_new_funcs=len(new_funcs),
            auto_ported=auto_ported_count,
            review_required=review_required_count,
            unmatched=unmatched_count,
            auto_port_rate=auto_port_rate,
            threshold=threshold,
            matches=sorted(matches, key=lambda m: m.old_addr),
            offset_map=offset_map,
            unmatched_old=sorted(unmatched_old_addrs),
            unmatched_new=sorted(unmatched_new_addrs),
        )

        # Persist VersionRecord in toothpick.db
        avg_similarity = (
            sum(m.similarity for m in matches) / len(matches) if matches else 0.0
        )
        # W20: the baseline's 64-D function embeddings are aggregated (mean of
        # unit vectors, re-normalized) and persisted as a 256-byte float32 blob
        # in `versions.embedding_blob` alongside the offset map (design §9).
        embedding_blob = self._aggregate_embedding_blob(
            old_funcs, old_edges_map, old_facts_map
        )

        version_rec = VersionRecord(
            binary_sha256=old_sha256,
            target_sha256=target_sha,
            similarity_score=avg_similarity,
            embedding_blob=embedding_blob,
            offset_map=offset_map,
        )
        self.db.add_version(version_rec)

        # Persist receipt
        self.db.add_receipt(
            binary_sha256=old_sha256,
            func_addr=0,
            phase="rebase",
            payload={
                "old_sha256": old_sha256,
                "new_sha256": target_sha,
                "auto_port_rate": report.auto_port_rate,
                "threshold": threshold,
                "auto_ported": report.auto_ported,
                "review_required": report.review_required,
                "unmatched": report.unmatched,
                "embedding_blob_bytes": len(embedding_blob),
                "embedding_sha256": hashlib.sha256(embedding_blob).hexdigest(),
                "semantic_matches": sum(
                    1 for m in matches if m.match_tier == MATCH_TIER_SEMANTIC_EMBEDDING
                ),
            },
        )

        return report

    @classmethod
    def port_mod_config(cls, config: ModConfig, report: RebaseReport) -> ModConfig:
        """Port a ModConfig's target_addr to the updated binary version."""
        if config.target_addr is None:
            raise RebaseError("ModConfig has no target_addr specified to rebase")

        old_addr_hex = f"0x{config.target_addr:x}"
        if old_addr_hex not in report.offset_map:
            raise RebaseError(
                f"Target offset {old_addr_hex} ({config.target_class}.{config.target_method}) was not "
                "auto-ported (similarity below threshold or function unmatched in new binary)"
            )

        new_addr = report.offset_map[old_addr_hex]
        return ModConfig(
            plugin_name=config.plugin_name,
            plugin_id=config.plugin_id,
            version=config.version,
            target_addr=new_addr,
            target_class=config.target_class,
            target_method=config.target_method,
            hook_type=config.hook_type,
            boot_token=config.boot_token,
            mod_logic=config.mod_logic,
            binary_sha256=report.new_sha256,
            author=config.author,
        )
