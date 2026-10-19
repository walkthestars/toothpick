"""
Toothpick phase-constrained query engine (W04).

Binding contract: ``PLAN-W04-mcp.md`` lines 8-17 (done criteria 1); governing
spec ``docs/design-document.md`` sections 11-12 ("MCP surface (phase-constrained
subsets, not flat 200-tool dump): ``def_use()``, ``callers()/callees()``,
``slice(back|fwd, source|sink, depth)``").

The engine answers call-graph, def-use, and intra-function program-slicing
questions **strictly from the W03 graph store** (``toothpick.db``, see
``toothpick.storage``) with no re-disassembly, no network access and no writes.
Every list-returning method is bounded by an explicit top-K budget
(:data:`DEFAULT_LIMIT` = 50) and never returns an unbounded list, so a dense
graph (the 10x context stress case in ``tests/test_query.py``) cannot blow up the
agent's context window -- Risk R1 mitigation.

P1 extension (``docs/units/UNIT-P1-PROFILE-PLAN.md`` section 1.3): :meth:`QueryEngine.hottest`
joins sampled-profile hotness (VTune/ETW/perf) into this surface and ``get_function``
carries a ``profile`` key.  Both delegate to :mod:`toothpick.profile`, imported **lazily
inside the methods** -- this module keeps no import-time dependency on the profile
subsystem (and the laziness is what keeps the two modules cycle-free).

Determinism contract: all reads are ordered by explicit sort keys, so repeated
queries with identical arguments return identical results (slice idempotence
property). Nothing here depends on dict/set iteration order.
"""

from __future__ import annotations

import sqlite3
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from toothpick.storage import Database, FunctionRecord, StorageError

__all__ = [
    "DEFAULT_LIMIT",
    "MAX_LIMIT",
    "MAX_DEPTH",
    "QueryError",
    "SliceDirection",
    "parse_address",
    "QueryEngine",
]


#: Default top-K budget for every query method (PLAN-W04-mcp.md line 14).
DEFAULT_LIMIT = 50

#: Hard ceiling: even an explicit ``limit`` is clamped to this, so a caller
#: cannot ask for an unbounded (or context-killing) response.
MAX_LIMIT = 1_000

#: Hard ceiling for slice traversal depth (def-use hops).
MAX_DEPTH = 64

PathLike = Union[str, Path]
DbLike = Union[Database, str, Path]


class QueryError(Exception):
    """Raised when a query cannot be answered.

    Examples: the database file does not exist, the queried binary is ambiguous
    (more than one ``binary_sha256`` in the store and none given), a ``limit`` of
    zero/negative, or a malformed address.
    """


class SliceDirection(str, Enum):
    """Program-slice direction (``backward`` = from sink, ``forward`` = from source)."""

    BACKWARD = "backward"
    FORWARD = "forward"

    @classmethod
    def coerce(cls, value: Union["SliceDirection", str]) -> "SliceDirection":
        """Accept the enum, ``"backward"``/``"forward"``, or ``"back"``/``"fwd"``."""
        if isinstance(value, cls):
            return value
        text = str(value).strip().lower()
        aliases = {"back": cls.BACKWARD, "fwd": cls.FORWARD, "forwards": cls.FORWARD}
        if text in aliases:
            return aliases[text]
        try:
            return cls(text)
        except ValueError as exc:
            raise QueryError(
                f"invalid slice direction {value!r}; expected 'backward' or 'forward'"
            ) from exc


def parse_address(value: Union[int, str]) -> int:
    """Parse an address from an int, ``0x...`` hex, bare hex, or decimal string.

    ``0x1000`` -> 4096, ``10f0`` -> 4336 (bare hex digits containing a-f),
    ``4096`` -> 4096 (decimal).
    """
    if isinstance(value, bool):  # bool is an int subclass; never a valid address
        raise QueryError(f"invalid address {value!r}")
    if isinstance(value, int):
        if value < 0:
            raise QueryError(f"address must be non-negative, got {value}")
        return value
    if not isinstance(value, str):
        raise QueryError(f"invalid address {value!r}; expected int or string")
    text = value.strip()
    if not text:
        raise QueryError("address must not be empty")
    negative = text.startswith("-")
    body = text[1:] if negative else text
    try:
        if body.lower().startswith("0x"):
            addr = int(body, 16)
        elif any(c in "abcdefABCDEF" for c in body):
            addr = int(body, 16)
        else:
            addr = int(body, 10)
    except ValueError as exc:
        raise QueryError(f"invalid address {value!r}") from exc
    if negative:
        raise QueryError(f"address must be non-negative, got {value}")
    return addr


def _base_reg(name: str) -> str:
    """``"rax_1"`` -> ``"rax"``; leaves non-SSA names untouched."""
    if not name:
        return ""
    return name.split("_", 1)[0]


def _hex_addr(addr: int) -> str:
    return f"0x{addr:x}"


def _fact_matches(fact: Dict[str, Any], needle: str) -> bool:
    """True when ``needle`` names this SSA fact (exact or base-register match)."""
    if fact.get("var_name") == needle or fact.get("ssa_reg") == needle:
        return True
    for key in ("var_name", "ssa_reg"):
        value = fact.get(key) or ""
        if value and (_base_reg(value) == needle or _base_reg(value) == _base_reg(needle)):
            return True
    return False


def _truncate_lines(text: Optional[str], max_lines: Optional[int]) -> Tuple[str, bool]:
    if not text:
        return "", False
    if max_lines is None or max_lines <= 0:
        return text, False
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text, False
    truncated = "\n".join(lines[:max_lines]) + f"\n... [truncated {len(lines) - max_lines} lines]"
    return truncated, True


_CALLER_SQL = """
SELECT e.src_addr AS addr,
       f.name AS name,
       f.size AS size,
       f.calling_convention AS calling_convention,
       f.prototype AS prototype
  FROM edges e
  LEFT JOIN funcs f
         ON f.binary_sha256 = e.binary_sha256
        AND f.func_addr = e.src_addr
 WHERE e.binary_sha256 = ? AND e.dst_addr = ? AND e.kind = 'call'
 ORDER BY e.src_addr
 LIMIT ?
"""

_CALLEE_SQL = """
SELECT e.dst_addr AS addr,
       f.name AS name,
       f.size AS size,
       f.calling_convention AS calling_convention,
       f.prototype AS prototype
  FROM edges e
  LEFT JOIN funcs f
         ON f.binary_sha256 = e.binary_sha256
        AND f.func_addr = e.dst_addr
 WHERE e.binary_sha256 = ? AND e.src_addr = ? AND e.kind = 'call'
 ORDER BY e.dst_addr
 LIMIT ?
"""

_COUNT_SQL = {
    "caller": (
        "SELECT COUNT(DISTINCT src_addr) AS n FROM edges "
        "WHERE binary_sha256 = ? AND dst_addr = ? AND kind = 'call'"
    ),
    "callee": (
        "SELECT COUNT(DISTINCT dst_addr) AS n FROM edges "
        "WHERE binary_sha256 = ? AND src_addr = ? AND kind = 'call'"
    ),
}


class QueryEngine:
    """Phase-constrained, budgeted reader over ``toothpick.db``.

    Usage::

        engine = QueryEngine(Path("toothpick.db"))
        engine.callers(sha, 0x1000, limit=50)          # -> [ {func_addr, name, ..} ]
        engine.def_use(sha, 0x2000, var_name="rax")    # -> [ {def_point, use_points, ..} ]
        engine.slice(sha, 0x2000, target="rax", direction="backward")
    """

    def __init__(self, db: DbLike, *, create: bool = False) -> None:
        """Bind to an open :class:`~toothpick.storage.Database` or a db path.

        A path must already exist -- silently creating an empty ``toothpick.db``
        for a typo'd ``--db`` would answer every question with ``[]``.
        """
        if isinstance(db, Database):
            self.db = db
            self._owned = False
        else:
            path = Path(db)
            if not path.exists():
                raise QueryError(f"database not found: {path}")
            try:
                self.db = Database(path, create=create)
            except StorageError as exc:
                raise QueryError(f"cannot open database {path}: {exc}") from exc
            self._owned = True
        self._cached_conn: Optional[sqlite3.Connection] = None

    def _get_read_conn(self) -> sqlite3.Connection:
        if self._cached_conn is None:
            self._cached_conn = self.db._new_connection()
        return self._cached_conn

    def close(self) -> None:
        """Close the cached connection and underlying database if this engine opened it."""
        if self._cached_conn is not None:
            try:
                self._cached_conn.close()
            except Exception:
                pass
            self._cached_conn = None
        if self._owned:
            self.db.close()

    def __enter__(self) -> "QueryEngine":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    # -- budget / input coercion ------------------------------------------

    def resolve_limit(self, limit: Optional[int] = None) -> int:
        """Public top-K coercion (MCP layer / CLI): positive int, clamped to MAX_LIMIT."""
        return self._limit(limit)

    def resolve_max_depth(self, max_depth: Optional[int] = None) -> int:
        """Public depth-cap coercion (MCP layer / CLI): non-negative, clamped to MAX_DEPTH."""
        return self._max_depth(max_depth)

    @staticmethod
    def _limit(limit: Optional[int]) -> int:
        """Coerce ``limit`` to a positive int clamped to :data:`MAX_LIMIT`."""
        if limit is None:
            return DEFAULT_LIMIT
        try:
            value = int(limit)
        except (TypeError, ValueError) as exc:
            raise QueryError(f"limit must be an integer, got {limit!r}") from exc
        if value <= 0:
            raise QueryError(f"limit must be positive, got {value}")
        return min(value, MAX_LIMIT)

    @staticmethod
    def _max_depth(max_depth: Optional[int]) -> int:
        if max_depth is None:
            return 10
        try:
            value = int(max_depth)
        except (TypeError, ValueError) as exc:
            raise QueryError(f"max_depth must be an integer, got {max_depth!r}") from exc
        if value < 0:
            raise QueryError(f"max_depth must be non-negative, got {value}")
        return min(value, MAX_DEPTH)

    def _fetch_all(self, sql: str, params: Tuple[Any, ...]) -> List[sqlite3.Row]:
        try:
            conn = self._get_read_conn()
            return conn.execute(sql, params).fetchall()
        except sqlite3.Error as exc:
            raise QueryError(f"query failed: {exc}") from exc

    # -- binary selection --------------------------------------------------

    def binaries(self) -> List[str]:
        """Distinct ``binary_sha256`` values present in the store (sorted)."""
        rows = self._fetch_all(
            "SELECT DISTINCT binary_sha256 FROM funcs ORDER BY binary_sha256", ()
        )
        return [str(r["binary_sha256"]) for r in rows]

    def resolve_sha(self, binary_sha256: Optional[str] = None) -> str:
        """Return ``binary_sha256``, inferring it when the store holds exactly one."""
        if binary_sha256:
            return str(binary_sha256)
        shas = self.binaries()
        if len(shas) == 1:
            return shas[0]
        if not shas:
            raise QueryError("database holds no binaries; pass binary_sha256/--sha")
        raise QueryError(
            f"database holds {len(shas)} binaries; pass binary_sha256/--sha explicitly"
        )

    # -- call graph --------------------------------------------------------

    def callers(
        self,
        binary_sha256: Optional[str],
        func_addr: Union[int, str],
        limit: Optional[int] = DEFAULT_LIMIT,
    ) -> List[Dict[str, Any]]:
        """Functions that call ``func_addr`` (call-graph predecessors, top-K)."""
        return self._neighbours(binary_sha256, func_addr, "caller", limit)

    def callees(
        self,
        binary_sha256: Optional[str],
        func_addr: Union[int, str],
        limit: Optional[int] = DEFAULT_LIMIT,
    ) -> List[Dict[str, Any]]:
        """Functions called by ``func_addr`` (call-graph successors, top-K)."""
        return self._neighbours(binary_sha256, func_addr, "callee", limit)

    def count_callers(self, binary_sha256: Optional[str], func_addr: Union[int, str]) -> int:
        """Total number of distinct caller edges (unbudgeted; for ``truncated`` flags)."""
        return self._count(binary_sha256, func_addr, "caller")

    def count_callees(self, binary_sha256: Optional[str], func_addr: Union[int, str]) -> int:
        """Total number of distinct callee edges (unbudgeted; for ``truncated`` flags)."""
        return self._count(binary_sha256, func_addr, "callee")

    def _count(
        self, binary_sha256: Optional[str], func_addr: Union[int, str], role: str
    ) -> int:
        sha = self.resolve_sha(binary_sha256)
        addr = parse_address(func_addr)
        rows = self._fetch_all(_COUNT_SQL[role], (sha, addr))
        return int(rows[0]["n"]) if rows else 0

    def _neighbours(
        self,
        binary_sha256: Optional[str],
        func_addr: Union[int, str],
        role: str,
        limit: Optional[int],
    ) -> List[Dict[str, Any]]:
        sha = self.resolve_sha(binary_sha256)
        addr = parse_address(func_addr)
        budget = self._limit(limit)
        sql = _CALLER_SQL if role == "caller" else _CALLEE_SQL
        rows = self._fetch_all(sql, (sha, addr, budget))
        items: List[Dict[str, Any]] = []
        for row in rows[:budget]:
            neighbour = int(row["addr"])
            items.append(
                {
                    "binary_sha256": sha,
                    "func_addr": neighbour,
                    "addr_hex": _hex_addr(neighbour),
                    "role": role,
                    "name": row["name"] or "",
                    "size": int(row["size"] or 0),
                    "calling_convention": row["calling_convention"] or "",
                    "prototype": row["prototype"] or "",
                    "query_func_addr": addr,
                    "query_addr_hex": _hex_addr(addr),
                }
            )
        return items

    # -- profile hotness (P1) ----------------------------------------------

    def hottest(
        self,
        binary_sha256: Optional[str],
        top_n: Optional[int] = DEFAULT_LIMIT,
    ) -> List[Dict[str, Any]]:
        """Functions of ``binary_sha256`` by sampled hotness, top-K and bounded (R1).

        Thin delegation to :func:`toothpick.profile.hottest` over this engine's own
        database, so a P1 profile imported through any format is reachable from the
        W04 query surface without re-disassembly.  ``top_n`` goes through the same
        budget path as every other list-returning method -- default
        :data:`DEFAULT_LIMIT` (50), clamped to :data:`MAX_LIMIT` (1000) -- so an
        unbounded hotness list is unreachable.

        ``func_profile`` is optional state: a store that never saw a profile import
        answers ``[]`` (reads never create the table).  Ordering is the module's
        deterministic ``hotness DESC, func_addr ASC`` key order, so repeated queries
        with identical arguments return identical lists.

        ``toothpick.profile`` is imported here rather than at module scope: the
        profile subsystem imports :mod:`toothpick.storage`, and deferring the import
        keeps ``query`` free of an import-time dependency (and of any cycle).
        """
        sha = self.resolve_sha(binary_sha256)
        budget = self._limit(top_n)
        from toothpick.profile import ProfileError
        from toothpick.profile import hottest as _profile_hottest

        try:
            return _profile_hottest(self.db, sha, budget)
        except ProfileError as exc:
            raise QueryError(f"profile query failed: {exc}") from exc

    def _profile_row(self, sha: str, addr: int) -> Optional[Dict[str, Any]]:
        """``toothpick.profile.get_profile`` result, or ``None`` when unprofiled.

        Lazily imported for the same reason as :meth:`hottest`; a store without a
        ``func_profile`` table (or without a row for this function) yields ``None``,
        which is what ``get_function`` puts in its ``profile`` key.
        """
        from toothpick.profile import ProfileError, get_profile

        try:
            return get_profile(self.db, sha, addr)
        except ProfileError as exc:
            raise QueryError(f"profile query failed: {exc}") from exc
    # -- functions ---------------------------------------------------------

    def get_function(
        self,
        binary_sha256: Optional[str],
        func_addr: Union[int, str],
        max_lines: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """One analysed function row (``None`` when the address is not in the store)."""
        sha = self.resolve_sha(binary_sha256)
        addr = parse_address(func_addr)
        try:
            record: Optional[FunctionRecord] = self.db.get_function(sha, addr)
        except StorageError as exc:
            raise QueryError(f"query failed: {exc}") from exc
        if record is None:
            return None

        pcode, pcode_trunc = _truncate_lines(record.ir_pcode, max_lines)
        decomp_c, decomp_trunc = _truncate_lines(record.decompiled_c, max_lines)
        profile_addr = int(record.func_addr)

        return {
            "binary_sha256": record.binary_sha256,
            "func_addr": profile_addr,
            "addr_hex": _hex_addr(profile_addr),
            "name": record.name,
            "size": int(record.size),
            "calling_convention": record.calling_convention,
            "prototype": record.prototype,
            "ir_pcode": pcode,
            "decompiled_c": decomp_c,
            "has_ir_pcode": bool(record.ir_pcode),
            "has_decompiled_c": bool(record.decompiled_c),
            "truncated": pcode_trunc or decomp_trunc,
            # P1: {sample_count, alloc_bytes, hotness} when a func_profile row
            # exists for this function, else None -- no schema break for callers
            # whose store carries no profile import.
            "profile": self._profile_row(sha, profile_addr),
        }

    # -- def-use -----------------------------------------------------------

    def _ssa_facts(self, binary_sha256: str, func_addr: int) -> List[Dict[str, Any]]:
        try:
            facts = self.db.get_ssa_facts(binary_sha256, func_addr)
        except StorageError as exc:
            raise QueryError(f"query failed: {exc}") from exc
        # Storage already orders by (def_point, fact_id); re-sort defensively so
        # this module owns its determinism guarantee.
        return sorted(facts, key=lambda f: (int(f["def_point"]), str(f["fact_id"])))

    def def_use(
        self,
        binary_sha256: Optional[str],
        func_addr: Union[int, str],
        var_name: Optional[str] = None,
        limit: Optional[int] = DEFAULT_LIMIT,
    ) -> List[Dict[str, Any]]:
        """Def-use chains (SSA facts) inside ``func_addr``, optionally for one register.

        ``var_name`` matches the base register (``"rax"`` matches ``"rax"`` and
        ``"rax_1"``) or an exact SSA register name. Results are capped at ``limit``.
        """
        sha = self.resolve_sha(binary_sha256)
        addr = parse_address(func_addr)
        budget = self._limit(limit)
        facts = self._ssa_facts(sha, addr)
        if var_name:
            needle = str(var_name).strip()
            facts = [f for f in facts if _fact_matches(f, needle)]
        items: List[Dict[str, Any]] = []
        for fact in facts[:budget]:
            def_point = int(fact["def_point"])
            use_points = [int(u) for u in fact["use_points"]]
            items.append(
                {
                    "fact_id": fact["fact_id"],
                    "binary_sha256": sha,
                    "func_addr": addr,
                    "func_addr_hex": _hex_addr(addr),
                    "var_name": fact["var_name"],
                    "ssa_reg": fact["ssa_reg"],
                    "def_point": def_point,
                    "def_hex": _hex_addr(def_point) if def_point >= 0 else None,
                    "use_points": use_points,
                    "use_hex": [_hex_addr(u) for u in use_points],
                    "use_count": len(use_points),
                }
            )
        return items

    # -- program slicing ---------------------------------------------------

    def slice(
        self,
        binary_sha256: Optional[str],
        func_addr: Union[int, str],
        target: Optional[str] = None,
        direction: Union[SliceDirection, str] = SliceDirection.BACKWARD,
        max_depth: int = 10,
        limit: Optional[int] = DEFAULT_LIMIT,
        *,
        target_reg: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Intra-function program slice over SSA def-use facts (budgeted).

        ``target`` (alias ``target_reg``, per ``docs/design-document.md`` line 67)
        names the register to slice from; ``direction="backward"`` walks from a
        sink towards its producers (def side), ``"forward"`` walks from a source
        towards its consumers (use side). Traversal is capped at ``max_depth``
        hops and the returned step list at ``limit`` entries.

        Deterministic and idempotent: the same arguments always yield the same
        ordered list (steps sorted by ``(depth, def_point, fact_id)``).
        """
        sha = self.resolve_sha(binary_sha256)
        addr = parse_address(func_addr)
        budget = self._limit(limit)
        hop_cap = self._max_depth(max_depth)
        route = SliceDirection.coerce(direction)
        seed_name = target if target is not None else target_reg
        needle = str(seed_name).strip() if seed_name is not None else None

        facts = self._ssa_facts(sha, addr)
        if not facts:
            return []
        if needle:
            seeds = [f for f in facts if _fact_matches(f, needle)]
        else:
            seeds = list(facts)
        if not seeds:
            return []

        frontier: List[Dict[str, Any]] = sorted(
            seeds, key=lambda f: (int(f["def_point"]), str(f["fact_id"]))
        )
        seen = {str(f["fact_id"]) for f in frontier}
        records: List[Tuple[int, int, str, Dict[str, Any]]] = []

        for depth in range(hop_cap + 1):
            if not frontier:
                break
            for fact in frontier:
                records.append(
                    (depth, int(fact["def_point"]), str(fact["fact_id"]), fact)
                )
                if len(records) >= budget:
                    break
            if len(records) >= budget:
                break
            discovered: List[Dict[str, Any]] = []
            for fact in frontier:
                if route is SliceDirection.BACKWARD:
                    # def-side: who produced the value this fact defines?
                    anchors = {int(fact["def_point"])}
                    linked = [
                        f
                        for f in facts
                        if str(f["fact_id"]) not in seen
                        and any(int(u) in anchors for u in f["use_points"])
                    ]
                else:
                    # use-side: who consumes the value this fact defines?
                    anchors = {int(u) for u in fact["use_points"]}
                    linked = [
                        f
                        for f in facts
                        if str(f["fact_id"]) not in seen
                        and int(f["def_point"]) in anchors
                    ]
                linked.sort(key=lambda f: (int(f["def_point"]), str(f["fact_id"])))
                for item in linked:
                    seen.add(str(item["fact_id"]))
                    discovered.append(item)
            frontier = discovered

        records.sort(key=lambda r: (r[0], r[1], r[2]))
        steps: List[Dict[str, Any]] = []
        for depth, def_point, fact_id, fact in records[:budget]:
            use_points = [int(u) for u in fact["use_points"]]
            steps.append(
                {
                    "fact_id": fact_id,
                    "binary_sha256": sha,
                    "func_addr": addr,
                    "func_addr_hex": _hex_addr(addr),
                    "var_name": fact["var_name"],
                    "ssa_reg": fact["ssa_reg"],
                    "def_point": def_point,
                    "def_hex": _hex_addr(def_point) if def_point >= 0 else None,
                    "use_points": use_points,
                    "use_hex": [_hex_addr(u) for u in use_points],
                    "depth": depth,
                    "direction": route.value,
                    "target": needle,
                }
            )
        return steps
