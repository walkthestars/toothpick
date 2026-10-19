"""
Toothpick Token & Cost Observability Ledger and Debug Overlay (W10).

Governing specs:
- design-document.md §14 (Observability: events.log pipeline trace + token/cost
  ledger per model class + toothpick show <addr> overlay; no telemetry leaves host).
- design-document.md §1, §9 (Model roles: orchestrator, solver, bulk worker; TBM-2 cost model).
- design-document.md §17 (Risk R6: LLM cost overrun -> token ledger + phase caps).
- execution-plan.md §1 (W10: Token/cost ledger + events.log + toothpick show).
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from toothpick.query import parse_address
from toothpick.storage import Database

__all__ = [
    "ModelClass",
    "TokenUsage",
    "ModelPricing",
    "CostEstimate",
    "TraceEvent",
    "LedgerSummary",
    "FunctionOverlay",
    "ObservabilityError",
    "InspectionError",
    "ObservabilityLedger",
    "FunctionInspector",
    "DEFAULT_PRICING",
]

PathLike = Union[str, Path]
DbLike = Union[Database, str, Path]


class ObservabilityError(Exception):
    """Base error for observability and ledger operations."""


class InspectionError(ObservabilityError):
    """Raised when inspecting a function fails or target is not found."""


class ModelClass(str, Enum):
    """Architectural model roles defined in Design §1."""

    ORCHESTRATOR = "orchestrator"  # Muse Spark 1.3 class: long-thread, high reasoning
    SOLVER = "solver"              # GLM-5.3 / Astra class: constraint solving, high precision
    BULK = "bulk"                  # DeepSeek V4.1 Flash class: cheap prefill, 1M context bulk
    CUSTOM = "custom"              # User-defined model class

    @classmethod
    def from_string(cls, value: Union["ModelClass", str]) -> "ModelClass":
        if isinstance(value, cls):
            return value
        text = str(value).strip().lower()
        for member in cls:
            if member.value == text:
                return member
        return cls.CUSTOM


@dataclass
class TokenUsage:
    """Prompt, completion, and cache token metrics."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> Dict[str, int]:
        return {
            "prompt_tokens": int(self.prompt_tokens),
            "completion_tokens": int(self.completion_tokens),
            "cached_tokens": int(self.cached_tokens),
            "total_tokens": int(self.total_tokens),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TokenUsage":
        return cls(
            prompt_tokens=int(data.get("prompt_tokens", 0)),
            completion_tokens=int(data.get("completion_tokens", 0)),
            cached_tokens=int(data.get("cached_tokens", 0)),
        )


@dataclass
class CostEstimate:
    """Estimated cost in USD for token consumption."""

    prompt_cost: float = 0.0
    completion_cost: float = 0.0

    @property
    def total_cost(self) -> float:
        return self.prompt_cost + self.completion_cost

    def to_dict(self) -> Dict[str, float]:
        return {
            "prompt_cost": round(self.prompt_cost, 6),
            "completion_cost": round(self.completion_cost, 6),
            "total_cost": round(self.total_cost, 6),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CostEstimate":
        return cls(
            prompt_cost=float(data.get("prompt_cost", 0.0)),
            completion_cost=float(data.get("completion_cost", 0.0)),
        )


@dataclass
class ModelPricing:
    """Rates in USD per 1,000,000 (1M) tokens."""

    prompt_per_1m: float
    completion_per_1m: float

    def calculate_cost(self, usage: TokenUsage) -> CostEstimate:
        p_cost = (usage.prompt_tokens / 1_000_000.0) * self.prompt_per_1m
        c_cost = (usage.completion_tokens / 1_000_000.0) * self.completion_per_1m
        return CostEstimate(prompt_cost=p_cost, completion_cost=c_cost)


#: Default pricing lookup table per 1M tokens based on Design §9 / §14 reference anchors
DEFAULT_PRICING: Dict[str, ModelPricing] = {
    # Model classes
    ModelClass.ORCHESTRATOR.value: ModelPricing(prompt_per_1m=3.00, completion_per_1m=15.00),
    ModelClass.SOLVER.value: ModelPricing(prompt_per_1m=2.00, completion_per_1m=6.00),
    ModelClass.BULK.value: ModelPricing(prompt_per_1m=0.14, completion_per_1m=0.28),
    ModelClass.CUSTOM.value: ModelPricing(prompt_per_1m=1.00, completion_per_1m=2.00),
    # Specific model identifiers
    "spark-1.3": ModelPricing(prompt_per_1m=3.00, completion_per_1m=15.00),
    "claude-3-5-sonnet": ModelPricing(prompt_per_1m=3.00, completion_per_1m=15.00),
    "glm-5.3": ModelPricing(prompt_per_1m=2.00, completion_per_1m=6.00),
    "astra": ModelPricing(prompt_per_1m=5.00, completion_per_1m=10.00),
    "deepseek-v4.1-flash": ModelPricing(prompt_per_1m=0.14, completion_per_1m=0.28),
    "deepseek-chat": ModelPricing(prompt_per_1m=0.14, completion_per_1m=0.28),
    "gpt-4o": ModelPricing(prompt_per_1m=2.50, completion_per_1m=10.00),
    "gpt-4o-mini": ModelPricing(prompt_per_1m=0.15, completion_per_1m=0.60),
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TraceEvent:
    """Single pipeline trace or LLM invocation event."""

    event_id: str
    timestamp: str
    binary_sha256: str = ""
    phase: str = "system"
    func_addr: Optional[int] = None
    event_type: str = "pipeline"
    model_class: Optional[str] = None
    model_name: Optional[str] = None
    token_usage: Optional[TokenUsage] = None
    cost_estimate: Optional[CostEstimate] = None
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "binary_sha256": self.binary_sha256,
            "phase": self.phase,
            "func_addr": self.func_addr,
            "func_addr_hex": f"0x{self.func_addr:x}" if self.func_addr is not None else None,
            "event_type": self.event_type,
            "model_class": self.model_class,
            "model_name": self.model_name,
            "token_usage": self.token_usage.to_dict() if self.token_usage else None,
            "cost_estimate": self.cost_estimate.to_dict() if self.cost_estimate else None,
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TraceEvent":
        tokens_dict = data.get("token_usage")
        cost_dict = data.get("cost_estimate")
        return cls(
            event_id=data["event_id"],
            timestamp=data.get("timestamp", _utc_now_iso()),
            binary_sha256=data.get("binary_sha256", ""),
            phase=data.get("phase", "system"),
            func_addr=data.get("func_addr"),
            event_type=data.get("event_type", "pipeline"),
            model_class=data.get("model_class"),
            model_name=data.get("model_name"),
            token_usage=TokenUsage.from_dict(tokens_dict) if tokens_dict else None,
            cost_estimate=CostEstimate.from_dict(cost_dict) if cost_dict else None,
            payload=data.get("payload", {}),
        )


@dataclass
class LedgerSummary:
    """Aggregated token and cost metrics across model classes and phases."""

    binary_sha256: Optional[str]
    total_events: int
    total_tokens: int
    total_prompt_tokens: int
    total_completion_tokens: int
    total_cached_tokens: int
    total_cost: float
    by_model_class: Dict[str, Dict[str, Any]]
    by_phase: Dict[str, Dict[str, Any]]
    by_model: Dict[str, Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "binary_sha256": self.binary_sha256,
            "total_events": self.total_events,
            "total_tokens": self.total_tokens,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_cached_tokens": self.total_cached_tokens,
            "total_cost": round(self.total_cost, 6),
            "by_model_class": self.by_model_class,
            "by_phase": self.by_phase,
            "by_model": self.by_model,
        }

    def format_table(self) -> str:
        lines: List[str] = []
        lines.append("================================================================================")
        lines.append("TOOTHPICK OBSERVABILITY LEDGER SUMMARY")
        if self.binary_sha256:
            lines.append(f"Target Binary SHA-256: {self.binary_sha256}")
        lines.append(f"Total Events: {self.total_events} | Total Tokens: {self.total_tokens:,} | Total Cost: ${self.total_cost:.4f}")
        lines.append("--------------------------------------------------------------------------------")
        lines.append(f"{'MODEL CLASS':<16} | {'EVENTS':<8} | {'PROMPT':<10} | {'COMPLETION':<10} | {'COST ($)':<10}")
        lines.append("--------------------------------------------------------------------------------")
        for cls_name, stats in sorted(self.by_model_class.items()):
            lines.append(
                f"{cls_name.upper():<16} | {stats['events']:<8} | {stats['prompt_tokens']:<10,}"
                f" | {stats['completion_tokens']:<10,} | ${stats['cost']:<10.4f}"
            )
        lines.append("--------------------------------------------------------------------------------")
        lines.append(f"{'PIPELINE PHASE':<16} | {'EVENTS':<8} | {'TOTAL TOKENS':<14} | {'COST ($)':<10}")
        lines.append("--------------------------------------------------------------------------------")
        for phase_name, stats in sorted(self.by_phase.items()):
            lines.append(
                f"{phase_name:<16} | {stats['events']:<8} | {stats['tokens']:<14,}"
                f" | ${stats['cost']:<10.4f}"
            )
        lines.append("================================================================================")
        return "\n".join(lines)


class ObservabilityLedger:
    """Engine managing events.log and SQLite trace_events synchronization."""

    def __init__(
        self,
        db: Optional[DbLike] = None,
        *,
        events_log_path: Optional[PathLike] = None,
        log_to_file: bool = True,
        log_to_db: bool = True,
        pricing: Optional[Dict[str, ModelPricing]] = None,
    ) -> None:
        self._lock = threading.Lock()
        self.log_to_file = log_to_file
        self.log_to_db = log_to_db
        self.pricing = dict(DEFAULT_PRICING)
        if pricing:
            self.pricing.update(pricing)

        # File log location
        if events_log_path is not None:
            self.events_log_path = Path(events_log_path)
        else:
            env_path = os.environ.get("TOOTHPICK_EVENTS_LOG")
            self.events_log_path = Path(env_path) if env_path else Path("events.log")

        # Database management
        self._owned_db: Optional[Database] = None
        self.db: Optional[Database] = None
        if db is not None:
            if isinstance(db, Database):
                self.db = db
            else:
                self._owned_db = Database(db)
                self.db = self._owned_db

    def __enter__(self) -> "ObservabilityLedger":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close owned database connection if any."""
        if self._owned_db is not None:
            self._owned_db.close()
            self._owned_db = None
            self.db = None

    def resolve_pricing(self, model_name: Optional[str], model_class: Optional[str]) -> ModelPricing:
        """Find appropriate pricing rates for a given model or class."""
        if model_name and model_name.lower() in self.pricing:
            return self.pricing[model_name.lower()]
        cls_key = ModelClass.from_string(model_class or "custom").value
        if cls_key in self.pricing:
            return self.pricing[cls_key]
        return self.pricing[ModelClass.CUSTOM.value]

    def record_event(self, event: TraceEvent) -> str:
        """Record an event to events.log (JSONL) and SQLite trace_events."""
        event_dict = event.to_dict()

        # 1. File log append (JSONL)
        if self.log_to_file and self.events_log_path:
            with self._lock:
                try:
                    self.events_log_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(self.events_log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(event_dict, separators=(",", ":")) + "\n")
                except Exception as exc:
                    # Invariant: file logging failures must not crash caller if DB succeeds
                    if not self.log_to_db or self.db is None:
                        raise ObservabilityError(f"Failed writing to {self.events_log_path}: {exc}") from exc

        # 2. SQLite trace_events dual-write
        if self.log_to_db and self.db is not None:
            try:
                # Prepare payload containing full event telemetry for faithful SQLite storage
                db_func_addr = int(event.func_addr) if event.func_addr is not None else -1
                db_payload = {
                    "phase": event.phase,
                    "func_addr": event.func_addr,
                    "model_class": event.model_class,
                    "model_name": event.model_name,
                    "token_usage": event.token_usage.to_dict() if event.token_usage else None,
                    "cost_estimate": event.cost_estimate.to_dict() if event.cost_estimate else None,
                    "payload": event.payload,
                }
                self.db.add_trace_event(
                    binary_sha256=event.binary_sha256,
                    func_addr=db_func_addr,
                    event_type=event.event_type,
                    payload=db_payload,
                    timestamp=event.timestamp,
                    event_id=event.event_id,
                )
            except Exception as exc:
                raise ObservabilityError(f"Failed writing event to database: {exc}") from exc

        return event.event_id

    def record_llm_call(
        self,
        model_name: str,
        model_class: Union[ModelClass, str],
        phase: str,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int = 0,
        binary_sha256: str = "",
        func_addr: Optional[int] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> TraceEvent:
        """Record an LLM API call with token usage and cost accounting."""
        m_cls = ModelClass.from_string(model_class).value
        usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
        )
        pricing = self.resolve_pricing(model_name, m_cls)
        cost = pricing.calculate_cost(usage)

        event = TraceEvent(
            event_id=f"trac_{uuid4().hex[:12]}",
            timestamp=_utc_now_iso(),
            binary_sha256=binary_sha256,
            phase=phase,
            func_addr=func_addr,
            event_type="llm_call",
            model_class=m_cls,
            model_name=model_name,
            token_usage=usage,
            cost_estimate=cost,
            payload=payload if payload is not None else {},
        )
        self.record_event(event)
        return event

    def record_phase_event(
        self,
        phase: str,
        event_type: str,
        binary_sha256: str = "",
        func_addr: Optional[int] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> TraceEvent:
        """Record a pipeline stage milestone (phase_start, phase_end, receipt, etc.)."""
        event = TraceEvent(
            event_id=f"trac_{uuid4().hex[:12]}",
            timestamp=_utc_now_iso(),
            binary_sha256=binary_sha256,
            phase=phase,
            func_addr=func_addr,
            event_type=event_type,
            payload=payload if payload is not None else {},
        )
        self.record_event(event)
        return event

    def get_events(
        self,
        binary_sha256: Optional[str] = None,
        phase: Optional[str] = None,
        func_addr: Optional[int] = None,
        model_class: Optional[str] = None,
        limit: int = 1_000,
    ) -> List[TraceEvent]:
        """Fetch trace events from DB if available, falling back to events.log."""
        events: List[TraceEvent] = []

        # Try reading from SQLite DB first
        if self.db is not None:
            if binary_sha256:
                target_shas = [binary_sha256]
            else:
                sha_rows = self.db._conn.execute(
                    "SELECT DISTINCT binary_sha256 FROM trace_events WHERE binary_sha256 != ''"
                ).fetchall()
                target_shas = [r[0] for r in sha_rows if r[0]]
            if not target_shas:
                # Query without sha filter
                rows = self.db._conn.execute("SELECT * FROM trace_events ORDER BY timestamp, event_id").fetchall()
                for r in rows:
                    payload = json.loads(r["payload_json"]) if r["payload_json"] else {}
                    t_usage = TokenUsage.from_dict(payload["token_usage"]) if payload.get("token_usage") else None
                    c_est = CostEstimate.from_dict(payload["cost_estimate"]) if payload.get("cost_estimate") else None
                    events.append(
                        TraceEvent(
                            event_id=r["event_id"],
                            timestamp=r["timestamp"],
                            binary_sha256=r["binary_sha256"],
                            func_addr=payload.get("func_addr") if "func_addr" in payload else (int(r["func_addr"]) if r["func_addr"] >= 0 else None),
                            event_type=r["event_type"],
                            phase=payload.get("phase", "system"),
                            model_class=payload.get("model_class"),
                            model_name=payload.get("model_name"),
                            token_usage=t_usage,
                            cost_estimate=c_est,
                            payload=payload.get("payload", {}),
                        )
                    )
            else:
                for sha in target_shas:
                    db_events = self.db.get_trace_events(sha, func_addr=func_addr)
                    for item in db_events:
                        payload = item["payload"]
                        t_usage = TokenUsage.from_dict(payload["token_usage"]) if payload.get("token_usage") else None
                        c_est = CostEstimate.from_dict(payload["cost_estimate"]) if payload.get("cost_estimate") else None
                        events.append(
                            TraceEvent(
                                event_id=item["event_id"],
                                timestamp=item["timestamp"],
                                binary_sha256=item["binary_sha256"],
                                func_addr=payload.get("func_addr") if "func_addr" in payload else (int(item["func_addr"]) if item["func_addr"] >= 0 else None),
                                event_type=item["event_type"],
                                phase=payload.get("phase", "system"),
                                model_class=payload.get("model_class"),
                                model_name=payload.get("model_name"),
                                token_usage=t_usage,
                                cost_estimate=c_est,
                                payload=payload.get("payload", {}),
                            )
                        )
        elif self.events_log_path and self.events_log_path.is_file():
            # Fall back to reading JSONL events.log
            with open(self.events_log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        events.append(TraceEvent.from_dict(data))
                    except Exception:
                        continue

        # Filter events
        filtered: List[TraceEvent] = []
        for e in events:
            if binary_sha256 and e.binary_sha256 and e.binary_sha256 != binary_sha256:
                continue
            if phase and e.phase != phase:
                continue
            if func_addr is not None and e.func_addr != func_addr:
                continue
            if model_class and e.model_class != model_class:
                continue
            filtered.append(e)
            if len(filtered) >= limit:
                break

        return filtered

    def get_summary(self, binary_sha256: Optional[str] = None) -> LedgerSummary:
        """Aggregate token usage and monetary cost by model class and phase."""
        events = self.get_events(binary_sha256=binary_sha256, limit=100_000)

        total_tokens = 0
        total_prompt = 0
        total_completion = 0
        total_cached = 0
        total_cost = 0.0

        by_class: Dict[str, Dict[str, Any]] = {
            cls.value: {
                "events": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cached_tokens": 0,
                "cost": 0.0,
            }
            for cls in ModelClass
        }
        by_phase: Dict[str, Dict[str, Any]] = {}
        by_model: Dict[str, Dict[str, Any]] = {}

        for e in events:
            p_tokens = e.token_usage.prompt_tokens if e.token_usage else 0
            c_tokens = e.token_usage.completion_tokens if e.token_usage else 0
            k_tokens = e.token_usage.cached_tokens if e.token_usage else 0
            e_tokens = p_tokens + c_tokens
            e_cost = e.cost_estimate.total_cost if e.cost_estimate else 0.0

            total_tokens += e_tokens
            total_prompt += p_tokens
            total_completion += c_tokens
            total_cached += k_tokens
            total_cost += e_cost

            # Group by model class
            cls_name = e.model_class or ModelClass.CUSTOM.value
            if cls_name not in by_class:
                by_class[cls_name] = {
                    "events": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "cached_tokens": 0,
                    "cost": 0.0,
                }
            by_class[cls_name]["events"] += 1
            by_class[cls_name]["prompt_tokens"] += p_tokens
            by_class[cls_name]["completion_tokens"] += c_tokens
            by_class[cls_name]["total_tokens"] += e_tokens
            by_class[cls_name]["cached_tokens"] += k_tokens
            by_class[cls_name]["cost"] = round(by_class[cls_name]["cost"] + e_cost, 6)

            # Group by phase
            phase = e.phase or "system"
            if phase not in by_phase:
                by_phase[phase] = {"events": 0, "tokens": 0, "cost": 0.0}
            by_phase[phase]["events"] += 1
            by_phase[phase]["tokens"] += e_tokens
            by_phase[phase]["cost"] = round(by_phase[phase]["cost"] + e_cost, 6)

            # Group by model name
            m_name = e.model_name or "none"
            if m_name not in by_model:
                by_model[m_name] = {"class": cls_name, "events": 0, "tokens": 0, "cost": 0.0}
            by_model[m_name]["events"] += 1
            by_model[m_name]["tokens"] += e_tokens
            by_model[m_name]["cost"] = round(by_model[m_name]["cost"] + e_cost, 6)

        return LedgerSummary(
            binary_sha256=binary_sha256,
            total_events=len(events),
            total_tokens=total_tokens,
            total_prompt_tokens=total_prompt,
            total_completion_tokens=total_completion,
            total_cached_tokens=total_cached,
            total_cost=round(total_cost, 6),
            by_model_class=by_class,
            by_phase=by_phase,
            by_model=by_model,
        )


@dataclass
class FunctionOverlay:
    """Complete function inspection record for toothpick show overlay."""

    func_addr: int
    binary_sha256: str
    name: str
    size: int
    calling_convention: str
    prototype: str
    decompiled_c: Optional[str]
    ir_pcode: Optional[str]
    callers: List[int]
    callees: List[int]
    cfg_edges: List[Dict[str, Any]]
    ssa_facts: List[Dict[str, Any]]
    receipts: List[Dict[str, Any]]
    types: List[Dict[str, Any]]
    trace_events: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "func_addr": self.func_addr,
            "func_addr_hex": f"0x{self.func_addr:x}",
            "binary_sha256": self.binary_sha256,
            "name": self.name,
            "size": self.size,
            "calling_convention": self.calling_convention,
            "prototype": self.prototype,
            "decompiled_c": self.decompiled_c,
            "ir_pcode": self.ir_pcode,
            "callers": [f"0x{c:x}" for c in self.callers],
            "callees": [f"0x{c:x}" for c in self.callees],
            "cfg_edges": self.cfg_edges,
            "ssa_facts_count": len(self.ssa_facts),
            "ssa_facts": self.ssa_facts,
            "receipts": self.receipts,
            "types": self.types,
            "trace_events": self.trace_events,
        }

    def format_text(self) -> str:
        """Format as clean, readable debug overlay text."""
        lines: List[str] = []
        lines.append("================================================================================")
        lines.append(f"TOOTHPICK FUNCTION DEBUG OVERLAY: 0x{self.func_addr:x} ({self.name or 'unnamed'})")
        lines.append(f"Binary SHA-256: {self.binary_sha256}")
        lines.append(
            f"Size: {self.size} bytes | Calling Conv: {self.calling_convention or 'default'}"
            f" | Prototype: {self.prototype or 'void ()'}"
        )
        lines.append("================================================================================")

        # Call Graph & CFG
        lines.append("\n[CONTROL FLOW & CALL GRAPH]")
        callers_str = ", ".join(f"0x{c:x}" for c in self.callers) if self.callers else "none"
        callees_str = ", ".join(f"0x{c:x}" for c in self.callees) if self.callees else "none"
        lines.append(f"  Incoming Callers ({len(self.callers)}): {callers_str}")
        lines.append(f"  Outgoing Callees ({len(self.callees)}): {callees_str}")
        if self.cfg_edges:
            lines.append(f"  CFG Edges ({len(self.cfg_edges)}):")
            for e in self.cfg_edges[:20]:
                lines.append(f"    0x{e['src_addr']:x} -> 0x{e['dst_addr']:x} ({e['kind']})")
            if len(self.cfg_edges) > 20:
                lines.append(f"    ... and {len(self.cfg_edges) - 20} more CFG edges")

        # SSA & Data Flow
        lines.append(f"\n[DATA FLOW & SSA FACTS] ({len(self.ssa_facts)} facts)")
        if self.ssa_facts:
            for fact in self.ssa_facts[:15]:
                var_name = fact.get("var_name", "")
                reg = fact.get("ssa_reg", "")
                def_pt = fact.get("def_point", -1)
                uses = fact.get("use_points", [])
                lines.append(
                    f"  {var_name:<10} (reg: {reg:<6}) def: 0x{def_pt:x} | uses: {[f'0x{u:x}' for u in uses]}"
                )
            if len(self.ssa_facts) > 15:
                lines.append(f"  ... and {len(self.ssa_facts) - 15} more SSA facts")
        else:
            lines.append("  (no SSA facts recorded)")

        # Code / IR
        lines.append("\n[DECOMPILED CODE / PSEUDOCODE]")
        if self.decompiled_c:
            for c_line in self.decompiled_c.strip().split("\n")[:30]:
                lines.append(f"  {c_line}")
            if len(self.decompiled_c.strip().split("\n")) > 30:
                lines.append("  ... [output truncated]")
        elif self.ir_pcode:
            lines.append("  (No decompiled C; normalized P-code IR:)")
            for ir_line in self.ir_pcode.strip().split("\n")[:20]:
                lines.append(f"    {ir_line}")
        else:
            lines.append("  (No decompilation or IR available)")

        # Receipts
        lines.append(f"\n[GROUNDING RECEIPTS] ({len(self.receipts)} receipts)")
        if self.receipts:
            for r in self.receipts:
                phase = r.get("phase", "unknown")
                ts = r.get("timestamp", "")
                r_id = r.get("receipt_id", "")
                lines.append(f"  [{phase}] ID: {r_id} at {ts}")
        else:
            lines.append("  (no dynamic or solver receipts recorded)")

        # Type State
        lines.append(f"\n[TYPE STATE] ({len(self.types)} types)")
        if self.types:
            for t in self.types[:5]:
                lines.append(f"  Type {t.get('name', 'unnamed')} ({t.get('kind', 'unknown')}, size: {t.get('size_bytes', 0)}B)")
        else:
            lines.append("  (no type records in lattice)")

        # Trace Events
        lines.append(f"\n[TRACE HISTORY] ({len(self.trace_events)} events)")
        if self.trace_events:
            for ev in self.trace_events[-10:]:
                ev_type = ev.get("event_type", "")
                ts = ev.get("timestamp", "")
                lines.append(f"  {ts} [{ev_type}]")
        else:
            lines.append("  (no trace events recorded)")

        lines.append("================================================================================")
        return "\n".join(lines)


class FunctionInspector:
    """Inspector for toothpick show overlay rendering."""

    def __init__(self, db: DbLike) -> None:
        self._owned_db: Optional[Database] = None
        if isinstance(db, Database):
            self.db = db
        else:
            self._owned_db = Database(db)
            self.db = self._owned_db

    def __enter__(self) -> "FunctionInspector":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def close(self) -> None:
        if self._owned_db is not None:
            self._owned_db.close()
            self._owned_db = None

    def resolve_sha(self, binary_sha256: Optional[str]) -> str:
        """Resolve binary SHA-256 or enforce unambiguous attribution (Trap 20)."""
        if binary_sha256:
            return str(binary_sha256)
        rows = self.db._conn.execute(
            "SELECT DISTINCT binary_sha256 FROM funcs WHERE binary_sha256 != '' ORDER BY binary_sha256"
        ).fetchall()
        shas = [str(r[0]) for r in rows if r[0]]
        if len(shas) == 1:
            return shas[0]
        if len(shas) > 1:
            raise InspectionError(
                f"Ambiguous binary SHA-256: database contains {len(shas)} binaries "
                f"({', '.join(shas)}); pass an explicit --sha"
            )
        return ""

    def inspect(
        self,
        func_addr: Union[int, str],
        binary_sha256: Optional[str] = None,
    ) -> FunctionOverlay:
        """Inspect all aspects of a function to produce FunctionOverlay."""
        addr = parse_address(func_addr)
        sha = self.resolve_sha(binary_sha256)
        if not sha:
            raise InspectionError("No binary analysis records found in database")

        func = self.db.get_function(sha, addr)
        if func is None:
            raise InspectionError(f"Function 0x{addr:x} not found in database for binary {sha}")

        # Callers & Callees
        callers = sorted(self.db.get_callers(sha, addr))
        callees = sorted(self.db.get_callees(sha, addr))

        # CFG Edges: get edges where src or dst is within the function range
        func_end = addr + max(func.size, 1)
        all_cfg = self.db.get_edges(sha, kind="cfg")
        func_cfg = [
            {
                "src_addr": e.src_addr,
                "dst_addr": e.dst_addr,
                "kind": str(e.kind.value if hasattr(e.kind, "value") else e.kind),
            }
            for e in all_cfg
            if (addr <= e.src_addr < func_end or addr <= e.dst_addr < func_end)
        ]

        # SSA Facts
        ssa_facts = self.db.get_ssa_facts(sha, addr)

        # Receipts
        receipts = self.db.get_receipts(sha, func_addr=addr)

        # Types
        types_records = self.db.get_types(sha)
        types_list = [
            {
                "type_id": t.type_id,
                "name": t.name,
                "kind": t.kind,
                "size_bytes": t.size_bytes,
                "layout": t.layout,
            }
            for t in types_records
        ]

        # Trace Events
        trace_events = self.db.get_trace_events(sha, func_addr=addr)

        return FunctionOverlay(
            func_addr=addr,
            binary_sha256=sha,
            name=func.name,
            size=func.size,
            calling_convention=func.calling_convention,
            prototype=func.prototype,
            decompiled_c=func.decompiled_c,
            ir_pcode=func.ir_pcode,
            callers=callers,
            callees=callees,
            cfg_edges=func_cfg,
            ssa_facts=ssa_facts,
            receipts=receipts,
            types=types_list,
            trace_events=trace_events,
        )
