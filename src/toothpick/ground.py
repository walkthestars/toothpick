"""
Toothpick dynamic grounding harness using headless Frida (unit W06).

Implements the W06 grounding contract per UNIT-W06-GROUND-PLAN.md,
design-document.md §11, ADR-06, Risk R3 and Risk R5:
- Live process attachment and target execution spawning via headless Frida scripts.
- Real-time CPU register context extraction and stack frame reading at function addresses.
- Mandatory `--static-only` fallback gate: when running in headless CI or offline environments
  without an emulator or VM, emits a deterministic static fallback receipt
  (ground_type="STATIC_FALLBACK", verifier_status="UNGROUNDED") without failing.
- Persistence of dynamic and static verification receipts directly into toothpick.db.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from uuid import uuid4

from toothpick.storage import Database

__all__ = [
    "GroundError",
    "FridaAttachError",
    "GroundTimeoutError",
    "GroundReceipt",
    "FridaRunner",
    "GroundingHarness",
]

PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class GroundError(Exception):
    """Base exception for grounding failures."""


class FridaAttachError(GroundError):
    """Raised when Frida cannot attach to or spawn a target process."""


class GroundTimeoutError(GroundError):
    """Raised when execution does not hit the targeted breakpoint within timeout."""


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class GroundReceipt:
    """Execution receipt persisted in the SQLite receipts table."""

    receipt_id: str
    binary_sha256: str
    func_addr: int
    ground_type: str  # "FRIDA_LIVE" | "STATIC_FALLBACK"
    timestamp: float
    live_reads: Dict[str, Any] = field(default_factory=dict)
    verifier_status: str = "GROUNDED"  # "GROUNDED" | "UNGROUNDED"
    phase: str = "ground"

    def to_row(self) -> Tuple[str, str, int, str, str, str]:
        """Serialize into a SQLite receipts table row."""
        payload = {
            "ground_type": self.ground_type,
            "verifier_status": self.verifier_status,
            "live_reads": self.live_reads,
        }
        return (
            self.receipt_id,
            self.binary_sha256,
            int(self.func_addr),
            self.phase,
            str(self.timestamp),
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert into a JSON-ready dictionary."""
        return {
            "receipt_id": self.receipt_id,
            "binary_sha256": self.binary_sha256,
            "func_addr": self.func_addr,
            "func_addr_hex": f"0x{self.func_addr:x}",
            "ground_type": self.ground_type,
            "timestamp": self.timestamp,
            "live_reads": self.live_reads,
            "verifier_status": self.verifier_status,
            "phase": self.phase,
        }

    def to_json(self) -> str:
        """Format as indented JSON."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str)

    @classmethod
    def from_row(cls, row: Any) -> GroundReceipt:
        """Construct a GroundReceipt from a sqlite3.Row or storage dict."""
        if isinstance(row, dict) and "payload" in row and isinstance(row["payload"], dict):
            payload = row["payload"]
        else:
            payload_str = (
                row["payload_json"]
                if (isinstance(row, dict) and "payload_json" in row)
                or (hasattr(row, "keys") and "payload_json" in row.keys())
                else "{}"
            )
            payload = json.loads(payload_str or "{}") if isinstance(payload_str, str) else (payload_str or {})
        raw_ts = row["timestamp"]
        try:
            ts = float(raw_ts)
        except (ValueError, TypeError):
            ts = time.time()
        phase = row["phase"] if (hasattr(row, "keys") and "phase" in row.keys()) or (isinstance(row, dict) and "phase" in row) else "ground"
        return cls(
            receipt_id=row["receipt_id"],
            binary_sha256=row["binary_sha256"],
            func_addr=int(row["func_addr"]),
            ground_type=payload.get("ground_type", "STATIC_FALLBACK"),
            timestamp=ts,
            live_reads=payload.get("live_reads", {}),
            verifier_status=payload.get("verifier_status", "UNGROUNDED"),
            phase=phase,
        )


# ---------------------------------------------------------------------------
# Frida Script Generation & Runner
# ---------------------------------------------------------------------------


def _generate_frida_script(addr: int) -> str:
    """Headless Frida script that traps on entry to addr, captures context, and reports."""
    return f"""
    const targetAddr = ptr("0x{addr:x}");
    Interceptor.attach(targetAddr, {{
        onEnter: function(args) {{
            const regs = {{}};
            const ctx = this.context;
            const regNames = [
                'rax', 'rbx', 'rcx', 'rdx', 'rsi', 'rdi', 'rbp', 'rsp',
                'r8', 'r9', 'r10', 'r11', 'r12', 'r13', 'r14', 'r15', 'rip',
                'eax', 'ebx', 'ecx', 'edx', 'esp', 'ebp', 'esi', 'edi', 'eip',
                'x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8',
                'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16',
                'x17', 'x18', 'x19', 'x20', 'x21', 'x22', 'x23', 'x24',
                'x25', 'x26', 'x27', 'x28', 'x29', 'x30', 'sp', 'pc', 'lr'
            ];
            for (let i = 0; i < regNames.length; i++) {{
                const r = regNames[i];
                if (ctx[r] !== undefined) {{
                    regs[r] = ctx[r].toString();
                }}
            }}
            const sp = ctx.sp || ctx.rsp || ctx.esp;
            const stackReads = [];
            if (sp) {{
                try {{
                    for (let i = 0; i < 4; i++) {{
                        stackReads.push(sp.add(i * Process.pointerSize).readPointer().toString());
                    }}
                }} catch (e) {{}}
            }}
            send({{
                status: "HIT",
                registers: regs,
                stack: stackReads
            }});
        }}
    }});
    """


class FridaRunner:
    """Manages headless Frida session lifecycle and message dispatching."""

    def __init__(self, frida_module: Optional[Any] = None) -> None:
        self._frida = frida_module

    def _get_frida(self) -> Any:
        if self._frida is not None:
            return self._frida
        try:
            import frida

            self._frida = frida
            return frida
        except ImportError as exc:
            raise FridaAttachError("frida module is not installed; use --static-only") from exc

    def attach_and_trap(
        self, pid: int, addr: int, timeout: float = 5.0
    ) -> Dict[str, Any]:
        """Attach to running PID, set breakpoint at addr, and return captured live reads."""
        frida = self._get_frida()
        try:
            session = frida.attach(pid)
        except Exception as exc:
            raise FridaAttachError(f"Cannot attach to PID {pid}: {exc}") from exc

        return self._trap_session(session, addr, timeout)

    def spawn_and_trap(
        self, target_path: str, addr: int, timeout: float = 5.0
    ) -> Dict[str, Any]:
        """Spawn target executable suspended, attach, set breakpoint, resume, and return reads."""
        frida = self._get_frida()
        pid: Optional[int] = None
        try:
            pid = frida.spawn(target_path)
            session = frida.attach(pid)
        except Exception as exc:
            if pid is not None:
                try:
                    frida.kill(pid)
                except Exception:
                    pass
            raise FridaAttachError(f"Cannot spawn {target_path}: {exc}") from exc

        try:
            res = self._trap_session(session, addr, timeout, on_ready=lambda: frida.resume(pid))
        finally:
            if pid is not None:
                try:
                    frida.kill(pid)
                except Exception:
                    pass
        return res

    def _trap_session(
        self,
        session: Any,
        addr: int,
        timeout: float,
        on_ready: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """Load interceptor script and await breakpoint trigger within timeout."""
        hit_event = threading.Event()
        captured_data: Dict[str, Any] = {}

        def on_message(message: Dict[str, Any], _data: Any) -> None:
            if message.get("type") == "send":
                payload = message.get("payload", {})
                if payload.get("status") == "HIT":
                    captured_data["registers"] = payload.get("registers", {})
                    captured_data["stack"] = payload.get("stack", [])
                    hit_event.set()

        js_source = _generate_frida_script(addr)
        try:
            script = session.create_script(js_source)
            script.on("message", on_message)
            script.load()
        except Exception as exc:
            try:
                session.detach()
            except Exception:
                pass
            raise FridaAttachError(f"Frida script compilation failed: {exc}") from exc

        if on_ready:
            on_ready()

        signaled = hit_event.wait(timeout=max(timeout, 0.001))
        try:
            script.unload()
        except Exception:
            pass
        try:
            session.detach()
        except Exception:
            pass

        if not signaled:
            raise GroundTimeoutError(
                f"Breakpoint at 0x{addr:x} was not hit within {timeout:.1f} seconds"
            )

        return captured_data


# ---------------------------------------------------------------------------
# Grounding Harness (GroundingHarness)
# ---------------------------------------------------------------------------


class GroundingHarness:
    """Coordinates dynamic execution grounding and offline static fallback."""

    def __init__(
        self,
        db: Optional[Union[Database, PathLike]] = None,
        binary_sha256: Optional[str] = None,
        frida_runner: Optional[FridaRunner] = None,
    ) -> None:
        self.db: Optional[Database] = None
        self._owned_db = False
        if db is not None:
            if isinstance(db, Database):
                self.db = db
            else:
                self.db = Database(Path(db), create=False)
                self._owned_db = True

        self.binary_sha256 = binary_sha256
        self.runner = frida_runner or FridaRunner()

    def close(self) -> None:
        if self._owned_db and self.db is not None:
            self.db.close()
            self.db = None

    def __enter__(self) -> GroundingHarness:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _resolve_sha(self) -> str:
        """Resolve the target binary SHA-256 used for receipt attribution.

        Inference is only permitted when the database holds exactly one binary.
        An ambiguous database (multiple distinct binaries) raises
        :class:`GroundError` rather than picking an arbitrary row, so a receipt
        is never attributed to the wrong binary (ADR-06); an empty database
        yields "" (receipt carries no binary attribution).
        """
        if self.binary_sha256:
            return self.binary_sha256
        if self.db is not None:
            rows = self.db._conn.execute("SELECT DISTINCT binary_sha256 FROM funcs").fetchall()
            shas = [r["binary_sha256"] for r in rows if r["binary_sha256"]]
            if len(shas) == 1:
                return shas[0]
            if len(shas) > 1:
                raise GroundError(
                    "Cannot infer binary SHA-256: database contains "
                    f"{len(shas)} distinct binaries ({', '.join(sorted(shas))}); "
                    "pass an explicit --sha"
                )
        return ""

    def attach_and_read(
        self, pid: int, addr: int, timeout: float = 5.0
    ) -> GroundReceipt:
        """Attach to running process, capture registers/stack at addr, and persist receipt."""
        reads = self.runner.attach_and_trap(pid, addr, timeout=timeout)
        sha = self._resolve_sha()
        receipt = GroundReceipt(
            receipt_id=f"rec_{uuid4().hex[:12]}",
            binary_sha256=sha,
            func_addr=addr,
            ground_type="FRIDA_LIVE",
            timestamp=time.time(),
            live_reads=reads,
            verifier_status="GROUNDED",
            phase="ground",
        )
        if self.db is not None:
            self.db.insert_receipt(receipt)
        return receipt

    def spawn_and_read(
        self, target_path: str, addr: int, timeout: float = 5.0
    ) -> GroundReceipt:
        """Spawn executable, capture registers/stack at addr, and persist receipt."""
        reads = self.runner.spawn_and_trap(target_path, addr, timeout=timeout)
        sha = self._resolve_sha()
        receipt = GroundReceipt(
            receipt_id=f"rec_{uuid4().hex[:12]}",
            binary_sha256=sha,
            func_addr=addr,
            ground_type="FRIDA_LIVE",
            timestamp=time.time(),
            live_reads=reads,
            verifier_status="GROUNDED",
            phase="ground",
        )
        if self.db is not None:
            self.db.insert_receipt(receipt)
        return receipt

    def fallback_read(
        self, addr: int, db_path: Optional[str] = None
    ) -> GroundReceipt:
        """Emit deterministic static fallback receipt without requiring VM or live process."""
        sha = self._resolve_sha()
        static_reads: Dict[str, Any] = {
            "mode": "STATIC_FALLBACK",
            "reason": "Static-only gate requested or live VM unavailable (ADR-06 / Risk R5)",
        }

        # Enrich with static facts if present in the database
        if self.db is not None:
            func = self.db.get_function(sha, addr)
            if func is not None:
                static_reads["name"] = func.name
                static_reads["size"] = func.size
                static_reads["prototype"] = func.prototype
                static_reads["calling_convention"] = func.calling_convention
            ssa = self.db.get_ssa_facts(sha, addr)
            if ssa:
                static_reads["ssa_variables"] = [f["var_name"] for f in ssa[:10]]

        receipt = GroundReceipt(
            receipt_id=f"rec_{uuid4().hex[:12]}",
            binary_sha256=sha,
            func_addr=addr,
            ground_type="STATIC_FALLBACK",
            timestamp=time.time(),
            live_reads=static_reads,
            verifier_status="UNGROUNDED",
            phase="ground",
        )
        if self.db is not None:
            self.db.insert_receipt(receipt)
        return receipt

    def get_receipts(
        self, func_addr: Optional[int] = None, phase: Optional[str] = "ground"
    ) -> List[GroundReceipt]:
        """Fetch stored ground receipts from database."""
        if self.db is None:
            return []
        sha = self._resolve_sha()
        raw_receipts = self.db.get_receipts(sha, func_addr)
        if phase is not None:
            raw_receipts = [r for r in raw_receipts if r.get("phase") == phase]
        return [GroundReceipt.from_row(r) for r in raw_receipts]
