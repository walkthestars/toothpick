"""
Toothpick MCP (Model Context Protocol) server over stdio (W04).

Binding contract: ``PLAN-W04-mcp.md`` lines 19-26 (done criteria 2) and
``docs/design-document.md`` sections 11-12: the server exposes the *phase
constrained* query subset -- callers, callees, def-use, slice, and one function
lookup -- never a flat 200-tool dump, and every list-shaped answer is capped by
the query engine's top-K budget (Risk R1: token blowup).

Two transports share one tool implementation:

* :meth:`ToothpickMcpServer.serve_stdio` -- a self-contained newline-delimited
  JSON-RPC 2.0 loop (``initialize``, ``tools/list``, ``tools/call``, ``ping``,
  ``resources/list``, ``prompts/list``). That is exactly the MCP stdio framing
  (one JSON-RPC message per line) and needs no third-party runtime, so it works
  under ``click.testing.CliRunner`` and inside a pipe.
* :attr:`ToothpickMcpServer.sdk` -- the official ``mcp`` SDK server
  (``mcp.server.MCPServer``) with the same five tools registered, selected by
  ``toothpick mcp --sdk`` for clients that want the full SDK stack.

A ``tools/call`` response is ``{"content": [{"type": "text", "text": "<json>"}],
"isError": false}``; :meth:`ToothpickMcpServer.call_tool` returns the raw JSON
payload instead, for in-process callers and tests.
"""

from __future__ import annotations

import asyncio
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, Tuple, Union

from toothpick.query import (
    DEFAULT_LIMIT,
    MAX_DEPTH,
    MAX_LIMIT,
    QueryEngine,
    QueryError,
    SliceDirection,
    parse_address,
)
from toothpick.storage import Database

__all__ = [
    "SERVER_NAME",
    "SERVER_VERSION",
    "PROTOCOL_VERSION",
    "JSONRPC_VERSION",
    "McpToolError",
    "McpMethodNotFound",
    "TOOL_SPECS",
    "TOOL_NAMES",
    "ToothpickMcpServer",
    "create_mcp_server",
]

SERVER_NAME = "toothpick"
SERVER_VERSION = "0.1.0"
JSONRPC_VERSION = "2.0"

try:  # pragma: no cover - depends on the installed mcp release
    from mcp.types import LATEST_PROTOCOL_VERSION as _SDK_PROTOCOL_VERSION
except Exception:  # pragma: no cover - SDK absent or symbol renamed
    _SDK_PROTOCOL_VERSION = ""

#: MCP protocol revision advertised in ``initialize`` (SDK's latest when known).
PROTOCOL_VERSION = _SDK_PROTOCOL_VERSION or "2025-06-18"

DbLike = Union[Database, str, Path]

_ADDR_SCHEMA = {
    "type": ["integer", "string"],
    "description": "Function address (int, or hex string such as '0x1000')",
}
_SHA_SCHEMA = {
    "type": "string",
    "description": (
        "SHA-256 of the analysed binary. Optional: inferred when the database "
        "holds exactly one binary."
    ),
}
_LIMIT_SCHEMA = {
    "type": "integer",
    "minimum": 1,
    "maximum": MAX_LIMIT,
    "default": DEFAULT_LIMIT,
    "description": f"Top-K budget for the response (default {DEFAULT_LIMIT})",
}

#: Top-K budget for ``toothpick_hottest`` (mirrors ``profile.DEFAULT_TOP_N``).
_TOP_N_SCHEMA = {
    "type": "integer",
    "minimum": 1,
    "maximum": MAX_LIMIT,
    "default": DEFAULT_LIMIT,
    "description": (
        "Number of hottest functions to return, hottest first "
        f"(default {DEFAULT_LIMIT}, ceiling {MAX_LIMIT})"
    ),
}

_ENGINE_NOTE = (
    "Answers come only from the toothpick.db graph (phase-constrained: no "
    "re-disassembly, no unbounded lists)."
)

#: The phase-constrained tool surface (PLAN-W04-mcp.md lines 21-25).
TOOL_SPECS: Tuple[Dict[str, Any], ...] = (
    {
        "name": "toothpick_callers",
        "description": (
            "List the functions that call func_addr (call-graph predecessors, "
            f"top-K, default {DEFAULT_LIMIT}). {_ENGINE_NOTE}"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "binary_sha256": _SHA_SCHEMA,
                "func_addr": _ADDR_SCHEMA,
                "limit": _LIMIT_SCHEMA,
            },
            "required": ["func_addr"],
            "additionalProperties": False,
        },
    },
    {
        "name": "toothpick_callees",
        "description": (
            "List the functions called by func_addr (call-graph successors, "
            f"top-K, default {DEFAULT_LIMIT}). {_ENGINE_NOTE}"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "binary_sha256": _SHA_SCHEMA,
                "func_addr": _ADDR_SCHEMA,
                "limit": _LIMIT_SCHEMA,
            },
            "required": ["func_addr"],
            "additionalProperties": False,
        },
    },
    {
        "name": "toothpick_def_use",
        "description": (
            "Return the SSA def-use chains inside func_addr, optionally filtered "
            f"to one register/SSA variable (top-K, default {DEFAULT_LIMIT}). {_ENGINE_NOTE}"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "binary_sha256": _SHA_SCHEMA,
                "func_addr": _ADDR_SCHEMA,
                "var_name": {
                    "type": "string",
                    "description": "Register or SSA variable name, e.g. 'rax' or 'rax_1'",
                },
                "limit": _LIMIT_SCHEMA,
            },
            "required": ["func_addr"],
            "additionalProperties": False,
        },
    },
    {
        "name": "toothpick_slice",
        "description": (
            "Intra-function program slice over SSA def-use facts: backward from a "
            "sink, or forward from a source, with depth and top-K caps "
            f"(default {DEFAULT_LIMIT}). {_ENGINE_NOTE}"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "binary_sha256": _SHA_SCHEMA,
                "func_addr": _ADDR_SCHEMA,
                "target": {
                    "type": "string",
                    "description": "Register/SSA variable to slice from, e.g. 'rax'",
                },
                "direction": {
                    "type": "string",
                    "enum": ["backward", "forward"],
                    "default": "backward",
                    "description": (
                        "backward = from sink to producers, "
                        "forward = from source to consumers"
                    ),
                },
                "max_depth": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": MAX_DEPTH,
                    "default": 10,
                    "description": "Maximum number of def-use hops to traverse",
                },
                "limit": _LIMIT_SCHEMA,
            },
            "required": ["func_addr"],
            "additionalProperties": False,
        },
    },
    {
        "name": "toothpick_get_func",
        "description": (
            "Fetch one analysed function record (name, prototype, size, IR "
            f"p-code, decompiled C). {_ENGINE_NOTE}"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "binary_sha256": _SHA_SCHEMA,
                "func_addr": _ADDR_SCHEMA,
                "max_lines": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 200,
                    "description": "Maximum number of lines to return for decompiled C and IR p-code (default 200)",
                },
            },
            "required": ["func_addr"],
            "additionalProperties": False,
        },
    },
    {
        "name": "toothpick_rebase",
        "description": (
            "Rebase function offsets across binary updates using differential "
            "analysis and multi-heuristic similarity scoring (sim >= 0.99 auto-ported). "
            f"{_ENGINE_NOTE}"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "old_sha256": {
                    "type": "string",
                    "description": "Baseline binary SHA-256 to rebase from",
                },
                "new_sha256": {
                    "type": "string",
                    "description": "Target updated binary SHA-256",
                },
                "threshold": {
                    "type": "number",
                    "default": 0.99,
                    "description": "Auto-port similarity threshold (default 0.99)",
                },
                "limit": _LIMIT_SCHEMA,
            },
            "required": ["old_sha256", "new_sha256"],
            "additionalProperties": False,
        },
    },
    {
        "name": "toothpick_show",
        "description": (
            "Render comprehensive function debug overlay (AST, decompiled C pseudocode, "
            f"CFG edges, SSA facts, receipts, and type state) per Design §14. {_ENGINE_NOTE}"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "binary_sha256": _SHA_SCHEMA,
                "func_addr": _ADDR_SCHEMA,
                "format": {
                    "type": "string",
                    "enum": ["text", "json"],
                    "default": "text",
                    "description": "Output format: text or json",
                },
            },
            "required": ["func_addr"],
            "additionalProperties": False,
        },
    },
    {
        "name": "toothpick_ledger",
        "description": (
            "Query token and cost observability ledger aggregated per model class and "
            f"pipeline phase per Design §14, Risk R6, and TBM-2. {_ENGINE_NOTE}"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "binary_sha256": _SHA_SCHEMA,
                "events_log": {
                    "type": "string",
                    "description": "Optional path to JSONL events.log file",
                },
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "toothpick_hottest",
        "description": (
            "List the hottest functions of the binary by sampled-profile hotness "
            "(VTune/ETW/perf imports), ordered hotness descending with "
            f"func_addr as the tie-break, top-K, default {DEFAULT_LIMIT}. {_ENGINE_NOTE}"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "binary_sha256": _SHA_SCHEMA,
                "top_n": _TOP_N_SCHEMA,
            },
            "required": [],
            "additionalProperties": False,
        },
    },
)

TOOL_NAMES: Tuple[str, ...] = tuple(spec["name"] for spec in TOOL_SPECS)

_TOOL_PROPERTIES: Dict[str, Tuple[str, ...]] = {
    spec["name"]: tuple(spec["inputSchema"]["properties"]) for spec in TOOL_SPECS
}


class McpToolError(Exception):
    """Unknown tool, or arguments the tool cannot accept (JSON-RPC -32602)."""


class McpMethodNotFound(Exception):
    """Unsupported JSON-RPC method (JSON-RPC -32601)."""


def _error_response(msg_id: Any, code: int, message: str) -> Dict[str, Any]:
    return {
        "jsonrpc": JSONRPC_VERSION,
        "id": msg_id,
        "error": {"code": code, "message": message},
    }


def _require_addr(args: Dict[str, Any], key: str = "func_addr") -> int:
    if key not in args or args[key] is None or args[key] == "":
        raise McpToolError(f"missing required argument: {key}")
    try:
        return parse_address(args[key])
    except QueryError as exc:
        raise McpToolError(str(exc)) from exc


class ToothpickMcpServer:
    """MCP server exposing the toothpick query engine over stdio.

    Transport-agnostic: :meth:`list_tools` / :meth:`call_tool` are the tool
    surface, :meth:`handle_message` is the MCP protocol layer, and
    :meth:`serve_stdio` drives it over stdin/stdout.
    """

    #: :meth:`serve` uses the built-in loop unless this is switched on.
    _prefer_sdk = False

    def __init__(
        self,
        db: DbLike,
        *,
        name: str = SERVER_NAME,
        version: str = SERVER_VERSION,
        use_sdk: bool = True,
    ) -> None:
        self.name = name
        self.version = version
        self.engine = QueryEngine(db)
        self._sdk: Any = None
        if use_sdk:
            self._sdk = self._build_sdk_server()

    # -- tool catalogue ----------------------------------------------------

    def list_tools(self) -> List[Dict[str, Any]]:
        """Tool descriptors (``name``/``description``/``inputSchema``), deep-copied."""
        return [copy.deepcopy(dict(spec)) for spec in TOOL_SPECS]

    def tool_names(self) -> List[str]:
        """Names of every exposed tool, in catalogue order."""
        return list(TOOL_NAMES)

    # -- tool dispatch -----------------------------------------------------

    def call_tool(
        self, name: str, arguments: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Run one tool and return its raw JSON-serialisable payload.

        Raises :class:`McpToolError` for an unknown tool or bad arguments.
        """
        if name not in TOOL_NAMES:
            raise McpToolError(f"unknown tool: {name!r}")
        args = dict(arguments or {})
        unknown = set(args) - set(_TOOL_PROPERTIES[name])
        if unknown:
            raise McpToolError(
                f"unexpected arguments for {name}: {', '.join(sorted(unknown))}"
            )
        try:
            if name == "toothpick_callers":
                return self._payload_callers(args)
            if name == "toothpick_callees":
                return self._payload_callees(args)
            if name == "toothpick_def_use":
                return self._payload_def_use(args)
            if name == "toothpick_slice":
                return self._payload_slice(args)
            if name == "toothpick_rebase":
                return self._payload_rebase(args)
            if name == "toothpick_show":
                return self._payload_show(args)
            if name == "toothpick_ledger":
                return self._payload_ledger(args)
            if name == "toothpick_hottest":
                return self._payload_hottest(args)
            return self._payload_get_func(args)
        except QueryError as exc:
            raise McpToolError(str(exc)) from exc
        except (TypeError, ValueError) as exc:
            raise McpToolError(f"invalid arguments for {name}: {exc}") from exc

    def _payload_show(self, args: Dict[str, Any]) -> Dict[str, Any]:
        from toothpick.observability import FunctionInspector, InspectionError
        addr = _require_addr(args)
        sha_arg = args.get("binary_sha256")
        fmt = str(args.get("format", "text")).lower()
        try:
            with FunctionInspector(db=self.engine.db) as inspector:
                overlay = inspector.inspect(addr, binary_sha256=sha_arg)
                return {
                    "tool": "toothpick_show",
                    "binary_sha256": overlay.binary_sha256,
                    "func_addr": overlay.func_addr,
                    "func_addr_hex": f"0x{overlay.func_addr:x}",
                    "format": fmt,
                    "overlay": overlay.to_dict() if fmt == "json" else overlay.format_text(),
                }
        except InspectionError as exc:
            raise McpToolError(str(exc)) from exc

    def _payload_ledger(self, args: Dict[str, Any]) -> Dict[str, Any]:
        from toothpick.observability import ObservabilityLedger, ObservabilityError
        sha_arg = args.get("binary_sha256")
        events_log = args.get("events_log")
        try:
            with ObservabilityLedger(db=self.engine.db, events_log_path=events_log) as obs:
                summary = obs.get_summary(binary_sha256=sha_arg)
                return {
                    "tool": "toothpick_ledger",
                    "summary": summary.to_dict(),
                    "table": summary.format_table(),
                }
        except ObservabilityError as exc:
            raise McpToolError(str(exc)) from exc

    def _payload_rebase(self, args: Dict[str, Any]) -> Dict[str, Any]:
        from toothpick.rebase import RebaseEngine, RebaseError
        old_sha = args.get("old_sha256")
        new_sha = args.get("new_sha256")
        if not old_sha or not new_sha:
            raise McpToolError("old_sha256 and new_sha256 are required for toothpick_rebase")
        threshold = float(args.get("threshold", 0.99))
        limit = args.get("limit", DEFAULT_LIMIT)
        try:
            with RebaseEngine(db=self.engine.db) as rebase_engine:
                report = rebase_engine.rebase(old_sha256=old_sha, new_target=new_sha, threshold=threshold)
                matches = [m.to_dict() for m in report.matches[:limit]]
                return {
                    "tool": "toothpick_rebase",
                    "old_sha256": report.old_sha256,
                    "new_sha256": report.new_sha256,
                    "total_old_funcs": report.total_old_funcs,
                    "total_new_funcs": report.total_new_funcs,
                    "auto_ported": report.auto_ported,
                    "review_required": report.review_required,
                    "unmatched": report.unmatched,
                    "auto_port_rate": round(report.auto_port_rate, 4),
                    "limit": self.engine.resolve_limit(limit),
                    "count": len(matches),
                    "truncated": len(matches) < len(report.matches),
                    "matches": matches,
                    "offset_map": report.to_dict()["offset_map"],
                }
        except RebaseError as exc:
            raise McpToolError(str(exc)) from exc

    def _payload_callers(self, args: Dict[str, Any]) -> Dict[str, Any]:
        sha_arg = args.get("binary_sha256")
        addr = _require_addr(args)
        limit = args.get("limit", DEFAULT_LIMIT)
        items = self.engine.callers(sha_arg, addr, limit)
        sha = self.engine.resolve_sha(sha_arg)
        total = self.engine.count_callers(sha, addr)
        return {
            "tool": "toothpick_callers",
            "binary_sha256": sha,
            "func_addr": addr,
            "func_addr_hex": f"0x{addr:x}",
            "limit": self.engine.resolve_limit(limit),
            "count": len(items),
            "total": total,
            "truncated": len(items) < total,
            "callers": items,
        }

    def _payload_callees(self, args: Dict[str, Any]) -> Dict[str, Any]:
        sha_arg = args.get("binary_sha256")
        addr = _require_addr(args)
        limit = args.get("limit", DEFAULT_LIMIT)
        items = self.engine.callees(sha_arg, addr, limit)
        sha = self.engine.resolve_sha(sha_arg)
        total = self.engine.count_callees(sha, addr)
        return {
            "tool": "toothpick_callees",
            "binary_sha256": sha,
            "func_addr": addr,
            "func_addr_hex": f"0x{addr:x}",
            "limit": self.engine.resolve_limit(limit),
            "count": len(items),
            "total": total,
            "truncated": len(items) < total,
            "callees": items,
        }

    def _payload_hottest(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Bounded hottest-function list, hotness-DESC (Risk R1).

        Thin wrapper over :meth:`QueryEngine.hottest`, which in turn delegates to
        :func:`toothpick.profile.hottest` over this engine's database: a profile
        imported through any P1 format is reachable here without touching the
        analysis state.  The list is capped by the query-engine top-K budget and
        ordered ``hotness DESC, func_addr ASC``, so the same call returns the
        same list (no unbounded or order-unstable response).
        """
        sha_arg = args.get("binary_sha256")
        top_n = args.get("top_n", DEFAULT_LIMIT)
        items = self.engine.hottest(sha_arg, top_n)
        sha = self.engine.resolve_sha(sha_arg)
        return {
            "tool": "toothpick_hottest",
            "binary_sha256": sha,
            "top_n": self.engine.resolve_limit(top_n),
            "count": len(items),
            "hottest": items,
        }

    def _payload_def_use(self, args: Dict[str, Any]) -> Dict[str, Any]:
        sha_arg = args.get("binary_sha256")
        addr = _require_addr(args)
        var_name = args.get("var_name")
        limit = args.get("limit", DEFAULT_LIMIT)
        facts = self.engine.def_use(sha_arg, addr, var_name, limit)
        sha = self.engine.resolve_sha(sha_arg)
        return {
            "tool": "toothpick_def_use",
            "binary_sha256": sha,
            "func_addr": addr,
            "func_addr_hex": f"0x{addr:x}",
            "var_name": var_name,
            "limit": self.engine.resolve_limit(limit),
            "count": len(facts),
            "facts": facts,
        }

    def _payload_slice(self, args: Dict[str, Any]) -> Dict[str, Any]:
        sha_arg = args.get("binary_sha256")
        addr = _require_addr(args)
        target = args.get("target", args.get("target_reg"))
        direction = SliceDirection.coerce(args.get("direction", SliceDirection.BACKWARD))
        max_depth = args.get("max_depth", 10)
        limit = args.get("limit", DEFAULT_LIMIT)
        steps = self.engine.slice(sha_arg, addr, target, direction, max_depth, limit)
        sha = self.engine.resolve_sha(sha_arg)
        return {
            "tool": "toothpick_slice",
            "binary_sha256": sha,
            "func_addr": addr,
            "func_addr_hex": f"0x{addr:x}",
            "target": target,
            "direction": direction.value,
            "max_depth": self.engine.resolve_max_depth(max_depth),
            "limit": self.engine.resolve_limit(limit),
            "count": len(steps),
            "slice": steps,
        }

    def _payload_get_func(self, args: Dict[str, Any]) -> Dict[str, Any]:
        sha_arg = args.get("binary_sha256")
        addr = _require_addr(args)
        max_lines = args.get("max_lines", 200)
        record = self.engine.get_function(sha_arg, addr, max_lines=max_lines)
        sha = self.engine.resolve_sha(sha_arg)
        return {
            "tool": "toothpick_get_func",
            "binary_sha256": sha,
            "func_addr": addr,
            "func_addr_hex": f"0x{addr:x}",
            "found": record is not None,
            "function": record,
        }

    # -- MCP protocol layer ------------------------------------------------

    def initialize_result(self, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """MCP ``initialize`` result (tools capability + server identity)."""
        del params  # we advertise our own protocol version
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": self.name, "version": self.version},
            "instructions": (
                "Phase-constrained binary queries over toothpick.db: callers, "
                "callees, def-use, program slice, function lookup. All list "
                f"responses are capped (default {DEFAULT_LIMIT})."
            ),
        }

    def _call_result(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        try:
            payload = self.call_tool(name, arguments)
        except McpToolError as exc:
            return {"content": [{"type": "text", "text": str(exc)}], "isError": True}
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return {"content": [{"type": "text", "text": text}], "isError": False}

    def _handle_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if method == "initialize":
            return self.initialize_result(params)
        if method == "ping":
            return {}
        if method == "tools/list":
            return {"tools": self.list_tools()}
        if method == "tools/call":
            name = params.get("name")
            if not name:
                raise McpToolError("tools/call requires a tool name")
            if name not in TOOL_NAMES:
                raise McpToolError(f"unknown tool: {name!r}")
            arguments = params.get("arguments")
            if arguments is None:
                arguments = {}
            if not isinstance(arguments, dict):
                raise McpToolError("tools/call arguments must be an object")
            return self._call_result(str(name), arguments)
        if method == "resources/list":
            return {"resources": []}
        if method == "resources/templates/list":
            return {"resourceTemplates": []}
        if method == "prompts/list":
            return {"prompts": []}
        raise McpMethodNotFound(method)

    def handle_message(self, message: Any) -> Optional[Dict[str, Any]]:
        """Handle one JSON-RPC message; ``None`` for notifications (no reply)."""
        if not isinstance(message, dict):
            return _error_response(None, -32600, "invalid request: expected a JSON object")
        msg_id = message.get("id")
        method = message.get("method")
        if method is None:
            return None  # a response to a server-initiated request: ignore it
        if msg_id is None:
            return None  # notification (e.g. notifications/initialized)
        params = message.get("params")
        if params is None:
            params = {}
        if not isinstance(params, dict):
            return _error_response(msg_id, -32602, "params must be an object")
        try:
            result = self._handle_request(str(method), params)
        except McpMethodNotFound as exc:
            return _error_response(msg_id, -32601, f"method not found: {exc}")
        except McpToolError as exc:
            return _error_response(msg_id, -32602, str(exc))
        except QueryError as exc:  # pragma: no cover - defensive
            return _error_response(msg_id, -32603, str(exc))
        return {"jsonrpc": JSONRPC_VERSION, "id": msg_id, "result": result}

    # -- stdio transports --------------------------------------------------

    def serve_stdio(
        self, stdin: Optional[TextIO] = None, stdout: Optional[TextIO] = None
    ) -> None:
        """Newline-delimited JSON-RPC loop until EOF (the MCP stdio framing)."""
        if stdin is None and hasattr(sys.stdin, "reconfigure"):
            try:
                sys.stdin.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
        if stdout is None and hasattr(sys.stdout, "reconfigure"):
            try:
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
        source = stdin if stdin is not None else sys.stdin
        sink = stdout if stdout is not None else sys.stdout
        for line in source:
            text = line.strip()
            if not text:
                continue
            message: Any = None
            try:
                message = json.loads(text)
                response = self.handle_message(message)
            except json.JSONDecodeError as exc:
                response = _error_response(None, -32700, f"parse error: {exc.msg}")
            except Exception as exc:  # pragma: no cover - defensive
                msg_id = message.get("id") if isinstance(message, dict) else None
                response = _error_response(msg_id, -32603, f"internal error: {exc}")
            if response is not None:
                sink.write(json.dumps(response, separators=(",", ":")) + "\n")
                sink.flush()

    def serve(self) -> None:
        """Blocking entry point: built-in stdio loop, or the SDK when armed."""
        if self._prefer_sdk and self._sdk is not None:
            self.serve_sdk()
        else:
            self.serve_stdio()

    def serve_sdk(self) -> None:
        """Run the official ``mcp`` SDK stdio transport (requires the SDK)."""
        if self._sdk is None:
            raise McpToolError("mcp SDK is not installed; use serve_stdio() instead")
        asyncio.run(self.run_stdio())

    async def run_stdio(self) -> None:
        """Async entry point: the SDK transport when available, else the loop."""
        if self._sdk is None:
            await asyncio.to_thread(self.serve_stdio)
            return
        await self._sdk.run_stdio_async()

    def use_sdk(self, enabled: bool = True) -> None:
        """Arm :meth:`serve` to use the SDK transport on its next call."""
        self._prefer_sdk = bool(enabled)

    # -- official SDK bridge ----------------------------------------------

    def _build_sdk_server(self) -> Any:
        """Register the same five tools on ``mcp.server.MCPServer`` (or ``None``)."""
        try:
            from mcp.server import MCPServer
        except Exception:  # pragma: no cover - SDK absent
            return None
        try:
            server = MCPServer(
                name=self.name,
                version=self.version,
                instructions=(
                    "Phase-constrained binary queries over toothpick.db "
                    "(callers, callees, def-use, slice, function lookup)."
                ),
            )

            def to_text(name: str, args: Dict[str, Any]) -> str:
                return json.dumps(self.call_tool(name, args), sort_keys=True)

            async def toothpick_callers(
                func_addr: Union[int, str],
                binary_sha256: Optional[str] = None,
                limit: int = DEFAULT_LIMIT,
            ) -> str:
                """List the functions that call func_addr (top-K)."""
                return to_text(
                    "toothpick_callers",
                    {
                        "func_addr": func_addr,
                        "binary_sha256": binary_sha256,
                        "limit": limit,
                    },
                )

            async def toothpick_callees(
                func_addr: Union[int, str],
                binary_sha256: Optional[str] = None,
                limit: int = DEFAULT_LIMIT,
            ) -> str:
                """List the functions called by func_addr (top-K)."""
                return to_text(
                    "toothpick_callees",
                    {
                        "func_addr": func_addr,
                        "binary_sha256": binary_sha256,
                        "limit": limit,
                    },
                )

            async def toothpick_def_use(
                func_addr: Union[int, str],
                binary_sha256: Optional[str] = None,
                var_name: Optional[str] = None,
                limit: int = DEFAULT_LIMIT,
            ) -> str:
                """Return the SSA def-use chains inside func_addr (top-K)."""
                return to_text(
                    "toothpick_def_use",
                    {
                        "func_addr": func_addr,
                        "binary_sha256": binary_sha256,
                        "var_name": var_name,
                        "limit": limit,
                    },
                )

            async def toothpick_slice(
                func_addr: Union[int, str],
                binary_sha256: Optional[str] = None,
                target: Optional[str] = None,
                direction: str = "backward",
                max_depth: int = 10,
                limit: int = DEFAULT_LIMIT,
            ) -> str:
                """Slice a function backward from a sink or forward from a source."""
                return to_text(
                    "toothpick_slice",
                    {
                        "func_addr": func_addr,
                        "binary_sha256": binary_sha256,
                        "target": target,
                        "direction": direction,
                        "max_depth": max_depth,
                        "limit": limit,
                    },
                )

            async def toothpick_get_func(
                func_addr: Union[int, str],
                binary_sha256: Optional[str] = None,
                max_lines: int = 200,
            ) -> str:
                """Fetch one analysed function record."""
                return to_text(
                    "toothpick_get_func",
                    {
                        "func_addr": func_addr,
                        "binary_sha256": binary_sha256,
                        "max_lines": max_lines,
                    },
                )

            async def toothpick_rebase(
                old_sha256: str,
                new_sha256: str,
                threshold: float = 0.99,
                limit: int = DEFAULT_LIMIT,
            ) -> str:
                """Rebase function offsets across binary updates."""
                return to_text(
                    "toothpick_rebase",
                    {
                        "old_sha256": old_sha256,
                        "new_sha256": new_sha256,
                        "threshold": threshold,
                        "limit": limit,
                    },
                )

            async def toothpick_show(
                func_addr: Union[int, str],
                binary_sha256: Optional[str] = None,
                format: str = "text",
            ) -> str:
                """Render function debug overlay."""
                return to_text(
                    "toothpick_show",
                    {
                        "func_addr": func_addr,
                        "binary_sha256": binary_sha256,
                        "format": format,
                    },
                )

            async def toothpick_ledger(
                binary_sha256: Optional[str] = None,
                events_log: Optional[str] = None,
            ) -> str:
                """Query token and cost observability ledger."""
                return to_text(
                    "toothpick_ledger",
                    {
                        "binary_sha256": binary_sha256,
                        "events_log": events_log,
                    },
                )

            async def toothpick_hottest(
                binary_sha256: Optional[str] = None,
                top_n: int = DEFAULT_LIMIT,
            ) -> str:
                """List the hottest functions by sampled-profile hotness."""
                return to_text(
                    "toothpick_hottest",
                    {
                        "binary_sha256": binary_sha256,
                        "top_n": top_n,
                    },
                )

            handlers = {
                "toothpick_callers": toothpick_callers,
                "toothpick_callees": toothpick_callees,
                "toothpick_def_use": toothpick_def_use,
                "toothpick_slice": toothpick_slice,
                "toothpick_get_func": toothpick_get_func,
                "toothpick_rebase": toothpick_rebase,
                "toothpick_show": toothpick_show,
                "toothpick_ledger": toothpick_ledger,
                "toothpick_hottest": toothpick_hottest,
            }
            for spec in TOOL_SPECS:
                server.add_tool(
                    handlers[spec["name"]],
                    name=spec["name"],
                    description=spec["description"],
                )
            return server
        except Exception:  # pragma: no cover - SDK API drift must not break us
            return None

    @property
    def sdk(self) -> Any:
        """The official ``mcp.server.MCPServer`` when available, else ``None``."""
        return self._sdk

    def __getattr__(self, item: str) -> Any:
        # Delegate anything else (run_stdio_async, sse_app, ...) to the SDK server.
        sdk = self.__dict__.get("_sdk")
        if sdk is not None and hasattr(sdk, item):
            return getattr(sdk, item)
        raise AttributeError(item)


def create_mcp_server(db: DbLike, *, use_sdk: bool = True) -> ToothpickMcpServer:
    """Build the toothpick MCP server bound to ``db`` (a path or :class:`Database`)."""
    return ToothpickMcpServer(db, use_sdk=use_sdk)
