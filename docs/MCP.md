# Toothpick MCP Server Guide

`toothpick mcp` exposes `toothpick.db` to AI coding agents over the Model Context Protocol (MCP) using stdio JSON-RPC 2.0.

## Why Phase-Constrained MCP?

Raw disassembler dumps and unbounded cross-reference lists quickly exhaust an LLM's context window. Toothpick's MCP server enforces three invariants on every tool call:

1. **Database-backed queries only**: Every tool reads pre-indexed rows from `toothpick.db` — no on-the-fly full-binary disassembly during a tool call.
2. **Hard top-K list bounds**: Every list-returning tool defaults to `limit = 50` and enforces a hard ceiling of `1,000` items.
3. **Line-bounded code output**: Function IR and decompiled C listings are capped at `max_lines = 200` with an explicit `[TRUNCATED]` marker when cut.

---

## Starting the Server

### Built-in stdio loop (default, zero extra runtime dependencies)

```bash
uv run toothpick mcp --db /path/to/toothpick.db
```

### Official `mcp` Python SDK transport

```bash
uv run toothpick mcp --db /path/to/toothpick.db --sdk
```

Both modes speak newline-delimited JSON-RPC 2.0 on `stdin`/`stdout` and register the exact same 9 tools. If `toothpick.db` contains multiple binaries, you can pin a default binary SHA-256 at server startup with `--sha <binary_sha256>`.

---

## Client Configuration

### Claude Desktop / Cursor (`mcp.json`)

```json
{
  "mcpServers": {
    "toothpick": {
      "command": "uv",
      "args": [
        "--directory",
        "C:/path/to/toothpick",
        "run",
        "toothpick",
        "mcp",
        "--db",
        "C:/path/to/toothpick.db"
      ]
    }
  }
}
```

### Polymorph (`config.toml`)

```toml
[[mcp_servers]]
id = "toothpick"
cmd = "uv"
args = ["--directory", "C:/path/to/toothpick", "run", "toothpick", "mcp", "--db", "C:/path/to/toothpick.db"]
```

---

## Tool Reference

All tools accept `func_addr` as either an integer (`5368713216`) or a hex string (`"0x140001000"`). `binary_sha256` is optional whenever `toothpick.db` holds a single binary.

### 1. `toothpick_callers`

Returns the functions that call `func_addr`, sorted deterministically by caller address.

**Arguments:**
- `func_addr` (`integer | string`, **required**)
- `binary_sha256` (`string`, optional)
- `limit` (`integer`, default `50`, max `1000`)

### 2. `toothpick_callees`

Returns the functions called by `func_addr`, sorted deterministically by callee address.

**Arguments:**
- `func_addr` (`integer | string`, **required**)
- `binary_sha256` (`string`, optional)
- `limit` (`integer`, default `50`, max `1000`)

### 3. `toothpick_def_use`

Returns SSA reaching-definition chains (`var_name`, `def_point`, `use_points`) inside `func_addr`.

**Arguments:**
- `func_addr` (`integer | string`, **required**)
- `var_name` (`string`, optional — e.g. `"rax"`, `"rcx"`)
- `binary_sha256` (`string`, optional)
- `limit` (`integer`, default `50`, max `1000`)

### 4. `toothpick_slice`

Computes a backward (sink → producers) or forward (source → consumers) intra-function program slice over SSA def-use edges.

**Arguments:**
- `func_addr` (`integer | string`, **required**)
- `target` (`string`, **required** — register or SSA variable name)
- `direction` (`"backward" | "forward"`, default `"backward"`)
- `max_depth` (`integer`, default `10`, max `64`)
- `binary_sha256` (`string`, optional)
- `limit` (`integer`, default `50`, max `1000`)

### 5. `toothpick_get_func`

Fetches the stored function record (name, prototype, calling convention, size, basic-block count, IR p-code, and decompiled C text).

**Arguments:**
- `func_addr` (`integer | string`, **required**)
- `binary_sha256` (`string`, optional)
- `max_lines` (`integer`, default `200`, max `1000`)

### 6. `toothpick_rebase`

Translates `func_addr` from `old_sha256` to `new_sha256` using a stored version-rebase mapping in `toothpick.db`.

**Arguments:**
- `old_sha256` (`string`, **required**)
- `new_sha256` (`string`, **required**)
- `func_addr` (`integer | string`, **required**)

### 7. `toothpick_show`

Returns the complete debug inspection card for `func_addr`: function metadata, callers, callees, SSA facts, verification receipts (`GROUNDED` / `UNGROUNDED`), and dynamic trace events.

**Arguments:**
- `func_addr` (`integer | string`, **required**)
- `binary_sha256` (`string`, optional)
- `limit` (`integer`, default `50`, max `1000`)

### 8. `toothpick_ledger`

Returns token and monetary cost accounting aggregated by model class (`Orchestrator`, `Solver`, `Bulk`, `Custom`).

**Arguments:**
- `events_log` (`string`, optional — path to JSONL events log to sync before summarizing)

### 9. `toothpick_hottest`

Returns the top-N highest-sample functions from imported ETW/WPR or JFR/async-profiler captures joined against `toothpick.db`.

**Arguments:**
- `binary_sha256` (`string`, optional)
- `top_n` (`integer`, default `50`, max `1000`)
