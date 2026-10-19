<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white" alt="Python 3.11+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-AGPLv3-blue" alt="License: AGPL-3.0-only"></a>
  <img src="https://img.shields.io/badge/interface-CLI%20%2B%20MCP-brightgreen" alt="CLI + MCP">
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/badge/code%20style-ruff-000000.svg" alt="Code style: ruff"></a>
</p>

<h1 align="center">Toothpick</h1>

<p align="center">A deterministic, multi-architecture binary reverse-engineering engine, MCP server, and singleplayer modding toolbox.<br>
Static CFG/SSA lifting, SMT path solving, live runtime grounding, differential rebase, and bounded agent tools backed by SQLite.</p>

<p align="center">
  <a href="#quick-start">Quick start</a> ·
  <a href="#pipeline-overview">Pipeline</a> ·
  <a href="#cli-reference">CLI</a> ·
  <a href="#mcp-server">MCP</a> ·
  <a href="docs/CLI.md">Full CLI Reference</a> ·
  <a href="docs/WORKFLOWS.md">Workflows</a>
</p>

---

Toothpick turns binary executables, game containers, JVM archives, and profiler captures into a single queryable SQLite WAL graph (`toothpick.db`). It unpacks runtime stubs, lifts instructions across five CPU architectures (x86, x86_64, ARM64, PowerPC, MIPS), computes control-flow edges and reaching-definitions SSA facts, solves branch reachability with Z3, records verified runtime register/memory receipts, rebases hooks across version updates, and scaffolds mods for Unity IL2CPP, Unreal Engine, Godot, native C++ shared libraries, and Lua/Steam Workshop targets.

All query, scan, and Model Context Protocol (MCP) endpoints enforce strict top-K and line-count caps so AI coding agents can inspect large binaries without blowing out their context window.

## Quick start

### Prerequisites

- **OS**: Windows 10/11 (x64) or Linux (x86_64)
- **Python**: 3.11+ with [`uv`](https://docs.astral.sh/uv/)
- **Optional host tools**:
  - `.NET SDK 8.0+` — only required when compiling generated C# BepInEx plugins (`toothpick build --compile`)
  - `Ghidra 11+` (`GHIDRA_INSTALL_DIR`) — only required when requesting PyGhidra decompilation (`toothpick static --prefer-ghidra`)
  - `cdb.exe` (Windows Debugging Tools) — only required for live WinDbg/cdb commands (`toothpick cdb`)

### Installation

```bash
git clone https://github.com/walkthestars/toothpick.git
cd toothpick
uv sync --locked
```

### Core workflow in 60 seconds

```bash
# 1. Triage container format, architecture, game engine, and packer signatures
uv run toothpick triage target.exe

# 2. Unpack simple protection stubs (UPX NRV2B/D/E, XOR loaders) if flagged
uv run toothpick unpack packed_target.exe --out target.exe

# 3. Lift functions, basic blocks, CFG edges, and SSA facts into toothpick.db
uv run toothpick static target.exe --db toothpick.db

# 4. Query the call graph and SSA def-use chains under bounded budgets
uv run toothpick query --db toothpick.db callers 0x140001000 --limit 50
uv run toothpick query --db toothpick.db def_use 0x140001000 --var rax
uv run toothpick query --db toothpick.db slice 0x140001000 --target rax --direction backward

# 5. Solve branch reachability constraints to a target basic block via Z3
uv run toothpick solve 0x140001000 --target-block 0x140001080 --db toothpick.db

# 6. Ground live via Frida (--pid) or scaffold offline with --allow-unverified
uv run toothpick ground 0x140001000 --static-only --db toothpick.db
uv run toothpick hook build --target 0x140001000 --db toothpick.db --out ./my_hook --name SpeedHook --allow-unverified
```

## Pipeline overview

```text
  Target Binary / Container / JAR / Dump / IDA Extract / Profiler Capture
         │
         ├──► triage & unpack      Format/engine/packer detection, UPX & Unicorn OEP unpack
         ├──► static / jvm ingest  Capstone CFG + SSA lifting or JVM class/JAR call graph
         ├──► ingest-ida           Schema-v1 IDA Pro JSON import + name-conflict arbitration
         ├──► rtti / struct        MSVC & Itanium C++ RTTI/vtable recovery + ReClass export
         ├──► profile import       Windows ETW/WPR & JVM JFR/async-profiler hot-function ranking
         │
         ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │                     toothpick.db (SQLite WAL)                       │
  │  funcs · edges · ssa_facts · types · receipts · versions · profiles │
  └─────────────────────────────────────────────────────────────────────┘
         │
         ├──► query / show / mcp   Top-K bounded callers, callees, def-use, slicing, stdio MCP
         ├──► solve / taint        Z3 BitVec path reachability & dynamic taint tracking
         ├──► ground / scan / cdb  Frida, CEServer, WinDbg/cdb & DBI runtime receipts
         ├──► rebase               Multi-pass structural graph diffing across binary updates
         └──► build / hook / lua   Epistemic-gated Unity, Unreal, Godot, C++ & Lua mod builders
```

## CLI reference

Run `uv run toothpick <command> --help` for full option lists, or see [docs/CLI.md](docs/CLI.md) for the complete reference.

| Command | Purpose |
|---------|---------|
| `toothpick triage <target>` | Identify container (`PE`, `ELF`, `NSO`, `NRO`, `SELF`, `DOL`, `raw`), CPU architecture, engine (`Unity IL2CPP/Mono`, `Unreal`, `Godot`, `.NET`, `Native`), and packer protection. Aborts with exit code `2` on hypervisor/VM protectors (`PACKED_ABORT(VM_PROTECTED)`). |
| `toothpick unpack <target>` | Decompress UPX/XOR stubs, trace packer entry points to the Original Entry Point (OEP) via Unicorn micro-emulation, hunt compiler prologues (`--hunt-oep`), or rebuild PE headers from a dump (`--reconstruct-pe --oep <hex>`). |
| `toothpick static <target>` | Lift instructions (x86, x86_64, ARM64, PPC, MIPS), partition basic blocks, compute CFG edges and reaching-definitions SSA facts, and upsert into `toothpick.db`. |
| `toothpick query <subcommand>` | Bounded graph queries: `callers`, `callees`, `def_use`, `slice` (`backward`/`forward`), and `func` inspection. |
| `toothpick solve <func_addr>` | Translate basic-block paths into Z3 `BitVec` constraints and solve for concrete register/memory inputs that reach `--target-block <block_addr>`. |
| `toothpick ground <func_addr>` | Capture runtime register/argument receipts via Frida (`--pid`/`--spawn`), CEServer (`--backend ceserver --host ... --port ...`), or offline verification (`--static-only`) to mark functions `GROUNDED`. |
| `toothpick scan <pattern>` | Bounded memory and pointer scanner over live PIDs, raw dumps (`--file`), or CEServer targets (`--type i32\|f32\|string\|hex\|pointer`, narrowing via `--session`). |
| `toothpick ingest-metadata <file>` | Parse Unity IL2CPP `global-metadata.dat` type definitions and method descriptors into `toothpick.db`. |
| `toothpick ingest-ida <doc.json>` | Ingest versioned `ida_extract.v1.json` documents produced by `icepick`, routing symbol collisions through the name-conflict queue. |
| `toothpick build <target>` | Generate Unity IL2CPP BepInEx/Harmony C# plugins, Unreal Engine UE4SS Lua/C++ mods (`--engine unreal`), or Godot GDExtension scaffolds (`--engine godot`). Enforces the Epistemic Grounding Gate unless `--allow-unverified` is passed. |
| `toothpick hook build` | Emit a self-contained CMake C++ MinHook DLL project (`--target <addr> --out <dir> --name <name>`) gated by grounding receipts. |
| `toothpick lua build\|pack\|patch-emit` | Scaffold (`--class`, `--method`, `--desc`), package, and emit diff patches for Lua/Steam Workshop mods. |
| `toothpick pck list\|extract` | Inspect and safely extract Godot `.pck` resource packs. |
| `toothpick rebase <new_target>` | Diff a new binary update against `--old-sha` via exact digest, structural CFG similarity, semantic embeddings (`--semantic`), and external diff priors (`--priors`), outputting a port map (`--out`). |
| `toothpick rtti scan <file>` | Recover MSVC (`_s__RTTICompleteObjectLocator`) C++ class hierarchies, vtables (`--base`, `--vtables`), and demangled names. |
| `toothpick struct` | Synthesize struct layouts from memory dumps (`--file`) or live processes (`--pid`) and export ReClass.NET `.rcnet`, C++, C#, or JSON (`--format rcnet\|cpp\|csharp\|json`). |
| `toothpick jvm ingest <jar_or_class>` | Parse JVM `.class` and `.jar` archives into `toothpick.db` methods, call edges, and string pools. |
| `toothpick profile import\|hottest` | Import VTune/ETW/perf CSVs or JVM JFR/async-profiler captures (`profile import`) and rank hottest functions (`profile hottest --top 50`). |
| `toothpick cdb` | Run headless WinDbg/`cdb.exe` crash-dump triage (`--dump`) or non-invasive live process inspection (`--pid`). |
| `toothpick dbi` | Run Unicorn dynamic binary instrumentation traces (`--target`), generate Pin/DynamoRIO tools (`--generate`), or parse trace logs (`--parse`). |
| `toothpick taint` | Run register/memory taint analysis (`--binary`, `--func`, `--source`, `--sink`) or export Triton/Manticore scripts (`--export`). |
| `toothpick bench run\|compare` | Execute timed command benchmarks and evaluate regression gates across runs. |
| `toothpick show <func_addr>` | Display the unified function inspection view (metadata, CFG edges, SSA facts, grounding receipts, and trace events). |
| `toothpick ledger` | Inspect token and monetary cost accounting across agent sessions (`--db`, `--events-log`). |
| `toothpick mcp` | Start the stdio Model Context Protocol server (`--db toothpick.db`, optional `--sdk`). |

## MCP server

Run `toothpick mcp --db toothpick.db` to expose 9 phase-constrained reverse-engineering tools over stdio JSON-RPC 2.0:

```json
{
  "mcpServers": {
    "toothpick": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/toothpick",
        "run",
        "toothpick",
        "mcp",
        "--db",
        "/path/to/toothpick.db"
      ]
    }
  }
}
```

### Exposed MCP tools

| Tool | Description |
|------|-------------|
| `toothpick_callers` | Call-graph predecessors of `func_addr` (top-K capped, default 50, max 1,000). |
| `toothpick_callees` | Call-graph successors called by `func_addr` (top-K capped). |
| `toothpick_def_use` | SSA reaching-definition chains inside `func_addr`, optionally filtered by `var_name`. |
| `toothpick_slice` | Backward or forward intra-function program slice from `target` register/variable with `max_depth` and `limit` caps. |
| `toothpick_get_func` | Single function record (signature, basic-block count, IR/decompiled text bounded by `max_lines`). |
| `toothpick_rebase` | Port an address from `old_sha256` to `new_sha256` using stored version-rebase mappings. |
| `toothpick_show` | Full inspection card for `func_addr` (function record, callers/callees, SSA facts, grounding receipts, trace events). |
| `toothpick_ledger` | Token and cost accounting breakdown across model classes (`Orchestrator`, `Solver`, `Bulk`). |
| `toothpick_hottest` | Top-N hottest functions from imported ETW/JFR performance profiles joined against `toothpick.db`. |

Full request/response schemas and client examples are documented in [docs/MCP.md](docs/MCP.md).

## Epistemic Grounding Gate

Mod generators (`toothpick build`, `toothpick hook build`, `toothpick lua build`) refuse to emit patches targeting unverified addresses by default. A target function must hold a dynamic `GROUNDED` receipt in `toothpick.db` (recorded via `toothpick ground --pid <PID>` or `--backend ceserver`) before code generation proceeds.

- **Dynamic grounding path**: Run `toothpick ground <func_addr> --pid <PID> --db toothpick.db` (or `--backend ceserver`), then run `toothpick build` or `toothpick hook build`.
- **Static-only / offline path**: `toothpick ground --static-only` records a `STATIC_FALLBACK` (`UNGROUNDED`) receipt. To generate a mod scaffold without a live process receipt, pass `--allow-unverified`; generated source files and READMEs are stamped with an explicit `UNVERIFIED` banner.
- **IDA imports**: Functions imported via `toothpick ingest-ida` carry `UNGROUNDED` provenance until verified by live `toothpick ground`, preventing decompiler heuristics from silently masquerading as runtime truth.

## Development

```bash
uv sync --locked --dev
uv run python scripts/ci.py          # runs all 5 CI gates (ruff, uv build, pytest, coverage, license audit)
uv run pytest -p no:faulthandler     # run the test suite directly
uv run ruff check .                  # run the linter
```

## Security & known limitations

- **Untrusted binaries**: Static commands (`triage`, `static`, `query`, `solve`, `rtti`, `jvm ingest`, `ingest-ida`) never execute the target binary on the host OS. Dynamic commands (`ground` without `--static-only`, `scan --pid`, `cdb --pid`, `dbi`) attach to live processes and should only be used against untrusted targets inside an isolated VM. See [SECURITY.md](SECURITY.md).
- **VM-protected binaries**: Commercial virtualizing protectors (Denuvo, VMProtect, Themida) trigger an immediate `PACKED_ABORT(VM_PROTECTED)` (exit code `2`) during `triage` and `unpack`. Use `toothpick unpack --detect-vm` for non-aborting forensic section/entropy inspection, or dump the process from memory at runtime and pass `--reconstruct-pe --oep <hex>`.
- **Intra-procedural slicing**: `toothpick query slice` operates over intra-function SSA facts; cross-function dataflow should be combined with `callers`/`callees` traversal.

## Contributing

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). Two things are asked of contributors: a `Signed-off-by` line on every commit (the [Developer Certificate of Origin](DCO), `git commit -s`; CI checks it), and a one-time acceptance of the [Contributor License Agreement](CLA.md) on a contributor's first pull request.

## License

AGPL-3.0-only, with commercial/enterprise dual licensing available. See [LICENSE](LICENSE), [COPYRIGHT](COPYRIGHT), and [CLA.md](CLA.md).

Running unmodified Toothpick locally places no obligations on your own code or analysis artifacts. Modifying Toothpick, distributing modified versions, or operating a modified Toothpick over a network triggers the AGPL-3.0 copyleft/network provisions — offer those modifications under the same license, or contact the Project Owner (`walkthestars`) for a commercial enterprise license.
