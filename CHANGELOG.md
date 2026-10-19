# Changelog

All notable changes to Toothpick will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-10-18

### Added

- Multi-container binary triage (`toothpick triage`) supporting PE/COFF, ELF32/64, Nintendo Switch NSO/NRO, PlayStation SELF, GameCube/Wii DOL, and raw firmware blobs.
- Static and Unicorn micro-emulated unpacking pre-pass (`toothpick unpack`) for UPX (NRV2B/D/E), XOR loaders, OEP hunting, and PE reconstruction.
- Multi-architecture static lifting and SQLite WAL graph storage (`toothpick static`) over Capstone (x86, x86_64, ARM64, PowerPC, MIPS) with optional PyGhidra decompilation, basic-block CFG reconstruction, and reaching-definitions SSA extraction.
- Bounded query engine (`toothpick query`) for `callers`, `callees`, `def_use`, `slice`, and `func` with top-K and line caps.
- Z3 BitVec SMT path-reachability solver (`toothpick solve`) with per-function timeout guard.
- Dynamic grounding harness (`toothpick ground`) via Frida and CEServer with `--static-only` offline fallback, plus bounded memory and pointer scanning (`toothpick scan`).
- Epistemic-gated mod scaffolding and build pipelines (`toothpick build`, `toothpick hook build`, `toothpick lua`) for Unity IL2CPP (BepInEx/Harmony), Unreal Engine (UE4SS), Godot (GDExtension + PCK inspection), native C++ shared-library hooks (MinHook/Detours/LD_PRELOAD), and Lua/Steam Workshop mods.
- Differential binary rebase engine (`toothpick rebase`) with structural graph matching and external diff-candidate priors.
- IDA Pro schema-v1 ingestion consumer (`toothpick ingest-ida`) with non-destructive name-conflict arbitration.
- C++ RTTI and vtable recovery (`toothpick rtti scan`), symbol demangling, string cross-reference labeling, library vendor exclusion, and script-to-RTTI correlation.
- JVM bytecode and call-graph ingestion (`toothpick jvm ingest`) and profile ingestion (`toothpick profile import` / `hottest`) for Windows ETW/WPR and JVM JFR/async-profiler.
- WinDbg/cdb crash-dump and live debugger integration (`toothpick cdb`), dynamic binary instrumentation (`toothpick dbi`), symbolic taint analysis (`toothpick taint`), ReClass.NET struct recovery (`toothpick struct`), and performance benchmark comparison (`toothpick bench`).
- Stdio Model Context Protocol server (`toothpick mcp`) exposing 9 bounded reverse-engineering tools over JSON-RPC 2.0 and the official MCP Python SDK.
