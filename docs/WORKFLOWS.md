# Toothpick End-User Workflows

Practical step-by-step guides for common reverse-engineering and modding tasks.

---

## Workflow 1: Triage, Unpack, and Index a Native Binary

1. **Triage the binary** to determine its container type, CPU architecture, engine, and whether it is packed:
   ```bash
   uv run toothpick triage target.exe
   ```
   - If the report shows `protection: "UPX"` or `"XOR"`, proceed to step 2.
   - If the report exits with code `2` (`PACKED_ABORT(VM_PROTECTED)`), the binary is wrapped in a commercial virtualizing protector. Run `uv run toothpick unpack target.exe --detect-vm` for a non-aborting section/entropy report, or capture a runtime memory dump and reconstruct the PE header:
     ```bash
     uv run toothpick unpack dump.bin --reconstruct-pe --oep 0x140001000 --out unpacked.exe
     ```

2. **Unpack simple protection stubs**:
   ```bash
   uv run toothpick unpack target.exe --out unpacked.exe
   ```

3. **Ingest into `toothpick.db`**:
   ```bash
   uv run toothpick static unpacked.exe --db toothpick.db
   ```

4. **Explore functions and dataflow**:
   ```bash
   uv run toothpick query --db toothpick.db callers 0x140001000
   uv run toothpick query --db toothpick.db slice 0x140001000 --target rax --direction backward
   uv run toothpick show 0x140001000 --db toothpick.db
   ```

---

## Workflow 2: Enriching `toothpick.db` with IDA Pro (`icepick`)

When you have an IDA Pro 9.4+ installation and [`icepick`](https://github.com/walkthestars/icepick), you can combine IDA's FLIRT/Hex-Rays output with Toothpick's static/dynamic graph:

1. **Extract a schema-v1 JSON inventory with `icepick`**:
   ```bash
   uv run python -m icepick extract target.exe --backend idalib \
     --ida-home "C:\Program Files\IDA Professional 9.5" --out ida_extract.v1.json
   ```

2. **Ingest the IDA document into Toothpick**:
   ```bash
   uv run toothpick ingest-ida ida_extract.v1.json --db toothpick.db
   ```
   - Empty function names in `toothpick.db` are populated from IDA.
   - If a function already has a `GROUNDED` name that disagrees with IDA's symbol, Toothpick preserves the grounded name and logs a pending record in the name-conflict queue rather than overwriting silently.
   - IDA-sourced functions receive an `UNGROUNDED` (`ida_extract`) receipt until verified via `toothpick ground`.

---

## Workflow 3: Grounding a Function and Building a Mod

Toothpick's mod builders require a `GROUNDED` receipt before generating hooks, preventing patches aimed at unverified offsets.

### Step 1: Record a grounding receipt

Attach to a live process with Frida or a `ceserver` endpoint to record a confirmed `GROUNDED` receipt (or use `--static-only` plus `--allow-unverified` on the build step when working purely offline):

```bash
# Live process attach via Frida (records a GROUNDED receipt)
uv run toothpick ground 0x140001200 --pid 4812 --db toothpick.db

# Offline static fallback receipt (records UNGROUNDED; pair with --allow-unverified on build)
uv run toothpick ground 0x140001200 --static-only --db toothpick.db
```

### Step 2A: Unity IL2CPP (BepInEx + Harmony C# plugin)

```bash
# Optional: ingest IL2CPP metadata for symbol and type resolution
uv run toothpick ingest-metadata global-metadata.dat --db toothpick.db

# Generate and compile the BepInEx plugin
uv run toothpick build 0x140001200 --engine unity --db toothpick.db \
  --out ./mods/FOVMod --name FOVMod --hook-type postfix --compile
```

### Step 2B: Unreal Engine (UE4SS) or Godot (GDExtension)

```bash
uv run toothpick build 0x140001200 --engine unreal --db toothpick.db \
  --out ./mods/UnrealPatch --name UnrealPatch

uv run toothpick build 0x140001200 --engine godot --db toothpick.db \
  --out ./mods/GodotPatch --name GodotPatch
```

### Step 2C: Native C/C++ Shared Library Hook (`MinHook` DLL)

```bash
uv run toothpick hook build --target 0x140001200 --db toothpick.db \
  --out ./mods/NativeHook --name NativeHook --arch x64
```

---

## Workflow 4: Rebasing Hooks Across a Game Update

When a target binary updates from `v1.exe` to `v2.exe`:

1. **Ingest the baseline binary into `toothpick.db`**:
   ```bash
   uv run toothpick static v1.exe --db toothpick.db
   ```

2. **Run differential rebase** against `v2.exe`:
   ```bash
   uv run toothpick rebase v2.exe --old-sha <v1_sha256> --db toothpick.db \
     --semantic --threshold 0.95 --out offset_map.json
   ```
   - Functions with exact byte/mnemonic digests match in Pass 1 (`1.0`).
   - Modified functions are scored by CFG topology, call-graph neighborhood, and 64-D semantic vectors in Pass 2.
   - If you also have a Diaphora export converted via `icepick diff-import`, pass `--priors candidates.json` to incorporate external similarity priors.

---

## Workflow 5: Hot-Path Profiling + C++ RTTI Recovery

To locate and reverse-engineer performance bottlenecks in large native C++ or JVM games:

1. **Recover C++ classes and vtables** (for native MSVC binaries) or **ingest JVM bytecode** (for `.jar` targets):
   ```bash
   uv run toothpick rtti scan game.exe --base 0x140000000 --vtables 0x14001000,0x14002000 --db toothpick.db
   # or for JVM targets:
   uv run toothpick jvm ingest game.jar --db toothpick.db
   ```

2. **Import a profiler capture** (VTune/ETW/perf CSV, async-profiler collapsed stacks, or `jfr print --json`):
   ```bash
   uv run toothpick profile import trace.csv --format etw --db toothpick.db
   ```

3. **Query the hottest functions**:
   ```bash
   uv run toothpick profile hottest --db toothpick.db --top 25
   ```
