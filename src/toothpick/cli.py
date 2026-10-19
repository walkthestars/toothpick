"""
Toothpick CLI interface.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import click

from toothpick.query import (
    DEFAULT_LIMIT,
    QueryEngine,
    QueryError,
    SliceDirection,
    parse_address,
)
from toothpick.storage import StorageError
from toothpick.triage import (
    PackedAbortError,
    TriageError,
    UnknownFormatError,
    triage_binary,
)


@click.group()
def cli() -> None:
    """Toothpick: Agent-driven binary reverse engineering & modding pipeline."""
    pass


@cli.command()
@click.argument("target", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--raw", is_flag=True, help="Treat target as raw binary blob")
@click.option("--arch", type=str, default=None, help="Explicit target architecture hint for raw targets")
@click.option("--base", type=str, default=None, help="Explicit base address hex for raw targets")
@click.option("--static-only", is_flag=True, help="Disable live VM/Frida/emulator grounding requirement")
def triage(target: Path, raw: bool, arch: Optional[str], base: Optional[str], static_only: bool) -> None:
    """Triage a binary file for format, architecture, engine, and packers."""
    raw_base_int = int(base, 16) if base else None
    try:
        report = triage_binary(
            target_path=target,
            is_raw=raw,
            raw_arch=arch,
            raw_base=raw_base_int,
            static_only=static_only,
        )
        click.echo(report.to_json())
    except PackedAbortError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(2)
    except UnknownFormatError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(3)
    except TriageError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("target", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--out", "output", type=click.Path(dir_okay=False, path_type=Path), default=None, help="Output path for unpacked binary")
@click.option(
    "--detect-vm",
    is_flag=True,
    help="Forensic VM-protection triage (entropy + structural heuristics, no abort)",
)
@click.option(
    "--hunt-oep",
    is_flag=True,
    help="Hunt the Original Entry Point via compiler-prologue heuristics",
)
@click.option(
    "--reconstruct-pe",
    is_flag=True,
    help="Rebuild a runnable PE from a memory dump (use with --oep)",
)
@click.option(
    "--oep",
    "oep_text",
    type=str,
    default=None,
    help="Original Entry Point address hex for --reconstruct-pe",
)
def unpack(
    target: Path,
    output: Optional[Path],
    detect_vm: bool,
    hunt_oep: bool,
    reconstruct_pe: bool,
    oep_text: Optional[str],
) -> None:
    """Run simple unpack pre-pass on target binary."""
    from toothpick.unpack import UnpackError, unpack_analysis, unpack_binary
    if detect_vm or hunt_oep or reconstruct_pe:
        try:
            analysis = unpack_analysis(
                Path(target),
                output,
                detect_vm=detect_vm,
                hunt_oep=hunt_oep,
                reconstruct_pe=reconstruct_pe,
                oep=oep_text,
            )
        except (UnpackError, FileNotFoundError) as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)
        analysis["command"] = "unpack"
        analysis["status"] = "ok"
        click.echo(json.dumps(analysis, indent=2, sort_keys=True, default=str))
        return
    try:
        report = unpack_binary(target, output_path=output)
        click.echo(report.to_json())
    except PackedAbortError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(2)
    except UnpackError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command("static")
@click.argument("target", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--db",
    "db_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("toothpick.db"),
    show_default=True,
    help="SQLite graph database (toothpick.db) to ingest into",
)
@click.option("--raw", is_flag=True, help="Treat target as a raw memory blob")
@click.option(
    "--arch",
    type=str,
    default=None,
    help="Architecture for raw targets: x86, x86_64, arm64, ppc, mips",
)
@click.option(
    "--base",
    type=str,
    default=None,
    help="Base address of a raw blob (hex, e.g. 0x10000)",
)
@click.option(
    "--static-only",
    is_flag=True,
    help="Force static-only analysis (no live/dynamic grounding)",
)
@click.option(
    "--prefer-ghidra",
    is_flag=True,
    help="Use PyGhidra C decompilation when a Ghidra install is available",
)
@click.option(
    "--json/--no-json",
    "as_json",
    default=True,
    show_default=True,
    help="Emit the analysis summary as JSON",
)
def static(
    target: Path,
    db_path: Path,
    raw: bool,
    arch: Optional[str],
    base: Optional[str],
    static_only: bool,
    prefer_ghidra: bool,
    as_json: bool,
) -> None:
    """Ingest a target into toothpick.db: functions, edges, and DB stats."""
    from toothpick.static_engine import (
        ContainerParseError,
        StaticEngine,
        StaticEngineError,
        UnsupportedArchitectureError,
    )
    from toothpick.storage import StorageError

    raw_base: Optional[int] = None
    if base:
        try:
            raw_base = int(base, 0 if base.lower().startswith("0x") else 16)
        except ValueError:
            click.echo(f"Error: --base must be a hex address, got {base!r}", err=True)
            sys.exit(1)
    if raw and not arch:
        click.echo(
            "Error: --raw requires --arch (x86, x86_64, arm64, ppc, mips)", err=True
        )
        sys.exit(1)

    try:
        engine = StaticEngine(db_path=db_path, prefer_ghidra=prefer_ghidra)
    except StorageError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    try:
        report = engine.analyze(
            target,
            is_raw=raw,
            raw_arch=arch,
            raw_base=raw_base,
            static_only=static_only,
            prefer_ghidra=prefer_ghidra,
        )
        click.echo(report.to_json(indent=2 if as_json else None))
    except UnsupportedArchitectureError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(4)
    except ContainerParseError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(3)
    except StaticEngineError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except StorageError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        engine.close()

# -- W04: phase-constrained query engine + MCP server (PLAN-W04-mcp.md 3) --


@cli.group()
@click.option(
    "--db",
    "db_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="SQLite graph database (toothpick.db) to query",
)
@click.option(
    "--sha",
    "binary_sha256",
    type=str,
    default=None,
    help="binary_sha256 to query (inferred when the db holds exactly one binary)",
)
@click.pass_context
def query(
    ctx: click.Context,
    db_path: Optional[Path],
    binary_sha256: Optional[str],
) -> None:
    """Phase-constrained queries over toothpick.db: callers, callees, def_use, slice."""
    ctx.obj = {"db_path": db_path, "binary_sha256": binary_sha256}


def _db_options(command: Any) -> Any:
    """Attach ``--sha``/``--db`` to a ``query`` subcommand.

    Group-level values (``toothpick query --sha S callers ADDR``) act as defaults
    and the subcommand-level options take precedence (see :func:`_engine_for`).
    """
    for decorator in (
        click.option(
            "--sha",
            "binary_sha256",
            type=str,
            default=None,
            help="binary_sha256 to query (inferred when the db holds exactly one binary)",
        ),
        click.option(
            "--db",
            "db_path",
            type=click.Path(dir_okay=False, path_type=Path),
            default=None,
            help="SQLite graph database (toothpick.db) to query",
        ),
    ):
        command = decorator(command)
    return command


def _query_options(command: Any) -> Any:
    """``--db``/``--sha`` plus the strict top-K ``--limit`` budget (default 50)."""
    return click.option(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        show_default=True,
        help="Top-K budget for the response (never an unbounded list)",
    )(_db_options(command))


def _emit_json(payload: Dict[str, Any]) -> None:
    """Print one JSON payload with a stable key order."""
    click.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _engine_for(
    ctx: click.Context, db_path: Optional[Path], binary_sha256: Optional[str]
) -> Tuple[QueryEngine, str]:
    """Open a :class:`QueryEngine` for a ``query`` subcommand, resolving --db/--sha.

    The database must already exist (a typo'd ``--db`` is an error, not an empty
    store) and the binary is inferred when the store holds exactly one of them.
    """
    parent = ctx.obj if isinstance(ctx.obj, dict) else {}
    path = db_path or parent.get("db_path") or Path("toothpick.db")
    sha_arg = binary_sha256 or parent.get("binary_sha256")
    try:
        engine = QueryEngine(path)
        sha = engine.resolve_sha(sha_arg)
    except StorageError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    except QueryError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    return engine, sha


@query.command("callers")
@click.argument("func_addr", type=str)
@_query_options
@click.pass_context
def query_callers(
    ctx: click.Context,
    func_addr: str,
    db_path: Optional[Path],
    binary_sha256: Optional[str],
    limit: int,
) -> None:
    """List the functions that call FUNC_ADDR (call-graph predecessors, top-K)."""
    engine, sha = _engine_for(ctx, db_path, binary_sha256)
    try:
        addr = parse_address(func_addr)
        items = engine.callers(sha, addr, limit)
        total = engine.count_callers(sha, addr)
        payload = {
            "query": "callers",
            "binary_sha256": sha,
            "func_addr": addr,
            "func_addr_hex": f"0x{addr:x}",
            "limit": engine.resolve_limit(limit),
            "count": len(items),
            "total": total,
            "truncated": len(items) < total,
            "results": items,
        }
    except QueryError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    finally:
        engine.close()
    _emit_json(payload)


@query.command("callees")
@click.argument("func_addr", type=str)
@_query_options
@click.pass_context
def query_callees(
    ctx: click.Context,
    func_addr: str,
    db_path: Optional[Path],
    binary_sha256: Optional[str],
    limit: int,
) -> None:
    """List the functions called by FUNC_ADDR (call-graph successors, top-K)."""
    engine, sha = _engine_for(ctx, db_path, binary_sha256)
    try:
        addr = parse_address(func_addr)
        items = engine.callees(sha, addr, limit)
        total = engine.count_callees(sha, addr)
        payload = {
            "query": "callees",
            "binary_sha256": sha,
            "func_addr": addr,
            "func_addr_hex": f"0x{addr:x}",
            "limit": engine.resolve_limit(limit),
            "count": len(items),
            "total": total,
            "truncated": len(items) < total,
            "results": items,
        }
    except QueryError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    finally:
        engine.close()
    _emit_json(payload)


@query.command("def_use")
@click.argument("func_addr", type=str)
@click.option(
    "--var",
    "--var-name",
    "var_name",
    type=str,
    default=None,
    help="Register or SSA variable name, e.g. 'rax' or 'rax_1'",
)
@_query_options
@click.pass_context
def query_def_use(
    ctx: click.Context,
    func_addr: str,
    var_name: Optional[str],
    db_path: Optional[Path],
    binary_sha256: Optional[str],
    limit: int,
) -> None:
    """Return the SSA def-use chains inside FUNC_ADDR (top-K)."""
    engine, sha = _engine_for(ctx, db_path, binary_sha256)
    try:
        addr = parse_address(func_addr)
        facts = engine.def_use(sha, addr, var_name, limit)
        payload = {
            "query": "def_use",
            "binary_sha256": sha,
            "func_addr": addr,
            "func_addr_hex": f"0x{addr:x}",
            "var_name": var_name,
            "limit": engine.resolve_limit(limit),
            "count": len(facts),
            "results": facts,
        }
    except QueryError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    finally:
        engine.close()
    _emit_json(payload)


@query.command("slice")
@click.argument("func_addr", type=str)
@click.option(
    "--target",
    "--reg",
    "target",
    type=str,
    default=None,
    help="Register/SSA variable to slice from, e.g. 'rax'",
)
@click.option(
    "--direction",
    type=click.Choice([direction.value for direction in SliceDirection]),
    default=SliceDirection.BACKWARD.value,
    show_default=True,
    help="backward = from sink to producers, forward = from source to consumers",
)
@click.option(
    "--max-depth",
    type=int,
    default=10,
    show_default=True,
    help="Maximum number of def-use hops to traverse",
)
@_query_options
@click.pass_context
def query_slice(
    ctx: click.Context,
    func_addr: str,
    target: Optional[str],
    direction: str,
    max_depth: int,
    db_path: Optional[Path],
    binary_sha256: Optional[str],
    limit: int,
) -> None:
    """Intra-function program slice of FUNC_ADDR (backward from a sink, or forward)."""
    engine, sha = _engine_for(ctx, db_path, binary_sha256)
    try:
        addr = parse_address(func_addr)
        steps = engine.slice(sha, addr, target, direction, max_depth, limit)
        payload = {
            "query": "slice",
            "binary_sha256": sha,
            "func_addr": addr,
            "func_addr_hex": f"0x{addr:x}",
            "target": target,
            "direction": SliceDirection.coerce(direction).value,
            "max_depth": engine.resolve_max_depth(max_depth),
            "limit": engine.resolve_limit(limit),
            "count": len(steps),
            "results": steps,
        }
    except QueryError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    finally:
        engine.close()
    _emit_json(payload)


@query.command("func")
@click.argument("func_addr", type=str)
@click.option(
    "--max-lines",
    type=int,
    default=None,
    help="Maximum lines of decompiled code/IR to return",
)
@_db_options
@click.pass_context
def query_func(
    ctx: click.Context,
    func_addr: str,
    max_lines: Optional[int],
    db_path: Optional[Path],
    binary_sha256: Optional[str],
) -> None:
    """Fetch one analysed function record (name, prototype, IR p-code, decompiled C)."""
    engine, sha = _engine_for(ctx, db_path, binary_sha256)
    try:
        addr = parse_address(func_addr)
        record = engine.get_function(sha, addr, max_lines=max_lines)
        payload = {
            "query": "func",
            "binary_sha256": sha,
            "func_addr": addr,
            "func_addr_hex": f"0x{addr:x}",
            "found": record is not None,
            "function": record,
        }
    except QueryError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    finally:
        engine.close()
    _emit_json(payload)


@cli.command("mcp")
@click.option(
    "--db",
    "db_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("toothpick.db"),
    show_default=True,
    help="SQLite graph database (toothpick.db) to serve over MCP",
)
@click.option(
    "--sdk",
    "use_sdk",
    is_flag=True,
    help="Also register the same tools on the official 'mcp' SDK server",
)
def mcp(db_path: Path, use_sdk: bool) -> None:
    """Launch the MCP (Model Context Protocol) stdio server over toothpick.db."""
    from toothpick.mcp_server import ToothpickMcpServer

    try:
        server = ToothpickMcpServer(db_path, use_sdk=use_sdk)
        if use_sdk:
            server.use_sdk(True)
    except StorageError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    except QueryError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    try:
        server.serve()
    except (BrokenPipeError, KeyboardInterrupt):  # pragma: no cover - pipe/interrupt
        pass
    finally:
        server.engine.close()


@cli.command("solve")
@click.argument("func_addr", type=str)
@click.option(
    "--target-block",
    type=str,
    default=None,
    help="Target basic block address hex or int to solve reachability for",
)
@click.option(
    "--timeout",
    type=float,
    default=5.0,
    show_default=True,
    help="Strict solver timeout in seconds (default 5.0)",
)
@click.option(
    "--db",
    "db_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("toothpick.db"),
    show_default=True,
    help="SQLite graph database (toothpick.db) to query",
)
@click.option(
    "--sha",
    "binary_sha256",
    type=str,
    default=None,
    help="Explicit binary SHA-256 (inferred if db contains exactly one binary)",
)
def solve(
    func_addr: str,
    target_block: Optional[str],
    timeout: float,
    db_path: Path,
    binary_sha256: Optional[str],
) -> None:
    """Solve SMT reachability and input constraints to reach a target basic block."""
    from toothpick.solver import SMTSolver, SolverError

    try:
        addr = parse_address(func_addr)
        target = parse_address(target_block) if target_block else None
    except QueryError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    if not db_path.exists():
        click.echo(f"Error: database not found: {db_path}", err=True)
        sys.exit(1)

    try:
        with SMTSolver(db=db_path) as solver:
            result = solver.solve_path(
                func_addr=addr,
                target_block_addr=target,
                timeout_sec=timeout,
                binary_sha256=binary_sha256,
            )
            payload = {
                "command": "solve",
                "func_addr": addr,
                "func_addr_hex": f"0x{addr:x}",
                "target_block": target,
                "target_block_hex": f"0x{target:x}" if target is not None else None,
                "status": result.status,
                "model": result.model,
                "constraints": result.constraints,
                "elapsed_sec": round(result.elapsed_sec, 4),
            }
            _emit_json(payload)
    except SolverError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


@cli.command("ground")
@click.argument("func_addr", type=str)
@click.option(
    "--pid",
    type=int,
    default=None,
    help="Process ID of a running target process to attach to",
)
@click.option(
    "--spawn",
    "spawn_target",
    type=str,
    default=None,
    help="Executable path to spawn and instrument under Frida",
)
@click.option(
    "--static-only",
    is_flag=True,
    help="Emit deterministic static fallback receipt without live instrumentation (ADR-06 / Risk R5)",
)
@click.option(
    "--timeout",
    type=float,
    default=5.0,
    show_default=True,
    help="Timeout in seconds for dynamic breakpoint trigger (default 5.0)",
)
@click.option(
    "--db",
    "db_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="SQLite graph database (toothpick.db) to persist receipts into",
)
@click.option(
    "--backend",
    "backend_type",
    type=click.Choice(["frida", "ceserver"], case_sensitive=False),
    default="frida",
    show_default=True,
    help="Dynamic grounding backend (frida or ceserver)",
)
@click.option(
    "--host",
    type=str,
    default="127.0.0.1",
    show_default=True,
    help="Remote host for ceserver backend",
)
@click.option(
    "--port",
    type=int,
    default=52736,
    show_default=True,
    help="Remote TCP port for ceserver backend",
)
@click.option(
    "--sha",
    "binary_sha256",
    type=str,
    default=None,
    help="Explicit binary SHA-256 (inferred if db contains exactly one binary)",
)
def ground(
    func_addr: str,
    pid: Optional[int],
    spawn_target: Optional[str],
    static_only: bool,
    timeout: float,
    backend_type: str,
    host: str,
    port: int,
    db_path: Optional[Path],
    binary_sha256: Optional[str],
) -> None:
    """Dynamically ground execution state via Frida/CEServer or emit a static fallback receipt."""
    from toothpick.ground import GroundError, GroundReceipt, GroundingHarness

    if backend_type.lower() == "ceserver":
        if spawn_target is not None or static_only:
            click.echo(
                "Error: --spawn and --static-only cannot be used with --backend ceserver",
                err=True,
            )
            sys.exit(1)
        if pid is None:
            click.echo("Error: --pid is required when using --backend ceserver", err=True)
            sys.exit(1)
    else:
        modes = sum(bool(x) for x in (pid is not None, spawn_target is not None, static_only))
        if modes == 0:
            click.echo("Error: Must specify at least one of --pid, --spawn, or --static-only", err=True)
            sys.exit(1)
        if modes > 1:
            click.echo("Error: Mutually exclusive modes: specify only one of --pid, --spawn, or --static-only", err=True)
            sys.exit(1)

    try:
        addr = parse_address(func_addr)
    except QueryError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    if db_path is not None and not db_path.exists():
        click.echo(f"Error: database not found: {db_path}", err=True)
        sys.exit(1)

    target_db = db_path
    if target_db is None:
        default_db = Path("toothpick.db")
        if default_db.is_file():
            target_db = default_db

    try:
        with GroundingHarness(db=target_db, binary_sha256=binary_sha256) as harness:
            if backend_type.lower() == "ceserver":
                if pid is None:
                    click.echo("Error: --pid is required when using --backend ceserver", err=True)
                    sys.exit(1)
                import time
                from uuid import uuid4
                from toothpick.ceserver import CEServerClient

                with CEServerClient() as client:
                    client.connect(host=host, port=port, timeout=timeout)
                    handle = client.open_process(pid)
                    try:
                        raw_bytes = client.read_memory(handle, addr, 64)
                        region = client.virtual_query(handle, addr)
                        live_reads = {
                            "backend": "ceserver",
                            "host": host,
                            "port": port,
                            "pid": pid,
                            "addr_bytes_hex": raw_bytes.hex(),
                            "region": {
                                "base": region.base if region else None,
                                "size": region.size if region else None,
                                "protection": region.protection if region else None,
                            }
                            if region
                            else None,
                        }
                        sha = binary_sha256 or harness._resolve_sha()
                        receipt = GroundReceipt(
                            receipt_id=f"rec_{uuid4().hex[:12]}",
                            binary_sha256=sha,
                            func_addr=addr,
                            ground_type="CESERVER_LIVE",
                            timestamp=time.time(),
                            live_reads=live_reads,
                            verifier_status="GROUNDED",
                            phase="ground",
                        )
                        if harness.db is not None:
                            harness.db.insert_receipt(receipt)
                    finally:
                        try:
                            client.close_handle(handle)
                        except Exception:
                            pass
            elif static_only:
                receipt = harness.fallback_read(addr)
            elif pid is not None:
                receipt = harness.attach_and_read(pid, addr, timeout=timeout)
            elif spawn_target is not None:
                receipt = harness.spawn_and_read(spawn_target, addr, timeout=timeout)
            else:  # pragma: no cover
                receipt = harness.fallback_read(addr)
            _emit_json(receipt.to_dict())
    except (GroundError, StorageError) as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


@cli.command("scan")
@click.argument("pattern", type=str)
@click.option(
    "--type",
    "data_type",
    type=click.Choice(
        [
            "i8",
            "u8",
            "i16",
            "u16",
            "i32",
            "u32",
            "i64",
            "u64",
            "f32",
            "f64",
            "string",
            "hex",
            "pointer",
        ],
        case_sensitive=False,
    ),
    default="i32",
    show_default=True,
    help="Data type to scan for",
)
@click.option(
    "--file",
    "file_target",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Offline memory dump or raw binary file to scan",
)
@click.option(
    "--pid",
    type=int,
    default=None,
    help="Process ID of a running target process to scan via Frida",
)
@click.option(
    "--base",
    "base_addr",
    type=str,
    default=None,
    help="Base address hex for raw file target (e.g. 0x140000000)",
)
@click.option(
    "--limit",
    type=int,
    default=50,
    show_default=True,
    help="Maximum number of matches to return (ceiling 1000)",
)
@click.option(
    "--session",
    "session_id",
    type=str,
    default=None,
    help="Existing scan session ID to refine/filter",
)
@click.option(
    "--comparison",
    "--filter",
    "comparison",
    type=click.Choice(
        ["exact", "increased", "decreased", "changed", "unchanged"],
        case_sensitive=False,
    ),
    default="exact",
    show_default=True,
    help="Refinement comparison operator when using --session",
)
@click.option(
    "--db",
    "db_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="SQLite graph database (toothpick.db) to persist scan receipts into",
)
@click.option(
    "--backend",
    "backend_choice",
    type=click.Choice(["buffer", "frida", "ceserver"], case_sensitive=False),
    default=None,
    help="Memory backend to use (buffer, frida, or ceserver)",
)
@click.option(
    "--host",
    type=str,
    default="127.0.0.1",
    show_default=True,
    help="Remote host for ceserver backend",
)
@click.option(
    "--port",
    type=int,
    default=52736,
    show_default=True,
    help="Remote TCP port for ceserver backend",
)
@click.option(
    "--sha",
    "binary_sha256",
    type=str,
    default=None,
    help="Explicit binary SHA-256 (inferred if db contains exactly one binary)",
)
@click.option(
    "--export-reclass",
    "export_reclass",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write match addresses as a ReClass.NET XML project to this path",
)
def scan(
    pattern: str,
    data_type: str,
    file_target: Optional[Path],
    pid: Optional[int],
    base_addr: Optional[str],
    limit: int,
    session_id: Optional[str],
    comparison: str,
    db_path: Optional[Path],
    binary_sha256: Optional[str],
    backend_choice: Optional[str],
    host: str,
    port: int,
    export_reclass: Optional[Path],
) -> None:
    """Bounded memory scanner searching live process or offline dumps."""
    if limit < 1 or limit > 1000:
        click.echo(
            f"Error: --limit must be between 1 and 1000 (got {limit})",
            err=True,
        )
        sys.exit(1)
    from toothpick.scanner import (
        BufferMemoryBackend,
        FridaMemoryBackend,
        MemoryScanner,
        ScannerError,
    )

    if file_target is not None and pid is not None:
        click.echo("Error: Mutually exclusive targets: specify only one of --file or --pid", err=True)
        sys.exit(1)

    if file_target is None and pid is None and session_id is None:
        click.echo("Error: Must specify either --file, --pid, or an active --session", err=True)
        sys.exit(1)

    try:
        base_int = parse_address(base_addr) if base_addr is not None else 0
    except QueryError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    if db_path is not None and not db_path.exists():
        click.echo(f"Error: database not found: {db_path}", err=True)
        sys.exit(1)

    target_db = db_path
    if target_db is None:
        default_db = Path("toothpick.db")
        if default_db.is_file():
            target_db = default_db

    backend = None
    choice = backend_choice.lower() if backend_choice else None
    if choice == "ceserver":
        if pid is None:
            click.echo("Error: --pid is required when using --backend ceserver", err=True)
            sys.exit(1)
        from toothpick.ceserver import CEServerMemoryBackend

        backend = CEServerMemoryBackend(host=host, port=port, pid=pid)
    elif file_target is not None or choice == "buffer":
        if file_target is None:
            click.echo("Error: --file is required when using --backend buffer", err=True)
            sys.exit(1)
        if not file_target.exists():
            click.echo(f"Error: target file not found: {file_target}", err=True)
            sys.exit(1)
        backend = BufferMemoryBackend.from_file(file_target, base_address=base_int)
    elif pid is not None or choice == "frida":
        if pid is None and session_id is None:
            click.echo("Error: --pid is required when using --backend frida", err=True)
            sys.exit(1)
        if pid is not None:
            backend = FridaMemoryBackend(pid)

    try:
        with MemoryScanner(db=target_db) as scanner:
            if session_id is not None:
                result = scanner.filter_scan(
                    session_id=session_id,
                    new_value=pattern,
                    comparison=comparison,
                    limit=limit,
                    binary_sha256=binary_sha256,
                    backend=backend,
                )
            elif data_type.lower() == "pointer":
                if backend is None:
                    click.echo("Error: Must specify --file, --pid, or --backend for pointer scan", err=True)
                    sys.exit(1)
                try:
                    target_addr = parse_address(pattern)
                except QueryError as exc:
                    click.echo(f"Error: {exc}", err=True)
                    sys.exit(1)
                result = scanner.scan_pointers(
                    backend=backend,
                    target_address=target_addr,
                    limit=limit,
                    binary_sha256=binary_sha256,
                )
            else:
                if backend is None:
                    click.echo("Error: Must specify --file, --pid, or --backend for initial scan", err=True)
                    sys.exit(1)
                result = scanner.first_scan(
                    backend=backend,
                    value=pattern,
                    data_type=data_type.lower(),
                    limit=limit,
                    binary_sha256=binary_sha256,
                )
            payload = result.to_dict()
            if export_reclass is not None:
                from toothpick.reclass import ReclassError, StructSynthesizer

                addrs = sorted({int(m.address) for m in result.matches})
                if not addrs:
                    click.echo("Error: no matches to export as ReClass XML", err=True)
                    sys.exit(1)
                base = addrs[0]
                span = addrs[-1] - base + 8
                try:
                    layout = StructSynthesizer().synthesize_from_accesses(
                        [(addr - base, "pointer") for addr in addrs],
                        span,
                        name="ScanMatches",
                    )
                    export_reclass.write_text(layout.to_reclass_xml(), encoding="utf-8")
                except ReclassError as exc:
                    click.echo(f"Error: {exc}", err=True)
                    sys.exit(1)
                payload["reclass_path"] = str(export_reclass)
            _emit_json(payload)
    except (ScannerError, StorageError) as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    finally:
        if backend is not None and hasattr(backend, "close"):
            backend.close()


@cli.command("ingest-metadata")
@click.argument("metadata_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--db",
    "db_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("toothpick.db"),
    show_default=True,
    help="SQLite database (toothpick.db) to ingest metadata into",
)
@click.option(
    "--sha",
    "binary_sha256",
    type=str,
    default=None,
    help="Explicit binary SHA-256 for metadata attribution (inferred if db contains 1 binary)",
)
def ingest_metadata(metadata_file: Path, db_path: Path, binary_sha256: Optional[str]) -> None:
    """Ingest Unity IL2CPP type metadata into toothpick.db."""
    from toothpick.mod_builder import Il2CppMetadataReader, MetadataParseError
    from toothpick.storage import Database, StorageError

    try:
        with Database(db_path, create=True) as db:
            sha = binary_sha256 or ""
            if not sha:
                rows = db._conn.execute(
                    "SELECT DISTINCT binary_sha256 FROM funcs WHERE binary_sha256 != '' "
                    "UNION SELECT DISTINCT binary_sha256 FROM types WHERE binary_sha256 != ''"
                ).fetchall()
                shas = [r[0] for r in rows if r[0]]
                if len(shas) == 1:
                    sha = shas[0]
                elif len(shas) > 1:
                    click.echo(
                        f"Error: Cannot infer binary SHA-256: database contains {len(shas)} "
                        f"distinct binaries ({', '.join(sorted(shas))}); pass an explicit --sha",
                        err=True,
                    )
                    sys.exit(1)

            res = Il2CppMetadataReader.ingest(metadata_file, db, binary_sha256=sha)
            _emit_json({
                "status": "ok",
                "file": str(metadata_file),
                "binary_sha256": sha,
                "types_ingested": res["types_ingested"],
                "funcs_ingested": res["funcs_ingested"],
            })
    except MetadataParseError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    except StorageError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


@cli.command("ingest-ida")
@click.argument("doc_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--db",
    "db_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("toothpick.db"),
    show_default=True,
    help="SQLite database (toothpick.db) to ingest the IDA extract into",
)
@click.option(
    "--sha",
    "binary_sha256",
    type=str,
    default=None,
    help="Explicit binary SHA-256 for row attribution "
    "(defaults to the document provenance idb_sha256, noted in the report)",
)
def ingest_ida(doc_file: Path, db_path: Path, binary_sha256: Optional[str]) -> None:
    """Ingest an icepick schema-v1 IDA extract document into toothpick.db.

    DOC_FILE is a validated-on-load ``ida_extract.v1.json`` document: funcs land
    in ``funcs`` (decompiled text only, no SSA), call edges land in ``edges``
    (kind ``call``), provenance lands in ``receipts`` (phase ``ida_extract``,
    verifier ``UNGROUNDED``). IDA-proposed names go through the name-conflict
    queue -- a grounded name is never silently overwritten. Schema-invalid
    input is refused before the store is touched (exit 1, ``Error:``).
    """
    # Lazy import, the same discipline as `jvm ingest` above: the consumer is
    # only loaded for its own command, so every other command stays cheap.
    from toothpick import ida_consumer

    try:
        report = ida_consumer.ingest_extract_file(
            db_path, doc_file, binary_sha256=binary_sha256
        )
        _emit_json({"status": "ok", "file": str(doc_file), **report})
    except ida_consumer.IdaConsumerError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


@cli.command("build")
@click.argument("target", type=str)
@click.option(
    "--out",
    "out_dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Output directory for generated mod project (default ./build/<name>)",
)
@click.option(
    "--name",
    "plugin_name",
    type=str,
    default=None,
    help="Name of the BepInEx mod plugin",
)
@click.option(
    "--db",
    "db_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="SQLite graph database (toothpick.db) to verify against",
)
@click.option(
    "--allow-unverified",
    is_flag=True,
    help="Override the Epistemic Grounding Gate and build without confirmed dynamic receipts (§16 #1)",
)
@click.option(
    "--compile",
    "do_compile",
    is_flag=True,
    help="Compile the mod project into a .dll assembly using dotnet build",
)
@click.option(
    "--sha",
    "binary_sha256",
    type=str,
    default=None,
    help="Explicit binary SHA-256 for target attribution",
)
@click.option(
    "--hook-type",
    type=click.Choice(["prefix", "postfix", "replace"], case_sensitive=False),
    default="postfix",
    show_default=True,
    help="Harmony hook type to generate",
)
@click.option(
    "--logic",
    "mod_logic",
    type=str,
    default=None,
    help="Optional custom code snippet to insert into the hook",
)
@click.option(
    "--engine",
    type=click.Choice(["unity", "unreal", "godot"], case_sensitive=False),
    default="unity",
    show_default=True,
    help="Target game engine (unity, unreal, godot)",
)
@click.option(
    "--mod-type",
    type=str,
    default=None,
    help="Mod variant inside the engine (bepinex_csharp, ue4ss_lua, native_cpp, gdextension, gdscript)",
)
@click.option(
    "--godot-version",
    type=int,
    default=4,
    show_default=True,
    help="Target Godot version (3 or 4)",
)
def build_mod(
    target: str,
    out_dir: Optional[Path],
    plugin_name: Optional[str],
    db_path: Optional[Path],
    allow_unverified: bool,
    do_compile: bool,
    binary_sha256: Optional[str],
    hook_type: str,
    mod_logic: Optional[str],
    engine: str,
    mod_type: Optional[str],
    godot_version: int,
) -> None:
    """Build a mod project targeting a recovered function/method (Unity, Unreal, Godot)."""
    from toothpick.mod_builder import (
        CompilationError,
        ModBuilder,
        ModBuildError,
        UngroundedTargetError,
    )
    from toothpick.storage import StorageError

    if db_path is not None and not db_path.exists():
        click.echo(f"Error: database not found: {db_path}", err=True)
        sys.exit(1)

    target_db = db_path
    if target_db is None:
        default_db = Path("toothpick.db")
        if default_db.is_file():
            target_db = default_db

    output_dir = out_dir
    if output_dir is None:
        safe_name = plugin_name or "ToothpickMod"
        output_dir = Path("build") / safe_name

    try:
        with ModBuilder(db=target_db, binary_sha256=binary_sha256) as builder:
            artifact = builder.build(
                target=target,
                out_dir=output_dir,
                name=plugin_name,
                allow_unverified=allow_unverified,
                compile=do_compile,
                hook_type=hook_type,
                mod_logic=mod_logic,
                binary_sha256=binary_sha256,
                engine=engine,
                mod_type=mod_type,
                godot_version=godot_version,
            )
            _emit_json(artifact.to_dict())
    except UngroundedTargetError as exc:
        click.echo(f"Error (Ungrounded Target): {exc}", err=True)
        sys.exit(2)
    except (ModBuildError, StorageError, CompilationError) as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


@cli.command("rebase")
@click.argument("new_target", type=str)
@click.option(
    "--old-sha",
    "old_sha256",
    type=str,
    default=None,
    help="SHA-256 of the baseline binary to rebase from (inferred if db contains 1 binary)",
)
@click.option(
    "--new-sha",
    "new_sha256",
    type=str,
    default=None,
    help="Explicit SHA-256 of the updated binary",
)
@click.option(
    "--db",
    "db_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="SQLite database (toothpick.db) containing binary analysis records",
)
@click.option(
    "--threshold",
    type=float,
    default=None,
    help="Similarity threshold for auto-porting offsets (default 0.99 per §16 #2, 0.85 with --semantic)",
)
@click.option(
    "--semantic",
    is_flag=True,
    help="Semantic-embedding rebase: match via 64-D function vectors (Tier 2 at cosine >= 0.85)",
)
@click.option(
    "--dump-embeddings",
    is_flag=True,
    help="Attach per-function embedding blobs of the baseline binary to the report",
)
@click.option(
    "--out",
    "out_file",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional path to write full JSON rebase report",
)
@click.option(
    "--priors",
    "priors_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional icepick diff-import JSON: candidate pairs seed the scorer "
    "as priors (Tier-1 boost on name-confirmed pairs, bounded Tier-2 uplift; "
    "never depresses, never moves the gates)",
)
def rebase(
    new_target: str,
    old_sha256: Optional[str],
    new_sha256: Optional[str],
    db_path: Optional[Path],
    threshold: Optional[float],
    semantic: bool,
    dump_embeddings: bool,
    out_file: Optional[Path],
    priors_path: Optional[Path],
) -> None:
    """Rebase function offsets across binary updates (diff + sim-score per §10)."""
    from toothpick.rebase import RebaseEngine, RebaseError, load_priors
    from toothpick.storage import StorageError

    target_db = db_path
    if target_db is None:
        default_db = Path("toothpick.db")
        if default_db.is_file():
            target_db = default_db
        else:
            click.echo("Error: database not found; pass --db <path>", err=True)
            sys.exit(1)
    elif not target_db.exists():
        click.echo(f"Error: database not found: {target_db}", err=True)
        sys.exit(1)

    try:
        with RebaseEngine(db=target_db) as engine:
            sha = old_sha256 or ""
            if not sha:
                rows = engine.db._conn.execute(
                    "SELECT DISTINCT binary_sha256 FROM funcs WHERE binary_sha256 != ''"
                ).fetchall()
                shas = [r[0] for r in rows if r[0]]
                # If new_target is a SHA in the DB, exclude it from candidate old SHAs
                candidate_shas = [s for s in shas if s != new_target]
                if len(candidate_shas) == 1:
                    sha = candidate_shas[0]
                elif len(candidate_shas) > 1:
                    click.echo(
                        f"Error: Cannot infer baseline binary SHA-256: database contains {len(candidate_shas)} "
                        f"distinct binaries ({', '.join(sorted(candidate_shas))}); pass an explicit --old-sha",
                        err=True,
                    )
                    sys.exit(1)
                elif len(shas) == 1:
                    sha = shas[0]
                else:
                    click.echo("Error: No baseline binary functions found in database; pass --old-sha", err=True)
                    sys.exit(1)

            effective_threshold = (
                float(threshold)
                if threshold is not None
                else (0.85 if semantic else 0.99)
            )
            priors = None
            if priors_path is not None:
                try:
                    priors_doc = json.loads(priors_path.read_text(encoding="utf-8"))
                except (OSError, ValueError) as exc:
                    click.echo(f"Error: cannot read priors {priors_path}: {exc}", err=True)
                    sys.exit(1)
                try:
                    priors = load_priors(priors_doc)
                except RebaseError as exc:
                    click.echo(f"Error: invalid priors document: {exc}", err=True)
                    sys.exit(1)
            report = engine.rebase(
                old_sha256=sha,
                new_target=new_target,
                new_sha256=new_sha256,
                threshold=effective_threshold,
                priors=priors,
            )
            report_dict = report.to_dict()
            report_dict["semantic"] = bool(semantic)
            if semantic or dump_embeddings:
                embeddings = engine.embeddings_for(sha)
                report_dict["embeddings"] = {
                    f"0x{addr:x}": embedding.to_bytes().hex() for addr, embedding in embeddings.items()
                }
                report_dict["embedding_dim"] = 64
            if out_file is not None:
                out_file.write_text(json.dumps(report_dict, indent=2), encoding="utf-8")
            _emit_json(report_dict)
    except RebaseError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    except StorageError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


@cli.command("show")
@click.argument("func_addr", type=str)
@click.option(
    "--db",
    "db_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="SQLite database (toothpick.db) containing function records",
)
@click.option(
    "--sha",
    "binary_sha256",
    type=str,
    default=None,
    help="Explicit binary SHA-256 for target attribution (inferred if db contains 1 binary)",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json"], case_sensitive=False),
    default="text",
    show_default=True,
    help="Output format: human-readable debug overlay text or JSON",
)
def show(
    func_addr: str,
    db_path: Optional[Path],
    binary_sha256: Optional[str],
    output_format: str,
) -> None:
    """Render function debug overlay: AST, CFG, SSA facts, receipts, and type state."""
    from toothpick.observability import FunctionInspector, InspectionError
    from toothpick.query import QueryError, parse_address
    from toothpick.storage import StorageError

    target_db = db_path
    if target_db is None:
        default_db = Path("toothpick.db")
        if default_db.is_file():
            target_db = default_db
        else:
            click.echo("Error: database not found; pass --db <path>", err=True)
            sys.exit(1)
    elif not target_db.exists():
        click.echo(f"Error: database not found: {target_db}", err=True)
        sys.exit(1)

    try:
        addr = parse_address(func_addr)
    except QueryError as exc:
        click.echo(f"Error: Invalid function address {func_addr!r}: {exc}", err=True)
        sys.exit(1)

    try:
        with FunctionInspector(db=target_db) as inspector:
            overlay = inspector.inspect(addr, binary_sha256=binary_sha256)
            if output_format.lower() == "json":
                _emit_json(overlay.to_dict())
            else:
                click.echo(overlay.format_text())
    except InspectionError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    except StorageError as exc:
        click.echo(f"Error: Database storage error: {exc}", err=True)
        sys.exit(1)


@cli.command("ledger")
@click.option(
    "--db",
    "db_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="SQLite database (toothpick.db) to read trace_events from",
)
@click.option(
    "--events-log",
    "events_log_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Path to JSONL events.log file (default: events.log or TOOTHPICK_EVENTS_LOG)",
)
@click.option(
    "--sha",
    "binary_sha256",
    type=str,
    default=None,
    help="Optional binary SHA-256 to filter ledger aggregation",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Output summary in JSON format instead of table",
)
def ledger(
    db_path: Optional[Path],
    events_log_path: Optional[Path],
    binary_sha256: Optional[str],
    as_json: bool,
) -> None:
    """Display token and cost observability ledger aggregated per model class and phase."""
    from toothpick.observability import ObservabilityLedger, ObservabilityError
    from toothpick.storage import StorageError

    target_db = db_path
    if target_db is not None and not target_db.exists():
        click.echo(f"Error: database not found: {target_db}", err=True)
        sys.exit(1)
    if target_db is None and events_log_path is None:
        default_db = Path("toothpick.db")
        if default_db.is_file():
            target_db = default_db

    try:
        with ObservabilityLedger(db=target_db, events_log_path=events_log_path) as obs:
            summary = obs.get_summary(binary_sha256=binary_sha256)
            if as_json:
                _emit_json(summary.to_dict())
            else:
                click.echo(summary.format_table())
    except ObservabilityError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    except StorageError as exc:
        click.echo(f"Error: Database storage error: {exc}", err=True)
        sys.exit(1)


def _safe_relative_pck_path(path_in_pck: str) -> Optional[Path]:
    """Map a PCK internal path to a safe relative filesystem path.

    Thin wrapper over :func:`toothpick.mod_builder.safe_relative_pck_path`, which
    strips the ``res://`` / ``user://`` virtual prefixes as *prefixes* (never
    with :meth:`str.lstrip`, whose argument is a character set and which would
    corrupt names such as ``res://scene.tscn`` -> ``cene.tscn``) and rejects
    absolute paths and ``..`` traversal segments.  Returns ``None`` when the
    path cannot be safely mapped.
    """
    from toothpick.mod_builder import safe_relative_pck_path

    return safe_relative_pck_path(path_in_pck)


@cli.group("pck")
def pck_group() -> None:
    """Inspect, extract, and manipulate Godot .pck container archives."""
    pass


@pck_group.command("list")
@click.argument("pck_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def pck_list(pck_file: Path) -> None:
    """List all virtual files contained within a Godot .pck package."""
    from toothpick.mod_builder import GodotPckReader, PckParseError

    try:
        reader = GodotPckReader(pck_file)
        entries = reader.parse()
        result = {
            "status": "success",
            "pck_file": str(pck_file),
            "format_version": reader.format_version,
            "godot_version": f"{reader.version_major}.{reader.version_minor}.{reader.version_patch}",
            "file_count": len(entries),
            "entries": [
                {
                    "path": e.path,
                    "offset": e.offset,
                    "size": e.size,
                    "md5": e.md5.hex() if e.md5 else None,
                    "flags": e.flags,
                }
                for e in entries
            ],
        }
        _emit_json(result)
    except PckParseError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


@pck_group.command("extract")
@click.argument("pck_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--out",
    "out_dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("extracted_pck"),
    show_default=True,
    help="Destination directory for extracted files",
)
@click.option(
    "--file",
    "file_in_pck",
    type=str,
    default=None,
    help="Specific file path in PCK to extract",
)
def pck_extract(pck_file: Path, out_dir: Path, file_in_pck: Optional[str]) -> None:
    """Extract files from a Godot .pck container archive."""
    from toothpick.mod_builder import GodotPckReader, PckParseError

    try:
        reader = GodotPckReader(pck_file)
        if file_in_pck:
            rel = _safe_relative_pck_path(file_in_pck)
            if rel is None:
                click.echo(
                    f"Error: refusing unsafe PCK path {file_in_pck!r} "
                    "(absolute or parent-traversal path)",
                    err=True,
                )
                sys.exit(1)
            dest = out_dir / rel
            extracted_path = reader.extract_file(file_in_pck, dest)
            result = {
                "status": "success",
                "pck_file": str(pck_file),
                "extracted_files": [str(extracted_path)],
            }
        else:
            extracted = reader.extract_all(out_dir)
            result = {
                "status": "success",
                "pck_file": str(pck_file),
                "out_dir": str(out_dir),
                "extracted_count": len(extracted),
                "extracted_files": [str(p) for p in extracted],
            }
        _emit_json(result)
    except PckParseError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)



# -- W15: headless WinDbg-cdb bridge + minidump crash analysis (Chunk 3) --


def _verify_module_address(addr: Optional[int], modules: Any) -> Optional[Dict[str, Any]]:
    """Report whether ``addr`` resolves inside one of the loaded modules.

    Returns ``None`` when no address was requested, so callers can omit the key
    entirely instead of emitting a meaningless ``resolved: false``.
    """
    if addr is None:
        return None
    address = int(addr)
    for module in modules:
        if module.contains(address):
            return {
                "addr": address,
                "addr_hex": f"0x{address:x}",
                "resolved": True,
                "module": module.to_dict(),
            }
    return {
        "addr": address,
        "addr_hex": f"0x{address:x}",
        "resolved": False,
        "module": None,
    }


def _resolve_cdb_sha(db: Any, binary_sha256: Optional[str]) -> str:
    """Resolve the binary SHA-256 stamped on a ``cdb_trace`` receipt.

    Inference is only allowed when the store holds exactly one binary; an
    ambiguous store is an error rather than a mis-attributed receipt, and an
    empty store yields ``""`` (no attribution), mirroring
    :meth:`toothpick.ground.GroundingHarness._resolve_sha`.
    """
    if binary_sha256:
        return str(binary_sha256)
    rows = db._conn.execute(
        "SELECT DISTINCT binary_sha256 FROM funcs WHERE binary_sha256 != ''"
    ).fetchall()
    shas = sorted({str(row[0]) for row in rows if row[0]})
    if len(shas) == 1:
        return shas[0]
    if len(shas) > 1:
        raise StorageError(
            f"Cannot infer binary SHA-256: database contains {len(shas)} distinct "
            f"binaries ({', '.join(shas)}); pass an explicit --sha"
        )
    return ""


@cli.command("cdb")
@click.option(
    "--pid",
    type=int,
    default=None,
    help="Process ID of a running target for a non-invasive cdb attach (cdb -pv -p)",
)
@click.option(
    "--dump",
    "dump_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Crash dump / minidump (.dmp) to analyze offline",
)
@click.option(
    "--cmd",
    "cdb_commands",
    multiple=True,
    help="CDB command to run in the live session (repeatable; default r, kp, lm, q)",
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional path to also write the JSON report to",
)
@click.option(
    "--verify-addr",
    "verify_addr_text",
    type=str,
    default=None,
    help="Address to verify against the module map of the session/crash dump",
)
@click.option(
    "--db",
    "db_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="SQLite database (toothpick.db) to persist the cdb_trace receipt into",
)
@click.option(
    "--sha",
    "binary_sha256",
    type=str,
    default=None,
    help="Explicit binary SHA-256 for receipt attribution (inferred when unambiguous)",
)
@click.option(
    "--timeout",
    type=float,
    default=None,
    help="Timeout in seconds for the non-invasive cdb session (default 30.0)",
)
def cdb(
    pid: Optional[int],
    dump_path: Optional[Path],
    cdb_commands: Tuple[str, ...],
    out_path: Optional[Path],
    verify_addr_text: Optional[str],
    db_path: Optional[Path],
    binary_sha256: Optional[str],
    timeout: Optional[float],
) -> None:
    """Inspect a live process non-invasively (--pid) or analyze a crash dump (--dump)."""
    from toothpick.storage import Database
    from toothpick.windbg import (
        CDB_PHASE,
        DEFAULT_ATTACH_COMMANDS,
        DEFAULT_CDB_TIMEOUT,
        CdbEngine,
        CdbError,
    )

    if pid is None and dump_path is None:
        click.echo(
            "Error: Must specify either --pid (non-invasive live attach) or "
            "--dump (crash dump analysis)",
            err=True,
        )
        sys.exit(1)
    if pid is not None and dump_path is not None:
        click.echo(
            "Error: Mutually exclusive targets: specify only one of --pid or --dump",
            err=True,
        )
        sys.exit(1)

    verify_addr: Optional[int] = None
    if verify_addr_text is not None:
        try:
            verify_addr = parse_address(verify_addr_text)
        except QueryError as exc:
            click.echo(f"Error: {exc}", err=True)
            sys.exit(1)

    active_path = Path(dump_path) if dump_path is not None else None
    if active_path is not None and not active_path.is_file():
        click.echo(f"Error: crash dump not found: {active_path}", err=True)
        sys.exit(1)

    budget = DEFAULT_CDB_TIMEOUT if timeout is None else float(timeout)
    engine = CdbEngine(timeout=budget)

    modules: Any = []
    report: Dict[str, Any]
    receipt_payload: Dict[str, Any]
    func_addr = int(verify_addr) if verify_addr is not None else 0

    if active_path is not None:
        try:
            crash = engine.analyze_crash_dump(active_path)
        except CdbError as exc:
            click.echo(f"Error: {exc}", err=True)
            sys.exit(1)
        crash_dict = crash.to_dict()
        modules = crash.modules
        func_addr = int(crash.faulting_ip)
        receipt_payload = crash.receipt_payload()
        report = {
            "command": "cdb",
            "status": "ok",
            "mode": "crash_dump",
            "phase": CDB_PHASE,
            "verifier_status": "GROUNDED",
            "dump_path": str(active_path),
            "exception_code": crash_dict["exception_code"],
            "exception_code_hex": crash_dict["exception_code_hex"],
            "exception_name": crash_dict["exception_name"],
            "faulting_ip": crash_dict["faulting_ip"],
            "faulting_ip_hex": crash_dict["faulting_ip_hex"],
            "faulting_module": crash_dict["faulting_module"],
            "crash": crash_dict,
        }
    else:
        commands = [str(c).strip() for c in cdb_commands if str(c).strip()]
        if not commands:
            commands = list(DEFAULT_ATTACH_COMMANDS)
        try:
            session = engine.attach_non_invasive(int(pid or 0), commands, timeout=budget)
        except CdbError as exc:
            click.echo(f"Error: {exc}", err=True)
            sys.exit(1)
        session_dict = session.to_dict()
        modules = session.modules
        receipt_payload = {
            "ground_type": "CDB_LIVE_ATTACH",
            "verifier_status": "GROUNDED",
            "phase": CDB_PHASE,
            "session": session_dict,
        }
        report = {
            "command": "cdb",
            "status": "ok",
            "mode": "non_invasive_attach",
            "phase": CDB_PHASE,
            "verifier_status": "GROUNDED",
            "pid": int(session.pid),
            "commands": list(session.commands),
            "cdb_executable": session.cdb_executable,
            "registers": session_dict["registers"],
            "stack_trace": session_dict["stack_trace"],
            "modules": session_dict["modules"],
            "session": session_dict,
        }

    if verify_addr is not None:
        verification = _verify_module_address(verify_addr, modules)
        if verification is not None:
            receipt_payload = dict(receipt_payload)
            receipt_payload["address_verification"] = verification
            report["address_verification"] = verification

    if db_path is not None:
        try:
            with Database(db_path, create=True) as db:
                sha = _resolve_cdb_sha(db, binary_sha256)
                receipt_id = db.add_receipt(sha, func_addr, CDB_PHASE, receipt_payload)
        except StorageError as exc:
            click.echo(f"Error: {exc}", err=True)
            sys.exit(1)
        report["binary_sha256"] = sha
        report["receipt_id"] = receipt_id
        report["receipt"] = {
            "receipt_id": receipt_id,
            "binary_sha256": sha,
            "func_addr": func_addr,
            "phase": CDB_PHASE,
            "payload": receipt_payload,
        }
    elif binary_sha256:
        report["binary_sha256"] = binary_sha256

    if out_path is not None:
        try:
            out_path.write_text(
                json.dumps(report, indent=2, sort_keys=True, default=str),
                encoding="utf-8",
            )
        except OSError as exc:
            click.echo(f"Error: cannot write {out_path}: {exc}", err=True)
            sys.exit(1)
        report["out_path"] = str(out_path)

    _emit_json(report)


@cli.command("dbi")
@click.option(
    "--target",
    "target_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Target binary file to trace under offline Unicorn emulation",
)
@click.option(
    "--engine",
    "engine_choice",
    type=click.Choice(["unicorn", "pin", "dynamorio"]),
    default="unicorn",
    show_default=True,
    help="DBI engine: unicorn (offline emulation) or pin/dynamorio (log parsing)",
)
@click.option(
    "--trace-bbs",
    is_flag=True,
    help="Record executed basic blocks during a Unicorn trace (default on)",
)
@click.option(
    "--generate",
    "generate_backend",
    type=click.Choice(["pin", "dynamorio"]),
    default=None,
    help="Generate an instrumentation tool project instead of tracing",
)
@click.option(
    "--tool",
    "tool_type",
    type=str,
    default="memtrace",
    show_default=True,
    help="Tool type to generate: bbtrace, inscount, or memtrace",
)
@click.option(
    "--parse",
    "parse_log",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Parse a raw Pin/DynamoRIO execution log into the DBI trace schema",
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional path to also write the JSON report to",
)
@click.option(
    "--db",
    "db_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="SQLite database (toothpick.db) to persist the dbi_trace receipt into",
)
@click.option(
    "--sha",
    "binary_sha256",
    type=str,
    default=None,
    help="Explicit binary SHA-256 for receipt attribution (inferred when unambiguous)",
)
@click.option(
    "--max-instructions",
    type=int,
    default=None,
    help="Instruction budget bounding one Unicorn emulation run",
)
@click.option(
    "--timeout",
    type=float,
    default=None,
    help="Wall-clock guard in seconds for one Unicorn emulation run",
)
def dbi(
    target_path: Optional[Path],
    engine_choice: str,
    trace_bbs: bool,
    generate_backend: Optional[str],
    tool_type: str,
    parse_log: Optional[Path],
    out_path: Optional[Path],
    db_path: Optional[Path],
    binary_sha256: Optional[str],
    max_instructions: Optional[int],
    timeout: Optional[float],
) -> None:
    """Trace execution with DBI backends, generate Pintools, or parse trace logs."""
    from toothpick.dbi import (
        DBI_PHASE,
        DbiError,
        DbiToolGenerator,
        TraceParser,
        UnicornDbiTracer,
    )
    from toothpick.storage import Database

    modes = sum(
        x is not None for x in (target_path, generate_backend, parse_log)
    )
    if modes == 0:
        click.echo(
            "Error: Must specify one of --target (Unicorn trace), "
            "--generate (tool scaffolding), or --parse (log ingestion)",
            err=True,
        )
        sys.exit(1)
    if modes > 1:
        click.echo(
            "Error: Mutually exclusive modes: specify only one of "
            "--target, --generate, or --parse",
            err=True,
        )
        sys.exit(1)

    report: Dict[str, Any]
    receipt_payload: Optional[Dict[str, Any]] = None
    func_addr = 0

    if generate_backend is not None:
        if out_path is None:
            click.echo("Error: --out directory is required with --generate", err=True)
            sys.exit(1)
        try:
            if generate_backend == "pin":
                source = DbiToolGenerator.generate_pintool(tool_type, out_path)
            else:
                source = DbiToolGenerator.generate_dynamorio_client(tool_type, out_path)
        except DbiError as exc:
            click.echo(f"Error: {exc}", err=True)
            sys.exit(1)
        report = {
            "command": "dbi",
            "status": "ok",
            "mode": "generate",
            "backend": generate_backend,
            "tool": tool_type,
            "source": str(source),
            "out_dir": str(out_path),
        }
    elif parse_log is not None:
        log_path = Path(parse_log)
        if not log_path.is_file():
            click.echo(f"Error: trace log not found: {log_path}", err=True)
            sys.exit(1)
        try:
            if engine_choice == "dynamorio":
                result = TraceParser.parse_dynamorio_log(log_path)
            else:
                result = TraceParser.parse_pin_log(log_path)
        except DbiError as exc:
            click.echo(f"Error: {exc}", err=True)
            sys.exit(1)
        payload = result.to_dict()
        func_addr = int(result.executed_bbs[0]) if result.executed_bbs else 0
        receipt_payload = {
            "ground_type": "DBI_LOG_PARSE",
            "verifier_status": "GROUNDED",
            "phase": DBI_PHASE,
            "engine": result.engine,
            "trace": payload,
        }
        report = {
            "command": "dbi",
            "status": "ok",
            "mode": "parse",
            "phase": DBI_PHASE,
            "verifier_status": "GROUNDED",
            "log_path": str(log_path),
            "trace": payload,
        }
    else:
        blob_path = Path(target_path) if target_path is not None else None
        if blob_path is None or not blob_path.is_file():
            click.echo(f"Error: target not found: {blob_path}", err=True)
            sys.exit(1)
        try:
            tracer_kwargs: Dict[str, Any] = {}
            if max_instructions is not None:
                tracer_kwargs["max_instructions"] = int(max_instructions)
            tracer = UnicornDbiTracer(**tracer_kwargs)
            trace_kwargs: Dict[str, Any] = {}
            if max_instructions is not None:
                trace_kwargs["max_instructions"] = int(max_instructions)
            if timeout is not None:
                trace_kwargs["timeout_sec"] = float(timeout)
            result = tracer.trace(blob_path, **trace_kwargs)
        except DbiError as exc:
            click.echo(f"Error: {exc}", err=True)
            sys.exit(1)
        payload = result.to_dict()
        if not trace_bbs:
            payload = dict(payload)
            payload["executed_bbs"] = []
        func_addr = int(result.executed_bbs[0]) if result.executed_bbs else 0
        receipt_payload = {
            "ground_type": "DBI_UNICORN_TRACE",
            "verifier_status": "GROUNDED",
            "phase": DBI_PHASE,
            "engine": result.engine,
            "stop_reason": tracer.last_stop_reason,
            "trace": payload,
        }
        report = {
            "command": "dbi",
            "status": "ok",
            "mode": "unicorn_trace",
            "phase": DBI_PHASE,
            "verifier_status": "GROUNDED",
            "target": str(blob_path),
            "trace": payload,
            "stop_reason": tracer.last_stop_reason,
        }

    if db_path is not None and receipt_payload is not None:
        try:
            with Database(db_path, create=True) as db:
                sha = _resolve_cdb_sha(db, binary_sha256)
                receipt_id = db.add_receipt(sha, func_addr, DBI_PHASE, receipt_payload)
        except StorageError as exc:
            click.echo(f"Error: {exc}", err=True)
            sys.exit(1)
        report["binary_sha256"] = sha
        report["receipt_id"] = receipt_id
        report["receipt"] = {
            "receipt_id": receipt_id,
            "binary_sha256": sha,
            "func_addr": func_addr,
            "phase": DBI_PHASE,
            "payload": receipt_payload,
        }
    elif binary_sha256:
        report["binary_sha256"] = binary_sha256

    if out_path is not None and generate_backend is None:
        try:
            out_path.write_text(
                json.dumps(report, indent=2, sort_keys=True, default=str),
                encoding="utf-8",
            )
        except OSError as exc:
            click.echo(f"Error: cannot write {out_path}: {exc}", err=True)
            sys.exit(1)
        report["out_path"] = str(out_path)
    elif out_path is not None:
        report["out_dir"] = str(out_path)

    _emit_json(report)


@cli.command("taint")
@click.option(
    "--binary",
    "binary_path",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="Target binary file whose bytes are analyzed",
)
@click.option(
    "--func",
    "func_text",
    type=str,
    default=None,
    help="Function address hex to analyze (defaults to the load base)",
)
@click.option(
    "--source",
    "source",
    type=str,
    default=None,
    help="Taint source: register name (rsi) or address (0x... / mem:0x...)",
)
@click.option(
    "--sink",
    "sink_text",
    type=str,
    default=None,
    help="Sink address hex checked for tainted-state use",
)
@click.option(
    "--arch",
    type=str,
    default="x86_64",
    show_default=True,
    help="Architecture used to decode flat binary bytes",
)
@click.option(
    "--base",
    "base_text",
    type=str,
    default="0x1000",
    show_default=True,
    help="Load base address hex for flat binary bytes",
)
@click.option(
    "--export",
    "export_framework",
    type=click.Choice(["triton", "manticore"]),
    default=None,
    help="Write a Triton/Manticore script instead of analyzing",
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional path to also write the JSON report (analyze) or script (export) to",
)
@click.option(
    "--db",
    "db_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="SQLite database (toothpick.db) to persist the taint_trace receipt into",
)
@click.option(
    "--sha",
    "binary_sha256",
    type=str,
    default=None,
    help="Explicit binary SHA-256 for receipt attribution (inferred when unambiguous)",
)
def taint(
    binary_path: Path,
    func_text: Optional[str],
    source: Optional[str],
    sink_text: Optional[str],
    arch: str,
    base_text: str,
    export_framework: Optional[str],
    out_path: Optional[Path],
    db_path: Optional[Path],
    binary_sha256: Optional[str],
) -> None:
    """Run register/memory taint analysis or export symbolic-framework scripts."""
    from toothpick.symbolic import (
        SymbolicError,
        TaintEngine,
        disassemble_bytes,
        exporter_for,
    )
    from toothpick.storage import Database

    target = Path(binary_path)
    if not target.is_file():
        click.echo(f"Error: binary not found: {target}", err=True)
        sys.exit(1)

    if export_framework is not None:
        try:
            exporter = exporter_for(str(export_framework), target)
            script = exporter.generate()
        except SymbolicError as exc:
            click.echo(f"Error: {exc}", err=True)
            sys.exit(1)
        report = {
            "command": "taint",
            "status": "ok",
            "mode": "export",
            "framework": str(export_framework),
            "binary": str(target),
            "script_bytes": len(script.encode("utf-8")),
        }
        if out_path is not None:
            try:
                out_path.write_text(script, encoding="utf-8")
            except OSError as exc:
                click.echo(f"Error: cannot write {out_path}: {exc}", err=True)
                sys.exit(1)
            report["out_path"] = str(out_path)
        _emit_json(report)
        return

    if not source:
        click.echo("Error: --source is required for taint analysis", err=True)
        sys.exit(1)
    if not sink_text:
        click.echo("Error: --sink is required for taint analysis", err=True)
        sys.exit(1)
    try:
        base = parse_address(base_text) if base_text else 0x1000
        func_addr = parse_address(func_text) if func_text else base
        sink_addr = parse_address(sink_text) if sink_text else 0
    except QueryError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    try:
        code = target.read_bytes()
    except OSError as exc:
        click.echo(f"Error: cannot read {target}: {exc}", err=True)
        sys.exit(1)
    if not code:
        click.echo(f"Error: binary is empty: {target}", err=True)
        sys.exit(1)
    try:
        engine = TaintEngine(arch=arch)
        offset = max(0, int(func_addr) - int(base))
        insns = disassemble_bytes(code[offset:], base_addr=int(func_addr), arch=arch)
        taint_report = engine.analyze_function(insns, source=source, sink_addr=sink_addr)
    except SymbolicError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    body = taint_report.to_dict()
    receipt_payload = {
        "ground_type": "TAINT_STATIC",
        "verifier_status": "GROUNDED" if body["is_sink_tainted"] else "UNGROUNDED",
        "phase": "taint_trace",
        "arch": arch,
        "report": body,
    }
    report = {
        "command": "taint",
        "status": "ok",
        "mode": "analyze",
        "phase": "taint_trace",
        "binary": str(target),
        "report": body,
    }
    if db_path is not None:
        try:
            with Database(db_path, create=True) as db:
                sha = _resolve_cdb_sha(db, binary_sha256)
                receipt_id = db.add_receipt(sha, int(func_addr), "taint_trace", receipt_payload)
        except StorageError as exc:
            click.echo(f"Error: {exc}", err=True)
            sys.exit(1)
        report["binary_sha256"] = sha
        report["receipt_id"] = receipt_id
        report["receipt"] = {
            "receipt_id": receipt_id,
            "binary_sha256": sha,
            "func_addr": int(func_addr),
            "phase": "taint_trace",
            "payload": receipt_payload,
        }
    elif binary_sha256:
        report["binary_sha256"] = binary_sha256
    if out_path is not None:
        try:
            out_path.write_text(
                json.dumps(report, indent=2, sort_keys=True, default=str),
                encoding="utf-8",
            )
        except OSError as exc:
            click.echo(f"Error: cannot write {out_path}: {exc}", err=True)
            sys.exit(1)
        report["out_path"] = str(out_path)
    _emit_json(report)
@cli.command("struct")
@click.option(
    "--file",
    "file_target",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Memory dump or raw binary file to synthesize the struct from",
)
@click.option(
    "--pid",
    type=int,
    default=None,
    help="Process ID of a running target to read memory from via Frida",
)
@click.option(
    "--address",
    "address_text",
    type=str,
    default=None,
    help="Base address hex of the struct (also used with --pid for live reads)",
)
@click.option(
    "--base",
    "base_text",
    type=str,
    default=None,
    help="Base address hex annotating the layout (defaults to --address or 0x0)",
)
@click.option(
    "--size",
    "size_text",
    type=str,
    default="0x100",
    show_default=True,
    help="Bytes to synthesize (hex or int)",
)
@click.option(
    "--name",
    "struct_name",
    type=str,
    default="SynthesizedStruct",
    show_default=True,
    help="Struct name used in generated code",
)
@click.option(
    "--format",
    "format_choice",
    type=click.Choice(["rcnet", "cpp", "csharp", "json"]),
    default="rcnet",
    show_default=True,
    help="Output format",
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional path to write the generated output to",
)
@click.option(
    "--db",
    "db_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="SQLite database (toothpick.db) to ingest the layout into",
)
@click.option(
    "--sha",
    "binary_sha256",
    type=str,
    default=None,
    help="Explicit binary SHA-256 for type attribution (inferred when unambiguous)",
)
def struct(
    file_target: Optional[Path],
    pid: Optional[int],
    address_text: Optional[str],
    base_text: Optional[str],
    size_text: str,
    struct_name: str,
    format_choice: str,
    out_path: Optional[Path],
    db_path: Optional[Path],
    binary_sha256: Optional[str],
) -> None:
    """Synthesize struct layouts from memory and export ReClass/C++/C# code."""
    from toothpick.reclass import ReclassError, StructSynthesizer, ingest_layout
    from toothpick.storage import Database

    if file_target is None and pid is None:
        click.echo("Error: Must specify either --file or --pid", err=True)
        sys.exit(1)
    if file_target is not None and pid is not None:
        click.echo("Error: Mutually exclusive sources: --file or --pid", err=True)
        sys.exit(1)
    try:
        base = parse_address(base_text) if base_text else (
            parse_address(address_text) if address_text else 0
        )
        size = parse_address(size_text) if size_text else 0x100
    except QueryError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    if size <= 0:
        click.echo(f"Error: --size must be positive (got {size_text})", err=True)
        sys.exit(1)
    try:
        if file_target is not None:
            data = Path(file_target).read_bytes()[: int(size)]
        else:
            from toothpick.scanner import FridaMemoryBackend

            backend = FridaMemoryBackend(int(pid or 0))
            try:
                addr = parse_address(address_text) if address_text else base
                data = backend.read_memory(int(addr), int(size))
            finally:
                backend.close()
    except QueryError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    except OSError as exc:
        click.echo(f"Error: cannot read source: {exc}", err=True)
        sys.exit(1)
    except Exception as exc:
        click.echo(f"Error: memory read failed: {exc}", err=True)
        sys.exit(1)
    if not data:
        click.echo("Error: no bytes available for synthesis", err=True)
        sys.exit(1)
    try:
        layout = StructSynthesizer().synthesize_from_memory(
            bytes(data), base_addr=int(base), name=struct_name
        )
    except ReclassError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    if format_choice == "cpp":
        rendered = layout.to_cpp()
    elif format_choice == "csharp":
        rendered = layout.to_csharp()
    elif format_choice == "json":
        rendered = layout.to_json()
    else:
        rendered = layout.to_reclass_xml()
    report: Dict[str, Any] = {
        "command": "struct",
        "status": "ok",
        "phase": "struct_synth",
        "name": layout.name,
        "format": format_choice,
        "layout": layout.to_dict(),
        "rendered": rendered,
    }
    if db_path is not None:
        try:
            with Database(db_path, create=True) as db:
                sha = _resolve_cdb_sha(db, binary_sha256)
                type_id = ingest_layout(db, layout, binary_sha256=sha)
        except StorageError as exc:
            click.echo(f"Error: {exc}", err=True)
            sys.exit(1)
        report["binary_sha256"] = sha
        report["type_id"] = type_id
        report["receipt"] = {
            "type_id": type_id,
            "binary_sha256": sha,
            "phase": "struct_synth",
        }
    elif binary_sha256:
        report["binary_sha256"] = binary_sha256
    if out_path is not None:
        try:
            out_path.write_text(rendered, encoding="utf-8")
        except OSError as exc:
            click.echo(f"Error: cannot write {out_path}: {exc}", err=True)
            sys.exit(1)
        report["out_path"] = str(out_path)
    _emit_json(report)


# -- P1: sampled-stack profile ingestion (UNIT-P1-PROFILE-PLAN.md section 2) --


@cli.group("profile")
def profile() -> None:
    """Sampled-stack profiles: import VTune/ETW/perf CSVs as per-function hotness."""
    pass


@profile.command("import")
@click.argument("csv_path", type=click.Path(dir_okay=False, path_type=Path))
@click.option(
    "--db",
    "db_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("toothpick.db"),
    show_default=True,
    help="SQLite graph database (toothpick.db) to attribute the profile into",
)
@click.option(
    "--sha",
    "binary_sha256",
    type=str,
    default=None,
    help="binary_sha256 the profile belongs to (inferred when the db holds one binary)",
)
@click.option(
    "--format",
    "profile_format",
    type=click.Choice(["auto", "vtune", "etw", "perf", "async", "jfr"], case_sensitive=False),
    default="auto",
    show_default=True,
    help="Profile format: auto sniffs the input; async = async-profiler collapsed stacks; jfr = jfr print --json (no binary .jfr decoder)",
)
def profile_import(
    csv_path: Path,
    db_path: Optional[Path],
    binary_sha256: Optional[str],
    profile_format: str,
) -> None:
    """Import a sampled-stack profile (auto|vtune|etw|perf|async|jfr) into toothpick.db.

    ``vtune``/``etw``/``perf`` are the P1 delimited formats.  ``async`` accepts
    async-profiler collapsed stacks (``frame;frame;frame <count> [<alloc>]``); ``jfr``
    accepts a ``jfr print --json`` event projection -- no binary ``.jfr`` decoder is
    built here (the ecosystem tools own that layer).  ``auto`` sniffs which of the
    five families the input is.

    Emits ``{imported, unattributed, rejected, total_samples}`` (plan section 1.4).
    Samples are attributed to functions the analysis has already observed; an
    unmatched sample is counted ``unattributed``, never fabricated into a row.
    """
    from toothpick.profile import ProfileError, import_samples, parse_profile_file
    from toothpick.storage import Database, StorageError

    target_db = db_path or Path("toothpick.db")
    if not target_db.exists():
        click.echo(f"Error: database not found: {target_db}", err=True)
        sys.exit(1)

    try:
        with Database(target_db) as db:
            sha = binary_sha256 or ""
            if not sha:
                # Rebase-command inference (see `rebase` above): only a store
                # holding exactly one binary may be assumed -- guessing among
                # several would mis-attribute every imported sample.
                rows = db._conn.execute(
                    "SELECT DISTINCT binary_sha256 FROM funcs WHERE binary_sha256 != ''"
                ).fetchall()
                shas = sorted({str(row[0]) for row in rows if row[0]})
                if len(shas) == 1:
                    sha = shas[0]
                elif len(shas) > 1:
                    click.echo(
                        "Error: Cannot infer binary SHA-256: database contains "
                        f"{len(shas)} distinct binaries ({', '.join(shas)}); "
                        "pass an explicit --sha",
                        err=True,
                    )
                    sys.exit(1)
                else:
                    click.echo(
                        "Error: Cannot infer binary SHA-256: database holds no "
                        "functions; pass an explicit --sha",
                        err=True,
                    )
                    sys.exit(1)

            samples, diagnostics = parse_profile_file(
                csv_path, format=profile_format
            )
            result = import_samples(db, sha, (samples, diagnostics))
    except (ProfileError, StorageError) as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    result["command"] = "profile import"
    result["status"] = "ok"
    result["db"] = str(target_db)
    result["profile_path"] = str(csv_path)
    result["format"] = diagnostics.get("format", profile_format)
    _emit_json(result)


@profile.command("hottest")
@click.option(
    "--db",
    "db_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("toothpick.db"),
    show_default=True,
    help="SQLite graph database (toothpick.db) holding the imported profile",
)
@click.option(
    "--sha",
    "binary_sha256",
    type=str,
    default=None,
    help="binary_sha256 to list hotness for (inferred when the db holds one binary)",
)
@click.option(
    "--top",
    "top_n",
    type=int,
    default=50,
    show_default=True,
    help="Maximum number of functions to return (ceiling 1000)",
)
def profile_hottest(
    db_path: Optional[Path],
    binary_sha256: Optional[str],
    top_n: int,
) -> None:
    """List the hottest functions of a profile-imported binary (top-K, R1).

    Emits ``{db, binary_sha256, top, count, hottest}``: a **bounded** list
    ordered ``hotness DESC, func_addr ASC`` (plan section 1.3), so the same call
    returns the same list and an unbounded response is unreachable.
    """
    # Risk R1 ceiling, the same shape as `scan --limit` above: an out-of-range
    # --top is refused with a clear error rather than silently clamped, so a
    # caller asking for 10_000 learns the budget is 1000 instead of believing it
    # received everything.
    if top_n < 1 or top_n > 1000:
        click.echo(
            f"Error: --top must be between 1 and 1000 (got {top_n})",
            err=True,
        )
        sys.exit(1)

    target_db = db_path or Path("toothpick.db")
    if not target_db.exists():
        click.echo(f"Error: database not found: {target_db}", err=True)
        sys.exit(1)

    try:
        # Imported here rather than at module scope (as `profile import` above):
        # the CLI keeps no import-time dependency on the query/profile
        # subsystems, and the store is opened only on the path that needs it.
        from toothpick.query import QueryEngine

        with QueryEngine(target_db) as engine:
            sha = engine.resolve_sha(binary_sha256)
            items = engine.hottest(sha, top_n)
            budget = engine.resolve_limit(top_n)
    except (QueryError, StorageError) as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    _emit_json(
        {
            "command": "profile hottest",
            "status": "ok",
            "db": str(target_db),
            "binary_sha256": sha,
            "top": budget,
            "count": len(items),
            "hottest": items,
        }
    )


# -- P2: generic native hook artifact (UNIT-P2-HOOK-PLAN.md section 2) --


@cli.group("hook")
def hook_group() -> None:
    """Generate native hook artifacts (injected MinHook DLL) for grounded targets."""
    pass


@hook_group.command("build")
@click.option(
    "--target",
    "target_text",
    type=str,
    required=True,
    help="Target function address to hook (hex, e.g. 0x140001000, or decimal)",
)
@click.option(
    "--db",
    "db_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("toothpick.db"),
    show_default=True,
    help="SQLite graph database (toothpick.db) holding the grounding receipts",
)
@click.option(
    "--sha",
    "binary_sha256",
    type=str,
    default=None,
    help="binary_sha256 the target belongs to (inferred when the db holds one binary)",
)
@click.option(
    "--out",
    "out_dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Output project directory for the generated hook project",
)
@click.option(
    "--name",
    type=str,
    default="hook",
    show_default=True,
    help="Project/DLL name of the generated hook artifact",
)
@click.option(
    "--arch",
    type=click.Choice(["x86", "x64"], case_sensitive=False),
    default="x64",
    show_default=True,
    help="Architecture of the injected DLL (one project per architecture)",
)
@click.option(
    "--prologue-hex",
    "prologue_hex",
    type=str,
    default=None,
    help="Optional caller-supplied prologue bytes (hex) to validate; never invented",
)
@click.option(
    "--allow-unverified",
    is_flag=True,
    help="Override the Epistemic Grounding Gate and emit an UNVERIFIED artifact (Design §16 #1)",
)
def hook_build(
    target_text: str,
    db_path: Optional[Path],
    binary_sha256: Optional[str],
    out_dir: Path,
    name: str,
    arch: str,
    prologue_hex: Optional[str],
    allow_unverified: bool,
) -> None:
    """Build a MinHook injected-DLL project for TARGET (receipts-or-refusal).

    ``toothpick.hook`` is imported lazily so the CLI keeps no import-time
    dependency on Capstone.  The Epistemic Grounding Gate refuses an address with
    no confirmed ``GROUNDED`` receipt unless ``--allow-unverified`` is passed, in
    which case the emitted project is stamped ``UNVERIFIED`` and never claims
    grounding.  A prologue is validated only when ``--prologue-hex`` supplies it
    (never invented here).
    """
    from toothpick.hook import (
        HookError,
        HookGate,
        HookTarget,
        generate_hook_project,
        validate_prologue,
    )
    from toothpick.mod_builder import UngroundedTargetError
    from toothpick.storage import Database, StorageError

    try:
        func_addr = parse_address(target_text)
    except QueryError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    # Prologue bytes are purely caller-supplied: when --prologue-hex is absent the
    # trampoline length stays UNVALIDATED rather than being guessed (Design §16 #1).
    prologue_bytes = b""
    if prologue_hex is not None:
        text = prologue_hex.strip()
        if text.lower().startswith("0x"):
            text = text[2:]
        text = "".join(text.split())
        try:
            prologue_bytes = bytes.fromhex(text)
        except ValueError:
            prologue_bytes = b""
        if not prologue_bytes:
            click.echo(
                "Error: --prologue-hex must be an even-length hex string of at "
                f"least one byte (got {prologue_hex!r}); nothing is invented for you",
                err=True,
            )
            sys.exit(1)
        verdict = validate_prologue(prologue_bytes, arch)
        if not verdict["ok"]:
            click.echo(f"Error: prologue rejected: {verdict['reason']}", err=True)
            sys.exit(1)

    try:
        target = HookTarget(
            func_addr=func_addr, arch=arch, prologue_bytes=prologue_bytes
        )
    except HookError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    target_db = db_path or Path("toothpick.db")
    if not target_db.exists():
        # A typo'd --db is an error, not an empty store (see _engine_for above):
        # without the store there is no receipt to enforce the gate against.
        click.echo(f"Error: database not found: {target_db}", err=True)
        sys.exit(1)

    try:
        with Database(target_db) as db:
            sha = binary_sha256 or ""
            if not sha:
                # Rebase/profile inference pattern: only a store holding exactly
                # one binary may be assumed -- guessing among several would pin
                # another binary's receipt to this address.
                rows = db._conn.execute(
                    "SELECT DISTINCT binary_sha256 FROM funcs WHERE binary_sha256 != '' "
                    "UNION SELECT DISTINCT binary_sha256 FROM receipts "
                    "WHERE binary_sha256 != ''"
                ).fetchall()
                shas = sorted({str(row[0]) for row in rows if row[0]})
                if len(shas) == 1:
                    sha = shas[0]
                elif len(shas) > 1:
                    click.echo(
                        "Error: Cannot infer binary SHA-256: database contains "
                        f"{len(shas)} distinct binaries ({', '.join(shas)}); "
                        "pass an explicit --sha",
                        err=True,
                    )
                    sys.exit(1)
                else:
                    click.echo(
                        "Error: Cannot infer binary SHA-256: database holds no "
                        "functions or receipts; pass an explicit --sha",
                        err=True,
                    )
                    sys.exit(1)

            # Raises UngroundedTargetError unless a GROUNDED receipt backs the
            # address or the operator explicitly waived the gate.
            grounded = HookGate.enforce(db, sha, target, allow_unverified)
    except UngroundedTargetError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    except StorageError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    try:
        result = generate_hook_project(
            target,
            out_dir,
            name=name,
            grounded=grounded,
        )
    except HookError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    result["command"] = "hook build"
    result["db"] = str(target_db)
    result["func_addr"] = func_addr
    result["func_addr_hex"] = f"0x{func_addr:x}"
    _emit_json(result)


# -- P3: MSVC RTTI scan (UNIT-P3-RTTI-PLAN.md section 2) --


@cli.group("rtti")
def rtti_group() -> None:
    """Recover MSVC RTTI class names and hierarchies from supplied image bytes."""
    pass


@rtti_group.command("scan")
@click.argument("image", metavar="FILE", type=click.Path(dir_okay=False, path_type=Path))
@click.option(
    "--base",
    "base_text",
    type=str,
    required=True,
    help="Image base / load address of the supplied bytes (hex, e.g. 0x140000000)",
)
@click.option(
    "--vtables",
    "vtables_text",
    type=str,
    required=True,
    help="Comma-separated vtable addresses to scan (hex or decimal, e.g. 0x14001000,0x14002000)",
)
@click.option(
    "--arch",
    type=click.Choice(["x86", "x64"], case_sensitive=False),
    default="x64",
    show_default=True,
    help="MSVC ABI variant: x64 resolves COL/CHD/BCD links as image-base RVAs, x86 as absolute VAs",
)
@click.option(
    "--db",
    "db_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("toothpick.db"),
    show_default=True,
    help="SQLite graph database (toothpick.db) to upsert the recovered classes into",
)
@click.option(
    "--sha",
    "binary_sha256",
    type=str,
    default=None,
    help="binary_sha256 the bytes belong to (inferred when the db holds exactly one binary)",
)
@click.option(
    "--max-vtables",
    "max_vtables",
    type=int,
    default=None,
    help="Risk R1 scan bound: at most N vtables scanned (default 50, clamped to the 1000 ceiling)",
)
def rtti_scan(
    image: Path,
    base_text: str,
    vtables_text: str,
    arch: str,
    db_path: Optional[Path],
    binary_sha256: Optional[str],
    max_vtables: Optional[int],
) -> None:
    """Scan VTABLE addresses of FILE for MSVC RTTI classes and store them.

    FILE is the raw image/memory-dump bytes and ``--base`` is the address those
    bytes are loaded at, so every recovered address belongs to *this* image: the
    store keeps no raw section bytes and the parser never invents an address
    (plan section 1).  Each ``--vtables`` candidate is a vftable whose
    ``vftable[-1]`` slot points at its CompleteObjectLocator; a candidate whose
    structures are truncated, misaligned or outside the supplied bytes is
    *rejected* with its reason instead of being emitted as a partial class row.

    Recovery is bounded (Risk R1): ``--max-vtables`` caps the scan (module
    default 50, ceiling 1000) and every candidate past the bound is reported in
    ``rejected``, so a bounded answer can never be mistaken for an exhaustive
    one.

    Classes land in the ``types`` table as ``type_id = 'rtti:<mangled>'``,
    ``kind = 'class'``, ``provenance: 'msvc-rtti'`` -- replace per sha, so
    re-scanning the same image converges instead of duplicating rows.
    """
    # Lazy import, the same discipline as `unpack` above: the CLI keeps no
    # import-time dependency on the RTTI subsystem, and the parser is only
    # reachable on a path that has actually been handed image bytes.
    from toothpick.rtti_parse import (
        RttiArch,
        RttiError,
        class_type_id,
        resolve_max_vtables,
        scan_vtables,
    )
    from toothpick.rtti_store import store
    from toothpick.storage import Database, StorageError

    # Addresses first: a malformed --base/--vtables is a caller error (never an
    # out-of-image rejection), so it fails before a byte is read.
    try:
        base_addr = parse_address(base_text)
        candidates = []
        for chunk in str(vtables_text).split(","):
            text = chunk.strip()
            if text:
                candidates.append(parse_address(text))
        if not candidates:
            raise QueryError("--vtables needs at least one vtable address")
        rtti_arch = RttiArch.coerce(arch)
        # R1 passthrough: `resolve_max_vtables` owns the default/ceiling rule,
        # so the bound reported back is the bound actually applied.
        budget = resolve_max_vtables(max_vtables)
    except QueryError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    except RttiError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    if not image.is_file():
        click.echo(f"Error: image not found: {image}", err=True)
        sys.exit(1)
    try:
        data = image.read_bytes()
    except OSError as exc:
        click.echo(f"Error: cannot read image bytes from {image}: {exc}", err=True)
        sys.exit(1)
    if not data:
        click.echo(
            f"Error: image is empty: {image} (no bytes to scan; nothing is invented)",
            err=True,
        )
        sys.exit(1)

    target_db = db_path or Path("toothpick.db")
    try:
        with Database(target_db, create=True) as db:
            sha = binary_sha256 or ""
            if not sha:
                # Rebase/profile-import inference pattern: only a store holding
                # exactly one binary may be assumed -- guessing among several
                # would attribute this image's classes to another binary.
                rows = db._conn.execute(
                    "SELECT DISTINCT binary_sha256 FROM funcs WHERE binary_sha256 != '' "
                    "UNION SELECT DISTINCT binary_sha256 FROM types WHERE binary_sha256 != ''"
                ).fetchall()
                shas = sorted({str(row[0]) for row in rows if row[0]})
                if len(shas) == 1:
                    sha = shas[0]
                elif len(shas) > 1:
                    click.echo(
                        "Error: Cannot infer binary SHA-256: database contains "
                        f"{len(shas)} distinct binaries ({', '.join(shas)}); "
                        "pass an explicit --sha",
                        err=True,
                    )
                    sys.exit(1)
                else:
                    click.echo(
                        "Error: Cannot infer binary SHA-256: database holds no "
                        "functions or types; pass an explicit --sha",
                        err=True,
                    )
                    sys.exit(1)

            result = scan_vtables(
                data, base_addr, rtti_arch, candidates, max_vtables=budget
            )
            stored = store(db, sha, result.classes, arch=rtti_arch.value)
    except RttiError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    except StorageError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    def _class_json(record: Any) -> Dict[str, Any]:
        """One recovered class as JSON: identity, evidence addresses, bases."""
        return {
            "type_id": class_type_id(record.mangled),
            "mangled": record.mangled,
            "name": record.name,
            "demangled": bool(record.demangled),
            "arch": record.arch,
            "vtable_addr": int(record.vtable_addr),
            "vtable_addr_hex": f"0x{int(record.vtable_addr):x}",
            "col_addr": int(record.col_addr),
            "col_addr_hex": f"0x{int(record.col_addr):x}",
            "td_addr": int(record.td_addr),
            "td_addr_hex": f"0x{int(record.td_addr):x}",
            "chd_addr": int(record.chd_addr),
            "chd_addr_hex": f"0x{int(record.chd_addr):x}",
            "bases": [
                {"mangled": base.mangled, "name": base.name, "mdisp": int(base.mdisp)}
                for base in record.bases
            ],
        }

    _emit_json(
        {
            "command": "rtti scan",
            "status": "ok",
            "db": str(target_db),
            "binary_sha256": sha,
            "image": str(image),
            "image_size": len(data),
            "base": base_addr,
            "base_hex": f"0x{base_addr:x}",
            "arch": rtti_arch.value,
            "vtables": [f"0x{addr:x}" for addr in candidates],
            "max_vtables": budget,
            "scanned": len(result.classes),
            "rejected_count": len(result.rejected),
            "classes": [_class_json(record) for record in result.classes],
            "rejected": [
                {"addr": int(addr), "addr_hex": f"0x{int(addr):x}", "reason": reason}
                for addr, reason in result.rejected
            ],
            "stored": stored,
        }
    )


# -- J1: JVM structural ingest (UNIT-J1-JVM-PLAN.md section 2) --


@cli.group("jvm")
def jvm_group() -> None:
    """JVM structural ingest: class-file / JAR structure into funcs/edges/types."""
    pass


@jvm_group.command("ingest")
@click.argument("target", metavar="PATH", type=click.Path(path_type=Path))
@click.option(
    "--db",
    "db_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("toothpick.db"),
    show_default=True,
    help="SQLite graph database (toothpick.db) to ingest the classes into",
)
@click.option(
    "--sha",
    "binary_sha256",
    type=str,
    default=None,
    help="binary_sha256 the classes belong to (inferred when the db holds one binary)",
)
def jvm_ingest(
    target: Path,
    db_path: Optional[Path],
    binary_sha256: Optional[str],
) -> None:
    """Ingest PATH (a JAR/ZIP or a single .class file) into toothpick.db.

    PATH is either an archive -- every ``*.class`` member, with ``META-INF/``,
    directory and non-``.class`` members skipped and counted -- or one class
    file.  The kind is decided by the bytes (``CAFEBABE`` vs ``PK``), with the
    suffix only breaking a tie, so neither branch trusts the name; a target that
    is neither is refused.  The target is parsed *before* the store is touched,
    so a missing, empty, truncated or otherwise unparseable input exits 1 with
    ``Error: <msg>`` on stderr and writes nothing.

    Methods land in ``funcs`` (``calling_convention='jvm'``, synthetic 1-based
    ordinals -- never addresses), resolved ``invoke*`` sites in ``edges``
    (kind ``call``), and classes/fields in ``types``, replace-per-sha and
    per-``jvm:`` namespace, exactly as :func:`toothpick.jvm.ingest` documents.
    A store holding one binary may have its ``--sha`` inferred; anything else
    must be named explicitly.
    """
    # Lazy import, the same discipline as `unpack` and `rtti scan` above: the
    # CLI keeps no import-time dependency on the JVM subsystem, so every other
    # command (and `toothpick --help`) stays cheap and import-clean.
    from toothpick.jvm import MAGIC, JvmError, ingest, parse_class_file, parse_jar
    from toothpick.storage import Database, StorageError

    path = Path(target)
    if not path.exists():
        click.echo(f"Error: target not found: {path}", err=True)
        sys.exit(1)
    if not path.is_file():
        click.echo(f"Error: target is not a file: {path}", err=True)
        sys.exit(1)
    try:
        data = path.read_bytes()
    except OSError as exc:
        click.echo(f"Error: cannot read {path}: {exc}", err=True)
        sys.exit(1)
    if not data:
        click.echo(
            f"Error: target is empty: {path} (no class file or JAR bytes to parse)",
            err=True,
        )
        sys.exit(1)

    # Magic first, suffix as the tiebreak only: a JAR named `.class`, or a class
    # file named `.jar`, is still read as what its bytes say.
    suffix = path.suffix.lower()
    if data[:4] == MAGIC:
        kind = "class"
    elif data[:2] == b"PK":
        kind = "jar"
    elif suffix in (".jar", ".zip"):
        kind = "jar"
    elif suffix == ".class":
        kind = "class"
    else:
        click.echo(
            f"Error: {path} is neither a class file (CAFEBABE) nor a ZIP/JAR "
            f"(PK..): first bytes {data[:4].hex()}",
            err=True,
        )
        sys.exit(1)

    # Parse before opening the store: a refused input fails without creating or
    # modifying a database, so a partial ingest is unreachable.
    try:
        if kind == "jar":
            classes, diagnostics = parse_jar(data)
        else:
            classes = [parse_class_file(data)]
            diagnostics = {"format": "class", "entries": 1, "skipped": 0, "classes": 1}
    except JvmError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    if not classes:
        click.echo(
            f"Error: {path} holds no class files (nothing to ingest)", err=True
        )
        sys.exit(1)

    target_db = db_path or Path("toothpick.db")
    try:
        with Database(target_db, create=True) as db:
            sha = binary_sha256 or ""
            if not sha:
                # Rebase/`profile import` inference pattern (see `rebase`
                # above): only a store holding exactly one binary may be
                # assumed -- guessing among several would attribute these
                # classes to another binary's sha.
                rows = db._conn.execute(
                    "SELECT DISTINCT binary_sha256 FROM funcs WHERE binary_sha256 != '' "
                    "UNION SELECT DISTINCT binary_sha256 FROM types WHERE binary_sha256 != ''"
                ).fetchall()
                shas = sorted({str(row[0]) for row in rows if row[0]})
                if len(shas) == 1:
                    sha = shas[0]
                elif len(shas) > 1:
                    click.echo(
                        "Error: Cannot infer binary SHA-256: database contains "
                        f"{len(shas)} distinct binaries ({', '.join(shas)}); "
                        "pass an explicit --sha",
                        err=True,
                    )
                    sys.exit(1)
                else:
                    click.echo(
                        "Error: Cannot infer binary SHA-256: database holds no "
                        "functions or types; pass an explicit --sha",
                        err=True,
                    )
                    sys.exit(1)

            result = ingest(db, sha, classes)
    except JvmError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    except StorageError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    # The diagnostics of the parse that produced these rows win over ingest's
    # default labels: entries/skipped describe the archive that was read.
    result["diagnostics"] = {**result.get("diagnostics", {}), **diagnostics}
    result["command"] = "jvm ingest"
    result["status"] = "ok"
    result["db"] = str(target_db)
    result["path"] = str(path)
    result["format"] = diagnostics.get("format", kind)
    _emit_json(result)


# -- J3: Lua mod scaffolding (UNIT-J3-LUA-PLAN.md section 2) ------------------


def _lua_slug(*parts: str) -> str:
    """Deterministic path-safe identifier (``[A-Za-z0-9_.-]+``) from name parts.

    Only ASCII alphanumerics and ``_``/``.``/``-`` survive, so a derived mod id
    can never escape its output directory; an empty or dotted-path-shaped result
    falls back to a fixed id rather than being rejected at a distance.
    """
    text = "_".join(part for part in parts if part).lower()
    cleaned = "".join(
        char if char.isascii() and (char.isalnum() or char in "_.-") else "_"
        for char in text
    )
    cleaned = cleaned.strip("_.-")
    if not cleaned or cleaned in (".", "..") or ".." in cleaned:
        cleaned = "toothpick_lua_mod"
    return cleaned


def _lua_method_ordinal(db: Any, binary_sha256: str, config: Any) -> Optional[int]:
    """J1 synthetic ordinal of ``config``'s ``Class.method:desc`` in ``db``.

    ``toothpick jvm ingest`` (J1) writes every JVM method as a ``funcs`` row whose
    ``func_addr`` is the deterministic synthetic ordinal of
    :func:`toothpick.jvm_parse.method_addr_index`, ``name`` the dotted
    ``a.b.C.m`` display name and ``prototype`` the descriptor.  The lookup
    mirrors those three columns exactly, so the ordinal handed to
    :class:`toothpick.hook.HookGate` is the one J1 wrote -- never a re-derived or
    guessed address.  A name/descriptor pair with no row is *unobserved*
    (``None``), which the caller refuses unless the gate was waived.
    """
    dotted = config.target_class.replace("/", ".")
    rows = db._conn.execute(
        "SELECT func_addr, calling_convention FROM funcs "
        "WHERE binary_sha256 = ? AND name = ? AND prototype = ? "
        "ORDER BY func_addr",
        (binary_sha256, f"{dotted}.{config.target_method}", config.target_desc),
    ).fetchall()
    if not rows:
        return None
    # The J1 (calling_convention='jvm') row is the ordinal the synthetic address
    # space defines; any other producer's same-named row is only a fallback, so a
    # coincidental native row can never be preferred over the real one.
    for func_addr, convention in rows:
        if str(convention or "") == "jvm":
            return int(func_addr)
    return int(rows[0][0])


@cli.group("lua")
def lua_group() -> None:
    """Lua mod scaffolding for JVM/descriptor-editable games (Project Zomboid class)."""
    pass


@lua_group.command("build")
@click.option(
    "--class",
    "class_name",
    type=str,
    required=True,
    help="JVM target class (internal 'com/foo/Bar' or dotted 'com.foo.Bar')",
)
@click.option(
    "--method",
    "method_name",
    type=str,
    required=True,
    help="JVM target method name (e.g. 'tick')",
)
@click.option(
    "--desc",
    "method_desc",
    type=str,
    required=True,
    help="JVM method descriptor, '()V'-shaped (e.g. '()V')",
)
@click.option(
    "--out",
    "out_dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Output project directory for the generated Lua mod project",
)
@click.option(
    "--name",
    "mod_name",
    type=str,
    default=None,
    help="Human-readable mod name (default: derived from --class/--method)",
)
@click.option(
    "--mod-id",
    "mod_id",
    type=str,
    default=None,
    help="Path-safe mod id (default: derived from --class/--method)",
)
@click.option(
    "--version",
    "version",
    type=str,
    default="1.0.0",
    show_default=True,
    help="Semver-ish mod version",
)
@click.option(
    "--author",
    "author",
    type=str,
    default="",
    help="Mod author recorded in README.md (optional)",
)
@click.option(
    "--db",
    "db_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="SQLite graph database (toothpick.db) holding the J1 method rows/receipts",
)
@click.option(
    "--sha",
    "binary_sha256",
    type=str,
    default=None,
    help=(
        "binary_sha256 the JVM method ordinal belongs to "
        "(inferred when the db holds exactly one binary)"
    ),
)
@click.option(
    "--allow-unverified",
    is_flag=True,
    help=(
        "Override the Epistemic Grounding Gate and emit an UNVERIFIED artifact "
        "(Design §16 #1)"
    ),
)
def lua_build(
    class_name: str,
    method_name: str,
    method_desc: str,
    out_dir: Path,
    mod_name: Optional[str],
    mod_id: Optional[str],
    version: str,
    author: str,
    db_path: Optional[Path],
    binary_sha256: Optional[str],
    allow_unverified: bool,
) -> None:
    """Scaffold a Project-Zomboid-class Lua mod project for a JVM method target.

    The target is named as ``Class.method:desc`` and lands in
    ``media/lua/client/<mod_id>/main.lua`` (boot-token print + hook stub), plus
    ``mod.info`` and ``README.md``.  Nothing is applied to a class file: the
    scaffold describes the intended hook point only.

    Receipts-or-refusal (Design §16 #1): when a store is available (``--db``, or
    ``./toothpick.db``) the target's J1 synthetic ordinal is looked up and
    :class:`toothpick.hook.HookGate` enforces a confirmed ``GROUNDED`` receipt on
    it.  An unobserved method, a missing store, or an ungrounded ordinal raises
    ``Error: ...`` and exits 1 unless ``--allow-unverified`` is passed, in which
    case the project is stamped ``UNVERIFIED`` and never claims grounding.
    ``toothpick.lua_mod`` and ``toothpick.hook`` are imported lazily so the CLI
    keeps no import-time dependency on either subsystem.
    """
    # Lazy imports, the same discipline as `hook build`/`jvm ingest` above: no
    # import-time dependency on the J3 module or Capstone for any other command.
    from toothpick.hook import HookGate
    from toothpick.lua_mod import LuaModConfig, LuaModError, generate_lua_mod
    from toothpick.mod_builder import UngroundedTargetError
    from toothpick.storage import Database, StorageError

    simple_class = class_name.replace("/", ".").rsplit(".", 1)[-1]
    try:
        config = LuaModConfig(
            name=mod_name or f"Toothpick {simple_class}.{method_name} mod",
            mod_id=mod_id or _lua_slug("toothpick", simple_class, method_name),
            version=version,
            target_class=class_name,
            target_method=method_name,
            target_desc=method_desc,
            author=author,
        )
    except LuaModError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    target_db = db_path
    if target_db is None:
        default_db = Path("toothpick.db")
        if default_db.is_file():
            target_db = default_db

    grounded = False
    sha = binary_sha256 or ""
    ordinal: Optional[int] = None
    if target_db is not None:
        if not target_db.exists():
            # A typo'd --db is an error, not an empty store (see `hook build`
            # above): without the store there is no ordinal to gate against.
            click.echo(f"Error: database not found: {target_db}", err=True)
            sys.exit(1)
        try:
            with Database(target_db) as db:
                if not sha:
                    # Same inference pattern as `hook build`/`jvm ingest`: only a
                    # store holding exactly one binary may be assumed -- guessing
                    # among several would pin another binary's receipt here.
                    rows = db._conn.execute(
                        "SELECT DISTINCT binary_sha256 FROM funcs WHERE binary_sha256 != '' "
                        "UNION SELECT DISTINCT binary_sha256 FROM receipts "
                        "WHERE binary_sha256 != ''"
                    ).fetchall()
                    shas = sorted({str(row[0]) for row in rows if row[0]})
                    if len(shas) == 1:
                        sha = shas[0]
                    elif len(shas) > 1:
                        click.echo(
                            "Error: Cannot infer binary SHA-256: database contains "
                            f"{len(shas)} distinct binaries ({', '.join(shas)}); "
                            "pass an explicit --sha",
                            err=True,
                        )
                        sys.exit(1)
                    else:
                        click.echo(
                            "Error: Cannot infer binary SHA-256: database holds no "
                            "functions or receipts; pass an explicit --sha",
                            err=True,
                        )
                        sys.exit(1)

                ordinal = _lua_method_ordinal(db, sha, config)
                if ordinal is None:
                    if not allow_unverified:
                        click.echo(
                            f"Error: JVM method {config.target_symbol} has no ingested "
                            f"ordinal in {target_db} (J1 'jvm ingest' writes it as a "
                            "funcs row); nothing was observed to ground against -- "
                            "pass --allow-unverified to emit an UNVERIFIED scaffold "
                            "(Design §16 #1)",
                            err=True,
                        )
                        sys.exit(1)
                else:
                    # Raises UngroundedTargetError unless a GROUNDED receipt backs
                    # this ordinal or the operator explicitly waived the gate.
                    grounded = HookGate.enforce(db, sha, ordinal, allow_unverified)
        except UngroundedTargetError as exc:
            click.echo(f"Error: {exc}", err=True)
            sys.exit(1)
        except StorageError as exc:
            click.echo(f"Error: {exc}", err=True)
            sys.exit(1)
    elif not allow_unverified:
        click.echo(
            f"Error: no toothpick.db available to ground {config.target_symbol} "
            "against (pass --db/--sha); nothing was observed, so pass "
            "--allow-unverified to emit an UNVERIFIED scaffold (Design §16 #1)",
            err=True,
        )
        sys.exit(1)

    try:
        result = generate_lua_mod(config, out_dir, grounded=grounded)
    except LuaModError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    result["command"] = "lua build"
    result["db"] = str(target_db) if target_db is not None else None
    result["binary_sha256"] = sha or None
    result["method_ordinal"] = ordinal
    result["allow_unverified"] = bool(allow_unverified)


@lua_group.command("pack")
@click.argument(
    "project_dir",
    type=click.Path(file_okay=False, path_type=Path),
)
@click.option(
    "--out",
    "out_dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("dist"),
    show_default=True,
    help="Output directory for <mod_id>.zip and <mod_id>.workshop.json",
)
def lua_pack(project_dir: Path, out_dir: Path) -> None:
    """Package a generated Lua mod project for Workshop distribution.

    ``PROJECT_DIR`` is a tree written by ``toothpick lua build``; this writes
    ``<mod_id>.zip`` (deterministic entries: fixed order, fixed metadata, no
    timestamps) plus the ``<mod_id>.workshop.json`` manifest carrying
    per-file ``path``/``size``/``sha256`` into ``--out`` (default ``dist``).
    Steam API uploads are out of scope: the manifest is the hand-off.

    A missing/unreadable project, an unreadable ``mod.info`` or an occupied
    output path is an ``Error: ...`` on stderr with exit code 1.  The
    ``toothpick.lua_mod`` module is imported lazily so no other command pays
    for it at import time (the same discipline as ``lua build`` above).
    """
    from toothpick.lua_mod import LuaModError, package_workshop

    try:
        result = package_workshop(project_dir, out_dir)
    except LuaModError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    result["command"] = "lua pack"
    result["project_dir"] = str(project_dir)
    _emit_json(result)


@lua_group.command("patch-emit")
@click.argument(
    "patches_json",
    type=click.Path(dir_okay=False, path_type=Path),
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("patch.json"),
    show_default=True,
    help="Output path for the emitted patch descriptor JSON",
)
def lua_patch_emit(patches_json: Path, out_path: Path) -> None:
    """Emit a class-file patch descriptor from the ``PATCHES_JSON`` list.

    ``PATCHES_JSON`` is a JSON file holding a list of
    ``{class, method, desc, op, lua_hook}`` objects (``op`` in
    ``prepend``/``append``/``replace``, ``desc`` ``()V``-shaped); it is validated
    before anything is written and the descriptor is emitted to ``--out``
    (default ``patch.json``) with a deterministic byte layout.

    The descriptor *describes* edits for external bytecode tools -- nothing here
    reads or rewrites a class file.  A missing/unreadable ``PATCHES_JSON``,
    malformed JSON, or any schema violation is an ``Error: ...`` on stderr with
    exit code 1.  ``toothpick.lua_mod`` is imported lazily.
    """
    from toothpick.lua_mod import LuaModError, emit_patch_descriptor

    try:
        document = json.loads(patches_json.read_text(encoding="utf-8"))
    except OSError as exc:
        click.echo(f"Error: cannot read {patches_json}: {exc}", err=True)
        sys.exit(1)
    except ValueError as exc:
        click.echo(f"Error: {patches_json} is not valid JSON: {exc}", err=True)
        sys.exit(1)

    try:
        result = emit_patch_descriptor(document, out_path)
    except LuaModError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    result["command"] = "lua patch-emit"
    result["patches_json"] = str(patches_json)
    _emit_json(result)


# -- P4: A/B bench harness (UNIT-P4-BENCH-PLAN.md section 2) --


@cli.group("bench")
def bench_group() -> None:
    """A/B wall-clock benches: sample a command, gate the delta, receipt the claim."""
    pass


def _load_run_result(path: Path) -> Any:
    """Load a ``bench run --out`` RunResult JSON file (errors are ``Error:`` + exit 1).

    ``toothpick.bench`` is imported lazily: the module is only paid for when a
    ``bench`` subcommand actually runs (the same discipline as ``lua``/``unpack``).
    """
    from toothpick.bench import BenchError, RunResult

    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        click.echo(f"Error: cannot read {path}: {exc}", err=True)
        sys.exit(1)
    except ValueError as exc:
        click.echo(f"Error: {path} is not valid JSON: {exc}", err=True)
        sys.exit(1)

    try:
        return RunResult.from_dict(document)
    except BenchError as exc:
        click.echo(f"Error: {path} is not a bench RunResult JSON: {exc}", err=True)
        sys.exit(1)


@bench_group.command("run")
@click.option(
    "--reps",
    type=int,
    default=None,
    help="Measured repetitions (default: the bench module's 11)",
)
@click.option(
    "--warmup",
    type=int,
    default=None,
    help="Discarded warmup runs, executed but never timed (default 2)",
)
@click.option(
    "--timeout",
    type=float,
    default=None,
    help="Per-run wall-clock timeout in seconds (default 300)",
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write the RunResult JSON that 'bench compare' consumes to this path",
)
@click.argument("cmd", nargs=-1, required=False, type=click.UNPROCESSED)
def bench_run(
    reps: Optional[int],
    warmup: Optional[int],
    timeout: Optional[float],
    out_path: Optional[Path],
    cmd: Tuple[str, ...],
) -> None:
    """Time CMD... and emit the RunResult JSON.

    Usage: ``toothpick bench run --reps 11 --warmup 2 --timeout 300 --out
    before.json -- python target.py``.  The ``--`` separator is what keeps CMD's
    own flags out of this command's parser: the argv is handed to the bench
    module as a list and run with ``shell=False``, never through a shell.

    An empty command, a non-positive ``--reps``/``--timeout``, a negative
    ``--warmup`` or a target that exits non-zero is an ``Error: ...`` on stderr
    with exit code 1 -- a crashing target is never folded into the timings.
    ``--out`` receives the plain RunResult JSON (the file ``bench compare``
    reads); the stdout payload adds ``command``/``out_path`` on top.
    """
    from toothpick.bench import (
        DEFAULT_REPS,
        DEFAULT_TIMEOUT,
        DEFAULT_WARMUP,
        BenchError,
        measure,
    )

    argv = list(cmd)
    if not argv:
        click.echo(
            "Error: bench run requires a command, e.g. "
            "toothpick bench run -- python target.py",
            err=True,
        )
        sys.exit(1)

    # Unset flags fall back to the module defaults (never to a CLI-local copy),
    # and a supplied 0/-1 still reaches measure() so its validation is the gate.
    try:
        result = measure(
            argv,
            reps=DEFAULT_REPS if reps is None else reps,
            warmup=DEFAULT_WARMUP if warmup is None else warmup,
            timeout=DEFAULT_TIMEOUT if timeout is None else timeout,
        )
    except BenchError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    payload = result.as_dict()
    payload["command"] = "bench run"
    if out_path is not None:
        try:
            out_path.write_text(
                json.dumps(result.as_dict(), indent=2, sort_keys=True, default=str),
                encoding="utf-8",
            )
        except OSError as exc:
            click.echo(f"Error: cannot write {out_path}: {exc}", err=True)
            sys.exit(1)
        payload["out_path"] = str(out_path)
    _emit_json(payload)


@bench_group.command("compare")
@click.argument(
    "before_json",
    type=click.Path(dir_okay=False, path_type=Path),
)
@click.argument(
    "after_json",
    type=click.Path(dir_okay=False, path_type=Path),
)
@click.option(
    "--db",
    "db_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="SQLite graph database (toothpick.db) to receipt the claim into",
)
@click.option(
    "--sha",
    "binary_sha256",
    type=str,
    default=None,
    help="binary_sha256 the claim is attributed to (required with --db)",
)
@click.option(
    "--label",
    type=str,
    default=None,
    help="Human-readable claim label, e.g. 'hook-tick' (required with --db)",
)
@click.option(
    "--min-effect",
    type=float,
    default=None,
    help="Minimum relative median improvement that counts as a win (default 0.05)",
)
def bench_compare(
    before_json: Path,
    after_json: Path,
    db_path: Optional[Path],
    binary_sha256: Optional[str],
    label: Optional[str],
    min_effect: Optional[float],
) -> None:
    """Gate AFTER_JSON against BEFORE_JSON, optionally receipting the claim.

    Both arguments are RunResult JSON files as ``bench run --out`` writes them
    (BEFORE_JSON is the pre-change run).  The verdict is printed as JSON:
    ``faster`` is ``true`` only when the median ratio cleared ``--min-effect``
    *and* the two IQR bands are disjoint -- overlapping spreads stay
    ``inconclusive``, so noise is never printed as a win.

    With ``--db`` the same verdict is recorded as a ``phase="bench"`` receipt
    (``func_addr`` sentinel ``-1``, trap 23) that always carries the binding
    ``verifier_status="UNVERIFIED (stability)"`` label: a wall-clock delta is
    not grounding evidence.  ``--sha`` and ``--label`` are required with
    ``--db`` so the row is attributed rather than guessed; the id is reported as
    ``receipt_id`` and replaying the identical claim updates that row instead of
    appending a duplicate.

    A missing/unreadable/malformed JSON argument, or a verdict this harness
    refuses (a bad ``--min-effect``, a run with no timing samples), is an
    ``Error: ...`` on stderr with exit code 1.
    """
    from toothpick.bench import (
        DEFAULT_MIN_EFFECT,
        VERIFIER_STATUS_STABILITY,
        BenchError,
        compare,
        record_claim,
    )

    before = _load_run_result(before_json)
    after = _load_run_result(after_json)
    effect = DEFAULT_MIN_EFFECT if min_effect is None else min_effect

    try:
        verdict = compare(before, after, min_effect=effect)
    except BenchError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    payload = verdict.as_dict()
    payload["command"] = "bench compare"
    payload["before_json"] = str(before_json)
    payload["after_json"] = str(after_json)
    payload["before_median_ms"] = before.stat("median_ms")
    payload["after_median_ms"] = after.stat("median_ms")

    if db_path is not None:
        if not binary_sha256:
            click.echo(
                "Error: --sha is required with --db (a bench receipt is "
                "attributed, never guessed)",
                err=True,
            )
            sys.exit(1)
        if not label:
            click.echo(
                "Error: --label is required with --db (the claim needs a "
                "human-readable name)",
                err=True,
            )
            sys.exit(1)
        try:
            receipt_id = record_claim(db_path, binary_sha256, label, verdict)
        except BenchError as exc:
            click.echo(f"Error: {exc}", err=True)
            sys.exit(1)
        payload["binary_sha256"] = str(binary_sha256)
        payload["label"] = str(label)
        payload["receipt_id"] = receipt_id
        payload["verifier_status"] = VERIFIER_STATUS_STABILITY

    _emit_json(payload)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
