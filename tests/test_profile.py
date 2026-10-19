"""P1 profile-ingestion tests: parsers, ``func_profile`` table, attribution, R1 caps.

Binding contract: ``docs/units/UNIT-P1-PROFILE-PLAN.md`` section 1 (unified
importer, attribution by address then name, hotness normalization) and section 2
(interfaces), with Risk R1 (bounded top-K, deterministic ordering) and the
Windows SQLite traps (close every handle -- a leaked WAL handle surfaces as
``PermissionError: [WinError 32]`` on teardown -- and no ``faulthandler``;
``pyproject.toml`` already pins ``-p no:faulthandler``).

Every synthetic profile here is header-only evidence: nothing in this file
invents a ``funcs`` row, which is the epistemic rule the module is built on.

The final sections drive the certified P1 slices end to end over a tmp db seeded
by :func:`_seed`: CLI (``profile import`` -> ``profile hottest``), ``QueryEngine``
(``hottest`` ordering and ``get_function.profile``) and MCP ``toothpick_hottest``.
Every store, engine and CLI invocation there is context-managed or explicitly
closed before teardown (trap 7).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from click.testing import CliRunner

from toothpick.cli import cli
from toothpick.mcp_server import TOOL_NAMES, McpToolError, create_mcp_server
from toothpick.profile import (
    DEFAULT_TOP_N,
    FUNC_PROFILE_DDL,
    FUNC_PROFILE_INDEX_DDL,
    MAX_TOP_N,
    ProfileError,
    ProfileFormat,
    ProfileSample,
    detect_format,
    ensure_profile_table,
    get_profile,
    hottest,
    import_samples,
    parse_etw_csv,
    parse_perf_csv,
    parse_profile_file,
    parse_profile_text,
    parse_vtune_csv,
    resolve_top_n,
)
from toothpick.storage import Database, FunctionRecord
from toothpick.query import DEFAULT_LIMIT, MAX_LIMIT, QueryEngine, QueryError

SHA = "sha_profile_test"

VTUNE_CSV = """Function,Function Address,CPU Time,Samples,Alloc Bytes
sub_140001000,0x140001000,1.5s,300,0
sub_140002000,0x140002000,0.5s,100,4096
ghost_symbol,0x140003000,0.2s,60,0
sub_140001000,0x140001000,0.1s,40,0
"""
VTUNE_TIME_CSV = """Function,Function Address,CPU Time
sub_140001000,0x140001000,1.5s
sub_140002000,0x140002000,250ms
"""

ETW_CSV = """Stack,Symbol,Address,Count,Alloc
k32.dll!sub_140001000,sub_140001000,0x140001000,10,1024
by_sym,by_sym,,5,
"""

PERF_CSV = """symbol,address,samples,alloc
sub_140001000,0x140001000,80,0
sub_140002000,0x140002000,20,2048
"""

MALFORMED_CSV = """symbol,address,samples
ok,0x1000,5
bad,0x1,notanumber
short
"""

#: Three-row perf-CSV covering every ``_seed`` function with distinct weights
#: (50/30/20 -> hotness 0.5/0.3/0.2): the ordering fixture for the CLI, engine
#: and MCP slices below.
TRIO_CSV = """symbol,address,samples,alloc
sub_140001000,0x140001000,50,0
sub_140002000,0x140002000,30,1024
by_sym,,20,0
"""

#: Equal weights with the addresses listed descending on purpose: the R1
#: tie-break must reorder to ``func_addr ASC``, not echo the file order.
TIE_CSV = """symbol,address,samples
sub_140002000,0x140002000,25
sub_140001000,0x140001000,25
"""


def _seed(db: Database, sha: str = SHA, *, write_receipt: bool = True) -> None:
    """Three known functions -- the only rows attribution may ever touch."""
    for addr, name in (
        (0x140001000, "sub_140001000"),
        (0x140002000, "sub_140002000"),
        (0x140005000, "by_sym"),
    ):
        db.upsert_function(
            FunctionRecord(binary_sha256=sha, func_addr=addr, name=name),
            write_receipt=write_receipt,
        )


# --------------------------------------------------------------------------- #
# parsers
# --------------------------------------------------------------------------- #


def test_parse_vtune_csv_addresses_symbols_samples_and_alloc() -> None:
    rows = parse_vtune_csv(VTUNE_CSV)
    # Deterministic address sort; the duplicate address keeps both observations
    # (aggregation is an import concern, not a parse concern).
    assert [r.func_addr for r in rows] == [
        0x140001000,
        0x140001000,
        0x140002000,
        0x140003000,
    ]
    assert sorted(r.samples for r in rows) == [40, 60, 100, 300]
    assert {r.symbol for r in rows} == {
        "sub_140001000",
        "sub_140002000",
        "ghost_symbol",
    }
    assert [r.alloc_bytes for r in rows if r.func_addr == 0x140002000] == [4096]


def test_parse_vtune_csv_without_sample_column_uses_cpu_time() -> None:
    rows = parse_vtune_csv(VTUNE_TIME_CSV)
    assert [r.func_addr for r in rows] == [0x140001000, 0x140002000]
    assert [r.samples for r in rows] == [1500, 250]  # seconds -> ms pseudo-count


def test_parse_etw_csv_symbol_address_and_optional_alloc() -> None:
    rows = parse_etw_csv(ETW_CSV)
    assert [(r.func_addr, r.symbol, r.samples, r.alloc_bytes) for r in rows] == [
        (0x140001000, "sub_140001000", 10, 1024),
        (None, "by_sym", 5, 0),
    ]


def test_parse_perf_csv_alloc_column_and_address_optional() -> None:
    rows = parse_perf_csv(PERF_CSV)
    assert [(r.func_addr, r.samples, r.alloc_bytes) for r in rows] == [
        (0x140001000, 80, 0),
        (0x140002000, 20, 2048),
    ]
    samples, diag = parse_profile_text("symbol,samples\nonly_name,7\n")
    assert [(r.func_addr, r.symbol, r.samples) for r in samples] == [
        (None, "only_name", 7)
    ]
    assert diag["format"] == "perf"


def test_detect_format_priority_and_refusal() -> None:
    assert detect_format(["Function", "CPU Time"]) is ProfileFormat.VTUNE
    assert detect_format(["Function", "Samples"]) is ProfileFormat.VTUNE
    assert detect_format(["Stack", "Count"]) is ProfileFormat.ETW
    assert detect_format(["symbol", "address", "samples"]) is ProfileFormat.PERF
    with pytest.raises(ProfileError):
        detect_format(["alpha", "beta"])


def test_parse_profile_text_counts_malformed_rows_instead_of_raising() -> None:
    samples, diag = parse_profile_text(MALFORMED_CSV)
    assert [s.symbol for s in samples] == ["ok"]
    assert diag["rejected"] == 2  # bad number + short row
    assert diag["rows"] == 1 and diag["format"] == "perf"


def test_parsers_refuse_unparsable_input() -> None:
    with pytest.raises(ProfileError):
        parse_profile_text("")
    with pytest.raises(ProfileError):
        parse_profile_text("alpha,beta\n1,2\n")  # no format matches the header
    with pytest.raises(ProfileError):
        parse_vtune_csv("Function,Notes\nsub_a,hello\n")  # no weight column


def test_parse_profile_file_auto_diagnostics_and_explicit_format(
    tmp_path: Path,
) -> None:
    path = tmp_path / "hotspots.csv"
    path.write_text(VTUNE_CSV, encoding="utf-8")
    samples, diag = parse_profile_file(path)
    assert len(samples) == 4 and diag["rejected"] == 0 and diag["rows"] == 4
    assert diag["format"] == "vtune" and diag["delimiter"] == ","
    assert diag["path"] == str(path) and diag["size_bytes"] == path.stat().st_size
    assert diag["header"] == [
        "Function",
        "Function Address",
        "CPU Time",
        "Samples",
        "Alloc Bytes",
    ]

    perf_path = tmp_path / "perf.csv"
    perf_path.write_text(PERF_CSV, encoding="utf-8")
    perf_samples, perf_diag = parse_profile_file(perf_path, format=ProfileFormat.PERF)
    assert perf_diag["format"] == "perf"
    assert [s.func_addr for s in perf_samples] == [0x140001000, 0x140002000]

    with pytest.raises(ProfileError):
        parse_profile_file(perf_path, format="bogus")
    with pytest.raises(ProfileError):
        parse_profile_file(tmp_path / "missing.csv")


def test_profile_format_coerce() -> None:
    assert ProfileFormat.coerce(None) is ProfileFormat.AUTO
    assert ProfileFormat.coerce(" VTUNE ") is ProfileFormat.VTUNE
    assert ProfileFormat.coerce(ProfileFormat.ETW) is ProfileFormat.ETW
    # J2 (UNIT-J2-JFR-PLAN.md chunk 2a): async-profiler collapsed and JFR JSON
    # are real members now, so they coerce instead of raising.
    assert ProfileFormat.coerce("jfr") is ProfileFormat.JFR
    assert ProfileFormat.coerce(" async ") is ProfileFormat.ASYNC
    with pytest.raises(ProfileError):
        ProfileFormat.coerce("jfr-binary")  # "jfr" exists; "jfr-binary" does not


def test_profile_sample_validates_and_serializes() -> None:
    sample = ProfileSample(symbol="  sub_a  ", func_addr="0x1000", samples=3)
    assert sample.symbol == "sub_a" and sample.func_addr == 0x1000
    assert sample.as_dict() == {
        "symbol": "sub_a",
        "func_addr": 0x1000,
        "addr_hex": "0x1000",
        "samples": 3,
        "alloc_bytes": 0,
    }
    with pytest.raises(ProfileError):
        ProfileSample(func_addr=-1)
    with pytest.raises(ProfileError):
        ProfileSample(func_addr=True)
    with pytest.raises(ProfileError):
        ProfileSample(samples=-5)
    with pytest.raises(ProfileError):
        ProfileSample(samples="not-a-number")


# --------------------------------------------------------------------------- #
# table DDL
# --------------------------------------------------------------------------- #


def test_ensure_profile_table_ddl_columns_index_and_idempotency(
    tmp_path: Path,
) -> None:
    with Database(tmp_path / "graph.db") as db:
        ensure_profile_table(db)
        ensure_profile_table(db)  # idempotent
        assert db.column_names("func_profile") == [
            "binary_sha256",
            "func_addr",
            "sample_count",
            "alloc_bytes",
            "hotness",
            "updated_at",
        ]
        with db.connection() as conn:
            indexes = [
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'index' "
                    "AND tbl_name = 'func_profile'"
                ).fetchall()
            ]
    assert "idx_func_profile_hot" in indexes
    assert "CREATE TABLE IF NOT EXISTS func_profile" in FUNC_PROFILE_DDL
    assert "PRIMARY KEY (binary_sha256, func_addr)" in FUNC_PROFILE_DDL
    assert "IF NOT EXISTS" in FUNC_PROFILE_INDEX_DDL


def test_func_profile_is_not_core_schema_until_ensure_is_called(tmp_path: Path) -> None:
    """Storage wiring is the Chunk 2 half of plan 1.2; the module adds the table."""
    with Database(tmp_path / "core.db") as db:
        assert "func_profile" not in db.table_names()
        ensure_profile_table(db)
        assert "func_profile" in db.table_names()


def test_ensure_profile_table_accepts_a_db_path_and_releases_it(tmp_path: Path) -> None:
    path = tmp_path / "ensure" / "profile.db"
    ensure_profile_table(path)
    conn = sqlite3.connect(str(path))
    try:
        names = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
    finally:
        conn.close()
    assert "func_profile" in names
    path.unlink()  # PermissionError [WinError 32] if the module leaked a handle


# --------------------------------------------------------------------------- #
# attribution + hotness
# --------------------------------------------------------------------------- #


def test_import_attributes_by_address_then_name_and_never_fabricates_funcs(
    tmp_path: Path,
) -> None:
    with Database(tmp_path / "graph.db") as db:
        _seed(db)
        before = db.count_functions(SHA)
        result = import_samples(db, SHA, VTUNE_CSV)
        assert result["binary_sha256"] == SHA
        assert result["imported"] == 3
        assert result["unattributed"] == 1
        assert result["unattributed_samples"] == 60  # ghost_symbol is not a funcs row
        assert result["attributed_samples"] == 440
        assert result["total_samples"] == 500
        assert result["functions"] == 2
        # Epistemic rule: no funcs row invented for the unmatched symbol.
        assert db.count_functions(SHA) == before == 3
        assert db.get_function(SHA, 0x140003000) is None
        assert {f.name for f in db.get_functions(SHA)} == {
            "sub_140001000",
            "sub_140002000",
            "by_sym",
        }


def test_hotness_is_attributed_share_and_ordering_is_hotness_desc_addr_asc(
    tmp_path: Path,
) -> None:
    with Database(tmp_path / "graph.db") as db:
        _seed(db)
        import_samples(db, SHA, VTUNE_CSV)
        listed = hottest(db, SHA)
        assert [(r["func_addr"], r["sample_count"]) for r in listed] == [
            (0x140001000, 340),  # 300 + 40 aggregated onto one (binary, addr) row
            (0x140002000, 100),
        ]
        assert round(listed[0]["hotness"], 12) == round(340 / 440, 12)
        assert round(sum(r["hotness"] for r in listed), 12) == 1.0
        assert listed[0]["alloc_bytes"] == 0 and listed[1]["alloc_bytes"] == 4096
        assert set(listed[0]) == {
            "func_addr",
            "addr_hex",
            "name",
            "sample_count",
            "alloc_bytes",
            "hotness",
        }
        assert listed[0]["addr_hex"] == "0x140001000"
        assert listed[0]["name"] == "sub_140001000"


def test_name_fallback_attribution_and_replace_semantics(tmp_path: Path) -> None:
    with Database(tmp_path / "graph.db") as db:
        _seed(db)
        assert import_samples(db, SHA, VTUNE_CSV)["functions"] == 2
        second = import_samples(db, SHA, ETW_CSV)
        assert second["imported"] == 2 and second["unattributed"] == 0
        rows = hottest(db, SHA, 100)
        # Address-less ETW row fell back to the exact name "by_sym"; the previous
        # import's rows are replaced, not accumulated.
        assert [(r["func_addr"], r["sample_count"]) for r in rows] == [
            (0x140001000, 10),
            (0x140005000, 5),
        ]
        assert round(sum(r["hotness"] for r in rows), 12) == 1.0


def test_get_profile_row_hex_lookup_and_miss(tmp_path: Path) -> None:
    with Database(tmp_path / "graph.db") as db:
        _seed(db)
        import_samples(db, SHA, ETW_CSV)
        prof = get_profile(db, SHA, 0x140001000)
        assert prof is not None
        assert set(prof) == {"sample_count", "alloc_bytes", "hotness"}
        assert prof["sample_count"] == 10 and prof["alloc_bytes"] == 1024
        assert round(prof["hotness"], 12) == round(10 / 15, 12)
        assert get_profile(db, SHA, "0x140001000") == prof  # hex string accepted
        assert get_profile(db, SHA, 0x140003000) is None
        assert get_profile(db, SHA, "zzz") is None
        with pytest.raises(ProfileError):
            get_profile(db, SHA, -1)


def test_import_samples_accepts_text_path_tuple_and_mapping(tmp_path: Path) -> None:
    with Database(tmp_path / "graph.db") as db:
        _seed(db)
        csv_path = tmp_path / "hotspots.csv"
        csv_path.write_text(VTUNE_CSV, encoding="utf-8")
        by_path = import_samples(db, SHA, csv_path)
        by_tuple = import_samples(db, SHA, parse_profile_file(csv_path))
        by_text = import_samples(db, SHA, VTUNE_CSV)
        for result in (by_path, by_tuple, by_text):
            assert result["total_samples"] == 500
            assert result["unattributed"] == 1 and result["rejected"] == 0
        by_mapping = import_samples(
            db,
            SHA,
            [
                {
                    "symbol": "sub_140001000",
                    "address": "0x140001000",
                    "samples": 5,
                    "alloc": 7,
                }
            ],
        )
        assert by_mapping["imported"] == 1 and by_mapping["total_samples"] == 5
        assert get_profile(db, SHA, 0x140001000)["alloc_bytes"] == 7
        with pytest.raises(ProfileError):
            import_samples(db, "", VTUNE_CSV)


def test_zero_weight_attribution_stores_hotness_zero(tmp_path: Path) -> None:
    with Database(tmp_path / "graph.db") as db:
        _seed(db)
        result = import_samples(
            db, SHA, [ProfileSample(symbol="sub_140001000", func_addr=0x140001000)]
        )
        assert result["imported"] == 1 and result["attributed_samples"] == 0
        prof = get_profile(db, SHA, 0x140001000)
        assert prof == {"sample_count": 0, "alloc_bytes": 0, "hotness": 0.0}
        assert hottest(db, SHA)[0]["hotness"] == 0.0


def test_reads_without_a_profile_table_return_empty_and_create_nothing(
    tmp_path: Path,
) -> None:
    with Database(tmp_path / "graph.db") as db:
        _seed(db)
        assert hottest(db, SHA) == []
        assert get_profile(db, SHA, 0x140001000) is None
        with db.connection() as conn:
            assert (
                conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE name = 'func_profile'"
                ).fetchone()
                is None
            )


# --------------------------------------------------------------------------- #
# R1 caps + determinism
# --------------------------------------------------------------------------- #


def test_hottest_default_cap_ceiling_and_deterministic_tie_order(tmp_path: Path) -> None:
    with Database(tmp_path / "caps.db") as db:
        addrs = list(range(0x1000, 0x1000 + 1200))
        for addr in addrs:
            db.upsert_function(
                FunctionRecord(binary_sha256=SHA, func_addr=addr, name=f"f_{addr:x}"),
                write_receipt=False,
            )
        result = import_samples(
            db,
            SHA,
            [ProfileSample(func_addr=addr, samples=1) for addr in addrs],
        )
        assert result["imported"] == 1200 and result["unattributed"] == 0

        default_rows = hottest(db, SHA)
        ceiling_rows = hottest(db, SHA, 10_000)
        assert len(default_rows) == DEFAULT_TOP_N == 50
        assert len(ceiling_rows) == MAX_TOP_N == 1000
        # Completed hotness normalizes to 1.0 over the attributed set; each equal
        # weight is 1/1200 and ties break on func_addr ASC.
        assert [r["func_addr"] for r in default_rows] == addrs[:50]
        assert [r["func_addr"] for r in ceiling_rows] == addrs[:1000]
        assert round(sum(r["hotness"] for r in ceiling_rows), 12) == round(
            1000 / 1200, 12
        )
        assert {round(r["hotness"], 12) for r in default_rows} == {round(1 / 1200, 12)}
        assert hottest(db, SHA) == default_rows  # repeated query is byte-identical


def test_resolve_top_n_defaults_clamps_and_refuses() -> None:
    assert resolve_top_n() == DEFAULT_TOP_N == 50
    assert resolve_top_n(None) == DEFAULT_TOP_N
    assert resolve_top_n(7) == 7
    assert resolve_top_n(10_000) == MAX_TOP_N == 1000
    for bad in (0, -1, "many", True):
        with pytest.raises(ProfileError):
            resolve_top_n(bad)
    db_target = None
    with pytest.raises(ProfileError):
        hottest(db_target, SHA, 0)


# --------------------------------------------------------------------------- #
# target handling: raw connections, paths, WinError 32
# --------------------------------------------------------------------------- #


def test_raw_connection_target_is_supported_and_left_unmutated(tmp_path: Path) -> None:
    path = tmp_path / "raw.db"
    with Database(path) as db:
        _seed(db)
    conn = sqlite3.connect(str(path))
    try:
        assert conn.row_factory is None and conn.isolation_level == ""
        ensure_profile_table(conn)
        result = import_samples(conn, SHA, PERF_CSV)
        assert result["imported"] == 2 and result["unattributed"] == 0
        rows = hottest(conn, SHA)
        assert [r["func_addr"] for r in rows] == [0x140001000, 0x140002000]
        assert rows[0]["name"] == "sub_140001000"
        assert get_profile(conn, SHA, 0x140001000)["sample_count"] == 80
        # Caller state (row_factory, isolation_level) is restored, not clobbered.
        assert conn.row_factory is None
        assert conn.isolation_level == ""
        assert conn.in_transaction is False
    finally:
        conn.close()
    path.unlink()


def test_path_target_bootstraps_schema_and_releases_handles(tmp_path: Path) -> None:
    path = tmp_path / "fresh" / "profile.db"
    result = import_samples(
        str(path), SHA, [ProfileSample(symbol="ghost", func_addr=0x10, samples=4)],
        create=True,
    )
    # A brand-new store has no known functions: the sample stays unattributed.
    assert result["imported"] == 0 and result["unattributed"] == 1
    assert result["functions"] == 0 and path.exists()
    # The checker's own handle must be closed too: ``with sqlite3.connect(...)``
    # only ends the transaction, it never closes the connection (traps 7), and a
    # leaked handle here would fail the unlink below for the wrong reason.
    check = sqlite3.connect(str(path))
    try:
        assert check.execute(
            "SELECT 1 FROM sqlite_master WHERE name = 'funcs'"
        ).fetchone() is not None
    finally:
        check.close()
    path.unlink()  # WinError 32 trap: every handle opened above must be closed


def test_missing_or_foreign_database_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ProfileError):
        import_samples(
            str(tmp_path / "nope.db"), SHA, [ProfileSample(symbol="f", samples=1)]
        )
    with pytest.raises(ProfileError):
        hottest(tmp_path / "nope.db", SHA)

    foreign = tmp_path / "foreign.db"
    conn = sqlite3.connect(str(foreign))
    conn.execute("CREATE TABLE unrelated (x INTEGER)")
    conn.commit()
    conn.close()
    with pytest.raises(ProfileError):
        import_samples(
            str(foreign), SHA, [ProfileSample(symbol="f", samples=1)], create=True
        )


def test_hottest_tolerates_a_profile_table_without_funcs(tmp_path: Path) -> None:
    path = tmp_path / "orphan.db"
    conn = sqlite3.connect(str(path))
    try:
        ensure_profile_table(conn)
        assert hottest(conn, SHA) == []  # no funcs table: still a bounded answer
        assert get_profile(conn, SHA, 1) is None
    finally:
        conn.close()
    path.unlink()


# --------------------------------------------------------------------------- #
# P1 slice 1: CLI -- ``profile import`` then ``profile hottest``
# --------------------------------------------------------------------------- #


def _seed_db_file(tmp_path: Path, name: str) -> Path:
    """Seeded db *file* (not handle) for the integration slices below.

    The ``with`` block closes the writer before any CLI/engine/server opens the
    store, so every later reader sees a released file (trap 7).
    """
    path = tmp_path / name
    with Database(path) as db:
        _seed(db)
    return path


def _write_profile(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_cli_profile_import_json_counts_and_replacement(tmp_path: Path) -> None:
    db_path = _seed_db_file(tmp_path, "cli_import.db")
    vtune_path = _write_profile(tmp_path, "hotspots.csv", VTUNE_CSV)
    runner = CliRunner()

    imported = runner.invoke(
        cli,
        ["profile", "import", str(vtune_path), "--db", str(db_path), "--sha", SHA],
    )
    assert imported.exit_code == 0, imported.output
    payload = json.loads(imported.output)
    assert payload["command"] == "profile import" and payload["status"] == "ok"
    assert payload["binary_sha256"] == SHA and payload["format"] == "vtune"
    assert (payload["imported"], payload["unattributed"]) == (3, 1)
    assert payload["rejected"] == 0
    assert payload["total_samples"] == 500
    assert payload["attributed_samples"] == 440
    assert payload["unattributed_samples"] == 60
    assert payload["functions"] == 2

    # A second import replaces the previous rows rather than accumulating.
    etw_path = _write_profile(tmp_path, "hotspots_etw.csv", ETW_CSV)
    second = runner.invoke(
        cli,
        [
            "profile",
            "import",
            str(etw_path),
            "--db",
            str(db_path),
            "--sha",
            SHA,
            "--format",
            "etw",
        ],
    )
    assert second.exit_code == 0, second.output
    second_payload = json.loads(second.output)
    assert second_payload["format"] == "etw"
    assert (second_payload["imported"], second_payload["unattributed"]) == (2, 0)
    assert second_payload["total_samples"] == 15

    listed = runner.invoke(
        cli, ["profile", "hottest", "--db", str(db_path), "--sha", SHA]
    )
    assert listed.exit_code == 0, listed.output
    body = json.loads(listed.output)
    assert body["command"] == "profile hottest" and body["status"] == "ok"
    assert body["count"] == 2 and body["top"] == DEFAULT_TOP_N
    assert [(r["func_addr"], r["sample_count"]) for r in body["hottest"]] == [
        (0x140001000, 10),  # the ETW import's rows, not the VTune ones
        (0x140005000, 5),  # by_sym: address-less row attributed by exact name
    ]
    assert round(sum(r["hotness"] for r in body["hottest"]), 12) == 1.0

    # WinError 32 trap: no handle from either invocation may outlive it.
    db_path.unlink()


def test_cli_profile_hottest_orders_hotness_desc_and_respects_top(tmp_path: Path) -> None:
    db_path = _seed_db_file(tmp_path, "cli_hottest.db")
    csv_path = _write_profile(tmp_path, "trio.csv", TRIO_CSV)
    runner = CliRunner()

    imported = runner.invoke(
        cli, ["profile", "import", str(csv_path), "--db", str(db_path), "--sha", SHA]
    )
    assert imported.exit_code == 0, imported.output
    assert json.loads(imported.output)["imported"] == 3

    listed = runner.invoke(
        cli, ["profile", "hottest", "--db", str(db_path), "--sha", SHA]
    )
    assert listed.exit_code == 0, listed.output
    body = json.loads(listed.output)
    assert [r["func_addr"] for r in body["hottest"]] == [
        0x140001000,
        0x140002000,
        0x140005000,
    ]
    assert [r["hotness"] for r in body["hottest"]] == [0.5, 0.3, 0.2]  # hotness DESC
    assert [r["addr_hex"] for r in body["hottest"]] == [
        "0x140001000",
        "0x140002000",
        "0x140005000",
    ]
    assert body["count"] == 3 and body["top"] == DEFAULT_TOP_N
    # Trap 14: the payload is plain JSON (lists/dicts), so it round-trips.
    assert json.loads(json.dumps(body["hottest"])) == body["hottest"]

    capped = runner.invoke(
        cli,
        ["profile", "hottest", "--db", str(db_path), "--sha", SHA, "--top", "2"],
    )
    assert capped.exit_code == 0, capped.output
    capped_body = json.loads(capped.output)
    assert capped_body["top"] == 2 and capped_body["count"] == 2
    assert capped_body["hottest"] == body["hottest"][:2]

    # R1 determinism: the same call returns the same bounded list.
    repeat = runner.invoke(
        cli, ["profile", "hottest", "--db", str(db_path), "--sha", SHA]
    )
    assert json.loads(repeat.output) == body

    db_path.unlink()


def test_cli_profile_missing_db_bad_top_and_unreadable_csv_exit_1(tmp_path: Path) -> None:
    db_path = _seed_db_file(tmp_path, "cli_errors.db")
    csv_path = _write_profile(tmp_path, "hotspots.csv", VTUNE_CSV)
    missing = tmp_path / "absent.db"
    runner = CliRunner()

    for argv in (
        ["profile", "hottest", "--db", str(missing), "--sha", SHA],
        ["profile", "import", str(csv_path), "--db", str(missing), "--sha", SHA],
    ):
        result = runner.invoke(cli, argv)
        assert result.exit_code == 1
        assert "database not found" in result.output
    # A typo'd --db is refused, never silently created (receipts-or-refusal).
    assert not missing.exists()

    for top in ("0", "5000"):
        result = runner.invoke(
            cli,
            ["profile", "hottest", "--db", str(db_path), "--sha", SHA, "--top", top],
        )
        assert result.exit_code == 1
        assert "must be between 1 and 1000" in result.output

    bad_csv = _write_profile(tmp_path, "bad.csv", "alpha,beta\n1,2\n")
    auto = runner.invoke(
        cli, ["profile", "import", str(bad_csv), "--db", str(db_path), "--sha", SHA]
    )
    assert auto.exit_code == 1
    assert "cannot determine profile format" in auto.output

    forced = runner.invoke(
        cli,
        [
            "profile",
            "import",
            str(bad_csv),
            "--db",
            str(db_path),
            "--sha",
            SHA,
            "--format",
            "vtune",
        ],
    )
    assert forced.exit_code == 1
    assert "header lacks a required column" in forced.output

    db_path.unlink()


def test_cli_profile_import_infers_sha_but_refuses_ambiguity(tmp_path: Path) -> None:
    single = _seed_db_file(tmp_path, "single.db")
    csv_path = _write_profile(tmp_path, "hotspots.csv", VTUNE_CSV)
    runner = CliRunner()

    inferred = runner.invoke(cli, ["profile", "import", str(csv_path), "--db", str(single)])
    assert inferred.exit_code == 0, inferred.output
    assert json.loads(inferred.output)["binary_sha256"] == SHA

    ambiguous = _seed_db_file(tmp_path, "ambiguous.db")
    with Database(ambiguous) as db:
        db.upsert_function(
            FunctionRecord(
                binary_sha256="other_sha", func_addr=0x1000, name="other_fn"
            ),
            write_receipt=True,
        )
    refused = runner.invoke(
        cli, ["profile", "import", str(csv_path), "--db", str(ambiguous)]
    )
    assert refused.exit_code == 1
    assert "Cannot infer binary SHA-256" in refused.output
    assert "2 distinct binaries" in refused.output

    empty = tmp_path / "empty.db"
    with Database(empty):
        pass  # schema exists, but no function was ever observed
    no_funcs = runner.invoke(
        cli, ["profile", "import", str(csv_path), "--db", str(empty)]
    )
    assert no_funcs.exit_code == 1
    assert "holds no functions" in no_funcs.output

    # An explicit --sha never guesses: the ambiguous store accepts the import.
    explicit = runner.invoke(
        cli, ["profile", "import", str(csv_path), "--db", str(ambiguous), "--sha", SHA]
    )
    assert explicit.exit_code == 0, explicit.output
    assert json.loads(explicit.output)["imported"] == 3

    for path in (single, ambiguous, empty):
        path.unlink()


# --------------------------------------------------------------------------- #
# P1 slice 2: QueryEngine -- hottest() delegation + get_function.profile
# --------------------------------------------------------------------------- #


def test_query_engine_hottest_delegates_and_orders_deterministically(
    tmp_path: Path,
) -> None:
    path = _seed_db_file(tmp_path, "engine.db")
    csv_path = _write_profile(tmp_path, "trio.csv", TRIO_CSV)
    with Database(path) as db:
        import_samples(db, SHA, csv_path)
        expected = hottest(db, SHA, 100)
    assert [r["func_addr"] for r in expected] == [
        0x140001000,
        0x140002000,
        0x140005000,
    ]

    with QueryEngine(path) as engine:
        assert engine.hottest(SHA, 100) == expected  # pure delegation to profile
        assert engine.hottest(None) == expected  # sha inferred from one binary
        assert engine.hottest(SHA) == expected  # default budget still bounded
        assert engine.hottest(SHA, 10_000) == expected  # clamped, never unbounded
        assert engine.resolve_limit(10_000) == MAX_LIMIT == MAX_TOP_N
        assert engine.resolve_limit(None) == DEFAULT_LIMIT == DEFAULT_TOP_N
        assert [r["hotness"] for r in expected] == [0.5, 0.3, 0.2]
        assert engine.hottest(SHA, 100) == engine.hottest(SHA, 100)  # deterministic

    # R1 tie-break: equal hotness falls through to func_addr ASC, not CSV order.
    tie_path = _write_profile(tmp_path, "tie.csv", TIE_CSV)
    with Database(path) as db:
        import_samples(db, SHA, tie_path)
    with QueryEngine(path) as engine:
        tied = engine.hottest(SHA, 100)
    assert [r["func_addr"] for r in tied] == [0x140001000, 0x140002000]
    assert [r["hotness"] for r in tied] == [0.5, 0.5]

    path.unlink()


def test_query_engine_get_function_carries_profile_dict_or_none(tmp_path: Path) -> None:
    path = _seed_db_file(tmp_path, "engine_func.db")
    csv_path = _write_profile(tmp_path, "perf.csv", PERF_CSV)
    with Database(path) as db:
        import_samples(db, SHA, csv_path)

    with QueryEngine(path) as engine:
        profiled = engine.get_function(SHA, 0x140001000)
        assert profiled is not None
        assert profiled["profile"] == {
            "sample_count": 80,
            "alloc_bytes": 0,
            "hotness": 0.8,
        }
        assert set(profiled["profile"]) == {"sample_count", "alloc_bytes", "hotness"}
        assert engine.get_function(SHA, "0x140002000")["profile"] == {
            "sample_count": 20,
            "alloc_bytes": 2048,
            "hotness": 0.2,
        }
        # Observed function without a profile row: key present, value None.
        unprofiled = engine.get_function(SHA, 0x140005000)
        assert unprofiled is not None and unprofiled["profile"] is None
        assert engine.get_function(SHA, 0x140009999) is None  # never observed

    # A store that never imported a profile answers None and creates no table.
    bare = _seed_db_file(tmp_path, "engine_bare.db")
    with QueryEngine(bare) as engine:
        assert engine.get_function(SHA, 0x140001000)["profile"] is None
        assert engine.hottest(SHA) == []
    conn = sqlite3.connect(str(bare))
    try:
        created = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name = 'func_profile'"
        ).fetchone()
    finally:
        conn.close()
    assert created is None

    path.unlink()
    bare.unlink()


def test_query_engine_hottest_refuses_bad_budget_and_missing_db(tmp_path: Path) -> None:
    path = _seed_db_file(tmp_path, "engine_caps.db")
    with QueryEngine(path) as engine:
        for bad in (0, -3):
            with pytest.raises(QueryError):
                engine.hottest(SHA, bad)
        with pytest.raises(QueryError):
            engine.resolve_limit("many")

    missing = tmp_path / "engine_absent.db"
    with pytest.raises(QueryError):
        QueryEngine(missing)  # a typo'd path is never silently created
    assert not missing.exists()

    path.unlink()


# --------------------------------------------------------------------------- #
# P1 slice 3: MCP -- toothpick_hottest catalogue entry + bounded call_tool
# --------------------------------------------------------------------------- #


def test_mcp_list_tools_exposes_toothpick_hottest_bounded_schema(tmp_path: Path) -> None:
    path = _seed_db_file(tmp_path, "mcp_tools.db")
    server = create_mcp_server(path, use_sdk=False)
    try:
        assert "toothpick_hottest" in TOOL_NAMES
        assert server.tool_names() == list(TOOL_NAMES)
        assert "toothpick_hottest" in [spec["name"] for spec in server.list_tools()]

        spec = next(s for s in server.list_tools() if s["name"] == "toothpick_hottest")
        schema = spec["inputSchema"]
        assert set(schema["properties"]) == {"binary_sha256", "top_n"}
        assert schema["required"] == []
        assert schema["additionalProperties"] is False
        top_n = schema["properties"]["top_n"]
        assert top_n["minimum"] == 1
        assert top_n["maximum"] == MAX_LIMIT == MAX_TOP_N
        assert top_n["default"] == DEFAULT_LIMIT == DEFAULT_TOP_N

        # list_tools hands out copies: a caller cannot mutate the catalogue.
        spec["description"] = "mutated"
        fresh = next(
            s for s in server.list_tools() if s["name"] == "toothpick_hottest"
        )
        assert fresh["description"] != "mutated"
    finally:
        server.engine.close()


def test_mcp_call_tool_hottest_returns_bounded_hotness_desc_list(tmp_path: Path) -> None:
    path = _seed_db_file(tmp_path, "mcp_call.db")
    csv_path = _write_profile(tmp_path, "trio.csv", TRIO_CSV)
    with Database(path) as db:
        import_samples(db, SHA, csv_path)
        expected = hottest(db, SHA)

    server = create_mcp_server(path, use_sdk=False)
    try:
        payload = server.call_tool("toothpick_hottest", {})
        assert payload["tool"] == "toothpick_hottest"
        assert payload["binary_sha256"] == SHA  # inferred: one binary in the store
        assert payload["top_n"] == DEFAULT_LIMIT == DEFAULT_TOP_N
        assert payload["count"] == 3
        assert payload["hottest"] == expected  # same list as the module slice
        assert [r["func_addr"] for r in payload["hottest"]] == [
            0x140001000,
            0x140002000,
            0x140005000,
        ]
        assert [r["hotness"] for r in payload["hottest"]] == [0.5, 0.3, 0.2]
        assert json.loads(json.dumps(payload)) == payload  # trap 14 round-trip

        capped = server.call_tool("toothpick_hottest", {"binary_sha256": SHA, "top_n": 1})
        assert capped["top_n"] == 1 and capped["count"] == 1
        assert capped["hottest"] == expected[:1]

        over = server.call_tool(
            "toothpick_hottest", {"binary_sha256": SHA, "top_n": 10_000}
        )
        assert over["top_n"] == MAX_LIMIT == MAX_TOP_N  # clamped, never unbounded
        assert over["count"] == 3
    finally:
        server.engine.close()


def test_mcp_call_tool_hottest_errors_and_empty_store(tmp_path: Path) -> None:
    path = _seed_db_file(tmp_path, "mcp_errors.db")
    server = create_mcp_server(path, use_sdk=False)
    try:
        # No profile imported yet: a bounded empty answer, not an error.
        empty = server.call_tool("toothpick_hottest", {"binary_sha256": SHA})
        assert empty["count"] == 0 and empty["hottest"] == []
        assert empty["top_n"] == DEFAULT_LIMIT

        with pytest.raises(McpToolError):
            server.call_tool("toothpick_nope", {})
        with pytest.raises(McpToolError):
            server.call_tool("toothpick_hottest", {"bogus": 1})
        with pytest.raises(McpToolError):
            server.call_tool("toothpick_hottest", {"top_n": 0})
    finally:
        server.engine.close()


# --------------------------------------------------------------------------- #
# trap 11: the suite keeps ``-p no:faulthandler``
# --------------------------------------------------------------------------- #


def test_suite_pins_no_faulthandler_addopt() -> None:
    config = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(
        encoding="utf-8"
    )
    assert "-p no:faulthandler" in config
