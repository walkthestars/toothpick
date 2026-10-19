"""P4 A/B bench tests: wall-clock sampling, IQR-gated verdicts, honest receipts.

Binding contract: ``docs/units/UNIT-P4-BENCH-PLAN.md`` section 1 (module --
``measure``, ``compare``, ``record_claim``, ``BenchError``, the three defaults)
and section 2 (interfaces, including the ``bench run``/``bench compare`` CLI),
plus the acceptance clause of section 1.3
every speed claim carries ``UNVERIFIED (stability)``.

Four properties are pinned here, one per block:

* **A crashing target is never benchmarked** -- non-zero exit and unrunnable
  argv both raise :class:`BenchError` instead of producing a fast sample.
* **Noise is never a win** -- the verdict tests use synthetic timers, so they
  assert the gate itself (median effect *and* disjoint IQR bands) rather than
  the noise of this machine.  Overlapping bands stay ``inconclusive`` even
  when the medians moved by far more than ``min_effect``.
* **A receipt exists and is honestly labelled** -- ``func_addr=-1`` sentinel
  with an explicit ``func_addr: null`` in the payload, ``phase="bench"`` and
  ``verifier_status="UNVERIFIED (stability)"``.
* **The CLI is wiring, not a second implementation** -- ``bench run`` writes the
  RunResult JSON that ``bench compare`` consumes, ``bench compare`` receipts the
  same verdict the module gates, both defer their defaults to this module (the
  flags default to ``None``), and every refusal is an ``Error: ...`` line with
  exit code 1.

The deterministic fixtures are pure stdlib arithmetic; only the ``measure`` and
``bench run`` tests start a process (this interpreter, ``shell=False``).  Every database
opened here is closed before teardown and the path target is unlinked to prove
it (Trap 7: a leaked WAL handle surfaces as ``PermissionError: [WinError 32]``
on Windows); no ``faulthandler`` is installed (``pyproject.toml`` pins
``-p no:faulthandler``).
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import click
import pytest
from click.testing import CliRunner, Result

from toothpick.bench import (
    BENCH_PHASE,
    DEFAULT_MIN_EFFECT,
    DEFAULT_REPS,
    DEFAULT_WARMUP,
    FUNC_ADDR_NONE_SENTINEL,
    VERIFIER_STATUS_STABILITY,
    BenchError,
    RunResult,
    Verdict,
    compare,
    measure,
    record_claim,
)
from toothpick.cli import cli
from toothpick.storage import Database

SHA = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

#: A command that runs and exits 0 without importing anything from this project.
NOOP_CMD = [sys.executable, "-c", "pass"]


def _ticking(value_ms: float, spread_ms: float, reps: int = 5) -> RunResult:
    """A synthetic run centred on ``value_ms`` with a deterministic spread."""
    offsets = [0.0, 0.2, -0.2, 0.1, -0.1][:reps]
    times = [value_ms + offset * spread_ms for offset in offsets]
    return RunResult(cmd=["synthetic"], reps=reps, times_ms=times)


# --------------------------------------------------------------------------- #
# measure: wall-clock, warmup excluded, failures refused
# --------------------------------------------------------------------------- #


def test_defaults_match_plan_section_one() -> None:
    assert (DEFAULT_REPS, DEFAULT_WARMUP, DEFAULT_MIN_EFFECT) == (11, 2, 0.05)
    assert FUNC_ADDR_NONE_SENTINEL == -1
    assert BENCH_PHASE == "bench"


def test_measure_shape_excludes_warmup_and_summarises_samples() -> None:
    result = measure(NOOP_CMD, reps=3, warmup=1, timeout=60)

    assert result.cmd == NOOP_CMD
    assert result.reps == 3
    assert result.warmup == 1
    assert len(result.times_ms) == 3, "warmup runs must not reach times_ms"
    assert all(isinstance(sample, float) and sample >= 0.0 for sample in result.times_ms)
    for statistic in (result.median_ms, result.mean_ms, result.min_ms, result.max_ms):
        assert isinstance(statistic, float) and statistic >= 0.0
    assert isinstance(result.stdev_ms, float) and result.stdev_ms >= 0.0
    assert result.min_ms <= result.median_ms <= result.max_ms
    assert result.min_ms <= result.mean_ms <= result.max_ms

    as_dict = result.as_dict()
    assert set(as_dict) == {
        "cmd",
        "reps",
        "warmup",
        "times_ms",
        "median_ms",
        "mean_ms",
        "stdev_ms",
        "min_ms",
        "max_ms",
    }
    assert json.loads(json.dumps(as_dict)) == as_dict
    assert as_dict["warmup"] == 1 and len(as_dict["times_ms"]) == 3


def test_measure_refuses_a_shell_string_and_an_invalid_plan() -> None:
    with pytest.raises(BenchError, match="argv list"):
        measure(f"{sys.executable} -c pass")  # type: ignore[arg-type]
    with pytest.raises(BenchError, match="must not be empty"):
        measure([])
    with pytest.raises(BenchError, match="reps"):
        measure(NOOP_CMD, reps=0)
    with pytest.raises(BenchError, match="reps"):
        measure(NOOP_CMD, reps=2.5)  # type: ignore[arg-type]
    with pytest.raises(BenchError, match="warmup"):
        measure(NOOP_CMD, warmup=-1)
    with pytest.raises(BenchError, match="timeout"):
        measure(NOOP_CMD, timeout=0)
    with pytest.raises(BenchError, match=r"cmd\[1\]"):
        measure([sys.executable, ""])


def test_measure_non_zero_exit_is_never_a_sample() -> None:
    with pytest.raises(BenchError, match="non-zero exit"):
        measure([sys.executable, "-c", "raise SystemExit(3)"], reps=2, warmup=0)
    # The message carries the exit status and the stderr snippet: a failed
    # target is diagnosable, not silently folded into the timings.
    with pytest.raises(BenchError, match="exited 3"):
        measure([sys.executable, "-c", "import sys; sys.exit(3)"], reps=1, warmup=0)


def test_measure_unrunnable_argv_raises_bench_error() -> None:
    with pytest.raises(BenchError, match="could not be started"):
        measure([str(Path("no") / "such" / "binary-xyz"), "--flag"], warmup=0)


# --------------------------------------------------------------------------- #
# compare: effect + IQR separation, noise stays inconclusive
# --------------------------------------------------------------------------- #


def test_compare_passes_a_real_delta_on_synthetic_timers() -> None:
    before = _ticking(20.0, 0.4)
    after = _ticking(10.0, 0.2)

    verdict = compare(before, after)

    assert verdict.faster is True
    assert verdict.reason.startswith("faster:")
    assert verdict.ratio == pytest.approx(0.5, rel=1e-9)
    assert verdict.delta_ms == pytest.approx(-10.0, rel=1e-9)
    assert verdict.overlap is False
    assert verdict.min_effect == DEFAULT_MIN_EFFECT


def test_compare_overlapping_bands_are_inconclusive_even_when_medians_move() -> None:
    before = RunResult(cmd=["synthetic"], times_ms=[10.0, 20.0, 30.0, 40.0, 50.0])
    after = RunResult(cmd=["synthetic"], times_ms=[11.0, 21.0, 31.0, 41.0, 51.0])

    verdict = compare(before, after)

    assert verdict.bands["before_iqr"] == [20.0, 40.0]
    assert verdict.bands["after_iqr"] == [21.0, 41.0]
    assert verdict.overlap is True
    assert verdict.faster is False
    assert verdict.reason.startswith("inconclusive:")

    # A median move worth far more than min_effect is still noise while the
    # spreads touch (30.0 -> 25.0, ratio 0.8333): the bands settle the verdict.
    shifted = RunResult(cmd=["synthetic"], times_ms=[15.0, 20.0, 25.0, 40.0, 45.0])
    still_noise = compare(before, shifted)
    assert still_noise.ratio <= 1.0 - DEFAULT_MIN_EFFECT
    assert still_noise.overlap is True
    assert still_noise.faster is False
    assert still_noise.reason.startswith("inconclusive:")


def test_compare_touching_bands_count_as_overlap() -> None:
    # Closed bands: Q3(before) == Q1(after) is still contact, not separation.
    before = RunResult(cmd=["synthetic"], times_ms=[8.0, 9.0, 10.0, 12.0, 12.0])
    after = RunResult(cmd=["synthetic"], times_ms=[10.0, 12.0, 13.0, 14.0, 15.0])

    verdict = compare(before, after)

    assert verdict.bands["before_iqr"][1] == verdict.bands["after_iqr"][0] == 12.0
    assert verdict.overlap is True
    assert verdict.faster is False


def test_compare_effect_threshold_is_inclusive_and_min_effect_gates() -> None:
    before = RunResult(cmd=["synthetic"], times_ms=[100.0, 101.0, 102.0, 103.0, 104.0])

    # ratio 0.95 == 1 - min_effect -> a pass (the boundary is inside the gate).
    at_threshold = RunResult(cmd=["synthetic"], times_ms=[94.0, 95.0, 96.0, 97.0, 98.0])
    assert compare(before, at_threshold, min_effect=0.05).faster is True

    # ratio 0.96 > 1 - 0.05 -> separated bands, but not enough of an effect.
    above = RunResult(cmd=["synthetic"], times_ms=[95.0, 96.0, 97.0, 98.0, 99.0])
    verdict = compare(before, above, min_effect=0.05)
    assert verdict.faster is False
    assert verdict.reason.startswith("not faster:")
    assert verdict.overlap is False

    # A strict min_effect turns the same pair into a win.
    assert compare(before, above, min_effect=0.05 - 0.005).faster is True


def test_compare_validates_rule_and_rejects_empty_runs() -> None:
    before = _ticking(20.0, 0.4)
    after = _ticking(10.0, 0.2)

    with pytest.raises(BenchError, match="overlap_rule"):
        compare(before, after, overlap_rule="bootstrap")
    with pytest.raises(BenchError, match="min_effect"):
        compare(before, after, min_effect=1.0)
    with pytest.raises(BenchError, match="min_effect"):
        compare(before, after, min_effect=-0.01)

    empty = RunResult(cmd=["synthetic"], times_ms=[1.0])  # a valid single sample
    statless = RunResult(
        cmd=["synthetic"],
        reps=1,
        median_ms=1.0,
        mean_ms=1.0,
        stdev_ms=0.0,
        min_ms=1.0,
        max_ms=1.0,
    )
    assert statless.times_ms == []
    assert statless.as_dict()["median_ms"] == 1.0
    with pytest.raises(BenchError, match="at least one timing sample"):
        compare(empty, statless)
    with pytest.raises(BenchError, match="positive"):
        compare(RunResult(cmd=["synthetic"], times_ms=[0.0]), empty)


def test_compare_is_deterministic_and_round_trips_through_dicts() -> None:
    before = _ticking(20.0, 0.4)
    after = _ticking(10.0, 0.2)

    first = compare(before, after).as_dict()
    second = compare(before.from_dict(before.as_dict()), after).as_dict()
    assert first == second
    assert json.loads(json.dumps(first)) == first

    restored = Verdict.from_dict(first)
    assert restored.faster is True
    assert restored.as_dict() == first
    assert restored.bands == first["bands"]

    with pytest.raises(BenchError, match="missing"):
        Verdict.from_dict({"faster": True, "ratio": 0.5})
    with pytest.raises(BenchError, match="faster must be a bool"):
        Verdict(faster=1, ratio=0.5, delta_ms=-1.0, reason="x", bands=first["bands"])  # type: ignore[arg-type]
    with pytest.raises(BenchError, match="ascending"):
        Verdict(
            faster=False,
            ratio=1.0,
            delta_ms=0.0,
            reason="x",
            bands={"before_iqr": [2.0, 1.0], "after_iqr": [1.0, 2.0]},
        )
    with pytest.raises(BenchError, match=r"times_ms\[1\]"):
        RunResult(cmd=["synthetic"], times_ms=[1.0, "fast"])  # type: ignore[list-item]
    with pytest.raises(BenchError, match="times_ms or explicit"):
        RunResult(cmd=["synthetic"], reps=1)
    with pytest.raises(BenchError, match="warmup"):
        RunResult(cmd=["synthetic"], times_ms=[1.0], warmup=-2)


# --------------------------------------------------------------------------- #
# record_claim: sentinel row, binding label, idempotent replay
# --------------------------------------------------------------------------- #


def _verdict() -> Verdict:
    return compare(_ticking(20.0, 0.4), _ticking(10.0, 0.2))


def test_record_claim_writes_sentinel_bench_receipt(tmp_path: Path) -> None:
    db_path = tmp_path / "bench.db"
    db = Database(db_path)
    try:
        receipt_id = record_claim(db, SHA, "hook-tick", _verdict())
    finally:
        db.close()

    assert receipt_id.startswith("rcpt_")

    with Database(db_path) as reopened:
        rows = reopened.get_receipts(SHA, FUNC_ADDR_NONE_SENTINEL)
    assert len(rows) == 1
    row = rows[0]
    assert row["receipt_id"] == receipt_id
    assert row["func_addr"] == -1, "Trap 23: negative sentinel, not 0"
    assert row["phase"] == BENCH_PHASE

    payload = row["payload"]
    assert payload["label"] == "hook-tick"
    assert payload["func_addr"] is None, "explicit null preserved in the payload"
    assert payload["verifier_status"] == VERIFIER_STATUS_STABILITY
    assert payload["faster"] is True
    assert payload["bands"] == _verdict().as_dict()["bands"]
    assert "median_ms" not in payload  # the claim is the verdict, not a raw run


def test_record_claim_accepts_a_verdict_mapping_and_validates_input(
    tmp_path: Path,
) -> None:
    # The plan's ``verdict_dict`` form is a supported input, not only Verdict
    # objects: the acceptance path is asserted, not merely the refusals.
    db_path = tmp_path / "mapping.db"
    receipt_id = record_claim(db_path, SHA, "hook-tick", _verdict().as_dict())
    with Database(db_path) as reopened:
        rows = reopened.get_receipts(SHA, FUNC_ADDR_NONE_SENTINEL)
    assert len(rows) == 1
    assert rows[0]["receipt_id"] == receipt_id
    assert rows[0]["payload"]["faster"] is True
    assert rows[0]["payload"]["verifier_status"] == VERIFIER_STATUS_STABILITY

    # A mapping missing required verdict keys is refused, never zero-filled.
    with pytest.raises(BenchError, match="binary sha"):
        record_claim("unused.db", "  ", "hook-tick", _verdict())
    with pytest.raises(BenchError, match="label"):
        record_claim("unused.db", SHA, "   ", _verdict())
    with pytest.raises(BenchError, match="Verdict"):
        record_claim("unused.db", SHA, "hook-tick", {"ratio": 0.5})
    with pytest.raises(BenchError, match="target must be"):
        record_claim(12345, SHA, "hook-tick", _verdict())  # type: ignore[arg-type]

    # A path target is owned by record_claim and closed in its ``finally``: the
    # file is removable here, which a leaked WAL handle would forbid (Trap 7).
    closed_path = tmp_path / "unused-closed.db"
    with pytest.raises(BenchError, match="closed"):
        closed = Database(closed_path)
        closed.close()
        record_claim(closed, SHA, "hook-tick", _verdict())
    for path in (db_path, closed_path):
        path.unlink()
        assert not path.exists()


def test_record_claim_replays_idempotently_and_closes_an_owned_path(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "bench.db"
    verdict = _verdict()

    first = record_claim(db_path, SHA, "hook-tick", verdict)
    second = record_claim(db_path, SHA, "hook-tick", verdict)
    assert first == second, "deterministic receipt id updates the row"

    with Database(db_path) as db:
        assert db.count_receipts(SHA) == 1

    third = record_claim(db_path, SHA, "hook-tick-v2", verdict)
    assert third != first, "a different label is a different claim"
    with Database(db_path) as db:
        assert db.count_receipts(SHA) == 2

    # Trap 7: a leaked WAL handle would make this raise PermissionError.
    db_path.unlink()
    assert not db_path.exists()
    for suffix in ("-wal", "-shm"):
        sidecar = Path(str(db_path) + suffix)
        if sidecar.exists():
            sidecar.unlink()


def test_record_claim_uses_a_raw_connection_without_closing_it(tmp_path: Path) -> None:
    db_path = tmp_path / "bench.db"
    with Database(db_path):
        pass  # create the schema/WAL, then let the Database handle go

    raw = sqlite3.connect(str(db_path))
    try:
        receipt_id = record_claim(raw, SHA, "hook-tick", _verdict())
        row = raw.execute(
            "SELECT receipt_id, binary_sha256, func_addr, phase, payload_json "
            "FROM receipts WHERE receipt_id = ?",
            (receipt_id,),
        ).fetchone()
        assert row is not None, "caller-supplied connection is still usable"
        assert (row[0], row[1], row[2], row[3]) == (
            receipt_id,
            SHA,
            FUNC_ADDR_NONE_SENTINEL,
            BENCH_PHASE,
        )
        assert json.loads(row[4])["verifier_status"] == VERIFIER_STATUS_STABILITY
    finally:
        raw.close()


# --------------------------------------------------------------------------- #
# CLI (Chunk 2a): the ``bench`` group wires straight through to the module
# --------------------------------------------------------------------------- #

NOOP_TARGET = "pass"
FAIL_TARGET = "raise SystemExit(3)"


def _cli(*args: str) -> Result:
    """Invoke the real root CLI (``toothpick.cli.cli``) under a ``CliRunner``."""
    return CliRunner().invoke(cli, list(args))


def _combined(result: Result) -> str:
    """Everything the CLI printed, stderr included (click 8.2 split the streams)."""
    return (result.output or "") + (getattr(result, "stderr", "") or "")


def _write_run(path: Path, value_ms: float, spread_ms: float) -> Path:
    """A synthetic RunResult JSON file -- a fixture, no process is started."""
    path.write_text(
        json.dumps(_ticking(value_ms, spread_ms).as_dict()), encoding="utf-8"
    )
    return path


def test_cli_bench_group_exposes_run_and_compare() -> None:
    result = _cli("bench", "--help")

    assert result.exit_code == 0, result.output
    assert "run" in result.output
    assert "compare" in result.output


def test_cli_bench_options_defer_to_the_module_defaults() -> None:
    # Every flag default is None: bench.py owns 11/2/300/0.05 and the CLI keeps no
    # second copy that could drift away from DEFAULT_REPS/DEFAULT_WARMUP.
    commands = cli.commands["bench"].commands
    run_options = {
        option.name: option.default
        for option in commands["run"].params
        if isinstance(option, click.Option)
    }
    compare_options = {
        option.name: option.default
        for option in commands["compare"].params
        if isinstance(option, click.Option)
    }

    assert set(run_options) >= {"reps", "warmup", "timeout", "out_path"}
    assert run_options["reps"] is None
    assert run_options["warmup"] is None
    assert run_options["timeout"] is None
    assert run_options["out_path"] is None
    assert compare_options["min_effect"] is None
    assert compare_options["db_path"] is None


def test_cli_imports_the_bench_module_lazily() -> None:
    # The CLI module must not pay for toothpick.bench at import time; the first
    # ``bench`` subcommand must pull it in (the discipline ``lua``/``unpack`` use).
    probe = (
        "import sys\n"
        "from click.testing import CliRunner\n"
        "import toothpick.cli\n"
        "eager = 'toothpick.bench' in sys.modules\n"
        "CliRunner().invoke(toothpick.cli.cli, ['bench', 'run'])\n"
        "print(eager, 'toothpick.bench' in sys.modules)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "False True", completed.stdout


def test_cli_bench_run_writes_out_json_and_emits_the_same_run(tmp_path: Path) -> None:
    out = tmp_path / "before.json"

    result = _cli(
        "bench",
        "run",
        "--reps",
        "2",
        "--warmup",
        "1",
        "--out",
        str(out),
        "--",
        sys.executable,
        "-c",
        NOOP_TARGET,
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["command"] == "bench run"
    assert payload["reps"] == 2, "--reps reaches measure()"
    assert payload["warmup"] == 1, "--warmup reaches measure()"
    assert len(payload["times_ms"]) == 2, "warmup runs never reach times_ms"
    assert payload["out_path"] == str(out)

    document = json.loads(out.read_text(encoding="utf-8"))
    assert "command" not in document and "out_path" not in document
    assert RunResult.from_dict(document).as_dict() == document
    assert payload["median_ms"] == pytest.approx(document["median_ms"])


def test_cli_bench_compare_gates_and_receipts_the_claim(tmp_path: Path) -> None:
    before = _write_run(tmp_path / "before.json", 20.0, 0.4)
    after = _write_run(tmp_path / "after.json", 10.0, 0.2)
    db_path = tmp_path / "game.db"

    result = _cli(
        "bench",
        "compare",
        str(before),
        str(after),
        "--db",
        str(db_path),
        "--sha",
        SHA,
        "--label",
        "hook-tick",
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["command"] == "bench compare"
    assert payload["faster"] is True, payload["reason"]
    assert payload["ratio"] == pytest.approx(0.5, rel=1e-9)
    assert payload["before_median_ms"] == pytest.approx(20.0)
    assert payload["after_median_ms"] == pytest.approx(10.0)
    assert payload["before_json"] == str(before)
    assert payload["after_json"] == str(after)
    assert payload["verifier_status"] == VERIFIER_STATUS_STABILITY
    assert payload["receipt_id"].startswith("rcpt_")

    with Database(db_path) as reopened:
        rows = reopened.get_receipts(SHA, FUNC_ADDR_NONE_SENTINEL)
    assert len(rows) == 1
    assert rows[0]["receipt_id"] == payload["receipt_id"]
    assert rows[0]["payload"]["label"] == "hook-tick"
    assert rows[0]["payload"]["faster"] is True
    assert rows[0]["payload"]["verifier_status"] == VERIFIER_STATUS_STABILITY


def test_cli_bench_compare_passes_min_effect_through(tmp_path: Path) -> None:
    # median 20.0 -> 19.2 is ratio 0.96 with disjoint bands: the verdict flips on
    # --min-effect alone, which is exactly the passthrough under test.
    before = _write_run(tmp_path / "before.json", 20.0, 0.4)
    near = _write_run(tmp_path / "near.json", 19.2, 0.4)

    default_gate = _cli("bench", "compare", str(before), str(near))
    explicit_gate = _cli(
        "bench", "compare", str(before), str(near), "--min-effect", "0.05"
    )
    relaxed_gate = _cli(
        "bench", "compare", str(before), str(near), "--min-effect", "0.03"
    )

    for result in (default_gate, explicit_gate, relaxed_gate):
        assert result.exit_code == 0, result.output
    default_payload = json.loads(default_gate.output)
    explicit_payload = json.loads(explicit_gate.output)
    relaxed_payload = json.loads(relaxed_gate.output)

    assert default_payload["min_effect"] == DEFAULT_MIN_EFFECT
    assert explicit_payload["min_effect"] == pytest.approx(0.05)
    assert relaxed_payload["min_effect"] == pytest.approx(0.03)
    assert default_payload["faster"] is False, "0.96 > 1 - 0.05 is not an effect"
    assert default_payload == explicit_payload, "the flag reproduces the default"
    assert relaxed_payload["faster"] is True, "0.96 <= 1 - 0.03 clears the gate"
    assert relaxed_payload["ratio"] == pytest.approx(0.96, rel=1e-9)


def test_cli_bench_error_paths_exit_one_with_an_error_line(tmp_path: Path) -> None:
    before = _write_run(tmp_path / "before.json", 20.0, 0.4)
    after = _write_run(tmp_path / "after.json", 10.0, 0.2)

    statless = tmp_path / "statless.json"
    statless.write_text(
        json.dumps({**_ticking(20.0, 0.4).as_dict(), "times_ms": []}),
        encoding="utf-8",
    )
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{not json", encoding="utf-8")
    verdict_like = tmp_path / "verdict.json"
    verdict_like.write_text(json.dumps(_verdict().as_dict()), encoding="utf-8")
    unused_db = tmp_path / "unused.db"

    run_argv = ["--", sys.executable, "-c", NOOP_TARGET]
    failing_argv = ["--", sys.executable, "-c", FAIL_TARGET]
    cases = [
        (["bench", "run"], "requires a command"),
        (["bench", "run", "--reps", "0", "--warmup", "0", *run_argv], "reps must be >= 1"),
        (["bench", "run", "--reps", "1", "--warmup", "0", *failing_argv], "exited 3"),
        (
            [
                "bench",
                "run",
                "--reps",
                "1",
                "--warmup",
                "0",
                "--out",
                str(tmp_path / "no" / "such" / "dir" / "out.json"),
                *run_argv,
            ],
            "cannot write",
        ),
        (["bench", "compare", str(tmp_path / "missing.json"), str(after)], "cannot read"),
        (["bench", "compare", str(malformed), str(after)], "not valid JSON"),
        (["bench", "compare", str(verdict_like), str(after)], "not a bench RunResult JSON"),
        (["bench", "compare", str(statless), str(after)], "at least one timing sample"),
        (["bench", "compare", str(before), str(after), "--min-effect", "1.0"], "min_effect"),
        (
            ["bench", "compare", str(before), str(after), "--db", str(unused_db)],
            "--sha is required",
        ),
        (
            [
                "bench",
                "compare",
                str(before),
                str(after),
                "--db",
                str(unused_db),
                "--sha",
                SHA,
            ],
            "--label is required",
        ),
    ]

    for args, expected in cases:
        result = _cli(*args)
        combined = _combined(result)
        assert result.exit_code == 1, f"{expected!r}: exit {result.exit_code} ({combined})"
        assert "Error: " in combined, f"{expected!r}: no 'Error: ' line ({combined})"
        assert expected in combined, f"{expected!r} missing from {combined!r}"
        assert "Traceback" not in combined

    # The --sha/--label guards fire before this harness writes anything.
    assert not unused_db.exists()
