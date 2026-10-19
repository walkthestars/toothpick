"""Distribution packaging and CLI entrypoint verification test suite.

Governed by design-document.md §13, §15; execution-plan.md §1 (W12).

Enforces:
1. Wheel and sdist distribution artifacts build cleanly via uv build.
2. Wheel package structure and PEP 621 METADATA conform to specification.
3. Console script entrypoint (toothpick) registers all subcommands.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import zipfile

from click.testing import CliRunner

from toothpick.cli import cli


def test_distribution_artifacts_build(tmp_path: Path) -> None:
    """Verify that uv build produces valid sdist and wheel packages in dist/."""
    repo_root = Path(__file__).resolve().parent.parent

    # Execute uv build
    res = subprocess.run(
        ["uv", "build", "--out-dir", str(tmp_path)],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, f"uv build failed: {res.stderr}\n{res.stdout}"

    wheels = list(tmp_path.glob("*.whl"))
    tarballs = list(tmp_path.glob("*.tar.gz"))

    assert len(wheels) == 1, f"Expected 1 wheel artifact, found: {wheels}"
    assert len(tarballs) == 1, f"Expected 1 sdist artifact, found: {tarballs}"

    wheel_path = wheels[0]
    tarball_path = tarballs[0]

    assert wheel_path.stat().st_size > 50_000, "Wheel unexpectedly small"
    assert tarball_path.stat().st_size > 50_000, "Sdist tarball unexpectedly small"

    # Verify wheel archive structure
    with zipfile.ZipFile(wheel_path, "r") as zf:
        namelist = zf.namelist()

        # Core source files present in package
        assert "toothpick/__init__.py" in namelist
        assert "toothpick/cli.py" in namelist
        assert "toothpick/static_engine.py" in namelist
        assert "toothpick/storage.py" in namelist
        assert "toothpick/solver.py" in namelist
        assert "toothpick/rebase.py" in namelist
        assert "toothpick/observability.py" in namelist
        assert "toothpick/mcp_server.py" in namelist
        assert "toothpick/mod_builder.py" in namelist

        # Metadata records
        metadata_files = [n for n in namelist if n.endswith(".dist-info/METADATA")]
        assert len(metadata_files) == 1, "Missing METADATA in wheel"

        metadata_content = zf.read(metadata_files[0]).decode("utf-8")
        assert "Name: toothpick" in metadata_content
        assert "Version: 0.1.0" in metadata_content
        assert (
            "License: AGPL-3.0-only" in metadata_content
            or "License-Expression: AGPL-3.0-only" in metadata_content
        )

        # Entry point script record
        entry_points_files = [n for n in namelist if n.endswith(".dist-info/entry_points.txt")]
        assert len(entry_points_files) == 1, "Missing entry_points.txt in wheel"
        entry_content = zf.read(entry_points_files[0]).decode("utf-8")
        assert "toothpick = toothpick.cli:cli" in entry_content


def test_cli_entrypoint_all_subcommands_registered() -> None:
    """Verify that toothpick CLI registers all 13 core subcommands."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0, f"toothpick --help failed: {result.output}"

    expected_subcommands = [
        "triage",
        "unpack",
        "static",
        "query",
        "solve",
        "ground",
        "scan",
        "ingest-metadata",
        "build",
        "rebase",
        "show",
        "ledger",
        "mcp",
    ]

    help_text = result.output.lower()
    missing_commands = [cmd for cmd in expected_subcommands if cmd not in help_text]
    assert not missing_commands, f"Missing CLI subcommands in --help: {missing_commands}"


def test_cli_subcommand_individual_help_screens() -> None:
    """Verify that every individual CLI subcommand responds to --help with status code 0."""
    runner = CliRunner()
    subcommands = [
        "triage",
        "unpack",
        "static",
        "query",
        "solve",
        "ground",
        "scan",
        "ingest-metadata",
        "build",
        "rebase",
        "show",
        "ledger",
        "mcp",
    ]

    for cmd in subcommands:
        res = runner.invoke(cli, [cmd, "--help"])
        assert res.exit_code == 0, f"toothpick {cmd} --help failed with exit code {res.exit_code}:\n{res.output}"
        assert cmd in res.output.lower()
        assert "options:" in res.output.lower() or "--help" in res.output.lower()
