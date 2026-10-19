# Contributing to Toothpick

Thanks for your interest in contributing to Toothpick!

## Getting Started

 Prerequisites:
- Python **3.11+**
- [`uv`](https://docs.astral.sh/uv/) for dependency management and virtual environments

```bash
git clone https://github.com/walkthestars/toothpick.git
cd toothpick
uv sync --locked --dev
uv run python scripts/ci.py
```

`scripts/ci.py` runs the full five-gate verification suite:
1. `uv run ruff check .` (linting and import hygiene)
2. `uv build` (wheel and sdist packaging verification)
3. `uv run pytest -p no:faulthandler` (full unit, integration, and parser-fuzz test suite)
4. `uv run python scripts/coverage_gate.py` (coverage floor: ≥80% total)
5. `uv run pytest tests/test_license_audit.py` (dependency license and binary-hygiene audit)

> **Windows note**: Always pass `-p no:faulthandler` when running `pytest` directly on Windows (`pyproject.toml` sets this automatically via `addopts`). Native Unicorn CPU emulation registers vectored exception handlers on Windows that conflict with Python's `faulthandler` plugin.

## Development Workflow

1. **Fork** the repository and create a descriptive branch (`feat/arm64-reloc-lifting`, `fix/nso-lz4-header`).
2. **Implement your change** with accompanying unit and edge-case tests under `tests/`.
3. **Verify resource cleanup**: SQLite connections (`Database`, `QueryEngine`) and sockets must be closed deterministically (`with` context managers or `try/finally`) so temporary files clean up without `WinError 32` locks on Windows NTFS.
4. **Maintain dependency hygiene**: All runtime and development dependencies must use permissive OSI-approved licenses (MIT, Apache-2.0, BSD, PSF). Copyleft (GPL/LGPL/AGPL/MPL) dependencies and precompiled binary blobs (`.exe`, `.dll`, `.so`, `.dylib`) in the source tree are rejected by `tests/test_license_audit.py`.
5. **Run the CI harness**: `uv run python scripts/ci.py`.
6. **Sign off your commit (DCO)**: `git commit -s -m "feat(query): ..."` (see below).
7. **Accept the CLA** (first-time contributors only, once ever) — see [Contributor License Agreement](#contributor-license-agreement-cla) below.

## Sign Your Work (DCO)

Toothpick uses the [Developer Certificate of Origin](DCO) **and** a Contributor License Agreement ([CLA.md](CLA.md)). They do different jobs and neither replaces the other: the DCO certifies **where the code came from** (every commit, `Signed-off-by`), while the CLA grants the project's maintainer the relicensing rights described in [CLA.md](CLA.md) (once per contributor). You retain copyright in your contribution under both.

Add a `Signed-off-by` line to each commit certifying that you wrote the code (or otherwise have the right to submit it) and are contributing it under AGPL-3.0-only and the terms of `CLA.md`. Pass `-s` to `git commit`:

```bash
git commit -s -m "fix(unpack): validate PE section alignment before rebuild"
```

which appends:

```text
Signed-off-by: Your Name <your.email@example.com>
```

## Contributor License Agreement (CLA)

In addition to the DCO, Toothpick asks first-time contributors to accept the [Contributor License Agreement](CLA.md) — **once, ever**, not per pull request. It grants the Project Owner the right to license your contribution as part of the project under other terms (such as a commercial or enterprise license), while your contribution itself remains public under AGPL-3.0-only as part of the open-source repository.

**How to sign:** on your first pull request, the CLA Assistant bot will comment with instructions. Read [CLA.md](CLA.md), then comment this exact phrase on the pull request:

```text
I have read the CLA Document and I hereby sign the CLA
```

The bot records your signature on the `cla-signatures` branch and marks the check green for all present and future pull requests.

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat(<scope>): ...` — new capability or CLI option
- `fix(<scope>): ...` — bug fix
- `docs(<scope>): ...` — documentation updates
- `test(<scope>): ...` — test additions or fuzz corpus updates
- `chore(<scope>): ...` — dependency updates or CI maintenance

## Pull Request Checklist

- [ ] `uv run python scripts/ci.py` passes all 5 gates locally
- [ ] All new query or scan outputs enforce explicit `limit` / `max_lines` bounds
- [ ] New features include deterministic offline tests under `tests/`
- [ ] User-facing CLI or MCP changes are reflected in `README.md` and `docs/`
- [ ] Commits carry a valid `Signed-off-by` line (`git commit -s`) and the [CLA](CLA.md) is signed (first-time contributors)
