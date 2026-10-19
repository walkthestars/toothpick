## Description

<!-- Summarize the changes in this pull request -->

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update
- [ ] CI/CD or packaging update

## Checklist

- [ ] Every commit is signed off (`git commit -s`) per the [DCO](../DCO) and the [CLA](../CLA.md) is accepted — see [CONTRIBUTING.md](../CONTRIBUTING.md#sign-your-work-dco)
- [ ] Full CI harness passes (`uv run python scripts/ci.py`)
- [ ] Linting passes with zero warnings (`uv run ruff check .`)
- [ ] All tests pass (`uv run pytest -p no:faulthandler`)
- [ ] New features include deterministic offline tests
- [ ] End-user documentation (`README.md`, `docs/`) is updated if CLI or MCP interfaces changed
- [ ] `CHANGELOG.md` is updated

## Related Issues

<!-- e.g., Fixes #12 -->
