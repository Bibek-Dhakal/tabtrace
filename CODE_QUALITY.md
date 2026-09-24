# Code Quality & Formatting

> Automated quality checks run on `git commit` (formatting, linting, staged files) and on `git commit --message`
> (commit message validation) via pre-commit hooks, and again repository-wide in CI on every push. Manual fallback
> commands for running everything yourself are below.

TabTrace uses **Ruff** for formatting, import sorting, and linting (it replaces Black + isort + flake8 in a single
tool) and **pytest** for tests. Commit messages are validated against the Conventional Commits spec.

## Environment Setup

Install the project with its dev dependencies, then register the git hooks:

```bash
pip install -e ".[dev]"
pre-commit install
pre-commit install --hook-type commit-msg
```

## Manual Execution Commands

Run every quality check across the whole repository:

```bash
pre-commit run --all-files
```

Run checks only on currently staged files:

```bash
pre-commit run
```

Run checks against a specific branch's changed files or a specific path:

```bash
pre-commit run --from-ref origin/main --to-ref HEAD
pre-commit run --files src/tabtrace/pipeline.py
```

## Isolated Tool Commands

```bash
# Formatter only
ruff format .

# Linter only (with autofix)
ruff check . --fix

# Tests only
pytest

# Tests with coverage report
pytest --cov=tabtrace --cov-report=term-missing
```

## Maintenance & Cache

```bash
# Update hook versions to their latest tagged releases
pre-commit autoupdate

# Clear the local pre-commit hook cache
pre-commit clean

# Clear Ruff's cache
ruff clean
```

## Emergency Bypassing

Use sparingly, and only for genuine hotfixes:

```bash
# Skip pre-commit hooks for a single commit
git commit --no-verify -m "fix: emergency hotfix"
```

Bypassing skips both formatting/lint checks and commit-message validation locally, but CI still runs the full suite
on push — a bypassed local commit will still be caught before merge.
