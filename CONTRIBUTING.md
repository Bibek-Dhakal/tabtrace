# Contributing to TabTrace

Thanks for considering a contribution. This is a portfolio project, but it follows the same process discipline as a
production repository, so the workflow itself is part of the demonstration.

## Getting Set Up

```bash
git clone <this-repo>
cd tabtrace
pip install -e ".[dev]"
pre-commit install
pre-commit install --hook-type commit-msg
```

See [CODE_QUALITY.md](CODE_QUALITY.md) for the full formatting/linting/testing command reference.

## Code Style

- Ruff enforces formatting and linting**** via pre-commit and CI — see [CODE_QUALITY.md](CODE_QUALITY.md).
- Keep preprocessing and feature-engineering logic in `src/tabtrace/`, as testable functions — never notebook-only.
- Every new engineered feature must be registered with a non-empty justification (`register_feature(...)` in
  `src/tabtrace/features/engineer.py`); the registry raises if you omit one.

## Testing

- Add or update tests under `tests/` for any behavioral change.
- Run the full suite locally before opening a PR: `pytest`.
- CI runs the same suite on every push, and blocks merge on failure — see `.github/workflows/`.

## Commit Messages: Conventional Commits (Required)

All commit messages **and PR titles** must follow the [Conventional Commits](https://www.conventionalcommits.org/)
format:

```text
type(scope): short description
```

Commit messages are not just style — they directly drive automated versioning and `CHANGELOG.md` generation via
**release-please**. The prefix you choose determines the next semantic version:

| Prefix                          | Meaning                              | Version bump    |
|----------------------------------|---------------------------------------|-----------------|
| `feat:`                          | A new feature                         | Minor (`0.X.0`) |
| `fix:`                           | A bug fix                             | Patch (`0.0.X`) |
| `feat!:` or `fix!:` (or a `BREAKING CHANGE:` footer) | A breaking change | Major (`X.0.0`) |
| `docs:`                          | Documentation only                    | None (changelog note only) |
| `chore:`                         | Tooling, deps, maintenance            | None (changelog note only) |
| `test:`                          | Adding or fixing tests                | None            |
| `refactor:`                      | Code change with no behavior change   | None            |

Examples:

```text
feat(features): add radius_worst_mean_ratio engineered feature
fix(split): correct stratification on val/test carve-out
docs(readme): clarify local run instructions
feat!: change report.md schema (breaking for downstream parsers)
```

## Release Process (release-please)

This repo uses Google's `release-please` GitHub Action (`.github/workflows/release-please.yml`). It watches `main`
and maintains a single, continuously updated **Release PR** (`chore: release x.y.z`) that aggregates every
conventional commit merged since the last release into a draft `CHANGELOG.md` entry and version bump.

- Pushing conventional commits to `main` updates the open Release PR — it does **not** immediately cut a release.
- A new version, Git tag (`vX.Y.Z`), and finalized `CHANGELOG.md` section are only created when the Release PR
  itself is merged.
- Git tags determine release boundaries strictly: commits between two tags belong to that release, commits
  after the latest tag are "unreleased" and roll into the next Release PR.

## Pull Requests

1. Branch from `main`.
2. Keep commits scoped and conventional.
3. Ensure `pre-commit run --all-files` and `pytest` pass locally.
4. Open a PR with a conventional-commit-style title; CI must pass (lint + tests) before merge.
5. Since this is currently a single-maintainer project, a self-review pass against a short checklist (tests added,
   docs updated if needed, CI green, no stray debug output) stands in for a second reviewer.
