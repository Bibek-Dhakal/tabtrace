# Roadmap

## Current Milestone (v0.1.0)

- [x] Deterministic, saved, leakage-checked train/val/test split.
- [x] Three justified engineered features, each unit-tested.
- [x] Baseline (Logistic Regression) + tuned (Random Forest, grid-searched) models.
- [x] Cross-validated evaluation reported as the primary number, held-out test as a secondary check.
- [x] Full unit test suite for all preprocessing/feature/evaluation logic.
- [x] CI (lint + test) gating merges to `main`.
- [x] Automated versioning/changelog via release-please.
- [x] Enforce a minimum coverage threshold in CI (`--cov-fail-under=90`).
- [x] Exploratory-analysis notebook included alongside the package documenting rationale.
- [x] Persist the fitted baseline/tuned models (e.g., via `joblib`) alongside the metrics report.

## Near-Term Goals

- Swap in a different tabular dataset by generalizing `tabtrace.data.ingest` beyond its current dataset-specific validation rules.
- Add a hash/fingerprint check to `get_or_create_split` to detect if the underlying data changed shape since the split was saved.
- Expand the hyperparameter grid in `tabtrace.models.tuned.PARAM_GRID` once moved off a laptop-scale CI runner.

## Known Technical Debt

- `tabtrace.report.build_report` string-builds Markdown by hand; a template engine (e.g., Jinja2) would scale better if the report grows more sections.
