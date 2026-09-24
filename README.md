# TabTrace

**Reproducible tabular ML pipeline with justified feature engineering and cross-validated evaluation.**

Most self-taught ML work lives in a single notebook: an unfixed split, features picked without justification, and
"tuning" that's really re-running cells until a number looks good. TabTrace is the opposite of that on purpose — every
stage is a pure, unit-tested function; every engineered feature carries a stated rationale enforced in code; every
reported metric comes from cross-validation, not a lucky holdout; and the whole pipeline halts loudly on a bad split
or bad data rather than producing a silently-wrong result.

## What It Does

Runs the Breast Cancer Wisconsin (Diagnostic) dataset (bundled with scikit-learn, no download needed) through:

```text
raw dataset
→ ingestion + validation (schema/null checks, halts on failure)
→ deterministic split (train/val/test, fixed seed, saved indices)
→ feature construction (3 engineered features, each justified + unit-tested)
→ baseline model (Logistic Regression)
→ tuned model (Random Forest, grid-searched under stratified 5-fold CV)
→ evaluation (baseline vs. tuned, CV metrics + held-out check + failure-case inspection)
→ report generation (outputs/report.md)
```

See [`docs/architecture/README.md`](docs/architecture/README.md) for a diagram and module map.

## Quickstart

```bash
pip install -e ".[dev, notebooks]"
python scripts/run_pipeline.py
pytest
```

Full usage details, commands, and output file reference: [`docs/usage/README.md`](docs/usage/README.md).

## Documentation

| Doc                                                 | Covers                                                     |
|-----------------------------------------------------|------------------------------------------------------------|
| [`docs/usage/`](docs/usage/README.md)               | Commands, outputs, environment variables.                  |
| [`docs/architecture/`](docs/architecture/README.md) | Pipeline data flow diagram, module responsibilities.       |
| [`docs/testing/`](docs/testing/README.md)           | Test tiers, execution, coverage, CI integration.           |
| [`docs/roadmap/`](docs/roadmap/README.md)           | Current milestone, near-term goals, known technical debt.  |
| [`docs/data/`](docs/data/README.md)                 | Dataset source, preprocessing, and split policy.           |
| [`CODE_QUALITY.md`](CODE_QUALITY.md)                | Formatting/linting/testing commands and pre-commit setup.  |
| [`CONTRIBUTING.md`](CONTRIBUTING.md)                | Conventional Commits, release-please workflow, PR process. |

## System Invariants

```text
No feature enters the model without a stated justification.
Split indices are fixed and saved — never re-randomized between runs.
Reported metrics must come from cross-validation, not a single split.
Preprocessing/feature logic must be unit-tested, not notebook-only.
```

## Project Structure

```text
tabtrace/
├── src/tabtrace/            # installable package — all pipeline logic
│   ├── data/                # ingestion + validation, deterministic split
│   ├── features/            # justified, registered feature engineering
│   ├── models/               # baseline + tuned (CV grid-searched) models
│   ├── evaluation/           # cross-validated + held-out metrics
│   ├── pipeline.py           # orchestrates the full journey
│   └── report.py             # renders outputs/report.md
├── tests/                    # unit tests mirroring src/tabtrace/
├── scripts/run_pipeline.py   # CLI entry point
├── notebooks/                # exploratory analysis (see notebooks/README.md)
├── outputs/                  # generated: split indices, metrics, report
└── docs/                     # see Documentation table above
```

## License

MIT — see [`LICENSE`](LICENSE).
