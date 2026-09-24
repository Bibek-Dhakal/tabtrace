# Usage

## Install

```bash
pip install -e ".[dev]"
# To also install dependencies for exploratory notebooks (like matplotlib and jupyter):
pip install -e ".[dev,notebooks]"
```

## Run the Full Pipeline

```bash
python scripts/run_pipeline.py
```

This runs ingestion → validation → split → feature construction → baseline training → tuned training → evaluation →
report generation, writing to `outputs/`:

| Output file                     | Contents                                                                    |
|---------------------------------|-----------------------------------------------------------------------------|
| `outputs/split_indices.json`    | The persisted, deterministic train/val/test row indices.                    |
| `outputs/cv_results.json`       | Machine-readable CV + held-out metrics for both models.                     |
| `outputs/report.md`             | Human-readable metrics table, tuning rationale, and feature justifications. |
| `outputs/baseline_model.joblib` | The fitted baseline Pipeline (Logistic Regression), for downstream use.     |
| `outputs/tuned_model.joblib`    | The refitted tuned Pipeline (Random Forest), for downstream use.            |

Re-running the script reuses the existing `outputs/split_indices.json` rather than generating a new split — delete
that file first if you deliberately want a fresh split with a different seed.

## Run Tests

```bash
pytest
pytest --cov=tabtrace --cov-report=term-missing   # with coverage
```

## Environment Variables

TabTrace has no external services, secrets, or network calls (the dataset ships with scikit-learn), so **no
environment variables are required to run it**. If you extend it with a config file or a different data source,
document new variables here as: `Name | Type | Default | Description`.

## Commands / Entry Points

| Command                          | Description                                                                                |
|----------------------------------|--------------------------------------------------------------------------------------------|
| `python scripts/run_pipeline.py` | Run the full pipeline end to end (equivalent to `tabtrace` console script once installed). |
| `tabtrace`                       | Same as above, available after `pip install -e .` registers the console script.            |
| `pytest`                         | Run the unit test suite.                                                                   |
| `ruff check .` / `ruff format .` | Lint / format the codebase.                                                                |
| `pre-commit run --all-files`     | Run all quality gates locally, matching CI.                                                |

---
