# Testing

## Tiers

TabTrace's test suite is entirely **unit-level**, matching the project's scope: every preprocessing, splitting, and
feature-engineering function is a pure function over in-memory data, so no integration or E2E harness is needed. The
full pipeline (`scripts/run_pipeline.py`) is exercised manually and via CI as a smoke check, but its component parts
are what's unit-tested.

| Suite     | File                     | Covers                                                                                      |
|-----------|--------------------------|---------------------------------------------------------------------------------------------|
| Ingestion | `tests/test_ingest.py`   | Shape consistency, null detection, dtype checks, label domain checks.                       |
| Split     | `tests/test_split.py`    | Determinism across calls/seeds, disjointness (no leakage), save/reload persistence.         |
| Features  | `tests/test_features.py` | Justification enforcement, correctness of each engineered feature on toy frames.            |
| Metrics   | `tests/test_metrics.py`  | Cross-validation output shape/bounds, held-out metric correctness, failure-case extraction. |

## Execution

```bash
pytest                                          # run everything
pytest tests/test_split.py                      # a single suite
pytest -k "leakage"                             # by keyword
pytest --cov=tabtrace --cov-report=term-missing # with coverage
```

## Mock / Fixture Strategy

Tests favor small, hand-built toy `DataFrame`s (see `tests/test_features.py`, `tests/test_metrics.py`) over mocking
frameworks, since the functions under test are pure and deterministic — a toy frame with known expected output is a
stronger check than a mocked call. `pytest.fixture(scope="module")` is used in `tests/test_split.py` to avoid
reloading the real dataset once per test.

## Coverage

Coverage is measured via `pytest-cov`, configured in `pyproject.toml`
(`addopts = "-ra -q --cov=tabtrace --cov-report=term-missing"`), so a plain `pytest` invocation always reports
coverage. There is no hard-enforced minimum threshold yet; CI currently reports coverage rather than gating on it —
see the roadmap for tightening this.

## CI Integration

`.github/workflows/ci.yml` runs on every push and pull request against `main`: `ruff check`, `ruff format --check`,
and the full `pytest` suite with coverage. A failing run blocks merge — see the System Invariants in the top-level
spec ("no untested code merges to main").
