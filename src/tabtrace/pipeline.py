"""Orchestrates the full TabTrace journey:

raw dataset -> ingestion+validation -> deterministic split -> feature
construction -> baseline training -> tuned training -> evaluation -> report.

Halts on the first failed invariant (schema violation, leakage, missing
feature justification) rather than silently producing a result.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib

from tabtrace.data.ingest import load_and_validate
from tabtrace.data.split import apply_split, get_or_create_split
from tabtrace.evaluation.metrics import cross_validated_scores, holdout_metrics
from tabtrace.features.engineer import apply_features
from tabtrace.models.baseline import train_baseline
from tabtrace.models.tuned import train_tuned
from tabtrace.report import build_report, save_report

OUTPUT_DIR = Path("outputs")
SPLIT_INDEX_PATH = OUTPUT_DIR / "split_indices.json"
REPORT_PATH = OUTPUT_DIR / "report.md"
CV_RESULTS_PATH = OUTPUT_DIR / "cv_results.json"

SEED = 42


def run_pipeline(seed: int = SEED, output_dir: Path | str = OUTPUT_DIR) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ingestion + validation (halts on schema/null issues)
    X_raw, y = load_and_validate()

    # 2. Deterministic split (halts on leakage between partitions)
    indices = get_or_create_split(X_raw, y, output_dir / "split_indices.json", seed=seed)
    X_train_raw, y_train, X_val_raw, y_val, X_test_raw, y_test = apply_split(X_raw, y, indices)

    # 3. Feature construction (every column added is justified; see features/engineer.py)
    X_train = apply_features(X_train_raw)
    X_val = apply_features(X_val_raw)
    X_test = apply_features(X_test_raw)

    # 4. Baseline model
    baseline_model = train_baseline(X_train, y_train, seed=seed)
    baseline_cv = cross_validated_scores(baseline_model, X_train, y_train, seed=seed)
    baseline_test = holdout_metrics(baseline_model, X_test, y_test)

    # 5. Tuned model (cross-validated grid search)
    tuned_model, tuned_search_summary = train_tuned(X_train, y_train, seed=seed)
    tuned_cv = cross_validated_scores(tuned_model, X_train, y_train, seed=seed)
    tuned_test = holdout_metrics(tuned_model, X_test, y_test)

    # 6. Report generation & Save models
    report_text = build_report(
        baseline_cv=baseline_cv,
        tuned_cv=tuned_cv,
        tuned_search_summary=tuned_search_summary,
        baseline_test=baseline_test,
        tuned_test=tuned_test,
    )
    save_report(report_text, output_dir / "report.md")

    joblib.dump(baseline_model, output_dir / "baseline_model.joblib")
    joblib.dump(tuned_model, output_dir / "tuned_model.joblib")

    results = {
        "baseline_cv": baseline_cv,
        "tuned_cv": tuned_cv,
        "tuned_search_summary": tuned_search_summary,
        "baseline_test": baseline_test,
        "tuned_test": tuned_test,
        "val_set_size": len(X_val),
    }
    with open(output_dir / "cv_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    run_pipeline()
