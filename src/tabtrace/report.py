"""Generates the human-readable metrics + rationale report."""

from __future__ import annotations

import json
from pathlib import Path

from tabtrace.features.engineer import feature_rationale_table


def build_report(
    baseline_cv: dict,
    tuned_cv: dict,
    tuned_search_summary: dict,
    baseline_test: dict,
    tuned_test: dict,
) -> str:
    rationale = feature_rationale_table()

    lines: list[str] = []
    lines.append("# TabTrace — Metrics & Rationale Report\n")

    lines.append("## Cross-Validated Model Comparison\n")
    lines.append("| Model | Scoring | CV Mean | CV Std | Folds |")
    lines.append("|---|---|---|---|---|")
    lines.append(
        f"| Baseline (Logistic Regression) | {baseline_cv['scoring']} | "
        f"{baseline_cv['mean']:.4f} | {baseline_cv['std']:.4f} | {baseline_cv['cv_folds']} |"
    )
    lines.append(
        f"| Tuned (Random Forest, grid-searched) | {tuned_cv['scoring']} | "
        f"{tuned_cv['mean']:.4f} | {tuned_cv['std']:.4f} | {tuned_cv['cv_folds']} |"
    )
    lines.append("")

    lines.append("## Tuning Decision\n")
    lines.append(
        f"Best hyperparameters selected by grid search under "
        f"{tuned_search_summary['cv_folds']}-fold stratified CV, optimizing "
        f"`{tuned_search_summary['scoring']}`:\n"
    )
    lines.append("```json")
    lines.append(json.dumps(tuned_search_summary["best_params"], indent=2))
    lines.append("```")
    lines.append(
        "\nRationale: a grid search over tree depth, leaf size, and estimator "
        "count was used (rather than manual re-running) so the selected "
        "configuration is the one that generalizes best across folds, not the "
        "one that happened to score highest on a single split.\n"
    )

    if baseline_cv["mean"] > tuned_cv["mean"]:
        lines.append(
            "**Observation**: The baseline Logistic Regression model outperformed the "
            "tuned Random Forest model on cross-validation. This suggests that for this "
            "specific dataset, a linear decision boundary is highly effective, and the "
            "added complexity of a tree ensemble may be prone to overfitting or simply "
            "unnecessary given the strong linearly-separable signal in the engineered features.\n"
        )
    elif baseline_cv["mean"] < tuned_cv["mean"]:
        lines.append(
            "**Observation**: The tuned Random Forest model outperformed the baseline "
            "Logistic Regression model. This suggests that the tree ensemble was able to "
            "leverage non-linear relationships in the engineered features effectively.\n"
        )

    lines.append("## Held-Out Test Metrics (secondary sanity check)\n")
    lines.append("| Model | Accuracy | Precision | Recall | F1 | ROC AUC |")
    lines.append("|---|---|---|---|---|---|")
    for label, m in (("Baseline", baseline_test), ("Tuned", tuned_test)):
        lines.append(
            f"| {label} | {m['accuracy']:.4f} | {m['precision']:.4f} | "
            f"{m['recall']:.4f} | {m['f1']:.4f} | {m.get('roc_auc', float('nan')):.4f} |"
        )
    lines.append("")

    lines.append("## Engineered Feature Rationale\n")
    lines.append("| Feature | Justification |")
    lines.append("|---|---|")
    for _, row in rationale.iterrows():
        lines.append(f"| `{row['feature']}` | {row['justification']} |")
    lines.append("")

    return "\n".join(lines)


def save_report(report_text: str, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report_text, encoding="utf-8")
