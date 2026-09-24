"""Evaluation utilities: cross-validated scores plus a held-out test check.

Cross-validation is the primary reported number (per the pipeline's
invariants); the held-out test set is used only as a final, secondary sanity
check and is always reported alongside, never as a substitute.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score


def cross_validated_scores(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int = 5,
    seed: int = 42,
    scoring: str = "f1",
) -> dict:
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    scores = cross_val_score(estimator, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    return {
        "scoring": scoring,
        "cv_folds": cv_folds,
        "fold_scores": [float(s) for s in scores],
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
    }


def holdout_metrics(estimator: BaseEstimator, X: pd.DataFrame, y: pd.Series) -> dict:
    """Compute a fixed set of metrics on a single held-out partition.

    Reported as a secondary sanity check alongside cross-validation, never
    as the sole evaluation number.
    """
    preds = estimator.predict(X)
    metrics = {
        "accuracy": float(accuracy_score(y, preds)),
        "precision": float(precision_score(y, preds)),
        "recall": float(recall_score(y, preds)),
        "f1": float(f1_score(y, preds)),
    }
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)[:, 1]
        metrics["roc_auc"] = float(roc_auc_score(y, proba))
    return metrics


def failure_cases(estimator: BaseEstimator, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Return the misclassified rows of a partition, for qualitative inspection."""
    preds = estimator.predict(X)
    mask = preds != y.values
    out = X.loc[mask].copy()
    out["true_label"] = y.values[mask]
    out["predicted_label"] = preds[mask]
    return out
