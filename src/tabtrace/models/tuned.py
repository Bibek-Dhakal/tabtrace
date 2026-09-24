"""Tuned model: hyperparameters selected via cross-validated grid search.

Invariant: reported metrics must come from cross-validation, not a single
holdout number. `GridSearchCV` here uses StratifiedKFold internally, and the
CV results (per-fold scores) are returned alongside the fitted best estimator
so the caller can report a mean +/- std rather than one lucky split.
"""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PARAM_GRID = {
    "clf__n_estimators": [100, 200],
    "clf__max_depth": [3, 5, None],
    "clf__min_samples_leaf": [1, 3],
}


def build_tuned_pipeline(seed: int = 42) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(random_state=seed)),
        ]
    )


def train_tuned(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    seed: int = 42,
    cv_folds: int = 5,
    scoring: str = "f1",
) -> tuple[Pipeline, dict]:
    """Grid-search a RandomForestClassifier under stratified k-fold CV.

    Returns:
        (best_estimator, cv_summary): the refit best pipeline, and a summary
        dict with per-config mean/std CV scores plus the winning params.
    """
    pipeline = build_tuned_pipeline(seed=seed)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    search = GridSearchCV(
        pipeline,
        param_grid=PARAM_GRID,
        scoring=scoring,
        cv=cv,
        refit=True,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)

    cv_summary = {
        "best_params": search.best_params_,
        "best_cv_score_mean": float(search.best_score_),
        "cv_folds": cv_folds,
        "scoring": scoring,
    }
    return search.best_estimator_, cv_summary
