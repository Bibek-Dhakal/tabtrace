"""Baseline model: fast, simple, establishes the floor to beat."""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_baseline_pipeline(seed: int = 42) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=seed)),
        ]
    )


def train_baseline(X_train: pd.DataFrame, y_train: pd.Series, seed: int = 42) -> Pipeline:
    pipeline = build_baseline_pipeline(seed=seed)
    pipeline.fit(X_train, y_train)
    return pipeline
