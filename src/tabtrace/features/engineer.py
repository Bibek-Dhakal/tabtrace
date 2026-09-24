"""Engineered features, each registered with a stated justification.

Invariant: no feature enters the model without a stated justification. This
is enforced in code, not just in prose: `register_feature` raises if a
justification string is missing, and `apply_features` only ever adds columns
that came through the registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd


@dataclass(frozen=True)
class Feature:
    name: str
    func: Callable[[pd.DataFrame], pd.Series]
    justification: str


_REGISTRY: list[Feature] = []


def register_feature(name: str, justification: str):
    """Decorator that registers a feature function together with its rationale.

    Raises:
        ValueError: if no non-trivial justification is supplied.
    """
    if not justification or not justification.strip():
        raise ValueError(f"Feature '{name}' must be registered with a non-empty justification.")

    def decorator(func: Callable[[pd.DataFrame], pd.Series]):
        _REGISTRY.append(Feature(name=name, func=func, justification=justification.strip()))
        return func

    return decorator


def list_features() -> list[Feature]:
    return list(_REGISTRY)


# ---------------------------------------------------------------------------
# Engineered features
# ---------------------------------------------------------------------------


@register_feature(
    name="area_perimeter_ratio",
    justification=(
        "Normalizes cell area by perimeter, capturing shape compactness "
        "independent of raw tumor size. Malignant cells tend to have a "
        "different compactness profile than benign ones even at matched size."
    ),
)
def _area_perimeter_ratio(X: pd.DataFrame) -> pd.Series:
    return X["mean area"] / X["mean perimeter"]


@register_feature(
    name="concavity_symmetry_interaction",
    justification=(
        "Multiplies mean concavity by mean symmetry. Malignant nuclei tend to "
        "be jointly concave AND asymmetric; the two source features are only "
        "weakly correlated on their own, so their product surfaces a combined "
        "effect neither captures individually."
    ),
)
def _concavity_symmetry_interaction(X: pd.DataFrame) -> pd.Series:
    return X["mean concavity"] * X["mean symmetry"]


@register_feature(
    name="radius_worst_mean_ratio",
    justification=(
        "Ratio of the most extreme ('worst') radius measurement to the mean "
        "radius. A high ratio indicates a heterogeneous, unevenly-growing "
        "mass, a pattern associated with malignancy that a single mean value "
        "would smooth over."
    ),
)
def _radius_worst_mean_ratio(X: pd.DataFrame) -> pd.Series:
    return X["worst radius"] / X["mean radius"]


def apply_features(X: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of X with every registered feature appended."""
    X_out = X.copy()
    for feature in _REGISTRY:
        X_out[feature.name] = feature.func(X)
    return X_out


def feature_rationale_table() -> pd.DataFrame:
    """Return a DataFrame of feature name -> justification, for reporting."""
    return pd.DataFrame([{"feature": f.name, "justification": f.justification} for f in _REGISTRY])
