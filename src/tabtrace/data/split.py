"""Deterministic train/val/test splitting with saved, re-loadable indices.

Invariant: split indices are fixed and saved -- never re-randomized between
runs. If a saved index file already exists, it is loaded and reused instead
of being regenerated, so re-running the pipeline reproduces the exact same
split every time.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_SEED = 42


class LeakageError(ValueError):
    """Raised when split partitions are not disjoint (a leakage condition)."""


def make_split_indices(
    n_rows: int,
    y: pd.Series,
    seed: int = DEFAULT_SEED,
    val_size: float = 0.15,
    test_size: float = 0.15,
) -> dict[str, list[int]]:
    """Compute stratified, deterministic train/val/test index lists.

    Stratification on the target keeps class balance consistent across
    partitions, which matters for a moderately imbalanced diagnostic dataset.
    """
    all_idx = np.arange(n_rows)

    train_val_idx, test_idx = train_test_split(
        all_idx,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    val_relative_size = val_size / (1 - test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_relative_size,
        random_state=seed,
        stratify=y.iloc[train_val_idx],
    )

    return {
        "seed": seed,
        "train": sorted(int(i) for i in train_idx),
        "val": sorted(int(i) for i in val_idx),
        "test": sorted(int(i) for i in test_idx),
    }


def validate_no_leakage(indices: dict) -> None:
    """Assert the three partitions are pairwise disjoint.

    Raises:
        LeakageError: if any row index appears in more than one partition.
    """
    train, val, test = set(indices["train"]), set(indices["val"]), set(indices["test"])
    overlaps = {
        "train_val": train & val,
        "train_test": train & test,
        "val_test": val & test,
    }
    leaking = {k: v for k, v in overlaps.items() if v}
    if leaking:
        raise LeakageError(f"Overlapping indices detected between partitions: {leaking}")


def save_split_indices(indices: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(indices, f, indent=2)


def load_split_indices(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def get_or_create_split(
    X: pd.DataFrame,
    y: pd.Series,
    path: str | Path,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Load a previously-saved split if present; otherwise create and save one.

    This is what makes the pipeline re-runnable: the same `path` always
    yields the same partition of rows, regardless of how many times the
    pipeline is executed.
    """
    path = Path(path)
    if path.exists():
        indices = load_split_indices(path)
    else:
        indices = make_split_indices(len(X), y, seed=seed)
        save_split_indices(indices, path)

    validate_no_leakage(indices)
    return indices


def apply_split(
    X: pd.DataFrame, y: pd.Series, indices: dict
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Slice X, y into train/val/test according to a saved index dict."""
    X_train, y_train = X.iloc[indices["train"]], y.iloc[indices["train"]]
    X_val, y_val = X.iloc[indices["val"]], y.iloc[indices["val"]]
    X_test, y_test = X.iloc[indices["test"]], y.iloc[indices["test"]]
    return X_train, y_train, X_val, y_val, X_test, y_test
