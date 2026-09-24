"""Dataset ingestion and validation.

The pipeline uses scikit-learn's built-in Breast Cancer Wisconsin diagnostic
dataset. It ships with scikit-learn (no network download required), is small
enough to iterate on in minutes on a CPU-only machine, and is a well-scoped
binary classification task suitable for demonstrating disciplined
preprocessing and evaluation.

Every function here is a pure, testable unit -- no notebook-only logic.
"""

from __future__ import annotations

import pandas as pd
from sklearn.datasets import load_breast_cancer


class SchemaValidationError(ValueError):
    """Raised when ingested data fails a schema or integrity check."""


def load_raw_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Load the raw feature matrix and target vector.

    Returns:
        (X, y): feature DataFrame and target Series ("malignant"=0, "benign"=1
        per scikit-learn's convention).
    """
    bunch = load_breast_cancer(as_frame=True)
    X = bunch.data.copy()
    y = bunch.target.copy()
    y.name = "target"
    return X, y


def validate_schema(X: pd.DataFrame, y: pd.Series) -> None:
    """Validate structural invariants of the raw dataset.

    Halts (raises) rather than silently continuing, per the pipeline's
    failure-handling contract: a validation failure must stop the run and
    surface its cause, not produce a silently-wrong result.

    Raises:
        SchemaValidationError: if any invariant is violated.
    """
    if X.empty:
        raise SchemaValidationError("Feature matrix is empty.")

    if len(X) != len(y):
        raise SchemaValidationError(
            f"Row count mismatch: X has {len(X)} rows, y has {len(y)} rows."
        )

    null_counts = X.isnull().sum()
    offending = null_counts[null_counts > 0]
    if not offending.empty:
        raise SchemaValidationError(f"Null values detected in columns: {offending.to_dict()}")

    if y.isnull().any():
        raise SchemaValidationError("Null values detected in target vector.")

    non_numeric = X.select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        raise SchemaValidationError(
            f"Non-numeric feature columns found (expected all-numeric): {non_numeric}"
        )

    unexpected_labels = set(y.unique()) - {0, 1}
    if unexpected_labels:
        raise SchemaValidationError(f"Target contains unexpected labels: {unexpected_labels}")


def load_and_validate() -> tuple[pd.DataFrame, pd.Series]:
    """Load the raw dataset and validate it in one step."""
    X, y = load_raw_dataset()
    validate_schema(X, y)
    return X, y
