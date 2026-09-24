import numpy as np
import pandas as pd
import pytest

from tabtrace.data.ingest import (
    SchemaValidationError,
    load_and_validate,
    load_raw_dataset,
    validate_schema,
)


def test_load_raw_dataset_shapes_match():
    X, y = load_raw_dataset()
    assert len(X) == len(y)
    assert len(X) > 0


def test_load_raw_dataset_no_nulls():
    X, y = load_raw_dataset()
    assert not X.isnull().values.any()
    assert not y.isnull().values.any()


def test_validate_schema_passes_on_clean_data():
    X, y = load_raw_dataset()
    validate_schema(X, y)  # should not raise


def test_validate_schema_rejects_row_mismatch():
    X, y = load_raw_dataset()
    with pytest.raises(SchemaValidationError):
        validate_schema(X, y.iloc[:-1])


def test_validate_schema_rejects_nulls():
    X, y = load_raw_dataset()
    X_bad = X.copy()
    X_bad.iloc[0, 0] = np.nan
    with pytest.raises(SchemaValidationError):
        validate_schema(X_bad, y)


def test_validate_schema_rejects_non_numeric_column():
    X, y = load_raw_dataset()
    X_bad = X.copy()
    X_bad["extra"] = "not-a-number"
    with pytest.raises(SchemaValidationError):
        validate_schema(X_bad, y)


def test_validate_schema_rejects_unexpected_labels():
    X, y = load_raw_dataset()
    y_bad = y.copy()
    y_bad.iloc[0] = 2
    with pytest.raises(SchemaValidationError):
        validate_schema(X, y_bad)


def test_validate_schema_rejects_empty_frame():
    with pytest.raises(SchemaValidationError):
        validate_schema(pd.DataFrame(), pd.Series(dtype=int))


def test_validate_schema_rejects_target_nulls():
    X, y = load_raw_dataset()
    y_bad = y.copy()
    y_bad.iloc[0] = np.nan
    with pytest.raises(SchemaValidationError):
        validate_schema(X, y_bad)


def test_load_and_validate():
    X, y = load_and_validate()
    assert len(X) == len(y)
    assert len(X) > 0
