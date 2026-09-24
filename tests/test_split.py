import json

import pytest

from tabtrace.data.ingest import load_raw_dataset
from tabtrace.data.split import (
    LeakageError,
    apply_split,
    get_or_create_split,
    make_split_indices,
    validate_no_leakage,
)


@pytest.fixture(scope="module")
def raw_data():
    return load_raw_dataset()


def test_split_is_deterministic_across_calls(raw_data):
    X, y = raw_data
    first = make_split_indices(len(X), y, seed=42)
    second = make_split_indices(len(X), y, seed=42)
    assert first["train"] == second["train"]
    assert first["val"] == second["val"]
    assert first["test"] == second["test"]


def test_different_seeds_produce_different_splits(raw_data):
    X, y = raw_data
    a = make_split_indices(len(X), y, seed=1)
    b = make_split_indices(len(X), y, seed=2)
    assert a["train"] != b["train"]


def test_partitions_are_disjoint_and_complete(raw_data):
    X, y = raw_data
    indices = make_split_indices(len(X), y, seed=42)
    validate_no_leakage(indices)  # should not raise
    all_idx = set(indices["train"]) | set(indices["val"]) | set(indices["test"])
    assert len(all_idx) == len(X)


def test_validate_no_leakage_catches_overlap():
    bad_indices = {"train": [0, 1, 2], "val": [2, 3], "test": [4, 5]}
    with pytest.raises(LeakageError):
        validate_no_leakage(bad_indices)


def test_get_or_create_split_persists_and_reloads(raw_data, tmp_path):
    X, y = raw_data
    path = tmp_path / "split.json"

    first = get_or_create_split(X, y, path, seed=42)
    assert path.exists()

    # Simulate a second pipeline run: must reload the exact same indices,
    # not regenerate them, even if called with a different seed argument.
    second = get_or_create_split(X, y, path, seed=999)
    assert first["train"] == second["train"]
    assert first["test"] == second["test"]

    with open(path) as f:
        on_disk = json.load(f)
    assert on_disk["train"] == first["train"]


def test_apply_split_slices_match_indices(raw_data):
    X, y = raw_data
    indices = make_split_indices(len(X), y, seed=42)
    X_train, y_train, X_val, y_val, X_test, y_test = apply_split(X, y, indices)
    assert len(X_train) == len(indices["train"])
    assert len(X_val) == len(indices["val"])
    assert len(X_test) == len(indices["test"])
