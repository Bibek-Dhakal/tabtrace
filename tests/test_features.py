import pandas as pd
import pytest

from tabtrace.data.ingest import load_raw_dataset
from tabtrace.features.engineer import (
    apply_features,
    feature_rationale_table,
    list_features,
    register_feature,
)


def test_every_registered_feature_has_a_justification():
    for feature in list_features():
        assert feature.justification.strip() != ""


def test_register_feature_rejects_empty_justification():
    with pytest.raises(ValueError):

        @register_feature(name="bad_feature", justification="   ")
        def _bad(X: pd.DataFrame) -> pd.Series:
            return X.iloc[:, 0]


def test_apply_features_adds_expected_columns():
    X, _ = load_raw_dataset()
    X_out = apply_features(X)
    expected = {f.name for f in list_features()}
    assert expected.issubset(set(X_out.columns))
    assert len(X_out) == len(X)


def test_area_perimeter_ratio_is_correct_on_toy_frame():
    toy = pd.DataFrame({"mean area": [10.0, 20.0], "mean perimeter": [2.0, 5.0]})
    X_out = apply_features(
        toy.reindex(
            columns=[
                "mean area",
                "mean perimeter",
                "mean concavity",
                "mean symmetry",
                "worst radius",
                "mean radius",
            ],
            fill_value=1.0,
        )
    )
    assert X_out["area_perimeter_ratio"].tolist() == pytest.approx([5.0, 4.0])


def test_radius_worst_mean_ratio_is_correct_on_toy_frame():
    toy = pd.DataFrame(
        {
            "mean area": [1.0, 1.0],
            "mean perimeter": [1.0, 1.0],
            "mean concavity": [1.0, 1.0],
            "mean symmetry": [1.0, 1.0],
            "worst radius": [30.0, 10.0],
            "mean radius": [10.0, 10.0],
        }
    )
    X_out = apply_features(toy)
    assert X_out["radius_worst_mean_ratio"].tolist() == pytest.approx([3.0, 1.0])


def test_feature_rationale_table_has_one_row_per_feature():
    table = feature_rationale_table()
    assert set(table["feature"]) == {f.name for f in list_features()}
    assert table["justification"].apply(lambda s: len(s) > 0).all()
