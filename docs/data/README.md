# Data

## Source

TabTrace uses the **Breast Cancer Wisconsin (Diagnostic)** dataset, bundled directly with scikit-learn via
`sklearn.datasets.load_breast_cancer()`. No download or external storage is required — the dataset is small
(569 rows, 30 numeric features, binary target) and ships with the library.

- Target: `0` = malignant, `1` = benign (scikit-learn's convention).
- Features: 30 real-valued measurements of cell nuclei (radius, texture, perimeter, area, concavity, symmetry, etc.),
  each reported as a mean, standard error, and "worst" (largest) value.

## Preprocessing

No missing-value imputation is needed — the source dataset is complete, and `tabtrace.data.ingest.validate_schema`
asserts this on every run (raising `SchemaValidationError` if it ever isn't). Numeric features are standardized
(zero mean, unit variance) inside each model pipeline via `StandardScaler`, fit only on the training partition to
avoid leakage.

## Engineered Features

See [`docs/architecture/README.md`](../architecture/README.md) and the feature rationale table generated in
`outputs/report.md` for the three engineered features and their justifications. The single source of truth for
justifications is the code itself: `src/tabtrace/features/engineer.py`.

## Splits

Train/validation/test indices are computed once (stratified by target, seed=42) and persisted to
`outputs/split_indices.json`. Re-running the pipeline reuses the saved file rather than re-randomizing — see
[`docs/testing/README.md`](../testing/README.md) for how this is verified.
