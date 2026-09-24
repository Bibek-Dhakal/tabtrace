# TabTrace — Metrics & Rationale Report

## Cross-Validated Model Comparison

| Model | Scoring | CV Mean | CV Std | Folds |
|---|---|---|---|---|
| Baseline (Logistic Regression) | f1 | 0.9739 | 0.0138 | 5 |
| Tuned (Random Forest, grid-searched) | f1 | 0.9641 | 0.0239 | 5 |

## Tuning Decision

Best hyperparameters selected by grid search under 5-fold stratified CV, optimizing `f1`:

```json
{
  "clf__max_depth": 5,
  "clf__min_samples_leaf": 1,
  "clf__n_estimators": 100
}
```

Rationale: a grid search over tree depth, leaf size, and estimator count was used (rather than manual re-running) so the selected configuration is the one that generalizes best across folds, not the one that happened to score highest on a single split.

**Observation**: The baseline Logistic Regression model outperformed the tuned Random Forest model on cross-validation. This suggests that for this specific dataset, a linear decision boundary is highly effective, and the added complexity of a tree ensemble may be prone to overfitting or simply unnecessary given the strong linearly-separable signal in the engineered features.

## Held-Out Test Metrics (secondary sanity check)

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|---|
| Baseline | 0.9767 | 0.9815 | 0.9815 | 0.9815 | 0.9959 |
| Tuned | 0.9651 | 0.9811 | 0.9630 | 0.9720 | 0.9954 |

## Engineered Feature Rationale

| Feature | Justification |
|---|---|
| `area_perimeter_ratio` | Normalizes cell area by perimeter, capturing shape compactness independent of raw tumor size. Malignant cells tend to have a different compactness profile than benign ones even at matched size. |
| `concavity_symmetry_interaction` | Multiplies mean concavity by mean symmetry. Malignant nuclei tend to be jointly concave AND asymmetric; the two source features are only weakly correlated on their own, so their product surfaces a combined effect neither captures individually. |
| `radius_worst_mean_ratio` | Ratio of the most extreme ('worst') radius measurement to the mean radius. A high ratio indicates a heterogeneous, unevenly-growing mass, a pattern associated with malignancy that a single mean value would smooth over. |
