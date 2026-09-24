# Notebooks

TabTrace deliberately keeps all pipeline logic out of notebooks (see the project's Target requirements: "expose all preprocessing/feature logic as testable functions, not notebook-only code"). This folder holds **exploratory analysis only** — nothing here is imported by `src/tabtrace/`.

## Setup

To run the exploratory notebooks, install the project with the `notebooks` extra (which includes Matplotlib and Jupyter):

```bash
pip install -e ".[notebooks]"
```

## Files

- `eda_exploration.ipynb` — A standard Jupyter notebook walking through: loading the raw dataset, checking class balance, and visualizing why each of the three engineered features in `src/tabtrace/features/engineer.py` separates the two classes.
