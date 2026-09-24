#!/usr/bin/env python
"""Run the full TabTrace pipeline end to end and print a summary.

Usage:
    python scripts/run_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tabtrace.pipeline import run_pipeline  # noqa: E402


def main() -> None:
    results = run_pipeline()
    print("TabTrace pipeline complete.\n")
    print(
        f"Baseline CV {results['baseline_cv']['scoring']}: "
        f"{results['baseline_cv']['mean']:.4f} (+/- {results['baseline_cv']['std']:.4f})"
    )
    print(
        f"Tuned CV {results['tuned_cv']['scoring']}:    "
        f"{results['tuned_cv']['mean']:.4f} (+/- {results['tuned_cv']['std']:.4f})"
    )
    print(f"Best tuned params: {results['tuned_search_summary']['best_params']}")
    print("\nFull report written to outputs/report.md")


if __name__ == "__main__":
    main()
