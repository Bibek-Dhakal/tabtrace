"""Command-line entry point. Delegates to the pipeline runner script."""

from tabtrace.pipeline import run_pipeline


def main() -> None:
    run_pipeline()


if __name__ == "__main__":
    main()
