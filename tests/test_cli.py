from unittest.mock import patch

from tabtrace.cli import main


@patch("tabtrace.cli.run_pipeline")
def test_cli_main(mock_run):
    main()
    mock_run.assert_called_once()
