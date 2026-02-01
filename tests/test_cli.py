import pytest
from typer.testing import CliRunner

from latentpool.cli import app, detect_block, main, train_model

runner = CliRunner()

def test_detect_command_defaults():
    """Verify detect command runs with default block."""
    result = runner.invoke(app, ["detect"])
    assert result.exit_code == 0
    assert "Analyzing block 19000000 with SAGE" in result.stdout

def test_detect_command_custom_args():
    """Verify detect command accepts custom block and model."""
    result = runner.invoke(app, ["detect", "123456", "--model", "GAT"])
    assert result.exit_code == 0
    assert "Analyzing block 123456 with GAT" in result.stdout

def test_train_command():
    """Verify train command options."""
    result = runner.invoke(app, ["train", "--model", "GAT", "--epochs", "10"])

    if result.exit_code != 0:
        print(f"Error output: {result.stdout}")

    assert result.exit_code == 0
    assert "Training GAT for 10 epochs" in result.stdout

def test_cli_help():
    """Ensure the help menu is descriptive."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Detect MEV in a block (default: latest)." in result.stdout


def test_baked_in_entry_points():
    """Verify the helper functions and secure 100% coverage."""
    assert detect_block() == 0
    assert train_model() == 0

def test_main_wrapper():
    """Execute main() to ensure the wrapper is covered."""
    # app() eventually calls sys.exit()
    with pytest.raises(SystemExit):
        main()
