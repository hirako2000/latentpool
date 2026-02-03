import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from latentpool.cli import app, detect_block, main, train_model

runner = CliRunner()

def test_main_wrapper() -> None:
    with pytest.raises(SystemExit):
        main()

def test_baked_in_entry_points() -> None:
    assert detect_block() == 0
    assert train_model() == 0

def test_detect_command_defaults() -> None:
    result = runner.invoke(app, ["detect"])
    assert result.exit_code == 0
    assert "Analyzing block 19000000 with SAGE" in result.stdout

def test_train_command() -> None:
    result = runner.invoke(app, ["train", "--epochs", "5"])
    assert result.exit_code == 0
    assert "Training SAGE for 5 epochs" in result.stdout

@patch("latentpool.cli.IngestionCoordinator")
@patch.dict(os.environ, {"ALCHEMY_API_KEY": "", "ALCHEMY_RPC_URL": ""}, clear=True)
def test_get_coordinator_error(mock_coord: MagicMock) -> None:
    result = runner.invoke(app, ["data", "hydrate"])
    assert result.exit_code == 1
    # assert "missing in .env" in result.stdout

@patch("latentpool.cli.IngestionCoordinator")
def test_data_hydrate(mock_coord_class: MagicMock, tmp_path: Path) -> None:
    arb_csv = tmp_path / "arb.csv"
    arb_csv.write_text("hash\n0x123")
    sand_csv = tmp_path / "sand.csv"
    sand_csv.write_text("hash\n0x456")

    silver_file = tmp_path / "edges.parquet"
    silver_file.write_text("dummy data")

    mock_instance = mock_coord_class.return_value
    mock_instance.run_hydration = AsyncMock(return_value=10)
    mock_instance.run_negative_sampling = AsyncMock(return_value=10)

    env = {"ALCHEMY_API_KEY": "test", "ALCHEMY_RPC_URL": "http://test/"}

    with patch.dict(os.environ, env):
        result = runner.invoke(app, [
            "data", "hydrate",
            "--arbitrage-csv", str(arb_csv),
            "--sandwich-csv", str(sand_csv),
            "--silver-path", str(silver_file)
        ])

        assert result.exit_code == 0
        assert "Step 1: Hydrating MEV" in result.stdout
        assert "MEV Hydration complete: 10 files" in result.stdout
        assert "Step 2: Sampling 'Normal'" in result.stdout
        assert "Normal Sampling complete: 10 files" in result.stdout

@patch("latentpool.cli.IngestionCoordinator")
def test_data_hydrate_csv_missing_exit(mock_coord: MagicMock, tmp_path: Path) -> None:
    missing_csv = str(tmp_path / "ghost.csv")
    result = runner.invoke(app, ["data", "hydrate", "--arbitrage-csv", missing_csv])
    assert result.exit_code == 1
    assert "Label CSV missing" in result.stdout

@patch("latentpool.cli.IngestionCoordinator")
def test_data_hydrate_step2_missing_silver_coverage(mock_coord_class: MagicMock, tmp_path: Path) -> None:
    arb_csv = tmp_path / "arb.csv"
    arb_csv.write_text("hash\n0x123")
    sand_csv = tmp_path / "sand.csv"
    sand_csv.write_text("hash\n0x456")

    missing_silver = str(tmp_path / "void_edges.parquet")

    mock_instance = mock_coord_class.return_value
    mock_instance.run_hydration = AsyncMock(return_value=5)

    env = {"ALCHEMY_API_KEY": "test", "ALCHEMY_RPC_URL": "http://test/"}

    with patch.dict(os.environ, env):
        result = runner.invoke(app, [
            "data", "hydrate",
            "--arbitrage-csv", str(arb_csv),
            "--sandwich-csv", str(sand_csv),
            "--silver-path", missing_silver
        ])

        assert result.exit_code == 0
        assert "MEV Hydration complete: 5 files" in result.stdout
        assert "Silver data not found" in result.stdout
        mock_instance.run_negative_sampling.assert_not_called()


@patch("latentpool.cli.IngestionCoordinator")
def test_data_check_command(mock_coord_class: MagicMock, tmp_path: Path) -> None:
    mock_instance = mock_coord_class.return_value
    mock_instance.config = MagicMock()
    mock_instance.config.raw_dir = str(tmp_path)
    mock_instance.extract_hashes.return_value = []

    env = {"ALCHEMY_API_KEY": "test", "ALCHEMY_RPC_URL": "http://test/"}
    with patch.dict(os.environ, env):
        result = runner.invoke(app, ["data", "check"])
        assert result.exit_code == 0
        assert "INGESTION HEALTH REPORT" in result.stdout

@patch("latentpool.cli.ParquetExporter")
def test_data_prepare_logic(mock_exp: MagicMock) -> None:
    result = runner.invoke(app, ["data", "prepare"])
    assert result.exit_code == 0
    mock_exp.return_value.process_all.assert_called_once()

@patch("latentpool.cli.Labeler")
def test_data_label_logic(mock_lab: MagicMock) -> None:
    result = runner.invoke(app, ["data", "label"])
    assert result.exit_code == 0
    mock_lab.return_value.run.assert_called_once()

@patch("latentpool.cli.GraphBuilder")
def test_data_process_logic(mock_builder: MagicMock) -> None:
    result = runner.invoke(app, ["data", "process"])
    assert result.exit_code == 0
    mock_builder.return_value.build_and_save.assert_called_once()

@patch("latentpool.cli.GraphExplorer")
def test_viz_flow_command(mock_explorer: MagicMock) -> None:
    result = runner.invoke(app, ["viz", "flow", "0xabc"])
    assert result.exit_code == 0
    mock_explorer.return_value.generate_tx_graph.assert_called_with("0xabc")

def test_viz_health_check_missing_silver(tmp_path: Path) -> None:
    """This branch uses 'raise typer.Exit(1)', so we expect exit_code 1."""
    missing_path = str(tmp_path / "missing.parquet")
    result = runner.invoke(app, ["viz", "health-check", "--silver-path", missing_path])
    assert result.exit_code == 1
    assert "Silver data missing" in result.stdout

@patch("latentpool.cli.IngestionValidator")
def test_viz_health_check_full_existence(mock_val: MagicMock, tmp_path: Path) -> None:
    silver = tmp_path / "s.parquet"
    silver.write_text("s")

    gold = tmp_path / "g.parquet"
    gold.write_text("g")

    graphs = tmp_path / "graphs"
    graphs.mkdir()

    result = runner.invoke(app, [
        "viz", "health-check",
        "--silver-path", str(silver),
        "--gold-path", str(gold),
        "--graphs-dir", str(graphs)
    ])
    assert result.exit_code == 0
    assert "Gold layer detected" in result.stdout
    assert "Graph tensors detected" in result.stdout


@pytest.mark.parametrize("cmd", ["structure", "depth", "diversity", "health", "temporal"])
def test_viz_silver_commands_missing_path(cmd: str, tmp_path: Path) -> None:
    missing_path = str(tmp_path / "ghost.parquet")
    result = runner.invoke(app, ["viz", "silver", cmd, "--silver-path", missing_path])
    assert result.exit_code == 1
    assert "Silver data missing" in result.stdout

@patch("latentpool.cli.analyze_structure")
def test_viz_silver_structure_logic(mock_fn: MagicMock, tmp_path: Path) -> None:
    silver = tmp_path / "exists.parquet"
    silver.write_text("data")
    result = runner.invoke(app, ["viz", "silver", "structure", "--silver-path", str(silver)])
    assert result.exit_code == 0
    mock_fn.assert_called_once()

@patch("latentpool.cli.analyze_node_diversity")
def test_viz_silver_diversity_logic(mock_fn: MagicMock, tmp_path: Path) -> None:
    silver = tmp_path / "exists.parquet"
    silver.write_text("data")
    result = runner.invoke(app, ["viz", "silver", "diversity", "--silver-path", str(silver)])
    assert result.exit_code == 0
    mock_fn.assert_called_once()

@patch("latentpool.cli.analyze_path_depth")
def test_viz_silver_depth_logic(mock_fn: MagicMock, tmp_path: Path) -> None:
    silver = tmp_path / "exists.parquet"
    silver.write_text("data")
    result = runner.invoke(app, ["viz", "silver", "depth", "--silver-path", str(silver)])
    assert result.exit_code == 0
    mock_fn.assert_called_once()

@patch("latentpool.cli.analyze_temporal_flow")
def test_viz_silver_temporal_logic(mock_fn: MagicMock, tmp_path: Path) -> None:
    silver = tmp_path / "exists.parquet"
    silver.write_text("data")
    result = runner.invoke(app, ["viz", "silver", "temporal", "--silver-path", str(silver)])
    assert result.exit_code == 0
    mock_fn.assert_called_once()

@patch("latentpool.cli.analyze_feature_health")
def test_viz_silver_health_logic(mock_fn: MagicMock, tmp_path: Path) -> None:
    silver = tmp_path / "exists.parquet"
    silver.write_text("data")
    result = runner.invoke(app, ["viz", "silver", "health", "--silver-path", str(silver)])
    assert result.exit_code == 0
    mock_fn.assert_called_once()

@patch("latentpool.cli.analyze_structure")
def test_viz_silver_structure_no_plot_coverage(mock_analyze: MagicMock, tmp_path: Path) -> None:
    """
    Covers the 'Metrics calculation complete' branch when --no-plot is True.
    """
    silver = tmp_path / "exists.parquet"
    silver.write_text("data")

    result = runner.invoke(app, [
        "viz", "silver", "structure",
        "--silver-path", str(silver),
        "--no-plot"
    ])

    assert result.exit_code == 0
    assert "Metrics calculation complete." in result.stdout
    mock_analyze.assert_called_once_with(silver_parquet=str(silver), output_dir="")
