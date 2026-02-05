import os
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import Result
from typer.testing import CliRunner

from latentpool.cli import app, detect_block, main, train_model

runner: CliRunner = CliRunner()

def test_main_wrapper() -> None:
    """Ensure main() triggers the Typer app correctly."""
    with patch("latentpool.cli.app") as mock_app:
        main()
        mock_app.assert_called_once()

@patch("latentpool.cli.detect")
@patch("latentpool.cli.train")
def test_baked_entry_points(mock_train: MagicMock, mock_detect: MagicMock) -> None:
    """Verify return codes for placeholder CLI scripts without running logic."""
    detect_block()
    mock_detect.assert_called_once()

    train_model()
    mock_train.assert_called_once()

def test_get_coordinator_exit_on_missing_env(tmp_path: Path) -> None:
    """COVERS: ALCHEMY env var missing check."""
    arb = tmp_path / "arb.csv"
    arb.write_text("h")
    with patch.dict(os.environ, {"ALCHEMY_API_KEY": ""}, clear=True):
        result: Result = runner.invoke(app, ["data", "hydrate", "--arbitrage-csv", str(arb)])
        assert result.exit_code == 1
        assert "missing in .env" in result.stdout

@patch("latentpool.cli.IngestionCoordinator")
def test_hydrate_csv_loop_missing(mock_coord: MagicMock, tmp_path: Path) -> None:
    """COVERS: Label CSV missing branch in for-loop."""
    arb = tmp_path / "exists.csv"
    arb.write_text("h")
    missing = str(tmp_path / "ghost.csv")
    result: Result = runner.invoke(app, ["data", "hydrate", "--arbitrage-csv", str(arb), "--sandwich-csv", missing])
    assert result.exit_code == 1
    assert f"Label CSV missing: {missing}" in result.stdout

@patch("latentpool.cli.IngestionCoordinator")
def test_hydrate_full_success_flow(mock_coord_class: MagicMock, tmp_path: Path) -> None:
    """COVERS: Step 1 (Hydration) and Step 2 (Normal Sampling) success paths."""
    arb = tmp_path / "arb.csv"
    arb.write_text("h")
    silver = tmp_path / "edges.parquet"
    silver.write_text("data")

    mock_inst = mock_coord_class.return_value
    mock_inst.run_hydration = cast(Any, AsyncMock(return_value=10))
    mock_inst.run_negative_sampling = cast(Any, AsyncMock(return_value=5))

    env = {"ALCHEMY_API_KEY": "k", "ALCHEMY_RPC_URL": "u"}
    with patch.dict(os.environ, env):
        result = runner.invoke(app, ["data", "hydrate", "--silver-path", str(silver), "--arbitrage-csv", str(arb)])
        assert "MEV Hydration complete: 10 files" in result.stdout
        assert "Normal Sampling complete: 5 files" in result.stdout

@patch("latentpool.cli.IngestionCoordinator")
def test_data_check_report_math(mock_coord_class: MagicMock, tmp_path: Path) -> None:
    """COVERS: Set logic for found/missing hashes and ASCII report table."""
    raw_dir = tmp_path / "raw_traces"
    raw_dir.mkdir()
    (raw_dir / "0x1.json").write_text("{}")

    mock_inst = mock_coord_class.return_value
    mock_inst.config.raw_dir = str(raw_dir)
    mock_inst.extract_hashes.side_effect = [["0x1"], ["0x2"]]

    with patch.dict(os.environ, {"ALCHEMY_API_KEY": "k", "ALCHEMY_RPC_URL": "u"}):
        result = runner.invoke(app, ["data", "check"])
    assert "Arbitrage" in result.stdout
    assert "Total Files on Disk: 1" in result.stdout

@patch("latentpool.cli.DatasetAssembler")
def test_data_label_logic(mock_ass: MagicMock) -> None:
    runner.invoke(app, ["data", "label"])
    cast(MagicMock, mock_ass.return_value).run.assert_called_once()

@patch("latentpool.cli.GraphBuilder")
def test_data_process_logic(mock_builder: MagicMock) -> None:
    runner.invoke(app, ["data", "process"])
    cast(MagicMock, mock_builder.return_value).build_and_save.assert_called_once()

@pytest.mark.parametrize("cmd, func_path", [
    ("structure", "latentpool.cli.analyze_structure"),
    ("depth", "latentpool.cli.analyze_path_depth"),
    ("diversity", "latentpool.cli.analyze_node_diversity"),
    ("health", "latentpool.cli.analyze_feature_health"),
    ("temporal", "latentpool.cli.analyze_temporal_flow"),
])
def test_viz_silver_variants(cmd: str, func_path: str, tmp_path: Path) -> None:
    """COVERS: Silver visualization variants and path guards."""
    silver = tmp_path / "exists.parquet"
    silver.write_text("d")
    with patch(func_path) as mock_func:
        result = runner.invoke(app, ["viz", "silver", cmd, "--silver-path", str(silver)])
        assert result.exit_code == 0
        mock_func.assert_called_once()

@pytest.mark.parametrize("cmd, func_path, flag", [
    ("balance", "latentpool.cli.analyze_gold_balance", "--gold-path"),
    ("complexity", "latentpool.cli.analyze_label_complexity", "--gold-path"),
    ("summary", "latentpool.cli.visualize_gold_parquet", "--parquet-path"),
])
def test_viz_gold_variants(cmd: str, func_path: str, flag: str, tmp_path: Path) -> None:
    """COVERS: Gold visualization command branches."""
    gold_file = tmp_path / "gold.parquet"
    gold_file.write_text("dummy")
    with patch(func_path) as mock_func:
        result = runner.invoke(app, ["viz", "gold", cmd, flag, str(gold_file)])
        assert result.exit_code == 0
        mock_func.assert_called_once()

@patch("latentpool.cli.analyze_tensor_features")
def test_viz_gold_health_logic(mock_analyze: MagicMock, tmp_path: Path) -> None:
    """COVERS: Tensor health check with directory validation and error handling."""
    graphs_dir = tmp_path / "graphs"
    graphs_dir.mkdir()

    result = runner.invoke(app, ["viz", "gold", "health", "--graphs-dir", str(graphs_dir)])
    assert result.exit_code == 1

    (graphs_dir / "sample.pt").write_text("pt")
    result = runner.invoke(app, ["viz", "gold", "health", "--graphs-dir", str(graphs_dir)])
    assert result.exit_code == 0
    mock_analyze.assert_called_once()

    mock_analyze.side_effect = RuntimeError("Mock crash")
    result = runner.invoke(app, ["viz", "gold", "health", "--graphs-dir", str(graphs_dir)])
    assert result.exit_code == 1
    assert "Critical Error" in result.stdout

@patch("latentpool.cli.run_training")
def test_train_command_logic(mock_train: MagicMock, tmp_path: Path) -> None:
    graphs_dir = tmp_path / "tensors"
    graphs_dir.mkdir()
    (graphs_dir / "data.pt").write_text("data")

    result = runner.invoke(app, ["train", "--epochs", "2", "--graphs-dir", str(graphs_dir)])
    assert result.exit_code == 0
    mock_train.assert_called_once_with(graphs_dir=str(graphs_dir), epochs=2, batch_size=256)

def test_detect_command_logic() -> None:
    result = runner.invoke(app, ["detect", "12345"])
    assert result.exit_code == 0
    assert "Analyzing block 12345" in result.stdout

@patch("latentpool.cli.IngestionValidator")
def test_viz_health_check_full_flow(mock_val_class: MagicMock, tmp_path: Path) -> None:
    silver = tmp_path / "s.parquet"
    silver.write_text("s")
    gold = tmp_path / "g.parquet"
    gold.write_text("g")
    graphs = tmp_path / "graphs"
    graphs.mkdir()

    result = runner.invoke(app, ["viz", "health-check",
                                 "--silver-path", str(silver),
                                 "--gold-path", str(gold),
                                 "--graphs-dir", str(graphs)])

    assert result.exit_code == 0
    assert "Gold layer detected" in result.stdout
    assert "Graph tensors detected" in result.stdout


@patch("latentpool.cli.IngestionCoordinator")
def test_hydrate_missing_silver_sampling_return(mock_coord_class: MagicMock, tmp_path: Path) -> None:
    arb = tmp_path / "arb.csv"
    arb.write_text("h")

    mock_inst = mock_coord_class.return_value
    mock_inst.run_hydration = AsyncMock(return_value=10)

    env = {"ALCHEMY_API_KEY": "k", "ALCHEMY_RPC_URL": "u"}
    with patch.dict(os.environ, env):
        result = runner.invoke(app, ["data", "hydrate", "--silver-path", "void.parquet", "--arbitrage-csv", str(arb)])

        assert result.exit_code == 0
        assert "Silver data not found" in result.stdout
        assert cast(AsyncMock, mock_inst.run_negative_sampling).call_count == 0

@patch("latentpool.cli.ParquetExporter")
def test_prepare_success_messages(mock_exporter_class: MagicMock) -> None:
    result = runner.invoke(app, ["data", "prepare", "--raw-dir", "raw_test", "--processed-dir", "proc_test"])

    assert result.exit_code == 0
    assert "Starting transformation from raw_test" in result.stdout
    assert "Parquet files ready in proc_test" in result.stdout
    mock_exporter_class.return_value.process_all.assert_called_once()

def test_viz_silver_missing_file_exit(tmp_path: Path) -> None:
    result = runner.invoke(app, ["viz", "silver", "structure", "--silver-path", "non_existent.parquet"])

    assert result.exit_code == 1
    assert "Silver data missing" in result.stdout

@patch("latentpool.cli.analyze_structure")
def test_viz_structure_metrics_only_branch(mock_analyze: MagicMock, tmp_path: Path) -> None:
    silver = tmp_path / "exists.parquet"
    silver.write_text("data")

    result = runner.invoke(app, ["viz", "silver", "structure", "--no-plot", "--silver-path", str(silver)])

    assert result.exit_code == 0
    assert "Metrics calculation complete" in result.stdout
    mock_analyze.assert_called_once_with(silver_parquet=str(silver), output_dir="")

@patch("latentpool.cli.analyze_gold_balance")
def test_viz_gold_balance_missing_file_exit(mock_func: MagicMock) -> None:
    result = runner.invoke(app, ["viz", "gold", "balance", "--gold-path", "missing_gold.parquet"])

    assert result.exit_code == 1
    assert "Gold data missing" in result.stdout
    mock_func.assert_not_called()

@patch("latentpool.cli.visualize_gold_parquet")
def test_viz_gold_summary_missing_exit(mock_func: MagicMock) -> None:
    result = runner.invoke(app, ["viz", "gold", "summary", "--parquet-path", "ghost.parquet"])
    assert result.exit_code == 1
    assert "Gold parquet not found" in result.stdout

def test_viz_silver_depth_missing_exit() -> None:
    result = runner.invoke(app, ["viz", "silver", "depth", "--silver-path", "no.parquet"])
    assert result.exit_code == 1
    assert "Silver data missing" in result.stdout

def test_viz_silver_diversity_missing_exit() -> None:
    result = runner.invoke(app, ["viz", "silver", "diversity", "--silver-path", "no.parquet"])
    assert result.exit_code == 1
    assert "Silver data missing" in result.stdout

def test_viz_silver_health_missing_exit() -> None:
    result = runner.invoke(app, ["viz", "silver", "health", "--silver-path", "no.parquet"])
    assert result.exit_code == 1
    assert "Silver data missing" in result.stdout

def test_viz_silver_temporal_missing_exit() -> None:
    result = runner.invoke(app, ["viz", "silver", "temporal", "--silver-path", "no.parquet"])
    assert result.exit_code == 1
    assert "Silver data missing" in result.stdout

@patch("latentpool.cli.GraphExplorer")
def test_visualize_flow_logic(mock_explorer_class: MagicMock) -> None:
    tx = "0xdeadbeef"
    result = runner.invoke(app, ["viz", "flow", tx])

    assert result.exit_code == 0
    mock_explorer_class.assert_called_once_with("data/processed/edges.parquet")
    mock_explorer_class.return_value.generate_tx_graph.assert_called_once_with(tx)

def test_viz_health_check_missing_silver_exit(tmp_path: Path) -> None:
    missing_path = str(tmp_path / "void.parquet")
    result = runner.invoke(app, ["viz", "health-check", "--silver-path", missing_path])

    assert result.exit_code == 1
    assert "Silver data missing" in result.stdout
    assert "Run 'just prepare' first" in result.stdout

def test_train_no_tensors_exit(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty_pt"
    empty_dir.mkdir()

    result = runner.invoke(app, ["train", "--graphs-dir", str(empty_dir)])

    assert result.exit_code == 1
    assert "No .pt tensors found" in result.stdout
    assert "Did you run the processing step yet?" in result.stdout

@patch("latentpool.cli.analyze_label_complexity")
def test_viz_gold_complexity_missing_exit(mock_analyze: MagicMock) -> None:
    result = runner.invoke(app, ["viz", "gold", "complexity", "--gold-path", "non_existent_gold.parquet"])

    assert result.exit_code == 1
    assert "Gold data missing" in result.stdout
    mock_analyze.assert_not_called()
