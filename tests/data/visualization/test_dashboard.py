import logging
from pathlib import Path
from typing import Any, Tuple
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from latentpool.data.visualization.dashboard import IngestionValidator

HASH_A = "0xAAA"
HASH_B = "0xBBB"
ADDR_WETH = "0xC02aaA"
ADDR_USDC = "0xA0b869"

LABEL_NORMAL = 0
LABEL_ARB = 1
LABEL_SAND = 2

VAL_DEFAULT = 1.0
EXPECTED_SILVER_EDGES = 3
EXPECTED_UNIQUE_ADDR = 2
TRANSFORMATION_PLOT_COUNT = 2

SILVER_FILE = "silver.parquet"
GOLD_FILE = "gold.parquet"
VIS_DIR_TRANS = "visualizations/transformation"
VIS_DIR_LABEL = "visualizations/labeling"
VIS_DIR_GEOM = "visualizations/geometry"

FILE_EDGE_DIST = "edge_distribution.png"
FILE_ADDR_USAGE = "token_usage.png"
FILE_CLASS_DIST = "class_balance.png"
FILE_GEOM_COMPLEX = "tensor_complexity.png"

@pytest.fixture
def mock_data_paths(tmp_path: Path) -> Tuple[str, str]:
    """Creates silver and gold parquets with a mismatch to test Unlabeled logic."""
    silver_path = tmp_path / SILVER_FILE
    gold_path = tmp_path / GOLD_FILE

    # HASH_A and HASH_B are known. HASH_C is "Unlabeled".
    silver_df = pd.DataFrame({
        "tx_hash": [HASH_A, HASH_B, "0xCCC"],
        "token": [ADDR_WETH, ADDR_USDC, ADDR_WETH],
        "value": [VAL_DEFAULT, VAL_DEFAULT, VAL_DEFAULT]
    })
    silver_df.to_parquet(silver_path)  # type: ignore[reportUnknownMemberType]

    # Gold only knows about A and B
    gold_df = pd.DataFrame({
        "tx_hash": [HASH_A, HASH_B],
        "label": [LABEL_ARB, LABEL_NORMAL]
    })
    gold_df.to_parquet(gold_path)  # type: ignore[reportUnknownMemberType]

    return str(silver_path), str(gold_path)

@patch("matplotlib.pyplot.savefig")
def test_print_aggregates_unlabeled_warning(mock_save: MagicMock, mock_data_paths: Tuple[str, str], capsys: Any) -> None:
    """Explicitly tests the '⚠️ Unlabeled' warning when Silver and Gold counts mismatch."""
    silver_p, gold_p = mock_data_paths
    validator = IngestionValidator(silver_p)

    validator.print_aggregates(gold_path=gold_p)
    out = capsys.readouterr().out

    # We have 3 unique txs in Silver (A, B, C) and 2 in Gold (A, B)
    assert "⚠️  Unlabeled:" in out
    assert "1" in out
    assert "(Pending Labeler run)" in out

@patch("matplotlib.pyplot.savefig")
def test_print_aggregates(mock_save: MagicMock, mock_data_paths: Tuple[str, str], capsys: Any) -> None:
    """Verifies the aggregate summary text output for both silver and gold."""
    silver_p, gold_p = mock_data_paths
    validator = IngestionValidator(silver_p)

    validator.print_aggregates()
    out_no_gold = capsys.readouterr().out
    assert "Total Transfers (Edges): 3" in out_no_gold
    assert "Gold layer not found" in out_no_gold

    validator.print_aggregates(gold_path=gold_p)
    out_gold = capsys.readouterr().out

    assert "LABEL DISTRIBUTION" in out_gold
    lines = [line.strip() for line in out_gold.split("\n")]
    assert any("Arbitrage:" in line and "1" in line for line in lines)
    assert any("Normal:" in line and "1" in line for line in lines)

@patch("matplotlib.pyplot.savefig")
def test_generate_transformation_plots(mock_save: MagicMock, mock_data_paths: Tuple[str, str]) -> None:
    silver_p, _ = mock_data_paths
    validator = IngestionValidator(silver_p)

    validator.generate_transformation_plots()

    assert Path(VIS_DIR_TRANS).exists()
    assert mock_save.call_count == TRANSFORMATION_PLOT_COUNT

    args_list = [str(call.args[0]) for call in mock_save.call_args_list]
    assert any(FILE_EDGE_DIST in arg for arg in args_list)
    assert any(FILE_ADDR_USAGE in arg for arg in args_list)

@patch("matplotlib.pyplot.savefig")
def test_generate_gold_plots(mock_save: MagicMock, mock_data_paths: Tuple[str, str]) -> None:
    """labeling distribution plots"""
    silver_p, gold_p = mock_data_paths
    validator = IngestionValidator(silver_p)

    validator.generate_gold_plots(gold_p)

    assert Path(VIS_DIR_LABEL).exists()
    mock_save.assert_called()
    assert FILE_CLASS_DIST in str(mock_save.call_args[0][0])

@patch("matplotlib.pyplot.savefig")
def test_generate_tensor_stats(mock_save: MagicMock, tmp_path: Path) -> None:
    """geometric complexity plotting with mocked .pt files."""
    graphs_dir = tmp_path / "graphs"
    graphs_dir.mkdir()

    mock_data = MagicMock()
    mock_data.x.size.return_value = 10
    mock_data.edge_index.size.return_value = 20

    (graphs_dir / "graph_0.pt").write_text("dummy")

    with patch("torch.load", return_value=mock_data):
        silver_empty = tmp_path / "empty.parquet"
        df_empty: Any = pd.DataFrame({"tx_hash": [], "token": []})
        df_empty.to_parquet(silver_empty)

        validator = IngestionValidator(str(silver_empty))

        validator.generate_tensor_stats(str(graphs_dir))
        assert Path(VIS_DIR_GEOM).exists()
        assert FILE_GEOM_COMPLEX in str(mock_save.call_args[0][0])

@patch("matplotlib.pyplot.savefig")
def test_generate_tensor_stats_empty_or_error(mock_save: MagicMock, tmp_path: Path, caplog: Any) -> None:
    dummy_p = tmp_path / SILVER_FILE
    df_dummy: Any = pd.DataFrame({"tx_hash": [], "token": []})
    df_dummy.to_parquet(dummy_p)

    validator = IngestionValidator(str(dummy_p))

    validator.generate_tensor_stats(str(tmp_path / "non_existent"))
    mock_save.assert_not_called()

    error_dir = tmp_path / "error_graphs"
    error_dir.mkdir()
    (error_dir / "bad.pt").write_text("not a tensor")

    with patch("torch.load", side_effect=RuntimeError("Corrupt file")):
        with caplog.at_level(logging.ERROR):
            validator.generate_tensor_stats(str(error_dir))
            assert "Failed to load graph" in caplog.text
