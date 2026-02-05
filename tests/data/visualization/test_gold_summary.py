from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from latentpool.data.visualization.gold_summary import visualize_gold_parquet

EXPECTED_PLOT_COUNT = 1

@pytest.fixture
def mock_gold_parquet(tmp_path: Path) -> str:
    """a gold parquet file with edge-list structure."""
    data = {
        "tx_hash": ["0x1", "0x1", "0x2", "0x3", "0x3", "0x3"],
        "label": [0, 0, 1, 2, 2, 2],
        "split": ["train", "train", "test", "train", "train", "train"],
        "from": ["a", "b", "c", "d", "e", "f"],
        "to": ["b", "c", "d", "e", "f", "g"]
    }
    df = pd.DataFrame(data)
    parquet_path = tmp_path / "summary_gold.parquet"
    df.to_parquet(parquet_path) # type: ignore
    return str(parquet_path)

@patch("latentpool.data.visualization.gold_summary.plt")
@patch("latentpool.data.visualization.gold_summary.sns")
def test_visualize_gold_parquet_execution(
    mock_sns: MagicMock,
    mock_plt: MagicMock,
    mock_gold_parquet: str,
    tmp_path: Path
):
    mock_axes = [MagicMock(), MagicMock()]
    mock_plt.subplots.return_value = (MagicMock(), mock_axes)

    output_dir = tmp_path / "summary_viz"
    visualize_gold_parquet(mock_gold_parquet, output_dir=str(output_dir))

    assert output_dir.exists()
    assert mock_plt.savefig.call_count == EXPECTED_PLOT_COUNT

    assert mock_sns.boxplot.called
    assert mock_sns.countplot.called

def test_visualize_gold_parquet_not_found():
    with pytest.raises(FileNotFoundError):
        visualize_gold_parquet("ghost.parquet")

@patch("latentpool.data.visualization.gold_summary.pd.read_parquet")
@patch("latentpool.data.visualization.gold_summary.plt")
def test_visualize_gold_parquet_empty_df(
    mock_plt: MagicMock,
    mock_read: MagicMock,
    tmp_path: Path
):
    """handles empty DataFrames without crashing."""
    empty_df: Any = pd.DataFrame(columns=['tx_hash', 'label', 'split'])
    mock_read.return_value = empty_df

    mock_plt.subplots.return_value = (MagicMock(), [MagicMock(), MagicMock()])

    output_dir = tmp_path / "empty_summary"
    visualize_gold_parquet("mock.parquet", output_dir=str(output_dir))

    assert output_dir.exists()
