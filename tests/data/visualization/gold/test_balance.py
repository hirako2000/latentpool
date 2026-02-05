from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from latentpool.data.visualization.gold.balance import analyze_gold_balance

EXPECTED_PLOT_COUNT = 2

@pytest.fixture
def mock_gold_data(tmp_path: Path) -> str:
    data = {
        "tx_hash": [f"0x{i}" for i in range(100)],
        "label": [0, 1, 2] * 33 + [0],
        "split": ["train"] * 70 + ["test"] * 30,
        "block_number": np.linspace(18000000, 18001000, 100).astype(int),
        "token": ["WETH"] * 100,
        "value": [1.0] * 100
    }
    df = pd.DataFrame(data)
    parquet_path = tmp_path / "test_gold.parquet"
    df.to_parquet(parquet_path)  # type: ignore
    return str(parquet_path)

@patch("latentpool.data.visualization.gold.balance.plt")
@patch("latentpool.data.visualization.gold.balance.sns")
def test_analyze_gold_balance_execution(
    mock_sns: MagicMock,
    mock_plt: MagicMock,
    mock_gold_data: str,
    tmp_path: Path
):
    """function should run, print report, and save images."""
    output_dir = tmp_path / "viz"

    analyze_gold_balance(mock_gold_data, output_dir=str(output_dir))

    assert output_dir.exists()

    assert mock_plt.savefig.call_count == EXPECTED_PLOT_COUNT
    assert mock_plt.close.call_count == EXPECTED_PLOT_COUNT

def test_analyze_gold_balance_invalid_path():
    with pytest.raises(FileNotFoundError):
        analyze_gold_balance("non_existent.parquet")

@patch("latentpool.data.visualization.gold.balance.plt")
@patch("latentpool.data.visualization.gold.balance.sns")
@patch("latentpool.data.visualization.gold.balance.pd.read_parquet")
def test_analyze_gold_balance_empty_df(
    mock_read: MagicMock,
    mock_sns: MagicMock,
    mock_plt: MagicMock,
    tmp_path: Path
):
    """empty dataset."""
    empty_df: Any = pd.DataFrame(columns=["tx_hash", "label", "split", "block_number"])
    mock_read.return_value = empty_df

    mock_plt.subplots.return_value = (MagicMock(), MagicMock())

    output_dir = tmp_path / "empty_viz"

    analyze_gold_balance("mock.parquet", output_dir=str(output_dir))

    assert output_dir.exists()
    call_count = 2
    assert mock_plt.savefig.call_count == call_count
