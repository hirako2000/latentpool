from pathlib import Path
from typing import Any, List
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from latentpool.data.visualization.gold.complexity import analyze_label_complexity

EXPECTED_PLOT_COUNT = 1

@pytest.fixture
def mock_complexity_data(tmp_path: Path) -> str:
    """Creates dummy edge-list data."""
    tx_hashes: List[str] = []
    labels: List[int] = []

    for i in range(10):
        tx_id = f"0x{i}"
        edge_count = (i + 1) * 2
        tx_hashes.extend([tx_id] * edge_count)
        labels.extend([i % 3] * edge_count)

    num_entries = len(tx_hashes)

    df = pd.DataFrame({
        "tx_hash": tx_hashes,
        "label": labels,
        "from": ["addr_a"] * num_entries,
        "to": ["addr_b"] * num_entries
    })

    parquet_path = tmp_path / "test_complexity.parquet"
    df.to_parquet(parquet_path) # type: ignore
    return str(parquet_path)

@patch("latentpool.data.visualization.gold.complexity.plt")
@patch("latentpool.data.visualization.gold.complexity.sns")
def test_analyze_label_complexity_execution(
    mock_sns: MagicMock,
    mock_plt: MagicMock,
    mock_complexity_data: str,
    tmp_path: Path
):
    output_dir = tmp_path / "viz_complexity"
    analyze_label_complexity(mock_complexity_data, output_dir=str(output_dir))

    assert output_dir.exists()
    assert mock_plt.savefig.call_count == EXPECTED_PLOT_COUNT
    mock_plt.yscale.assert_called_with("log")

def test_analyze_label_complexity_file_not_found():
    with pytest.raises(FileNotFoundError):
        analyze_label_complexity("missing.parquet")

@patch("latentpool.data.visualization.gold.complexity.pd.read_parquet")
@patch("latentpool.data.visualization.gold.complexity.plt")
def test_analyze_label_complexity_empty_df(
    mock_plt: MagicMock,
    mock_read: MagicMock,
    tmp_path: Path
):
    empty_df: Any = pd.DataFrame(columns=["tx_hash", "label", "from", "to"])
    mock_read.return_value = empty_df

    output_dir = tmp_path / "empty_complexity"
    analyze_label_complexity("mock.parquet", output_dir=str(output_dir))

    assert mock_plt.savefig.call_count == 1
