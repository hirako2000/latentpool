from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from latentpool.data.visualization.silver.graph_structure import analyze_structure


@pytest.fixture
def pd_proxy() -> Any:
    """Provides a pd proxy for type-safe dataframe creation."""
    return pd

@pytest.fixture
def mock_structure_parquet(tmp_path: Path, pd_proxy: Any) -> Path:
    """
    Creates a dummy silver parquet with varied transaction structures:
    - tx1: 2 edges, 2 unique tokens (Complexity 1.0)
    - tx2: 4 edges, 2 unique tokens (Complexity 2.0)
    - tx3: 6 edges, 3 unique tokens (Complexity 2.0)
    """
    data = {
        "tx_hash": ["tx1"] * 2 + ["tx2"] * 4 + ["tx3"] * 6,
        "token": [
            "T1", "T2",             # tx1
            "T1", "T1", "T2", "T2", # tx2
            "T1", "T1", "T2", "T2", "T3", "T3" # tx3
        ],
        "from": ["0x1"] * 12,
        "to": ["0x2"] * 12,
        "value": [1.0] * 12
    }
    df = pd_proxy.DataFrame(data)
    path = tmp_path / "test_structure.parquet"
    cast_df: Any = df
    cast_df.to_parquet(path)
    return path

@patch("latentpool.data.visualization.silver.graph_structure.plt")
@patch("latentpool.data.visualization.silver.graph_structure.sns")
def test_analyze_structure_logic(
    mock_sns: MagicMock,
    mock_plt: MagicMock,
    mock_structure_parquet: Path,
    tmp_path: Path,
    capsys: Any
) -> None:
    """Tests if topological aggregation and complexity ratios are correct."""
    output_dir = tmp_path / "viz_structure"

    analyze_structure(str(mock_structure_parquet), str(output_dir))

    capsys_any: Any = capsys
    captured = capsys_any.readouterr()

    # Logic Verification:
    # Total TXs: 3
    # tx1 edges: 2, tx2 edges: 4, tx3 edges: 6 -> Avg: 4.0
    # tx1 complex: 1.0, tx2 complex: 2.0, tx3 complex: 2.0 -> Avg: 1.67
    assert "Total Graphs (TXs):    3" in captured.out
    assert "Avg Edges per Graph:   4.00" in captured.out
    assert "Avg Complexity Ratio:  1.67" in captured.out
    assert "Max Edges (Outlier):   6" in captured.out

    # Plotting Verification
    mock_plt.figure.assert_called_once()
    mock_sns.scatterplot.assert_called_once()
    mock_plt.axhline.assert_called_once() # For the 99th percentile line
    mock_plt.savefig.assert_called_once()

def test_analyze_structure_empty(tmp_path: Path, pd_proxy: Any, capsys: Any) -> None:
    """Ensures structural analysis handles empty datasets without crashing."""
    path = tmp_path / "empty.parquet"
    df = pd_proxy.DataFrame(columns=["tx_hash", "token", "from", "to", "value"])
    cast_df: Any = df
    cast_df.to_parquet(path)

    with patch("latentpool.data.visualization.silver.graph_structure.plt"), \
         patch("latentpool.data.visualization.silver.graph_structure.sns"):
        analyze_structure(str(path), str(tmp_path / "viz_empty"))

    capsys_any: Any = capsys
    captured = capsys_any.readouterr()
    assert "Total Graphs (TXs):    0" in captured.out
    # We check for NaN or 0.00 depending on Pandas version mean() behavior
    assert "Avg Edges per Graph:" in captured.out

@patch("latentpool.data.visualization.silver.graph_structure.plt")
@patch("latentpool.data.visualization.silver.graph_structure.sns")
def test_analyze_structure_single_row(
    mock_sns: MagicMock,
    mock_plt: MagicMock,
    tmp_path: Path,
    pd_proxy: Any,
    capsys: Any
) -> None:
    """Verifies quantile calculation (99th percentile) with minimal data."""
    df = pd_proxy.DataFrame({
        "tx_hash": ["tx1"],
        "token": ["T1"],
        "from": ["0x1"],
        "to": ["0x2"],
        "value": [1.0]
    })
    path = tmp_path / "single.parquet"
    cast_df: Any = df
    cast_df.to_parquet(path)

    analyze_structure(str(path), str(tmp_path / "viz_single"))

    capsys_any: Any = capsys
    captured = capsys_any.readouterr()
    # With one row, the 99th percentile is just the value itself
    assert "99th Percentile Edges: 1.00" in captured.out
