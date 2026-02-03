from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from latentpool.data.visualization.silver.node_diversity import analyze_node_diversity


@pytest.fixture
def pd_proxy() -> Any:
    """Provides a pd proxy for type-safe dataframe creation."""
    return pd

@pytest.fixture
def mock_diversity_parquet(tmp_path: Path, pd_proxy: Any) -> Path:
    """
    Creates a dummy silver parquet with a clear hierarchy:
    - One massive Hub (0xHUB) with degree 10
    - 5 Mid-tier nodes with degree 1
    - 5 Leaf nodes with degree 1
    Total edges = 10, Total unique nodes = 11.
    """
    data = {
        "tx_hash": [f"tx{i}" for i in range(10)],
        "from": ["0xHUB"] * 5 + [f"0xLeaf{i}" for i in range(5)],
        "to": [f"0xMid{i}" for i in range(5)] + ["0xHUB"] * 5,
        "token": ["t1"] * 10,
        "value": [1.0] * 10
    }
    df = pd_proxy.DataFrame(data)
    path = tmp_path / "test_diversity.parquet"
    cast_df: Any = df
    cast_df.to_parquet(path)
    return path

@patch("latentpool.data.visualization.silver.node_diversity.plt")
@patch("latentpool.data.visualization.silver.node_diversity.sns")
def test_analyze_node_diversity_logic(
    mock_sns: MagicMock,
    mock_plt: MagicMock,
    mock_diversity_parquet: Path,
    tmp_path: Path,
    capsys: Any
) -> None:
    """Tests if diversity metrics (Hub activity and 1-edge nodes) are correct."""
    output_dir = tmp_path / "viz_diversity"

    analyze_node_diversity(str(mock_diversity_parquet), str(output_dir))

    capsys_any: Any = capsys
    captured = capsys_any.readouterr()

    # Assertions based on our 11-node, 10-edge mock setup
    assert "Total Unique Nodes:   11" in captured.out
    assert "Top Node Activity:    10 edges" in captured.out
    assert "Nodes with 1 Edge:    10" in captured.out

    mock_plt.figure.assert_called_once()
    mock_plt.loglog.assert_called_once()
    mock_plt.annotate.assert_called_once()
    mock_plt.savefig.assert_called_once()
    mock_plt.close.assert_called_once()

@patch("latentpool.data.visualization.silver.node_diversity.plt")
@patch("latentpool.data.visualization.silver.node_diversity.sns")
def test_analyze_node_diversity_minimal(
    mock_sns: MagicMock,
    mock_plt: MagicMock,
    tmp_path: Path,
    pd_proxy: Any,
    capsys: Any
) -> None:
    """Tests handling of a minimal 2-node graph."""
    df = pd_proxy.DataFrame({
        "from": ["0x1"],
        "to": ["0x2"],
        "tx_hash": ["tx1"],
        "token": ["t1"],
        "value": [1.0]
    })
    path = tmp_path / "minimal.parquet"
    cast_df: Any = df
    cast_df.to_parquet(path)

    analyze_node_diversity(str(path), str(tmp_path / "viz_min"))

    capsys_any: Any = capsys
    captured = capsys_any.readouterr()
    assert "Total Unique Nodes:   2" in captured.out
    assert "Top 1% Node Share:" in captured.out

def test_analyze_node_diversity_empty(tmp_path: Path, pd_proxy: Any, capsys: Any) -> None:
    """Ensures empty data handles division by zero gracefully."""
    path = tmp_path / "empty.parquet"
    df = pd_proxy.DataFrame(columns=["from", "to", "tx_hash", "token", "value"])
    cast_df: Any = df
    cast_df.to_parquet(path)

    with patch("latentpool.data.visualization.silver.node_diversity.plt"), \
         patch("latentpool.data.visualization.silver.node_diversity.sns"):
        # This will pass if the source has the denominator > 0 guard
        analyze_node_diversity(str(path), str(tmp_path / "viz_empty"))

    capsys_any: Any = capsys
    captured = capsys_any.readouterr()
    assert "🧬 NODE DIVERSITY DIAGNOSTICS" in captured.out
    assert "Total Unique Nodes:   0" in captured.out
    assert "Top Node Activity:    0 edges" in captured.out
