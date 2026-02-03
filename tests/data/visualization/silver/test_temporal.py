from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from latentpool.data.visualization.silver.temporal import analyze_temporal_flow


@pytest.fixture
def mock_silver_parquet(tmp_path: Path) -> Path:
    """Creates a dummy silver parquet file for testing."""
    pd_any: Any = pd
    df = pd_any.DataFrame({
        "block_number": [100, 100, 101, 102, 102, 102],
        "tx_hash": ["tx1", "tx2", "tx3", "tx4", "tx4", "tx5"],
        "from": ["a", "b", "c", "d", "e", "f"],
        "to": ["g", "h", "i", "j", "k", "l"],
        "token": ["t1", "t1", "t1", "t1", "t1", "t1"],
        "value": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    })
    path = tmp_path / "test_silver.parquet"
    df_any: Any = df
    df_any.to_parquet(path)
    return path

@patch("latentpool.data.visualization.silver.temporal.plt")
@patch("latentpool.data.visualization.silver.temporal.sns")
def test_analyze_temporal_flow_logic(
    mock_sns: MagicMock,
    mock_plt: MagicMock,
    mock_silver_parquet: Path,
    tmp_path: Path,
    capsys: Any
) -> None:
    """Tests if the temporal analysis calculates metrics correctly and calls plot."""
    output_dir = tmp_path / "viz"

    analyze_temporal_flow(str(mock_silver_parquet), str(output_dir))

    assert output_dir.exists()

    capsys_any: Any = capsys
    captured = capsys_any.readouterr()
    assert "Blocks Covered:      3" in captured.out
    assert "Avg Txs per Block:   1.67" in captured.out
    assert "Max Txs in a Block:  2" in captured.out

    mock_plt.figure.assert_called_once()
    mock_sns.lineplot.assert_called_once()
    mock_plt.savefig.assert_called_once()
    mock_plt.close.assert_called_once()

    expected_plot_path = output_dir / "temporal_density.png"
    assert f"✅ Temporal plot saved to: {expected_plot_path}" in captured.out

def test_analyze_temporal_flow_empty_df(tmp_path: Path, capsys: Any) -> None:
    """Ensures the code doesn't crash with an empty dataframe."""
    pd_any: Any = pd
    path = tmp_path / "empty.parquet"
    df = pd_any.DataFrame(columns=["block_number", "tx_hash"])
    df_any: Any = df
    df_any.to_parquet(path)

    output_dir = tmp_path / "viz_empty"

    with patch("latentpool.data.visualization.silver.temporal.plt"), \
         patch("latentpool.data.visualization.silver.temporal.sns"):
        analyze_temporal_flow(str(path), str(output_dir))

    capsys_any: Any = capsys
    captured = capsys_any.readouterr()
    assert "Blocks Covered:      0" in captured.out
