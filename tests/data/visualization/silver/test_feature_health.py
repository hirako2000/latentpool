from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from latentpool.data.visualization.silver.feature_health import analyze_feature_health


@pytest.fixture
def pd_proxy() -> Any:
    """Provides a pd proxy for type-safe dataframe creation."""
    return pd

@pytest.fixture
def mock_health_parquet(tmp_path: Path, pd_proxy: Any) -> Path:
    """
    Creates a dummy silver parquet with varied value magnitudes:
    - 2 zero values
    - 3 non-zero values (1.0, 100.0, 10000.0)
    """
    data = {
        "tx_hash": [f"tx{i}" for i in range(5)],
        "from": ["0x1"] * 5,
        "to": ["0x2"] * 5,
        "token": ["T1"] * 5,
        "value": [0.0, 0, 1.0, 100.0, 10000.0]
    }
    df = pd_proxy.DataFrame(data)
    path = tmp_path / "test_health.parquet"
    cast_df: Any = df
    cast_df.to_parquet(path)
    return path

@patch("latentpool.data.visualization.silver.feature_health.plt")
@patch("latentpool.data.visualization.silver.feature_health.sns")
def test_analyze_feature_health_logic(
    mock_sns: MagicMock,
    mock_plt: MagicMock,
    mock_health_parquet: Path,
    tmp_path: Path,
    capsys: Any
) -> None:
    """Tests if value distributions and zero-value percentages are correct."""
    output_dir = tmp_path / "viz_health"

    analyze_feature_health(str(mock_health_parquet), str(output_dir))

    capsys_any: Any = capsys
    captured = capsys_any.readouterr()

    # Logic: 5 total edges, 2 are zeros (40%)
    # Non-zero Min: 1.0, Max: 10000.0, Median: 100.0
    assert "Total Edges Analyzed: 5" in captured.out
    assert "Zero-Value Edges:    2 (40.00%)" in captured.out
    assert "Min Non-Zero Value:  1.000000" in captured.out
    assert "Max Value:           10,000.00" in captured.out
    assert "Median Value:        100.000000" in captured.out

    # Verify Plotting (Histplot and log10 transformation)
    mock_plt.figure.assert_called_once()
    mock_sns.histplot.assert_called_once()
    mock_plt.savefig.assert_called_once()

def test_analyze_feature_health_empty(tmp_path: Path, pd_proxy: Any, capsys: Any) -> None:
    """Ensures empty dataset doesn't crash and reports 0 edges."""
    path = tmp_path / "empty.parquet"
    df = pd_proxy.DataFrame(columns=["tx_hash", "from", "to", "token", "value"])
    cast_df: Any = df
    cast_df.to_parquet(path)

    with patch("latentpool.data.visualization.silver.feature_health.plt"), \
         patch("latentpool.data.visualization.silver.feature_health.sns"):
        analyze_feature_health(str(path), str(tmp_path / "viz_empty"))

    capsys_any: Any = capsys
    captured = capsys_any.readouterr()
    assert "Total Edges Analyzed: 0" in captured.out
    assert "Zero-Value Edges:    0 (0.00%)" in captured.out
    assert "⚠️ No non-zero values found to plot." in captured.out

@patch("latentpool.data.visualization.silver.feature_health.plt")
@patch("latentpool.data.visualization.silver.feature_health.sns")
def test_analyze_feature_health_all_zeros(
    mock_sns: MagicMock,
    mock_plt: MagicMock,
    tmp_path: Path,
    pd_proxy: Any,
    capsys: Any
) -> None:
    """Tests the scenario where all transfers are zero-value (e.g., failed extractions)."""
    df = pd_proxy.DataFrame({
        "tx_hash": ["tx1"],
        "from": ["0x1"],
        "to": ["0x2"],
        "token": ["T1"],
        "value": ["0x0"] # numeric conversion of hex-style zero
    })
    path = tmp_path / "zeros.parquet"
    cast_df: Any = df
    cast_df.to_parquet(path)

    analyze_feature_health(str(path), str(tmp_path / "viz_zeros"))

    capsys_any: Any = capsys
    captured = capsys_any.readouterr()
    assert "Zero-Value Edges:    1 (100.00%)" in captured.out
    assert "⚠️ No non-zero values found to plot." in captured.out
