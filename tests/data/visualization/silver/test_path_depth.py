from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from latentpool.data.visualization.silver.path_depth import analyze_path_depth


@pytest.fixture
def pd_proxy() -> Any:
    """Provides a pd proxy for type-safe dataframe creation."""
    return pd

@patch("latentpool.data.visualization.silver.path_depth.plt")
@patch("latentpool.data.visualization.silver.path_depth.sns")
def test_analyze_path_depth_dag_flow(
    mock_sns: MagicMock,
    mock_plt: MagicMock,
    tmp_path: Path,
    pd_proxy: Any,
    capsys: Any
) -> None:
    """
    Covers the 'if' block: nx_any.dag_longest_path_length(G).
    This is a standard linear swap (DAG).
    """
    # Chain: User -> Router -> Pool -> User (3 edges, 3 hops)
    df = pd_proxy.DataFrame({
        "tx_hash": ["dag_tx"] * 3,
        "from": ["0xUser", "0xRouter", "0xPool"],
        "to": ["0xRouter", "0xPool", "0xUser_Back"], # Distinct nodes to ensure DAG
        "token": ["t1"] * 3,
        "value": [1.0] * 3
    })
    path = tmp_path / "dag.parquet"
    cast_df: Any = df
    cast_df.to_parquet(path)

    analyze_path_depth(str(path), str(tmp_path / "viz_dag"))

    capsys_any: Any = capsys
    captured = capsys_any.readouterr()

    # 3 edges in a sequence = 3 hops
    assert "Max Path Found:      3 hops" in captured.out
    assert "Avg Path Depth:      3.00 hops" in captured.out

@patch("latentpool.data.visualization.silver.path_depth.plt")
@patch("latentpool.data.visualization.silver.path_depth.sns")
def test_analyze_path_depth_cyclic(
    mock_sns: MagicMock,
    mock_plt: MagicMock,
    tmp_path: Path,
    pd_proxy: Any,
    capsys: Any
) -> None:
    """Covers the 'else' block: nx_any.diameter(G.to_undirected())."""
    df = pd_proxy.DataFrame({
        "tx_hash": ["cycle_tx"] * 2,
        "from": ["A", "B"],
        "to": ["B", "A"], # Simple 2-node cycle
        "token": ["t1"] * 2,
        "value": [1.0] * 2
    })
    path = tmp_path / "cycle.parquet"
    cast_df: Any = df
    cast_df.to_parquet(path)

    analyze_path_depth(str(path), str(tmp_path / "viz_cycle"))

    capsys_any: Any = capsys
    captured = capsys_any.readouterr()
    assert "Max Path Found:" in captured.out

@patch("latentpool.data.visualization.silver.path_depth.plt")
@patch("latentpool.data.visualization.silver.path_depth.sns")
def test_analyze_path_depth_exception(
    mock_sns: MagicMock,
    mock_plt: MagicMock,
    tmp_path: Path,
    pd_proxy: Any,
    capsys: Any
) -> None:
    """Covers the 'except Exception' block."""
    df = pd_proxy.DataFrame({
        "tx_hash": ["fail_tx"],
        "from": ["A"],
        "to": ["B"],
        "token": ["t1"],
        "value": [1.0]
    })
    path = tmp_path / "fail.parquet"
    cast_df: Any = df
    cast_df.to_parquet(path)

    with patch("networkx.is_directed_acyclic_graph", side_effect=Exception("Simulated Failure")):
        analyze_path_depth(str(path), str(tmp_path / "viz_fail"))

    capsys_any: Any = capsys
    captured = capsys_any.readouterr()
    assert "Max Path Found:      0 hops" in captured.out
