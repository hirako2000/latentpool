from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch_geometric.data import Data

from latentpool.data.visualization.gold.tensor_health import analyze_tensor_features

EXPECTED_PLOT_COUNT = 2

@pytest.fixture
def mock_graph_dir(tmp_path: Path) -> str:
    graphs_dir = tmp_path / "graphs"
    graphs_dir.mkdir()

    for i in range(5):
        x = torch.randn((10, 11))
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index)

        torch.save(data, graphs_dir / f"graph_{i}.pt")

    return str(graphs_dir)

@patch("latentpool.data.visualization.gold.tensor_health.plt")
@patch("latentpool.data.visualization.gold.tensor_health.sns")
def test_analyze_tensor_features_execution(
    mock_sns: MagicMock,
    mock_plt: MagicMock,
    mock_graph_dir: str,
    tmp_path: Path
):
    output_dir = tmp_path / "viz_health"

    analyze_tensor_features(graphs_dir=mock_graph_dir, output_dir=str(output_dir))

    assert output_dir.exists()
    assert mock_plt.savefig.call_count == EXPECTED_PLOT_COUNT
    assert mock_plt.close.call_count == EXPECTED_PLOT_COUNT

@patch("latentpool.data.visualization.gold.tensor_health.plt")
def test_analyze_tensor_features_no_files(mock_plt: MagicMock, tmp_path: Path):
    """no .pt present."""
    empty_dir = tmp_path / "empty_graphs"
    empty_dir.mkdir()

    output_dir = tmp_path / "viz_empty"

    analyze_tensor_features(graphs_dir=str(empty_dir), output_dir=str(output_dir))

    assert mock_plt.savefig.call_count == 0

@patch("latentpool.data.visualization.gold.tensor_health.torch.load")
@patch("latentpool.data.visualization.gold.tensor_health.plt")
def test_analyze_tensor_features_corrupt_files(
    mock_plt: MagicMock,
    mock_load: MagicMock,
    mock_graph_dir: str,
    tmp_path: Path
):
    mock_load.side_effect = RuntimeError("Corrupt file")

    output_dir = tmp_path / "viz_corrupt"
    analyze_tensor_features(graphs_dir=mock_graph_dir, output_dir=str(output_dir))

    assert mock_plt.savefig.call_count == 0

def test_feature_matrix_alignment(mock_graph_dir: str, tmp_path: Path):
    """
    Correctly stacks features.
    without mocking the plotting internals.
    """
    with patch("latentpool.data.visualization.gold.tensor_health._generate_feature_reports") as mock_report:
        analyze_tensor_features(graphs_dir=mock_graph_dir, output_dir=str(tmp_path))

        # 5 graphs * 10 nodes = 50 rows; 11 features = 11 columns
        args, _ = mock_report.call_args
        feat_matrix = args[0]
        assert feat_matrix.shape == (50, 11)

        # Verify feature labels list length
        labels = args[1]
        expected_label_count = 11
        assert len(labels) == expected_label_count
        assert labels[0] == "Token In Count"

@patch("latentpool.data.visualization.gold.tensor_health.plt")
@patch("latentpool.data.visualization.gold.tensor_health.sns")
def test_analyze_tensor_features_handles_1d_tensors(
    mock_sns: MagicMock,
    mock_plt: MagicMock,
    tmp_path: Path
):
    """
    1D tensors are correctly reshaped to (1, -1) before stacking.
    """
    graphs_dir = tmp_path / "single_vector_graphs"
    graphs_dir.mkdir()

    x_1d = torch.randn(11)
    data = Data(x=x_1d)

    torch.save(data, graphs_dir / "vector_graph.pt")

    output_dir = tmp_path / "viz_1d"

    with patch("latentpool.data.visualization.gold.tensor_health._generate_feature_reports") as mock_report:
        analyze_tensor_features(graphs_dir=str(graphs_dir), output_dir=str(output_dir))

        args, _ = mock_report.call_args
        feat_matrix = args[0]

        expected_feat_dim = 2
        assert feat_matrix.ndim == expected_feat_dim
        assert feat_matrix.shape == (1, 11)
