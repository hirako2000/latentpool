import pytest
import torch

from latentpool.models.sage import MEVGraphSAGE


class MockConfig:
    in_channels = 11
    hidden_channels = 64
    out_channels = 3
    mlp_layer_1_dim = 32
    mlp_layer_2_dim = 16
    dropout_rate = 0.5
    output_temperature = 1.0

@pytest.fixture
def model() -> MEVGraphSAGE:
    config = MockConfig()
    return MEVGraphSAGE(config) # type: ignore

def test_sage_forward_shape(model: MEVGraphSAGE) -> None:
    # 10 nodes, 11 features
    x = torch.randn((10, 11))
    # 2 edges
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    # Single graph
    batch = torch.zeros(10, dtype=torch.long)

    output = model(x, edge_index, batch)

    # Batch size is 1, out_channels is 3
    assert output.shape == (1, 3)

def test_sage_batch_forward_shape(model: MEVGraphSAGE) -> None:
    # 2 graphs with 5 nodes each
    x = torch.randn((10, 11))
    edge_index = torch.tensor([[0, 1, 5, 6], [1, 0, 6, 5]], dtype=torch.long)
    batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.long)

    output = model(x, edge_index, batch)

    # 2 graphs in batch, out_channels is 3
    assert output.shape == (2, 3)
