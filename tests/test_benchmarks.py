from typing import Any, Callable

import pytest
import torch

from latentpool.gnn import TransactionGraphModel


@pytest.mark.benchmark(group="gnn-inference")
def test_gnn_inference_latency(benchmark: Callable[..., Any]) -> None:
    # Setup: 100 nodes, 16 features each
    model = TransactionGraphModel(16, 32, 8)
    x = torch.randn(100, 16).to(model.device)
    edge_index = torch.randint(0, 100, (2, 500)).to(model.device)
    batch = torch.zeros(100, dtype=torch.long).to(model.device)

    model.eval()

    with torch.no_grad():
        benchmark(model, x, edge_index, batch)
