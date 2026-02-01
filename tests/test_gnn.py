from unittest.mock import patch

import msgspec
import pytest
import torch

from latentpool.gnn import TransactionGraphModel, TransactionTrace


def test_trace_decoding():
    """
    Verifies msgspec decoding of a Parity-style transaction trace.
    """
    # Realistic trace JSON payload
    raw_json = b"""
    {
        "action": {
            "from": "0x1234567890123456789012345678901234567890",
            "to": "0x0987654321098765432109876543210987654321",
            "value": "0xde0b6b3a7640000",
            "gas": "0x5208",
            "input": "0x",
            "callType": "call"
        },
        "result": {
            "gasUsed": "0x5208",
            "output": "0x"
        },
        "subtraces": 0,
        "traceAddress": [],
        "type": "call"
    }
    """
    trace = msgspec.json.decode(raw_json, type=TransactionTrace)

    assert trace.action.from_address == "0x1234567890123456789012345678901234567890"
    assert trace.action.call_type == "call"
    assert trace.result is not None
    assert trace.result.gas_used == "0x5208"
    assert trace.trace_address == []

def test_gnn_forward_pass():
    """
    Verifies the GNN architecture processes tensors of expected dimensions.
    """
    input_dim = 16
    hidden_dim = 32
    output_dim = 8

    model = TransactionGraphModel(input_dim, hidden_dim, output_dim)
    device = model.device

    # Mock graph: 4 nodes, 16 features each
    x = torch.randn(4, input_dim).to(device)
    # 3 edges (0->1, 1->2, 2->3)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long).to(device)
    # All nodes belong to the same graph in this batch
    batch = torch.zeros(4, dtype=torch.long).to(device)

    output = model(x, edge_index, batch)

    # Verify execution on the detected device
    assert output.device.type == device.type
    assert output.shape == (1, output_dim)
    assert not torch.isnan(output).any()

if __name__ == "__main__":
    pytest.main([__file__])

def test_gnn_device_selection_mps(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    # We mock __init__ so it doesn't call self.to()
    with patch.object(TransactionGraphModel, "__init__", return_value=None):
        model = TransactionGraphModel(1, 1, 1)
        assert model._get_best_device().type == "mps" # type: ignore[reportPrivateUsage]

def test_gnn_device_selection_cuda(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    with patch.object(TransactionGraphModel, "__init__", return_value=None):
        model = TransactionGraphModel(1, 1, 1)
        assert model._get_best_device().type == "cuda" # type: ignore[reportPrivateUsage]

def test_gnn_device_selection_cpu(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with patch.object(TransactionGraphModel, "__init__", return_value=None):
        model = TransactionGraphModel(1, 1, 1)
        assert model._get_best_device().type == "cpu" # type: ignore[reportPrivateUsage]
