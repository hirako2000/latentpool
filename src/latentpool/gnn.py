from typing import List, Optional

import msgspec
import torch
from torch_geometric.nn import SAGEConv, global_mean_pool


class TraceAction(msgspec.Struct):
    """
    Represents the 'action' field in a transaction trace.
    """
    from_address: str = msgspec.field(name="from")
    to: Optional[str] = None
    value: str = "0x0"
    gas: str = "0x0"
    input: str = "0x"
    call_type: Optional[str] = msgspec.field(name="callType", default=None)


class TraceResult(msgspec.Struct):
    """
    Represents the 'result' field in a transaction trace.
    """
    gas_used: str = msgspec.field(name="gasUsed", default="0x0")
    output: str = "0x"


class TransactionTrace(msgspec.Struct):
    """
    High-performance schema for a single trace entry.
    """
    action: TraceAction
    result: Optional[TraceResult] = None
    subtraces: int = 0
    trace_address: List[int] = msgspec.field(name="traceAddress", default_factory=list)
    type: str = "call"
    error: Optional[str] = None


class TransactionGraphModel(torch.nn.Module):
    """
    GNN for transaction behavior analysis using GraphSAGE layers.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.device = self._get_best_device()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, output_dim)
        self.to(self.device)

    def _get_best_device(self) -> torch.device:
        """
        Determines the most efficient available hardware backend.
        """
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass over the batch of transaction graphs.
        """
        x = self.process_layers(x, edge_index)
        return self.aggregate_and_predict(x, batch)

    def process_layers(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Applies graph convolutional layers with ReLU activation.
        """
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

    def aggregate_and_predict(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Global pooling and final prediction head.
        """
        x = global_mean_pool(x, batch)
        return x
