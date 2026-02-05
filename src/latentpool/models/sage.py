from typing import Protocol

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import BatchNorm, SAGEConv, global_max_pool, global_mean_pool


class ModelConfig(Protocol):
    in_channels: int
    hidden_channels: int
    out_channels: int
    mlp_layer_1_dim: int
    mlp_layer_2_dim: int
    dropout_rate: float
    output_temperature: float

class MEVGraphSAGE(torch.nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        in_ch: int = config.in_channels
        hid_ch: int = config.hidden_channels

        self.input_bn = BatchNorm(in_ch)

        self.conv1 = SAGEConv(in_ch, hid_ch)
        self.conv2 = SAGEConv(hid_ch, hid_ch)
        self.conv3 = SAGEConv(hid_ch, hid_ch)
        self.conv4 = SAGEConv(hid_ch, hid_ch)

        pooled_dim: int = hid_ch * 2
        self.post_pool_bn = torch.nn.BatchNorm1d(pooled_dim)

        self.lin1 = torch.nn.Linear(pooled_dim, config.mlp_layer_1_dim)
        self.lin2 = torch.nn.Linear(config.mlp_layer_1_dim, config.mlp_layer_2_dim)
        self.lin3 = torch.nn.Linear(config.mlp_layer_2_dim, config.out_channels)

        self.dropout_rate: float = config.dropout_rate
        self.temperature: float = config.output_temperature

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        x = self.input_bn(x)
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()
        x = self.conv4(x, edge_index).relu()

        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)

        if x.shape[0] > 1:
            x = self.post_pool_bn(x)

        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))

        out: Tensor = self.lin3(x) / self.temperature
        return out
