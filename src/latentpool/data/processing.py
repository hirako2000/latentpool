# latentpool/data/processing.py

from pathlib import Path
from typing import Any, Dict, List, Set, cast

import pandas as pd
import torch
import typer
from torch_geometric.data import Data


class GraphBuilder:
    def __init__(self, gold_parquet: str):
        self.df: Any = pd.read_parquet(gold_parquet) # type: ignore

    def _compute_node_features(self, group: pd.DataFrame, node_map: Dict[Any, int], num_nodes: int) -> torch.Tensor:
        """Helper to calculate and standardize feature matrix x."""
        x = torch.zeros((num_nodes, 11), dtype=torch.float)
        tokens_in: Dict[int, Set[str]] = {n: set() for n in range(num_nodes)}
        tokens_out: Dict[int, Set[str]] = {n: set() for n in range(num_nodes)}
        neighbors: Dict[int, Set[int]] = {n: set() for n in range(num_nodes)}

        for _, row_raw in group.iterrows():
            row: Any = row_raw
            u, v = node_map[row["from"]], node_map[row["to"]]
            token = str(row["token"])

            x[u, 3] += 1
            x[v, 2] += 1
            tokens_out[u].add(token)
            tokens_in[v].add(token)
            neighbors[u].add(v)
            neighbors[v].add(u)

            val_log = torch.tensor(float(row["value"])).log1p()

            x[u, 6] += val_log
            x[v, 5] += val_log
            if u == v:
                x[u, 10] += 1

        for n in range(num_nodes):
            x[n, 0] = float(len(tokens_in[n]))
            x[n, 1] = float(len(tokens_out[n]))
            x[n, 4] = float(len(neighbors[n]))
            x[n, 7] = x[n, 5] - x[n, 6]
            x[n, 8] = 1.0 if x[n, 2] == 0 else 0.0
            x[n, 9] = 1.0 if x[n, 3] == 0 else 0.0

        for idx in [5, 6, 7]:
            feat_slice = x[:, idx]
            if num_nodes > 1:
                f_std = feat_slice.std()
                if f_std > 0:
                    x[:, idx] = (feat_slice - feat_slice.mean()) / f_std
            else:
                x[:, idx] = torch.clamp(feat_slice, min=-5.0, max=5.0)

        return x

    def build_and_save(self, output_dir: str):
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        groups: Any = self.df.groupby("tx_hash") # type: ignore
        total = len(groups)
        processed_rows: List[Dict[str, Any]] = []

        for i, (tx_hash, group_raw) in enumerate(groups):
            group: Any = group_raw

            combined_nodes: Any = pd.concat([group["from"], group["to"]])
            nodes_raw: List[Any] = list(combined_nodes.unique().tolist()) # type: ignore

            node_map: Dict[Any, int] = {addr: j for j, addr in enumerate(nodes_raw)}
            num_nodes = len(nodes_raw)

            f_vals: List[Any] = list(group["from"].tolist()) # type: ignore
            t_vals: List[Any] = list(group["to"].tolist()) # type: ignore

            edge_index = torch.tensor([
                [node_map[f] for f in f_vals],
                [node_map[t] for t in t_vals]
            ], dtype=torch.long)

            x = self._compute_node_features(cast(pd.DataFrame, group), node_map, num_nodes)

            # iloc type is unknown in groupby iterations
            label = int(group["label"].iloc[0]) # type: ignore
            split = str(group["split"].iloc[0]) # type: ignore

            for n_idx, address in enumerate(nodes_raw):
                processed_rows.append({
                    "tx_hash": tx_hash, "address": address, "label": label, "split": split,
                    "token_in_count": x[n_idx, 0].item(), "token_out_count": x[n_idx, 1].item(),
                    "transfers_in": x[n_idx, 2].item(), "transfers_out": x[n_idx, 3].item(),
                    "neighbor_count": x[n_idx, 4].item(), "log_vol_in_std": x[n_idx, 5].item(),
                    "log_vol_out_std": x[n_idx, 6].item(), "net_flow_std": x[n_idx, 7].item(),
                    "is_source": x[n_idx, 8].item(), "is_sink": x[n_idx, 9].item(),
                    "self_transfers": x[n_idx, 10].item()
                })

            data = Data(x=x, edge_index=edge_index, y=torch.tensor([label], dtype=torch.long))
            data.tx_hash = tx_hash
            data.train_mask = (split == "train")
            torch.save(data, out_path / f"{tx_hash}.pt")

            if i % 1000 == 0:
                typer.echo(f"  Processed {i}/{total} tensors...")

        final_parquet_path = out_path.parent / "gold_processed.parquet"
        pd.DataFrame(processed_rows).to_parquet(final_parquet_path, index=False) # type: ignore

        typer.secho(f"✨ Tensors saved to {output_dir}", fg=typer.colors.GREEN)
        typer.secho(f"📊 Processed Table saved to {final_parquet_path}", fg=typer.colors.CYAN)
