from pathlib import Path
from typing import Any, Dict, List, Set, cast

import pandas as pd
import torch
import typer
from torch_geometric.data import Data


class GraphBuilder:
    def __init__(self, gold_parquet: str):
        """
        Reads the Gold Parquet which has 'label', 'split', 'token', and 'value'.
        """
        self.df: Any = pd.read_parquet(gold_parquet)  # type: ignore

    def build_and_save(self, output_dir: str):
        """
        Iterates through transactions and builds 14-feature node tensors.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        groups: Any = self.df.groupby("tx_hash")
        total = len(groups)

        for i, (tx_hash, group) in enumerate(groups):
            combined_nodes: Any = pd.concat([group["from"], group["to"]])
            nodes_list: Any = combined_nodes.unique().tolist()
            nodes_raw = cast(List[Any], nodes_list)

            node_map: Dict[Any, int] = {addr: j for j, addr in enumerate(nodes_raw)}
            num_nodes = len(nodes_raw)

            edge_index = torch.tensor([
                [node_map[f] for f in group["from"]],
                [node_map[t] for t in group["to"]]
            ], dtype=torch.long)

            x = torch.zeros((num_nodes, 14), dtype=torch.float)

            tokens_in: Dict[int, Set[str]] = {n: set() for n in range(num_nodes)}
            tokens_out: Dict[int, Set[str]] = {n: set() for n in range(num_nodes)}
            neighbors: Dict[int, Set[int]] = {n: set() for n in range(num_nodes)}

            for _, row in group.iterrows():  # type: ignore
                u, v = node_map[row["from"]], node_map[row["to"]]
                token = str(row["token"])

                # Degrees
                x[u, 4] += 1  # total_transfers_out
                x[v, 3] += 1  # total_transfers_in

                # Variety
                tokens_out[u].add(token)
                tokens_in[v].add(token)

                # Neighbor tracking (Index 7)
                neighbors[u].add(v)
                neighbors[v].add(u)

                # Volumes
                raw_val = float(str(row["value"]))
                val_log = torch.tensor(raw_val + 1).log()
                x[u, 9] += val_log  # log_volume_out
                x[v, 8] += val_log  # log_volume_in

                # Self-transfers
                if u == v:
                    x[u, 13] += 1

            # features that require aggregation
            for n in range(num_nodes):
                x[n, 1] = float(len(tokens_in[n]))   # token_in_count
                x[n, 2] = float(len(tokens_out[n]))  # token_out_count
                x[n, 7] = float(len(neighbors[n]))   # neighbor_count
                x[n, 10] = x[n, 8] - x[n, 9]         # net_token_flow

                # Topology Flags
                if x[n, 3] == 0:
                    x[n, 11] = 1.0      # is_source
                if x[n, 4] == 0:
                    x[n, 12] = 1.0      # is_sink

            y = torch.tensor([group["label"].iloc[0]], dtype=torch.long)  # type: ignore
            data: Any = Data(x=x, edge_index=edge_index, y=y)
            data.tx_hash = tx_hash
            # Embed the split info (True if train, False if test)
            data.train_mask = (group["split"].iloc[0] == "train")  # type: ignore

            torch.save(data, out_path / f"{tx_hash}.pt")

            if i % 1000 == 0:
                typer.echo(f"  Processed {i}/{total} tensors...")

        typer.secho(f"✨ Successfully saved {total} tensors to {output_dir}", fg=typer.colors.GREEN)
