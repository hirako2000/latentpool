from typing import Any

import networkx as nx
import pandas as pd
from pyvis.network import Network


class GraphExplorer:
    def __init__(self, parquet_path: str):
        self.df: Any = pd.read_parquet(parquet_path) # type: ignore

    def generate_tx_graph(self, tx_hash: str, output_path: str = "graph.html"):
        tx_edges: Any = self.df[self.df['tx_hash'] == tx_hash.lower()]

        if tx_edges.empty: # type: ignore
            print(f"No data found for hash {tx_hash}")
            return

        G: Any = nx.MultiDiGraph() # type: ignore

        for _, row in tx_edges.iterrows(): # type: ignore
            G.add_edge( # type: ignore
                row['from'],
                row['to'],
                label=f"Token: {str(row['token'])[:6]}...",
                title=f"Value: {row['value']}"
            )

        net: Any = Network(notebook=False, directed=True, height="750px", width="100%")
        net.from_nx(G) # type: ignore
        net.show(output_path) # type: ignore
        print(f"✨ Visualization saved to {output_path}")
