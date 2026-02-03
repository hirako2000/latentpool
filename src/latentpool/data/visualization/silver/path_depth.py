import logging
from pathlib import Path
from typing import Any, List

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

def analyze_path_depth(silver_parquet: str, output_dir: str = "visualizations/structure") -> None:
    """Calculates the max swap chain length for each transaction."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    pd_any: Any = pd
    nx_any: Any = nx
    plt_any: Any = plt
    sns_any: Any = sns

    sns_any.set_theme(style="whitegrid")

    df = pd_any.read_parquet(silver_parquet)
    grouped = df.groupby("tx_hash")

    path_lengths: List[int] = []

    print(f"🧬 Processing {len(grouped):,} transactions for path depth...")

    for _, tx_df in grouped:
        # graph using isolated nx
        G = nx_any.from_pandas_edgelist(
            tx_df,
            source="from",
            target="to",
            create_using=nx_any.DiGraph()
        )

        try:
            if nx_any.is_directed_acyclic_graph(G):
                depth = nx_any.dag_longest_path_length(G)
            else:
                depth = nx_any.diameter(G.to_undirected())
            path_lengths.append(int(depth))
        except Exception:
            path_lengths.append(0)

    depth_series = pd.Series(path_lengths)
    series_any: Any = depth_series

    avg_d = float(series_any.mean()) if not series_any.empty else 0.0
    max_d = int(series_any.max()) if not series_any.empty else 0

    modes = series_any.mode()
    mode_d = int(modes.iloc[0]) if not modes.empty else 0

    complex_ratio = float((series_any > 1).mean() * 100) if not series_any.empty else 0.0

    print("\n" + "═" * 40)
    print("🏗️  GRAPH PATH DEPTH DIAGNOSTICS")
    print("═" * 40)
    print(f"Avg Path Depth:      {avg_d:.2f} hops")
    print(f"Most Frequent Depth: {mode_d} hops")
    print(f"Max Path Found:      {max_d} hops")
    print(f"Multi-hop Txs (>1):  {complex_ratio:.2f}%")
    print("-" * 40)

    # calls via isolated plt/sns
    plt_any.figure(figsize=(10, 6))
    sns_any.countplot(x=path_lengths, color="salmon")

    plt_any.title("Distribution of Transaction Path Depths")
    plt_any.xlabel("Max Path Length (Hops)")
    plt_any.ylabel("Transaction Count")

    plot_path = Path(output_dir) / "path_depth_dist.png"
    plt_any.savefig(str(plot_path))
    plt_any.close()

    print(f"✅ Path depth plot saved to: {plot_path}")
    print("═" * 40 + "\n")
