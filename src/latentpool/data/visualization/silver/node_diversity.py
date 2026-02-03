import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

def analyze_node_diversity(silver_parquet: str, output_dir: str = "visualizations/structure") -> None:
    """Analyzes if the graph is a 'WETH Monoculture' or a diverse network."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    pd_any: Any = pd
    plt_any: Any = plt
    sns_any: Any = sns
    np_any: Any = np

    sns_any.set_theme(style="whitegrid")

    df = pd_any.read_parquet(silver_parquet)

    # degree (in + out) for every unique address
    all_nodes = pd_any.concat([df["from"], df["to"]])
    degree_counts = all_nodes.value_counts()

    counts_any: Any = degree_counts

    total_nodes = int(len(counts_any))
    total_edges = int(len(df))

    # Concentration metrics
    top_1_percent_count = max(1, total_nodes // 100)
    top_sum = float(counts_any.head(top_1_percent_count).sum())

    # safe against division by zero for empty graphs
    denominator = total_edges * 2
    top_1_percent_share = (top_sum / denominator) * 100 if denominator > 0 else 0.0

    print("\n" + "═" * 40)
    print("🧬 NODE DIVERSITY DIAGNOSTICS")
    print("═" * 40)
    print(f"Total Unique Nodes:   {total_nodes:,}")

    # robust against empty series for iloc access
    top_node_activity = int(counts_any.iloc[0]) if not counts_any.empty else 0
    print(f"Top Node Activity:    {top_node_activity:,} edges")

    single_edge_nodes = int(len(counts_any[counts_any == 1]))
    print(f"Nodes with 1 Edge:    {single_edge_nodes:,}")
    print(f"Top 1% Node Share:    {top_1_percent_share:.2f}%")
    print("-" * 40)

    # log-log degree distribution
    if total_nodes > 0:
        plt_any.figure(figsize=(10, 6))

        # We plot Rank vs. Frequency (Zipf's Law style)
        ranks = np_any.arange(1, total_nodes + 1)
        plt_any.loglog(ranks, counts_any.values, color="teal", linewidth=2)

        plt_any.fill_between(ranks, counts_any.values, color="teal", alpha=0.1)

        plt_any.title("Node Degree Distribution (Log-Log Scale)")
        plt_any.xlabel("Node Rank (Log Scale)")
        plt_any.ylabel("Degree / Edge Count (Log Scale)")

        top_label = str(counts_any.index[0])[:10]
        plt_any.annotate(
            f"Top Hub: {top_label}...",
            xy=(1, top_node_activity),
            xytext=(10, top_node_activity),
            arrowprops={"facecolor": "black", "shrink": 0.05}
        )

        plot_path = Path(output_dir) / "node_degree_diversity.png"
        plt_any.savefig(str(plot_path))
        plt_any.close()

        print(f"✅ Diversity plot saved to: {plot_path}")
    else:
        print("⚠️ No nodes found to visualize.")

    print("═" * 40 + "\n")
