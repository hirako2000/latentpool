import logging
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

def analyze_structure(silver_parquet: str, output_dir: str = "visualizations/structure") -> None:
    """Analyzes the topological health of the Silver layer."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    pd_any: Any = pd
    plt_any: Any = plt
    sns_any: Any = sns

    df = pd_any.read_parquet(silver_parquet)

    stats: Any = df.groupby("tx_hash").agg(
        edge_count=("token", "count"),
        unique_tokens=("token", "nunique")
    )

    # Node-to-Edge ratio
    stats["complexity"] = stats["edge_count"] / stats["unique_tokens"]

    edge_col: Any = stats["edge_count"]
    comp_col: Any = stats["complexity"]

    metrics: Dict[str, Any] = {
        "avg_edges": float(edge_col.mean()) if not edge_col.empty else 0.0,
        "max_edges": int(edge_col.max()) if not edge_col.empty else 0,
        "avg_complexity": float(comp_col.mean()) if not comp_col.empty else 0.0,
        "outlier_threshold": float(edge_col.quantile(0.99)) if not edge_col.empty else 0.0,
        "total_graphs": int(len(stats))
    }

    print("\n" + "═" * 40)
    print("🏗️  GRAPH STRUCTURE DIAGNOSTICS")
    print("═" * 40)
    print(f"Total Graphs (TXs):    {metrics['total_graphs']:,}")
    print(f"Avg Edges per Graph:   {metrics['avg_edges']:.2f}")
    print(f"Max Edges (Outlier):   {metrics['max_edges']:,}")
    print(f"Avg Complexity Ratio:  {metrics['avg_complexity']:.2f}")
    print(f"99th Percentile Edges: {metrics['outlier_threshold']:.2f}")
    print("-" * 40)

    # Complexity vs. Scale
    plt_any.figure(figsize=(10, 6))
    sns_any.set_theme(style="whitegrid")

    sns_any.scatterplot(
        data=stats,
        x="unique_tokens",
        y="edge_count",
        alpha=0.4,
        color="purple"
    )

    plt_any.title("Graph Topology: Nodes vs. Edges")
    plt_any.xlabel("Unique Tokens (Nodes)")
    plt_any.ylabel("Total Transfers (Edges)")

    # Black Hole zone
    plt_any.axhline(y=metrics['outlier_threshold'], color='r', linestyle='--', label='99th %-tile')
    plt_any.legend()

    plot_path = Path(output_dir) / "node_edge_ratio.png"
    plt_any.savefig(str(plot_path))
    plt_any.close()

    print(f"✅ Structural plot saved to: {plot_path}")
    print("═" * 40 + "\n")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        analyze_structure(sys.argv[1])
