import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

def analyze_label_complexity(gold_parquet: str, output_dir: str = "visualizations/gold") -> None:
    """
    Compares the structural complexity (edges and hops) across MEV labels.
    Helps verify if MEV attacks have a distinct 'fingerprint' compared to Normal txs.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    pd_any: Any = pd
    plt_any: Any = plt
    sns_any: Any = sns

    sns_any.set_theme(style="whitegrid")

    df = pd_any.read_parquet(gold_parquet)
    label_map = {0: "Normal", 1: "Arbitrage", 2: "Sandwich"}
    df["label_name"] = df["label"].map(label_map)

    # Calculate complexity per transaction
    # Since it's an edge list, count rows per tx_hash
    stats = df.groupby(["tx_hash", "label_name"]).size().reset_index(name="edge_count")

    print("\n" + "═" * 50)
    print("📊 LABEL COMPLEXITY DIAGNOSTICS")
    print("═" * 50)

    avg_stats = stats.groupby("label_name")["edge_count"].mean()
    print("Average Edges per Transaction Type:")
    print(avg_stats.to_string())
    print("-" * 50)

    plt_any.figure(figsize=(12, 6))

    sns_any.violinplot(data=stats, x="label_name", y="edge_count", hue="label_name", palette="magma", legend=False)
    plt_any.yscale("log")

    plt_any.title("Structural Complexity: Edge Count Distribution by Label")
    plt_any.xlabel("Transaction Type")
    plt_any.ylabel("Edges per Tx (Log Scale)")

    complexity_path = Path(output_dir) / "label_complexity_comparison.png"
    plt_any.savefig(str(complexity_path))
    plt_any.close()

    print(f"✅ Complexity plot saved: {complexity_path}")
    print("═" * 50 + "\n")
