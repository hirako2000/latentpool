import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

def analyze_gold_balance(gold_parquet: str, output_dir: str = "visualizations/gold") -> None:
    """
    Analyzes class distribution and temporal splitting in the Gold layer.
    Ensures that labels are balanced and that the train/test split is consistent.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    pd_any: Any = pd
    plt_any: Any = plt
    sns_any: Any = sns

    sns_any.set_theme(style="whitegrid")

    df = pd_any.read_parquet(gold_parquet)

    label_map = {0: "Normal", 1: "Arbitrage", 2: "Sandwich"}
    df["label_name"] = df["label"].map(label_map)

    unique_txs = df.drop_duplicates(subset=["tx_hash"])
    total_unique = len(unique_txs)

    stats = unique_txs.groupby(["label_name", "split"]).size().unstack(fill_value=0)

    print("\n" + "═" * 50)
    print("📀 GOLD CLASS BALANCE & SPLIT REPORT")
    print("═" * 50)
    print(f"Total Unique Transactions: {total_unique:,}")
    print("-" * 50)
    print(stats.to_string())
    print("-" * 50)

    plt_any.figure(figsize=(10, 6))

    plot_df = unique_txs.groupby(["label_name", "split"]).size().reset_index(name="count")
    sns_any.barplot(data=plot_df, x="label_name", y="count", hue="split", palette="viridis")

    plt_any.title("Class Distribution by Dataset Split (Gold Layer)")
    plt_any.xlabel("Transaction Type")
    plt_any.ylabel("Unique Transaction Count")
    plt_any.legend(title="Split")

    balance_path = Path(output_dir) / "class_balance_split.png"
    plt_any.savefig(str(balance_path))
    plt_any.close()

    plt_any.figure(figsize=(12, 6))

    sns_any.histplot(
        data=unique_txs,
        x="block_number",
        hue="split",
        element="step",
        bins=100,
        palette={"train": "blue", "test": "orange"}
    )

    plt_any.title("Temporal Split Integrity: Block Number Distribution")
    plt_any.xlabel("Block Number")
    plt_any.ylabel("Tx Count")

    temporal_path = Path(output_dir) / "temporal_split_integrity.png"
    plt_any.savefig(str(temporal_path))
    plt_any.close()

    print(f"✅ Balance plot saved: {balance_path}")
    print(f"✅ Temporal plot saved: {temporal_path}")
    print("═" * 50 + "\n")

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/processed/gold_labeled.parquet"
    analyze_gold_balance(path)
