# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

def analyze_temporal_flow(silver_parquet: str, output_dir: str = "visualizations/structure") -> None:
    """Visualizes transaction density across blocks to identify competitive bursts."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    df: pd.DataFrame = pd.read_parquet(silver_parquet)

    # by block for density
    block_counts = df.groupby("block_number")["tx_hash"].nunique().reset_index()
    block_counts.columns = ["block_number", "tx_count"]

    print("\n" + "═" * 40)
    print("🕒 TEMPORAL FLOW DIAGNOSTICS")
    print("═" * 40)
    print(f"Blocks Covered:      {len(block_counts):,}")
    print(f"Avg Txs per Block:   {block_counts['tx_count'].mean():.2f}")
    print(f"Max Txs in a Block:  {block_counts['tx_count'].max():,}")
    print("-" * 40)

    # transaction density
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=block_counts, x="block_number", y="tx_count", color="blue")

    plt.title("Transaction Density per Block")
    plt.xlabel("Block Number")
    plt.ylabel("Unique Transactions")

    plot_path = Path(output_dir) / "temporal_density.png"
    plt.savefig(str(plot_path))
    plt.close()

    print(f"✅ Temporal plot saved to: {plot_path}")
    print("═" * 40 + "\n")
