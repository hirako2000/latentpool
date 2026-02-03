import logging
from pathlib import Path
from typing import Any, Dict, Optional, cast

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch

logger = logging.getLogger(__name__)


class IngestionValidator:
    def __init__(self, silver_parquet: str):
        self.df: pd.DataFrame = pd.read_parquet(silver_parquet) # type: ignore[reportUnknownMemberType]

    def print_aggregates(self, gold_path: Optional[str] = None) -> None:
        """Combined summary of Silver (Tokens/Edges) and Gold (Labels)."""
        unique_txs = int(self.df["tx_hash"].nunique()) # type: ignore[reportUnknownMemberType]
        unique_tokens = int(self.df["token"].nunique()) # type: ignore[reportUnknownMemberType]
        total_edges = len(self.df)

        print("\n" + "═" * 40)
        print("📊 DATASET AGGREGATE SUMMARY")
        print("═" * 40)
        print(f"Total Transfers (Edges): {total_edges:,}")
        print(f"Unique Transactions:    {unique_txs:,}")
        print(f"Unique Tokens:          {unique_tokens:,}")
        print(f"Avg Transfers per TX:   {total_edges / unique_txs:.2f}")

        if gold_path and Path(gold_path).exists():
            gold_df: pd.DataFrame = pd.read_parquet(gold_path) # type: ignore[reportUnknownMemberType]
            unique_labels = gold_df.drop_duplicates("tx_hash")

            # 1. Get the counts
            # 2. Convert to dict
            # 3. Cast to Dict[int, int] so Pyright knows the inputs to int() are valid
            counts_raw = unique_labels["label"].value_counts().to_dict() # type: ignore[reportUnknownMemberType]
            label_counts = cast(Dict[int, int], counts_raw)

            # Now Pyright knows these are ints, but we use .get() for safety
            normal_count = int(label_counts.get(0, 0))
            arb_count = int(label_counts.get(1, 0))
            sand_count = int(label_counts.get(2, 0))

            print("-" * 40)
            print("🏷️  LABEL DISTRIBUTION (Ground Truth):")
            print(f"  Normal:    {normal_count:>10,}")
            print(f"  Arbitrage: {arb_count:>10,}")
            print(f"  Sandwich:  {sand_count:>10,}")

            missing = unique_txs - (normal_count + arb_count + sand_count)
            if missing > 0:
                print(f"  ⚠️  Unlabeled: {missing:>9,} (Pending Labeler run)")
        else:
            print("-" * 40)
            print("🏷️  LABEL DISTRIBUTION:")
            print("  [!] Gold layer not found. Run 'label' command to assign classes.")

        print("-" * 40)
        print("🔝 TOP 5 MOST ACTIVE TOKENS:")
        token_counts: Any = self.df["token"].value_counts().head(5) # type: ignore[reportUnknownMemberType]
        print(str(token_counts.to_string())) # type: ignore[reportUnknownMemberType]
        print("═" * 40 + "\n")

    def generate_transformation_plots(self) -> None:
        """Basic Silver-level structural plots."""
        out_dir = Path("visualizations/transformation")
        out_dir.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(10, 6)) # type: ignore[reportUnknownMemberType]
        edge_counts: Any = self.df.groupby("tx_hash").size() # type: ignore[reportUnknownMemberType]
        sns.histplot(data=edge_counts, bins=30, kde=True, color="teal")
        plt.title("Transfers per Transaction") # type: ignore[reportUnknownMemberType]
        plt.xlabel("Edge Count") # type: ignore[reportUnknownMemberType]
        plt.savefig(str(out_dir / "edge_distribution.png")) # type: ignore[reportUnknownMemberType]
        plt.close()

        plt.figure(figsize=(10, 6)) # type: ignore[reportUnknownMemberType]
        top_tokens: Any = self.df["token"].value_counts().head(10) # type: ignore[reportUnknownMemberType]
        top_tokens.plot(kind="barh", color="orange") # type: ignore[reportUnknownMemberType]
        plt.title("Top 10 Tokens") # type: ignore[reportUnknownMemberType]
        plt.tight_layout() # type: ignore[reportUnknownMemberType]
        plt.savefig(str(out_dir / "token_usage.png")) # type: ignore[reportUnknownMemberType]
        plt.close()

    def generate_gold_plots(self, gold_parquet: str) -> None:
        """Visualizes class distribution after labeling."""
        out_dir = Path("visualizations/labeling")
        out_dir.mkdir(parents=True, exist_ok=True)

        gold_df: pd.DataFrame = pd.read_parquet(gold_parquet) # type: ignore[reportUnknownMemberType]
        unique_txs = gold_df.drop_duplicates("tx_hash").copy()

        class_map: Dict[int, str] = {0: "Normal", 1: "Arb", 2: "Sand"}
        unique_txs["class_name"] = unique_txs["label"].map(class_map) # type: ignore[reportUnknownMemberType]

        plt.figure(figsize=(8, 6)) # type: ignore[reportUnknownMemberType]
        sns.countplot(
            data=unique_txs, x="class_name", hue="class_name", palette="magma", legend=False
        )
        plt.title("Class Distribution") # type: ignore[reportUnknownMemberType]
        plt.savefig(str(out_dir / "class_balance.png")) # type: ignore[reportUnknownMemberType]
        plt.close()

    def generate_tensor_stats(self, graphs_dir: str) -> None:
        """Samples graph complexity from generated .pt files."""
        out_dir = Path("visualizations/geometry")
        out_dir.mkdir(parents=True, exist_ok=True)

        files = [f for f in Path(graphs_dir).glob("*.pt") if f.is_file()]
        if not files:
            return

        nodes_list: list[int] = []
        edges_list: list[int] = []
        for f in files[:500]:
            try:
                graph_data: Any = torch.load(str(f), weights_only=False)
                nodes_list.append(int(graph_data.x.size(0)))
                edges_list.append(int(graph_data.edge_index.size(1)))
            except Exception as e:
                logger.error("Failed to load graph %s: %s", f, e)
                continue

        if nodes_list:
            plt.figure(figsize=(10, 5)) # type: ignore[reportUnknownMemberType]
            plt.hist(nodes_list, bins=20, alpha=0.5, label="Nodes", color="blue") # type: ignore[reportUnknownMemberType]
            plt.hist(edges_list, bins=20, alpha=0.5, label="Edges", color="orange") # type: ignore[reportUnknownMemberType]
            plt.legend() # type: ignore[reportUnknownMemberType]
            plt.title("Geometric Complexity (Sample)") # type: ignore[reportUnknownMemberType]
            plt.savefig(str(out_dir / "tensor_complexity.png")) # type: ignore[reportUnknownMemberType]
            plt.close()
