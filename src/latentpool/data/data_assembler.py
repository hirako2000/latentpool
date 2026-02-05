import re
from enum import IntEnum
from pathlib import Path
from typing import Any, List, Set, cast

import numpy as np
import pandas as pd
import typer


class MEVLabel(IntEnum):
    NORMAL = 0
    ARBITRAGE = 1
    SANDWICH = 2


class DatasetAssembler:
    """
    Assembles the Gold Layer from Silver edges + CSV Labels.
    Enforces class balancing and temporal splitting.
    """

    def __init__(self, edges_path: str, arb_csv: str, sand_csv: str):
        self.edges_path = Path(edges_path)
        self.arb_csv = Path(arb_csv)
        self.sand_csv = Path(sand_csv)
        self.hash_pattern = re.compile(r"0x[a-fA-F0-9]{64}")

    def _extract_hashes(self, path: Path) -> Set[str]:
        found: Set[str] = set()
        if not path.exists():
            return found

        with open(path, "r") as f:
            for line in f:
                for m in self.hash_pattern.findall(line):
                    found.add(m.lower())
        return found

    def run(self, output_path: str, train_ratio: float = 0.8, max_normal_ratio: float = 2.0):
        if not self.edges_path.exists():
            raise FileNotFoundError(f"No Silver data at {self.edges_path}")

        typer.echo("📀 Assembling Gold Dataset...")

        df = cast(pd.DataFrame, cast(Any, pd).read_parquet(self.edges_path))
        df_any: Any = df

        arb_h = self._extract_hashes(self.arb_csv)
        sand_h = self._extract_hashes(self.sand_csv)

        def get_label(tx_hash: Any) -> int:
            h = str(tx_hash).lower()
            if h in arb_h:
                return MEVLabel.ARBITRAGE.value
            if h in sand_h:
                return MEVLabel.SANDWICH.value
            return MEVLabel.NORMAL.value

        df["label"] = df_any["tx_hash"].apply(get_label)

        mev_count = int(df_any[df["label"] != MEVLabel.NORMAL.value]["tx_hash"].nunique())
        normal_txs: Any = df_any[df["label"] == MEVLabel.NORMAL.value]["tx_hash"].unique()

        limit = int(mev_count * max_normal_ratio)

        if len(normal_txs) > limit > 0:
            typer.secho(
                f"⚖️  Downsampling Normal transactions to {limit} for balance.",
                fg=typer.colors.CYAN,
            )
            keep_normals: Any = np.random.choice(normal_txs, limit, replace=False)

            mask = (df["label"] != MEVLabel.NORMAL.value) | (df_any["tx_hash"].isin(keep_normals))  # type: ignore
            df = cast(pd.DataFrame, df[mask])  # type: ignore

        unique_blocks_raw: Any = df_any["block_number"].unique()
        unique_blocks: List[Any] = sorted(unique_blocks_raw)

        if not unique_blocks:
            typer.secho("❌ No blocks found in silver data.", fg=typer.colors.RED)
            return

        split_idx = max(1, int(len(unique_blocks) * train_ratio))
        cutoff = unique_blocks[split_idx - 1]

        np_any: Any = np
        df["split"] = np_any.where(df["block_number"] <= cutoff, "train", "test")

        df_any.to_parquet(output_path, index=False)
        self._print_report(df, output_path)

    def _print_report(self, df: pd.DataFrame, path: str):
        label_map = {0: "Normal", 1: "Arb", 2: "Sand"}
        df_any: Any = df

        stats = df_any.groupby("label")["tx_hash"].nunique().rename(index=label_map)
        splits = df_any.groupby("split")["tx_hash"].nunique()

        print("\n" + "═" * 50)
        print(f"✅ GOLD DATASET READY: {path}")
        print("-" * 50)
        print(f"Unique Transactions:\n{stats.to_string()}")
        print(f"\nTemporal Split:\n{splits.to_string()}")
        print("═" * 50 + "\n")
