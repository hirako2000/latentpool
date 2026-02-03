import re
from pathlib import Path
from typing import Any, Set

import pandas as pd
import typer


class Labeler:
    def __init__(self, edges_path: str, arb_csv: str, sand_csv: str):
        self.edges_path = Path(edges_path)
        self.arb_csv = Path(arb_csv)
        self.sand_csv = Path(sand_csv)
        self.hash_pattern = re.compile(r'0x[a-fA-F0-9]{64}')

    def _extract_hashes_from_file(self, path: Path) -> Set[str]:
        """Reuse the clever regex logic to bypass malformed CSV issues."""
        found_hashes: Set[str] = set()
        if not path.exists():
            typer.secho(f"⚠️  Label file not found: {path}", fg=typer.colors.YELLOW)
            return found_hashes

        try:
            with open(path, 'r') as f:
                # line-by-line reading for large label files
                for line in f:
                    matches = self.hash_pattern.findall(line)
                    for m in matches:
                        found_hashes.add(m.lower())
            return found_hashes
        except Exception as e:
            typer.secho(f"❌ Failed to mine hashes from {path.name}: {e}", fg=typer.colors.RED)
            return found_hashes

    def run(self, output_path: str):
        if not self.edges_path.exists():
            raise FileNotFoundError(f"Missing Silver data at {self.edges_path}")

        typer.echo("📊 Loading silver edges...")
        df: Any = pd.read_parquet(self.edges_path) # type: ignore

        typer.echo("⛏️  Mining MEV hashes from labels...")
        arb_h = self._extract_hashes_from_file(self.arb_csv)
        sand_h = self._extract_hashes_from_file(self.sand_csv)

        # Apply Multi-Class Labels
        # todo: refactor to enums
        def get_label(h: Any) -> int:
            h_low = str(h).lower()
            if h_low in arb_h:
                return 1  # Arbitrage
            if h_low in sand_h:
                return 2  # Sandwich
            return 0      # Normal

        typer.echo("🏷️  Mapping labels (0: Normal, 1: Arb, 2: Sand)...")
        df['label'] = df['tx_hash'].map(get_label) # type: ignore

        # Temporal Split (80% Train / 20% Test by Block Number)
        typer.echo("📅 Performing temporal split by block number...")
        unique_blocks: Any = sorted(df['block_number'].unique())
        split_idx = int(len(unique_blocks) * 0.8)
        train_blocks = set(unique_blocks[:split_idx])

        df['split'] = df['block_number'].map(
            lambda x: 'train' if x in train_blocks else 'test' # type: ignore
        ) # type: ignore

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False) # type: ignore

        print("\n" + "═"*50)
        print("🏷️  LABELING COMPLETE (GOLD LAYER)")
        print("═"*50)

        label_stats: Any = df['label'].value_counts().rename({0: 'Normal', 1: 'Arb', 2: 'Sand'}) # type: ignore
        split_stats: Any = df['split'].value_counts() # type: ignore

        print(f"Distribution:\n{label_stats}")
        print(f"\nSplit Summary:\n{split_stats}")
        print(f"\n📍 Saved to: {output_path}")
        print("═"*50 + "\n")
