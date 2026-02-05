from pathlib import Path
from typing import Any, Dict, List, cast

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def visualize_gold_parquet(parquet_path: str, output_dir: str = "visualizations/gold") -> None:
    df = cast(pd.DataFrame, pd.read_parquet(parquet_path))  # type: ignore
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    label_map: Dict[int, str] = {0: "Normal", 1: "Arbitrage", 2: "Sandwich"}

    labels_raw = df['label'].map(lambda x: label_map.get(cast(int, x), "Unknown"))  # type: ignore
    df['label_name'] = cast(pd.Series, labels_raw)

    cols: List[str] = ['tx_hash', 'label_name', 'split']
    gb_complexity = df.groupby(cols)  # type: ignore
    size_series = cast(pd.Series, gb_complexity.size())
    complexity = size_series.reset_index(name='edge_count')

    print("\n" + "═" * 60)
    print("📊 FINAL GOLD DATASET AGGREGATE REPORT")
    print("═" * 60)

    gb_split = complexity.groupby(['label_name', 'split'])  # type: ignore
    split_counts = cast(pd.Series, gb_split.size())
    stats = split_counts.unstack(fill_value=0)

    print("--- Distribution by Label & Split ---")
    print(stats)
    print("-" * 60)

    # Complexity
    gb_comp = cast(Any, complexity.groupby('label_name'))['edge_count'] # type: ignore
    comp_agg = cast(pd.DataFrame, gb_comp.agg(['mean', 'std', 'min', 'max', 'median']))

    print("--- Structural Complexity (Edges per Tx) ---")
    print(comp_agg.round(2))  # type: ignore
    print("-" * 60)

    sns.set_theme(style="whitegrid")

    # subplots typing is inconsistent in stubs
    fig_res = plt.subplots(1, 2, figsize=(15, 6))  # type: ignore
    _, axes = cast(tuple[Any, Any], fig_res)

    sns.boxplot(ax=axes[0], x='label_name', y='edge_count', data=complexity, palette="magma", showfliers=False)
    axes[0].set_title("Edge Complexity by Label")

    sns.countplot(ax=axes[1], x='label_name', hue='split', data=complexity, palette="viridis")
    axes[1].set_title("Train/Test Class Balance")

    plot_path = Path(output_dir) / "gold_parquet_summary.png"
    plt.savefig(plot_path, bbox_inches='tight')  # type: ignore
    plt.close()

    print(f"✅ Summary plot saved to: {plot_path}")
    print("═" * 60 + "\n")
