import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

def analyze_feature_health(silver_parquet: str, output_dir: str = "visualizations/structure") -> None:
    """Checks the distribution of transfer values to detect scaling issues or data gaps."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    pd_any: Any = pd
    plt_any: Any = plt
    sns_any: Any = sns
    np_any: Any = np

    sns_any.set_theme(style="whitegrid")

    df = pd_any.read_parquet(silver_parquet)

    # ensure value is numeric
    df["value"] = pd_any.to_numeric(df["value"], errors="coerce").fillna(0)

    # zeros vs non-zeros
    total_edges = int(len(df))
    zero_values = int(len(df[df["value"] == 0]))

    # filtered series in Any to handle .empty and statistical methods
    non_zero_series: Any = df[df["value"] > 0]["value"]

    print("\n" + "═" * 40)
    print("🌡️  FEATURE HEALTH DIAGNOSTICS")
    print("═" * 40)
    print(f"Total Edges Analyzed: {total_edges:,}")

    zero_perc = (zero_values / total_edges) * 100 if total_edges > 0 else 0
    print(f"Zero-Value Edges:    {zero_values:,} ({zero_perc:.2f}%)")

    if not non_zero_series.empty:
        v_min = float(non_zero_series.min())
        v_max = float(non_zero_series.max())
        v_med = float(non_zero_series.median())
        print(f"Min Non-Zero Value:  {v_min:.6f}")
        print(f"Max Value:           {v_max:,.2f}")
        print(f"Median Value:        {v_med:.6f}")
    print("-" * 40)

    # Log-scale distribution of non-zero values
    if not non_zero_series.empty:
        plt_any.figure(figsize=(10, 6))

        # log10 to handle the massive range of ETH/Token values
        log_values = np_any.log10(non_zero_series)
        sns_any.histplot(log_values, bins=50, kde=True, color="gold")

        plt_any.title("Distribution of Transfer Magnitudes (Log10 Scale)")
        plt_any.xlabel("Value Magnitude (10^x)")
        plt_any.ylabel("Frequency")

        plot_path = Path(output_dir) / "value_distribution.png"
        plt_any.savefig(str(plot_path))
        plt_any.close()
        print(f"✅ Feature health plot saved to: {plot_path}")
    else:
        print("⚠️ No non-zero values found to plot.")
    print("═" * 40 + "\n")
