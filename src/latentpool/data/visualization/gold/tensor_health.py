import logging
from pathlib import Path
from typing import Any, List, cast

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch_geometric.data import Data

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "Token In Count",
    "Token Out Count",
    "Transfers In",
    "Transfers Out",
    "Neighbor Count",
    "Log Vol In",
    "Log Vol Out",
    "Net Flow",
    "Is Source",
    "Is Sink",
    "Self Transfers"
]

def _generate_feature_reports(feat_matrix: np.ndarray[Any, Any], current_labels: List[str], output_dir: str) -> None:
    """Helper to handle reporting and visualization logic."""
    num_feats = feat_matrix.shape[1]

    print("\n" + "═" * 60)
    print("🧬 FINAL TENSOR FEATURE SCALING REPORT")
    print("═" * 60)
    print(f"Total Vectors Sampled: {feat_matrix.shape[0]:,}")
    print("-" * 60)

    print(f"{'Feat':<18} | {'Min':<10} | {'Max':<10} | {'Mean':<10} | {'Std':<10}")
    print("-" * 60)
    for i in range(num_feats):
        col = feat_matrix[:, i]
        name = current_labels[i]
        print(f"{name:<18} | {col.min():>10.4f} | {col.max():>10.4f} | {col.mean():>10.4f} | {col.std():>10.4f}")

    # Heatmap with tha Correlation Matrix
    plt.figure(figsize=(14, 12)) # type: ignore
    sns.set_theme(style="white")
    corr = np.corrcoef(feat_matrix.T)
    mask = np.triu(np.ones_like(corr, dtype=bool))

    sns.heatmap( # type: ignore
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        square=True,
        linewidths=.5,
        xticklabels=current_labels,
        yticklabels=current_labels,
        cbar_kws={"shrink": .8}
    )

    plt.xticks(rotation=45, ha='right') # type: ignore
    plt.title("Feature Correlation Matrix (MEV Graph Features)") # type: ignore
    corr_path = Path(output_dir) / "tensor_feature_correlation.png"
    plt.savefig(str(corr_path), bbox_inches='tight') # type: ignore
    plt.close()

    # the Boxplot is Feature Spread
    plt.figure(figsize=(16, 6)) # type: ignore
    sns.boxplot(data=feat_matrix, palette="Set3")
    plt.xticks(range(num_feats), current_labels, rotation=45) # type: ignore
    plt.title("Numerical Range per Feature (Scaling Verification)") # type: ignore
    plt.ylabel("Value") # type: ignore

    spread_path = Path(output_dir) / "tensor_feature_spread.png"
    plt.savefig(str(spread_path), bbox_inches='tight') # type: ignore
    plt.close()

    print("-" * 60)
    print(f"✅ Correlation heatmap saved: {corr_path}")
    print(f"✅ Feature spread boxplot saved: {spread_path}")
    print("═" * 60 + "\n")

def analyze_tensor_features(graphs_dir: str = "data/processed/graphs", output_dir: str = "visualizations/gold") -> None:
    """
    Analyzes the numerical distribution and correlation of the graph features.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    torch.serialization.add_safe_globals([Data]) # type: ignore

    sample_files = list(Path(graphs_dir).glob("*.pt"))[:200]
    if not sample_files:
        print(f"❌ No .pt files found in {graphs_dir}")
        return

    all_features: List[np.ndarray[Any, Any]] = []
    print(f"🧬 Sampling {len(sample_files)} tensors for feature distribution...")

    for f in sample_files:
        try:
            data = cast(Data, torch.load(str(f), map_location="cpu", weights_only=False))
            # Accessing 'x' directly as it's the standard PyG feature attribute
            target_attr = getattr(data, 'x', getattr(data, 'edge_attr', None))

            if target_attr is not None and isinstance(target_attr, torch.Tensor):
                feat_np = target_attr.numpy()
                if feat_np.ndim == 1:
                    feat_np = feat_np.reshape(1, -1)
                all_features.append(feat_np)
        except Exception as e:
            logger.warning(f"Could not load {f.name}: {e}")

    if not all_features:
        print("⚠️ No valid feature tensors found in samples.")
        return

    feat_matrix = np.vstack(all_features)
    num_feats = feat_matrix.shape[1]
    current_labels = FEATURE_NAMES[:num_feats] if num_feats <= len(FEATURE_NAMES) else [f"F{i}" for i in range(num_feats)]

    _generate_feature_reports(feat_matrix, current_labels, output_dir)
