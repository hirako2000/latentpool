import concurrent.futures
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast

import torch
import torch.nn.functional as F
import typer
from sklearn.metrics import classification_report  # type: ignore
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from latentpool.models.sage import MEVGraphSAGE

PyGDataLoader = Any

@dataclass
class ModelConfig:
    in_channels: int = 11
    hidden_channels: int = 512
    out_channels: int = 3
    mlp_layer_1_dim: int = 128
    mlp_layer_2_dim: int = 64
    dropout_rate: float = 0.2
    output_temperature: float = 1.5

@dataclass
class TrainConfig:
    graphs_dir: str = "data/processed/graphs"
    model_save_path: str = "data/models/sage_mev_genesis_v3_3.pth"
    epochs: int = 10
    batch_size: int = 512
    learning_rate: float = 0.0003
    weight_decay: float = 1e-5
    focal_gamma: float = 2.5
    importance_weights: List[float] = field(default_factory=lambda: [1.0, 4.0, 12.0])
    lr_reduction_factor: float = 0.5
    lr_patience: int = 2

class MEVFocalLoss(nn.Module):
    def __init__(self, alpha: Tensor, gamma: float):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce_loss)
        res = (1 - pt) ** self.gamma * ce_loss
        return res.mean()

def get_compute_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def load_and_partition(config: TrainConfig) -> Tuple[PyGDataLoader, PyGDataLoader]:
    all_files = list(Path(config.graphs_dir).glob("*.pt"))
    typer.echo(f"📦 Loading {len(all_files)} tensors...")

    def _load(f: Path) -> Data:
        return cast(Data, torch.load(f, map_location="cpu", weights_only=False))

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        dataset: List[Data] = list(
            tqdm(executor.map(_load, all_files), total=len(all_files), desc="📥 Dataset")
        )

    train_data = [d for d in dataset if getattr(d, "train_mask", False)]
    test_data = [d for d in dataset if not getattr(d, "train_mask", False)]

    return (
        DataLoader(train_data, batch_size=config.batch_size, shuffle=True),
        DataLoader(test_data, batch_size=config.batch_size),
    )

def train_epoch(
    model: nn.Module,
    loader: PyGDataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device
) -> float:
    model.train()
    total_loss: float = 0.0
    for data in tqdm(loader, leave=False, desc="Training"):
        batch = data.to(device)
        optimizer.zero_grad()
        logits: Tensor = model(batch.x, batch.edge_index.long(), batch.batch)
        loss: Tensor = criterion(logits, batch.y)

        cast(Any, loss).backward()

        optimizer.step()
        total_loss += float(loss.item())
    return total_loss / len(loader)

def validate(
    model: nn.Module,
    loader: PyGDataLoader,
    device: torch.device
) -> Tuple[List[int], List[int]]:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    with torch.no_grad():
        for data in loader:
            batch = data.to(device)
            logits: Tensor = model(batch.x, batch.edge_index.long(), batch.batch)

            preds: List[int] = cast(Any, logits.argmax(dim=1).detach().cpu()).tolist()
            actuals: List[int] = batch.y.detach().cpu().tolist()

            y_pred.extend(preds)
            y_true.extend(actuals)
    return y_true, y_pred

def run_training(
    graphs_dir: str = "data/processed/graphs", epochs: int = 10, batch_size: int = 512
) -> None:
    m_cfg = ModelConfig()
    t_cfg = TrainConfig(graphs_dir=graphs_dir, epochs=epochs, batch_size=batch_size)

    device = get_compute_device()
    typer.secho(f"Training on {device}", fg=typer.colors.CYAN)

    train_loader, test_loader = load_and_partition(t_cfg)
    model = MEVGraphSAGE(m_cfg).to(device)

    alpha = torch.tensor(t_cfg.importance_weights).to(device)
    criterion = MEVFocalLoss(alpha=alpha, gamma=t_cfg.focal_gamma)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=t_cfg.learning_rate, weight_decay=t_cfg.weight_decay
    )
    scheduler = ReduceLROnPlateau(
        optimizer, factor=t_cfg.lr_reduction_factor, patience=t_cfg.lr_patience
    )

    for epoch in range(1, t_cfg.epochs + 1):
        loss = train_epoch(model, train_loader, criterion, optimizer, device)
        y_true, y_pred = validate(model, test_loader, device)

        cast(Any, scheduler).step(loss)

        typer.secho(f"\n📊 Epoch {epoch:02d} Metrics", fg=typer.colors.MAGENTA)

        report = cast(Any, classification_report(
            y_true, y_pred, target_names=["Normal", "Arb", "Sandwich"], zero_division="0"
        ))

        print(report)

        pred_dist: Dict[int, int] = dict(sorted(Counter(y_pred).items()))
        typer.echo(f"Pred Dist: {pred_dist}")

    Path(t_cfg.model_save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), t_cfg.model_save_path)
    typer.secho(f"💾 Model archived to {t_cfg.model_save_path}", fg=typer.colors.GREEN)

if __name__ == "__main__":
    typer.run(run_training)
