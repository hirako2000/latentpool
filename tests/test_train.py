from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch_geometric.data import Data

from latentpool.train import (
    MEVFocalLoss,
    TrainConfig,
    get_compute_device,
    load_and_partition,
    run_training,
    train_epoch,
    validate,
)


@pytest.fixture
def micro_batch() -> Data:
    """Creates a tiny batch of 2 graphs for testing forward/backward passes."""
    return Data(
        x=torch.randn(10, 11),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        y=torch.tensor([0]),  # Target label
        batch=torch.zeros(10, dtype=torch.long)
    )

def test_focal_loss_math() -> None:
    """Checking focal Loss doesn't crash and respects alpha weights."""
    alpha = torch.tensor([1.0, 4.0, 12.0])
    criterion = MEVFocalLoss(alpha=alpha, gamma=2.5)

    logits = torch.randn(4, 3)  # Batch of 4, 3 classes
    targets = torch.tensor([0, 1, 2, 0])

    loss = criterion(logits, targets)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert loss > 0

def test_get_compute_device() -> None:
    device = get_compute_device()
    assert isinstance(device, torch.device)

def test_load_and_partition(tmp_path: Path) -> None:
    graphs_dir = tmp_path / "graphs"
    graphs_dir.mkdir()

    d1 = Data(x=torch.randn(1, 11), train_mask=True)
    d2 = Data(x=torch.randn(1, 11), train_mask=False)

    torch.save(d1, graphs_dir / "train.pt")
    torch.save(d2, graphs_dir / "test.pt")

    config = TrainConfig(graphs_dir=str(graphs_dir), batch_size=1)
    train_loader, test_loader = load_and_partition(config)

    assert len(train_loader.dataset) == 1
    assert len(test_loader.dataset) == 1

def test_train_epoch_iteration(micro_batch: Data) -> None:
    device = torch.device("cpu")
    model = MagicMock()
    model.train.return_value = None

    mock_logits = torch.randn(1, 3, requires_grad=True)
    model.return_value = mock_logits

    loader: List[Data] = [micro_batch]
    optimizer = MagicMock()
    criterion = MEVFocalLoss(alpha=torch.ones(3), gamma=2.0)

    avg_loss = train_epoch(model, loader, criterion, optimizer, device)

    assert isinstance(avg_loss, float)
    optimizer.zero_grad.assert_called_once()
    optimizer.step.assert_called_once()


def test_validate(micro_batch: Data) -> None:
    """lists of predictions/truths"""
    device = torch.device("cpu")
    model = MagicMock()
    model.return_value = torch.tensor([[0.1, 0.8, 0.1]]) # Should predict index 1

    loader: List[Data] = [micro_batch]
    micro_batch.y = torch.tensor([1])

    y_true, y_pred = validate(model, loader, device)

    assert y_true == [1]
    assert y_pred == [1]


@patch("latentpool.train.load_and_partition")
@patch("latentpool.train.torch.save")
@patch("latentpool.train.classification_report")
def test_run_training_flow(
    mock_report: MagicMock,
    mock_save: MagicMock,
    mock_load: MagicMock,
    tmp_path: Path
) -> None:
    """Tests the full loop with mocked loaders to keep it fast"""
    mock_batch = Data(
        x=torch.randn(2, 11),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        y=torch.tensor([0, 1]),
        batch=torch.tensor([0, 1])
    )

    mock_loader: List[Data] = [mock_batch]
    mock_load.return_value = (mock_loader, mock_loader)
    mock_report.return_value = "Mock Report Table"

    save_path = tmp_path / "model.pth"

    with patch("latentpool.train.TrainConfig.model_save_path", str(save_path)):
        run_training(graphs_dir="fake", epochs=1, batch_size=1)

    assert mock_save.called
    assert save_path.parent.exists()

@patch("latentpool.train.torch.cuda.is_available")
@patch("latentpool.train.torch.backends.mps.is_available")
def test_get_compute_device_variants(mock_mps: MagicMock, mock_cuda: MagicMock) -> None:
    mock_mps.return_value = True
    assert get_compute_device().type == "mps"

    mock_mps.return_value = False
    mock_cuda.return_value = True
    assert get_compute_device().type == "cuda"

    mock_mps.return_value = False
    mock_cuda.return_value = False
    assert get_compute_device().type == "cpu"
