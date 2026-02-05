from pathlib import Path
from typing import Any, Callable

import pytest
import torch
import typer

from latentpool.models.sage import MEVGraphSAGE
from latentpool.train import ModelConfig, TrainConfig, get_compute_device


@pytest.mark.benchmark(group="gnn-inference")
def test_mev_sage_inference_latency(benchmark: Callable[..., Any]) -> None:
    m_cfg = ModelConfig()
    t_cfg = TrainConfig()
    device = get_compute_device()

    model = MEVGraphSAGE(m_cfg).to(device)

    typer.echo("\n" + "="*50)
    typer.secho("🚀 BENCHMARKING MODEL ARCHITECTURE", fg=typer.colors.CYAN, bold=True)
    print(model) # Prints the layer-by-layer architecture

    if Path(t_cfg.model_save_path).exists():
        typer.secho(f"📦 Loading weights from: {t_cfg.model_save_path}", fg=typer.colors.GREEN)
        model.load_state_dict(torch.load(t_cfg.model_save_path, map_location=device))
    else:
        typer.secho("⚠️ No saved weights found. Benchmarking raw architecture (Cold Start).", fg=typer.colors.YELLOW)

    typer.secho(f"💻 Device: {device}", fg=typer.colors.MAGENTA)
    typer.echo("="*50 + "\n")

    model.eval()
    x = torch.randn(50, m_cfg.in_channels).to(device)
    edge_index = torch.randint(0, 50, (2, 150)).to(device)
    batch = torch.zeros(50, dtype=torch.long).to(device)

    with torch.no_grad():
        benchmark(model, x, edge_index, batch)


    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2

    typer.echo(f"🔢 Total Parameters: {total_params:,}")
    typer.echo(f"🛠️  Trainable Parameters: {trainable_params:,}")
    typer.secho(f"💾 Model Size: {size_all_mb:.2f} MB", fg=typer.colors.BLUE)
