import asyncio
import os
import traceback
from pathlib import Path
from typing import TypedDict

import typer
from dotenv import load_dotenv

from latentpool.configs.data_config import ArchiveConfig, DataConfig
from latentpool.data.data_assembler import DatasetAssembler
from latentpool.data.ingestion.coordinator import IngestionCoordinator
from latentpool.data.processing import GraphBuilder
from latentpool.data.transformation.parquet_exporter import ParquetExporter
from latentpool.data.visualization.dashboard import IngestionValidator
from latentpool.data.visualization.explorer import GraphExplorer
from latentpool.data.visualization.gold.balance import analyze_gold_balance
from latentpool.data.visualization.gold.complexity import analyze_label_complexity
from latentpool.data.visualization.gold.tensor_health import analyze_tensor_features
from latentpool.data.visualization.gold_summary import visualize_gold_parquet
from latentpool.data.visualization.silver.feature_health import analyze_feature_health
from latentpool.data.visualization.silver.graph_structure import analyze_structure
from latentpool.data.visualization.silver.node_diversity import analyze_node_diversity
from latentpool.data.visualization.silver.path_depth import analyze_path_depth
from latentpool.data.visualization.silver.temporal import analyze_temporal_flow
from latentpool.train import run_training

load_dotenv()

DEFAULT_MAX_RPS = 15
DEFAULT_BLOCK_NUMBER = 19000000
DEFAULT_TRAIN_EPOCHS = 40
COL_WIDTH_TYPE = 12
COL_WIDTH_DATA = 10

class CoverageStats(TypedDict):
    Type: str
    Total: int
    Found: int
    Missing: int
    Coverage: str

app = typer.Typer(help="LatentPool: MEV Detection using GNNs")

# data group
data_app = typer.Typer(help="Data lifecycle: Ingestion, Preparation, and Processing.")
app.add_typer(data_app, name="data")

# viz group
viz_app = typer.Typer(help="Visualization and diagnostic tools.")
app.add_typer(viz_app, name="viz")

# viz silver sub-group
silver_viz_app = typer.Typer(help="Diagnostics for the Silver (unlabeled) layer.")
viz_app.add_typer(silver_viz_app, name="silver")

# viz gold sub-group
gold_viz_app = typer.Typer(help="Diagnostics for the Gold (labeled) layer.")
viz_app.add_typer(gold_viz_app, name="gold")

def get_coordinator(max_rps: int = DEFAULT_MAX_RPS):
    rpc_url_template = os.getenv("ALCHEMY_RPC_URL", "")
    api_key = os.getenv("ALCHEMY_API_KEY", "")
    full_rpc_url = f"{rpc_url_template}{api_key}"

    config = DataConfig(
        archive_node=ArchiveConfig(rpc_url=full_rpc_url, max_rps=max_rps),
        raw_dir="data/raw/traces"
    )

    if not api_key or not rpc_url_template:
        typer.secho("❌ ALCHEMY_API_KEY or ALCHEMY_RPC_URL missing in .env", fg=typer.colors.RED)
        raise typer.Exit(1)

    return IngestionCoordinator(config=config)

# --- 📂 DATA COMMANDS ---

@data_app.command("hydrate")
def hydrate(
    arbitrage_csv: str = typer.Option("data/labeled_arbitrage.csv", help="Path to arb labels"),
    sandwich_csv: str = typer.Option("data/labeled_sandwich.csv", help="Path to sandwich labels"),
    silver_path: str = typer.Option("data/processed/edges.parquet", help="Path to silver data for sampling"),
    max_rps: int = DEFAULT_MAX_RPS,
    sample_normals: bool = typer.Option(True, help="Whether to sample 'Normal' txs from blocks")
):
    """Phase 1 & 2: Fetch MEV receipts and sample Normal sibling transactions."""
    csv_paths = [arbitrage_csv, sandwich_csv]
    for p in csv_paths:
        if not Path(p).exists():
            typer.secho(f"❌ Label CSV missing: {p}", fg=typer.colors.RED)
            raise typer.Exit(1)

    coordinator = get_coordinator(max_rps)

    async def _execute():
        typer.echo("📖 Step 1: Hydrating MEV from CSVs...")
        mev_count = await coordinator.run_hydration(csv_paths)
        typer.secho(f"✅ MEV Hydration complete: {mev_count} files.", fg=typer.colors.CYAN)

        if sample_normals:
            if not Path(silver_path).exists():
                typer.secho(f"⚠️ Silver data not found at {silver_path}. Run 'data prepare' before sampling normals.", fg=typer.colors.YELLOW)
                return

            typer.echo("🧬 Step 2: Sampling 'Normal' transactions from sibling blocks...")
            normal_count = await coordinator.run_negative_sampling(silver_path, target_count=mev_count)
            typer.secho(f"✅ Normal Sampling complete: {normal_count} files.", fg=typer.colors.GREEN)

    asyncio.run(_execute())

@data_app.command("check")
def hydrate_check(
    arbitrage_csv: str = typer.Option("data/labeled_arbitrage.csv"),
    sandwich_csv: str = typer.Option("data/labeled_sandwich.csv")
):
    """Check current data coverage including MEV and sampled Normal data."""
    coordinator = get_coordinator(max_rps=1)
    raw_dir = Path(coordinator.config.raw_dir)
    downloaded = {f.stem.lower() for f in raw_dir.glob("*.json")}

    arb_hashes = set(coordinator.extract_hashes([arbitrage_csv]))
    sand_hashes = set(coordinator.extract_hashes([sandwich_csv]))

    arb_found = len([h for h in arb_hashes if h in downloaded])
    sand_found = len([h for h in sand_hashes if h in downloaded])
    normal_found = len(downloaded) - arb_found - sand_found

    typer.echo("\n" + "="*45)
    typer.echo("🧪 INGESTION HEALTH REPORT")
    typer.echo("="*45)
    typer.echo(f"{'Type':<15} | {'Found':<10} | {'CSV Total':<10}")
    typer.echo("-" * 45)
    typer.echo(f"{'Arbitrage':<15} | {arb_found:<10} | {len(arb_hashes):<10}")
    typer.echo(f"{'Sandwich':<15} | {sand_found:<10} | {len(sand_hashes):<10}")
    typer.echo(f"{'Normal':<15} | {normal_found:<10} | {'N/A':<10}")
    typer.echo("-" * 45)
    typer.echo(f"Total Files on Disk: {len(downloaded):,}")
    typer.echo("="*45 + "\n")

@data_app.command("prepare")
def prepare(
    raw_dir: str = "data/raw/traces",
    processed_dir: str = "data/processed"
):
    """Transform raw JSON receipts into structured Parquet"""
    typer.echo(f"🏗️  Starting transformation from {raw_dir}...")
    exporter = ParquetExporter(raw_dir=raw_dir, processed_dir=processed_dir)
    exporter.process_all()
    typer.secho(f"✅ Parquet files ready in {processed_dir}", fg=typer.colors.GREEN)

@data_app.command("label")
def label(
    train_ratio: float = typer.Option(0.8, help="Ratio of blocks for training"),
    normal_ratio: float = typer.Option(2.0, help="Max ratio of Normal vs MEV for balancing")
):
    """
    Create the labelled Gold training set.
    Includes class balancing and temporal splitting.
    """
    # Hardcoded paths to satisfy linting (limit function arguments)
    edges_parquet = "data/processed/edges.parquet"
    arbitrage_csv = "data/labeled_arbitrage.csv"
    sandwich_csv = "data/labeled_sandwich.csv"
    output_path = "data/processed/gold_labeled.parquet"

    assembler = DatasetAssembler(
        edges_path=edges_parquet,
        arb_csv=arbitrage_csv,
        sand_csv=sandwich_csv
    )
    assembler.run(output_path, train_ratio=train_ratio, max_normal_ratio=normal_ratio)

@data_app.command("process")
def process(
    gold_parquet: str = typer.Option("data/processed/gold_labeled.parquet", help="Source gold data"),
    output_dir: str = typer.Option("data/processed/graphs", help="Directory for .pt tensors")
):
    """
    Convert Gold Parquet into PyTorch Geometric tensors.
    Also applies Symmetric Log Scaling to volumes.
    """
    typer.echo(f"🧬 Constructing graph tensors in {output_dir}...")
    builder = GraphBuilder(gold_parquet)
    builder.build_and_save(output_dir)

# --- 🥈 VIZ SILVER COMMANDS ---

@silver_viz_app.command("structure")
def viz_structure(
    silver_path: str = typer.Option("data/processed/edges.parquet", help="Path to silver parquet"),
    output_dir: str = typer.Option("visualizations/structure", help="Folder for plots"),
    no_plot: bool = typer.Option(False, "--no-plot", help="Only output metrics to terminal")
):
    """
    Possible 'Black Holes' (outliers) and graph connectivity
    in the Silver layer before GNN processing.
    """
    if not Path(silver_path).exists():
        typer.secho(f"❌ Silver data missing at {silver_path}", fg=typer.colors.RED)
        raise typer.Exit(1)

    typer.echo(f"🔍 Analyzing graph topology for {silver_path}...")
    analyze_structure(silver_parquet=silver_path, output_dir=output_dir if not no_plot else "")

    if not no_plot:
        typer.secho(f"✅ Structural diagnostics complete. Plot saved to {output_dir}", fg=typer.colors.GREEN)
    else:
        typer.secho("✅ Metrics calculation complete.", fg=typer.colors.CYAN)

@silver_viz_app.command("depth")
def viz_path_depth(
    silver_path: str = "data/processed/edges.parquet",
    output_dir: str = "visualizations/structure"
):
    """Depth of swap sequences in the Silver layer."""
    if not Path(silver_path).exists():
        typer.secho(f"❌ Silver data missing at {silver_path}", fg=typer.colors.RED)
        raise typer.Exit(1)
    analyze_path_depth(silver_path, output_dir)

@silver_viz_app.command("diversity")
def viz_node_diversity(
    silver_path: str = "data/processed/edges.parquet",
    output_dir: str = "visualizations/structure"
):
    """graph is dominated by a few hubs or widely distributed"""
    if not Path(silver_path).exists():
        typer.secho(f"❌ Silver data missing at {silver_path}", fg=typer.colors.RED)
        raise typer.Exit(1)
    analyze_node_diversity(silver_path, output_dir)

@silver_viz_app.command("health")
def viz_feature_health(
    silver_path: str = "data/processed/edges.parquet",
    output_dir: str = "visualizations/structure"
):
    """edge 'value' distribution to ensure features aren't skewed or empty"""
    if not Path(silver_path).exists():
        typer.secho(f"❌ Silver data missing at {silver_path}", fg=typer.colors.RED)
        raise typer.Exit(1)
    analyze_feature_health(silver_path, output_dir)

@silver_viz_app.command("temporal")
def viz_temporal(
    silver_path: str = "data/processed/edges.parquet",
    output_dir: str = "visualizations/structure"
):
    """block-level transaction density to identify activity bursts."""
    if not Path(silver_path).exists():
        typer.secho(f"❌ Silver data missing at {silver_path}", fg=typer.colors.RED)
        raise typer.Exit(1)
    analyze_temporal_flow(silver_path, output_dir)

@gold_viz_app.command("balance")
def viz_gold_balance(
    gold_path: str = typer.Option(
        "data/processed/gold_labeled.parquet",
        help="Path to the labeled gold parquet"
    ),
    output_dir: str = typer.Option(
        "visualizations/gold",
        help="Folder for balance and temporal plots"
    )
):
    """
    Diagnostic: Check class balance and temporal split integrity in the Gold layer.
    Ensures that labels (Normal/Arb/Sandwich) are properly distributed.
    """
    if not Path(gold_path).exists():
        typer.secho(f"❌ Gold data missing at {gold_path}. Run 'latent data label' first.", fg=typer.colors.RED)
        raise typer.Exit(1)

    typer.echo(f"🔍 Analyzing class balance and split for {gold_path}...")
    analyze_gold_balance(gold_parquet=gold_path, output_dir=output_dir)

    typer.secho(f"✅ Gold diagnostics complete. Results in {output_dir}", fg=typer.colors.GREEN)

@gold_viz_app.command("complexity")
def viz_gold_complexity(
    gold_path: str = typer.Option("data/processed/gold_labeled.parquet"),
    output_dir: str = typer.Option("visualizations/gold")
):
    """
    Compare edge counts across Normal, Arb, and Sandwich labels.
    """
    if not Path(gold_path).exists():
        typer.secho("❌ Gold data missing.", fg=typer.colors.RED)
        raise typer.Exit(1)

    typer.echo("🔍 Comparing structural complexity by label...")
    analyze_label_complexity(gold_parquet=gold_path, output_dir=output_dir)

@gold_viz_app.command("health")
def viz_gold_tensor_health(
    graphs_dir: str = typer.Option(
        "data/processed/graphs",
        help="Directory containing the .pt graph tensors"
    ),
    output_dir: str = typer.Option(
        "visualizations/gold",
        help="Folder for the tensor report"
    )
):
    """
    Diagnostic: Check the numerical health of processed PyTorch Geometric tensors.
    Verifies that features are properly scaled, not exploding.
    """
    graphs_path = Path(graphs_dir)
    if not graphs_path.exists() or not any(graphs_path.iterdir()):
        typer.secho(
            f"❌ No graph tensors found in {graphs_dir}. Run 'latent data process' first.",
            fg=typer.colors.RED
        )
        raise typer.Exit(1)

    typer.echo(f"🧬 Sampling tensors in {graphs_dir} for feature distribution...")
    try:
        analyze_tensor_features(graphs_dir=graphs_dir, output_dir=output_dir)
    except Exception as e:
        typer.secho(f"❌ Critical Error in analyze_tensor_features: {e}", fg=typer.colors.RED)
        traceback.print_exc()  # to reveal potential hidden PyG or Numpy error
        raise typer.Exit(1) from e

@gold_viz_app.command("summary")
def viz_gold_summary(
    parquet_path: str = typer.Option(
        "data/processed/gold_processed.parquet",
        help="Path to the assembled gold parquet file"
    ),
    output_dir: str = typer.Option(
        "visualizations/gold",
        help="Folder to save the summary plot"
    )
):
    """
    Final Gold Check: Prints dataset statistics and generates structural complexity plots.
    To check the output of the DatasetAssembler.
    """

    if not Path(parquet_path).exists():
        typer.secho(f"❌ Gold parquet not found at {parquet_path}", fg=typer.colors.RED)
        raise typer.Exit(1)

    typer.echo(f"📊 Analyzing Gold Dataset: {parquet_path}...")
    visualize_gold_parquet(parquet_path=parquet_path, output_dir=output_dir)

@viz_app.command("flow")
def visualize_flow(tx_hash: str):
    explorer = GraphExplorer("data/processed/edges.parquet")
    explorer.generate_tx_graph(tx_hash)

@viz_app.command("health-check")
def check_health(
    silver_path: str = "data/processed/edges.parquet",
    gold_path: str = "data/processed/gold_labeled.parquet",
    graphs_dir: str = "data/processed/graphs"
):
    if not Path(silver_path).exists():
        typer.secho(f"❌ Silver data missing at {silver_path}. Run 'just prepare' first.", fg=typer.colors.RED)
        raise typer.Exit(1)

    validator = IngestionValidator(silver_path)
    validator.print_aggregates(gold_path=gold_path if Path(gold_path).exists() else None)

    typer.echo("📈 Generating Silver transformation plots...")
    validator.generate_transformation_plots()

    if Path(gold_path).exists():
        typer.echo("📀 Gold layer detected, generating label plots...")
        validator.generate_gold_plots(gold_path)

    if Path(graphs_dir).exists():
        typer.echo("🧬 Graph tensors detected, sampling geometric stats...")
        validator.generate_tensor_stats(graphs_dir)

    typer.secho("\n✅ Health check complete. See 'visualizations/' for details.", fg=typer.colors.GREEN)

@app.command()
def detect(
    block: int = typer.Argument(DEFAULT_BLOCK_NUMBER),
    model: str = "SAGE",
):
    print(f"Analyzing block {block} with {model}")
    return 0

@app.command()
def train(
    epochs: int = 40,
    batch_size: int = 256,
    graphs_dir: str = "data/processed/graphs"
):
    """Train the model on already processed tensors."""
    # Quick check before calling the trainer
    path = Path(graphs_dir)
    if not path.exists() or not any(path.glob("*.pt")):
        typer.secho(f"❌ No .pt tensors found in {graphs_dir}", fg=typer.colors.RED)
        typer.echo("Did you run the processing step yet?")
        raise typer.Exit(1)

    run_training(graphs_dir=graphs_dir, epochs=epochs, batch_size=batch_size)

def main():
    app()

def detect_block():
    return detect(DEFAULT_BLOCK_NUMBER, "SAGE")

def train_model():
    return train(epochs=DEFAULT_TRAIN_EPOCHS)

if __name__ == "__main__":
    main()
