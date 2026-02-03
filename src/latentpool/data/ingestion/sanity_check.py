from pathlib import Path

import typer

from latentpool.data.ingestion.coordinator import IngestionCoordinator


def check_status() -> None:
    raw_dir = Path("data/raw/traces")
    arbs_csv = "data/labeled_arbitrage.csv"
    sand_csv = "data/labeled_sandwich.csv"

    downloaded_files = list(raw_dir.glob("*.json"))
    downloaded_hashes = {f.stem.lower() for f in downloaded_files}

    coord = IngestionCoordinator(None) # Config-less for extraction only
    arb_hashes = set(coord.extract_hashes([arbs_csv]))
    sand_hashes = set(coord.extract_hashes([sand_csv]))

    arb_found = downloaded_hashes.intersection(arb_hashes)
    sand_found = downloaded_hashes.intersection(sand_hashes)

    # Normal = (Everything on disk) minus (Known MEV)
    # identifies transactions brought in via run_negative_sampling
    normal_found = downloaded_hashes - arb_hashes - sand_hashes

    typer.secho("\n📁 Data Layer Inventory", fg=typer.colors.CYAN, bold=True)
    print(f"Total files on disk: {len(downloaded_hashes):,}")
    print("-" * 30)
    print(f"✅ Arbitrage: {len(arb_found):,}")
    print(f"✅ Sandwich:  {len(sand_found):,}")
    print(f"✅ Normal:    {len(normal_found):,}")

    WARN_IF_BELOW = 90
    typer.secho("\n📊 MEV Ingestion Coverage", fg=typer.colors.CYAN, bold=True)
    for label, found, total_required in [
        ("Arbitrage", len(arb_found), len(arb_hashes)),
        ("Sandwich ", len(sand_found), len(sand_hashes))
    ]:
        coverage = (found / total_required * 100) if total_required > 0 else 0
        color = typer.colors.GREEN if coverage > WARN_IF_BELOW else typer.colors.YELLOW
        typer.secho(f"{label}: {found:,}/{total_required:,} ({coverage:.1f}%)", fg=color)

    if len(normal_found) < (len(arb_found) + len(sand_found)):
        typer.secho(
            "\n⚠️ Warning: Dataset is imbalanced. MEV samples outnumber Normal samples.",
            fg=typer.colors.RED,
            bold=True
        )
        print("Suggested: Increase target_count in run_negative_sampling.")
