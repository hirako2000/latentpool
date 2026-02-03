import logging
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Set

import pandas as pd
import typer
from tqdm import tqdm

from .archive_client import ArchiveNodeClient
from .hydrator import Hydrator

logger = logging.getLogger(__name__)

class IngestionCoordinator:
    def __init__(self, config: Any):
        self.config = config
        self.client = ArchiveNodeClient(self.config.archive_node)
        self.hydrator = Hydrator(client=self.client, storage_dir=self.config.raw_dir)

    def extract_hashes(self, csv_paths: List[str]) -> List[str]:
        hash_pattern = re.compile(r'0x[a-fA-F0-9]{64}')
        all_hashes: List[str] = []
        for path in csv_paths:
            if not os.path.exists(path):
                continue
            with open(path, 'r') as f:
                matches = hash_pattern.findall(f.read())
                all_hashes.extend([m.lower() for m in matches])
        return list(set(all_hashes))

    async def run_hydration(self, csv_paths: List[str]) -> int:
        hashes = self.extract_hashes(csv_paths)
        if not hashes:
            return 0
        await self.hydrator.run(hashes)
        return len(hashes)

    async def run_negative_sampling(self, silver_parquet_path: str, target_count: int) -> int:
        if not os.path.exists(silver_parquet_path):
            typer.echo("❌ Silver parquet not found.")
            return 0

        df: Any = pd.read_parquet(silver_parquet_path) # type: ignore

        mev_hashes: Set[str] = set(df['tx_hash'].unique())
        relevant_blocks: List[Any] = df['block_number'].unique().tolist()

        random.shuffle(relevant_blocks)
        normal_pool: List[str] = []

        pbar = tqdm(total=target_count, desc="🔍 Scanning blocks for Normal txs")

        for b in relevant_blocks:
            if len(normal_pool) >= target_count:
                break

            try:
                block_txs: List[str] = await self.client.get_block_tx_hashes(b)
                normals = [h for h in block_txs if h.lower() not in mev_hashes]

                if normals:
                    take_count = min(len(normals), 10)
                    sampled = random.sample(normals, take_count)
                    normal_pool.extend(sampled)
                    pbar.update(take_count)
            except Exception as e:
                logger.error(f"Failed to scan block {b}: {e}")
                continue

        pbar.close()

        if normal_pool:
            typer.secho(
                f"📥 Starting hydration for {len(normal_pool):,} normal transactions...",
                fg=typer.colors.MAGENTA
            )
            await self.hydrator.run(normal_pool)

        return len(normal_pool)

    def get_coverage_stats(self, csv_paths: List[str]) -> List[Dict[str, Any]]:
        raw_dir = Path(self.config.raw_dir)
        downloaded_hashes = {f.stem.lower() for f in raw_dir.glob("*.json")}
        stats: List[Dict[str, Any]] = []
        for path in csv_paths:
            csv_hashes = self.extract_hashes([path])
            total = len(csv_hashes)
            found = sum(1 for h in csv_hashes if h in downloaded_hashes)
            label = "Arbitrage" if "arb" in path.lower() else "Sandwich"
            stats.append({
                "Type": label,
                "Total": total,
                "Found": found,
                "Missing": total - found,
                "Coverage": f"{(found/total)*100:.1f}%" if total > 0 else "0%"
            })
        return stats
