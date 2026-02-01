import asyncio
import json
from pathlib import Path

from .archive_client import ArchiveNodeClient


class Hydrator:
    def __init__(self, client: "ArchiveNodeClient", storage_dir: str):
        self.client = client
        self.storage_path = Path(storage_dir)
        self.storage_path.mkdir(parents=True, exist_ok=True)

    async def run(self, tx_hashes: list[str]):
        """Main entry point to hydrate a list of hashes."""

        tasks = [self.hydrate_single(h) for h in tx_hashes]
        await asyncio.gather(*tasks)

    async def hydrate_single(self, tx_hash: str) -> bool:
        file_path = self.storage_path / f"{tx_hash}.json"
        if file_path.exists():
            return True

        data = await self.client.get_archive_data(tx_hash)
        if data:
            file_path.write_text(json.dumps(data))
            return True
        return False
