import asyncio
import json
from pathlib import Path
from typing import Any


class Hydrator:
    def __init__(self, client: Any, storage_dir: str):
        self.client = client
        self.storage_path = Path(storage_dir)

    async def run(self, tx_hashes: list[str]):
        total = len(tx_hashes)
        for i in range(0, total, 100):
            batch = tx_hashes[i:i+100]
            await asyncio.gather(*[self.hydrate_single(h) for h in batch])
            print(f"📥 Hydration: {i+len(batch)}/{total} files processed...")

    async def hydrate_single(self, tx_hash: str) -> bool:
        file_path = self.storage_path / f"{tx_hash.lower()}.json"

        if file_path.exists():
            return True

        data = await self.client.get_archive_data(tx_hash)
        if data:
            file_path.write_text(json.dumps(data))
            return True
        return False
