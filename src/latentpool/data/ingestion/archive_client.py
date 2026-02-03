import asyncio
from typing import Any, Dict, List, Optional

import httpx

from latentpool.configs.data_config import ArchiveConfig


class ArchiveNodeClient:
    def __init__(self, config: ArchiveConfig, client: Optional[httpx.AsyncClient] = None):
        self.config = config
        self.client = client or httpx.AsyncClient(timeout=config.timeout)
        self.semaphore = asyncio.Semaphore(config.max_rps)

    async def get_archive_data(self, tx_hash: str) -> Optional[Dict[str, Any]]:
        async with self.semaphore:
            return await self._fetch_with_backoff("eth_getTransactionReceipt", [tx_hash])

    async def get_block_tx_hashes(self, block_number: int) -> List[str]:
        """Fetch all sibling hashes in a block to find 'Normal' transactions."""
        async with self.semaphore:
            result = await self._fetch_with_backoff("eth_getBlockByNumber", [hex(block_number), False])
            return result.get("transactions", []) if result else []

    async def _fetch_with_backoff(self, method: str, params: List[Any]) -> Optional[Dict[str, Any]]:
        BAD_REQUEST = 400
        for attempt in range(self.config.retry_count):
            try:
                payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
                resp = await self.client.post(self.config.rpc_url, json=payload)

                if resp.status_code == BAD_REQUEST:
                    return None

                resp.raise_for_status()
                return resp.json().get("result")
            except Exception:
                if attempt == self.config.retry_count - 1:
                    break

                await asyncio.sleep((2**attempt) * self.config.initial_backoff)
        return None
