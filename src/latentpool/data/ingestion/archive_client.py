import asyncio
from http import HTTPStatus
from typing import Any, Dict, Optional

import httpx

from latentpool.configs.data_config import ArchiveConfig


class ArchiveNodeClient:
    def __init__(self, config: ArchiveConfig, client: Optional[httpx.AsyncClient] = None):
        self.config = config
        self.client = client or httpx.AsyncClient(timeout=config.timeout)
        self.semaphore = asyncio.Semaphore(config.max_rps)

    async def get_archive_data(self, tx_hash: str) -> Optional[Dict[str, Any]]:
        async with self.semaphore:
            return await self._fetch_with_backoff(tx_hash)

    async def _fetch_with_backoff(self, tx_hash: str) -> Optional[Dict[str, Any]]:
        for attempt in range(self.config.retry_count):
            try:
                payload = {
                    "jsonrpc": "2.0", "id": 1,
                    "method": "debug_traceTransaction",
                    "params": [tx_hash, {"tracer": "callTracer"}]
                }
                resp = await self.client.post(self.config.rpc_url, json=payload)

                if resp.status_code == HTTPStatus.TOO_MANY_REQUESTS:
                    wait_time = (2**attempt) * self.config.initial_backoff
                    await asyncio.sleep(wait_time)
                    continue

                resp.raise_for_status()
                return resp.json().get("result")
            except (httpx.HTTPError, Exception):
                # we've exhausted retries, just exit
                if attempt == self.config.retry_count - 1:
                    break
        return None
