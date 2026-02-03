from http import HTTPStatus

import httpx
import pytest
import respx

from latentpool.configs.data_config import ArchiveConfig
from latentpool.data.ingestion.archive_client import ArchiveNodeClient

SAMPLE_HASH = "0x1234567890"

@pytest.mark.asyncio
async def test_get_archive_data_success() -> None:
    config = ArchiveConfig(rpc_url="https://mock.node", max_rps=1)
    with respx.mock:
        respx.post("https://mock.node").mock(
            return_value=httpx.Response(200, json={"result": {"status": "ok"}})
        )

        async with httpx.AsyncClient() as client:
            node_client = ArchiveNodeClient(config, client=client)
            result = await node_client.get_archive_data(SAMPLE_HASH)

        assert result == {"status": "ok"}

@pytest.mark.asyncio
async def test_get_archive_data_bad_request() -> None:
    config = ArchiveConfig(rpc_url="https://mock.node")
    with respx.mock:
        respx.post("https://mock.node").mock(
            return_value=httpx.Response(400)
        )
        async with httpx.AsyncClient() as client:
            node_client = ArchiveNodeClient(config, client=client)
            result = await node_client.get_archive_data(SAMPLE_HASH)
            assert result is None

@pytest.mark.asyncio
async def test_get_archive_data_total_failure_logging() -> None:
    config = ArchiveConfig(rpc_url="https://mock.node", retry_count=1)

    with respx.mock:
        respx.post("https://mock.node").side_effect = Exception("Connection Refused")

        async with httpx.AsyncClient() as client:
            node_client = ArchiveNodeClient(config, client=client)
            result = await node_client.get_archive_data(SAMPLE_HASH)
            assert result is None

@pytest.mark.asyncio
async def test_get_archive_data_rate_limit_retry() -> None:
    """Verifies backoff logic works by retrying on 500/429 (raise_for_status)."""
    config = ArchiveConfig(
        rpc_url="https://mock.node",
        retry_count=2,
        initial_backoff=0.01
    )

    with respx.mock:
        route = respx.post("https://mock.node")
        # First call fails (Exception via raise_for_status), second call succeeds
        route.side_effect = [
            httpx.Response(HTTPStatus.TOO_MANY_REQUESTS),
            httpx.Response(HTTPStatus.OK, json={"result": "success"})
        ]

        async with httpx.AsyncClient() as client:
            node_client = ArchiveNodeClient(config, client=client)
            result = await node_client.get_archive_data(SAMPLE_HASH)

        assert result == "success"
        expected_route_call_count = 2
        assert route.call_count == expected_route_call_count

@pytest.mark.asyncio
async def test_get_archive_data_result_none() -> None:
    """Fix: Client returns None if 'result' key is None. Removed print assertion."""
    config = ArchiveConfig(rpc_url="https://mock.node")
    with respx.mock:
        respx.post("https://mock.node").mock(
            return_value=httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": None})
        )

        async with httpx.AsyncClient() as client:
            node_client = ArchiveNodeClient(config, client=client)
            result = await node_client.get_archive_data(SAMPLE_HASH)
            assert result is None

@pytest.mark.asyncio
async def test_get_block_tx_hashes_success() -> None:
    """Coverage for get_block_tx_hashes method."""
    config = ArchiveConfig(rpc_url="https://mock.node")
    with respx.mock:
        respx.post("https://mock.node").mock(
            return_value=httpx.Response(
                200,
                json={"result": {"transactions": ["0x1", "0x2"]}}
            )
        )

        async with httpx.AsyncClient() as client:
            node_client = ArchiveNodeClient(config, client=client)
            result = await node_client.get_block_tx_hashes(12345)

        assert result == ["0x1", "0x2"]
