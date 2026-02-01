# tests/data/test_archive_client.py
from http import HTTPStatus

import httpx
import pytest
import respx

from latentpool.configs.data_config import ArchiveConfig
from latentpool.data.ingestion.archive_client import ArchiveNodeClient


@pytest.mark.asyncio
async def test_get_archive_data_success():
    config = ArchiveConfig(rpc_url="https://mock.node", max_rps=1)

    with respx.mock:
        respx.post("https://mock.node").mock(return_value=httpx.Response(200, json={"result": {"status": "ok"}}))

        async with httpx.AsyncClient() as client:
            node_client = ArchiveNodeClient(config, client=client)
            result = await node_client.get_archive_data("0x123")

        assert result == {"status": "ok"}

@pytest.mark.asyncio
async def test_get_archive_data_rate_limit_retry():
    EXPECTED_CALL_COUNT = 2
    config = ArchiveConfig(
        rpc_url="https://mock.node",
        retry_count=EXPECTED_CALL_COUNT,
        initial_backoff=0.01
    )

    with respx.mock:
        route = respx.post("https://mock.node")
        route.side_effect = [
            httpx.Response(HTTPStatus.TOO_MANY_REQUESTS),
            httpx.Response(HTTPStatus.OK, json={"result": "success"})
        ]

        async with httpx.AsyncClient() as client:
            node_client = ArchiveNodeClient(config, client=client)
            result = await node_client.get_archive_data("0x123")

        assert result == "success"
        assert route.call_count == EXPECTED_CALL_COUNT

@pytest.mark.asyncio
async def test_get_archive_data_total_failure():
    config = ArchiveConfig(rpc_url="https://mock.node", retry_count=2)

    with respx.mock:
        respx.post("https://mock.node").mock(return_value=httpx.Response(500))

        async with httpx.AsyncClient() as client:
            node_client = ArchiveNodeClient(config, client=client)
            result = await node_client.get_archive_data("0x123")

        assert result is None
