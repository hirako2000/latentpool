from typing import Any, Dict

import httpx
import msgspec
import pytest
import respx

from latentpool.provider import NodeProvider


@pytest.mark.anyio
async def test_node_provider_trace_transaction(url: str, provider: NodeProvider):
    """
    Verifies NodeProvider correctly decodes a transaction trace response.
    """
    mock_response: Dict[str, Any] = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": [
            {
                "action": {
                    "from": "0x123",
                    "to": "0x456",
                    "value": "0x0",
                    "gas": "0x5208",
                    "input": "0x",
                    "callType": "call",
                },
                "result": {"gasUsed": "0x5208", "output": "0x"},
                "subtraces": 0,
                "traceAddress": [],
                "type": "call",
            }
        ],
    }

    with respx.mock:
        respx.post(url).mock(return_value=httpx.Response(200, json=mock_response))

        traces = await provider.get_transaction_trace("0xabc")

        assert len(traces) == 1
        assert traces[0].action.from_address == "0x123"


@pytest.mark.anyio
async def test_node_provider_error_handling(url: str, provider: NodeProvider):
    """
    Verifies that the provider raises a RuntimeError on RPC errors.
    """
    error_response: Dict[str, Any] = {
        "jsonrpc": "2.0",
        "id": 1,
        "error": {"code": -32000, "message": "execution reverted"},
    }

    with respx.mock:
        respx.post(url).mock(return_value=httpx.Response(200, json=error_response))

        with pytest.raises(RuntimeError, match="Node RPC Error"):
            await provider.get_transaction_trace("0xabc")


@pytest.mark.anyio
async def test_node_provider_get_block_traces(url: str, provider: NodeProvider):
    """
    Verifies block traces are fetched and block numbers are correctly hex-encoded.
    """
    block_num = 12345  # 0x3039 in hex

    mock_response: Dict[str, Any] = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": []
    }

    with respx.mock:
        route = respx.post(url).mock(return_value=httpx.Response(200, json=mock_response))

        await provider.get_block_traces(block_num)

        request_body: Dict[str, Any] = msgspec.json.decode(route.calls.last.request.content)
        assert request_body["params"] == ["0x3039"]


@pytest.mark.anyio
async def test_node_provider_empty_result(url: str, provider: NodeProvider):
    """
    Verifies that a 'null' (None) result from the RPC returns an empty list.
    """
    mock_response: Dict[str, Any] = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": None
    }

    with respx.mock:
        respx.post(url).mock(return_value=httpx.Response(200, json=mock_response))
        traces = await provider.get_transaction_trace("0xabc")

        assert traces == []

@pytest.mark.anyio
async def test_node_provider_connection_persistence(url: str, provider: NodeProvider):
    """
    Verifies that the provider uses the exact same client instance across calls.
    """
    mock_response: Dict[str, Any] = {"jsonrpc": "2.0", "id": 1, "result": []}

    initial_client_id = id(provider.client)

    with respx.mock:
        respx.post(url).mock(return_value=httpx.Response(200, json=mock_response))

        await provider.get_block_traces(1)

        # Intermediate check: ensure it's still open and the same instance
        assert id(provider.client) == initial_client_id
        assert provider.client.is_closed is False

        await provider.get_block_traces(2)

        # id() must still match, proving a new AsyncClient was never instantiated
        assert id(provider.client) == initial_client_id
        assert provider.client.is_closed is False
