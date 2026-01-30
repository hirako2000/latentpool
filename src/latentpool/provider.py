from typing import Any, List, Optional, cast

import httpx
import msgspec

from latentpool.gnn import TransactionTrace


class RPCRequest(msgspec.Struct):
    """
    Standard JSON-RPC request schema.
    """
    method: str
    params: list[Any]
    id: int = 1
    jsonrpc: str = "2.0"


class RPCResponse(msgspec.Struct):
    """
    Generic JSON-RPC response envelope.
    Using Any for result to handle various JSON structures before specific decoding.
    """
    result: Any = None
    error: Optional[dict[str, Any]] = None

class NodeProvider:
    """
    Block and transaction trace provider.
    """

    def __init__(self, rpc_url: str):
        self.rpc_url = rpc_url
        # for speed, pre-compile the decoder for speed
        self._list_decoder = msgspec.json.Decoder(List[TransactionTrace])

    async def get_transaction_trace(self, tx_hash: str) -> List[TransactionTrace]:
        """
        Fetches internal traces for a specific transaction hash.
        """
        payload = RPCRequest(method="trace_transaction", params=[tx_hash])
        return await self._execute(payload)

    async def get_block_traces(self, block_number: int) -> List[TransactionTrace]:
        """
        Fetches all internal traces for a given block number.
        """
        hex_block = hex(block_number)
        payload = RPCRequest(method="trace_block", params=[hex_block])
        return await self._execute(payload)

    async def _execute(self, request: RPCRequest) -> List[TransactionTrace]:
        """
        Executes the RPC call and handles response decoding.
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                self.rpc_url,
                content=msgspec.json.encode(request),
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()

            envelope = msgspec.json.decode(response.content, type=RPCResponse)

            if envelope.error:
                err_val = cast(Any, envelope.error)
                raise RuntimeError(f"Node RPC Error: {err_val}")

            if envelope.result is None:
                return []

            # Re-encode the result part to bytes so our optimized decoder can process it
            # This is faster than standard dict-to-object mapping libraries
            result_bytes = msgspec.json.encode(envelope.result)
            return self._list_decoder.decode(result_bytes)
