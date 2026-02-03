import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from latentpool.data.ingestion.archive_client import ArchiveNodeClient
from latentpool.data.ingestion.hydrator import Hydrator


@pytest.fixture
def mock_node_client() -> MagicMock:
    return MagicMock(spec=ArchiveNodeClient)

@pytest.fixture
def hydrator(mock_node_client: MagicMock) -> Hydrator:
    return Hydrator(client=mock_node_client, storage_dir="dummy_dir")

@pytest.mark.asyncio
async def test_hydrate_single_cache_hit(hydrator: Hydrator, mock_node_client: MagicMock):
    tx_hash: str = "0xabc123"

    with patch.object(Path, "exists", return_value=True):
        result: bool = await hydrator.hydrate_single(tx_hash)

    assert result is True
    mock_node_client.get_archive_data.assert_not_called()

@pytest.mark.asyncio
async def test_hydrate_single_success(hydrator: Hydrator, mock_node_client: MagicMock):
    tx_hash: str = "0xdef456"
    mock_data: Dict[str, Any] = {"receipt": {}, "trace": {}}
    mock_node_client.get_archive_data = AsyncMock(return_value=mock_data)

    with patch.object(Path, "exists", return_value=False), \
         patch.object(Path, "write_text") as mock_write:

        result: bool = await hydrator.hydrate_single(tx_hash)

        assert result is True
        mock_write.assert_called_once_with(json.dumps(mock_data))
        mock_node_client.get_archive_data.assert_called_once_with(tx_hash)

@pytest.mark.asyncio
async def test_hydrate_single_node_failure(hydrator: Hydrator, mock_node_client: MagicMock):
    tx_hash: str = "0xno_data"
    mock_node_client.get_archive_data = AsyncMock(return_value=None)

    with patch.object(Path, "exists", return_value=False), \
         patch.object(Path, "write_text") as mock_write:

        result: bool = await hydrator.hydrate_single(tx_hash)

        assert result is False
        mock_write.assert_not_called()

@pytest.mark.asyncio
async def test_run_batch(hydrator: Hydrator, mock_node_client: MagicMock):
    tx_hashes: List[str] = ["0x1", "0x2"]
    EXPECTED_TOTAL_HYDRATIONS: int = 2

    with patch.object(hydrator, "hydrate_single", AsyncMock(return_value=True)) as mock_single:
        await hydrator.run(tx_hashes)
        assert mock_single.call_count == EXPECTED_TOTAL_HYDRATIONS
