from pathlib import Path
from typing import Any, List, cast
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

from latentpool.configs.data_config import ArchiveConfig, DataConfig
from latentpool.data.ingestion.coordinator import IngestionCoordinator

MOCK_RPC_URL = "http://mock.com"
MAX_RPS = 1

HEX_LEN = 64
TX_HASH_1 = "0x" + "a" * HEX_LEN
TX_HASH_2 = "0x" + "b" * HEX_LEN
TX_HASH_3 = "0x" + "c" * HEX_LEN
INVALID_HASH = "0x123"
NON_HEX_HASH = "g" * HEX_LEN
ARBITRAGE_FILE = "arbitrage.csv"
RAW_EXT = ".json"

@pytest.fixture
def mock_config(tmp_path: Path) -> DataConfig:
    raw_path = tmp_path / "raw"
    return DataConfig(
        archive_node=ArchiveConfig(rpc_url=MOCK_RPC_URL, max_rps=MAX_RPS),
        raw_dir=str(raw_path)
    )

@pytest.fixture
def coordinator(mock_config: DataConfig) -> IngestionCoordinator:
    mock_client = MagicMock()
    mock_client.client = MagicMock()
    mock_client.client.aclose = AsyncMock()

    mock_hydrator = MagicMock()
    mock_hydrator.run = AsyncMock()

    coord = IngestionCoordinator(mock_config)
    coord.client = mock_client
    coord.hydrator = mock_hydrator
    return coord

def test_extract_hashes_valid_extraction(coordinator: IngestionCoordinator, tmp_path: Path) -> None:
    test_file = tmp_path / "data.csv"
    content = f"header\n{TX_HASH_1}\n{INVALID_HASH}\n{TX_HASH_1},some_val\n{TX_HASH_2}"
    test_file.write_text(content, encoding="utf-8")

    hashes = coordinator.extract_hashes([str(test_file)])

    expected_set = {TX_HASH_1, TX_HASH_2}
    assert set(hashes) == expected_set
    assert INVALID_HASH not in hashes

def test_get_coverage_stats(coordinator: IngestionCoordinator, tmp_path: Path) -> None:
    config = cast(DataConfig, coordinator.config)
    raw_dir = Path(config.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / f"{TX_HASH_1}{RAW_EXT}").write_text("{}")

    csv_path = tmp_path / ARBITRAGE_FILE
    csv_path.write_text(f"tx_hash\n{TX_HASH_1}\n{TX_HASH_2}", encoding="utf-8")

    stats = coordinator.get_coverage_stats([str(csv_path)])

    assert len(stats) == 1
    assert stats[0]["Type"] == "Arbitrage"

    expected_total = 2  # TX_HASH_1 and TX_HASH_2
    assert stats[0]["Total"] == expected_total

    # downloaded TX_HASH_1, but not TX_HASH_2
    assert stats[0]["Found"] == 1
    assert stats[0]["Missing"] == 1
    assert stats[0]["Coverage"] == "50.0%"

@pytest.mark.asyncio
async def test_run_negative_sampling_basic(coordinator: IngestionCoordinator, tmp_path: Path) -> None:
    """Ensure coverage for negative sampling logic."""
    silver_path = tmp_path / "silver.parquet"

    df: Any = pd.DataFrame({
        "tx_hash": [TX_HASH_1],
        "block_number": [100]
    })
    df.to_parquet(silver_path) # type: ignore

    coordinator.client.get_block_tx_hashes = AsyncMock(return_value=[TX_HASH_2, TX_HASH_3])

    # We also need to mock the hydrator's run method, as it's awaited at the end
    coordinator.hydrator.run = AsyncMock()

    # Since TX_HASH_2 and TX_HASH_3 are not in mev_hashes (which only contains TX_HASH_1),
    # the pool should fill up.
    count = await coordinator.run_negative_sampling(str(silver_path), target_count=1)

    # If count is 0, it means the loop didn't find 'normals' or mev_hashes check failed.
    assert count >= 1
    coordinator.hydrator.run.assert_called_once()

@pytest.mark.asyncio
async def test_run_hydration(coordinator: IngestionCoordinator, tmp_path: Path) -> None:
    config = cast(DataConfig, coordinator.config)
    raw_dir = Path(config.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tmp_path / "test.csv"
    csv_path.write_text(f"hash\n{TX_HASH_1}", encoding="utf-8")

    async def mock_run(hashes: List[str]) -> None:
        for h in hashes:
            (raw_dir / f"{h}{RAW_EXT}").write_text("{}")

    cast(AsyncMock, coordinator.hydrator.run).side_effect = mock_run

    count = await coordinator.run_hydration([str(csv_path)])

    assert count == 1
    cast(AsyncMock, coordinator.hydrator.run).assert_called_once()

def test_extract_hashes_unrecoverable(coordinator: IngestionCoordinator, tmp_path: Path) -> None:
    """
    Coordinator uses os.path.exists() and then open().
    Since the current code doesn't catch the IsADirectoryError,
    we test that it handles missing files gracefully as per current impl.
    """
    hashes = coordinator.extract_hashes(["/non/existent/path.csv"])
    assert len(hashes) == 0

def test_extract_hashes_non_hex_filter(coordinator: IngestionCoordinator, tmp_path: Path) -> None:
    csv = tmp_path / "hex.csv"
    csv.write_text(f"{TX_HASH_1}\n{NON_HEX_HASH}\n{INVALID_HASH}", encoding="utf-8")

    hashes = coordinator.extract_hashes([str(csv)])
    assert set(hashes) == {TX_HASH_1}

@pytest.mark.asyncio
async def test_run_hydration_no_hashes_found(coordinator: IngestionCoordinator, tmp_path: Path) -> None:
    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("nothing here", encoding="utf-8")

    count = await coordinator.run_hydration([str(empty_csv)])
    assert count == 0
    cast(AsyncMock, coordinator.hydrator.run).assert_not_called()

def test_extract_hashes_skips_non_existent_file(coordinator: IngestionCoordinator) -> None:
    """Non existing path should continue."""
    hashes = coordinator.extract_hashes(["this_file_is_missing.csv"])
    assert hashes == []

@pytest.mark.asyncio
async def test_run_negative_sampling_missing_parquet(coordinator: IngestionCoordinator) -> None:
    count = await coordinator.run_negative_sampling("non_existent.parquet", target_count=10)
    assert count == 0

@pytest.mark.asyncio
async def test_run_negative_sampling_target_break(coordinator: IngestionCoordinator, tmp_path: Path) -> None:
    silver_path = tmp_path / "sampling_break.parquet"
    df: Any = pd.DataFrame({
        "tx_hash": [TX_HASH_1, TX_HASH_2],
        "block_number": [100, 200]
    })
    df.to_parquet(silver_path) # type: ignore

    coordinator.client.get_block_tx_hashes = AsyncMock(return_value=[f"0x{i:064x}" for i in range(5)])

    count = await coordinator.run_negative_sampling(str(silver_path), target_count=2)

    expected_count = 2
    assert count >= expected_count
    assert coordinator.client.get_block_tx_hashes.call_count == 1

@pytest.mark.asyncio
async def test_run_negative_sampling_exception_handling(coordinator: IngestionCoordinator, tmp_path: Path) -> None:
    silver_path = tmp_path / "sampling_exception.parquet"
    df: Any = pd.DataFrame({
        "tx_hash": [TX_HASH_1, "0x000"],
        "block_number": [100, 200]
    })
    df.to_parquet(silver_path) # type: ignore

    coordinator.client.get_block_tx_hashes = AsyncMock(side_effect=[
        Exception("RPC Timeout"),
        [TX_HASH_3]
    ])

    count = await coordinator.run_negative_sampling(str(silver_path), target_count=1)

    assert count == 1
    expected_call_count = 2
    assert coordinator.client.get_block_tx_hashes.call_count == expected_call_count
