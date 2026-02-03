import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
import pytest

from latentpool.data.transformation.parquet_exporter import ParquetExporter

TX_HASH_SAMPLE = "0x" + "f" * 64
BLOCK_HEX = "0x123456"
BLOCK_INT = 1193046
TOKEN_ADDR = "0x" + "d" * 40
SENDER_ADDR = "0x" + "1" * 40
RECEIVER_ADDR = "0x" + "2" * 40
TOPIC_TRANSFER = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
VALUE_HEX = "0xde0b6b3a7640000"
VALUE_NUMERIC = 1e18
VALUE_ZERO_NUMERIC = 0.0
HEX_PREFIX = "0x"

ADDRESS_CHAR_COUNT = 40
PADDING_ZERO_COUNT = 24
EXPECTED_ROW_COUNT = 1

# Padded topics for ERC-20 (32 bytes / 64 chars + 0x)
TOPIC_FROM = HEX_PREFIX + ("0" * PADDING_ZERO_COUNT) + SENDER_ADDR[2:]
TOPIC_TO = HEX_PREFIX + ("0" * PADDING_ZERO_COUNT) + RECEIVER_ADDR[2:]

@pytest.fixture
def exporter_setup(tmp_path: Path) -> Tuple[Path, Path, ParquetExporter]:
    """Provides typed paths and the exporter instance."""
    raw_dir = tmp_path / "raw"
    proc_dir = tmp_path / "processed"
    raw_dir.mkdir()
    exporter = ParquetExporter(str(raw_dir), str(proc_dir))
    return raw_dir, proc_dir, exporter

def create_mock_json(path: Path, data: Dict[str, Any]) -> None:
    """Helper to write test data."""
    with open(path, "w") as f:
        json.dump(data, f)

def test_process_all_success(exporter_setup: Tuple[Path, Path, ParquetExporter]) -> None:
    """Verifies successful extraction of transfer events to parquet."""
    raw_dir, proc_dir, exporter = exporter_setup
    pd_any: Any = pd

    mock_receipt: Dict[str, Any] = {
        "transactionHash": TX_HASH_SAMPLE,
        "blockNumber": BLOCK_HEX,
        "logs": [{
            "address": TOKEN_ADDR,
            "topics": [TOPIC_TRANSFER, TOPIC_FROM, TOPIC_TO],
            "data": VALUE_HEX
        }]
    }
    create_mock_json(raw_dir / "tx1.json", mock_receipt)

    exporter.process_all()

    output_file = proc_dir / "edges.parquet"
    assert output_file.exists()

    df: Any = pd_any.read_parquet(output_file)
    assert len(df) == EXPECTED_ROW_COUNT
    assert df["tx_hash"].iloc[0] == TX_HASH_SAMPLE.lower()
    assert df["from"].iloc[0] == SENDER_ADDR.lower()
    assert df["to"].iloc[0] == RECEIVER_ADDR.lower()

    assert float(df["value"].iloc[0]) == VALUE_NUMERIC
    assert int(df["block_number"].iloc[0]) == BLOCK_INT

def test_process_all_no_json_files(exporter_setup: Tuple[Path, Path, ParquetExporter], capsys: Any) -> None:
    """Covers the branch where the raw directory is empty."""
    _, _, exporter = exporter_setup
    exporter.process_all()
    captured = capsys.readouterr()
    assert "No JSON files found" in captured.out

def test_process_all_missing_tx_hash(exporter_setup: Tuple[Path, Path, ParquetExporter]) -> None:
    """Covers continue when transactionHash is missing."""
    raw_dir, _, exporter = exporter_setup
    create_mock_json(raw_dir / "invalid.json", {"blockNumber": BLOCK_HEX})

    exporter.process_all()
    assert not (Path(exporter.processed_dir) / "edges.parquet").exists()

def test_process_all_filter_logic(exporter_setup: Tuple[Path, Path, ParquetExporter], capsys: Any) -> None:
    """Covers logs that are NOT transfers (wrong topics or prefix)."""
    raw_dir, _, exporter = exporter_setup
    mock_receipt: Dict[str, Any] = {
        "transactionHash": TX_HASH_SAMPLE,
        "blockNumber": BLOCK_HEX,
        "logs": [
            {"topics": ["0x1234"], "data": HEX_PREFIX},
            {"topics": ["0xeeee", TOPIC_FROM, TOPIC_TO], "data": HEX_PREFIX}
        ]
    }
    create_mock_json(raw_dir / "filtered.json", mock_receipt)

    exporter.process_all()
    captured = capsys.readouterr()
    assert "No valid ERC-20 transfer edges found" in captured.out

def test_process_all_empty_data_field(exporter_setup: Tuple[Path, Path, ParquetExporter]) -> None:
    """Verifies that '0x' or empty data is converted to 0.0 float."""
    raw_dir, proc_dir, exporter = exporter_setup
    pd_any: Any = pd

    mock_receipt: Dict[str, Any] = {
        "transactionHash": TX_HASH_SAMPLE,
        "blockNumber": BLOCK_HEX,
        "logs": [{
            "address": TOKEN_ADDR,
            "topics": [TOPIC_TRANSFER, TOPIC_FROM, TOPIC_TO],
            "data": HEX_PREFIX
        }]
    }
    create_mock_json(raw_dir / "empty_data.json", mock_receipt)

    exporter.process_all()

    # read_parquet via proxy
    df: Any = pd_any.read_parquet(proc_dir / "edges.parquet")
    assert float(df["value"].iloc[0]) == VALUE_ZERO_NUMERIC

def test_process_all_exception_handling(exporter_setup: Tuple[Path, Path, ParquetExporter], caplog: Any) -> None:
    """Covers the try-except block when a file is malformed."""
    raw_dir, _, exporter = exporter_setup
    broken_file = raw_dir / "broken.json"
    broken_file.write_text("{ broken: ")

    with caplog.at_level(logging.ERROR):
        exporter.process_all()

    assert f"Error processing file {broken_file}" in caplog.text
