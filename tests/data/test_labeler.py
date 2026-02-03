from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest

from latentpool.data.labeler import Labeler

TX_HASH_NORMAL = "0x" + "0" * 64
TX_HASH_ARB = "0x" + "a" * 64
TX_HASH_SAND = "0x" + "b" * 64
TX_HASH_EXTRA = "0x" + "1" * 64

LABEL_NORMAL = 0
LABEL_ARB = 1
LABEL_SAND = 2

EXPECTED_TRAIN_COUNT = 3
EXPECTED_TEST_COUNT = 1
BLOCKS_LIST = [100, 101, 102, 103]

@pytest.fixture
def sample_edges_path(tmp_path: Path) -> Path:
    path = tmp_path / "edges.parquet"
    df: Any = pd.DataFrame({
        "tx_hash": [TX_HASH_NORMAL, TX_HASH_ARB, TX_HASH_SAND, TX_HASH_EXTRA],
        "block_number": BLOCKS_LIST
    })
    df.to_parquet(path)
    return path

@pytest.fixture
def labeler(tmp_path: Path, sample_edges_path: Path) -> Labeler:
    arb_csv = tmp_path / "arb.csv"
    sand_csv = tmp_path / "sand.csv"
    # making usre hashes are written in a way the regex will definitely pick up
    arb_csv.write_text(f"hash\n{TX_HASH_ARB}")
    sand_csv.write_text(f"hash\n{TX_HASH_SAND}")
    return Labeler(str(sample_edges_path), str(arb_csv), str(sand_csv))

def test_labeler_run_success(labeler: Labeler, tmp_path: Path) -> None:
    """Tests the full labeling flow, including multi-class mapping and temporal split."""
    output_path = tmp_path / "gold.parquet"
    labeler.run(str(output_path))

    assert output_path.exists()
    df: Any = pd.read_parquet(output_path)  # type: ignore


    assert int(df.loc[df['tx_hash'] == TX_HASH_NORMAL, 'label'].iloc[0]) == LABEL_NORMAL
    assert int(df.loc[df['tx_hash'] == TX_HASH_ARB, 'label'].iloc[0]) == LABEL_ARB
    assert int(df.loc[df['tx_hash'] == TX_HASH_SAND, 'label'].iloc[0]) == LABEL_SAND

    # we split (80% of 4 unique blocks = 3 blocks for train)
    split_counts: Any = df['split'].value_counts()
    assert split_counts['train'] == EXPECTED_TRAIN_COUNT
    assert split_counts['test'] == EXPECTED_TEST_COUNT

def test_labeler_missing_edges_raises_error(tmp_path: Path) -> None:
    """Covers the FileNotFoundError when silver data is missing."""
    labeler = Labeler("non_existent.parquet", "arb.csv", "sand.csv")
    with pytest.raises(FileNotFoundError, match="Missing Silver data"):
        labeler.run("output.parquet")

def test_extract_hashes_file_not_found(labeler: Labeler, capsys: Any) -> None:
    """Covers the branch where a label file is missing (typer.secho yellow)."""
    hashes = labeler._extract_hashes_from_file(Path("missing_labels.csv"))  # type: ignore
    assert len(hashes) == 0
    captured = capsys.readouterr()
    assert "⚠️" in captured.out or "Label file not found" in captured.out

def test_extract_hashes_exception_handling(labeler: Labeler, tmp_path: Path, capsys: Any) -> None:
    """Covers the 'except Exception' block in hash mining (typer.secho red)."""
    bad_file = tmp_path / "broken.csv"
    bad_file.write_text("some data")

    with patch("builtins.open", side_effect=PermissionError("Denied")):
        hashes = labeler._extract_hashes_from_file(bad_file)  # type: ignore
        assert len(hashes) == 0
        captured = capsys.readouterr()
        assert "❌" in captured.out or "Failed to mine hashes" in captured.out

def test_labeler_run_creates_output_dirs(labeler: Labeler, tmp_path: Path) -> None:
    """Ensures nested output directories are created automatically."""
    deep_output = tmp_path / "deep" / "dir" / "gold.parquet"
    labeler.run(str(deep_output))
    assert deep_output.exists()
