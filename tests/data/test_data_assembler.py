from pathlib import Path
from typing import Any, Dict, cast

import numpy as np
import pandas as pd
import pytest

from latentpool.data.data_assembler import DatasetAssembler, MEVLabel


class TestConfig:
    HASH_PREFIX = "0x"
    HEX_LEN = 64
    MEV_COUNT = 1
    NORMAL_COUNT = 5
    MAX_NORMAL_RATIO = 2.0

@pytest.fixture
def assembler_setup(tmp_path: Path) -> Dict[str, Any]:
    edges_path = tmp_path / "silver.parquet"
    arb_csv = tmp_path / "arb.csv"
    sand_csv = tmp_path / "sand.csv"
    output_path = tmp_path / "gold.parquet"

    # 1 MEV, 5 Normals
    hashes = [f"{TestConfig.HASH_PREFIX}{str(i).zfill(TestConfig.HEX_LEN)}" for i in range(6)]

    df = pd.DataFrame({
        "tx_hash": hashes,
        "block_number": np.arange(6, dtype=np.int64)
    })

    df_any: Any = df
    df_any.to_parquet(edges_path, index=False)

    arb_csv.write_text(hashes[0])
    sand_csv.write_text("")

    return {
        "assembler": DatasetAssembler(str(edges_path), str(arb_csv), str(sand_csv)),
        "output": str(output_path),
        "mev_hash": hashes[0]
    }

def test_pipeline_execution_and_labeling(assembler_setup: Dict[str, Any]):
    assembler: DatasetAssembler = assembler_setup["assembler"]
    output: str = assembler_setup["output"]

    # ratios to exercise the downsampling
    assembler.run(output, max_normal_ratio=TestConfig.MAX_NORMAL_RATIO)

    pd_any: Any = pd
    res: Any = pd_any.read_parquet(output)

    assert Path(output).exists()
    assert "label" in res.columns

    # labeling for the MEV hash
    mev_row = res[res["tx_hash"] == assembler_setup["mev_hash"]]
    assert int(mev_row["label"].iloc[0]) == MEVLabel.ARBITRAGE.value

def test_label_coverage_sandwich(tmp_path: Path):
    edges_path = tmp_path / "sand.parquet"
    sand_csv = tmp_path / "sand.csv"
    h = f"{TestConfig.HASH_PREFIX}{'f' * TestConfig.HEX_LEN}"

    df = pd.DataFrame({"tx_hash": [h], "block_number": [1000]})
    cast(Any, df).to_parquet(edges_path, index=False)

    sand_csv.write_text(h)

    assembler = DatasetAssembler(str(edges_path), "empty.csv", str(sand_csv))
    out = str(tmp_path / "out.parquet")
    assembler.run(out)

    res: Any = cast(Any, pd).read_parquet(out)
    assert int(res["label"].iloc[0]) == MEVLabel.SANDWICH.value

def test_early_exit_no_blocks(tmp_path: Path):
    empty_path = tmp_path / "empty.parquet"
    df = pd.DataFrame(columns=["tx_hash", "block_number"])
    cast(Any, df).to_parquet(empty_path, index=False)

    assembler = DatasetAssembler(str(empty_path), "a.csv", "s.csv")
    out_file = tmp_path / "void.parquet"

    assembler.run(str(out_file))
    assert not out_file.exists()

def test_missing_silver_file():
    assembler = DatasetAssembler("missing.parquet", "a.csv", "s.csv")
    with pytest.raises(FileNotFoundError):
        assembler.run("out.parquet")
