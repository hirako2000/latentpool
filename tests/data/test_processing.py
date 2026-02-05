from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest
import torch

from latentpool.data.processing import GraphBuilder

TX_HASH_KEY = "tx_hash"
FROM_KEY = "from"
TO_KEY = "to"
VALUE_KEY = "value"
TOKEN_KEY = "token" # noqa: S105
LABEL_KEY = "label"
SPLIT_KEY = "split"

XFER_IN_IDX = 2
XFER_OUT_IDX = 3
VOL_IN_STD_IDX = 5
SOURCE_IDX = 8
SELF_XFER_IDX = 10
FEAT_DIM = 11

EXPECTED_ROWS = 2
EXPECTED_NODES = 3
EXPECTED_EDGES = 2

LABEL_ARB = 1
LABEL_NORMAL = 0
RAW_VAL_1 = "100.0"
RAW_VAL_2 = "200.0"
TX_HASH_SAMPLE = "0x" + "a" * 64
ADDR_A = "0xAAAA"
ADDR_B = "0xBBBB"
ADDR_C = "0xCCCC"

@pytest.fixture
def gold_parquet_path(tmp_path: Path) -> str:
    path = tmp_path / "gold.parquet"
    df_data: Any = {
        TX_HASH_KEY: [TX_HASH_SAMPLE, TX_HASH_SAMPLE],
        FROM_KEY: [ADDR_A, ADDR_B],
        TO_KEY: [ADDR_B, ADDR_C],
        TOKEN_KEY: ["WETH", "DAI"],
        VALUE_KEY: [RAW_VAL_1, RAW_VAL_2],
        LABEL_KEY: [LABEL_ARB, LABEL_ARB],
        SPLIT_KEY: ["train", "train"]
    }
    cast(Any, pd.DataFrame(df_data)).to_parquet(path)
    return str(path)

def test_build_and_save_logic(gold_parquet_path: str, tmp_path: Path) -> None:
    output_dir = tmp_path / "graphs"
    builder = GraphBuilder(gold_parquet_path)
    builder.build_and_save(str(output_dir))

    expected_file = output_dir / f"{TX_HASH_SAMPLE}.pt"
    assert expected_file.exists()

    data: Any = torch.load(expected_file, weights_only=False)
    x = cast(torch.Tensor, data.x)

    assert x.shape == (EXPECTED_NODES, FEAT_DIM)

    # A is only a "from", should be a source
    # Index 8 is is_source
    assert (x[:, SOURCE_IDX] == 1.0).any()

    # Address B has 1 in (Index 2) and 1 out (Index 3)
    mid_node_mask = (x[:, XFER_IN_IDX] == 1.0) & (x[:, XFER_OUT_IDX] == 1.0)
    assert mid_node_mask.any()

def test_log_volume_calculation(gold_parquet_path: str, tmp_path: Path) -> None:
    output_dir = tmp_path / "graphs_val"
    builder = GraphBuilder(gold_parquet_path)
    builder.build_and_save(str(output_dir))

    data: Any = torch.load(output_dir / f"{TX_HASH_SAMPLE}.pt", weights_only=False)
    x = cast(torch.Tensor, data.x)

    # non-zero/transformed
    assert (x[:, VOL_IN_STD_IDX] != 0).any()

def test_self_transfer_detection(tmp_path: Path) -> None:
    path = tmp_path / "self_xfer.parquet"
    tx_hash = "0x7777"
    addr_self = "0xSELF"

    df_data: Any = {
        TX_HASH_KEY: [tx_hash],
        FROM_KEY: [addr_self],
        TO_KEY: [addr_self],
        TOKEN_KEY: ["SELF_COIN"],
        VALUE_KEY: ["1.0"],
        LABEL_KEY: [LABEL_NORMAL],
        SPLIT_KEY: ["test"]
    }
    cast(Any, pd.DataFrame(df_data)).to_parquet(path)

    output_dir = tmp_path / "out_self"
    builder = GraphBuilder(str(path))
    builder.build_and_save(str(output_dir))

    data: Any = torch.load(output_dir / f"{tx_hash}.pt", weights_only=False)
    x = cast(torch.Tensor, data.x)

    assert x[0, SELF_XFER_IDX] == 1.0
