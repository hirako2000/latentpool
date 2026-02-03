from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest
import torch
from torch_geometric.data import Data

from latentpool.data.processing import GraphBuilder

TX_HASH_KEY = "tx_hash"
FROM_KEY = "from"
TO_KEY = "to"
VALUE_KEY = "value"
TOKEN_KEY = "token"  # noqa: S105
LABEL_KEY = "label"
SPLIT_KEY = "split"

VAL_TRAIN = "train"
VAL_TEST = "test"

IN_COUNT_IDX = 3    # total_transfers_in
OUT_COUNT_IDX = 4   # total_transfers_out
VOL_IN_IDX = 8      # log_volume_in
SOURCE_IDX = 11     # is_source
SELF_XFER_IDX = 13  # self_transfer_count
FEAT_DIM = 14

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
    """Creates a dummy gold parquet with the updated schema."""
    path = tmp_path / "gold.parquet"
    df_data: Any = {
        TX_HASH_KEY: [TX_HASH_SAMPLE, TX_HASH_SAMPLE],
        FROM_KEY: [ADDR_A, ADDR_B],
        TO_KEY: [ADDR_B, ADDR_C],
        TOKEN_KEY: ["WETH", "DAI"],
        VALUE_KEY: [RAW_VAL_1, RAW_VAL_2],
        LABEL_KEY: [LABEL_ARB, LABEL_ARB],
        SPLIT_KEY: [VAL_TRAIN, VAL_TRAIN]
    }
    df: Any = pd.DataFrame(df_data)
    df.to_parquet(path)
    return str(path)


def test_graph_builder_init(gold_parquet_path: str) -> None:
    builder = GraphBuilder(gold_parquet_path)
    assert len(builder.df) == EXPECTED_ROWS
    assert TX_HASH_KEY in builder.df.columns


def test_build_and_save_logic(gold_parquet_path: str, tmp_path: Path) -> None:
    output_dir = tmp_path / "graphs"
    builder = GraphBuilder(gold_parquet_path)
    builder.build_and_save(str(output_dir))

    expected_file = output_dir / f"{TX_HASH_SAMPLE}.pt"
    assert expected_file.exists()

    data: Any = torch.load(expected_file, weights_only=False)
    assert isinstance(data, Data)

    x = cast(torch.Tensor, data.x)
    edge_index = cast(torch.Tensor, data.edge_index)

    assert data.num_nodes == EXPECTED_NODES
    assert x.shape == (EXPECTED_NODES, FEAT_DIM)
    assert edge_index.shape[1] == EXPECTED_EDGES

    assert (x[:, SOURCE_IDX] == 1.0).any()

    # Address B should have 1 in (Index 3) and 1 out (Index 4)
    mid_node_mask = (x[:, IN_COUNT_IDX] == 1.0) & (x[:, OUT_COUNT_IDX] == 1.0)
    assert mid_node_mask.any()


def test_log_volume_calculation(gold_parquet_path: str, tmp_path: Path) -> None:
    output_dir = tmp_path / "graphs_val"
    builder = GraphBuilder(gold_parquet_path)
    builder.build_and_save(str(output_dir))

    data: Any = torch.load(output_dir / f"{TX_HASH_SAMPLE}.pt", weights_only=False)
    x = cast(torch.Tensor, data.x)

    val_float = float(RAW_VAL_1)
    expected_log = torch.tensor(val_float + 1.0).log()

    # allow small float precision errors
    assert torch.isclose(x[:, VOL_IN_IDX], expected_log, atol=1e-4).any()


def test_self_transfer_detection(tmp_path: Path) -> None:
    """Explicitly tests the 'if u == v' branch for feature index 13."""
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
        SPLIT_KEY: [VAL_TEST]
    }
    df: Any = pd.DataFrame(df_data)
    df.to_parquet(path)

    output_dir = tmp_path / "out_self"
    builder = GraphBuilder(str(path))
    builder.build_and_save(str(output_dir))

    data: Any = torch.load(output_dir / f"{tx_hash}.pt", weights_only=False)
    x = cast(torch.Tensor, data.x)

    # In a self-transfer with one row, the single node should have 1.0 at index 13
    assert x[0, SELF_XFER_IDX] == 1.0
