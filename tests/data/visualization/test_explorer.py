from pathlib import Path
from typing import Any, Tuple
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from latentpool.data.visualization.explorer import GraphExplorer

TX_HASH_VALID = "0x" + "a" * 64
TX_HASH_EMPTY = "0x" + "b" * 64
ADDR_FROM = "0x123"
ADDR_TO = "0x456"
ADDR_CONTRACT = "0xTOKEN_ADDRESS"
TRANSFER_VALUE = "1000"

PARQUET_FILE = "test_data.parquet"
OUTPUT_HTML = "test_graph.html"

EXPECTED_NODES = 2
EXPECTED_EDGES = 1

# For substring slicing check
TOKEN_LABEL_SLICE = 6

@pytest.fixture
def explorer_setup(tmp_path: Path) -> Tuple[str, GraphExplorer]:
    """Creates a temporary parquet and initializes the explorer."""
    path = tmp_path / PARQUET_FILE
    df: Any = pd.DataFrame({
        "tx_hash": [TX_HASH_VALID],
        "from": [ADDR_FROM],
        "to": [ADDR_TO],
        "token": [ADDR_CONTRACT],
        "value": [TRANSFER_VALUE]
    })
    df.to_parquet(path)
    return str(path), GraphExplorer(str(path))

@patch("latentpool.data.visualization.explorer.Network")
def test_generate_tx_graph_success(
    mock_network_class: MagicMock,
    explorer_setup: Tuple[str, GraphExplorer],
    capsys: Any
) -> None:
    """Verifies graph generation, edge labeling, and pyvis integration."""
    _, explorer = explorer_setup
    mock_net_instance = mock_network_class.return_value

    explorer.generate_tx_graph(TX_HASH_VALID, output_path=OUTPUT_HTML)

    assert mock_net_instance.from_nx.called

    generated_graph = mock_net_instance.from_nx.call_args[0][0]

    assert generated_graph.number_of_nodes() == EXPECTED_NODES
    assert generated_graph.number_of_edges() == EXPECTED_EDGES

    # edge attributes (labels and titles)
    edge_data = list(generated_graph.edges(data=True))[0][2]
    expected_label = f"Token: {ADDR_CONTRACT[:TOKEN_LABEL_SLICE]}..."
    assert edge_data["label"] == expected_label
    assert edge_data["title"] == f"Value: {TRANSFER_VALUE}"

    mock_net_instance.show.assert_called_with(OUTPUT_HTML)

    captured = capsys.readouterr()
    assert f"Visualization saved to {OUTPUT_HTML}" in captured.out

def test_generate_tx_graph_no_data(explorer_setup: Tuple[str, GraphExplorer], capsys: Any) -> None:
    """Covers the branch where a tx_hash does not exist in the dataframe."""
    _, explorer = explorer_setup

    explorer.generate_tx_graph(TX_HASH_EMPTY)

    captured = capsys.readouterr()
    assert f"No data found for hash {TX_HASH_EMPTY}" in captured.out

@patch("latentpool.data.visualization.explorer.Network")
def test_generate_tx_graph_case_insensitivity(
    mock_network_class: MagicMock,
    explorer_setup: Tuple[str, GraphExplorer]
) -> None:
    """Ensures hashes are handled case-insensitively."""
    _, explorer = explorer_setup

    explorer.generate_tx_graph(TX_HASH_VALID.upper())

    #  still processed? (from_nx should be called)
    mock_net_instance = mock_network_class.return_value
    assert mock_net_instance.from_nx.called

def test_init_loading(explorer_setup: Tuple[str, GraphExplorer]) -> None:
    """Verifies the dataframe is loaded correctly on init."""
    _, explorer = explorer_setup
    assert not explorer.df.empty
    assert explorer.df["tx_hash"].iloc[0] == TX_HASH_VALID
