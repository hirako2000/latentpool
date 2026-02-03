from enum import Enum


class MEVType(Enum):
    NORMAL = 0
    ARBITRAGE = 1
    SANDWICH = 2

class EdgeSchema:
    TX_HASH = "tx_hash"
    FROM_ADDR = "from_address"
    TO_ADDR = "to_address"
    TOKEN_ADDR = "token_address" # noqa: S105
    VALUE = "value_raw"
    LOG_INDEX = "log_index"

    TYPES = {
        TX_HASH: "string",
        FROM_ADDR: "string",
        TO_ADDR: "string",
        TOKEN_ADDR: "string",
        VALUE: "float64", # need to normalize that for some GNN
        LOG_INDEX: "int32"
    }

NODE_FEATURE_MAP = {
    0: "eth_balance_change",    # n/a
    1: "token_in_count",        # Unique tokens received
    2: "token_out_count",       # Unique tokens sent
    3: "total_transfers_in",    # In-degree
    4: "total_transfers_out",   # Out-degree
    5: "is_contract",           # n/a
    6: "is_eoa",                # n/a
    7: "neighbor_count",        # Unique addresses interacted with
    8: "log_volume_in",         # Log volume received
    9: "log_volume_out",        # Log volume sent
    10: "net_token_flow",       # In - Out volume
    11: "is_source",            # Binary: No inputs
    12: "is_sink",              # Binary: No outputs
    13: "self_transfer_count",  # from == to
}
