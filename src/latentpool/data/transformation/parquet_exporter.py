import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

MIN_TRANSFER_TOPICS = 3
TRANSFER_EVENT_SIG_PREFIX = "0xddf2"


class ParquetExporter:
    """Converts raw JSON receipts into a structured Parquet edge list."""

    def __init__(self, raw_dir: str, processed_dir: str):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def process_all(self) -> None:
        """Iterates over all JSON receipts and extracts ERC-20 transfer events."""
        all_edges: List[Dict[str, Any]] = []
        json_files = list(self.raw_dir.glob("*.json"))

        if not json_files:
            print(f"❌ No JSON files found in {self.raw_dir}")
            return

        print(f"🔍 Found {len(json_files):,} files. Processing...")

        for json_file in tqdm(json_files, desc="Converting JSON to Edges"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    receipt: Dict[str, Any] = json.load(f)

                    tx_hash: Optional[str] = receipt.get("transactionHash")
                    raw_block: Optional[str] = receipt.get("blockNumber")

                    if not tx_hash or not raw_block:
                        continue

                    block_number = int(raw_block, 16)
                    logs: List[Dict[str, Any]] = receipt.get("logs", [])

                    for log in logs:
                        topics: List[str] = log.get("topics", [])

                        if (
                            len(topics) >= MIN_TRANSFER_TOPICS
                            and topics[0].lower().startswith(TRANSFER_EVENT_SIG_PREFIX)
                        ):
                            all_edges.append({
                                "tx_hash": tx_hash.lower(),
                                "from": f"0x{topics[1][-40:]}".lower(),
                                "to": f"0x{topics[2][-40:]}".lower(),
                                "token": str(log.get("address", "")).lower(),
                                # Store as float to allow numeric operations later
                                "value": float(int(log.get("data", "0x0"), 16))
                                        if log.get("data") not in (None, "0x") else 0.0,
                                "block_number": block_number
                            })
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.error("Error processing file %s: %s", json_file, e)
                continue

        if not all_edges:
            print("❌ No valid ERC-20 transfer edges found.")
            return

        print(f"📊 Creating DataFrame from {len(all_edges):,} edges...")

        pd_any: Any = pd
        df = pd_any.DataFrame(all_edges)

        val_col: Any = df["value"]
        block_col: Any = df["block_number"]
        hash_col: Any = df["tx_hash"]

        df["block_number"] = block_col.astype(int)
        df["value"] = val_col.astype(float)
        df["tx_hash"] = hash_col.astype(str)

        output_file = self.processed_dir / "edges.parquet"

        print(f"💾 Saving to {output_file}...")

        # to_parquet via the Any-typed dataframe to bypass overload checks
        df_any: Any = df
        df_any.to_parquet(
            output_file,
            engine="pyarrow",
            index=False
        )
        print(f"✅ Export complete: {len(df):,} rows.")
