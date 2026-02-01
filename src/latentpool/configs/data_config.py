import os

import msgspec


class ArchiveConfig(msgspec.Struct):
    rpc_url: str
    max_rps: int = 20
    timeout: float = 60.0
    retry_count: int = 5
    initial_backoff: float = 1.0

    def __post_init__(self):
        if "${ALCHEMY_API_KEY}" in self.rpc_url:
            key = os.getenv("ALCHEMY_API_KEY", "missing_key")
            self.rpc_url = self.rpc_url.replace("${ALCHEMY_API_KEY}", key)

class DataConfig(msgspec.Struct):
    archive_node: ArchiveConfig
    raw_dir: str = "data/raw/traces"
