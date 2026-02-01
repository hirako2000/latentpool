import os

from latentpool.configs.data_config import ArchiveConfig, DataConfig


def test_archive_config_env_substitution():
    """Verifies that ${ALCHEMY_API_KEY} is correctly replaced."""
    os.environ["ALCHEMY_API_KEY"] = "test_secret_key"

    config = ArchiveConfig(
        rpc_url="https://eth-mainnet.g.alchemy.com/v2/${ALCHEMY_API_KEY}"
    )

    assert "test_secret_key" in config.rpc_url
    assert "${ALCHEMY_API_KEY}" not in config.rpc_url

def test_archive_config_missing_env():
    """Verifies fallback when the environment variable is missing."""
    if "ALCHEMY_API_KEY" in os.environ:
        del os.environ["ALCHEMY_API_KEY"]

    config = ArchiveConfig(
        rpc_url="https://eth-mainnet.g.alchemy.com/v2/${ALCHEMY_API_KEY}"
    )

    assert "missing_key" in config.rpc_url

def test_data_config_defaults():
    """Verifies the parent struct and its nested default values."""
    baseline = ArchiveConfig(rpc_url="base")
    DEFAULT_RAW_DIR = "data/raw/traces"

    archive = ArchiveConfig(rpc_url="https://localhost:8545")
    data_cfg = DataConfig(archive_node=archive)

    assert data_cfg.raw_dir == DEFAULT_RAW_DIR
    assert data_cfg.archive_node.max_rps == baseline.max_rps
    assert data_cfg.archive_node.timeout == baseline.timeout
