import pytest

from latentpool.provider import NodeProvider


@pytest.fixture
def url() -> str:
    """Shared RPC URL for all provider tests."""
    return "http://localhost:8545"

@pytest.fixture
async def provider(url: str):
    """
    Yields a provider and ensures the persistent client is closed after the test.
    """
    p = NodeProvider(url)
    yield p
    await p.aclose()
