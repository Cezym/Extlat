import pytest

# Import everything that will be reused across tests
try:
    from src.base_miner import BaseMiner
except Exception:
    BaseMiner = None


@pytest.fixture
def dummy_algo():
    """A minimal miner that just sleeps and allocates a tiny list."""

    class DummyMiner(BaseMiner):
        def find_frequent_itemsets(self):
            import time

            time.sleep(0.05)  # fake work
            _ = [0] * 10000  # allocate memory
            return {}

    return DummyMiner


@pytest.fixture
def tmp_dataset(tmp_path):
    """Create a tiny transaction file for the benchmark tests."""
    data_file = tmp_path / "data.txt"
    data_file.write_text("1 2\n3")
    return str(data_file)
