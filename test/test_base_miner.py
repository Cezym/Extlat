import math

from src.base_miner import BaseMiner


class DummyMiner(BaseMiner):
    def find_frequent_itemsets(self):
        return {}


def test_min_support_count_ceil():
    """min_support_count must be rounded up."""
    miner = DummyMiner(min_support=0.25, dataset=[{1}, {2}])
    assert miner.min_support_count == math.ceil(0.25 * 2) == 1


def test_calculate_support_zero_rows():
    """When total_rows is zero we avoid division by zero."""
    miner = DummyMiner(min_support=0.5, dataset=[])
    assert miner.calculate_support(count=10, total_rows=0) == 0.0
