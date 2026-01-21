import pytest
from src.alg_eclat import EclatMiner


@pytest.fixture
def small_dataset():
    return [{1, 2}, {1, 3}, {2, 3}]


def test_basic_frequent_itemsets(small_dataset):
    """Typical case - supports > min_support."""
    miner = EclatMiner(min_support=0.33, dataset=small_dataset)
    results = miner.find_frequent_itemsets()
    expected = {
        frozenset([1]): 2,
        frozenset([2]): 2,
        frozenset([3]): 2,
        frozenset([1, 2]): 1,
        frozenset([1, 3]): 1,
        frozenset([2, 3]): 1,
    }
    assert results == expected


def test_no_frequent_items(small_dataset):
    """All itemsets have support < min_support -> empty result."""
    miner = EclatMiner(min_support=1.0, dataset=small_dataset)
    assert miner.find_frequent_itemsets() == {}


def test_single_transaction():
    """Dataset with one transaction -> all subsets are frequent."""
    dataset = [{1, 2, 3}]
    miner = EclatMiner(min_support=0.5, dataset=dataset)
    results = miner.find_frequent_itemsets()

    expected = {
        frozenset([1]): 1,
        frozenset([2]): 1,
        frozenset([3]): 1,
        frozenset([1, 2]): 1,
        frozenset([1, 3]): 1,
        frozenset([2, 3]): 1,
        frozenset([1, 2, 3]): 1,
    }
    assert results == expected


def test_single_transaction_singletons_only():
    """Only singletons should be frequent because pairs/triplets fall below threshold."""
    dataset = [{1, 2}, {1, 3}]
    miner = EclatMiner(min_support=0.75, dataset=dataset)
    results = miner.find_frequent_itemsets()

    expected = {frozenset([1]): 2}
    assert results == expected


@pytest.mark.parametrize("support", [0.0])
def test_support_zero(small_dataset, support):
    """min_support=0 -> min_support_count = 0 -> every subset qualifies."""
    miner = EclatMiner(min_support=support, dataset=small_dataset)
    results = miner.find_frequent_itemsets()
    # All singletons and pairs should be present (no triplet).
    assert frozenset([1]) in results
    assert frozenset([2]) in results
    assert frozenset([3]) in results
    assert frozenset([1, 2]) in results
