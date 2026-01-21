import pytest

# Import only if bitarray is available - otherwise skip the whole module.
try:
    from src.alg_advanced_eclat import AdvancedEclatMiner
except Exception:
    AdvancedEclatMiner = None


@pytest.fixture
def dummy_dataset():
    return [{1, 2}, {2, 3}]


@pytest.mark.skipif(AdvancedEclatMiner is None, reason="bitarray not installed")
def test_basic_frequent_itemsets(dummy_dataset):
    """Verify that the bitarray implementation reproduces the expected support counts."""
    miner = AdvancedEclatMiner(min_support=0.5, dataset=dummy_dataset)
    results = miner.find_frequent_itemsets()
    expected = {
        frozenset([1]): 1,
        frozenset([2]): 2,
        frozenset([3]): 1,
        frozenset([1, 2]): 1,
        frozenset([2, 3]): 1,
    }
    assert results == expected


@pytest.mark.skipif(AdvancedEclatMiner is None, reason="bitarray not installed")
def test_sort_items_by_support_flag():
    """When sorting by support the algorithm still returns correct itemsets."""
    dataset = [{1}, {1, 2}, {1, 2, 3}]
    miner_sorted = AdvancedEclatMiner(
        min_support=0.5, dataset=dataset, sort_items_by_support=True
    )
    sorted_results = miner_sorted.find_frequent_itemsets()

    miner_unsorted = AdvancedEclatMiner(
        min_support=0.5, dataset=dataset, sort_items_by_support=False
    )
    unsorted_results = miner_unsorted.find_frequent_itemsets()

    # The two runs must produce identical results
    assert sorted_results == unsorted_results


@pytest.mark.skipif(AdvancedEclatMiner is None, reason="bitarray not installed")
def test_empty_dataset():
    """Empty dataset -> empty result - no crash."""
    miner = AdvancedEclatMiner(min_support=0.5, dataset=[])
    assert miner.find_frequent_itemsets() == {}
