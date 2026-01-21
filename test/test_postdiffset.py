import pytest
from src.alg_postdiffset import PostdiffsetMiner
from src.alg_eclat import EclatMiner


@pytest.fixture
def small_dataset():
    return [{1, 2}, {1, 3}, {2, 3}]


def test_same_output_as_eclat(small_dataset):
    """Both algorithms should produce identical results on the same data."""
    eclat_results = EclatMiner(
        min_support=0.33, dataset=small_dataset
    ).find_frequent_itemsets()
    postdiff_results = PostdiffsetMiner(
        min_support=0.33, dataset=small_dataset
    ).find_frequent_itemsets()
    assert postdiff_results == eclat_results


def test_no_frequent_items(small_dataset):
    """No frequent itemsets -> empty dictionary."""
    results = PostdiffsetMiner(
        min_support=1.0, dataset=small_dataset
    ).find_frequent_itemsets()
    assert results == {}


@pytest.mark.parametrize("support", [0.0])
def test_support_zero(small_dataset, support):
    """Support 0 should return all subsets (singletons and pairs)."""
    results = PostdiffsetMiner(
        min_support=support, dataset=small_dataset
    ).find_frequent_itemsets()
    assert frozenset([1]) in results
    assert frozenset([2]) in results
    assert frozenset([3]) in results
