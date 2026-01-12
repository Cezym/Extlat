from __future__ import annotations

from typing import List, Set, Dict, FrozenSet, Tuple
from base_miner import BaseMiner


class AdvancedEclatMiner(BaseMiner):
    """
    A feature‑rich implementation of the ECLAT frequent‑itemset miner.

    Parameters
    ----------
    min_support : float
        Minimum support threshold expressed as a fraction of the total
        number of transactions (e.g. 0.2 for 20%).
    dataset : List[Set[int]]
        The transaction database; each element is a set containing item IDs.
    sort_items_by_support : bool, optional
        If ``True`` (default) candidate items are sorted by their support count
        in ascending order before recursion – this usually speeds up pruning.
        Set to ``False`` if you want the original behaviour of sorting by item ID.

    Attributes
    ----------
    frequent_itemsets : Dict[FrozenSet[int], int]
        Mapping from each discovered frequent itemset (as a frozenset) to its absolute support count.
    """

    def __init__(
        self,
        min_support: float,
        dataset: List[Set[int]],
        *,
        sort_items_by_support: bool = True,
    ) -> None:
        super().__init__(min_support, dataset)
        self.frequent_itemsets: Dict[FrozenSet[int], int] = {}
        self.sort_items_by_support = sort_items_by_support

    # ----------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------
    def find_frequent_itemsets(self) -> Dict[FrozenSet[int], int]:
        """
        Execute the ECLAT algorithm and return all frequent itemsets.

        Returns
        -------
        Dict[FrozenSet[int], int]
            Mapping from each frequent itemset to its support count.
        """
        # 1 Convert the horizontal database into a vertical tid‑list.
        vertical = self._build_vertical()

        # 2 Keep only items that meet the minimum support threshold.
        candidate_items: List[Tuple[int, Set[int]]] = [
            (item, tids) for item, tids in vertical.items()
            if len(tids) >= self.min_support_count
        ]

        # 3 Sort candidates – either by support or by item ID.
        if self.sort_items_by_support:
            candidate_items.sort(key=lambda x: len(x[1]))
        else:
            candidate_items.sort(key=lambda x: x[0])

        # 4 Recursively enumerate all frequent itemsets.
        self._mine(prefix=[], items=candidate_items)
        return self.frequent_itemsets

    def pretty_print(self) -> None:
        """
        Print the discovered frequent itemsets in a human‑readable form.

        The output is sorted by decreasing support, then lexicographically
        by the itemset contents.
        """
        sorted_items = sorted(
            self.frequent_itemsets.items(),
            key=lambda kv: (-kv[1], tuple(sorted(kv[0])))
        )
        for itemset, count in sorted_items:
            print(f"Itemset: {sorted(itemset)}  Support: {count}")

    # ----------------------------------------------------------------------
    # Internals
    # ----------------------------------------------------------------------
    def _build_vertical(self) -> Dict[int, Set[int]]:
        """
        Build the vertical tid‑list representation.

        Returns
        -------
        Dict[int, Set[int]]
            Mapping from an item to the set of transaction IDs containing it.
        """
        vertical: Dict[int, Set[int]] = {}
        for tid, transaction in enumerate(self.dataset):
            for item in transaction:
                vertical.setdefault(item, set()).add(tid)
        return vertical

    def _mine(
        self,
        prefix: List[int],
        items: List[Tuple[int, Set[int]]]
    ) -> None:
        """
        Recursive depth‑first search that enumerates all frequent itemsets.

        Parameters
        ----------
        prefix : List[int]
            Current itemset being extended.
        items : List[Tuple[int, Set[int]]]
            Candidate items (item ID and its tid set) to be combined with the prefix.
        """
        for i, (curr_item, curr_tids) in enumerate(items):
            # New frequent itemset
            new_prefix = prefix + [curr_item]
            support_count = len(curr_tids)
            self.frequent_itemsets[frozenset(new_prefix)] = support_count

            # Build candidates for the next recursion level
            next_candidates: List[Tuple[int, Set[int]]] = []

            for j in range(i + 1, len(items)):
                nxt_item, nxt_tids = items[j]
                intersection = curr_tids & nxt_tids
                if len(intersection) >= self.min_support_count:
                    next_candidates.append((nxt_item, intersection))

            # Recurse only if we have candidates to extend with
            if next_candidates:
                # Keep the same ordering as before (by support or item ID)
                if self.sort_items_by_support:
                    next_candidates.sort(key=lambda x: len(x[1]))
                else:
                    next_candidates.sort(key=lambda x: x[0])
                self._mine(prefix=new_prefix, items=next_candidates)


# ----------------------------------------------------------------------
# Demo / usage example
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Dummy data – 4 transactions
    dummy_data = [
        {1, 3, 4},
        {2, 3, 5},
        {1, 2, 3, 5},
        {2, 5}
    ]

    # Load a real file (uncomment if you have a dataset)
    from data_manager import TransactionLoader
    loader = TransactionLoader()
    miner_dataset = loader.load(r"data/retail.txt")

    # For the demo we use the dummy data
    miner = AdvancedEclatMiner(min_support=0.1, dataset=miner_dataset)
    results = miner.find_frequent_itemsets()

    print(f"\n--- Results (Support threshold: {miner.min_support_count}) ---")
    for itemset, count in sorted(
        results.items(), key=lambda kv: (-kv[1], tuple(sorted(kv[0])))
    ):
        print(f"Itemset: {sorted(itemset)} | Support: {count}")
