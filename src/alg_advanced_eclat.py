from typing import List, Set, Dict, FrozenSet, Tuple
from src.base_miner import BaseMiner

from bitarray import bitarray


class AdvancedEclatMiner(BaseMiner):

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
        self.num_transactions = len(dataset)

    def find_frequent_itemsets(self) -> Dict[FrozenSet[int], int]:

        bit_vertical = self._build_vertical_bitsets()

        candidate_items: List[Tuple[int, bitarray]] = []

        for item, bitmask in bit_vertical.items():
            support = bitmask.count()
            if support >= self.min_support_count:
                candidate_items.append((item, bitmask))

        if self.sort_items_by_support:
            candidate_items.sort(key=lambda x: x[1].count())
        else:
            candidate_items.sort(key=lambda x: x[0])

        self._mine_bitset(prefix=[], items=candidate_items)

        return self.frequent_itemsets

    def _build_vertical_bitsets(self) -> Dict[int, bitarray]:

        vertical_bits: Dict[int, bitarray] = {}

        for tid, transaction in enumerate(self.dataset):
            for item in transaction:
                if item not in vertical_bits:
                    ba = bitarray(self.num_transactions)
                    ba.setall(0)
                    vertical_bits[item] = ba


                vertical_bits[item][tid] = 1

        return vertical_bits

    def _mine_bitset(
        self, prefix: List[int], items: List[Tuple[int, bitarray]]
    ) -> None:

        for i in range(len(items)):
            curr_item, curr_mask = items[i]


            new_prefix = prefix + [curr_item]
            support_count = curr_mask.count()
            self.frequent_itemsets[frozenset(new_prefix)] = support_count

            next_candidates: List[Tuple[int, bitarray]] = []

            for j in range(i + 1, len(items)):
                nxt_item, nxt_mask = items[j]

                intersection_mask = curr_mask & nxt_mask

                if intersection_mask.count() >= self.min_support_count:
                    next_candidates.append((nxt_item, intersection_mask))

            if next_candidates:
                self._mine_bitset(prefix=new_prefix, items=next_candidates)


# Przykład użycia
if __name__ == "__main__":
    dummy_data = [{1, 3, 4}, {2, 3, 5}, {1, 2, 3, 5}, {2, 5}]

    print("--- Test Advanced Eclat (bitarray library) ---")
    try:
        miner = AdvancedEclatMiner(min_support=0.6, dataset=dummy_data)
        results = miner.find_frequent_itemsets()

        sorted_results = sorted(
            results.items(), key=lambda kv: (-kv[1], tuple(sorted(kv[0])))
        )

        for itemset, count in sorted_results:
            print(f"Itemset: {sorted(itemset)} | Support: {count}")

    except ImportError as e:
        print(e)
