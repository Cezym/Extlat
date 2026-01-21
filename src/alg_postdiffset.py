from src.base_miner import BaseMiner
from src.data_manager import TransactionLoader


class PostdiffsetMiner(BaseMiner):
    def __init__(self, min_support: float, dataset: list[set[int]]):
        super().__init__(min_support, dataset)
        self.frequent_itemsets = {}

    def find_frequent_itemsets(self):
        loader = TransactionLoader()

        self.dataset = loader.to_vertical(self.dataset)

        frequent_l1 = []
        for item, tids in self.dataset.items():
            support = len(tids)
            if support >= self.min_support_count:
                self.frequent_itemsets[frozenset([item])] = support
                frequent_l1.append((item, tids, support))

        frequent_l1.sort(key=lambda x: x[0])

        self._first_loop_tidset(frequent_l1)

        return self.frequent_itemsets

    def _first_loop_tidset(self, items: list):
        for i in range(len(items)):
            item_i, tids_i, support_i = items[i]
            candidates_for_diffset = []

            for j in range(i + 1, len(items)):
                item_j, tids_j, support_j = items[j]


                intersection_tids = tids_i & tids_j
                new_support = len(intersection_tids)

                if new_support >= self.min_support_count:
                    new_itemset = [item_i, item_j]
                    self.frequent_itemsets[frozenset(new_itemset)] = new_support

                    candidates_for_diffset.append(
                        (item_j, intersection_tids, new_support)
                    )

            if candidates_for_diffset:
                self._next_loops_diffset([item_i], candidates_for_diffset)

    def _next_loops_diffset(self, prefix: list, items: list):
        for i in range(len(items)):
            item_i, set_i, support_i = items[i]
            new_prefix = prefix + [item_i]
            next_level_candidates = []

            for j in range(i + 1, len(items)):
                item_j, set_j, support_j = items[j]

                diff = set_i - set_j
                new_support = support_i - len(diff)

                if new_support >= self.min_support_count:
                    full_itemset = new_prefix + [item_j]
                    self.frequent_itemsets[frozenset(full_itemset)] = new_support
                    next_level_candidates.append((item_j, diff, new_support))

            if next_level_candidates:
                self._next_loops_diffset(new_prefix, next_level_candidates)


if __name__ == "__main__":
    # Test
    dummy_data = [{1, 3, 4}, {2, 3, 5}, {1, 2, 3, 5}, {2, 5}]
    loader = TransactionLoader()
    data = loader.load(r"data/chess.txt")


    miner = PostdiffsetMiner(min_support=0.8, dataset=data)
    results = miner.find_frequent_itemsets()

    print(f"--- Wyniki dla danych testowych (Support: {miner.min_support_count}) ---")
    for itemset, count in results.items():
        print(f"Zbiór: {set(itemset)} | Liczba wystąpień: {count}")
