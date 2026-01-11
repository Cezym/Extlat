from base_miner import BaseMiner
from data_manager import TransactionLoader


class EclatMiner(BaseMiner):
    def __init__(self, min_support: float, dataset: list[set[int]]):
        super().__init__(min_support, dataset)
        self.frequent_itemsets = {}

    def find_frequent_itemsets(self):

        loader = TransactionLoader()

        self.dataset = loader.to_vertical(self.dataset)



        frequent_items = []
        for item, tids in self.dataset.items():
            if len(tids) >= self.min_support_count:
                frequent_items.append((item, tids))

        frequent_items.sort(key=lambda x: x[0])

        self._explore_frequent_itemsets([], frequent_items)

        return self.frequent_itemsets

    def _explore_frequent_itemsets(self, prefix: list[int], items: list[tuple]):
        for i in range(len(items)):
            current_item, current_tids = items[i]

            new_itemset = prefix + [current_item]
            self.frequent_itemsets[frozenset(new_itemset)] = len(current_tids)

            next_level_candidates = []

            for j in range(i + 1, len(items)):
                next_item, next_tids = items[j]

                intersection_tids = current_tids & next_tids

                if len(intersection_tids) >= self.min_support_count:
                    next_level_candidates.append((next_item, intersection_tids))

            if next_level_candidates:
                self._explore_frequent_itemsets(new_itemset, next_level_candidates)


if __name__ == "__main__":
    # Symulacja danych: 4 transakcje
    dummy_data = [
        {1, 3, 4},
        {2, 3, 5},
        {1, 2, 3, 5},
        {2, 5}
    ]
    loader = TransactionLoader()
    data = loader.load(r"data/chess.txt")

    # Ustawiamy support na 0.5 (czyli 50% -> min. 2 wystąpienia)
    miner = EclatMiner(min_support=0.8, dataset=data)
    results = miner.find_frequent_itemsets()

    print(f"--- Wyniki dla danych testowych (Support: {miner.min_support_count}) ---")
    for itemset, count in results.items():
        print(f"Zbiór: {set(itemset)} | Liczba wystąpień: {count}")