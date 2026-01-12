from typing import List, Set, Dict, FrozenSet, Tuple
from base_miner import BaseMiner

from bitarray import bitarray



class AdvancedEclatMiner(BaseMiner):
    """
    Implementacja algorytmu Advanced Eclat wykorzystująca bibliotekę `bitarray`.

    Zalety względem int (native Python):
    - Jawna kontrola nad pamięcią (1 bit to fizycznie 1 bit w pamięci).
    - Metody dedykowane do operacji na bitach.

    Kluczowe zmiany:
    1. Transpozycja: Używamy obiektu bitarray(length) zamiast int.
    2. Operacje: Używamy operatora & (AND) na obiektach bitarray.
    3. Zliczanie: Używamy metody .count() zamiast .bit_count().
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
        # Musimy znać liczbę transakcji, aby zainicjować wektory o odpowiedniej długości
        self.num_transactions = len(dataset)

    def find_frequent_itemsets(self) -> Dict[FrozenSet[int], int]:
        """
        Uruchamia algorytm Advanced Eclat (wersja bitarray).
        """
        # Krok 1: Transformacja do formatu wertykalnego (Item -> bitarray)

        bit_vertical = self._build_vertical_bitsets()

        # Krok 2: Filtracja L1
        candidate_items: List[Tuple[int, bitarray]] = []

        for item, bitmask in bit_vertical.items():
            # bitarray posiada metodę .count() zwracającą liczbę jedynek
            support = bitmask.count()
            if support >= self.min_support_count:
                candidate_items.append((item, bitmask))

        # Sortowanie (Least Frequent First lub wg ID)
        if self.sort_items_by_support:
            candidate_items.sort(key=lambda x: x[1].count())
        else:
            candidate_items.sort(key=lambda x: x[0])

        # Krok 3: Rekurencyjne przeszukiwanie (DFS)

        self._mine_bitset(prefix=[], items=candidate_items)

        return self.frequent_itemsets

    def _build_vertical_bitsets(self) -> Dict[int, bitarray]:
        """
        Tworzy bazę wertykalną używając obiektów bitarray.
        """
        vertical_bits: Dict[int, bitarray] = {}

        for tid, transaction in enumerate(self.dataset):
            for item in transaction:
                # Jeśli widzimy produkt pierwszy raz, inicjalizujemy dla niego bitarray
                if item not in vertical_bits:
                    # Tworzymy wektor o długości równej liczbie transakcji
                    ba = bitarray(self.num_transactions)
                    ba.setall(0)  # Zerujemy wszystkie bity
                    vertical_bits[item] = ba

                # Ustawiamy bit odpowiadający ID transakcji na 1
                vertical_bits[item][tid] = 1

        return vertical_bits

    def _mine_bitset(
            self,
            prefix: List[int],
            items: List[Tuple[int, bitarray]]
    ) -> None:
        """
        Rekurencyjna procedura DFS na obiektach bitarray.
        """
        for i in range(len(items)):
            curr_item, curr_mask = items[i]

            # Zapisujemy wynik
            new_prefix = prefix + [curr_item]
            support_count = curr_mask.count()
            self.frequent_itemsets[frozenset(new_prefix)] = support_count

            next_candidates: List[Tuple[int, bitarray]] = []

            for j in range(i + 1, len(items)):
                nxt_item, nxt_mask = items[j]

                # Operacja bitowa AND na obiektach bitarray

                intersection_mask = curr_mask & nxt_mask

                # Sprawdzenie wsparcia
                if intersection_mask.count() >= self.min_support_count:
                    next_candidates.append((nxt_item, intersection_mask))

            if next_candidates:
                self._mine_bitset(prefix=new_prefix, items=next_candidates)


# ----------------------------------------------------------------------
# Przykład użycia
# ----------------------------------------------------------------------
if __name__ == "__main__":
    dummy_data = [
        {1, 3, 4},
        {2, 3, 5},
        {1, 2, 3, 5},
        {2, 5}
    ]

    print("--- Test Advanced Eclat (bitarray library) ---")
    try:
        miner = AdvancedEclatMiner(min_support=0.6, dataset=dummy_data)
        results = miner.find_frequent_itemsets()

        sorted_results = sorted(
            results.items(),
            key=lambda kv: (-kv[1], tuple(sorted(kv[0])))
        )

        for itemset, count in sorted_results:
            print(f"Itemset: {sorted(itemset)} | Support: {count}")

    except ImportError as e:
        print(e)