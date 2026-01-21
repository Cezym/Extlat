import math
from abc import ABC, abstractmethod
from typing import List, Set


class BaseMiner(ABC):
    def __init__(self, min_support: float, dataset: List[Set[int]]):
        self.min_support = min_support
        self.dataset = dataset
        self.total_transactions = len(dataset) if dataset else 0
        self.min_support_count = math.ceil(min_support * self.total_transactions)

    def calculate_support(self, count: int, total_rows: int) -> float:
        if total_rows == 0:
            return 0.0
        return count / total_rows

    @abstractmethod
    def find_frequent_itemsets(self):
        pass
