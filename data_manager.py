import pathlib
from typing import Union, Dict, Set, List

class TransactionLoader:
    _instance = None

    def __new__(cls):
        """
        Wzorzec Singleton.
        Gwarantuje, że zawsze zwracany jest ten sam obiekt loadera.
        """
        if cls._instance is None:
            cls._instance = super(TransactionLoader, cls).__new__(cls)
        return cls._instance

    def load(self, path: Union[str, pathlib.Path]) -> List[Set[int]]:
        """
        Wczytuje plik i zwraca listę transakcji.
        """
        p = pathlib.Path(path)

        if not p.is_file():
            raise FileNotFoundError(f"File '{p}' does not exist")

        transactions = []

        with p.open("r", encoding="utf-8") as fh:
            for line_no, raw_line in enumerate(fh, start=1):
                stripped = raw_line.strip()
                if not stripped:
                    continue

                try:
                    items = {int(tok) for tok in stripped.split()}
                except ValueError as exc:
                    raise ValueError(f"Non‑integer token on line {line_no} of '{p}'") from exc

                transactions.append(items)

        return transactions

    def to_vertical(self, dataset: List[Set[int]]) -> Dict[int, Set[int]]:
        """
        Pobiera dataset, konwertuje go na format wertykalny i zwraca wynik.
        """
        vertical: Dict[int, Set[int]] = {}
        for tid, trans in enumerate(dataset):
            for item in trans:
                vertical.setdefault(item, set()).add(tid)
        return vertical