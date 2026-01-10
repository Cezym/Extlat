import pathlib
from typing import Union, Dict, Set, List


class TransactionLoader:
    def __init__(self) -> None:
        self.transactions: List[Set[int]] = []
        self.vertical_map: Union[Dict[int, Set[int]], None] = None

    def load(self, path: Union[str, pathlib.Path]) -> None:
        """
        Read the file and populate :attr:`transactions`.

        Parameters
        ----------
        path
            Path to the input file.  The file must contain one transaction per
            line with items separated by whitespace.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        ValueError
            If a non‑integer token is encountered in any line.
        """
        p = pathlib.Path(path)

        if not p.is_file():
            raise FileNotFoundError(f"File '{p}' does not exist")

        self.transactions.clear()

        with p.open("r", encoding="utf-8") as fh:
            for line_no, raw_line in enumerate(fh, start=1):
                stripped = raw_line.strip()
                if not stripped:  # skip empty lines
                    continue

                try:
                    items = {int(tok) for tok in stripped.split()}
                except ValueError as exc:
                    raise ValueError(f"Non‑integer token on line {line_no} of '{p}'") from exc

                self.transactions.append(items)

        # Invalidate the vertical cache – data has changed
        self.vertical_map = None

    def to_vertical(self) -> Dict[int, Set[int]]:
        """
        Convert transactions into a vertical map.

        Returns
        -------
        dict[int, set[int]]
            Mapping from an item id to the set of transaction IDs (TIDs)
            where that item occurs.  Transaction IDs are zero‑based indices
            corresponding to the order in ``self.transactions``.
        """
        if self.vertical_map is not None:
            return self.vertical_map

        vertical: Dict[int, Set[int]] = {}
        for tid, trans in enumerate(self.transactions):
            for item in trans:
                vertical.setdefault(item, set()).add(tid)

        self.vertical_map = vertical
        return vertical

if __name__ == "__main__":
    transaction_loader = TransactionLoader()
    transaction_loader.load("data/chess.txt")
    print(transaction_loader.transactions[:10])
    print(transaction_loader.to_vertical()[1])