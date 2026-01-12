
"""
Profile the three frequent‑itemset mining algorithms.

*  fim.apriori          – classic Apriori implementation (FIM library)
*  alg_eclat.EclatMiner – original ECLAT implementation you already have
*  alg_advanced_eclat.AdvancedEclatMiner – new, feature‑rich version

The script measures:
    • wall time
    • CPU user / system time
    • peak memory (tracemalloc)
    • RSS of the current process

All algorithms are wrapped in a small helper that accepts the *dataset* as its only argument,
so you can add or remove miners without touching the profiling logic.
"""

import time
import tracemalloc
from pathlib import Path

import psutil
from fim import apriori

# Local imports – adjust if your modules live elsewhere
from alg_advanced_eclat import AdvancedEclatMiner
from alg_eclat import EclatMiner
from data_manager import TransactionLoader


def profile_algorithm(
    algo_func,
    algo_name: str,
    data_path: Path,
) -> object:
    """
    Run *algo_func* on the dataset located at *data_path* and print profiling data.

    Parameters
    ----------
    algo_func : Callable[[List[Set[int]]], Any]
        Function that takes a list of transaction sets and returns an answer.
    algo_name : str
        Human‑readable name used in the output header.
    data_path : Path
        Path to the file containing the transactions.

    Returns
    -------
    object
        Whatever *algo_func* returned (e.g. a dict of frequent itemsets).
    """
    print(f"=== Profil dla {data_path.name} ({algo_name}) ===")

    # 1️⃣ Load the data
    loader = TransactionLoader()
    dataset = loader.load(data_path)

    # 2️⃣ Start measurement
    start_wall = time.perf_counter()
    tracemalloc.start()

    t = psutil.Process().cpu_times()
    cpu_user_start, cpu_system_start = t.user, t.system

    # 3️⃣ Run the algorithm
    answer = algo_func(dataset)

    # 4️⃣ Stop measurement
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    end_wall = time.perf_counter()
    t = psutil.Process().cpu_times()
    cpu_user_end, cpu_system_end = t.user, t.system

    rss_mb = psutil.Process().memory_info().rss / 1024 ** 2

    # 5️⃣ Print results
    print(f"Answer          : {answer}")
    print(f"Wall‑time       : {end_wall - start_wall:.6f}s")
    print(f"CPU user time   : {cpu_user_end - cpu_user_start:.6f}s")
    print(f"CPU system time : {cpu_system_end - cpu_system_start:.6f}s")
    print(f"Peak memory     : {peak / 1024:.2f} KiB (tracemalloc)")
    print(f"RSS process     : {rss_mb:.2f} MB\n")

    return answer


# ----------------------------------------------------------------------
# Helper wrappers – they convert the raw dataset into what each miner expects
# ----------------------------------------------------------------------
def run_eclat(dataset: list[set[int]]) -> dict[frozenset[int], int]:
    """Instantiate and run the original ECLAT miner."""
    miner = EclatMiner(min_support=0.2, dataset=dataset)
    return miner.find_frequent_itemsets()


def run_advanced_eclat(dataset: list[set[int]]) -> dict[frozenset[int], int]:
    """Instantiate and run the new AdvancedEclat miner."""
    miner = AdvancedEclatMiner(min_support=0.2, dataset=dataset)
    return miner.find_frequent_itemsets()


# ----------------------------------------------------------------------
# Main routine
# ----------------------------------------------------------------------
def main() -> None:
    data_folder = Path("data")

    # Define all algorithms – name + callable that accepts the *dataset*
    algorithms = [
        ("Apriori", lambda ds: apriori(ds)),
        ("ECLAT", run_eclat),
        ("Advanced ECLAT", run_advanced_eclat),
    ]

    for algo_name, algo_func in algorithms:
        print(f"=== Algorytm: {algo_name} ===")
        # The original script only processed retail.txt – keep that behaviour
        for file_path in data_folder.glob("*.txt"):
            profile_algorithm(algo_func, algo_name, file_path)


if __name__ == "__main__":
    main()