import time
import tracemalloc
from pathlib import Path

import psutil
from fim import apriori
from data_manager import TransactionLoader


def profile_algorithm(alg, data_path):
    print(f"=== Profil dla {data_path.name} ===")

    # 1. Ładujemy dane
    tl = TransactionLoader()
    tl.load(data_path)

    # 2. Rozpoczynamy pomiar
    start_wall = time.perf_counter()
    tracemalloc.start()

    t = psutil.Process().cpu_times()
    cpu_user_start, cpu_system_start = t.user, t.system

    # 3. Uruchamiamy algorytm
    answer = alg(tl.transactions)

    # 4. Kończymy pomiar
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    end_wall = time.perf_counter()
    t = psutil.Process().cpu_times()
    cpu_user_end, cpu_system_end = t.user, t.system

    rss_mb = psutil.Process().memory_info().rss / 1024 ** 2

    # 5. Wynik
    print(f"Answer          : {answer}")
    print(f"Wall‑time       : {end_wall - start_wall:.6f}s")
    print(f"CPU user time   : {cpu_user_end - cpu_user_start:.6f}s")
    print(f"CPU system time : {cpu_system_end - cpu_system_start:.6f}s")
    print(f"Peak memory     : {peak / 1024:.2f} KiB (tracemalloc)")
    print(f"RSS process     : {rss_mb:.2f} MB\n")

    return answer


def main():
    data_folder = Path("data")
    algorithms = [apriori]

    for alg in algorithms:
        print(f"=== Algorytm: {alg.__name__} ===")
        for file_path in data_folder.glob("*.txt"):
            profile_algorithm(alg, file_path)


if __name__ == "__main__":
    main()
