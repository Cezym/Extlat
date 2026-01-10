import time, tracemalloc, psutil, csv
from pathlib import Path
from fim import apriori
from data_manager import TransactionLoader


def profile_algorithm(alg, data_path):
    tl = TransactionLoader()
    tl.load(data_path)

    start_wall = time.perf_counter()
    tracemalloc.start()
    proc = psutil.Process()

    answer = alg(tl.transactions)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    end_wall = time.perf_counter()
    rss = proc.memory_info().rss / 1024 ** 2
    wall_time = end_wall - start_wall

    return {
        "answer": answer,
        "wall_time": wall_time,
        "peak_mem_kib": peak / 1024,
        "rss_mb": rss,
    }


def main():
    data_folder = Path("data")
    algs = [apriori]

    results_csv = []
    for alg in algs:
        print(f"\n=== Algorithm: {alg.__name__} ===")
        for data_file in data_folder.glob("*.txt"):
            print(f"File: {data_file.name}")
            res = profile_algorithm(alg, data_file)
            print(f"  wall_time   : {res['wall_time']:.6f}s")
            print(f"  cpu_user    : {res['cpu_user']:.6f}s")
            print(f"  cpu_system  : {res['cpu_system']:.6f}s")
            print(f"  peak_mem_kb : {res['peak_mem_kib']:.2f} KiB")
            print(f"  rss_mb      : {res['rss_mb']:.2f} MB\n")

            # do csv
            results_csv.append({
                "algorithm": alg.__name__,
                "file": data_file.name,
                **res
            })

    # zapis do csv
    csv_path = Path("profiling_results.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results_csv[0].keys())
        writer.writeheader()
        writer.writerows(results_csv)


if __name__ == "__main__":
    main()
