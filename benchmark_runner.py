import time
import gc
from pathlib import Path

import psutil
import threading
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Type, Tuple

# --- IMPORTY KLAS BAZOWYCH ---
from data_manager import TransactionLoader
from base_miner import BaseMiner
import pandas as pd

# --- IMPORTY ALGORYTMÓW ---
try:
    from alg_eclat import EclatMiner
    from alg_postdiffset import PostdiffsetMiner
    from alg_advanced_eclat import AdvancedEclatMiner
except ImportError as e:
    print(f"UWAGA: Nie znaleziono pliku z algorytmem: {e}")

# --- IMPORT BIBLIOTEKI FIM ---
try:
    import fim

    FIM_AVAILABLE = True
except ImportError:
    print("UWAGA: Biblioteka 'fim' nie jest zainstalowana (pip install fim).")
    FIM_AVAILABLE = False


# --- ADAPTERY DLA BIBLIOTEKI FIM ---
class DirectFimApriori(BaseMiner):
    def find_frequent_itemsets(self):
        results = fim.apriori(self.dataset, supp=-self.min_support_count, report="s")
        return len(results)


class DirectFimEclat(BaseMiner):
    def find_frequent_itemsets(self):
        results = fim.eclat(self.dataset, supp=-self.min_support_count, report="s")
        return len(results)


class BenchmarkRunner:
    def __init__(self, algorithms: Dict[str, Type[BaseMiner]]):
        self.algorithms = algorithms
        self.results = pd.DataFrame(
            columns=["iteration", "dataset", "algorithm", "support", "time", "memory"]
        )
        self.results_avg: pd.DataFrame

    def measure_execution(
        self,
        algorithm_class: Type[BaseMiner],
        dataset: list[set[int]],
        min_support: float,
    ):
        """
        Mierzy czas oraz RZECZYWISTE zużycie RAM całego procesu (w tym C extensions).
        Używa osobnego wątku do monitorowania szczytowego zużycia (Peak RSS).
        """
        # 1. Sprzątanie przed testem
        gc.collect()
        time.sleep(0.2)  # Krótka pauza dla stabilizacji systemu

        # 2. Pobranie procesu i pamięci początkowej (Baseline)
        process = psutil.Process(os.getpid())
        baseline_mem = (
            process.memory_info().rss
        )  # RSS = Resident Set Size (Fizyczna pamięć)

        # Zmienne współdzielone z wątkiem monitorującym
        memory_stats = {"peak": baseline_mem}
        stop_event = threading.Event()

        # 3. Definicja funkcji monitorującej (działa w tle)
        def monitor_memory():
            while not stop_event.is_set():
                # Pobieramy aktualne zużycie
                current_mem = process.memory_info().rss
                # Aktualizujemy szczyt, jeśli jest wyższy
                if current_mem > memory_stats["peak"]:
                    memory_stats["peak"] = current_mem
                # Próbkujemy co 1ms (bardzo często, żeby złapać "szpilki" pamięciowe)
                time.sleep(0.001)

        # 4. Start wątku monitorującego
        monitor_thread = threading.Thread(target=monitor_memory)
        monitor_thread.start()

        # 5. Uruchomienie algorytmu (Główny pomiar)
        start_time = time.time()
        try:
            # Tworzenie instancji i uruchomienie
            miner = algorithm_class(min_support, dataset)
            miner.find_frequent_itemsets()
        finally:
            # Zatrzymujemy monitorowanie niezależnie od błędów
            stop_event.set()
            monitor_thread.join()

        end_time = time.time()

        # 6. Obliczenia
        execution_time = end_time - start_time

        # Net Memory Usage = Szczyt - Pamięć Bazowa (ile algorytm 'dodał' do procesu)
        # Czasem lepiej pokazywać Peak absolutny, ale Net jest lepszy do porównań
        peak_memory_bytes = memory_stats["peak"] - baseline_mem

        # Zabezpieczenie: Jeśli algorytm był super szybki lub zwolnił pamięć, wynik może być < 0
        if peak_memory_bytes < 0:
            peak_memory_bytes = 0

        peak_memory_mb = peak_memory_bytes / (1024 * 1024)  # Konwersja na MB

        return execution_time, peak_memory_mb

    def run_comparison(
        self,
        datasets_supports: Tuple[Tuple[Dict, List]],
        output_file_path: str | Path,
        iter: int = 1,
    ):
        for i in range(1, iter + 1):
            print(f"-=== Iteracja {i}/{iter} ===-")
            loader = TransactionLoader()

            for datasets, support_range in datasets_supports:
                for data_name, data_path in datasets.items():
                    print("\n==========================================")
                    print(f" ZBIÓR: {data_name} ({data_path})")
                    print("==========================================")

                    if not os.path.exists(data_path):
                        print("BŁĄD: Plik nie istnieje!")
                        continue

                    current_dataset = loader.load(data_path)
                    print(f"-> Załadowano {len(current_dataset)} transakcji.")

                    for algo_name, algo_class in self.algorithms.items():
                        print(f"\n>>> Algorytm: {algo_name}")

                        for support in support_range:
                            print(f"    MinSup: {support:<4} ... ", end="", flush=True)

                            try:
                                exec_time, mem_net_peak = self.measure_execution(
                                    algo_class, current_dataset, support
                                )

                                new_row = {
                                    "iteration": i,
                                    "dataset": data_name,
                                    "algorithm": algo_name,
                                    "support": support,
                                    "time": exec_time,
                                    "memory": mem_net_peak,
                                }

                                self.results.loc[len(self.results)] = new_row
                                print(
                                    f"OK | Czas: {exec_time:.4f}s | RAM (Net Peak): {mem_net_peak:.2f}MB"
                                )
                            except Exception as e:
                                print(f"BŁĄD ({e})")
                                import traceback

                                traceback.print_exc()

        # Save the DataFrame to CSV, creating necessary directories if they don't exist
        output_dir = os.path.dirname(output_file_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.results.to_csv(output_file_path, index=False)
        self.results_avg = self.results.groupby(
            ["dataset", "algorithm", "support"], as_index=False
        ).agg(
            time=("time", "mean"),
            memory=("memory", "mean"),
        )

    def plot_results(self, metric="time"):
        if self.results_avg.empty:
            print("Brak wyników do wyświetlenia.")
            return

        if metric == "time":
            y_label = "Czas wykonania (s)"
            title_prefix = "Wydajność Czasowa"
        else:
            y_label = "Przyrost RAM (MB)"
            title_prefix = "Zużycie Pamięci (Net Peak)"

        for data_name, algo_results in self.results_avg.groupby("dataset"):
            plt.figure(figsize=(12, 7))

            for algo_name, data_points in algo_results.groupby("algorithm"):
                sorted_points = data_points.sort_values(by="support", ascending=False)

                supports = sorted_points["support"].tolist()
                values = sorted_points[metric].tolist()

                if supports:
                    is_library = "FIM" in algo_name or "Lib" in algo_name
                    line_style = "--" if is_library else "-"
                    marker = "x" if is_library else "o"

                    plt.plot(
                        supports,
                        values,
                        marker=marker,
                        linestyle=line_style,
                        label=algo_name,
                        linewidth=2,
                    )

            plt.title(f"{title_prefix}: {data_name}")
            plt.xlabel("Minimum Support")
            plt.ylabel(y_label)
            plt.legend()
            plt.grid(True, which="both", linestyle="--", alpha=0.7)
            plt.gca().invert_xaxis()
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    # 1. Algorytmy
    algos_to_test = {
        "My Eclat": EclatMiner,
        "My Postdiffset": PostdiffsetMiner,
        "My Adv. Eclat": AdvancedEclatMiner,
    }

    if FIM_AVAILABLE:
        algos_to_test["FIM Apriori"] = DirectFimApriori
        algos_to_test["FIM Eclat"] = DirectFimEclat

    # 2. Dane
    datasets_rare_map = {
        "Retail": "data/retail.txt",
        "Kosarak": "data/kosarak.dat.txt",
    }

    datasets_dense_map = {
        "Chess": "data/chess.txt",
        "Mushrooms": "data/mushrooms.txt",
    }

    # 3. Supporty
    supports_rare = [i / 10 for i in range(1, 10)]
    supports_dense = [i / 10 for i in range(7, 10)]

    # 4. Start
    runner = BenchmarkRunner(algos_to_test)
    runner.run_comparison(
        ((datasets_rare_map, supports_rare), (datasets_dense_map, supports_dense)),
        "results/results.csv",
        10,
    )

    print("\nRysowanie wykresów...")
    runner.plot_results(metric="time")
    runner.plot_results(metric="memory")
