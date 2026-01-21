import time
import gc
from pathlib import Path
import pandas as pd
import psutil
import threading
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Type, Any
import logging

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO)

# --- IMPORTY KLAS BAZOWYCH ---
from src.data_manager import TransactionLoader
from src.base_miner import BaseMiner

# --- IMPORTY ALGORYTMÓW ---
try:
    from alg_eclat import EclatMiner
    from alg_postdiffset import PostdiffsetMiner
    from alg_advanced_eclat import AdvancedEclatMiner
except ImportError as e:
    logging.warning(f"UWAGA: Nie znaleziono pliku z algorytmem: {e}")

# --- IMPORT BIBLIOTEKI FIM ---
try:
    import fim

    FIM_AVAILABLE = True
except ImportError:
    logging.warning("UWAGA: Biblioteka 'fim' nie jest zainstalowana (pip install fim).")
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
        peak_memory_bytes = memory_stats["peak"] - baseline_mem

        # Zabezpieczenie: Jeśli algorytm był super szybki lub zwolnił pamięć, wynik może być < 0
        if peak_memory_bytes < 0:
            peak_memory_bytes = 0

        peak_memory_mb = peak_memory_bytes / (1024 * 1024)  # Konwersja na MB

        return execution_time, peak_memory_mb

    def run_comparison(
        self,
        datasets_config: List[Dict[str , Any]],
        output_file_path: str | Path,
        iter: int = 1,
        log_file_path: str | Path = None,
    ):
        # Konfiguracja loggera jeśli podano ścieżkę do pliku logu
        if log_file_path:
            file_handler = logging.FileHandler(log_file_path)
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            file_handler.setFormatter(formatter)
            logger = logging.getLogger(__name__)
            logger.addHandler(file_handler)

        for i in range(1, iter + 1):
            if log_file_path:
                logger.info(f"-=== Iteracja {i}/{iter} ===-")

            loader = TransactionLoader()

            for dataset_config in datasets_config:
                data_name = dataset_config["name"]
                data_path = dataset_config["dataset_path"]
                support_range = dataset_config["min_supports"]
                if log_file_path:
                    logger.info(
                        "\n==========================================\n"
                        + f" ZBIÓR: {data_name} ({data_path})\n"
                        + "=========================================="
                    )

                if not os.path.exists(data_path):
                    if log_file_path:
                        logger.error("BŁĄD: Plik nie istnieje!")
                    continue

                current_dataset = loader.load(data_path)
                if log_file_path:
                    logger.info(f"-> Załadowano {len(current_dataset)} transakcji.")

                for algo_name, algo_class in self.algorithms.items():
                    if log_file_path:
                        logger.info(f"\n>>> Algorytm: {algo_name}")

                    for support in support_range:
                        if log_file_path:
                            logger.info(f"    MinSup: {support:<4} ... ")

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
                            if log_file_path:
                                logger.info(
                                    f"OK | Czas: {exec_time:.4f}s | RAM (Net Peak): {mem_net_peak:.2f}MB"
                                )
                        except Exception as e:
                            if log_file_path:
                                logger.error(f"BŁĄD ({e})")

                            import traceback

                            error_traceback = traceback.format_exc()
                            if log_file_path:
                                logger.error(error_traceback)

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

    def plot_results(self, metric="time", figures_path: str | Path = None):
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
            if figures_path:
                plt.savefig(
                    fname=Path(figures_path) / str(data_name.lower() + "_" + metric)
                )
            plt.show()
