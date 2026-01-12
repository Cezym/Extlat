import time
import gc
import tracemalloc
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Type

# --- IMPORTY KLAS BAZOWYCH ---
from data_manager import TransactionLoader
from base_miner import BaseMiner

# --- IMPORTY TWOICH ALGORYTMÓW (Bezpośrednio) ---
# Zakładamy, że te pliki istnieją i działają
try:
    from alg_eclat import EclatMiner
    from alg_postdiffset import PostdiffsetMiner
    from alg_advanced_eclat import AdvancedEclatMiner
except ImportError as e:
    print(f"UWAGA: Nie znaleziono pliku z algorytmem: {e}")

# --- IMPORT BIBLIOTEKI FIM (Bezpośrednio) ---
try:
    import fim

    FIM_AVAILABLE = True
except ImportError:
    print("UWAGA: Biblioteka 'fim' nie jest zainstalowana (pip install fim).")
    FIM_AVAILABLE = False


# --- ADAPTERY DLA BIBLIOTEKI FIM ---
# Niezbędne, aby ujednolicić interfejs (Klasa vs Funkcja) wewnątrz Benchmarkera
class DirectFimApriori(BaseMiner):
    def find_frequent_itemsets(self):
        # fim używa ujemnych wartości dla count (np. -10 oznacza min 10 wystąpień)
        results = fim.apriori(self.dataset, supp=-self.min_support_count, report='s')
        return len(results)


class DirectFimEclat(BaseMiner):
    def find_frequent_itemsets(self):
        results = fim.eclat(self.dataset, supp=-self.min_support_count, report='s')
        return len(results)


class BenchmarkRunner:
    def __init__(self, algorithms: Dict[str, Type[BaseMiner]]):
        """
        Args:
            algorithms: Słownik { "Nazwa na wykresie": KlasaAlgorytmu }
        """
        self.algorithms = algorithms
        self.results = {}

    def measure_execution(self, algorithm_class: Type[BaseMiner], dataset: list[set[int]], min_support: float):
        """
        Mierzy czas (s) i szczytowe zużycie pamięci (MB).
        """
        # 1. Sprzątanie pamięci przed testem (kluczowe dla dokładności)
        gc.collect()

        # 2. Start śledzenia pamięci
        tracemalloc.start()

        # 3. Inicjalizacja algorytmu
        # Tworzymy instancję klasy (np. PostdiffsetMiner lub DirectFimEclat)
        miner = algorithm_class(min_support, dataset)

        # 4. Start pomiaru czasu
        start_time = time.time()

        # Wykonanie (szukanie zbiorów)
        miner.find_frequent_itemsets()

        # 5. Stop pomiaru czasu
        end_time = time.time()

        # 6. Odczyt zużycia pamięci
        _, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        execution_time = end_time - start_time
        peak_memory_mb = peak_memory / (1024 * 1024)  # Konwersja Bajty -> MB

        return execution_time, peak_memory_mb

    def run_comparison(self, datasets: Dict[str, str], support_range: List[float]):
        """
        Główna pętla testująca: Datasets -> Algorithms -> Supports
        """
        loader = TransactionLoader()
        self.results = {}

        for data_name, data_path in datasets.items():
            print(f"\n==========================================")
            print(f" PRZETWARZANIE ZBIORU: {data_name}")
            print(f" Plik: {data_path}")
            print(f"==========================================")

            if not os.path.exists(data_path):
                print(f"BŁĄD: Plik nie istnieje!")
                continue

            # Wczytujemy dane RAZ (singleton loadera)
            current_dataset = loader.load(data_path)
            print(f"-> Załadowano {len(current_dataset)} transakcji.")

            self.results[data_name] = {}

            for algo_name, algo_class in self.algorithms.items():
                print(f"\n>>> Algorytm: {algo_name}")
                self.results[data_name][algo_name] = {}

                for support in support_range:
                    # Wyświetlanie postępu
                    print(f"    MinSup: {support:<4} ... ", end="", flush=True)

                    try:
                        exec_time, mem_peak = self.measure_execution(algo_class, current_dataset, support)

                        # Zapis wyników
                        self.results[data_name][algo_name][support] = {
                            "time": exec_time,
                            "memory": mem_peak
                        }
                        print(f"OK | Czas: {exec_time:.4f}s | RAM: {mem_peak:.2f}MB")
                    except Exception as e:
                        print(f"BŁĄD ({e})")
                        self.results[data_name][algo_name][support] = None

    def plot_results(self, metric="time"):
        """
        Rysuje wykresy.
        metric: 'time' lub 'memory'
        """
        if not self.results:
            print("Brak wyników do wyświetlenia.")
            return

        # Konfiguracja etykiet
        if metric == "time":
            y_label = "Czas wykonania (sekundy)"
            title_prefix = "Wydajność Czasowa"
        else:
            y_label = "Zużycie RAM (MB)"
            title_prefix = "Zużycie Pamięci"

        for data_name, algo_results in self.results.items():
            plt.figure(figsize=(12, 7))

            # Iteracja po algorytmach
            for algo_name, data_points in algo_results.items():
                # Sortowanie punktów po supporcie (oś X)
                # Sortujemy malejąco, żeby linie się ładnie łączyły
                sorted_points = sorted(data_points.items(), key=lambda x: x[0], reverse=True)

                supports = []
                values = []

                for supp, res in sorted_points:
                    if res is not None:
                        supports.append(supp)
                        values.append(res[metric])

                if supports:
                    # Stylizacja: FIM linią przerywaną, nasze ciągłą
                    is_library = "FIM" in algo_name or "Lib" in algo_name
                    line_style = '--' if is_library else '-'
                    marker = 'x' if is_library else 'o'

                    plt.plot(supports, values, marker=marker, linestyle=line_style, label=algo_name, linewidth=2)

            plt.title(f"{title_prefix}: {data_name}")
            plt.xlabel("Minimum Support (im mniej, tym trudniej)")
            plt.ylabel(y_label)
            plt.legend()
            plt.grid(True, which="both", linestyle='--', alpha=0.7)

            # Odwracamy oś X (startujemy od łatwego dużego supportu do trudnego małego)
            plt.gca().invert_xaxis()

            plt.tight_layout()
            plt.show()


# --- KONFIGURACJA I URUCHOMIENIE ---
if __name__ == "__main__":
    # 1. Definicja algorytmów do porównania
    algos_to_test = {
        # Twoje implementacje
        "My Eclat": EclatMiner,
        "My Postdiffset": PostdiffsetMiner,
        "My Adv. Eclat": AdvancedEclatMiner,
    }

    # Dodajemy biblioteczne FIM (jeśli dostępne)
    if FIM_AVAILABLE:
        algos_to_test["FIM Apriori"] = DirectFimApriori
        algos_to_test["FIM Eclat"] = DirectFimEclat

    # 2. Zbiory danych
    datasets_map = {
        #"Chess": "data/chess.txt",
        "Retail": "data/retail.txt", # Odkomentuj jeśli masz ten plik
        # "Mushroom": "data/mushroom.txt"
    }

    # 3. Zakresy supportu (od łatwego do trudnego)
    # Dla chess.txt: 0.9 (bardzo łatwe) -> 0.6 (trudniejsze)
    supports = [0.1, 0.2, 0.3]

    # 4. Uruchomienie benchmarka
    runner = BenchmarkRunner(algos_to_test)
    runner.run_comparison(datasets_map, supports)

    # 5. Generowanie wykresów
    print("\nRysowanie wykresu czasu...")
    runner.plot_results(metric="time")

    print("Rysowanie wykresu pamięci...")
    runner.plot_results(metric="memory")