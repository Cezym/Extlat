import time
import gc
import matplotlib.pyplot as plt
from typing import Dict, List, Type
import os

# --- IMPORTY KLAS POMOCNICZYCH ---
from data_manager import TransactionLoader
from base_miner import BaseMiner

# --- IMPORTY TWOICH ALGORYTMÓW ---
try:
    from alg_eclat import EclatMiner
    from alg_postdiffset import PostdiffsetMiner
    from alg_advanced_eclat import AdvancedEclatMiner
except ImportError as e:
    print(f"Ostrzeżenie: Nie znaleziono pliku algorytmu: {e}")

# --- IMPORTY BIBLIOTEKI FIM (Wrappery) ---
# Zakładam, że masz zainstalowane: pip install fim
try:
    from fim import apriori, eclat

    FIM_AVAILABLE = True
except ImportError:
    print("Ostrzeżenie: Biblioteka 'fim' nie jest zainstalowana. (pip install fim)")
    FIM_AVAILABLE = False


# Definicje wrapperów FIM (wewnątrz pliku, żebyś nie musiał tworzyć osobnego, jeśli nie chcesz)
class LibAprioriMiner(BaseMiner):
    def find_frequent_itemsets(self):
        # supp=-min_support_count (ujemna wartość oznacza liczbę wystąpień, a nie %)
        results = apriori(self.dataset, supp=-self.min_support_count, report='s')
        # fim zwraca listę krotek, my nic nie musimy z tym robić w benchmarku poza zmierzeniem czasu
        return len(results)


class LibEclatMiner(BaseMiner):
    def find_frequent_itemsets(self):
        results = eclat(self.dataset, supp=-self.min_support_count, report='s')
        return len(results)


class BenchmarkRunner:
    def __init__(self, algorithms: Dict[str, Type[BaseMiner]]):
        """
        Args:
            algorithms: Słownik { "Nazwa": KlasaAlgorytmu }
        """
        self.algorithms = algorithms
        self.results = {}

    def measure_execution(self, algorithm_class: Type[BaseMiner], dataset: list[set[int]], min_support: float) -> float:
        """
        Mierzy czas wykonania jednego algorytmu.
        """
        # Czyścimy pamięć przed testem
        gc.collect()

        # Inicjalizacja (czas tworzenia obiektu pomijamy lub wliczamy zależnie od potrzeb,
        # tutaj wliczamy tylko find_frequent_itemsets)
        miner = algorithm_class(min_support, dataset)

        start_time = time.time()
        # Uruchomienie
        miner.find_frequent_itemsets()
        end_time = time.time()

        return end_time - start_time

    def run_comparison(self, datasets: Dict[str, str], support_range: List[float]):
        """
        Główna pętla testowa.
        """
        loader = TransactionLoader()
        self.results = {}

        for data_name, data_path in datasets.items():
            print(f"\n=== ZBIÓR DANYCH: {data_name} ===")

            # Wczytujemy dane RAZ dla wszystkich algorytmów
            if not os.path.exists(data_path):
                print(f"Błąd: Plik {data_path} nie istnieje!")
                continue

            current_dataset = loader.load(data_path)
            print(f"Liczba transakcji: {len(current_dataset)}")

            self.results[data_name] = {}

            for algo_name, algo_class in self.algorithms.items():
                print(f"\n>>> Testowanie: {algo_name}")
                self.results[data_name][algo_name] = {}

                for support in support_range:
                    print(f"    Support: {support} ... ", end="", flush=True)
                    try:
                        exec_time = self.measure_execution(algo_class, current_dataset, support)
                        self.results[data_name][algo_name][support] = exec_time
                        print(f"{exec_time:.4f} s")
                    except Exception as e:
                        print(f"BŁĄD ({e})")
                        self.results[data_name][algo_name][support] = None

    def plot_results(self):
        """
        Rysuje wykresy wyników.
        """
        if not self.results:
            print("Brak wyników do wyświetlenia.")
            return

        for data_name, algo_results in self.results.items():
            plt.figure(figsize=(12, 8))

            for algo_name, data_points in algo_results.items():
                # Sortujemy punkty po supporcie malejąco
                sorted_points = sorted(data_points.items(), key=lambda x: x[0], reverse=True)

                supports = [p[0] for p in sorted_points if p[1] is not None]
                times = [p[1] for p in sorted_points if p[1] is not None]

                if supports:
                    # Styl linii: FIM przerywana, nasze ciągła
                    linestyle = '--' if "Lib" in algo_name or "FIM" in algo_name else '-'
                    marker = 'x' if "Lib" in algo_name or "FIM" in algo_name else 'o'

                    plt.plot(supports, times, marker=marker, linestyle=linestyle, label=algo_name, linewidth=2)

            plt.title(f"Porównanie wydajności: {data_name}")
            plt.xlabel("Minimum Support")
            plt.ylabel("Czas wykonania (s)")
            plt.legend()
            plt.grid(True, which="both", ls="-", alpha=0.5)

            # Odwracamy oś X (od dużego supportu do małego - trudniejszego)
            plt.gca().invert_xaxis()
            plt.tight_layout()
            plt.show()


# --- KONFIGURACJA TESTÓW ---
if __name__ == "__main__":
    # 1. Lista algorytmów do sprawdzenia
    algos_to_test = {
        "My Eclat (Tidset)": EclatMiner,
        "My Postdiffset": PostdiffsetMiner,
        "My Advanced Eclat": AdvancedEclatMiner,
    }

    # Dodajemy biblioteczne tylko jeśli są dostępne
    if FIM_AVAILABLE:
        algos_to_test["FIM Apriori"] = LibAprioriMiner
        algos_to_test["FIM Eclat"] = LibEclatMiner

    # 2. Definicja zbiorów danych
    # Upewnij się, że pliki istnieją w folderze data/
    datasets_map = {
        # "Chess (Dense)": "data/chess.txt",
        "Retail (Sparse)": "data/retail.txt", # Odkomentuj jeśli masz
        # "Mushroom": "data/mushroom.txt"       # Odkomentuj jeśli masz
    }

    # 3. Zakresy supportu do testów
    # UWAGA: Dla FIM w Pythonie supporty mogą być bardzo niskie,
    # ale dla implementacji w czystym Pythonie zacznij od wyższych (np. 0.6 - 0.9 dla Chess)
    supports = [0.8]

    # 4. Uruchomienie
    runner = BenchmarkRunner(algos_to_test)
    runner.run_comparison(datasets_map, supports)
    runner.plot_results()