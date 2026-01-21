import argparse
import datetime
import os
from pathlib import Path

from alg_advanced_eclat import AdvancedEclatMiner
from alg_eclat import EclatMiner
from alg_postdiffset import PostdiffsetMiner
from benchmark_runner import (
    BenchmarkRunner,
    FIM_AVAILABLE,
    DirectFimApriori,
    DirectFimEclat,
)
import yaml


def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Runner for Frequent Itemset Mining Algorithms"
    )

    # Results file paths
    parser.add_argument(
        "--results_file",
        type=str,
        default="results/results.csv",
        help="Path to save the detailed results in CSV format",
    )
    parser.add_argument(
        "--results_avg_file",
        type=str,
        default="results/average_results.csv",
        help="Path to save the averaged results in CSV format",
    )

    # Number of iterations
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of times to run each algorithm on the dataset (default: 1)",
    )

    # Input configuration
    parser.add_argument(
        "--input_config",
        type=str,
        default="configs/datasets_config.yaml",
        help="Path to the input configuration YAML file",
    )

    # Log file path
    parser.add_argument(
        "--log_file",
        type=str,
        default=f"logs/benchmark-{str(datetime.datetime.now()).replace(' ', '_').replace(':', '.')}.log",
        help="Path to save the output prints in log format",
    )

    args = parser.parse_args()

    # Utworzenie folderów do pliku log
    os.makedirs(Path(args.log_file).parent, exist_ok=True)

    # Załadowanie konfiguracji datasetów
    config = load_config(args.input_config)

    # Algorytmy do testowania
    algos_to_test = {
        "My Eclat": EclatMiner,
        "My Postdiffset": PostdiffsetMiner,
        "My Adv. Eclat": AdvancedEclatMiner,
    }

    if FIM_AVAILABLE:
        algos_to_test["FIM Apriori"] = DirectFimApriori
        algos_to_test["FIM Eclat"] = DirectFimEclat

    # Uruchomienie benchmark runnera
    runner = BenchmarkRunner(algos_to_test)
    runner.run_comparison(
        config, args.results_file, args.iterations, log_file_path=args.log_file
    )

    print("\nRysowanie wykresów...")
    runner.plot_results(metric="time")
    runner.plot_results(metric="memory")


if __name__ == "__main__":
    main()
