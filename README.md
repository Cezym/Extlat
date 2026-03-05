# Extlat - Frequent Itemset Mining

A project developed as part of the Data Mining course. It is designed to analyze and extract frequent itemsets from transactional databases. The software provides custom implementations of selected algorithms and a benchmarking module to evaluate execution time and peak RAM usage.

## Implemented Algorithms

The project provides the following algorithms:

- **Classic Eclat**: A custom implementation utilizing depth-first search (DFS) and intersection operations on a vertical data representation (TID-sets).
- **Postdiffset**: A hybrid approach that optimizes memory usage by switching data representations during recursion (transitioning from full TID-sets to Diffsets).
- **Advanced Eclat**: A highly optimized implementation based on bit vectors (using the `bitarray` library), replacing standard ID sets for faster intersections.
- **PyFIM Apriori & Eclat**: Adapters that allow running and comparing reference implementations from the external `pyfim` library.

## Project Structure

The key source files are located in the `src/` directory:

- `data_manager.py` - Responsible for loading transactions from files and transforming them into a vertical format.
- `base_miner.py` - Abstract base class inherited by all algorithms.
- `alg_eclat.py` - Implementation of the classic Eclat algorithm.
- `alg_postdiffset.py` - Implementation of the Postdiffset algorithm.
- `alg_advanced_eclat.py` - Implementation of the Advanced Eclat algorithm.
- `benchmark_runner.py` - Module responsible for running experiments, measuring time, and aggregating peak RAM usage via the `psutil` library.

## Requirements and Installation

To run the project, the environment must meet the following criteria:

- Python 3.14
- `uv` tool (package and environment manager)
- Python dependencies: `bitarray`, `psutil`, `pyfim` (along with `pandas` and `matplotlib` for data handling and plotting).

To install all required dependencies using the `uv` environment, run the following command:

```bash
uv sync

```

## Running the Application

The main entry point for running benchmarks and testing the algorithms is the `main.py` script.

Example of a basic run:

```bash
uv run python main.py

```

### Available Command-Line Arguments

* 
`--results_file`: Path to the CSV file with detailed results (default: `results/results.csv`).


* 
`--results_avg_file`: Path to the CSV file with averaged results across all iterations (default: `results/average_results.csv`).


* 
`--figures_path`: Directory to save the generated plots (default: `results/figures/`).


* 
`--iterations`: Number of test repetitions for each dataset and algorithm (default: `1`).


* 
`--input_config`: Path to the datasets configuration file in YAML format (default: `configs/datasets_config.yaml`).


* `--log_file`: Full path to the log file. If omitted, a file is automatically created in the `logs/` directory with the current date and time.



Example with extended configuration:

```bash
uv run python main.py --iterations 3 --results_file results/detailed.csv --results_avg_file results/avg.csv --figures_path figures/ --log_file logs/run.log

```

## Configuration (YAML)

The dataset configuration file (`datasets_config.yaml`) describes the datasets to be processed and their parameters.

Example configuration file content:

```yaml
- name: 'Retail'
  min_supports: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
  dataset_path: 'data/retail.txt'

- name: 'Chess'
  min_supports: [0.7, 0.8, 0.9]
  dataset_path: 'data/chess.txt'

```

## Input Data Format

Input data files (`.txt`) must meet the following conditions:
Each row represents a single transaction, and individual items are separated by a space.

Example input file:

```text
1 2 3
4 5 6
2 5
7 8 9

```

## Output Data

After the benchmark finishes, three main groups of results are generated:

1. 
**Detailed CSV File**: Contains information about individual runs (iteration number, dataset name, algorithm name, support value, execution time in seconds, and net peak memory usage in MB).


2. 
**Averaged CSV File**: Aggregated time and memory results for each combination of dataset, algorithm, and support.


3. 
**PNG Plots**: Performance and memory usage plots automatically exported to the specified directory, showing the relationship between performance and minimum support.
