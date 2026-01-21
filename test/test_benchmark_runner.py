import pandas as pd
import pytest

from src.base_miner import BaseMiner

# Import the BenchmarkRunner - skip if dependencies are missing.
try:
    from src.benchmark_runner import BenchmarkRunner
except Exception:
    BenchmarkRunner = None


@pytest.fixture
def dummy_algo():
    """Minimal miner used only for measuring execution time / memory."""

    class DummyMiner(BaseMiner):
        def find_frequent_itemsets(self):
            import time

            time.sleep(0.05)  # simulate work
            _ = [0] * 10000  # allocate a bit of RAM
            return {}

    return DummyMiner


@pytest.mark.skipif(
    BenchmarkRunner is None, reason="BenchmarkRunner cannot be imported"
)
def test_measure_execution(dummy_algo):
    runner = BenchmarkRunner(algorithms={"Dummy": dummy_algo})
    dataset = [{1, 2}, {3}]
    exec_time, mem_mb = runner.measure_execution(dummy_algo, dataset, min_support=0.5)

    assert exec_time >= 0
    assert mem_mb >= 0


@pytest.mark.skipif(
    BenchmarkRunner is None, reason="BenchmarkRunner cannot be imported"
)
def test_run_comparison(tmp_path):
    """Run a full benchmark with one tiny dataset and confirm the .csv output."""
    # Create a tiny transaction file
    data_file = tmp_path / "data.txt"
    data_file.write_text("1 2\n3")

    # YAML configuration for the runner
    config_content = [
        {"name": "tiny", "dataset_path": str(data_file), "min_supports": [0.5]}
    ]
    import yaml

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config_content))

    # Only use the native Eclat implementation for speed.
    from src.alg_eclat import EclatMiner

    runner = BenchmarkRunner(algorithms={"Eclat": EclatMiner})
    output_csv = tmp_path / "results.csv"

    runner.run_comparison(
        datasets_config=yaml.safe_load(config_path.read_text()),
        output_file_path=str(output_csv),
        iteration_num=1,
        log_file_path=None,
    )

    # CSV should exist and contain exactly one row.
    assert output_csv.exists()
    df = pd.read_csv(output_csv)
    assert len(df) == 1

    row = df.iloc[0]
    assert row["iteration"] == 1
    assert row["dataset"] == "tiny"
    assert row["algorithm"] == "Eclat"
    assert row["support"] == 0.5

    # The aggregated DataFrame should also contain the same values.
    avg_df = runner.results_avg
    assert len(avg_df) == 1
    avg_row = avg_df.iloc[0]
    assert abs(avg_row["time"] - row["time"]) < 1e-6
    assert abs(avg_row["memory"] - row["memory"]) < 1e-6
