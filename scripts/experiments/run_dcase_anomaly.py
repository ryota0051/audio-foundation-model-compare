"""実験実行 CLI: uv run python scripts/experiments/run_dcase_anomaly.py --config configs/experiment/dcase_anomaly.yaml"""

import argparse

from egaudio.experiments.dcase_anomaly import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment/dcase_anomaly.yaml")
    args = parser.parse_args()
    print(run(args.config).to_string())
