"""実験実行 CLI: uv run python scripts/experiments/run_esc50_retrieval.py --config configs/experiment/esc50_retrieval.yaml"""

import argparse

from egaudio.experiments.esc50_retrieval import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment/esc50_retrieval.yaml")
    args = parser.parse_args()
    print(run(args.config).to_string())
