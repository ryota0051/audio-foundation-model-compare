"""実験実行 CLI: uv run python scripts/experiments/run_esc50_zeroshot.py --config configs/experiment/esc50_zeroshot.yaml"""

import argparse

from egaudio.experiments.esc50_zeroshot import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment/esc50_zeroshot.yaml")
    args = parser.parse_args()
    print(run(args.config).to_string())
