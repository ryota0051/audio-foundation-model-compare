"""音声⇔テキスト類似度を試すブラウザアプリを起動する。

docker compose run --rm --service-ports app uv run python scripts/app/run_playground.py
→ ホストのブラウザで http://localhost:7860 を開く
"""

import argparse

from egaudio.config.loader import load_yaml
from egaudio.embedding.model import AudioTextEmbedder
from egaudio.playground.app import build_ui


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/app/playground.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    embedder = AudioTextEmbedder(load_yaml(cfg["model_config"]))
    demo = build_ui(embedder, cfg)
    demo.launch(server_name=cfg["server"]["host"], server_port=cfg["server"]["port"])


if __name__ == "__main__":
    main()
