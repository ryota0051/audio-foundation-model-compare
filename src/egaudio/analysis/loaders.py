"""分析スクリプト共通: キャッシュ済み埋め込みの読み込み（GPU 不要）。"""

from pathlib import Path

import numpy as np
import pandas as pd

from egaudio.config.loader import load_yaml
from egaudio.embedding.mrl import truncate_and_normalize


def load_esc50_retrieval_embeddings(retrieval_cfg: dict, label: str, n_clips: int) -> np.ndarray:
    """Step1b（esc50_retrieval）でキャッシュした音声埋め込みを L2 正規化して返す。"""
    model_cfg_path = Path(retrieval_cfg["models"][label])
    path = Path(retrieval_cfg["cache_dir"]) / model_cfg_path.stem / f"audio_n{n_clips}.npy"
    if not path.is_file():
        raise FileNotFoundError(f"{path} (run scripts/experiments/run_esc50_retrieval.py first)")
    return truncate_and_normalize(np.load(path), load_yaml(model_cfg_path)["full_dim"])


def load_zeroshot_embeddings(zeroshot_cfg: dict, filenames: list[str], classes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Step1 ゼロショットの最良条件（best.json）の (音声, クラステキスト) 埋め込みを全次元・L2 正規化で返す。"""
    import json

    from egaudio.experiments.esc50_zeroshot import _key
    from egaudio.zeroshot.prompts import build_texts

    best = json.loads((Path(zeroshot_cfg["output_dir"]) / "best.json").read_text(encoding="utf-8"))
    texts = build_texts(classes, zeroshot_cfg["templates"][best["template"]],
                        zeroshot_cfg["prefixes"][best["prefix"]], zeroshot_cfg["natural_labels"])
    cache = Path(zeroshot_cfg["cache_dir"])
    dim = load_yaml(zeroshot_cfg["model_config"])["full_dim"]
    audio = truncate_and_normalize(np.load(cache / f"audio_{_key(filenames)}.npy"), dim)
    text = truncate_and_normalize(np.load(cache / f"text_{_key(texts)}.npy"), dim)
    return audio, text


def load_dcase_whole(cache_dir: str | Path, machine: str, dim: int | None) -> tuple[np.ndarray, np.ndarray]:
    """DCASE のクリップ全体埋め込み (train, test) を dim 次元（None = 全次元）・L2 正規化で返す。"""
    d = Path(cache_dir) / machine
    tr, te = np.load(d / "train_whole.npy"), np.load(d / "test_whole.npy")
    dim = dim or tr.shape[1]
    return truncate_and_normalize(tr, dim), truncate_and_normalize(te, dim)


def dcase_domain_labels(clips: pd.DataFrame) -> np.ndarray:
    """'source normal' / 'target normal' / 'anomaly' の 3 区分。"""
    return np.where(clips["is_anomaly"].to_numpy() == 1, "anomaly", clips["domain"].to_numpy() + " normal")
