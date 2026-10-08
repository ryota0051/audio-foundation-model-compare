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


def load_dcase_whole(cache_dir: str | Path, machine: str, dim: int) -> tuple[np.ndarray, np.ndarray]:
    """DCASE のクリップ全体埋め込み (train, test) を dim 次元・L2 正規化で返す。"""
    d = Path(cache_dir) / machine
    return (
        truncate_and_normalize(np.load(d / "train_whole.npy"), dim),
        truncate_and_normalize(np.load(d / "test_whole.npy"), dim),
    )


def dcase_domain_labels(clips: pd.DataFrame) -> np.ndarray:
    """'source normal' / 'target normal' / 'anomaly' の 3 区分。"""
    return np.where(clips["is_anomaly"].to_numpy() == 1, "anomaly", clips["domain"].to_numpy() + " normal")
