"""コサイン類似度によるゼロショット分類と評価。"""

import numpy as np
import pandas as pd


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _normalize(a) @ _normalize(b).T


def topk_accuracy(sim: np.ndarray, y: np.ndarray, k: int) -> float:
    topk = np.argsort(-sim, axis=1)[:, :k]
    return float((topk == y[:, None]).any(axis=1).mean())


def confusion_pairs(y: np.ndarray, pred: np.ndarray, classes: list[str], top_n: int) -> pd.DataFrame:
    """真値≠予測 のペアを件数の多い順に返す。"""
    mask = y != pred
    names = np.asarray(classes)
    pairs = pd.DataFrame({"true": names[y[mask]], "pred": names[pred[mask]]})
    counts = pairs.value_counts().reset_index(name="count")
    counts = counts.sort_values(["count", "true", "pred"], ascending=[False, True, True])
    return counts.head(top_n).reset_index(drop=True)
