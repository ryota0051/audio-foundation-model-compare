"""Matryoshka (MRL) 次元切り詰め。"""

import numpy as np


def truncate_and_normalize(emb: np.ndarray, dim: int) -> np.ndarray:
    """先頭 dim 次元に切り詰めて L2 再正規化する。"""
    emb = np.asarray(emb, dtype=np.float32)
    if dim > emb.shape[-1]:
        raise ValueError(f"dim {dim} > embedding size {emb.shape[-1]}")
    y = emb[..., :dim]
    norm = np.linalg.norm(y, axis=-1, keepdims=True)
    return y / np.maximum(norm, 1e-12)
