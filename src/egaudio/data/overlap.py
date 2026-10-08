"""評価データと外部学習データ（Freesound ID）の重複判定。"""

import math
from pathlib import Path

import numpy as np
import pandas as pd


def load_freesound_ids(csv_path: str | Path, column: str = "freesound_id") -> set[str]:
    """メタデータ CSV の Freesound ID 列だけを文字列の集合で読む。"""
    return set(pd.read_csv(csv_path, usecols=[column], dtype=str)[column].dropna())


def overlap_mask(src_ids: pd.Series, ids: set[str]) -> np.ndarray:
    """元音源 ID が ids に含まれるクリップを True にする（型の違いを避けるため文字列で比較）。"""
    return src_ids.astype(str).isin(ids).to_numpy()


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """二項比率の Wilson 信頼区間（既定 95%）。"""
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, center - half), min(1.0, center + half)
