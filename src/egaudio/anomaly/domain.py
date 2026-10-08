"""ソース／ターゲットのドメイン別プールの統合。"""

import numpy as np


def domain_min_scores(source_scores: np.ndarray, target_scores: np.ndarray) -> np.ndarray:
    """近い方（スコアが小さい方）のドメインのスコアを採用する。"""
    return np.minimum(source_scores, target_scores)
