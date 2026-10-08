"""正常音メモリバンクに対する距離ベースの異常スコア。"""

import numpy as np
from sklearn.covariance import EmpiricalCovariance, LedoitWolf


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


class KnnCosineScorer:
    """k 近傍のコサイン距離 (1 - cos) の平均。"""

    def __init__(self, k: int):
        self.k = k

    def fit(self, bank: np.ndarray) -> "KnnCosineScorer":
        self.bank = _normalize(np.asarray(bank, dtype=np.float32))
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        dist = 1.0 - _normalize(np.asarray(x, dtype=np.float32)) @ self.bank.T
        k = min(self.k, self.bank.shape[0])
        return np.sort(dist, axis=1)[:, :k].mean(axis=1)

    def score_loo(self) -> np.ndarray:
        """バンク自身のスコアを、自分自身を除いた近傍で計算する（閾値決め用）。"""
        dist = 1.0 - self.bank @ self.bank.T
        np.fill_diagonal(dist, np.inf)
        k = min(self.k, self.bank.shape[0] - 1)
        return np.sort(dist, axis=1)[:, :k].mean(axis=1)


class MahalanobisScorer:
    """マハラノビス距離。"""

    _ESTIMATORS = {"ledoit_wolf": LedoitWolf, "empirical": EmpiricalCovariance}

    def __init__(self, covariance: str = "ledoit_wolf"):
        if covariance not in self._ESTIMATORS:
            raise ValueError(f"unknown covariance estimator: {covariance}")
        self.covariance = covariance

    def fit(self, bank: np.ndarray) -> "MahalanobisScorer":
        est = self._ESTIMATORS[self.covariance]().fit(np.asarray(bank, dtype=np.float64))
        self.mean = est.location_
        self.precision = est.precision_
        return self

    @classmethod
    def with_shared_precision(cls, base: "MahalanobisScorer", bank: np.ndarray) -> "MahalanobisScorer":
        """共分散は base（ソース）を流用し、平均だけ bank（ターゲット少数）から求める。"""
        sc = cls(base.covariance)
        sc.mean = np.asarray(bank, dtype=np.float64).mean(axis=0)
        sc.precision = base.precision
        return sc

    def score(self, x: np.ndarray) -> np.ndarray:
        d = np.asarray(x, dtype=np.float64) - self.mean
        return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", d, self.precision, d), 0.0))
