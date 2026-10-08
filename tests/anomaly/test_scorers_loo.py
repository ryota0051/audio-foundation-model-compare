import numpy as np

from egaudio.anomaly.scorers import KnnCosineScorer


def test_score_loo_excludes_self():
    bank = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    s = KnnCosineScorer(k=1).fit(bank).score_loo()
    # 0, 1 番は互いに一致（距離 0）、2 番は他と直交（距離 1）
    np.testing.assert_allclose(s, [0.0, 0.0, 1.0], atol=1e-6)


def test_score_loo_k_clipped():
    bank = np.array([[1.0, 0.0], [0.0, 1.0]])
    s = KnnCosineScorer(k=5).fit(bank).score_loo()
    np.testing.assert_allclose(s, [1.0, 1.0], atol=1e-6)
