import numpy as np
import pytest

from egaudio.anomaly.scorers import KnnCosineScorer, MahalanobisScorer


def test_knn_cosine_k1_and_k2():
    bank = np.array([[1.0, 0.0], [0.0, 1.0]])
    test = np.array([[1.0, 0.0], [1.0, 1.0]])
    s1 = KnnCosineScorer(k=1).fit(bank).score(test)
    np.testing.assert_allclose(s1, [0.0, 1 - 1 / np.sqrt(2)], atol=1e-6)
    s2 = KnnCosineScorer(k=2).fit(bank).score(test)
    np.testing.assert_allclose(s2, [0.5, 1 - 1 / np.sqrt(2)], atol=1e-6)


def test_knn_k_clipped_to_bank_size():
    bank = np.array([[1.0, 0.0]])
    s = KnnCosineScorer(k=10).fit(bank).score(np.array([[0.0, 1.0]]))
    np.testing.assert_allclose(s, [1.0])


def test_mahalanobis_identity_cov_is_euclidean():
    rng = np.random.default_rng(0)
    bank = rng.standard_normal((5000, 3))
    sc = MahalanobisScorer(covariance="empirical").fit(bank)
    s = sc.score(np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]))
    assert s[0] < 0.2 and abs(s[1] - 3.0) < 0.2


def test_mahalanobis_shared_precision_target_mean():
    rng = np.random.default_rng(0)
    src = rng.standard_normal((2000, 2))
    base = MahalanobisScorer(covariance="ledoit_wolf").fit(src)
    tgt = np.full((10, 2), 5.0)
    sc = MahalanobisScorer.with_shared_precision(base, tgt)
    s = sc.score(np.array([[5.0, 5.0]]))
    assert s[0] < 1e-6


def test_mahalanobis_unknown_cov_raises():
    with pytest.raises(ValueError):
        MahalanobisScorer(covariance="foo")
