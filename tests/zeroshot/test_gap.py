import numpy as np
import pytest

from egaudio.zeroshot.gap import (
    center,
    column_zscore,
    corrected_similarity,
    csls,
    prediction_concentration,
)


def _unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_center_removes_mean_and_normalizes():
    x = _unit(np.array([[1.0, 1.0], [1.0, -1.0], [2.0, 0.5]]))
    mu = x.mean(axis=0)
    c = center(x, mu)
    np.testing.assert_allclose(np.linalg.norm(c, axis=1), 1.0, rtol=1e-6)
    ref = _unit(x - mu)
    np.testing.assert_allclose(c, ref, rtol=1e-6)


def test_column_zscore():
    sim = np.array([[1.0, 10.0], [3.0, 30.0], [5.0, 50.0]])
    z = column_zscore(sim)
    np.testing.assert_allclose(z.mean(axis=0), 0.0, atol=1e-9)
    np.testing.assert_allclose(z.std(axis=0), 1.0, rtol=1e-6)
    # 列ごとのスケール差（ハブ）が消える
    np.testing.assert_allclose(z[:, 0], z[:, 1])


def test_csls_penalizes_hub():
    # テキスト 0 は全音声に近いハブ。CSLS ではテキスト 0 の優位が消える
    sim = np.array([[0.9, 0.85], [0.9, 0.2], [0.9, 0.2]])
    assert sim[0].argmax() == 0
    c = csls(sim, k=2)
    assert c[0].argmax() == 1


def test_corrected_similarity_none_is_cosine():
    rng = np.random.default_rng(0)
    a, t = _unit(rng.standard_normal((6, 4))), _unit(rng.standard_normal((3, 4)))
    folds = np.array([1, 1, 2, 2, 3, 3])
    np.testing.assert_allclose(corrected_similarity("none", a, t, folds, {}), a @ t.T, rtol=1e-6)


def test_center_both_oof_uses_other_folds_only():
    a = _unit(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 1.0]]))
    t = _unit(np.array([[1.0, 0.2], [0.2, 1.0]]))
    folds = np.array([1, 1, 2, 2])
    sim = corrected_similarity("center_both_oof", a, t, folds, {})
    mu_t = t.mean(axis=0)
    tc = _unit(t - mu_t)
    # fold 1 の行は fold 2 の音声平均だけで中心化される
    exp_f1 = _unit(a[:2] - a[2:].mean(axis=0)) @ tc.T
    np.testing.assert_allclose(sim[:2], exp_f1, rtol=1e-6)


def test_corrected_similarity_unknown():
    with pytest.raises(ValueError):
        corrected_similarity("foo", np.eye(2), np.eye(2), np.array([1, 2]), {})


def test_prediction_concentration():
    pred = np.array([0, 0, 0, 1])
    r = prediction_concentration(pred, n_classes=3)
    assert r == {"max_pred_count": 3, "n_never_predicted": 1}
