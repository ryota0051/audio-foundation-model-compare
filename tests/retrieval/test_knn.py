import numpy as np
import pytest

from egaudio.retrieval.knn import (
    knn_predict,
    linear_probe_accuracy,
    linear_probe_predict,
    out_of_fold_predict,
    precision_at_k,
)


def _db():
    # クラス 0 は x 軸寄り、クラス 1 は y 軸寄り
    emb = np.array([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.0, 1.0], [0.1, 0.9]])
    y = np.array([0, 0, 0, 1, 1])
    return emb, y


def test_knn_predict_k1():
    emb, y = _db()
    q = np.array([[1.0, 0.05], [0.05, 1.0]])
    np.testing.assert_array_equal(knn_predict(emb, y, q, k=1), [0, 1])


def test_knn_predict_majority_vote():
    emb, y = _db()
    # 最近傍はクラス 1 だが全 5 件ではクラス 0 が多数
    q = np.array([[0.45, 0.55]])
    assert knn_predict(emb, y, q, k=1)[0] == 1
    assert knn_predict(emb, y, q, k=5)[0] == 0


def test_knn_predict_tie_broken_by_similarity_sum():
    emb = np.array([[1.0, 0.0], [0.0, 1.0]])
    y = np.array([0, 1])
    q = np.array([[0.9, 0.1]])
    assert knn_predict(emb, y, q, k=2)[0] == 0


def test_precision_at_k():
    emb, y = _db()
    q = np.array([[1.0, 0.0]])
    assert precision_at_k(emb, y, q, np.array([0]), k=3) == pytest.approx(1.0)
    assert precision_at_k(emb, y, q, np.array([0]), k=5) == pytest.approx(3 / 5)


def test_linear_probe_separable():
    rng = np.random.default_rng(0)
    x0 = rng.normal([2, 0], 0.1, (50, 2))
    x1 = rng.normal([0, 2], 0.1, (50, 2))
    x = np.vstack([x0, x1])
    y = np.array([0] * 50 + [1] * 50)
    assert linear_probe_accuracy(x, y, x, y, c=1.0, max_iter=200) == 1.0


def test_linear_probe_predict_matches_accuracy():
    rng = np.random.default_rng(0)
    x = np.vstack([rng.normal([2, 0], 0.5, (30, 2)), rng.normal([0, 2], 0.5, (30, 2))])
    y = np.array([0] * 30 + [1] * 30)
    pred = linear_probe_predict(x[::2], y[::2], x[1::2], c=1.0, max_iter=200)
    assert pred.shape == (30,)
    assert float((pred == y[1::2]).mean()) == linear_probe_accuracy(x[::2], y[::2], x[1::2], y[1::2], 1.0, 200)


def test_out_of_fold_predict_uses_only_other_folds():
    emb = np.arange(6, dtype=float)[:, None]
    y = np.array([0, 1, 2, 3, 4, 5])
    folds = np.array([1, 1, 2, 2, 3, 3])
    seen = []

    def predict(db, db_y, q):
        seen.append((set(db_y.tolist()), q[:, 0].tolist()))
        return np.full(len(q), db_y.min())

    pred = out_of_fold_predict(predict, emb, y, folds)
    # 各 fold の予測は、その fold を除いた検索対象の最小ラベル
    np.testing.assert_array_equal(pred, [2, 2, 0, 0, 0, 0])
    for db_labels, q in seen:
        assert not db_labels & {int(v) for v in q}
