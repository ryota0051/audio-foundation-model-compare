"""音声埋め込みどうしの検索による分類評価。"""

from collections.abc import Callable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from egaudio.zeroshot.classify import cosine_similarity


def _topk(db: np.ndarray, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    sim = cosine_similarity(query, db)
    k = min(k, db.shape[0])
    idx = np.argsort(-sim, axis=1)[:, :k]
    return idx, np.take_along_axis(sim, idx, axis=1)


def knn_predict(db: np.ndarray, db_y: np.ndarray, query: np.ndarray, k: int) -> np.ndarray:
    """上位 k 件の多数決。同票は類似度の合計が大きいクラスを採用。"""
    idx, sims = _topk(db, query, k)
    labels = db_y[idx]
    preds = np.empty(len(query), dtype=db_y.dtype)
    for i in range(len(query)):
        classes, counts = np.unique(labels[i], return_counts=True)
        sim_sum = np.array([sims[i][labels[i] == c].sum() for c in classes])
        preds[i] = classes[np.lexsort((sim_sum, counts))[-1]]
    return preds


def precision_at_k(db: np.ndarray, db_y: np.ndarray, query: np.ndarray, query_y: np.ndarray, k: int) -> float:
    """上位 k 件のうちクエリと同じクラスの割合（クエリ平均）。"""
    idx, _ = _topk(db, query, k)
    return float((db_y[idx] == query_y[:, None]).mean())


def linear_probe_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, c: float, max_iter: int) -> np.ndarray:
    """標準化 + ロジスティック回帰（標準化の統計量も学習データのみから推定）。"""
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=max_iter))
    clf.fit(train_x, train_y)
    return clf.predict(test_x)


def linear_probe_accuracy(
    train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray, c: float, max_iter: int
) -> float:
    return float((linear_probe_predict(train_x, train_y, test_x, c, max_iter) == test_y).mean())


def out_of_fold_predict(
    predict: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    emb: np.ndarray, y: np.ndarray, folds: np.ndarray,
) -> np.ndarray:
    """各 fold を、それ以外の fold を検索対象・学習データにした predict(db, db_y, query) で予測して結合する。"""
    pred = np.empty_like(y)
    for f in np.unique(folds):
        te = folds == f
        pred[te] = predict(emb[~te], y[~te], emb[te])
    return pred
