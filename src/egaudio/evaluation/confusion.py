"""混同行列と閾値判定。"""

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """行 = 真値, 列 = 予測 の件数行列。"""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return cm


def row_normalize(cm: np.ndarray) -> np.ndarray:
    """各行を真値クラスの件数で割る（対角 = クラス別再現率）。空行は 0。"""
    total = cm.sum(axis=1, keepdims=True)
    return np.divide(cm, total, out=np.zeros(cm.shape, dtype=float), where=total > 0)


def group_confusion(cm: np.ndarray, groups: np.ndarray, n_groups: int) -> np.ndarray:
    """クラス→グループの対応 groups で、混同行列をグループ単位に集約する。"""
    onehot = np.eye(n_groups, dtype=int)[groups]
    return onehot.T @ cm @ onehot


def threshold_from_normal(normal_scores: np.ndarray, quantile: float) -> float:
    """学習用正常音スコアの分位点を判定閾値にする。"""
    return float(np.quantile(normal_scores, quantile))


def binary_report(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """スコア > 閾値 を異常と判定したときの混同行列と precision / recall / F1。"""
    y_true = np.asarray(y_true).astype(bool)
    pred = np.asarray(scores) > threshold
    tp = int((pred & y_true).sum())
    fp = int((pred & ~y_true).sum())
    fn = int((~pred & y_true).sum())
    tn = int((~pred & ~y_true).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp, "precision": precision, "recall": recall, "f1": f1}
