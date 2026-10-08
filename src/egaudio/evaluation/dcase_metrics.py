"""DCASE 2026 Task 2 公式 evaluator と同じ定義の AUC / pAUC。

- AUC (domain): 当該ドメインの正常 + 全ドメインの異常
- pAUC: 全クリップ, sklearn の max_fpr（McClish 補正）
- 集約: epsilon でクリップした調和平均
"""

import sys

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

DOMAINS = {"source": 0, "target": 1}


def evaluate_section(y_true: np.ndarray, y_score: np.ndarray, domain: np.ndarray, max_fpr: float) -> dict:
    y_true, y_score, domain = map(np.asarray, (y_true, y_score, domain))
    out = {}
    for name, idx in DOMAINS.items():
        mask = (domain == idx) | (y_true != 0)
        out[f"auc_{name}"] = float(roc_auc_score(y_true[mask], y_score[mask]))
    out["pauc"] = float(roc_auc_score(y_true, y_score, max_fpr=max_fpr))
    return out


def hmean(values) -> float:
    v = np.maximum(np.asarray(values, dtype=float), sys.float_info.epsilon)
    return float(stats.hmean(v))
