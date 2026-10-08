import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from egaudio.evaluation.dcase_metrics import evaluate_section, hmean


def test_evaluate_section_follows_official_definition():
    # domain: 0=source, 1=target
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    dom = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.9, 0.3, 0.5, 0.8, 0.4, 0.7])
    r = evaluate_section(y, s, dom, max_fpr=0.1)
    src_mask = (dom == 0) | (y == 1)
    tgt_mask = (dom == 1) | (y == 1)
    assert r["auc_source"] == pytest.approx(roc_auc_score(y[src_mask], s[src_mask]))
    assert r["auc_target"] == pytest.approx(roc_auc_score(y[tgt_mask], s[tgt_mask]))
    assert r["pauc"] == pytest.approx(roc_auc_score(y, s, max_fpr=0.1))
    assert r["auc_source"] == 1.0


def test_hmean_clips_to_eps():
    assert hmean([0.5, 0.5]) == pytest.approx(0.5)
    assert hmean([0.0, 1.0]) < 1e-10
