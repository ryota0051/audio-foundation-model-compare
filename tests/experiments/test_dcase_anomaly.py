import numpy as np
import pandas as pd
import pytest

from egaudio.experiments.dcase_anomaly import (
    aggregate_segments,
    baseline_table,
    build_scorers,
    score_clips,
    summarize,
)


def _bank():
    src = np.array([[1.0, 0.0], [0.99, 0.1]])
    tgt = np.array([[0.0, 1.0]])
    return np.vstack([src, tgt]), np.array(["source", "source", "target"])


def test_score_clips_single_and_domain_min_knn():
    emb, dom = _bank()
    fit_src, fit_tgt = build_scorers({"knn_k": [1], "mahalanobis": False, "covariance": "ledoit_wolf"})["knn1"]
    test = np.array([[0.0, 1.0]])
    np.testing.assert_allclose(score_clips(emb, dom, test, fit_src, fit_tgt, "single"), [0.0], atol=1e-6)
    np.testing.assert_allclose(score_clips(emb, dom, test, fit_src, fit_tgt, "domain_min"), [0.0], atol=1e-6)


def test_score_clips_unknown_pooling():
    emb, dom = _bank()
    fit_src, fit_tgt = build_scorers({"knn_k": [1], "mahalanobis": False, "covariance": "ledoit_wolf"})["knn1"]
    with pytest.raises(ValueError):
        score_clips(emb, dom, emb, fit_src, fit_tgt, "foo")


def test_build_scorers_names():
    names = build_scorers({"knn_k": [1, 5], "mahalanobis": True, "covariance": "ledoit_wolf"})
    assert list(names) == ["knn1", "knn5", "mahalanobis"]


def test_aggregate_segments():
    s = np.array([1.0, 3.0, 2.0, 4.0, 6.0])
    idx = np.array([0, 0, 0, 1, 1])
    np.testing.assert_allclose(aggregate_segments(s, idx, 2, "max"), [3.0, 6.0])
    np.testing.assert_allclose(aggregate_segments(s, idx, 2, "mean"), [2.0, 5.0])


def test_summarize_official_score_is_hmean_over_all_values():
    df = pd.DataFrame(
        [
            {"machine": "a", "mode": "whole", "dim": 768, "pooling": "single", "scorer": "knn1",
             "auc_source": 0.5, "auc_target": 0.5, "pauc": 0.5},
            {"machine": "b", "mode": "whole", "dim": 768, "pooling": "single", "scorer": "knn1",
             "auc_source": 1.0, "auc_target": 1.0, "pauc": 1.0},
        ]
    )
    s = summarize(df)
    assert s.loc[0, "official_score"] == pytest.approx(2 / 3)


def test_baseline_table_converts_percent():
    cfg = {"MAHALA": {"fan": {"auc_source": 60.0, "auc_target": 50.0, "pauc": 50.0}}}
    t = baseline_table(cfg, ["fan"])
    assert t.iloc[0]["auc_source"] == pytest.approx(0.6)
    assert t.iloc[1]["machine"] == "official_score"
