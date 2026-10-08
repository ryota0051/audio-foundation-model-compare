import numpy as np

from egaudio.experiments.esc50_retrieval import evaluate_folds


def test_evaluate_folds_perfect_clusters():
    rng = np.random.default_rng(0)
    centers = np.eye(3, 8)
    y = np.repeat(np.arange(3), 10)
    emb = centers[y] + rng.normal(0, 0.01, (30, 8))
    folds = np.tile(np.arange(1, 6), 6)
    cfg = {"knn_k": [1, 3], "precision_k": [5], "linear_probe": {"enabled": True, "c": 1.0, "max_iter": 200}}
    rows = evaluate_folds(emb, y, folds, cfg)
    assert {r["fold"] for r in rows} == {1, 2, 3, 4, 5}
    metrics = {r["metric"] for r in rows}
    assert metrics == {"knn_acc@1", "knn_acc@3", "precision@5", "linear_probe_acc"}
    assert all(r["value"] == 1.0 for r in rows)


def test_evaluate_folds_query_never_in_db():
    # 各クリップが自分自身しか近傍を持たない状況でも、同 fold を検索対象に含めなければ自己一致しない
    emb = np.eye(10)
    y = np.arange(10)
    folds = np.repeat([1, 2], 5)
    cfg = {"knn_k": [1], "precision_k": [], "linear_probe": {"enabled": False}}
    rows = evaluate_folds(emb, y, folds, cfg)
    assert all(r["value"] == 0.0 for r in rows)
