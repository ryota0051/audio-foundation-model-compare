import numpy as np

from egaudio.zeroshot.classify import confusion_pairs, cosine_similarity, topk_accuracy


def test_cosine_similarity_normalizes():
    a = np.array([[2.0, 0.0]])
    b = np.array([[1.0, 0.0], [0.0, 3.0]])
    np.testing.assert_allclose(cosine_similarity(a, b), [[1.0, 0.0]])


def test_topk_accuracy():
    sim = np.array([[0.9, 0.5, 0.1], [0.2, 0.3, 0.8], [0.6, 0.5, 0.4]])
    y = np.array([0, 1, 2])
    assert topk_accuracy(sim, y, 1) == 1 / 3
    assert topk_accuracy(sim, y, 2) == 2 / 3
    assert topk_accuracy(sim, y, 3) == 1.0


def test_confusion_pairs_counts_offdiag():
    y = np.array([0, 0, 0, 1, 1, 2])
    pred = np.array([1, 1, 0, 0, 1, 2])
    df = confusion_pairs(y, pred, ["a", "b", "c"], top_n=5)
    assert df.iloc[0].to_dict() == {"true": "a", "pred": "b", "count": 2}
    assert len(df) == 2
