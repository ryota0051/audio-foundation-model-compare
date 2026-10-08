import numpy as np
import pytest

from egaudio.evaluation.confusion import (
    binary_report,
    confusion_matrix,
    group_confusion,
    row_normalize,
    threshold_from_normal,
)


def test_confusion_matrix_counts():
    y = np.array([0, 0, 1, 2, 2])
    p = np.array([0, 1, 1, 2, 0])
    cm = confusion_matrix(y, p, 3)
    np.testing.assert_array_equal(cm, [[1, 1, 0], [0, 1, 0], [1, 0, 1]])


def test_row_normalize_handles_empty_rows():
    cm = np.array([[2, 2], [0, 0]])
    np.testing.assert_allclose(row_normalize(cm), [[0.5, 0.5], [0.0, 0.0]])


def test_group_confusion_sums_blocks():
    cm = np.arange(16).reshape(4, 4)
    g = group_confusion(cm, np.array([0, 0, 1, 1]), 2)
    np.testing.assert_array_equal(g, [[0 + 1 + 4 + 5, 2 + 3 + 6 + 7], [8 + 9 + 12 + 13, 10 + 11 + 14 + 15]])


def test_threshold_from_normal_quantile():
    assert threshold_from_normal(np.arange(101, dtype=float), 0.9) == pytest.approx(90.0)


def test_binary_report():
    y = np.array([0, 0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.9, 0.8, 0.3])
    r = binary_report(y, s, threshold=0.5)
    assert (r["tn"], r["fp"], r["fn"], r["tp"]) == (2, 1, 1, 1)
    assert r["precision"] == pytest.approx(0.5)
    assert r["recall"] == pytest.approx(0.5)
    assert r["f1"] == pytest.approx(0.5)
