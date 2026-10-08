import numpy as np

from egaudio.experiments.esc50_dedup import stratified_difference, subset_accuracy_rows


def test_subset_accuracy_rows():
    correct = np.array([True, True, False, True])
    subsets = {"all": np.array([True] * 4), "clean": np.array([False, True, True, False])}
    rows = subset_accuracy_rows(correct, subsets)
    by = {r["subset"]: r for r in rows}
    assert by["all"]["n"] == 4 and by["all"]["n_correct"] == 3 and by["all"]["top1"] == 0.75
    assert by["clean"]["n"] == 2 and by["clean"]["top1"] == 0.5
    assert by["clean"]["ci95_low"] < 0.5 < by["clean"]["ci95_high"]


def test_stratified_difference_controls_class_mix():
    # クラス 0 は易しく（全問正解）、クラス 1 は難しい（全問不正解）。
    # overlap はクラス 0 に偏るので単純比較では差が出るが、クラス内で比べると差は 0
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    correct = y == 0
    overlap = np.array([True, True, True, False, True, False, False, False])
    diff, lo, hi, n_cls = stratified_difference(correct, y, overlap, n_boot=200, seed=0)
    assert correct[overlap].mean() - correct[~overlap].mean() > 0.4
    assert diff == 0.0 and lo == 0.0 and hi == 0.0 and n_cls == 2
