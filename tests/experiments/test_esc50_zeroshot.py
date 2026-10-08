import numpy as np

from egaudio.experiments.esc50_zeroshot import evaluate_condition


def test_evaluate_condition_perfect_alignment():
    text = np.eye(3, 4, dtype=np.float32)
    audio = np.repeat(text, 2, axis=0)
    y = np.repeat(np.arange(3), 2)
    folds = np.array([1, 2] * 3)
    rows = evaluate_condition(audio, text, y, folds, dims=[4, 2], topk=[1])
    assert rows[0] == {"dim": 4, "top1": 1.0, "top1_fold_mean": 1.0}
    assert rows[1]["dim"] == 2
