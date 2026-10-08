import numpy as np

from egaudio.experiments.esc50_zeroshot_gap import evaluate_methods


def test_evaluate_methods_rows_and_gap_fix():
    # 音声もテキストも共通の大きな成分（ギャップ）+ クラス固有の小さな成分。
    # テキストには全音声に近いハブ成分を足す
    rng = np.random.default_rng(0)
    n_cls, dim = 4, 16
    cls_dir = np.eye(n_cls, dim)
    common_a, common_t = np.zeros(dim), np.zeros(dim)
    common_a[-1], common_t[-2] = 3.0, 3.0
    y = np.repeat(np.arange(n_cls), 5)
    audio = common_a + 0.5 * cls_dir[y] + rng.normal(0, 0.05, (len(y), dim))
    text = common_t + 0.5 * cls_dir
    text[0] += 0.4 * common_a  # ハブ
    folds = np.tile(np.arange(1, 6), n_cls)
    rows = evaluate_methods(audio, text, y, folds, ["none", "center_both_oof"], dims=[dim], topk=[1],
                            params={"csls_k": 3})
    by = {r["method"]: r for r in rows}
    assert set(by) == {"none", "center_both_oof"}
    assert by["none"]["top1"] < by["center_both_oof"]["top1"]
    assert by["none"]["max_pred_count"] > by["center_both_oof"]["max_pred_count"]
