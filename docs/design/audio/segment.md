# 固定長分割

`segment(x, sr, window_sec, hop_sec) -> list[np.ndarray]`

- 窓長 `window_sec`、ホップ `hop_sec` で切り出す
- 最後の窓が終端に届かない場合、終端に揃えた窓を 1 本追加（取りこぼし防止）
- 入力が窓長以下なら入力全体を 1 本返す

DCASE（10 秒）で 2 秒窓 / 1 秒ホップ → 9 セグメント。設定: `configs/experiment/dcase_anomaly.yaml` の `segmentation.segment`
