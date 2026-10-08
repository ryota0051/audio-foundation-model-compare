# 音声前処理

`load_audio(path, target_sr, channel) -> np.ndarray[float32]`

1. `soundfile` で `(samples, channels)` として読む
2. `select_channel`
   - `channel: int` … そのチャンネルを抽出（DCASE は `0` = ch1、機械に近いマイク）
   - `channel: null` … 全チャンネル平均でモノラル化（ESC-50）
   - 範囲外は `ValueError`
3. `resample`（soxr）で `target_sr` へ。同一 sr ならそのまま

設定: チャンネルは `configs/dataset/{esc50,dcase}.yaml` の `channel`。`target_sr` は**モデル設定の `sample_rate`**（EmbeddingGemma 2 / BEATs は 16kHz、CLAP は 48kHz）を各実験ランナーが渡す

## 注意
EmbeddingGemma 2 のプロセッサは `sampling_rate` を受け取るが**リサンプルはしない**（ADR 0004）。必ずここで 16kHz にしてから渡す。
