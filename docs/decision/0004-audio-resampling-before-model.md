# 0004 モデル投入前に自前で 16kHz 化

- 日付: 2026-10-08
- ステータス: 採用

## 背景
プローブ（`scripts/probe/probe_sampling_rate.py`）で、44.1kHz の配列を `sampling_rate=44100` 付きで渡しても、sampling_rate なしで渡したときと同一の埋め込みになった。16kHz 版との cos は 0.907。プロセッサは sampling_rate を検証（バッチ内混在の検出）に使うだけでリサンプルしない。

## 決定
`egaudio.audio.preprocess.load_audio` で必ず 16kHz モノラルにし、`{"array": wav, "sampling_rate": 16000}` でモデルに渡す。
