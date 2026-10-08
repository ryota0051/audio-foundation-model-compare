# BEATs ラッパ

`BeatsEmbedder(cfg)`（`src/egaudio/embedding/beats.py`）。`configs/model/beats_*.yaml` の `backend: beats` で選択（`egaudio.embedding.factory.build_audio_embedder`）。

- モデル定義は microsoft/unilm の公式実装を `src/egaudio/embedding/beats_vendor/` に取り込み（MIT, import を相対化したのみ）
- チェックポイントは `models/beats/`（git 管理外, compose で読み取り専用マウント）。`torch.load(weights_only=True)` で読む
- 前処理は BEATs 内蔵（kaldi fbank 128 mel, 16kHz）。torchaudio が必要
- `layer`: 1 始まりの層番号（`null` = 最終層 12）。エンコーダの `tgt_layer` で指定層の出力を取り、**時間方向平均**で 768 次元に
- ファインチューニング済み（`cfg.finetuned_model`）でも分類ヘッド（527 クラス）は通さない
- 同じ長さの波形だけでバッチを組む（`group_by_length`）のでパディング不要
- `dtype: float32`（小さいモデルなので bf16 にしない）
- MRL 学習されていないため、実験設定では `dims: [768]` のみ

| 設定 | チェックポイント | 層 | 性質 |
|---|---|---|---|
| beats_iter3 | BEATs_iter3.pt | 12 | 自己教師ありのみ |
| beats_iter3_l6 | BEATs_iter3.pt | 6 | 同・中間層 |
| beats_iter3_plus_as2m | BEATs_iter3_plus_AS2M.pt | 12 | 自己教師あり（トークナイザが AS2M ラベル付きモデル由来） |
| beats_iter3_plus_as2m_ft | BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt | 12 | AS2M ファインチューニング済み |
