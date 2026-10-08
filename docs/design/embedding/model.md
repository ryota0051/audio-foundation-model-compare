# EmbeddingGemma 2 ラッパ

`AudioTextEmbedder(cfg)`（cfg = `configs/model/embeddinggemma2.yaml`）

- `SentenceTransformer(model_id, config_kwargs={"vision_config": None}, model_kwargs={"torch_dtype": dtype})` でテキスト＋音声のみロード
- `resolve_dtype`: `bfloat16` / `float32` のみ許可。`float16` 系は `ValueError`（NaN・劣化のため）。bf16 非対応 GPU では fp32 にフォールバック
- `encode_text(texts)`: プレフィックスは呼び出し側で付与済みの文字列を渡す
- `encode_audio(wavs)`: 16kHz モノラル float32 配列を `{"array": wav, "sampling_rate": 16000}` で渡す（音声にプレフィックスは付けない）
- 出力は `normalize_embeddings=True` の float32。非有限値があれば `FloatingPointError`
- バッチサイズは `batch_size.text` / `batch_size.audio`

## 確認済みの挙動（2026-10-08, sentence-transformers 6.1.0 / transformers 5.19.0）
- 受け付ける音声形式: `np.ndarray`、`{"array", "sampling_rate"}`、`{"audio": ...}`。`torch.Tensor` は不可
- 1 バッチ内で sampling_rate が混在するとエラー
- `sentence-transformers[image]`（Pillow）が必須（プロセッサが要求）
- bf16 と fp32、バッチ 1 と 8 で ESC-50 精度はほぼ同一（`scripts/probe/probe_esc50_sanity.py`）
