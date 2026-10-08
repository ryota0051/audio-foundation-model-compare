# CLAP ラッパ

`ClapEmbedder(cfg)`（`src/egaudio/embedding/clap.py`）。`configs/model/clap_larger_general.yaml` の `backend: clap` で選択。

- transformers の `ClapModel` / `ClapProcessor`（`laion/larger_clap_general`）。`get_audio_features` / `get_text_features` の `pooler_output`（512 次元, 射影後）を L2 正規化
- 入力は **48kHz** モノラル（モデル設定 `sample_rate`）。テキストにタスクプレフィックスは付けない
- プロセッサは 10 秒超の音声を**ランダムに切り出す**（`truncation: rand_trunc`）。結果を決定的にするため、`max_seconds` 窓に分割（末尾は終端揃え、`chunk_audio`）し、窓ごとの埋め込みを平均して L2 正規化（`mean_of_chunks`）。DCASE の ToyCar（11 秒）・ToyCarEmu（12 秒）が該当
- 10 秒未満はプロセッサの `repeatpad`（繰り返して埋める）に任せる
- MRL 学習されていないため実験設定では `dims: [512]` のみ
