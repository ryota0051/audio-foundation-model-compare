# キャッシュ済み埋め込みの読み込み

`src/egaudio/analysis/loaders.py`。作図スクリプト（混同行列・UMAP）で共通に使う。

- `load_esc50_retrieval_embeddings(retrieval_cfg, label, n_clips)`: Step1b のキャッシュ（`<cache_dir>/<モデル設定名>/audio_n<本数>.npy`）を `full_dim` で L2 正規化。無ければ `FileNotFoundError`（先に Step1b を実行）
- `load_dcase_whole(cache_dir, machine, dim)`: DCASE の `train_whole.npy` / `test_whole.npy` を dim 次元・L2 正規化
- `dcase_domain_labels(clips)`: `source normal` / `target normal` / `anomaly` の 3 区分
