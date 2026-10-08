# 0001 Docker + uv による実行環境

- 日付: 2026-10-08
- ステータス: 採用

## 背景
CLAUDE.md で Docker と uv の利用が指定。ホストは Windows + RTX 4090。

## 決定
- `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04` に uv を入れ、`uv sync` で仮想環境 `/opt/venv` を構築（`docker/Dockerfile`, `compose.yaml`）
- torch / torchvision は PyTorch の cu128 インデックスから取得（PyPI 版 torchvision は CUDA torch と不整合で `torchvision::nms` エラー）
- `sentence-transformers[image]` を使う（EmbeddingGemma2Processor が Pillow を要求）。`[audio]` は kenlm のビルドにコンパイラが要るため不採用
- `data/` は読み取り専用、`outputs/` と `.hf_cache/` はボリュームマウント
