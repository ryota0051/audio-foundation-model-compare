# Step 1b: ESC-50 音声⇔音声検索によるモデル比較

`src/egaudio/experiments/esc50_retrieval.py` / 設定 `configs/experiment/esc50_retrieval.yaml`

テキストを介さないため、テキスト非対応の BEATs と EmbeddingGemma 2 を同条件で比べられる。

1. `models` に並べたモデル設定ごとに 2,000 クリップを埋め込み（`outputs/cache/esc50_retrieval/<モデル設定名>/`）、L2 正規化
2. 公式 5 fold で交差検証: テスト fold をクエリ、残り 4 fold（1,600 本）を検索対象に
3. k-NN 分類正解率（`knn_k`）、Precision@k（`precision_k`）、線形プローブ正解率（`linear_probe`）

公式 fold は同一音源由来のクリップを同じ fold にまとめているため、検索対象とクエリで音源が重複しない。

```bash
docker compose run --rm app uv run python scripts/experiments/run_esc50_retrieval.py --config configs/experiment/esc50_retrieval.yaml
```

## 出力（`outputs/results/esc50_retrieval/`）
- `per_fold.csv`: model, fold, metric, value
- `summary.csv`: モデル × 指標の fold 平均
