# UMAP 可視化

`scripts/analysis/umap_figures.py`（設定 `configs/analysis/umap.yaml`。`base` で `confusion.yaml` を継承し、データ設定と配色を共有）

```bash
docker compose run --rm app uv run python scripts/analysis/umap_figures.py
```

UMAP パラメータ（`umap`）: n_neighbors 15, min_dist 0.1, metric cosine, random_state 0。

## 出力（`outputs/analysis/umap/`、各図の座標 CSV つき）
| 図 | 内容 |
|---|---|
| `esc50_umap_categories.png` | モデル（行）× 大分類（列）。該当大分類のみ着色し他は灰色の small multiples |
| `esc50_umap_audio_text.png` | EmbeddingGemma 2 の音声 2,000 本とクラステキスト 50 本を同一 UMAP に。平均コサイン類似度（`esc50_modality_cosine.csv`）と、ゼロショットで予測が集中したテキスト上位を併記 |
| `dcase_umap.png` | 機種（行）× モデル（列）で train + test の source 正常 / target 正常 / 異常 |

## 設計上の注意
- 散布図で 4 色以上は判別できない（dataviz 既定パレットの all-pairs 検証は先頭 3 色まで）ため、ESC-50 の 5 大分類は色分けせず small multiples にした
- DCASE の 3 区分は先頭 3 色 + マーカー形状（○ □ △）の二重符号化
- UMAP は大域的な距離・クラスタ間距離を保たない。クラスタの有無や重なりの定性確認に限り、定量的な主張はコサイン類似度や k-NN などの数値で行う
