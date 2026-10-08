# Step 1: ESC-50 ゼロショット分類

`src/egaudio/experiments/esc50_zeroshot.py` / 設定 `configs/experiment/esc50_zeroshot.yaml`

1. メタ CSV 読み込み（`folds` で絞り込み可）→ 44.1kHz を 16kHz モノラルに
2. 2,000 クリップを 768 次元で埋め込み（キャッシュ）
3. `prefixes` × `templates` の全組み合わせでクラステキストを埋め込み
4. `dims` ごとに MRL 切り詰め → コサイン類似度 → Top-1 / Top-5（全体と fold 平均）

## 出力（`outputs/results/esc50/`）
- `summary.csv`: prefix, template, dim, top1, top1_fold_mean, top5, top5_fold_mean
- `best.json`: 768 次元で Top-1 最良の条件
- `confusion_pairs.csv`: 最良条件の誤分類ペア上位
- `per_class_top1.csv`: 最良条件のクラス別正解率
