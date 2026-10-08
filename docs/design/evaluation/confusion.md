# 混同行列と閾値判定

`src/egaudio/evaluation/confusion.py`

- `confusion_matrix(y_true, y_pred, n_classes)`: 行 = 真値, 列 = 予測 の件数
- `row_normalize(cm)`: 行を真値件数で割る（対角 = クラス別再現率）。空行は 0
- `group_confusion(cm, groups, n_groups)`: クラス→グループ対応で集約（ESC-50 の 5 大分類 = target // 10）
- `threshold_from_normal(scores, quantile)`: 学習用正常音スコアの分位点を閾値に（公式ベースラインと同じ 90%）
- `binary_report(y_true, scores, threshold)`: スコア > 閾値 を異常として tn / fp / fn / tp, precision / recall / F1

DCASE の閾値用スコアは `KnnCosineScorer.score_loo()`（自分自身を除いた近傍）で計算する。自分自身を含めると距離 0 になり閾値が不当に低くなるため。

## 作図スクリプト
`scripts/analysis/confusion_figures.py`（設定 `configs/analysis/confusion.yaml`、キャッシュ済み埋め込みのみ使用）

```bash
docker compose run --rm app uv run python scripts/analysis/confusion_figures.py
```

出力（`outputs/analysis/confusion/`、各 PNG に同名 CSV）:
- `esc50_cm_<条件>.png`: 50x50（行正規化）。ゼロショットは Step1 の best.json の条件、k-NN は 5 fold の out-of-fold 予測
- `esc50_category_cm.png`: 5 大分類に集約
- `esc50_per_class_recall.png`: クラス別再現率（ゼロショット / EmbeddingGemma2 k-NN / BEATs iter3 k-NN）
- `dcase_confusion.png`, `dcase_binary_report.csv`: 機種 × ドメインの 2x2（whole / single / kNN k=1 / 768）

配色は dataviz 既定パレット（sequential blue、3 系列は categorical の先頭 3 色。aqua は地とのコントラスト 3:1 未満のため凡例・マーカー形状・CSV を併記）。
