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
- `esc50_cm_<条件>.png`: 50x50 の件数。1 本以上のセルのみ数字を表示し、行合計を右軸・予測合計を上軸に表示。予測合計がクラスサイズの 2 倍以上（=予測の集中）なら太字。ゼロショットは Step1 の best.json の条件、k-NN は 5 fold の out-of-fold 予測
- `esc50_category_cm.png`: 5 大分類に集約した件数（行合計・予測合計つき）
- `esc50_per_class_recall.png`: クラス別再現率（ゼロショット / EmbeddingGemma2 k-NN / BEATs iter3 k-NN）
- `dcase_confusion.png`, `dcase_binary_report.csv`: 機種 × ドメインの 2x2 件数（whole / single / kNN k=1 / 768）

配色は dataviz 既定パレット（sequential blue、3 系列は categorical の先頭 3 色。aqua は地とのコントラスト 3:1 未満のため凡例・マーカー形状・CSV を併記）。

## 表示方針
混同行列は **% ではなく件数** で表示する（色も件数、上限 = 1 行の件数）。
- ESC-50（各クラス 40 本・各大分類 400 本）も DCASE（各ドメイン 正常 50 / 異常 50）もクラスが均衡しており、行正規化しても情報は同じ。件数のほうが母数（1 本 = 2.5% など）が伝わる
- ゼロショットの「予測の集中」は列方向の現象で、行正規化の % では見えないため、予測合計を併記する
- クラス別再現率の図（`esc50_per_class_recall.png`）は再現率そのものの比較なので % のまま

## ESC-50 の大分類の決め方
- **出典**: ESC-50 公式 README（`data/ESC-50/README.md`, https://github.com/karolpiczak/ESC-50）の 5 大分類 × 各 10 クラスの表。クラスと大分類の対応は README を正とし、本リポジトリには書き写さない
- **割り当て規則**: メタデータ（`esc50.csv`）に大分類の列はないが、`target` 番号が README の表と同じ順（大分類ごとに 10 個ずつ連番）で振られているため、`target // (クラス数 / 大分類数)` で割り当てる（`scripts/analysis/confusion_figures.py`）
- **表示名**: `configs/analysis/confusion.yaml` の `esc50.category_groups`（README の表の列順に並べる）
- **前提**: 「番号が大分類ごとに連番」は ESC-50 固有。別データセットに流用する場合は対応表を明示的に持つ必要がある
