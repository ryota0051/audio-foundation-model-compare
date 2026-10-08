# Step 2: DCASE 距離ベース異常検知

`src/egaudio/experiments/dcase_anomaly.py` / 設定 `configs/experiment/dcase_anomaly.yaml`

機種ごとに:
1. train / test の wav を列挙しファイル名から domain・正常/異常を取得（`egaudio.data.dcase`）
2. ch1 を 16kHz で読み込み、以下のモードで埋め込み（キャッシュ）
   - `whole`: クリップ全体を 1 埋め込み
   - `segment_{max,mean}`: 2 秒窓・1 秒ホップで分割して埋め込み。バンクは全セグメント、テストはセグメントスコアを max / mean でクリップに集約
3. `dims` × `pooling` × スコアラ（kNN k=1,5,10 / マハラノビス）でスコア → 公式定義の AUC / pAUC

## 出力（`outputs/results/dcase/`）
- `per_machine.csv`: 機種 × 条件ごとの auc_source / auc_target / pauc
- `summary.csv`: 条件ごとの機種横断調和平均と official_score（降順）
- `baseline.csv`: 公式ベースライン（`configs/experiment/dcase_baseline_scores.yaml`）
- `comparison_best_vs_baseline.csv`: 最良条件と MAHALA / MSE の機種別比較

## 分析スクリプト
`scripts/analysis/mahalanobis_distribution.py`: キャッシュ済み whole 埋め込みでマハラノビス距離（単一プール）の分布をヒストグラム化し、公式 AUC に加えてドメイン内 AUC（同ドメインの正常 vs 異常）とスコアでドメインを当てる AUC を出力（`outputs/analysis/mahalanobis_distribution/`）。matplotlib は依存に含めず `uv run --with matplotlib` で実行する。
```bash
docker compose run --rm app uv run --with matplotlib python scripts/analysis/mahalanobis_distribution.py --dim 768
```

## モデル切り替え・設定継承
- モデル設定の `backend`（`embeddinggemma2` / `beats`）で埋め込みモデルを選ぶ
- 設定ファイルは `base: <path>` で他の設定を継承し差分だけ書ける（`egaudio.config.loader`）。BEATs 用は `configs/experiment/dcase_beats/*.yaml`

```bash
docker compose run --rm app uv run python scripts/experiments/run_dcase_anomaly.py --config configs/experiment/dcase_beats/beats_iter3.yaml
```

## モデル間比較
`scripts/analysis/compare_models.py`（設定 `configs/analysis/model_comparison.yaml`）: 各モデルの最良条件と固定条件（whole / 768 / single / マハラノビス）を公式ベースラインと並べて `outputs/analysis/model_comparison/` に出力。
