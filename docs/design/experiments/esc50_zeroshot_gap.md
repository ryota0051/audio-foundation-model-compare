# Step 1c: ESC-50 ゼロショットの補正比較

`src/egaudio/experiments/esc50_zeroshot_gap.py` / 設定 `configs/experiment/esc50_zeroshot_gap.yaml`（`base` で Step 1 の設定を継承）

Step 1 と同じプレフィックス × テンプレート × 次元で、`gap_correction.methods` の各補正を評価する。埋め込みは Step 1 のキャッシュを再利用し、無ければ計算する。

```bash
docker compose run --rm app uv run python scripts/experiments/run_esc50_zeroshot_gap.py --config configs/experiment/esc50_zeroshot_gap.yaml
```

## 出力（`outputs/results/esc50_zeroshot_gap/`）
- `summary.csv`: prefix, template, dim, method, top1, top5, max_pred_count, n_never_predicted
- `best_per_method.csv`: 補正方法ごとの Top-1 最良条件
