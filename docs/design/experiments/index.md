# experiments

- [esc50_zeroshot.md](esc50_zeroshot.md): Step 1 ESC-50 ゼロショット分類
- [esc50_retrieval.md](esc50_retrieval.md): Step 1b ESC-50 音声⇔音声検索によるモデル比較
- [dcase_anomaly.md](dcase_anomaly.md): Step 2 DCASE 異常検知

## 実行方法
```bash
docker compose build
docker compose run --rm app uv run pytest
docker compose run --rm app uv run python scripts/experiments/run_esc50_zeroshot.py --config configs/experiment/esc50_zeroshot.yaml
docker compose run --rm app uv run python scripts/experiments/run_dcase_anomaly.py --config configs/experiment/dcase_anomaly.yaml
```
スモーク用設定は `configs/experiment/smoke/`（ESC-50 fold1 / DCASE fan のみ）。
