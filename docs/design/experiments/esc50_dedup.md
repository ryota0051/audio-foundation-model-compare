# Step 1d: ESC-50 ゼロショットの重複除去再評価

`src/egaudio/experiments/esc50_dedup.py` / 設定 `configs/experiment/esc50_dedup.yaml`。GPU 不要（Step 1 のキャッシュを使う）。

```bash
docker compose run --rm app uv run python scripts/experiments/run_esc50_dedup.py --config configs/experiment/esc50_dedup.yaml
```

## 評価集合
`larger_clap_general` が Freesound の完全版・重複除去版のどちらで学習されたか明記がないため、**完全版と重複しないクリップ（clean）**を、どちらの場合でも汚染のない集合とする。

| 集合 | 内容 |
|---|---|
| all | 全 2,000 本 |
| overlap | 完全版と元音源が重複 |
| clean | 完全版と重複しない |
| fold1_all / fold1_clean | CLAP 公式の評価（第 1 fold）と比べる用 |

## 対照比較
EmbeddingGemma 2 を対照に、overlap と clean の正解率差をモデル間で比べる（学習時の記憶による底上げか、クリップの難易度差かを切り分ける）。clean はクラス構成が偏る（0〜36 本/クラス）ため、`stratified_difference` で**クラス内**の差を平均し、クラス内ブートストラップで 95% 区間を付ける。

## 出力（`outputs/results/esc50_dedup/`）
- `clip_overlap.csv`: クリップごとの重複フラグ（版ごと）
- `summary.csv`: モデル × 条件 × 集合の Top-1 と Wilson 95% 区間
- `stratified_difference.csv`: クラス内比較の overlap − clean と 95% 区間
