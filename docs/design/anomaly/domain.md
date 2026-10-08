# ドメイン別プール

`configs/experiment/dcase_anomaly.yaml` の `pooling`:

- `single`: 学習用正常音 1,000 本（source 990 + target 10）を 1 つのバンクにする
- `domain_min`: source バンクと target バンクで別々にスコアを出し、`domain_min_scores` で要素ごとに小さい方（近い方のドメイン）を採用
