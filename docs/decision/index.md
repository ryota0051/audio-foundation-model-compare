# ADR 一覧

| No | タイトル | 概要 |
|---|---|---|
| [0001](0001-docker-uv-environment.md) | Docker + uv による実行環境 | CUDA コンテナ内で uv 管理、依存の注意点 |
| [0002](0002-dcase-channel-and-domain-pooling.md) | DCASE の入力チャンネルとドメイン別プール | ch1 のみ、source/target 別プールの扱い |
| [0003](0003-evaluation-metrics.md) | 評価指標の実装方針 | 公式 evaluator を移植し自前で計算 |
| [0004](0004-audio-resampling-before-model.md) | モデル投入前に自前で 16kHz 化 | プロセッサはリサンプルしない |
