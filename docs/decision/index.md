# ADR 一覧

| No | タイトル | 概要 |
|---|---|---|
| [0001](0001-docker-uv-environment.md) | Docker + uv による実行環境 | CUDA コンテナ内で uv 管理、依存の注意点 |
| [0002](0002-dcase-channel-and-domain-pooling.md) | DCASE の入力チャンネルとドメイン別プール | ch1 のみ、source/target 別プールの扱い |
| [0003](0003-evaluation-metrics.md) | 評価指標の実装方針 | 公式 evaluator を移植し自前で計算 |
| [0004](0004-audio-resampling-before-model.md) | モデル投入前に自前で 16kHz 化 | プロセッサはリサンプルしない |
| [0005](0005-beats-vendoring-and-checkpoints.md) | BEATs の取り込みとチェックポイントの扱い | 公式実装を vendoring、チェックポイントは手動配置 |
| [0006](0006-clap-comparison.md) | CLAP を比較対象に追加 | larger_clap_general のみ、48kHz、決定的窓分割、ESC-50 重複の注意 |
| [0007](0007-playground-gradio-app.md) | 発話音声とテキストの近さは Gradio アプリで試す | localhost のみ公開、順位と平均との差で表示、コーパス評価は保留 |
