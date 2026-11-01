# EmbeddingGemma 2 で音声のクラス分類と異常検知をやってみた

## リポジトリについて

マルチモーダル埋め込みモデル **EmbeddingGemma 2**（`google/embeddinggemma-2`）の音声埋め込みが、どこまで音声タスクに使えるかを調べたリポジトリです。次の 2 つのタスクを試しました。

- **クラス分類**: 環境音データセット ESC-50 を使い、テキストラベルとの類似度で分類する（ゼロショット）。音声どうしの検索（k-NN）と線形プローブでも評価する
- **異常検知**: DCASE 2026 Task 2 の開発データを使い、正常音の埋め込みからの距離（k-NN / マハラノビス距離）で機械音の異常を検知する

比較対象として、同じパイプラインで次のモデルも評価しています。

| モデル | 種類 | 主な比較先 |
|---|---|---|
| EmbeddingGemma 2 | テキスト・画像・音声の汎用埋め込み | - |
| CLAP（`laion/larger_clap_general`） | 音声⇔テキストの対照学習モデル | ゼロショット / 異常検知 |
| BEATs（iter3 / iter3+AS2M / AS2M FT） | 音響の自己教師あり学習モデル | 音声検索 / 異常検知 |

設計や判断の経緯は [docs/](docs/) にまとめています。

- 仕様書: [docs/design/index.md](docs/design/index.md)
- ADR（設計判断）: [docs/decision/index.md](docs/decision/index.md)
- 作業ログ（詳しい結果と考察）: [docs/work_log/index.md](docs/work_log/index.md)

### ディレクトリ構成

```
.
├── configs/       # 実験・モデル・データセット・分析の設定（YAML）
├── docker/        # Dockerfile
├── docs/          # 仕様書・ADR・作業ログ
├── scripts/
│   ├── experiments/  # 実験の実行スクリプト
│   ├── analysis/     # 図表作成スクリプト
│   └── probe/        # モデル API の動作確認用
├── src/egaudio/   # 本体（前処理・埋め込み・ゼロショット・異常検知・評価）
├── tests/         # pytest
├── data/          # データセット（git 管理外、手動配置）
├── models/        # BEATs チェックポイント（git 管理外、手動配置）
└── outputs/       # 結果・埋め込みキャッシュ・図（git 管理外）
```

## 環境構築

### 前提

- NVIDIA GPU と NVIDIA Container Toolkit が使える Docker 環境（動作確認: RTX 4090）
- Docker Compose

コンテナは `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04` の上に uv で Python 3.12 環境を作ります（[ADR 0001](docs/decision/0001-docker-uv-environment.md)）。

### Docker のビルド

```bash
docker compose build
```

Hugging Face のモデル取得に認証が必要な場合は、環境変数 `HF_TOKEN` を設定してから実行してください（`compose.yaml` からコンテナに渡されます）。ダウンロードしたモデルは `.hf_cache/` にキャッシュされます。

ビルド後、テストが通ることを確認します。

```bash
docker compose run --rm app uv run pytest
```

### データ配置

データとチェックポイントは git 管理外なので、手動で次のように配置します。`data/` と `models/` はコンテナに読み取り専用でマウントされます。

```
data/
├── ESC-50/                  # https://github.com/karolpiczak/ESC-50 を clone したもの
│   ├── audio/               # *.wav（2,000 クリップ）
│   └── meta/esc50.csv
├── DCASE/                   # DCASE 2026 Task 2 開発データ
│   ├── ToyCar/
│   │   ├── attributes_00.csv
│   │   ├── train/           # section_00_source_train_normal_0000_spd_1.wav など
│   │   └── test/            # section_00_source_test_anomaly_0000_spd_1.wav など
│   ├── ToyCarEmu/
│   ├── bearingEmu/
│   ├── fan/
│   ├── gearboxEmu/
│   ├── sliderEmu/
│   └── valveEmu/
└── laion_freesound_meta/    # ESC-50 の重複除去評価でのみ使用
    ├── freesound_meta.csv
    └── freesound_no_overlap_meta.csv
models/
└── beats/                   # BEATs 比較でのみ使用
    ├── BEATs_iter3.pt
    ├── BEATs_iter3_plus_AS2M.pt
    └── BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt
```

| データ | 入手先 | 備考 |
|---|---|---|
| ESC-50 | [karolpiczak/ESC-50](https://github.com/karolpiczak/ESC-50) | |
| DCASE 2026 Task 2 開発データ | [DCASE 2026 Task 2](https://dcase.community/challenge2026/task-first-shot-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring) のページから Zenodo へ | CC BY-NC-SA 4.0。2ch のうち ch1 のみ使用（[ADR 0002](docs/decision/0002-dcase-channel-and-domain-pooling.md)） |
| LAION-Audio-630K Freesound メタデータ | Hugging Face `Meranti/CLAP_freesound` | 研究目的に限り利用可 |
| BEATs チェックポイント | [microsoft/unilm の BEATs](https://github.com/microsoft/unilm/tree/master/beats) | OneDrive 配布のため手動ダウンロード（[ADR 0005](docs/decision/0005-beats-vendoring-and-checkpoints.md)） |

データの置き場所を変える場合は `configs/dataset/*.yaml` を、チェックポイントの場所を変える場合は `configs/model/beats_*.yaml` を編集してください。

## 実験内容

実験はすべて `--config` に YAML を渡して実行します。条件（次元、テンプレート、スコア方式など）は設定ファイルで変更できます。埋め込みは `outputs/cache/` にキャッシュされるので、2 回目以降の実行は速くなります。結果は `outputs/results/` に出力されます。

以下のコマンドはすべて次の形で実行します。

```bash
docker compose run --rm app uv run python <スクリプト> --config <設定ファイル>
```

### Step 1: ESC-50 ゼロショット分類

音声埋め込みと、クラス名から作ったテキスト埋め込みのコサイン類似度で分類します。テキスト側のプレフィックス（なし / `task: classification` / `task: search result`）、ラベルテンプレート（`{label}`、`the sound of {label}`、自然文 `the sound of a dog barking` など）、MRL による次元（768 / 512 / 256 / 128）の組み合わせを比較します。

| 内容 | スクリプト | 設定 |
|---|---|---|
| ゼロショット分類 | `scripts/experiments/run_esc50_zeroshot.py` | `configs/experiment/esc50_zeroshot.yaml`<br>`configs/experiment/clap/esc50_zeroshot.yaml` |
| モダリティギャップ・ハブ化の補正（中心化・列標準化・CSLS） | `scripts/experiments/run_esc50_zeroshot_gap.py` | `configs/experiment/esc50_zeroshot_gap.yaml`<br>`configs/experiment/clap/esc50_zeroshot_gap.yaml` |
| 音声⇔音声検索（5 fold の k-NN・Precision@k・線形プローブ、全モデル） | `scripts/experiments/run_esc50_retrieval.py` | `configs/experiment/esc50_retrieval.yaml` |
| CLAP の学習データ重複を除いた再評価 | `scripts/experiments/run_esc50_dedup.py` | `configs/experiment/esc50_dedup.yaml` |

補正と重複除去の実験はゼロショット分類のキャッシュを再利用するので、先にゼロショット分類を実行してください。

### Step 2: DCASE 異常検知

学習用の正常音の埋め込みをプールし、テスト音の埋め込みとの距離を異常スコアにします。スコア方式（kNN k=1/5/10、マハラノビス距離）、正常音プール（全体 / source・target 別）、クリップ全体か 2 秒窓に分割するか、を比較します。評価指標は公式の official score（全機種の AUC(source)、AUC(target)、pAUC の調和平均）です（[ADR 0003](docs/decision/0003-evaluation-metrics.md)）。

| モデル | 設定 |
|---|---|
| EmbeddingGemma 2 | `configs/experiment/dcase_anomaly.yaml` |
| CLAP | `configs/experiment/clap/dcase_anomaly.yaml` |
| BEATs | `configs/experiment/dcase_beats/*.yaml` |

スクリプトはいずれも `scripts/experiments/run_dcase_anomaly.py` です。

### 実行例

```bash
# ESC-50 ゼロショット（EmbeddingGemma 2）
docker compose run --rm app uv run python scripts/experiments/run_esc50_zeroshot.py --config configs/experiment/esc50_zeroshot.yaml

# DCASE 異常検知（EmbeddingGemma 2）
docker compose run --rm app uv run python scripts/experiments/run_dcase_anomaly.py --config configs/experiment/dcase_anomaly.yaml

# 動作確認用の小さい設定（ESC-50 fold1 のみ / DCASE fan のみ）
docker compose run --rm app uv run python scripts/experiments/run_esc50_zeroshot.py --config configs/experiment/smoke/esc50_zeroshot.yaml
docker compose run --rm app uv run python scripts/experiments/run_dcase_anomaly.py --config configs/experiment/smoke/dcase_anomaly.yaml
```

### 分析・図表

実験結果やキャッシュから図表を作ります（出力先は `outputs/analysis/`）。対応する実験を先に実行しておく必要があります。

| 内容 | スクリプト | 設定 |
|---|---|---|
| DCASE のモデル比較表 | `scripts/analysis/compare_models.py` | `configs/analysis/model_comparison.yaml` |
| 混同行列（ESC-50 / DCASE） | `scripts/analysis/confusion_figures.py` | `configs/analysis/confusion.yaml` |
| 埋め込みの UMAP | `scripts/analysis/umap_figures.py` | `configs/analysis/umap.yaml` |
| DCASE のマハラノビス距離の分布 | `scripts/analysis/mahalanobis_distribution.py` | `configs/experiment/dcase_anomaly.yaml` |

## 実験結果

詳しい表と考察は [作業ログ](docs/work_log/20261008.md) にあります。

### ESC-50 クラス分類（50 クラス、チャンスレート 2%）

| モデル | ゼロショット Top-1 | 音声 k-NN@1 | 線形プローブ |
|---|---|---|---|
| EmbeddingGemma 2 | 27.4% | 70.2% | 81.8% |
| EmbeddingGemma 2 + CSLS 補正 | 29.8% | - | - |
| BEATs iter3（自己教師ありのみ） | - | 72.0% | 88.3% |
| BEATs iter3+AS2M FT | - | 95.2% | 96.6% |
| CLAP | **95.1%** | **97.0%** | **97.6%** |

> **集計方法**: 音声 k-NN@1 と線形プローブは、公式の 5 fold 交差検証（4 fold を検索対象・学習データ、残り 1 fold を評価データ）の 5 回の平均です。ゼロショットは学習を行わないので、全 2,000 クリップの正解率です（各 fold が 400 本ずつ同数なので、fold ごとの正解率の平均と同じ値になります）。重複除去の評価は、重複あり / なしの部分集合ごとの正解率です。

- EmbeddingGemma 2 のゼロショットは 27.4% で、CLAP（95.1%）に大きく届かない。テンプレートを CLAP と揃えても改善しない
- 一方、音声どうしの k-NN は 70%、線形プローブは 82% で、**音声埋め込み自体はクラス情報をかなり持っている**。弱いのは音声とテキストの対応付けの側
- EmbeddingGemma 2 では音声とテキストの埋め込みが離れており（モダリティギャップ）、予測が一部のラベル（engine、wind など）に集中する（ハブ化）。補正でハブ化は解消するが、Top-1 の改善は +2.4pt にとどまる
- ESC-50 のクリップの 82% は CLAP の学習データ（LAION-Audio-630K）と元音源が重複しうるが、重複しない 354 クリップでも CLAP は 91% 台、EmbeddingGemma 2 は 21〜27% で、結論は変わらない

### DCASE 異常検知（official score）

| モデル | 最良条件 |
|---|---|
| BEATs iter3（自己教師ありのみ） | **0.584** |
| 公式ベースライン MAHALA | 0.577 |
| 公式ベースライン MSE（AE） | 0.567 |
| CLAP | 0.561 |
| EmbeddingGemma 2 | 0.522 |

> **集計方法**: DCASE には fold がないので、公式の train / test 分割（各機種で学習用の正常音 1,000 本、テスト 200 本）で 1 回だけ評価した値です。乱数を含まない単一の実行なので、ばらつきは推定していません。「最良条件」は、スコア方式・正常音プール・分割の有無・次元の組み合わせのうち、official score が最も高かったものです。

- EmbeddingGemma 2 はどの条件でも公式ベースラインに届かない。ドメイン内で見ても AUC は 0.5〜0.6 で、**埋め込み上で正常音と異常音がほぼ重なっている**
- 自己教師ありの BEATs iter3 は公式ベースラインをわずかに上回る。BEATs ではラベル情報が強いほど（iter3 → +AS2M → FT）スコアが下がり、異常より動作条件（ドメイン）の違いに反応しやすくなる傾向があった
- 意味寄りのモデル（CLAP、EmbeddingGemma 2）はどちらも BEATs iter3 に届かず、「意味的な埋め込みは機械音の細かい違いに弱い」という仮説と合う結果になった

### まとめ

EmbeddingGemma 2 の音声埋め込みは「何の音か」という大まかな意味はとらえていて、音声どうしの検索では自己教師あり BEATs と同程度に使えます。ただし、テキストラベルを使うゼロショット分類は音声専用の CLAP に大きく劣り、同じ機械の正常音と異常音の細かい違いもほとんど表現できていません。
