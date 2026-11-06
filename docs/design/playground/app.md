# 音声⇔テキスト類似度アプリ

設定: `configs/app/playground.yaml`（モデル設定のパス、待ち受けホスト・ポート、録音の最大長、テキストのプレフィックス、初期の例文）

## 起動

```bash
docker compose run --rm --service-ports app uv run python scripts/app/run_playground.py
```

ブラウザで http://localhost:7860 を開く。`compose.yaml` はポートをホストの `127.0.0.1` にだけ公開する（LAN からは見えない）。ポートは `--service-ports` を付けたときだけ公開されるので、ほかの実験コマンドには影響しない。

## 画面
- 入力: テキスト候補（1 行 1 候補）、テキストのプレフィックス（`text_prefixes` から選ぶ）、音声（マイク録音またはファイル）
- 出力: 類似度の表（`rank`, `text`, `cosine`, `diff_from_mean`）と補足（音声の長さ、候補数、テキストどうしの平均類似度）

想定する使い方は「テキスト候補 = 社内ナレッジなどの文書、音声 = 問い合わせ」で、話した質問に答えている文書が上位に来るかを見る。そのため既定のプレフィックスは `document`（`title: none | text: {text}`）にしている。これはモデル同梱の `config_sentence_transformers.json` に定義された文書用のプロンプトと同じ文字列。音声側にはプレフィックスを付けない。

音声とテキストの埋め込みにはモダリティギャップがあり（ESC-50 では音声と正解テキストの類似度 0.70、音声と全テキストでは 0.66）、類似度の絶対値は解釈しにくい。そのため候補を複数並べ、**順位**と**候補平均との差**で比べる。テキストどうしの平均類似度は、絶対値を読むときの目安として表示する。

## 処理（`src/egaudio/playground/`）
| 関数 | 内容 |
|---|---|
| `similarity.parse_texts(raw)` | 改行で分割し、前後の空白を除いて空行を捨てる。重複はそのまま残す |
| `similarity.apply_prefix(text, template)` | `{text}` を置き換える。`{text}` がなければ `ValueError` |
| `similarity.prepare_mic_audio((sr, x), target_sr, max_sec)` | 整数 PCM を [-1, 1) の float32 に変換し、チャンネル平均でモノラル化して `target_sr`（16kHz）にリサンプル。`max_sec` を超えた分は捨てる。音声がない・空のときは `ValueError` |
| `similarity.rank_similarities(audio_emb, text_emb, texts)` | 正規化済みの埋め込みの内積（= コサイン類似度）を降順に並べ、候補平均との差を付ける |
| `app.compute(embedder, cfg, raw_texts, prefix_name, audio)` | 上の関数をつなぐ処理本体。テキストにだけプレフィックスを付ける（音声には付けない） |
| `app.build_ui(embedder, cfg)` | Gradio の画面を作る。`ValueError` は画面上のエラーとして表示 |

モデルは起動時に 1 回だけ読み込む（`AudioTextEmbedder`、[embedding/model.md](../embedding/model.md)）。

## 依存
`gradio` は依存グループ `app` に入れ、`[tool.uv] default-groups` で既定でインストールする（2026-11-06 時点で gradio 6.31.0）。
