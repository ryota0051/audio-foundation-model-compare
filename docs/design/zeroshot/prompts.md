# テキストクエリ生成

`build_texts(classes, template, prefix, natural)`

- `template` の `{label}` はクラス名の `_` を空白にしたもの、`{natural}` は `natural_labels` 辞書の自然文（例 `dog` → `a dog barking`）
- `prefix` の `{text}` に template 展開結果を埋め込む
- テンプレート・プレフィックス・自然文辞書はすべて `configs/experiment/esc50_zeroshot.yaml`

| prefix 名 | 文字列 |
|---|---|
| none | `{text}` |
| classification | `task: classification \| query: {text}` |
| search | `task: search result \| query: {text}` |
