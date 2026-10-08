# MRL 切り詰め

`truncate_and_normalize(emb, dim)`: 先頭 `dim` 次元に切り詰め、**L2 再正規化**する。`dim` > 元次元は `ValueError`。

全次元（768）で埋め込みを 1 回だけ計算・キャッシュし、512 / 256 / 128 は後処理で作る。
