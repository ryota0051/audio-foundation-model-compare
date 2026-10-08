# 埋め込みキャッシュ

`cached_array(path, fn)`: `path` (.npy) があれば読み込み、なければ `fn()` を計算して保存。

- ESC-50: `outputs/cache/esc50/audio_<filename一覧のsha1>.npy`, `text_<テキスト一覧のsha1>.npy`
- DCASE: `outputs/cache/dcase/<machine>/{train,test}_whole.npy`, `{split}_seg_w{win}_h{hop}.npy` と `_index.npy`（セグメント→クリップ番号）

モデルや前処理を変えたときは該当キャッシュを削除すること。
