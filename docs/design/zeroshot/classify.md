# ゼロショット分類

- `cosine_similarity(a, b)`: 行ごとに L2 正規化した内積
- `topk_accuracy(sim, y, k)`: 類似度上位 k に正解が含まれる割合
- `confusion_pairs(y, pred, classes, top_n)`: 誤分類 (真値, 予測) ペアを件数降順
