# 異常スコア

いずれも `fit(bank) -> self`, `score(x) -> np.ndarray`（大きいほど異常）。

## KnnCosineScorer(k)
正常メモリバンクとのコサイン距離 `1 - cos` の小さい方から k 個の平均。`k` はバンクサイズで頭打ち（ターゲットは 10 本しかない）。

## MahalanobisScorer(covariance)
- `covariance`: `ledoit_wolf`（既定, 縮約）/ `empirical`
- スコアはマハラノビス距離（平方根を取った値）
- `with_shared_precision(base, bank)`: ターゲット 10 本では共分散を推定できないため、**精度行列はソースのものを共有し平均だけターゲットから**求める
