# 音声⇔音声検索の評価関数

類似度はすべてコサイン類似度（`egaudio.zeroshot.classify.cosine_similarity`）。

- `knn_predict(db, db_y, query, k)`: 上位 k 件のラベルの多数決。同票なら類似度の合計が大きいクラス
- `precision_at_k(db, db_y, query, query_y, k)`: 上位 k 件のうちクエリと同じクラスの割合の平均
- `linear_probe_accuracy(train_x, train_y, test_x, test_y, c, max_iter)`: StandardScaler + ロジスティック回帰の正解率（埋め込みに含まれる情報量の目安）
