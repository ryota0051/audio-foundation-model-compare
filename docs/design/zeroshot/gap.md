# モダリティギャップ・ハブ化の補正

`src/egaudio/zeroshot/gap.py`。いずれも**正解ラベルは使わない**。方法の一覧と評価データの使い方はモジュールの docstring を正とする。

- `corrected_similarity(method, audio, text, folds, params)`: 補正後の (音声 x テキスト) 類似度
  - `center_text`: クラステキストの平均を引いて再正規化
  - `center_both_oof`: テキスト平均に加え、音声平均を**評価 fold 以外**から推定して引く（評価データ不使用）
  - `center_both_all`: 音声平均を全クリップから推定（ラベルなしで評価データを使うトランスダクティブ版）
  - `column_zscore`: 類似度をテキスト（列）ごとに標準化
  - `csls`: CSLS。`2*cos - r_audio - r_text`（r は相手側上位 `csls_k` 件との類似度平均）
- `prediction_concentration(pred, n_classes)`: 最多予測ラベルの回数と、一度も予測されないクラス数
