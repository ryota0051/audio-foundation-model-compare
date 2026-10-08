# 0005 BEATs の取り込みとチェックポイントの扱い

- 日付: 2026-10-08
- ステータス: 採用

## 背景
EmbeddingGemma 2 の DCASE 性能の低さが「意味的埋め込みゆえ」かを確かめるため、音響的な自己教師ありモデル BEATs を同一パイプラインで比較する。BEATs は pip パッケージがなく、チェックポイントは OneDrive 配布（匿名 API ダウンロード不可）。

## 決定
- 公式実装 3 ファイル（BEATs.py, backbone.py, modules.py）を `src/egaudio/embedding/beats_vendor/` に MIT ライセンスごと取り込む。変更は相対 import 化のみ
- チェックポイントは手動ダウンロードして `models/beats/` に置く（git 管理外）。`weights_only=True` で読み込む
- 比較するチェックポイントは「自己教師ありのみ（iter3）」「トークナイザがラベル由来（iter3+ AS2M）」「AS2M ファインチューニング済み（cpt2）」の 3 段階 + iter3 の中間層
- 分類ヘッドは使わずエンコーダ出力の時間平均を埋め込みとする
