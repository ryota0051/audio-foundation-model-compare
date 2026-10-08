# 外部学習データとの重複判定

`src/egaudio/data/overlap.py`

- `load_freesound_ids(csv_path, column="freesound_id")`: メタデータ CSV の ID 列だけを文字列集合で読む
- `overlap_mask(src_ids, ids)`: 元音源 ID（ESC-50 の `src_file` = Freesound ID）が集合に含まれるクリップを True。型の違いを避けるため文字列で比較
- `wilson_interval(successes, n)`: 二項比率の Wilson 95% 信頼区間

ESC-50 のクリップは元音源（Freesound の録音）から 5 秒を切り出したもの。判定は**元音源単位**なので、同じ録音の別区間が学習に含まれていても重複とみなす（保守的）。

## 外部データ
LAION-Audio-630K Freesound のメタデータ（Hugging Face `Meranti/CLAP_freesound`）を `data/laion_freesound_meta/` に手動配置（git 管理外）。**ダウンロード時点で研究目的に限る利用条件に同意**したものとみなされる。

| ファイル | 内容 |
|---|---|
| `freesound_meta.csv`（105MB） | 完全版 515,581 件 |
| `freesound_no_overlap_meta.csv`（94MB） | ESC-50 / FSD50K / UrbanSound8K / Clotho と重複するものを除いた版 460,801 件 |
