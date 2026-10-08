"""ESC-50 メタデータ。"""

from pathlib import Path

import pandas as pd


def label_text(category: str) -> str:
    return category.replace("_", " ")


def load_meta(meta_csv: str | Path, audio_dir: str | Path, folds: list[int] | None) -> pd.DataFrame:
    df = pd.read_csv(meta_csv)
    if folds is not None:
        df = df[df["fold"].isin(folds)]
    df = df.sort_values("filename").reset_index(drop=True)
    df["path"] = [str(Path(audio_dir) / f) for f in df["filename"]]
    return df
