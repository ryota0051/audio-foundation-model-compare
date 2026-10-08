"""Step 1d: CLAP の学習データ（LAION-Audio-630K Freesound）と重複する ESC-50 クリップを除いた再評価。

larger_clap_general が Freesound の完全版・重複除去版のどちらで学習されたかは明記されていないため、
完全版の ID と重複しないクリップ（clean）をどちらの場合でも汚染のない評価集合とみなす。
重複あり/なしの差をモデル間で比べ、学習時の記憶による底上げかクリップ難易度の差かを切り分ける。
ゼロショットの埋め込みは各モデルの Step 1 キャッシュを使う（GPU 不要）。
"""

from pathlib import Path

import numpy as np
import pandas as pd

from egaudio.config.loader import load_yaml
from egaudio.data.esc50 import load_meta
from egaudio.data.overlap import load_freesound_ids, overlap_mask, wilson_interval
from egaudio.embedding.mrl import truncate_and_normalize
from egaudio.experiments.esc50_zeroshot import _key
from egaudio.zeroshot.prompts import build_texts


def subset_accuracy_rows(correct: np.ndarray, subsets: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    for name, mask in subsets.items():
        n, k = int(mask.sum()), int(correct[mask].sum())
        lo, hi = wilson_interval(k, n)
        rows.append({"subset": name, "n": n, "n_correct": k, "top1": k / n if n else float("nan"),
                     "ci95_low": lo, "ci95_high": hi})
    return rows


def stratified_difference(correct: np.ndarray, y: np.ndarray, overlap: np.ndarray, n_boot: int, seed: int):
    """クラス内で (overlap の正解率 - clean の正解率) を求め、両方を含むクラスで平均する。

    クラス構成の偏りを除くため。95% 区間はクラス内でクリップを復元抽出するブートストラップ。
    戻り値: (差, 下限, 上限, 比較に使ったクラス数)
    """
    rng = np.random.default_rng(seed)
    classes = [k for k in np.unique(y) if (overlap & (y == k)).any() and (~overlap & (y == k)).any()]
    groups = [(correct[overlap & (y == k)], correct[~overlap & (y == k)]) for k in classes]
    diff = float(np.mean([a.mean() - b.mean() for a, b in groups]))
    boots = [np.mean([rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for a, b in groups])
             for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return diff, float(lo), float(hi), len(classes)


def run(config_path: str | Path) -> pd.DataFrame:
    exp = load_yaml(config_path)
    ds = load_yaml(exp["dataset_config"])
    out_dir = Path(exp["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = load_meta(ds["meta_csv"], ds["audio_dir"], None)
    classes = meta.drop_duplicates("target").sort_values("target")["category"].tolist()
    y = meta["target"].to_numpy()

    flags = {name: overlap_mask(meta["src_file"], load_freesound_ids(path))
             for name, path in exp["overlap_sources"].items()}
    meta.assign(**{f"in_{k}": v for k, v in flags.items()}).to_csv(out_dir / "clip_overlap.csv", index=False)
    full = flags[exp["contamination_source"]]
    fold1 = meta["fold"].to_numpy() == 1
    subsets = {
        "all": np.ones(len(meta), dtype=bool),
        "overlap": full,
        "clean": ~full,
        "fold1_all": fold1,
        "fold1_clean": fold1 & ~full,
    }

    rows, strat = [], []
    for model, zs_path in exp["models"].items():
        zs = load_yaml(zs_path)
        dim = load_yaml(zs["model_config"])["full_dim"]
        cache = Path(zs["cache_dir"])
        audio = truncate_and_normalize(np.load(cache / f"audio_{_key(meta['filename'].tolist())}.npy"), dim)
        for prefix_name, tmpl_name in [tuple(c) for c in exp["conditions"][model]]:
            texts = build_texts(classes, zs["templates"][tmpl_name], zs["prefixes"][prefix_name], zs["natural_labels"])
            text = truncate_and_normalize(np.load(cache / f"text_{_key(texts)}.npy"), dim)
            correct = (audio @ text.T).argmax(axis=1) == y
            for r in subset_accuracy_rows(correct, subsets):
                rows.append({"model": model, "prefix": prefix_name, "template": tmpl_name, "dim": dim, **r})
            diff, lo, hi, n_cls = stratified_difference(correct, y, full, exp["bootstrap"]["n"], exp["bootstrap"]["seed"])
            strat.append({"model": model, "prefix": prefix_name, "template": tmpl_name,
                          "overlap_minus_clean": diff, "ci95_low": lo, "ci95_high": hi, "n_classes": n_cls})

    pd.DataFrame(strat).to_csv(out_dir / "stratified_difference.csv", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "summary.csv", index=False)
    return df
