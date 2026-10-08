"""Step 1c: ESC-50 ゼロショットのモダリティギャップ・ハブ化補正の比較。

Step 1 と同じプレフィックス × テンプレート × 次元で、補正方法（egaudio.zeroshot.gap.METHODS）ごとに
Top-k 正解率と予測の偏りを測る。埋め込みは Step 1 のキャッシュを再利用（無ければ計算）。
"""

from pathlib import Path

import pandas as pd

from egaudio.audio.preprocess import load_audio
from egaudio.config.loader import load_yaml
from egaudio.data.esc50 import load_meta
from egaudio.embedding.cache import cached_array
from egaudio.embedding.mrl import truncate_and_normalize
from egaudio.experiments.esc50_zeroshot import _key
from egaudio.zeroshot.classify import topk_accuracy
from egaudio.zeroshot.gap import corrected_similarity, prediction_concentration
from egaudio.zeroshot.prompts import build_texts


def evaluate_methods(audio, text, y, folds, methods, dims, topk, params) -> list[dict]:
    rows = []
    for dim in dims:
        a, t = truncate_and_normalize(audio, dim), truncate_and_normalize(text, dim)
        for method in methods:
            sim = corrected_similarity(method, a, t, folds, params)
            row = {"dim": dim, "method": method, **{f"top{k}": topk_accuracy(sim, y, k) for k in topk}}
            row.update(prediction_concentration(sim.argmax(axis=1), t.shape[0]))
            rows.append(row)
    return rows


def run(config_path: str | Path) -> pd.DataFrame:
    exp = load_yaml(config_path)
    ds = load_yaml(exp["dataset_config"])
    out_dir, cache_dir = Path(exp["output_dir"]), Path(exp["cache_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = load_meta(ds["meta_csv"], ds["audio_dir"], exp["folds"])
    classes = meta.drop_duplicates("target").sort_values("target")["category"].tolist()
    y, folds = meta["target"].to_numpy(), meta["fold"].to_numpy()

    model_cfg = load_yaml(exp["model_config"])
    embedder = None

    def _embedder():
        nonlocal embedder
        if embedder is None:
            from egaudio.embedding.factory import build_audio_embedder

            embedder = build_audio_embedder(model_cfg)
        return embedder

    audio = cached_array(
        cache_dir / f"audio_{_key(meta['filename'].tolist())}.npy",
        lambda: _embedder().encode_audio([load_audio(p, model_cfg["sample_rate"], ds["channel"]) for p in meta["path"]]),
    )

    rows = []
    for prefix_name, prefix in exp["prefixes"].items():
        for tmpl_name, tmpl in exp["templates"].items():
            texts = build_texts(classes, tmpl, prefix, exp["natural_labels"])
            text = cached_array(cache_dir / f"text_{_key(texts)}.npy", lambda t=texts: _embedder().encode_text(t))
            for r in evaluate_methods(audio, text, y, folds, exp["gap_correction"]["methods"], exp["dims"],
                                      exp["topk"], exp["gap_correction"]):
                rows.append({"prefix": prefix_name, "template": tmpl_name, **r})

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    best = (summary.sort_values("top1", ascending=False).groupby("method", sort=False).head(1)
            .set_index("method").reindex(exp["gap_correction"]["methods"]).reset_index())
    best.to_csv(out_dir / "best_per_method.csv", index=False)
    return best
