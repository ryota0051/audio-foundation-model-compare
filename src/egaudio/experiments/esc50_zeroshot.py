"""Step 1: ESC-50 ゼロショット分類。"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from egaudio.audio.preprocess import load_audio
from egaudio.config.loader import load_yaml
from egaudio.data.esc50 import load_meta
from egaudio.embedding.cache import cached_array
from egaudio.embedding.mrl import truncate_and_normalize
from egaudio.zeroshot.classify import confusion_pairs, cosine_similarity, topk_accuracy
from egaudio.zeroshot.prompts import build_texts


def _key(texts: list[str]) -> str:
    return hashlib.sha1("\n".join(texts).encode("utf-8")).hexdigest()[:12]


def evaluate_condition(audio_emb, text_emb, y, folds, dims, topk) -> list[dict]:
    rows = []
    for dim in dims:
        sim = cosine_similarity(truncate_and_normalize(audio_emb, dim), truncate_and_normalize(text_emb, dim))
        row = {"dim": dim}
        for k in topk:
            row[f"top{k}"] = topk_accuracy(sim, y, k)
            row[f"top{k}_fold_mean"] = float(
                np.mean([topk_accuracy(sim[folds == f], y[folds == f], k) for f in np.unique(folds)])
            )
        rows.append(row)
    return rows


def run(config_path: str | Path) -> pd.DataFrame:
    exp = load_yaml(config_path)
    model_cfg = load_yaml(exp["model_config"])
    ds = load_yaml(exp["dataset_config"])
    out_dir, cache_dir = Path(exp["output_dir"]), Path(exp["cache_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = load_meta(ds["meta_csv"], ds["audio_dir"], exp["folds"])
    classes = meta.drop_duplicates("target").sort_values("target")["category"].tolist()
    y = meta["target"].to_numpy()
    folds = meta["fold"].to_numpy()

    from egaudio.embedding.factory import build_audio_embedder

    embedder = build_audio_embedder(model_cfg)

    def _embed_audio():
        wavs = [load_audio(p, model_cfg["sample_rate"], ds["channel"]) for p in meta["path"]]
        return embedder.encode_audio(wavs)

    audio_key = _key(meta["filename"].tolist())
    audio_emb = cached_array(cache_dir / f"audio_{audio_key}.npy", _embed_audio)

    rows, sims = [], {}
    for prefix_name, prefix in exp["prefixes"].items():
        for tmpl_name, tmpl in exp["templates"].items():
            texts = build_texts(classes, tmpl, prefix, exp["natural_labels"])
            text_emb = cached_array(cache_dir / f"text_{_key(texts)}.npy", lambda t=texts: embedder.encode_text(t))
            for r in evaluate_condition(audio_emb, text_emb, y, folds, exp["dims"], exp["topk"]):
                rows.append({"prefix": prefix_name, "template": tmpl_name, **r})
            sims[(prefix_name, tmpl_name)] = cosine_similarity(audio_emb, text_emb)

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "summary.csv", index=False)

    full = summary[summary["dim"] == model_cfg["full_dim"]]
    best = full.sort_values("top1", ascending=False).iloc[0]
    pred = sims[(best["prefix"], best["template"])].argmax(axis=1)
    confusion_pairs(y, pred, classes, exp["confusion_top_n"]).to_csv(out_dir / "confusion_pairs.csv", index=False)
    per_class = pd.DataFrame({"category": np.asarray(classes)[y], "correct": pred == y})
    per_class.groupby("category")["correct"].mean().sort_values().to_csv(out_dir / "per_class_top1.csv")
    (out_dir / "best.json").write_text(
        json.dumps({"prefix": best["prefix"], "template": best["template"], "top1": float(best["top1"])}, indent=2),
        encoding="utf-8",
    )
    return summary
