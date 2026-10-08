"""埋め込みの UMAP 可視化（キャッシュ済み埋め込みのみ使用、GPU 不要）。

出力（configs/analysis/umap.yaml の output_dir、各 PNG に座標 CSV）:
- esc50_umap_categories.png : モデル x 大分類の small multiples（該当大分類のみ着色、他は灰色）
- esc50_umap_audio_text.png : EmbeddingGemma 2 の音声 2,000 本とクラステキスト 50 本を同一 UMAP に
- dcase_umap.png            : 機種 x モデルで source 正常 / target 正常 / 異常
UMAP は大域的な距離を保たないため、クラスタの有無・重なりの定性的な確認にのみ使う。
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap

from egaudio.analysis.loaders import dcase_domain_labels, load_dcase_whole, load_esc50_retrieval_embeddings
from egaudio.config.loader import load_yaml
from egaudio.data.dcase import list_clips
from egaudio.data.esc50 import label_text, load_meta
from egaudio.embedding.mrl import truncate_and_normalize
from egaudio.experiments.esc50_zeroshot import _key
from egaudio.zeroshot.prompts import build_texts

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/analysis/umap.yaml")
args = parser.parse_args()
cfg = load_yaml(args.config)
style = cfg["style"]
out = Path(cfg["output_dir"])
out.mkdir(parents=True, exist_ok=True)
CONTEXT = "#c3c2b7"  # 背景（非着目）点の色: 地に溶けすぎない中間グレー

plt.rcParams.update({
    "figure.facecolor": style["surface"], "axes.facecolor": style["surface"], "savefig.facecolor": style["surface"],
    "text.color": style["text_primary"], "axes.labelcolor": style["text_secondary"],
    "axes.edgecolor": style["grid"], "font.size": 9,
})


def embed_2d(x: np.ndarray) -> np.ndarray:
    return umap.UMAP(**cfg["umap"]).fit_transform(x)


def clean(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(style["grid"])


# ---------------- ESC-50: 大分類 small multiples ----------------
e = cfg["esc50"]
ds = load_yaml(e["dataset_config"])
meta = load_meta(ds["meta_csv"], ds["audio_dir"], None)
y = meta["target"].to_numpy()
classes = meta.drop_duplicates("target").sort_values("target")["category"].tolist()
G = len(e["category_groups"])
group_of = y // (len(classes) // G)
rt = load_yaml(e["retrieval_config"])

models = cfg["esc50_umap"]["models"]
coords = {}
for label in models:
    coords[label] = embed_2d(load_esc50_retrieval_embeddings(rt, label, len(meta)))
    pd.DataFrame({"x": coords[label][:, 0], "y": coords[label][:, 1], "category": meta["category"],
                  "group": [e["category_groups"][g] for g in group_of]}).to_csv(
        out / f"esc50_umap_{label.replace('+', 'plus')}.csv", index=False)

fig, axes = plt.subplots(len(models), G, figsize=(2.9 * G, 3.0 * len(models)))
for r, label in enumerate(models):
    xy = coords[label]
    for g in range(G):
        ax = axes[r, g]
        sel = group_of == g
        ax.scatter(xy[~sel, 0], xy[~sel, 1], s=3, color=CONTEXT, alpha=0.5, linewidths=0)
        ax.scatter(xy[sel, 0], xy[sel, 1], s=5, color=style["series"][0], linewidths=0)
        clean(ax)
        if r == 0:
            ax.set_title(e["category_groups"][g], fontsize=10)
        if g == 0:
            ax.set_ylabel(label, fontsize=10)
fig.suptitle("ESC-50 audio embeddings (UMAP, cosine). Blue = clips of the column's major category, gray = all others",
             x=0.01, ha="left", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(out / "esc50_umap_categories.png", dpi=130)
plt.close(fig)

# ---------------- ESC-50: 音声とテキストを同一空間に（EmbeddingGemma 2） ----------------
zs = load_yaml(e["zeroshot_config"])
best = json.loads((Path(zs["output_dir"]) / "best.json").read_text(encoding="utf-8"))
texts = build_texts(classes, zs["templates"][best["template"]], zs["prefixes"][best["prefix"]], zs["natural_labels"])
zc = Path(zs["cache_dir"])
a = truncate_and_normalize(np.load(zc / f"audio_{_key(meta['filename'].tolist())}.npy"), 768)
t = truncate_and_normalize(np.load(zc / f"text_{_key(texts)}.npy"), 768)
xy = embed_2d(np.vstack([a, t]))
xa, xt = xy[: len(a)], xy[len(a):]
pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "modality": ["audio"] * len(a) + ["text"] * len(t),
              "label": list(meta["category"]) + classes}).to_csv(out / "esc50_umap_audio_text.csv", index=False)

# 数値の裏付け: 音声-音声 / テキスト-テキスト / 音声-正解テキスト の平均コサイン類似度
cos = {
    "audio-audio (random pairs)": float(np.mean(np.sum(a * a[np.random.default_rng(0).permutation(len(a))], axis=1))),
    "text-text (all pairs)": float((t @ t.T)[~np.eye(len(t), dtype=bool)].mean()),
    "audio-text (correct class)": float(np.mean(np.sum(a * t[y], axis=1))),
    "audio-text (all pairs)": float((a @ t.T).mean()),
}
pd.Series(cos).to_csv(out / "esc50_modality_cosine.csv", header=["mean_cosine"])

fig, ax = plt.subplots(figsize=(8, 7))
ax.scatter(xa[:, 0], xa[:, 1], s=4, color=style["series"][0], alpha=0.6, linewidths=0, label="audio clip (2,000)")
ax.scatter(xt[:, 0], xt[:, 1], s=28, marker="D", color=style["series"][1], edgecolors=style["surface"],
           linewidths=1.0, label="class text (50)")
# テキスト 50 点は 1 か所に固まり個別ラベルが重なるため、点群に 1 つだけ注記し、予測集中の上位は表で示す
pred_count = np.bincount((a @ t.T).argmax(axis=1), minlength=len(classes))
ax.annotate("all 50 class texts", xt.mean(axis=0), xytext=(10, -14), textcoords="offset points",
            fontsize=8, color=style["text_secondary"])
clean(ax)
ax.legend(loc="upper left", frameon=False)
ax.set_title("EmbeddingGemma 2: audio and class-text embeddings in one UMAP (modality gap)", loc="left", fontsize=11)
top = "\n".join(f"  {label_text(classes[i])}: {pred_count[i]}" for i in np.argsort(-pred_count)[:6])
ax.text(0.01, 0.01, "most-predicted texts in zero-shot\n(clips of 2,000; 40 per class expected)\n" + top,
        transform=ax.transAxes, ha="left", va="bottom", fontsize=8, color=style["text_secondary"])
txt = "\n".join(f"{k}: {v:.3f}" for k, v in cos.items())
ax.text(0.99, 0.01, "mean cosine similarity (768-d)\n" + txt, transform=ax.transAxes, ha="right", va="bottom",
        fontsize=8, color=style["text_secondary"])
fig.tight_layout()
fig.savefig(out / "esc50_umap_audio_text.png", dpi=130)
plt.close(fig)

# ---------------- DCASE ----------------
du = cfg["dcase_umap"]
dds = load_yaml(cfg["dcase"]["dataset_config"])
machines = dds["machines"]
order = [("source normal", style["series"][0], "o", 4), ("target normal", style["series"][1], "s", 14),
         ("anomaly", style["series"][2], "^", 16)]
fig, axes = plt.subplots(len(machines), len(du["models"]), figsize=(4.0 * len(du["models"]), 3.4 * len(machines)))
rows = []
for r, m in enumerate(machines):
    clips = pd.concat([list_clips(dds["root"], m, "train"), list_clips(dds["root"], m, "test")], ignore_index=True)
    lab = dcase_domain_labels(clips)
    for c, (label, cache) in enumerate(du["models"].items()):
        tr, te = load_dcase_whole(cache, m, du["dim"])
        xy = embed_2d(np.vstack([tr, te]))
        rows.append(pd.DataFrame({"machine": m, "model": label, "x": xy[:, 0], "y": xy[:, 1], "kind": lab,
                                  "split": clips["split"]}))
        ax = axes[r, c]
        for kind, color, marker, size in order:
            sel = lab == kind
            ax.scatter(xy[sel, 0], xy[sel, 1], s=size, color=color, marker=marker, linewidths=0, alpha=0.8,
                       label=f"{kind} ({sel.sum()})")
        clean(ax)
        if r == 0:
            ax.set_title(label, fontsize=10)
        if c == 0:
            ax.set_ylabel(m, fontsize=10)
        if r == 0 and c == len(du["models"]) - 1:
            ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.32), frameon=False, fontsize=8, markerscale=1.5)
pd.concat(rows).to_csv(out / "dcase_umap.csv", index=False)
fig.suptitle("DCASE whole-clip embeddings (UMAP, cosine): train + test, per machine", x=0.01, ha="left", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.975))
fig.savefig(out / "dcase_umap.png", dpi=130)
plt.close(fig)
print(pd.Series(cos).round(3).to_string())
