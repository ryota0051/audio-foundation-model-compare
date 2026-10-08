"""ESC-50 / DCASE の混同行列を作図する（キャッシュ済み埋め込みのみ使用、GPU 不要）。

出力（configs/analysis/confusion.yaml の output_dir）:
- esc50_cm_<条件>.png / .csv : 50x50 混同行列（行正規化 = クラス別再現率）
- esc50_category_cm.png / .csv : 5 大分類に集約した混同行列
- esc50_per_class_recall.png / .csv : クラス別再現率の比較
- dcase_confusion.png / dcase_binary_report.csv : 閾値判定の 2x2 混同行列（機種 x ドメイン）
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_rgb

from egaudio.anomaly.scorers import KnnCosineScorer
from egaudio.config.loader import load_yaml
from egaudio.data.dcase import list_clips
from egaudio.data.esc50 import load_meta
from egaudio.embedding.mrl import truncate_and_normalize
from egaudio.evaluation.confusion import (
    binary_report,
    confusion_matrix,
    group_confusion,
    row_normalize,
    threshold_from_normal,
)
from egaudio.experiments.esc50_zeroshot import _key
from egaudio.retrieval.knn import knn_predict
from egaudio.zeroshot.prompts import build_texts

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/analysis/confusion.yaml")
args = parser.parse_args()
cfg = load_yaml(args.config)
style = cfg["style"]
out = Path(cfg["output_dir"])
out.mkdir(parents=True, exist_ok=True)

CMAP = LinearSegmentedColormap.from_list("seq", style["sequential"])
plt.rcParams.update({
    "figure.facecolor": style["surface"], "axes.facecolor": style["surface"], "savefig.facecolor": style["surface"],
    "text.color": style["text_primary"], "axes.labelcolor": style["text_secondary"],
    "xtick.color": style["muted"], "ytick.color": style["muted"], "axes.edgecolor": style["grid"],
    "font.size": 9,
})


def ink_for(value: float) -> str:
    """セル背景の明るさに応じて文字色を切り替える。"""
    r, g, b = to_rgb(CMAP(value))
    return style["surface"] if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.5 else style["text_primary"]


# ---------------- ESC-50 ----------------
e = cfg["esc50"]
ds = load_yaml(e["dataset_config"])
meta = load_meta(ds["meta_csv"], ds["audio_dir"], None)
y, folds = meta["target"].to_numpy(), meta["fold"].to_numpy()
classes = meta.drop_duplicates("target").sort_values("target")["category"].tolist()
n = len(classes)
groups = np.arange(n) // (n // len(e["category_groups"]))

preds: dict[str, np.ndarray] = {}

zs = load_yaml(e["zeroshot_config"])
best = json.loads((Path(zs["output_dir"]) / "best.json").read_text(encoding="utf-8"))
texts = build_texts(classes, zs["templates"][best["template"]], zs["prefixes"][best["prefix"]], zs["natural_labels"])
zs_cache = Path(zs["cache_dir"])
a = truncate_and_normalize(np.load(zs_cache / f"audio_{_key(meta['filename'].tolist())}.npy"), 768)
t = truncate_and_normalize(np.load(zs_cache / f"text_{_key(texts)}.npy"), 768)
preds["EmbeddingGemma2 zero-shot"] = (a @ t.T).argmax(axis=1)

rt = load_yaml(e["retrieval_config"])
for label in e["knn_models"]:
    model_cfg_path = Path(rt["models"][label])
    emb = np.load(Path(rt["cache_dir"]) / model_cfg_path.stem / f"audio_n{len(meta)}.npy")
    emb = truncate_and_normalize(emb, load_yaml(model_cfg_path)["full_dim"])
    p = np.empty_like(y)
    for f in np.unique(folds):
        te = folds == f
        p[te] = knn_predict(emb[~te], y[~te], emb[te], e["knn_k"])
    preds[f"{label} k-NN@{e['knn_k']}"] = p

boundaries = np.arange(1, len(e["category_groups"])) * (n // len(e["category_groups"])) - 0.5
for name, p in preds.items():
    fname = name.replace(" ", "_").replace("+", "plus").replace("@", "")
    cm = confusion_matrix(y, p, n)
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(out / f"esc50_cm_{fname}.csv")
    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(row_normalize(cm), cmap=CMAP, vmin=0, vmax=1)
    for b in boundaries:
        ax.axhline(b, color=style["muted"], linewidth=0.8)
        ax.axvline(b, color=style["muted"], linewidth=0.8)
    ax.set_xticks(range(n), classes, rotation=90, fontsize=7)
    ax.set_yticks(range(n), classes, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"ESC-50  {name}   accuracy {np.mean(p == y):.1%}", loc="left", fontsize=12)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    cb.set_label("Share of true-class clips (row-normalized)")
    cb.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(out / f"esc50_cm_{fname}.png", dpi=130)
    plt.close(fig)

# 大分類の混同行列（small multiples）
names = list(preds)
G = len(e["category_groups"])
fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 4.6))
rows = []
for ax, name in zip(axes, names):
    g = row_normalize(group_confusion(confusion_matrix(y, preds[name], n), groups, G))
    ax.imshow(g, cmap=CMAP, vmin=0, vmax=1)
    for i in range(G):
        for j in range(G):
            ax.text(j, i, f"{g[i, j]:.0%}", ha="center", va="center", fontsize=8, color=ink_for(g[i, j]))
            rows.append({"condition": name, "true": e["category_groups"][i], "pred": e["category_groups"][j], "share": g[i, j]})
    ax.set_xticks(range(G), e["category_groups"], rotation=45, ha="right")
    ax.set_yticks(range(G), e["category_groups"] if ax is axes[0] else [""] * G)
    ax.set_title(name, fontsize=10, loc="left")
    ax.set_xlabel("Predicted category")
axes[0].set_ylabel("True category")
fig.suptitle("ESC-50 confusion by major category (row-normalized)", x=0.01, ha="left", fontsize=12)
fig.tight_layout()
fig.savefig(out / "esc50_category_cm.png", dpi=130)
plt.close(fig)
pd.DataFrame(rows).to_csv(out / "esc50_category_cm.csv", index=False)

# クラス別再現率の比較（EmbeddingGemma2 ゼロショット / 同 k-NN / BEATs iter3 k-NN）
compare = names[:3]
recall = pd.DataFrame({nm: np.diag(row_normalize(confusion_matrix(y, preds[nm], n))) for nm in compare}, index=classes)
recall = recall.sort_values(compare[0])
recall.to_csv(out / "esc50_per_class_recall.csv")
fig, ax = plt.subplots(figsize=(8, 12))
# 同じ値で重ならないよう系列ごとに縦方向へ少しずらす
for off, nm, color, marker in zip([-0.25, 0.0, 0.25], compare, style["series"], ["o", "s", "D"]):
    ax.scatter(recall[nm], np.arange(n) + off, color=color, marker=marker, s=30, label=nm, zorder=3,
               edgecolors=style["surface"], linewidths=1.2)
ax.set_yticks(range(n), recall.index, fontsize=7)
ax.set_ylim(n - 0.5, -0.5)
ax.set_xlim(-0.03, 1.03)
ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.grid(axis="x", color=style["grid"], linewidth=0.6)
ax.set_axisbelow(True)
for s in ["top", "right", "left"]:
    ax.spines[s].set_visible(False)
ax.set_xlabel("Per-class recall")
ax.set_title("ESC-50 per-class recall (sorted by zero-shot)", loc="left", fontsize=12, pad=30)
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(out / "esc50_per_class_recall.png", dpi=130)
plt.close(fig)

# ---------------- DCASE ----------------
d = cfg["dcase"]
dds = load_yaml(d["dataset_config"])
reports = []
models = list(d["models"])
machines = dds["machines"]
fig, axes = plt.subplots(len(machines), 2 * len(models), figsize=(2.3 * 2 * len(models), 2.2 * len(machines)))
for r, m in enumerate(machines):
    test = list_clips(dds["root"], m, "test")
    y_t, dom = test["is_anomaly"].to_numpy(), test["domain"].to_numpy()
    for mi, label in enumerate(models):
        cache = Path(d["models"][label]) / m
        tr = truncate_and_normalize(np.load(cache / "train_whole.npy"), d["dim"])
        te = truncate_and_normalize(np.load(cache / "test_whole.npy"), d["dim"])
        scorer = KnnCosineScorer(d["knn_k"]).fit(tr)
        thr = threshold_from_normal(scorer.score_loo(), d["threshold_quantile"])
        s = scorer.score(te)
        for di, domain in enumerate(["source", "target"]):
            sel = dom == domain
            rep = binary_report(y_t[sel], s[sel], thr)
            reports.append({"model": label, "machine": m, "domain": domain, "threshold": thr, **rep})
            mat = np.array([[rep["tn"], rep["fp"]], [rep["fn"], rep["tp"]]])
            share = row_normalize(mat)
            ax = axes[r, 2 * mi + di]
            ax.imshow(share, cmap=CMAP, vmin=0, vmax=1)
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f"{mat[i, j]}", ha="center", va="center", fontsize=10, color=ink_for(share[i, j]))
            ax.set_xticks([0, 1], ["normal", "anomaly"] if r == len(machines) - 1 else ["", ""])
            ax.set_yticks([0, 1], ["normal", "anomaly"] if 2 * mi + di == 0 else ["", ""])
            if r == 0:
                ax.set_title(f"{label}\n{domain}", fontsize=9)
            if 2 * mi + di == 0:
                ax.set_ylabel(f"{m}\ntrue", fontsize=9)
            ax.set_xlabel(f"F1 {rep['f1']:.2f}", fontsize=8)
fig.suptitle(
    "DCASE thresholded decisions (rows = true, columns = predicted)\n"
    f"threshold: {d['threshold_quantile']:.0%} quantile of train-normal LOO scores · whole / k-NN k={d['knn_k']} / dim {d['dim']}",
    x=0.01, ha="left", fontsize=10,
)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(out / "dcase_confusion.png", dpi=130)
plt.close(fig)
rep_df = pd.DataFrame(reports)
rep_df.to_csv(out / "dcase_binary_report.csv", index=False)
print(rep_df.groupby(["model", "domain"])[["precision", "recall", "f1"]].mean().round(3).to_string())
print(pd.DataFrame(rows).query("true == pred").pivot(index="true", columns="condition", values="share").round(3).to_string())
