"""マハラノビススコアの分布を可視化し、ドメイン内での分離度を確認する。

公式 AUC はドメインをまたいで異常音を混ぜるため、ドメイン間でスコアのスケールがずれると下がる。
ここではドメイン内（同じドメインの正常 vs 異常）AUC も出して切り分ける。
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from egaudio.anomaly.scorers import MahalanobisScorer
from egaudio.config.loader import load_yaml
from egaudio.data.dcase import list_clips
from egaudio.embedding.mrl import truncate_and_normalize
from egaudio.evaluation.dcase_metrics import DOMAINS, evaluate_section

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/experiment/dcase_anomaly.yaml")
parser.add_argument("--dim", type=int, default=768)
parser.add_argument("--out", default="outputs/analysis/mahalanobis_distribution")
args = parser.parse_args()

exp = load_yaml(args.config)
ds = load_yaml(exp["dataset_config"])
cache = Path(exp["cache_dir"])
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

machines = [m for m in (exp["machines"] or ds["machines"]) if (cache / m / "test_whole.npy").is_file()]
rows = []
fig, axes = plt.subplots(len(machines), 1, figsize=(8, 2.6 * len(machines)), squeeze=False)
for ax, m in zip(axes[:, 0], machines):
    train, test = list_clips(ds["root"], m, "train"), list_clips(ds["root"], m, "test")
    tr = truncate_and_normalize(np.load(cache / m / "train_whole.npy"), args.dim)
    te = truncate_and_normalize(np.load(cache / m / "test_whole.npy"), args.dim)
    s = MahalanobisScorer(exp["scorers"]["covariance"]).fit(tr).score(te)
    y = test["is_anomaly"].to_numpy()
    d = test["domain"].to_numpy()

    row = {"machine": m, **evaluate_section(y, s, test["domain"].map(DOMAINS).to_numpy(), exp["max_fpr"])}
    for dom in DOMAINS:
        mask = d == dom
        row[f"within_auc_{dom}"] = roc_auc_score(y[mask], s[mask])
    # 異常ラベルに関係なく、スコアがドメインをどれだけ当てられるか（ドメインずれの大きさ）
    row["domain_auc"] = roc_auc_score((d == "target").astype(int), s)
    rows.append(row)

    bins = np.linspace(s.min(), s.max(), 40)
    for dom, ls in [("source", "-"), ("target", "--")]:
        for lab, color in [(0, "tab:blue"), (1, "tab:red")]:
            sel = (d == dom) & (y == lab)
            ax.hist(s[sel], bins=bins, histtype="step", linestyle=ls, color=color, linewidth=1.5,
                    label=f"{dom} {'anomaly' if lab else 'normal'}")
    ax.set_title(f"{m}  (Mahalanobis, single pool, dim={args.dim})")
    ax.legend(fontsize=7)
axes[-1, 0].set_xlabel("Mahalanobis distance")
fig.tight_layout()
fig.savefig(out / f"hist_dim{args.dim}.png", dpi=120)

df = pd.DataFrame(rows)
df.to_csv(out / f"auc_dim{args.dim}.csv", index=False)
print(df.round(3).to_string(index=False))
