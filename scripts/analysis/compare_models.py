"""DCASE 異常検知結果をモデル間・公式ベースラインと比較する。"""

import argparse
from pathlib import Path

import pandas as pd

from egaudio.config.loader import load_yaml
from egaudio.experiments.dcase_anomaly import baseline_table

KEYS = ["mode", "dim", "pooling", "scorer"]
METRICS = ["auc_source", "auc_target", "pauc"]

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/analysis/model_comparison.yaml")
args = parser.parse_args()
cfg = load_yaml(args.config)
out = Path(cfg["output_dir"])
out.mkdir(parents=True, exist_ok=True)

summary_rows, machine_rows = [], []
for label, d in cfg["models"].items():
    d = Path(d)
    if not (d / "summary.csv").is_file():
        print(f"skip {label}: {d} not found")
        continue
    summary = pd.read_csv(d / "summary.csv")
    per = pd.read_csv(d / "per_machine.csv")
    fixed = cfg["fixed_condition"]
    picks = {"best": summary.iloc[0][KEYS].to_dict(), "fixed": fixed}
    for kind, cond in picks.items():
        sel = (per[KEYS] == pd.Series(cond)[KEYS]).all(axis=1)
        srow = summary[(summary[KEYS] == pd.Series(cond)[KEYS]).all(axis=1)]
        if srow.empty:
            continue
        summary_rows.append({"model": label, "pick": kind, **cond,
                             **srow.iloc[0][[c for c in srow.columns if c.startswith("hmean") or c == "official_score"]]})
        for _, r in per[sel].iterrows():
            machine_rows.append({"model": label, "pick": kind, "machine": r["machine"], **r[METRICS]})

base = baseline_table(load_yaml(cfg["baseline_config"]), sorted({r["machine"] for r in machine_rows}))
for system in base["system"].unique():
    b = base[base["system"] == system]
    summary_rows.append({"model": f"official_{system}", "pick": "-",
                         "official_score": b[b["machine"] == "official_score"]["auc_source"].iloc[0]})
    for _, r in b[b["machine"] != "official_score"].iterrows():
        machine_rows.append({"model": f"official_{system}", "pick": "-", "machine": r["machine"], **r[METRICS]})

summary_df = pd.DataFrame(summary_rows)
machine_df = pd.DataFrame(machine_rows)
summary_df.to_csv(out / "summary.csv", index=False)
machine_df.to_csv(out / "per_machine.csv", index=False)
print(summary_df.round(3).to_string(index=False))
best = machine_df[machine_df["pick"].isin(["best", "-"])]
print(best.pivot_table(index="machine", columns="model", values=METRICS).round(3).to_string())
