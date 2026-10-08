"""Step 2: DCASE Task 2 距離ベース異常検知。"""

from pathlib import Path

import numpy as np
import pandas as pd

from egaudio.anomaly.domain import domain_min_scores
from egaudio.anomaly.scorers import KnnCosineScorer, MahalanobisScorer
from egaudio.audio.preprocess import load_audio
from egaudio.audio.segment import segment
from egaudio.config.loader import load_yaml
from egaudio.data.dcase import list_clips
from egaudio.embedding.cache import cached_array
from egaudio.embedding.mrl import truncate_and_normalize
from egaudio.evaluation.dcase_metrics import DOMAINS, evaluate_section, hmean


def build_scorers(scorer_cfg: dict) -> dict:
    """{名前: (source/single 用 fit 関数, target 用 fit 関数)}。"""
    out = {}
    for k in scorer_cfg["knn_k"]:
        out[f"knn{k}"] = (
            lambda bank, k=k: KnnCosineScorer(k).fit(bank),
            lambda base, bank, k=k: KnnCosineScorer(k).fit(bank),
        )
    if scorer_cfg["mahalanobis"]:
        cov = scorer_cfg["covariance"]
        out["mahalanobis"] = (
            lambda bank: MahalanobisScorer(cov).fit(bank),
            MahalanobisScorer.with_shared_precision,
        )
    return out


def score_clips(train_emb, train_domain, test_emb, fit_src, fit_tgt, pooling: str) -> np.ndarray:
    if pooling == "single":
        return fit_src(train_emb).score(test_emb)
    if pooling == "domain_min":
        src = fit_src(train_emb[train_domain == "source"])
        tgt = fit_tgt(src, train_emb[train_domain == "target"])
        return domain_min_scores(src.score(test_emb), tgt.score(test_emb))
    raise ValueError(f"unknown pooling: {pooling}")


def aggregate_segments(seg_scores: np.ndarray, clip_index: np.ndarray, n_clips: int, how: str) -> np.ndarray:
    fn = {"max": np.max, "mean": np.mean}[how]
    return np.array([fn(seg_scores[clip_index == i]) for i in range(n_clips)])


class _Embeddings:
    """機種ごとのクリップ／セグメント埋め込み（キャッシュ付き）。"""

    def __init__(self, embedder, ds: dict, cache_dir: Path, seg_cfg: dict):
        self.embedder, self.ds, self.cache_dir, self.seg_cfg = embedder, ds, cache_dir, seg_cfg

    def _wavs(self, paths):
        return [load_audio(p, self.ds["target_sr"], self.ds["channel"]) for p in paths]

    def whole(self, machine: str, split: str, paths) -> np.ndarray:
        return cached_array(
            self.cache_dir / machine / f"{split}_whole.npy", lambda: self.embedder.encode_audio(self._wavs(paths))
        )

    def segments(self, machine: str, split: str, paths) -> tuple[np.ndarray, np.ndarray]:
        w, h = self.seg_cfg["window_sec"], self.seg_cfg["hop_sec"]
        tag = f"{split}_seg_w{w}_h{h}"
        sr = self.ds["target_sr"]

        def _index():
            return np.concatenate(
                [np.full(len(segment(x, sr, w, h)), i) for i, x in enumerate(self._wavs(paths))]
            )

        def _emb():
            segs = [s for x in self._wavs(paths) for s in segment(x, sr, w, h)]
            return self.embedder.encode_audio(segs)

        idx = cached_array(self.cache_dir / machine / f"{tag}_index.npy", _index)
        return cached_array(self.cache_dir / machine / f"{tag}.npy", _emb), idx


def evaluate_machine(machine, train, test, emb: _Embeddings, exp: dict) -> list[dict]:
    y = test["is_anomaly"].to_numpy()
    dom = test["domain"].map(DOMAINS).to_numpy()
    tr_dom = train["domain"].to_numpy()
    scorers = build_scorers(exp["scorers"])
    seg = exp["segmentation"]

    # (モード名, train埋め込み, train ドメイン, test埋め込み, 集約関数)
    modes = []
    if seg["whole"]["enabled"]:
        modes.append(("whole", emb.whole(machine, "train", train["path"]), tr_dom,
                      emb.whole(machine, "test", test["path"]), lambda s: s))
    if seg["segment"]["enabled"]:
        tr_e, tr_i = emb.segments(machine, "train", train["path"])
        te_e, te_i = emb.segments(machine, "test", test["path"])
        for how in seg["segment"]["aggregations"]:
            modes.append((f"segment_{how}", tr_e, tr_dom[tr_i], te_e,
                          lambda s, how=how, te_i=te_i: aggregate_segments(s, te_i, len(test), how)))

    rows = []
    for mode, tr_e, tr_d, te_e, agg in modes:
        for dim in exp["dims"]:
            tr_x, te_x = truncate_and_normalize(tr_e, dim), truncate_and_normalize(te_e, dim)
            for pooling in exp["pooling"]:
                for name, (fit_src, fit_tgt) in scorers.items():
                    s = agg(score_clips(tr_x, tr_d, te_x, fit_src, fit_tgt, pooling))
                    rows.append({"machine": machine, "mode": mode, "dim": dim, "pooling": pooling, "scorer": name,
                                 **evaluate_section(y, s, dom, exp["max_fpr"])})
    return rows


def summarize(per_machine: pd.DataFrame) -> pd.DataFrame:
    """機種×指標を公式と同様に調和平均で集約。"""
    keys = ["mode", "dim", "pooling", "scorer"]
    metrics = ["auc_source", "auc_target", "pauc"]
    rows = []
    for k, g in per_machine.groupby(keys):
        row = dict(zip(keys, k))
        for m in metrics:
            row[f"hmean_{m}"] = hmean(g[m])
        row["official_score"] = hmean(g[metrics].to_numpy().ravel())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("official_score", ascending=False).reset_index(drop=True)


def baseline_table(baseline_cfg: dict, machines: list[str]) -> pd.DataFrame:
    rows = []
    for system, per in baseline_cfg.items():
        vals = []
        for m in machines:
            v = per[m]
            rows.append({"system": system, "machine": m, **{k: v[k] / 100 for k in ("auc_source", "auc_target", "pauc")}})
            vals += [v["auc_source"] / 100, v["auc_target"] / 100, v["pauc"] / 100]
        rows.append({"system": system, "machine": "official_score", "auc_source": hmean(vals)})
    return pd.DataFrame(rows)


def run(config_path: str | Path) -> pd.DataFrame:
    exp = load_yaml(config_path)
    model_cfg = load_yaml(exp["model_config"])
    ds = load_yaml(exp["dataset_config"])
    machines = exp["machines"] or ds["machines"]
    out_dir = Path(exp["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    from egaudio.embedding.factory import build_audio_embedder

    emb = _Embeddings(build_audio_embedder(model_cfg), ds, Path(exp["cache_dir"]), exp["segmentation"]["segment"])
    rows = []
    for m in machines:
        train, test = list_clips(ds["root"], m, "train"), list_clips(ds["root"], m, "test")
        rows += evaluate_machine(m, train, test, emb, exp)
        pd.DataFrame(rows).to_csv(out_dir / "per_machine.csv", index=False)

    per_machine = pd.DataFrame(rows)
    summary = summarize(per_machine)
    summary.to_csv(out_dir / "summary.csv", index=False)

    base = baseline_table(load_yaml(exp["baseline_config"]), machines)
    base.to_csv(out_dir / "baseline.csv", index=False)
    best = summary.iloc[0]
    sel = (per_machine[["mode", "dim", "pooling", "scorer"]] == best[["mode", "dim", "pooling", "scorer"]].values).all(axis=1)
    cmp = per_machine[sel].set_index("machine")[["auc_source", "auc_target", "pauc"]].add_prefix("ours_")
    for system in base["system"].unique():
        b = base[(base["system"] == system) & (base["machine"] != "official_score")].set_index("machine")
        cmp = cmp.join(b[["auc_source", "auc_target", "pauc"]].add_prefix(f"{system}_"))
    cmp.to_csv(out_dir / "comparison_best_vs_baseline.csv")
    return summary
