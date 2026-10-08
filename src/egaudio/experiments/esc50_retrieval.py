"""Step 1b: ESC-50 音声⇔音声検索による埋め込み比較（5 分割交差検証）。

テストに使う fold をクエリ、残り 4 fold を検索対象にして k-NN 分類・Precision@k・線形プローブを評価する。
テキストを介さないので、テキスト非対応の BEATs とも同条件で比較できる。
"""

from pathlib import Path

import numpy as np
import pandas as pd

from egaudio.audio.preprocess import load_audio
from egaudio.config.loader import load_yaml
from egaudio.data.esc50 import load_meta
from egaudio.embedding.cache import cached_array
from egaudio.embedding.mrl import truncate_and_normalize
from egaudio.retrieval.knn import knn_predict, linear_probe_accuracy, precision_at_k


def evaluate_folds(emb: np.ndarray, y: np.ndarray, folds: np.ndarray, cfg: dict) -> list[dict]:
    rows = []
    lp = cfg["linear_probe"]
    for f in np.unique(folds):
        te, tr = folds == f, folds != f
        db, db_y, q, q_y = emb[tr], y[tr], emb[te], y[te]
        for k in cfg["knn_k"]:
            acc = float((knn_predict(db, db_y, q, k) == q_y).mean())
            rows.append({"fold": int(f), "metric": f"knn_acc@{k}", "value": acc})
        for k in cfg["precision_k"]:
            rows.append({"fold": int(f), "metric": f"precision@{k}", "value": precision_at_k(db, db_y, q, q_y, k)})
        if lp["enabled"]:
            acc = linear_probe_accuracy(db, db_y, q, q_y, c=lp["c"], max_iter=lp["max_iter"])
            rows.append({"fold": int(f), "metric": "linear_probe_acc", "value": acc})
    return rows


def run(config_path: str | Path) -> pd.DataFrame:
    exp = load_yaml(config_path)
    ds = load_yaml(exp["dataset_config"])
    out_dir, cache_dir = Path(exp["output_dir"]), Path(exp["cache_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = load_meta(ds["meta_csv"], ds["audio_dir"], exp["folds"])
    y, folds = meta["target"].to_numpy(), meta["fold"].to_numpy()

    from egaudio.embedding.factory import build_audio_embedder

    rows = []
    for label, model_config in exp["models"].items():
        model_cfg = load_yaml(model_config)

        def _embed(model_cfg=model_cfg):
            wavs = [load_audio(p, ds["target_sr"], ds["channel"]) for p in meta["path"]]
            return build_audio_embedder(model_cfg).encode_audio(wavs)

        raw = cached_array(cache_dir / Path(model_config).stem / f"audio_n{len(meta)}.npy", _embed)
        emb = truncate_and_normalize(raw, model_cfg["full_dim"])
        rows += [{"model": label, **r} for r in evaluate_folds(emb, y, folds, exp)]
        pd.DataFrame(rows).to_csv(out_dir / "per_fold.csv", index=False)

    per_fold = pd.DataFrame(rows)
    summary = (
        per_fold.groupby(["model", "metric"], sort=False)["value"].agg(["mean", "std"]).reset_index()
        .pivot(index="model", columns="metric", values="mean")
        .reindex(list(exp["models"]))
    )
    summary.to_csv(out_dir / "summary.csv")
    return summary
