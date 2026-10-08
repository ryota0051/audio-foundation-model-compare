"""音声⇔テキストのモダリティギャップ・ハブ化の補正（いずれも正解ラベル不使用）。

| method | 内容 | 評価データの使い方 |
|---|---|---|
| none | そのままのコサイン類似度 | - |
| center_text | クラステキストの平均を引いて再正規化 | 使わない |
| center_both_oof | テキスト平均 + 音声平均（評価 fold 以外から推定）を引く | 使わない |
| center_both_all | テキスト平均 + 音声平均（全クリップ）を引く | ラベルなしで使う（トランスダクティブ） |
| column_zscore | 類似度をテキスト（列）ごとに標準化 | 同上 |
| csls | Cross-domain Similarity Local Scaling（双方向の近傍平均で補正） | 同上 |
"""

import numpy as np

METHODS = ["none", "center_text", "center_both_oof", "center_both_all", "column_zscore", "csls"]


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def center(x: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """平均ベクトルを引いて L2 再正規化。"""
    return _unit(np.asarray(x, dtype=np.float64) - mean)


def column_zscore(sim: np.ndarray) -> np.ndarray:
    """列（テキスト）ごとに平均 0・標準偏差 1 にする。全音声に近いハブの優位を消す。"""
    return (sim - sim.mean(axis=0, keepdims=True)) / np.maximum(sim.std(axis=0, keepdims=True), 1e-12)


def csls(sim: np.ndarray, k: int) -> np.ndarray:
    """CSLS: 2*cos - r_audio(i) - r_text(j)。r は相手側の上位 k 近傍との類似度平均。"""
    k_a = min(k, sim.shape[1])
    k_t = min(k, sim.shape[0])
    r_audio = np.sort(sim, axis=1)[:, -k_a:].mean(axis=1, keepdims=True)
    r_text = np.sort(sim, axis=0)[-k_t:, :].mean(axis=0, keepdims=True)
    return 2 * sim - r_audio - r_text


def corrected_similarity(method: str, audio: np.ndarray, text: np.ndarray, folds: np.ndarray, params: dict) -> np.ndarray:
    """補正後の (音声 x テキスト) 類似度。audio / text は L2 正規化済みを想定。"""
    audio = np.asarray(audio, dtype=np.float64)
    text = np.asarray(text, dtype=np.float64)
    if method == "none":
        return audio @ text.T
    if method == "center_text":
        return audio @ center(text, text.mean(axis=0)).T
    if method == "center_both_all":
        return center(audio, audio.mean(axis=0)) @ center(text, text.mean(axis=0)).T
    if method == "center_both_oof":
        tc = center(text, text.mean(axis=0))
        sim = np.empty((len(audio), len(text)))
        for f in np.unique(folds):
            te = folds == f
            sim[te] = center(audio[te], audio[~te].mean(axis=0)) @ tc.T
        return sim
    if method == "column_zscore":
        return column_zscore(audio @ text.T)
    if method == "csls":
        return csls(audio @ text.T, params["csls_k"])
    raise ValueError(f"unknown method: {method}")


def prediction_concentration(pred: np.ndarray, n_classes: int) -> dict:
    """予測の偏り: 最多予測ラベルの回数と、一度も予測されないクラス数。"""
    counts = np.bincount(pred, minlength=n_classes)
    return {"max_pred_count": int(counts.max()), "n_never_predicted": int((counts == 0).sum())}
