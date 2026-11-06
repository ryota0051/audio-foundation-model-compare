"""ブラウザアプリ（playground）用の入力整形と類似度ランキング。モデルに依存しない純粋関数のみ。"""

from collections.abc import Sequence

import numpy as np

from egaudio.audio.preprocess import resample, select_channel


def parse_texts(raw: str | None) -> list[str]:
    """1 行 1 テキストとして分割し、前後の空白を除いて空行を捨てる。"""
    if not raw:
        return []
    return [line.strip() for line in raw.splitlines() if line.strip()]


def apply_prefix(text: str, template: str) -> str:
    """`{text}` を含むテンプレートにテキストを埋め込む。"""
    if "{text}" not in template:
        raise ValueError(f"prefix template must contain '{{text}}': {template!r}")
    return template.replace("{text}", text)


def _to_float32(x: np.ndarray) -> np.ndarray:
    if np.issubdtype(x.dtype, np.integer):
        return (x.astype(np.float32) / float(-np.iinfo(x.dtype).min)).astype(np.float32)
    return x.astype(np.float32)


def prepare_mic_audio(audio: tuple[int, np.ndarray] | None, target_sr: int, max_sec: float) -> np.ndarray:
    """Gradio の `(sampling_rate, array)` を target_sr のモノラル float32 に変換し、max_sec で打ち切る。"""
    if audio is None:
        raise ValueError("audio is empty")
    sr, x = audio
    x = np.asarray(x)
    if x.size == 0:
        raise ValueError("audio is empty")
    x = select_channel(_to_float32(x), channel=None)
    x = resample(x, int(sr), target_sr)
    return x[: int(round(max_sec * target_sr))]


def rank_similarities(audio_emb: np.ndarray, text_emb: np.ndarray, texts: Sequence[str]) -> list[dict]:
    """正規化済み埋め込みのコサイン類似度を降順に並べる。

    モダリティギャップで絶対値が全体的に高め／低めに偏るため、候補平均との差も返す。
    """
    audio_emb = np.asarray(audio_emb, dtype=np.float32).reshape(-1)
    text_emb = np.asarray(text_emb, dtype=np.float32).reshape(len(text_emb), -1)
    if len(text_emb) != len(texts):
        raise ValueError(f"texts ({len(texts)}) and embeddings ({len(text_emb)}) length mismatch")
    sims = text_emb @ audio_emb
    mean = float(sims.mean())
    order = np.argsort(-sims, kind="stable")
    return [
        {"rank": i + 1, "text": texts[j], "cosine": float(sims[j]), "diff_from_mean": float(sims[j]) - mean}
        for i, j in enumerate(order)
    ]
