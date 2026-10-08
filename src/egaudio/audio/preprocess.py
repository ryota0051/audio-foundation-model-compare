"""音声の読み込み・チャンネル選択・リサンプル。"""

from pathlib import Path

import numpy as np
import soundfile as sf
import soxr


def select_channel(x: np.ndarray, channel: int | None) -> np.ndarray:
    """(samples, channels) からチャンネルを選ぶ。channel=None なら平均でモノラル化。"""
    if x.ndim == 1:
        return x
    if channel is None:
        return x.mean(axis=1)
    if not 0 <= channel < x.shape[1]:
        raise ValueError(f"channel {channel} out of range for {x.shape[1]} channels")
    return x[:, channel]


def resample(x: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if orig_sr == target_sr:
        return x
    return soxr.resample(x, orig_sr, target_sr).astype(np.float32)


def load_audio(path: str | Path, target_sr: int, channel: int | None) -> np.ndarray:
    """モノラル float32、target_sr の波形を返す。"""
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return resample(select_channel(x, channel), sr, target_sr)
