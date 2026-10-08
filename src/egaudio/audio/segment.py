"""固定長窓による分割。"""

import numpy as np


def segment(x: np.ndarray, sr: int, window_sec: float, hop_sec: float) -> list[np.ndarray]:
    """窓長 window_sec・ホップ hop_sec で分割。末尾が余る場合は終端に揃えた窓を追加する。"""
    win = int(round(window_sec * sr))
    hop = int(round(hop_sec * sr))
    if len(x) <= win:
        return [x]
    starts = list(range(0, len(x) - win + 1, hop))
    if starts[-1] + win < len(x):
        starts.append(len(x) - win)
    return [x[s : s + win] for s in starts]
