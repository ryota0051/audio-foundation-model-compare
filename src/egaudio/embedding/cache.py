"""埋め込みの .npy キャッシュ。"""

from collections.abc import Callable
from pathlib import Path

import numpy as np


def cached_array(path: str | Path, fn: Callable[[], np.ndarray]) -> np.ndarray:
    path = Path(path)
    if path.is_file():
        return np.load(path)
    arr = np.asarray(fn())
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    return arr
