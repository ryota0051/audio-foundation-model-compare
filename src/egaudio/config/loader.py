"""YAML 設定ファイルの読み込み。"""

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}
