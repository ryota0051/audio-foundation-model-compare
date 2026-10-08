"""YAML 設定ファイルの読み込み。

`base: <path>` を書くと、そのファイルを読み込んだ上で自身の値を深くマージする（パスはリポジトリルート基準）。
辞書に `_replace: true` を書くと、その辞書はマージせず丸ごと置き換える。
"""

from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and v.get("_replace"):
            out[k] = {kk: vv for kk, vv in v.items() if kk != "_replace"}
        elif isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    base = data.pop("base", None)
    if base is None:
        return data
    return deep_merge(load_yaml(base), data)
