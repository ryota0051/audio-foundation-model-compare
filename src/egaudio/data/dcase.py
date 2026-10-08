"""DCASE Task 2 のファイル一覧とファイル名パース。"""

import re
from pathlib import Path

import pandas as pd

_PATTERN = re.compile(r"^section_(\d+)_(source|target)_(train|test)_(normal|anomaly)_")


def parse_filename(name: str) -> dict:
    m = _PATTERN.match(name)
    if m is None:
        raise ValueError(f"unexpected DCASE filename: {name}")
    section, domain, split, label = m.groups()
    return {
        "section": section,
        "domain": domain,
        "split": split,
        "label": label,
        "is_anomaly": int(label == "anomaly"),
    }


def list_clips(root: str | Path, machine: str, split: str) -> pd.DataFrame:
    d = Path(root) / machine / split
    rows = [{"path": str(p), "filename": p.name, **parse_filename(p.name)} for p in sorted(d.glob("*.wav"))]
    return pd.DataFrame(rows)
