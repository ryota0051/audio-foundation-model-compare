from pathlib import Path

import pytest

from egaudio.config.loader import load_yaml


def test_load_yaml_returns_dict(tmp_path: Path):
    p = tmp_path / "a.yaml"
    p.write_text("x: 1\ny:\n  z: null\n", encoding="utf-8")
    assert load_yaml(p) == {"x": 1, "y": {"z": None}}


def test_load_yaml_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_yaml(tmp_path / "none.yaml")


def test_repo_configs_are_loadable():
    root = Path(__file__).resolve().parents[2] / "configs"
    files = list(root.rglob("*.yaml"))
    assert files
    for f in files:
        assert isinstance(load_yaml(f), dict)
