from pathlib import Path

import pytest

from egaudio.config.loader import deep_merge, load_yaml

ROOT = Path(__file__).resolve().parents[2]


def test_load_yaml_returns_dict(tmp_path: Path):
    p = tmp_path / "a.yaml"
    p.write_text("x: 1\ny:\n  z: null\n", encoding="utf-8")
    assert load_yaml(p) == {"x": 1, "y": {"z": None}}


def test_load_yaml_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_yaml(tmp_path / "none.yaml")


def test_deep_merge_nested():
    assert deep_merge({"a": {"b": 1, "c": 2}, "d": 1}, {"a": {"c": 3}, "e": 4}) == {
        "a": {"b": 1, "c": 3},
        "d": 1,
        "e": 4,
    }


def test_load_yaml_base_inheritance(tmp_path: Path):
    base = tmp_path / "base.yaml"
    base.write_text("x: 1\nnested:\n  a: 1\n  b: 2\n", encoding="utf-8")
    child = tmp_path / "child.yaml"
    child.write_text(f"base: {base.as_posix()}\nnested:\n  b: 3\n", encoding="utf-8")
    assert load_yaml(child) == {"x": 1, "nested": {"a": 1, "b": 3}}


def test_repo_configs_are_loadable(monkeypatch):
    monkeypatch.chdir(ROOT)
    files = list((ROOT / "configs").rglob("*.yaml"))
    assert files
    for f in files:
        assert isinstance(load_yaml(f), dict)
