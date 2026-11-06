import numpy as np
import pytest

from egaudio.config.loader import load_yaml
from egaudio.playground.app import compute


class FakeEmbedder:
    """テキストは文字数、音声は長さで決まる 2 次元ベクトルを返す。"""

    sample_rate = 16000

    def __init__(self):
        self.seen_texts = []

    def encode_text(self, texts):
        self.seen_texts.extend(texts)
        v = np.array([[1.0, float(len(t))] for t in texts], dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def encode_audio(self, wavs):
        return np.array([[1.0, 0.0]] * len(wavs), dtype=np.float32)


CFG = {
    "audio": {"max_sec": 1.0},
    "text_prefixes": {"none": "{text}", "search": "task: search result | query: {text}"},
}


def _audio(sec=2.0):
    return (16000, np.zeros(int(16000 * sec), dtype=np.int16))


def test_compute_returns_sorted_table_and_info():
    emb = FakeEmbedder()
    table, info = compute(emb, CFG, "あ\nあいうえお\n", "none", _audio())
    # 短いテキストほど (1,0) に近い
    assert [row[1] for row in table] == ["あ", "あいうえお"]
    assert [row[0] for row in table] == [1, 2]
    assert "1.00 秒" in info  # max_sec で打ち切り
    assert "テキスト候補: 2 件" in info


def test_compute_applies_prefix_to_texts_only():
    emb = FakeEmbedder()
    compute(emb, CFG, "犬", "search", _audio())
    assert emb.seen_texts == ["task: search result | query: 犬"]


def test_compute_requires_texts():
    with pytest.raises(ValueError):
        compute(FakeEmbedder(), CFG, "\n  \n", "none", _audio())


def test_compute_requires_audio():
    with pytest.raises(ValueError):
        compute(FakeEmbedder(), CFG, "犬", "none", None)


def test_playground_config_prefixes_have_placeholder():
    cfg = load_yaml("configs/app/playground.yaml")
    assert cfg["text_prefixes"]
    assert all("{text}" in t for t in cfg["text_prefixes"].values())
    assert cfg["audio"]["max_sec"] > 0
