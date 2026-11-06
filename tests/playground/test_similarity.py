import numpy as np
import pytest

from egaudio.playground.similarity import (
    apply_prefix,
    parse_texts,
    prepare_mic_audio,
    rank_similarities,
)


# ---- parse_texts ----

def test_parse_texts_splits_lines_and_drops_blank():
    assert parse_texts("おはよう\n\n  こんにちは  \n") == ["おはよう", "こんにちは"]


def test_parse_texts_keeps_duplicates_in_order():
    assert parse_texts("a\nb\na") == ["a", "b", "a"]


def test_parse_texts_empty():
    assert parse_texts("") == []
    assert parse_texts(None) == []


# ---- apply_prefix ----

def test_apply_prefix_formats_text():
    assert apply_prefix("犬", "task: search result | query: {text}") == "task: search result | query: 犬"


def test_apply_prefix_none_template_is_identity():
    assert apply_prefix("犬", "{text}") == "犬"


def test_apply_prefix_requires_placeholder():
    with pytest.raises(ValueError):
        apply_prefix("犬", "no placeholder")


# ---- prepare_mic_audio ----

def test_prepare_int16_scaled_to_float():
    x = np.array([0, 16384, -32768], dtype=np.int16)
    out = prepare_mic_audio((16000, x), target_sr=16000, max_sec=10.0)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, [0.0, 0.5, -1.0])


def test_prepare_int32_scaled_to_float():
    x = np.array([0, 2**30], dtype=np.int32)
    out = prepare_mic_audio((16000, x), target_sr=16000, max_sec=10.0)
    np.testing.assert_allclose(out, [0.0, 0.5])


def test_prepare_stereo_is_averaged():
    x = np.stack([np.full(100, 0.2), np.full(100, 0.4)], axis=1).astype(np.float32)
    out = prepare_mic_audio((16000, x), target_sr=16000, max_sec=10.0)
    assert out.shape == (100,)
    np.testing.assert_allclose(out, 0.3, rtol=1e-6)


def test_prepare_resamples_to_target():
    x = np.zeros(48000, dtype=np.float32)
    out = prepare_mic_audio((48000, x), target_sr=16000, max_sec=10.0)
    assert out.shape == (16000,)


def test_prepare_truncates_to_max_sec():
    x = np.zeros(16000 * 5, dtype=np.float32)
    out = prepare_mic_audio((16000, x), target_sr=16000, max_sec=2.0)
    assert out.shape == (32000,)


def test_prepare_none_raises():
    with pytest.raises(ValueError):
        prepare_mic_audio(None, target_sr=16000, max_sec=10.0)


def test_prepare_empty_raises():
    with pytest.raises(ValueError):
        prepare_mic_audio((16000, np.zeros(0, dtype=np.int16)), target_sr=16000, max_sec=10.0)


# ---- rank_similarities ----

def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def test_rank_similarities_sorted_desc_with_rank():
    audio = _unit([1.0, 0.0])
    texts = ["a", "b", "c"]
    text_emb = _unit([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    rows = rank_similarities(audio, text_emb, texts)
    assert [r["text"] for r in rows] == ["b", "c", "a"]
    assert [r["rank"] for r in rows] == [1, 2, 3]
    assert rows[0]["cosine"] == pytest.approx(1.0)
    assert rows[1]["cosine"] == pytest.approx(np.sqrt(0.5))
    assert rows[2]["cosine"] == pytest.approx(0.0, abs=1e-6)


def test_rank_similarities_margin_from_mean():
    audio = _unit([1.0, 0.0])
    text_emb = _unit([[1.0, 0.0], [0.0, 1.0]])
    rows = rank_similarities(audio, text_emb, ["x", "y"])
    # 平均 0.5 からの差
    assert rows[0]["diff_from_mean"] == pytest.approx(0.5)
    assert rows[1]["diff_from_mean"] == pytest.approx(-0.5)


def test_rank_similarities_length_mismatch():
    with pytest.raises(ValueError):
        rank_similarities(_unit([1.0, 0.0]), _unit([[1.0, 0.0]]), ["a", "b"])
