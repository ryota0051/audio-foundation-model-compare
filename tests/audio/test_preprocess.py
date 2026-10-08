import numpy as np
import pytest
import soundfile as sf

from egaudio.audio.preprocess import load_audio, resample, select_channel


def test_select_channel_picks_index():
    x = np.stack([np.ones(10), np.zeros(10)], axis=1)
    np.testing.assert_array_equal(select_channel(x, 0), np.ones(10))
    np.testing.assert_array_equal(select_channel(x, 1), np.zeros(10))


def test_select_channel_none_averages():
    x = np.stack([np.ones(10), np.zeros(10)], axis=1)
    np.testing.assert_allclose(select_channel(x, None), np.full(10, 0.5))


def test_select_channel_mono_passthrough():
    x = np.arange(5, dtype=np.float32)
    np.testing.assert_array_equal(select_channel(x, 0), x)


def test_select_channel_out_of_range():
    with pytest.raises(ValueError):
        select_channel(np.zeros((10, 2)), 2)


def test_resample_length():
    x = np.random.default_rng(0).standard_normal(44100).astype(np.float32)
    y = resample(x, 44100, 16000)
    assert y.shape == (16000,)
    assert y.dtype == np.float32


def test_resample_identity():
    x = np.arange(100, dtype=np.float32)
    np.testing.assert_array_equal(resample(x, 16000, 16000), x)


def test_load_audio_stereo_to_ch0_and_resample(tmp_path):
    sr = 32000
    t = np.arange(sr) / sr
    stereo = np.stack([np.sin(2 * np.pi * 440 * t), np.zeros(sr)], axis=1)
    path = tmp_path / "a.wav"
    sf.write(path, stereo, sr)
    y = load_audio(path, target_sr=16000, channel=0)
    assert y.ndim == 1 and y.shape[0] == 16000 and y.dtype == np.float32
    assert np.abs(y).max() > 0.5
    y1 = load_audio(path, target_sr=16000, channel=1)
    assert np.abs(y1).max() < 1e-3
