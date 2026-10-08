import numpy as np

from egaudio.audio.segment import segment


def test_segment_counts_and_covers_end():
    x = np.arange(10 * 4, dtype=np.float32)  # sr=4, 10s
    segs = segment(x, sr=4, window_sec=2.0, hop_sec=1.0)
    assert len(segs) == 9
    assert all(s.shape == (8,) for s in segs)
    np.testing.assert_array_equal(segs[-1], x[-8:])


def test_segment_adds_tail_window_when_not_aligned():
    x = np.arange(int(10.5 * 4), dtype=np.float32)  # 42 samples
    segs = segment(x, sr=4, window_sec=2.0, hop_sec=1.0)
    np.testing.assert_array_equal(segs[-1], x[-8:])
    assert len(segs) == 10


def test_segment_short_input_returns_whole():
    x = np.arange(5, dtype=np.float32)
    segs = segment(x, sr=4, window_sec=2.0, hop_sec=1.0)
    assert len(segs) == 1
    np.testing.assert_array_equal(segs[0], x)
