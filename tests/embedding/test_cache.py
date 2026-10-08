import numpy as np

from egaudio.embedding.cache import cached_array


def test_cached_array_computes_once(tmp_path):
    calls = []

    def fn():
        calls.append(1)
        return np.arange(3, dtype=np.float32)

    p = tmp_path / "sub" / "x.npy"
    a = cached_array(p, fn)
    b = cached_array(p, fn)
    np.testing.assert_array_equal(a, b)
    assert len(calls) == 1
