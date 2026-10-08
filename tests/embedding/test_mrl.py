import numpy as np
import pytest

from egaudio.embedding.mrl import truncate_and_normalize


def test_truncate_shape_and_unit_norm():
    x = np.random.default_rng(0).standard_normal((4, 768)).astype(np.float32)
    y = truncate_and_normalize(x, 256)
    assert y.shape == (4, 256)
    np.testing.assert_allclose(np.linalg.norm(y, axis=1), 1.0, rtol=1e-5)
    # 方向は先頭 256 次元と一致
    ref = x[:, :256] / np.linalg.norm(x[:, :256], axis=1, keepdims=True)
    np.testing.assert_allclose(y, ref, rtol=1e-5)


def test_truncate_full_dim_still_normalizes():
    x = np.full((2, 4), 2.0, dtype=np.float32)
    np.testing.assert_allclose(np.linalg.norm(truncate_and_normalize(x, 4), axis=1), 1.0)


def test_truncate_too_large_dim_raises():
    with pytest.raises(ValueError):
        truncate_and_normalize(np.zeros((1, 4)), 8)
