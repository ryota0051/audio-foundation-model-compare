import numpy as np

from egaudio.embedding.clap import chunk_audio, mean_of_chunks


def test_chunk_audio_short_is_single():
    x = np.arange(5 * 10, dtype=np.float32)  # sr=10, 5 s
    chunks = chunk_audio(x, sr=10, max_seconds=10)
    assert len(chunks) == 1
    np.testing.assert_array_equal(chunks[0], x)


def test_chunk_audio_long_is_deterministic_and_covers_end():
    x = np.arange(12 * 10, dtype=np.float32)  # 12 s
    chunks = chunk_audio(x, sr=10, max_seconds=10)
    assert len(chunks) == 2
    np.testing.assert_array_equal(chunks[0], x[:100])
    np.testing.assert_array_equal(chunks[1], x[-100:])
    assert all(len(c) == 100 for c in chunks)


def test_mean_of_chunks_groups_and_normalizes():
    emb = np.array([[1.0, 0.0], [0.0, 1.0], [3.0, 0.0]])
    owner = np.array([0, 0, 1])
    out = mean_of_chunks(emb, owner, n=2)
    np.testing.assert_allclose(out[0], np.array([1.0, 1.0]) / np.sqrt(2), rtol=1e-6)
    np.testing.assert_allclose(out[1], [1.0, 0.0])
