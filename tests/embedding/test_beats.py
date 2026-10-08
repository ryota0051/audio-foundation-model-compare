import numpy as np
import pytest

from egaudio.embedding.beats import group_by_length, resolve_layer
from egaudio.embedding.factory import build_audio_embedder


def test_group_by_length_same_length_batches():
    wavs = [np.zeros(3), np.zeros(5), np.zeros(3), np.zeros(3), np.zeros(5)]
    groups = group_by_length(wavs, batch_size=2)
    assert sorted(i for g in groups for i in g) == [0, 1, 2, 3, 4]
    for g in groups:
        assert len(g) <= 2
        assert len({len(wavs[i]) for i in g}) == 1


def test_resolve_layer():
    assert resolve_layer(None, 12) == 11
    assert resolve_layer(12, 12) == 11
    assert resolve_layer(1, 12) == 0
    with pytest.raises(ValueError):
        resolve_layer(0, 12)
    with pytest.raises(ValueError):
        resolve_layer(13, 12)


def test_factory_unknown_backend():
    with pytest.raises(ValueError):
        build_audio_embedder({"backend": "foo"})
