import numpy as np
import pandas as pd
import pytest

from egaudio.analysis.loaders import (
    dcase_domain_labels,
    load_dcase_whole,
    load_esc50_retrieval_embeddings,
)


def _unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_load_esc50_retrieval_embeddings(tmp_path):
    model_cfg = tmp_path / "configs" / "m.yaml"
    model_cfg.parent.mkdir()
    model_cfg.write_text("full_dim: 4\n", encoding="utf-8")
    cache = tmp_path / "cache"
    (cache / "m").mkdir(parents=True)
    raw = np.random.default_rng(0).standard_normal((3, 4)).astype(np.float32)
    np.save(cache / "m" / "audio_n3.npy", raw)
    rt_cfg = {"cache_dir": str(cache), "models": {"M": str(model_cfg)}}
    emb = load_esc50_retrieval_embeddings(rt_cfg, "M", n_clips=3)
    np.testing.assert_allclose(emb, _unit(raw), rtol=1e-5)


def test_load_esc50_retrieval_embeddings_missing_raises(tmp_path):
    model_cfg = tmp_path / "m.yaml"
    model_cfg.write_text("full_dim: 4\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        load_esc50_retrieval_embeddings({"cache_dir": str(tmp_path), "models": {"M": str(model_cfg)}}, "M", 3)


def test_load_dcase_whole_truncates(tmp_path):
    (tmp_path / "fan").mkdir()
    np.save(tmp_path / "fan" / "train_whole.npy", np.ones((2, 8), dtype=np.float32))
    np.save(tmp_path / "fan" / "test_whole.npy", np.ones((3, 8), dtype=np.float32))
    tr, te = load_dcase_whole(tmp_path, "fan", dim=4)
    assert tr.shape == (2, 4) and te.shape == (3, 4)
    np.testing.assert_allclose(np.linalg.norm(te, axis=1), 1.0, rtol=1e-6)


def test_dcase_domain_labels():
    df = pd.DataFrame({"domain": ["source", "target", "source"], "is_anomaly": [0, 0, 1]})
    assert dcase_domain_labels(df).tolist() == ["source normal", "target normal", "anomaly"]
