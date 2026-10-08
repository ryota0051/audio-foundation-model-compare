import numpy as np
import pandas as pd
import pytest

from egaudio.data.overlap import load_freesound_ids, overlap_mask, wilson_interval


def test_load_freesound_ids(tmp_path):
    p = tmp_path / "meta.csv"
    pd.DataFrame({"audio_filename": ["a", "b", "c"], "freesound_id": [10, 20, 10], "x": [1, 2, 3]}).to_csv(p, index=False)
    assert load_freesound_ids(p) == {"10", "20"}


def test_overlap_mask_compares_as_strings():
    src = pd.Series([10, 30, 20])
    np.testing.assert_array_equal(overlap_mask(src, {"10", "20"}), [True, False, True])


def test_wilson_interval():
    lo, hi = wilson_interval(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3) and hi == pytest.approx(0.5962, abs=1e-3)
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0 and 0.0 < hi < 0.35
    assert wilson_interval(0, 0) == (0.0, 0.0)
