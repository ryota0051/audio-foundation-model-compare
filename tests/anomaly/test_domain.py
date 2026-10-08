import numpy as np

from egaudio.anomaly.domain import domain_min_scores


def test_domain_min_takes_elementwise_min():
    np.testing.assert_array_equal(domain_min_scores(np.array([1.0, 0.2]), np.array([0.5, 0.9])), [0.5, 0.2])
