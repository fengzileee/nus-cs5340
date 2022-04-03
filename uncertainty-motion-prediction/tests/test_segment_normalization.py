import numpy as np

from uncertainty_motion_prediction.predictor.hmm_latent_segments import (
    denormalize_segment,
)


def test_denormalize_segment():
    canonical = np.array([[0, 0], [1, 0], [5, 0]])
    segment = denormalize_segment(canonical, 3, [0, 0.5], [2, 2])
    np.testing.assert_almost_equal(segment, np.array([[2, 2], [2, 2.5], [2, 4.5]]))
