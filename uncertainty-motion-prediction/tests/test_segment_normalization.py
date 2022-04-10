import pytest
import numpy as np

from uncertainty_motion_prediction.predictor.hmm_latent_segments import (
    denormalize_segment,
    segmentize_trajectory,
)


def test_denormalize_segment():
    canonical = np.array([[0, 0], [1, 0], [5, 0]])
    segment = denormalize_segment(canonical, 3, [0, 0.5], [2, 2], 0.5)
    np.testing.assert_almost_equal(segment, np.array([[2, 2], [2, 2.5], [2, 4.5]]))


@pytest.mark.parametrize(
    "traj_length, segment_length, overlap, shape",
    [
        (10, 4, 1, (3, 4, 4)),
        (13, 5, 1, (3, 5, 4)),
        (5, 3, 2, (3, 3, 4)),
    ],
)
def test_segmentize_trajectory(traj_length, segment_length, overlap, shape):
    traj = np.random.random([traj_length, 4])
    segments = segmentize_trajectory(traj, segment_length, overlap=overlap)
    assert segments.shape == shape
    np.testing.assert_almost_equal(segments[0][-overlap], segments[1][0])
