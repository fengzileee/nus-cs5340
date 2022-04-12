import pytest
import numpy as np

from uncertainty_motion_prediction.predictor.hmm_latent_segments import (
    denormalize_segment,
    segmentize_trajectory,
    desegmentize_trajectory,
)


def test_denormalize_segment():
    canonical = np.array([[0, 0], [1, 0], [5, 0]])
    segment = denormalize_segment(canonical, 3, [0, 0.5], [2, 2], 0.5)
    np.testing.assert_almost_equal(segment, np.array([[2, 2], [2, 2.5], [2, 4.5]]))


@pytest.mark.parametrize(
    "traj_length, segment_length, shape",
    [
        (10, 4, (3, 4, 4)),
        (13, 5, (3, 5, 4)),
    ],
)
def test_segmentize_trajectory(traj_length, segment_length, shape):
    traj = np.random.random([traj_length, 4])
    segments = segmentize_trajectory(traj, segment_length)
    assert segments.shape == shape
    np.testing.assert_almost_equal(segments[0][-1], segments[1][0])


def test_desegmentize_trajectory():
    segments = [np.random.random([4, 4])]
    for i in range(1, 3):
        segments.append(np.random.random([4, 4]))
        segments[i][0, :] = segments[i - 1][-1, :]
    segments = np.array(segments)
    traj = desegmentize_trajectory(segments)
    assert traj.shape == (10, 4)
    np.testing.assert_almost_equal(traj[3, :], segments[1, 0, :])
