import pickle
from typing import List, Sequence
from pathlib import Path

import numpy as np
from hmmlearn.hmm import GaussianHMM
from .abstract import TrajPredictor
#from .hmm_latent_segments import (
#    normalise_segment, normalise_segment_batch, denormalize_segment
#)

# Get scaling matrix to scale all trajectory points
def _get_canonical_scaling(dir):
    norm = np.linalg.norm(dir)
    scaling = np.array([[1.0 / norm, 0.0], [0.0, 1.0 / norm]])
    return scaling


def _get_inverse_canonical_scaling(dir):
    norm = np.linalg.norm(dir)
    scaling = np.array([[norm, 0.0], [0.0, norm]])
    return scaling


# Returns the rotation matrix that transforms a unit direction vector
# to the canonical orientation aligned with the x-axis, i.e. [x y] --> [1 0],
# where |[x y]| == 1.
def _get_canonical_rotation(dir):
    if np.linalg.norm(dir) > 1e-3:
        x, y = dir / np.linalg.norm(dir)
        return np.array([[x, y], [-y, x]]) #np.identity(2)
    else:
        return np.eye(2)


def get_segment_length(segment):
    length_segment = 0
    for i in range(len(segment) - 1):
        length_segment += np.linalg.norm(segment[i] - segment[i + 1])
    return length_segment


def normalise_segment(
    segment: np.ndarray, segment_length: int, estimate_vel: bool = False
):
    segment = np.array(segment)
    assert len(segment) == segment_length

    # Translate start of the trajectory to origin
    translated = segment[:, 0:2] - segment[0, 0:2]

    # Compute the initial heading and magnitude of velocity. Either compute this
    # by taking the difference of the first two trajectory points, or use the
    # velocity in the data directly
    dir = np.array([0, 0])
    if estimate_vel:
        i = 0
        while np.linalg.norm(dir) < 1e-2 and i != len(segment) - 1:
            dir = translated[i + 1] - translated[0]
            i += 1
    else:
        dir = segment[0, 2:4]

    length_segment = get_segment_length(segment)
    if length_segment < 1e-3:
        length_segment = 1
    # S = _get_canonical_scaling(dir)
    S = np.array([[1.0 / length_segment, 0.0], [0.0, 1.0 / length_segment]])
    R = _get_canonical_rotation(dir)
    canonical = np.dot(R, np.dot(S, translated.T))
    return canonical


def denormalize_segment(
    canonical: np.ndarray,
    segment_length,
    init_vel_vector: Sequence[float],
    init_position: Sequence[float],
    scale: float,
):
    """Denormalize a segment.

    Args:
        canonical: The normalized segment. An Nx2 array.
        segment_length: Length of the segment.
        init_vel_vector: A 2-element sequence. The initial velocity [vx, vy].
        init_position: A 2-element sequence. The initial position [x, y].

    Returns:
        An Nx2 array.
    """
    canonical = np.array(canonical[:, 0:2])
    assert len(canonical) == segment_length
    S = np.array([[scale, 0.0], [0.0, scale]])
    R = _get_canonical_rotation(init_vel_vector)
    segment = np.dot(R.T, np.dot(S, canonical.T)).T
    segment = segment + np.array(init_position).reshape([1, 2])
    return segment


def normalise_segment_batch(
    trajs, segment_length, estimate_vel: bool = False
) -> np.ndarray:
    normalised = []
    num_segments = trajs.shape[1] // segment_length
    for traj in trajs:
        for i in range(num_segments):
            segment = traj[i * segment_length : (i + 1) * segment_length, :]
            normalised.append(
                normalise_segment(segment, segment_length, estimate_vel).flatten()
            )

    normalised = np.array(normalised)

    return normalised



class HMMContinuousEmissionsPredictor(TrajPredictor):
    def __init__(
        self,
        hmm: GaussianHMM,
        N_future_segment: int = 1,
        dt: float = 0.4,
        seg_len: int = 4
    ):
        self._hmm: GaussianHMM = hmm
        self._N_future_segment = N_future_segment
        self._dt = dt
        self._seg_len: int = seg_len

    def predict(self, traj: np.ndarray):
        assert(traj.shape[0] % self._seg_len == 0)
        traj_segments = traj[:, 0:4].reshape(-1, self._seg_len, 4)
        normalised = []

        for segment in traj_segments:
            normalised.append(normalise_segment(segment, len(segment)).flatten())
        normalised = np.array(normalised)
        
        Z = self._hmm.predict(normalised)
        current_state = Z[-1]

        # Based on the fitted latent states, we predict the next latent state
        # and take the mode of the Gaussian observation distribution as the
        # predicted MLE observation
        predicted_normalised_segments = []
        for _ in range(self._N_future_segment):
            predicted_next_z = np.argmax(self._hmm.transmat_[current_state])
            predicted_normalised_segments.append(self._hmm.means_[predicted_next_z].reshape([2, -1]).T)
            current_state = predicted_next_z

        disp = traj_segments[-1, -1, 0:2] - traj_segments[-1, -2, 0:2]
        pos = traj_segments[-1, -1, 0:2] + disp
        scale = get_segment_length(traj_segments[-1])

        predicted_denormalised_segments = []
        for s in predicted_normalised_segments:
            denormalized = denormalize_segment(s, self._seg_len, disp, pos, scale)
            disp = denormalized[-1, 0:2] - denormalized[-2, 0:2]
            pos = denormalized[-1, 0:2] + disp
            predicted_denormalised_segments.append(denormalized)

        predicted = np.array(predicted_denormalised_segments).reshape([-1, 2])
        return predicted

    def sample(self, traj: np.ndarray, count: int = 1):
        raise NotImplementedError("Not implemented!")