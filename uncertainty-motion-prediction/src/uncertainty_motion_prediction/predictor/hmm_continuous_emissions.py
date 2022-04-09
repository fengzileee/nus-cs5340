import pickle
from typing import List, Sequence
from pathlib import Path

import numpy as np
from hmmlearn.hmm import GaussianHMM
from .abstract import TrajPredictor
from .hmm_latent_segments import (
    normalise_segment, normalise_segment_batch, denormalize_segment
)

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
        print(traj)
        normalised = normalise_segment(traj, traj.shape[0])
        print(normalised)
        traj_segments = normalised[:, 0:2].reshape(-1, self._seg_len * 2)
        Z = self._hmm.predict(traj_segments)
        current_state = Z[-1]

        # Based on the fitted latent states, we predict the next latent state
        # and take the mode of the Gaussian observation distribution as the
        # predicted MLE observation
        predicted_normalised_segments = []
        for _ in range(self._N_future_segment):
            predicted_next_z = np.argmax(self._hmm.transmat_[current_state])
            predicted_normalised_segments.append(self._hmm.means_[predicted_next_z].reshape([-1, 2]))
            current_state = predicted_next_z

        unnormalised_segments = traj[:, :, 0:4].reshape(-1, self._seg_len, 4)
        disp = unnormalised_segments[-1, -1, 0:2] - unnormalised_segments[-1, -2, 0:2]
        pos = unnormalised_segments[-1, -1, 0:2] + disp
        predicted_denormalised_segments = []
        for s in predicted_normalised_segments:
            denormalized = denormalize_segment(s, self._seg_len, disp, pos)
            disp = denormalized[-1, 0:2] - denormalized[-2, 0:2]
            pos = denormalized[-1, 0:2] + disp
            predicted_denormalised_segments.append(denormalized)

        predicted = np.array(predicted_denormalised_segments).reshape([-1, 2])
        return predicted

    def sample(self, traj: np.ndarray, count: int = 1):
        raise NotImplementedError("Not implemented!")