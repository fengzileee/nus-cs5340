"""A base class for trajectory predictor."""
import numpy as np
from abc import ABC, abstractmethod


class TrajPredictor(ABC):
    """The abstract basic class of a trajectory predictor."""

    @abstractmethod
    def predict(self, traj: np.ndarray):
        """Perform a point estimate of future traj given a past traj.

        Args:
            traj: a Tx5 array ["pos_x", "pos_y", "vel_x", "vel_y", "time"],
                T is the number of timesteps.

        Returns:
            A Nx2 array ["pos_x", "pos_y"], N is the number of future steps.
        """
        pass

    @abstractmethod
    def sample(self, traj: np.ndarray, count: int = 1):
        """Sample a future traj given a past traj.

        Sample from the conditional distribution p(future traj|past traj). For
        deterministic predictors, this is equivalent to sampling from delta
        distribution.

        Might not be necessary when the condition distribution has closed form
        expression (e.g., Normal distribution).
        """
        pass


class ConstantVelocityPredictor(TrajPredictor):
    """A simple constant velocity motion predictor.

    Args:
        N_future: number of steps to predict in the future trajectory.
        dt: Time duration per step
    """

    def __init__(self, N_future: int = 4, dt: float = 0.4):
        self._N = N_future
        self._dt = dt

    def predict(self, traj):
        p = traj[-1, 0:2]
        v = traj[-1, 2:4]
        dt = np.linspace(self._dt, self._N * self._dt, self._N)
        dtdt = np.vstack([dt, dt]).T
        dp = dtdt * v.reshape((1, 2))
        return dp + np.tile(p, (self._N, 1))

    def sample(self, traj, count: int = 1):
        # ConstantVelocity predictor is equivalent to a delta distribution
        return self.predict(traj)
