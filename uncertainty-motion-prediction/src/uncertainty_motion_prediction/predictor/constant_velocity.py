import numpy as np
from .abstract import TrajPredictor


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
