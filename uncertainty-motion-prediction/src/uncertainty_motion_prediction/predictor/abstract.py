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

        The purpose is to visualize and to analyze the (co)variance.

        Might not be necessary when the conditionional distribution has closed
        form expression (e.g., Gaussian distribution).
        """
        pass
