"""The visualization module."""
import numpy as np

from .predictor import TrajPredictor


class SamplingVisualizer:
    def __init__(self, N_future: int = 4, sample_size: int = 100):
        self._N = N_future
        self._sample_size = sample_size

    def visualize_to_axis(self, axis, predictor: TrajPredictor, trajectory: np.ndarray):
        """Sample predictions and plot to aixs.

        Args:
            axis: matplotlib axis object.
            predictor: A trajectory predictor.
            trajectory: A Nx5 numpy array. Each row: [x, y, vx, vy, t]
        """
        traj_length = trajectory.shape[0]
        point_est = predictor.predict(trajectory[0 : traj_length - self._N, :])
        samples = predictor.sample(
            trajectory[0 : traj_length - self._N, :], count=self._sample_size
        )
        for s in samples:
            (h_samples,) = axis.plot(
                s[:, 0], s[:, 1], linewidth=0.5, alpha=0.5, c=[0.5, 0.5, 0.5]
            )
        (h_data,) = axis.plot(trajectory[:, 0], trajectory[:, 1], linewidth=2)
        (h_pred,) = axis.plot(point_est[:, 0], point_est[:, 1], linewidth=2)
        axis.legend(
            handles=[h_data, h_pred, h_samples],
            labels=["Groundtruth", "Point Estimate", "Samples"],
        )
