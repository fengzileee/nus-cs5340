"""The visualization module."""
import numpy as np

from .predictor import TrajPredictor

from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


class SamplingVisualizer:
    """Visualize the prediction of a trajectory to a matplotlib axis.

    Args:
        N_future: number of future steps used for evaluation.
        N_history: number of history steps. This parameter is needed when there
            is overlapping between future and history trajecotories. If None,
            use (trajectory_length - N_future).
        sampling_size: number of samples to draw.
    """
    def __init__(self, N_future: int = 4, N_history: int = None, sample_size: int = 100):
        self._N = N_future
        self._N_his = N_history
        self._sample_size = sample_size

    def confidence_ellipse(self, x, y, ax, n_std=3.0, facecolor="none", **kwargs):

        cov = np.cov(x, y)
        pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
        # Using a special case to obtain the eigenvalues of this
        # two-dimensionl dataset.
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = Ellipse(
            (0, 0),
            width=ell_radius_x * 2,
            height=ell_radius_y * 2,
            facecolor=facecolor,
            **kwargs
        )

        scale_x = np.sqrt(cov[0, 0]) * n_std
        mean_x = np.mean(x)

        # calculating the stdandard deviation of y ...
        scale_y = np.sqrt(cov[1, 1]) * n_std
        mean_y = np.mean(y)

        transf = (
            transforms.Affine2D()
            .rotate_deg(0)
            .scale(scale_x, scale_y)
            .translate(mean_x, mean_y)
        )

        ellipse.set_transform(transf + ax.transData)
        return ax.add_patch(ellipse)

    def cep(self, x, y, axis, color, alpha_scale=1.0):

        self.confidence_ellipse(
            x,
            y,
            axis,
            n_std=3,
            label=r"$3\sigma$",
            alpha=0.1 * alpha_scale,
            facecolor=color,
            edgecolor=color,
            linewidth=1,
        )
        self.confidence_ellipse(
            x,
            y,
            axis,
            n_std=2,
            label=r"$2\sigma$",
            alpha=0.2 * alpha_scale,
            facecolor=color,
            edgecolor=color,
            linewidth=1,
        )
        self.confidence_ellipse(
            x,
            y,
            axis,
            n_std=1,
            label=r"$1\sigma$",
            alpha=0.5 * alpha_scale,
            facecolor=color,
            edgecolor=color,
            linewidth=2,
        )

    def visualize_to_axis(self, axis, predictor: TrajPredictor, trajectory: np.ndarray):
        """Sample predictions and plot to aixs.

        Args:
            axis: matplotlib axis object.
            predictor: A trajectory predictor.
            trajectory: A Nx5 numpy array. Each row: [x, y, vx, vy, t]
        """
        traj_length = trajectory.shape[0]
        N_his = self._N_his if self._N_his is not None else (traj_length - self._N)
        point_est = predictor.predict(trajectory[0 : N_his, :])
        samples = predictor.sample(
            trajectory[0 : N_his, :], count=self._sample_size
        )

        for s in samples:
            (h_samples,) = axis.plot(
                s[:, 0], s[:, 1], linewidth=0.5, alpha=0.2, c=[0.5, 0.5, 0.5]
            )
        (h_data,) = axis.plot(trajectory[:, 0], trajectory[:, 1], linewidth=2)
        (h_pred,) = axis.plot(point_est[:, 0], point_est[:, 1], "-o", linewidth=2)

        # self.cep(
        #     [s[1, 0] for s in samples],
        #     [s[1, 1] for s in samples],
        #     axis,
        #     "green",
        #     alpha_scale=0.5,
        # )

        # self.cep([s[0, 0] for s in samples], [s[0, 1] for s in samples], axis, "red")

        axis.legend(
            handles=[h_data, h_pred, h_samples],
            labels=["Groundtruth", "Point Estimate", "Samples"],
        )
