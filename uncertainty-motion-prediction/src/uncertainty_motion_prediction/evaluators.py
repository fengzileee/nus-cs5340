"""A module contains utilities and classes for evaluation."""
import numpy as np

from uncertainty_motion_prediction.predictor import TrajPredictor


class DistanceErrorEvaluator:
    """An evaluator to compute FDE and ADE of point estimates.

    Args:
        N_future: number of future steps used for evaluation.
    """

    def __init__(self, N_future: int = 4):
        self._N = N_future

    def evaluate(self, predictor: TrajPredictor, trajectories: np.ndarray) -> None:
        traj_length = trajectories.shape[1]
        assert traj_length > self._N, f"Trajectories too short! {traj_length}"
        # errors [dx, dy], dx = x_prediction - x_groundtruth
        errors = np.zeros([trajectories.shape[0], self._N, 2])
        for i in range(trajectories.shape[0]):
            prediction = predictor.predict(
                trajectories[i, 0 : traj_length - self._N, :]
            )
            errors[i, :, :] = prediction - trajectories[i, -self._N :, 0:2]

        distance_errors = np.linalg.norm(errors, axis=2, ord=2)
        fde = distance_errors[:, -1]
        ade = np.mean(distance_errors, axis=1)
        return {"fde": fde, "ade": ade}
