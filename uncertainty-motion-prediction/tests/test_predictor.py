import numpy as np
from uncertainty_motion_prediction.predictor import ConstantVelocityPredictor


def test_constant_velocity_predictor_predict():
    p = ConstantVelocityPredictor(N_future=3, dt=1)
    pred = p.predict(np.array([[1, 1, 1, -1, 0]]))
    np.testing.assert_almost_equal(pred, np.array([[2, 0], [3, -1], [4, -2]]))
