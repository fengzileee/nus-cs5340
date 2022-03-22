import numpy as np
# from .abstract import TrajPredictor

# class ConstantVelocityKFPredictor(TrajPredictor):
class ConstantVelocityKFPredictor():
    def __init__(self, 
                 N_future: int = 4, 
                 dt: float = 0.4,
                 P = None, # uncertainty covariance
                 Q = None, # process uncertainty
                 R = None # measurement/observation covariance
                 ):
        self._N = N_future
        self._dt = dt

        # Hard-coded state and observation dimensions. We assume that the state
        # and observations are both [px, py, vx, vy]
        self.state_dim = 4
        self.obs_dim = 4

        # Set covariances
        self.P = np.eye(self.state_dim) if P is None else P
        self.Q = np.eye(self.state_dim) if Q is None else Q
        self.R = np.eye(self.obs_dim) if R is None else R

        # Initialise placeholder state
        self.state_x = np.zeros((self.state_dim, 1))
        self.state_P = self.P.copy()

        # State transition function for constant velocity: p_t = p_(t-1) + v * dt
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float64)

        # Measurement update function for constant velocity model where state and
        # observations have the same form
        self.H = np.eye(self.state_dim)

        # Identity matrix for Kalman update
        self.id = np.eye(self.state_dim)

        # Saving the predictions of the future trajectory
        self.future_x = []
        self.future_P = []


    # Returns predicted trajectories
    def predict_batch(self, trajs):
        if trajs.ndim == 1:
            raise Exception("Trajectory contains only 1 dimension")
        elif trajs.ndim == 2:
            trajs = np.expand_dims(trajs, axis=0)
        
        predicted_trajs = []
        for traj in trajs:
            predicted, _ = self._predict_single(traj[:, :4])
            predicted_trajs.append(predicted)
        return np.array(predicted_trajs)


    # Returns point estimate of most likely future trajectory, along with the
    # uncertainty at each point, modelled as Gaussian noise
    def sample(self, traj, count: int = 1):
        if traj.ndim == 1:
            raise Exception("Trajectory contains only 1 dimension")
        elif traj.ndim == 2:
            traj = np.expand_dims(traj, axis=0)

        predicted_traj = []
        covariances = []
        for t in traj:
            predicted, covariance = self._predict_single(t[:, :4])
            predicted_traj.append(predicted)
            covariances.append(covariance)

        return np.array(predicted_traj), np.array(covariances)


    # Returns predicted trajectory
    def predict(self, traj):
        if traj.ndim == 1:
            raise Exception("Trajectory contains only 1 dimension")

        predicted, _ = self._predict_single(traj[:, :4])
        return predicted[:, 0:2]


    def _predict_single(self, traj):
        self.state_x = traj[0, :]
        self.state_P = self.P.copy()
        self.future_x = []
        self.future_P = []

        trajlen = traj.shape[0]

        for i in range(1, trajlen):
            self._predict()
            self._update(traj[i, :])

        for j in range(self._N):
            self._predict()
            self.future_x.append(self.state_x)
            self.future_P.append(self.state_P)

        return np.array(self.future_x), np.array(self.future_P)


    def _predict(self):
        self.state_x = np.dot(self.F, self.state_x)
        self.state_P = np.dot(np.dot(self.F, self.state_P), self.F.T) + self.Q
    

    def _update(self, z):
        y = z - np.dot(self.H, self.state_x)

        # Project system uncertainty into observation/measurement space
        P_HT = np.dot(self.state_P, self.H.T)
        S = np.dot(self.H, P_HT) + self.R # i.e. H * P * H' + R

        # Compute Kalman gain 
        K = np.dot(P_HT, np.linalg.inv(S))

        # Update state and covariance using Kalman gain
        self.state_x = self.state_x + np.dot(K, y)
        I_KH = self.id - np.dot(K, self.H)
        self.state_P = np.dot(np.dot(I_KH, self.state_P), I_KH.T) + np.dot(np.dot(K, self.R), K.T)