import numpy as np
from copy import deepcopy
from .abstract import TrajPredictor


class ConstantVelocityKFPredictor(TrajPredictor):

    def get_q(self, var=1.0):
        dt = self._dt

        G = np.matrix([[0.5*dt**2],
               [dt]])
        Q = G*G.T

        return self._reorder_q(Q) * var

    def _reorder_q(self,Q):

        D = np.zeros((4, 4))

        Q = np.array(Q)
        for i, x in enumerate(Q.ravel()):
            f = np.eye(2) * x
            ix, iy = (i // 2) * 2, (i % 2) * 2
            D[ix:ix+2, iy:iy+2] = f

        return D
    def update_r(self, r):
        self.R = r


    def __init__(self, 
                 N_future: int = 4, 
                 dt: float = 0.4,
                 P = None, # uncertainty covariance
                 Q = None, # process uncertainty
                 R = None, # measurement/observation covariance
                 process_var: float = 0.81
                 ):
        self._N = N_future
        self._dt = dt

        # Hard-coded state and observation dimensions. We assume that the state
        # and observations are both [px, py, vx, vy]
        self.state_dim = 4
        self.obs_dim = 4

        # Set covariances
        self.P = np.eye(self.state_dim) if P is None else P
        self.Q = self.get_q(process_var) if Q is None else Q

        if R is None:
            self.is_adaptive_r = True
        else:
            self.is_adaptive_r = False
        # init with default
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

    # Returns predicted trajectory
    def predict(self, traj):
        if traj.ndim == 1:
            raise Exception("Trajectory contains only 1 dimension")

        predicted, _ = self._predict_single(traj[:, :4])
        return predicted[:, 0:2]


    def _predict_single(self, traj):
        self.state_x = traj[0, :]
        self.state_P = self.P.copy()
        future_x = []
        future_P = []

        trajlen = traj.shape[0]


        obs_v = traj[:,2:4]

        var_v = np.matrix([[np.std(obs_v[:,0])**2, 0.0],
                      [0.0, np.std(obs_v[:,1])**2]])

        if self.is_adaptive_r:
            R = np.eye(self.obs_dim)
            R[2,2] = var_v[0,0]
            R[3,3] = var_v[1,1]
            self.update_r(R)




        for i in range(1, trajlen):
            self._predict()
            self._update(traj[i, :])

        for j in range(self._N):
            self._predict()
            future_x.append(self.state_x)
            future_P.append(self.state_P)

        return np.array(future_x), np.array(future_P)


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

    # Returns point estimate of most likely future trajectory, along with the
    # uncertainty at each point, modelled as Gaussian noise
    def sample(self, traj, count: int = 1):
        if traj.ndim == 1:
            raise Exception("Trajectory contains only 1 dimension")

        self.state_x = traj[0, 0:4]
        self.state_P = self.P.copy()
        samples = []

        trajlen = traj.shape[0]

        for i in range(1, trajlen):
            self._predict()
            self._update(traj[i, 0:4])

        ############

        for _ in range(count):
            future_x = []
            current_state = deepcopy(self.state_x)
            current_cov = deepcopy(self.state_P)
            for j in range(self._N):
                mean = np.dot(self.F, current_state)
                current_cov = np.dot(np.dot(self.F, current_cov), self.F.T) + self.Q
                current_state = np.random.multivariate_normal(mean, current_cov)
                future_x.append(current_state)
            samples.append(np.array(future_x)[:, 0:2])

        return np.array(samples)
