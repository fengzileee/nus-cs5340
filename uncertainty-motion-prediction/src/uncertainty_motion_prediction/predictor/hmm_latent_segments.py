import numpy as np
# from .abstract import TrajPredictor

from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer

# class HMMLatentSegmentsPredictor(TrajPredictor):
class HMMLatentSegmentsPredictor():
    def __init__(self, 
                 N_future: int = 4,
                 dt: float = 0.4,
                 segment_len: int = 7,
                 estimate_velocity: bool = False
                 ):
        if segment_len < 3:
            raise Exception("Segment must have at least 3 points")

        self._seglen = segment_len
        self._N = N_future
        self._dt = dt
        self._estimate_vel = estimate_velocity


    def predict(self, traj):
        pass


    def sample(self, traj, count = 1):
        pass


    def learn_latent_segments(self, trajs):
        normalised = []
        num_segments = trajs.shape[1] // self._seglen
        for traj in trajs:
            for i in range(num_segments):
                segment = traj[i*self._seglen:(i+1)*self._seglen, :]
                normalised.append(self._normalise_segment(segment).flatten())

        print("Got ", len(normalised), " segments")
        normalised = np.array(normalised)

        for n in normalised:
            print(np.array(n).reshape(2, 3).T)
        print("====")

        # Prepare initial centers - amount of initial centers defines amount of clusters from which X-Means will
        # start analysis.
        min_centres = 3
        max_centres = 12
        initial_centers= kmeans_plusplus_initializer(normalised, min_centres).initialize()

        # Create instance of X-Means algorithm. The algorithm will analyse all possible numbers of cluster centers
        # in the range [min_centers, max_centres], and use BIC to determine the optimal no. of clusters
        clustering = xmeans(normalised, initial_centers, max_centres, criterion=0)
        clustering.process()

        # Extract clustering results: clusters and their centers
        clusters = clustering.get_clusters()
        centres = clustering.get_centers()
        return clusters, centres
            

    def _normalise_segment(self, segment):
        assert(len(segment) == self._seglen)

        # Translate start of the trajectory to origin
        translated = segment[:, 0:2] - segment[0, 0:2]

        # Compute the initial heading and magnitude of velocity. Either compute this
        # by taking the difference of the first two trajectory points, or use the
        # velocity in the data directly
        dir = translated[1] - translated[0] if self._estimate_vel else segment[0, 2:4]
        S = self._get_canonical_scaling(dir)
        R = self._get_canonical_rotation(dir)
        canonical = np.dot(R, np.dot(S, translated.T))

        return canonical

        
    # Get scaling matrix to scale all trajectory points
    def _get_canonical_scaling(self, dir):
        norm = np.linalg.norm(dir)
        scaling = np.array([[1.0 / norm, 0.0],
                            [0.0, 1.0 / norm]])
        return scaling


    # Returns the rotation matrix that transforms a unit direction vector
    # to the canonical orientation aligned with the x-axis, i.e. [x y] --> [1 0],
    # where |[x y]| == 1.
    def _get_canonical_rotation(self, dir):
        x, y = dir / np.linalg.norm(dir)
        return np.array([[x, y], [-y, x]])


if __name__ == "__main__":
    seg_type1 = [[0, 0, 1, 0], [1, 0, 5, 0], [6, 0, 5, 0]]
    seg_type2 = [[0, 0, 1, 0], [1, 0, 1, 0], [2, 0, 1, 0]]
    seg_type3 = [[0, 0, 1, 0], [1, 0, 1, 1], [2, 1, 1, 1]]
    seg_type4 = [[0, 0, 1, 0], [1, 0, 1, -1], [2, -1, 1, -1]]
    seg_type5 = [[0, 0, 1, 0], [1, 0, -2, 0], [-1, 0, -2, 0]]

    seg_types = np.array([seg_type1, seg_type2, seg_type3, seg_type4, seg_type5])
    samples = []

    for seg_type in seg_types:
        print(seg_type)
        for i in range(3):
            angle = np.random.uniform(-2*np.pi, 2*np.pi)
            scale = np.random.uniform(0.5, 1.5)
            trans = np.array([[np.random.uniform(-10, 10), np.random.uniform(-10, 10)]]).T

            R = np.array([[np.cos(angle), -np.sin(angle)],
                          [np.sin(angle), np.cos(angle)]])
            S = np.array([[scale, 0], 
                          [0, scale]])
            
            seg = np.dot(S, np.dot(R, seg_type[:, 0:2].T))
            seg = seg + trans
            seg_vels = np.dot(S, np.dot(R, seg_type[:, 2:].T))
            sample = np.hstack((seg.T, seg_vels.T))
            samples.append(sample)

    pred = HMMLatentSegmentsPredictor(segment_len=3, estimate_velocity=True)
    clusters, centres = pred.learn_latent_segments(np.array(samples))

    print("Got ", len(clusters), " clusters")
    for cluster in clusters:
        print(cluster)

    print(">>>>>>>>>\nCentres")
    for centre in centres:
        print(np.array(centre).reshape(2, 3).T)