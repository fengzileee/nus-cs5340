import numpy as np
from .abstract import TrajPredictor

from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.silhouette import silhouette
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer

from .hmm_multinomial import HMMMultinomialFirstOrder


class HMMLatentSegmentsPredictor(TrajPredictor):
    def __init__(self, 
                 N_future: int = 4,
                 dt: float = 0.4,
                 segment_len: int = 7,
                 estimate_velocity: bool = False,
                 n_min_centres: int = 3,
                 n_max_centres: int = 10
                 ):
        if segment_len < 3:
            raise Exception("Segment must have at least 3 points")

        self._seglen = segment_len
        self._N = N_future
        self._dt = dt
        self._estimate_vel = estimate_velocity

        self._n_min_centres = n_min_centres
        self._n_max_centres = n_max_centres


    def predict(self, traj):
        pass


    def sample(self, traj, count = 1):
        pass


    def learn_latent_segments_xmeans(self, trajs):
        normalised = self._normalise_segment_batch(trajs)

        # Prepare initial centers - amount of initial centers defines amount of clusters from which X-Means will
        # start analysis.
        initial_centres = kmeans_plusplus_initializer(normalised, self._n_min_centres).initialize()

        print("Using the seed centers:")
        print(initial_centres)

        # Create instance of X-Means algorithm. The algorithm will analyse all possible numbers of cluster centers
        # in the range [min_centers, max_centres], and use BIC to determine the optimal no. of clusters
        clustering = xmeans(normalised, initial_centres, self._n_max_centres, criterion=0)
        clustering.process()

        # Extract clustering results: clusters and their centers
        clusters = clustering.get_clusters()
        centres = clustering.get_centers()
        return clusters, centres


    def learn_latent_segments_manual_kmeans(self, trajs):
        normalised = self._normalise_segment_batch(trajs)

        tested_wce = []
        tested_silhouette = []
        tested_clusters = []
        tested_centres = []
        for num_centres in range(self._n_min_centres, self._n_max_centres + 1):
            # Prepare initial centers which K-means will use as a seed
            initial_centres = kmeans_plusplus_initializer(normalised, num_centres).initialize()

            # Run clustering
            clustering = kmeans(normalised, initial_centres, ccore=True)
            clustering.process()

            # Compute the silhouette score
            clusters = clustering.get_clusters()
            silhouettes = silhouette(normalised, clusters).process().get_score()
            silhouette_score = np.mean(np.array(silhouettes))

            # Save informations
            tested_wce.append(clustering.get_total_wce())
            tested_silhouette.append(silhouette_score)
            tested_clusters.append(clusters)
            tested_centres.append(clustering.get_centers())

        # Return all the information and clusters for visualisation and manual cluster selection
        return tested_wce, tested_silhouette, tested_clusters, tested_centres

    
    def _normalise_segment_batch(self, trajs):
        normalised = []
        num_segments = trajs.shape[1] // self._seglen
        for traj in trajs:
            for i in range(num_segments):
                segment = traj[i*self._seglen:(i+1)*self._seglen, :]
                normalised.append(self._normalise_segment(segment).flatten())

        normalised = np.array(normalised)

        return normalised


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