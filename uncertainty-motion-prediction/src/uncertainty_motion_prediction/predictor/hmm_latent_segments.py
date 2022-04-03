import pickle
from typing import List, Sequence
from pathlib import Path

import numpy as np
from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.silhouette import silhouette
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer

from .abstract import TrajPredictor
from .hmm_multinomial import HMMMultinomialFirstOrder


# Get scaling matrix to scale all trajectory points
def _get_canonical_scaling(dir):
    norm = np.linalg.norm(dir)
    scaling = np.array([[1.0 / norm, 0.0], [0.0, 1.0 / norm]])
    return scaling


def _get_inverse_canonical_scaling(dir):
    norm = np.linalg.norm(dir)
    scaling = np.array([[norm, 0.0], [0.0, norm]])
    return scaling


# Returns the rotation matrix that transforms a unit direction vector
# to the canonical orientation aligned with the x-axis, i.e. [x y] --> [1 0],
# where |[x y]| == 1.
def _get_canonical_rotation(dir):
    x, y = dir / np.linalg.norm(dir)
    return np.array([[x, y], [-y, x]])


def normalise_segment(
    segment: np.ndarray, segment_length: int, estimate_vel: bool = False
):
    assert len(segment) == segment_length

    # Translate start of the trajectory to origin
    translated = segment[:, 0:2] - segment[0, 0:2]

    # Compute the initial heading and magnitude of velocity. Either compute this
    # by taking the difference of the first two trajectory points, or use the
    # velocity in the data directly
    dir = np.array([0, 0])
    if estimate_vel:
        i = 0
        while np.linalg.norm(dir) < 1e-2 and i != len(segment) - 1:
            dir = translated[i + 1] - translated[i]
    else:
        dir = segment[0, 2:4]

    if np.linalg.norm(dir) > 1e-2:
        S = _get_canonical_scaling(dir)
        R = _get_canonical_rotation(dir)
        canonical = np.dot(R, np.dot(S, translated.T))
    else:
        canonical = np.array(translated.T)
    return canonical


def denormalize_segment(
    canonical: np.ndarray,
    segment_length,
    init_vel_vector: Sequence[float],
    init_position: Sequence[float],
):
    """Denormalize a segment.

    Args:
        canonical: The normalized segment. An Nx2 array.
        segment_length: Length of the segment.
        init_vel_vector: A 2-element sequence. The initial velocity [vx, vy].
        init_position: A 2-element sequence. The initial position [x, y].

    Returns:
        An Nx2 array.
    """
    canonical = np.array(canonical[:, 0:2])
    assert len(canonical) == segment_length
    S = _get_inverse_canonical_scaling(init_vel_vector)
    R = _get_canonical_rotation(init_vel_vector)
    segment = np.dot(R.T, np.dot(S, canonical.T)).T
    segment = segment + np.array(init_position).reshape([1, 2])
    return segment


def normalise_segment_batch(
    trajs, segment_length, estimate_vel: bool = False
) -> np.ndarray:
    normalised = []
    num_segments = trajs.shape[1] // segment_length
    for traj in trajs:
        for i in range(num_segments):
            segment = traj[i * segment_length : (i + 1) * segment_length, :]
            normalised.append(
                normalise_segment(segment, segment_length, estimate_vel).flatten()
            )

    normalised = np.array(normalised)

    return normalised


class KMeansOutcome:
    """A class to perform clustering prediction.

    Args:
        segment_len: number of steps in an observation segment
        centers: An Nx2T array. T is the number of steps. Each row is a
            flattened trajectory.
    """

    def __init__(self, segment_length: int, centers: np.ndarray):
        self._seg_len = segment_length
        self._centers = np.array(centers)

    @property
    def N(self):
        """Number of trajectory kinds."""
        return len(self._centers)

    @property
    def segment_length(self):
        return self._seg_len

    def classify(self, traj: np.ndarray) -> int:
        """Classify which cluster the trajectory belongs to.

        Args:
            traj: An Tx5 or Tx4 array representing a single trajectory
        """
        traj = np.array(traj)[:, 0:4]
        normalized_traj = normalise_segment(traj, self._seg_len)
        diff = self._centers - normalized_traj.reshape([1, -1])
        diff_norm = np.linalg.norm(diff, axis=1)
        return np.argmin(diff_norm.flatten())

    def classify_batch(self, trajs: np.ndarray) -> List[int]:
        ret = []
        for t in trajs:
            ret.append(self.classify(t))
        return ret

    def generate(self, traj: np.ndarray, cluster_idx: int) -> np.ndarray:
        """Generate the trajectory given the history trajectory.

        Returns:
        """
        pass

    def save_to_file(self, file_path):
        p = Path(file_path).expanduser().resolve()
        with open(p, "wb") as _file:
            pickle.dump(self, _file)

    @staticmethod
    def load_from_file(file_path):
        p = Path(file_path).expanduser().resolve()
        with open(p, "rb") as _file:
            return pickle.load(_file)


class HMMLatentSegmentsExtractor:
    def __init__(
        self,
        dt: float = 0.4,
        segment_len: int = 7,
        estimate_velocity: bool = False,
        n_min_centres: int = 3,
        n_max_centres: int = 10,
    ):
        if segment_len < 3:
            raise Exception("Segment must have at least 3 points")

        self._seglen = segment_len
        self._dt = dt
        self._estimate_vel = estimate_velocity

        self._n_min_centres = n_min_centres
        self._n_max_centres = n_max_centres

    def learn_latent_segments_xmeans(self, trajs):
        normalised = normalise_segment_batch(trajs, self._seglen, self._estimate_vel)

        # Prepare initial centers - amount of initial centers defines amount of clusters from which X-Means will
        # start analysis.
        initial_centres = kmeans_plusplus_initializer(
            normalised, self._n_min_centres
        ).initialize()

        print("Using the seed centers:")
        print(initial_centres)

        # Create instance of X-Means algorithm. The algorithm will analyse all possible numbers of cluster centers
        # in the range [min_centers, max_centres], and use BIC to determine the optimal no. of clusters
        clustering = xmeans(
            normalised, initial_centres, self._n_max_centres, criterion=0
        )
        clustering.process()

        # Extract clustering results: clusters and their centers
        clusters = clustering.get_clusters()
        centres = clustering.get_centers()
        return clusters, centres

    def learn_latent_segments_manual_kmeans(self, trajs):
        """Learn clusters for all numbers of clusters."""
        normalised = normalise_segment_batch(trajs, self._seglen, self._estimate_vel)

        tested_wce = []
        tested_silhouette = []
        tested_clusters = []
        tested_centres = []
        for num_centres in range(self._n_min_centres, self._n_max_centres + 1):
            # Prepare initial centers which K-means will use as a seed
            initial_centres = kmeans_plusplus_initializer(
                normalised, num_centres
            ).initialize()

            # Run clustering
            clustering = kmeans(normalised, initial_centres, ccore=True)
            clustering.process()

            # Compute the silhouette score
            clusters = clustering.get_clusters()
            silhouettes = silhouette(normalised, clusters).process().get_score()
            silhouettes = np.array(silhouettes)
            silhouettes[np.isnan(silhouettes)] = 0.0  # get nan if a(i) = b(i)
            silhouette_score = np.mean(silhouettes)

            # Save informations
            tested_wce.append(clustering.get_total_wce())
            tested_silhouette.append(silhouette_score)
            tested_clusters.append(clusters)
            tested_centres.append(clustering.get_centers())

        # Return all the information and clusters for visualisation and manual cluster selection
        return tested_wce, tested_silhouette, tested_clusters, tested_centres
