"""Semantic Progress Function solver.

Default hyper-parameters match the paper:
    k=30  (neighbourhood window), sigma=20 (Gaussian weighting bandwidth),
    lambda=1e-5 (Tikhonov regularisation), angular distance metric, p=1.
"""

from typing import List, Tuple

import numpy as np
from scipy.sparse import coo_matrix


def _gaussian_weight(i: float, j: float, sigma: float) -> float:
    # Paper Eq.: w_ij = exp( -(i-j)^2 / (2 sigma^2) )
    return np.exp(-((i - j) ** 2) / (2.0 * sigma ** 2))


def pairwise_angular_distance(
    embeddings: np.ndarray,
    k: int = 30,
    p: float = 1.0,
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """Compute angular distances between nearby frame embeddings.

    Args:
        embeddings: ``(N, D)`` normalised embeddings.
        k: Neighbourhood window — each frame is compared to its next *k* neighbours.
        p: Distance power.  ``d_ij^p`` modulates contrast (default 1; use 2
           for sharper segmentation).

    Returns:
        pairs: List of ``(i, j)`` index pairs.
        distances: Angular distance (raised to power *p*) for each pair.
    """
    pairs: List[Tuple[int, int]] = []
    distances: List[float] = []
    n = len(embeddings)

    for i in range(n):
        for j in range(i + 1, min(n, i + k + 1)):
            pairs.append((i, j))
            cos_sim = float(np.dot(embeddings[i], embeddings[j]))
            d = np.arccos(np.clip(cos_sim, -1.0 + 1e-7, 1.0 - 1e-7))
            distances.append(d ** p)

    return pairs, np.array(distances)


def _build_system(
    pairs: List[Tuple[int, int]],
    distances: np.ndarray,
    sigma: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the weighted linear system  W A x = W b.

    Note on sign convention: the paper writes A with +1 at i and -1 at j,
    encoding S_i - S_j ~ d_ij.  Because pairs satisfy i < j and d_ij > 0,
    that convention produces a *decreasing* raw solution.  We flip the signs
    (-1 at i, +1 at j) so the raw solution is *increasing*, which is what the
    monotone-normalize post-processing expects (it clips negative gradients).
    The final normalised SPF is identical either way.
    """
    rows, cols, vals = [], [], []
    weights = []

    for row, (i, j) in enumerate(pairs):
        rows.extend([row, row])
        cols.extend([i, j])
        vals.extend([-1, 1])
        weights.append(_gaussian_weight(i, j, sigma))

    n = max(max(i, j) for i, j in pairs) + 1
    P = coo_matrix((vals, (rows, cols)), shape=(len(distances), n)).toarray()
    return P, distances, np.array(weights)


class SolveDirect:
    """Compute the Semantic Potential Function via weighted least squares.

    The potential maps each frame index to a perceptual-progress value in [0, 1].
    Frames where "more happens" visually have steeper potential; static segments
    are flat.

    Args:
        k: Neighbourhood window for pairwise distances.
        sigma: Gaussian weighting bandwidth.
        lmd: Tikhonov regularisation strength.
        p: Distance power (``d^p``).  Default 1; use 2 for sharper segmentation.
        normalize: If ``True``, enforce monotonicity and scale to [0, 1].
    """

    def __init__(
        self,
        k: int = 30,
        sigma: float = 20.0,
        lmd: float = 1e-5,
        p: float = 1.0,
        normalize: bool = True,
    ):
        self.k = k
        self.sigma = sigma
        self.lmd = lmd
        self.p = p
        self.normalize = normalize

    def __call__(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute potential from frame embeddings.

        Args:
            embeddings: ``(N, D)`` array of normalised frame embeddings.

        Returns:
            Potential curve of shape ``(N,)`` with values in [0, 1].
        """
        n = len(embeddings)
        if n < 2:
            return np.zeros(n)

        pairs, distances = pairwise_angular_distance(embeddings, k=self.k, p=self.p)
        P, y, w = _build_system(pairs, distances, self.sigma)

        # Pw = diag(w) @ P via row-broadcasting (avoids M×M diagonal allocation)
        Pw = P * w[:, None]
        I = np.eye(P.shape[1])
        x = np.linalg.solve(P.T @ Pw + self.lmd * I, Pw.T @ y)

        if self.normalize:
            x = self._monotone_normalize(x)
        return x

    @staticmethod
    def _monotone_normalize(x: np.ndarray) -> np.ndarray:
        """Clip negative gradients, cumsum, and scale to [0, 1]."""
        grad = np.gradient(x)
        grad = np.clip(grad, 0.0, None)
        x = np.cumsum(grad)
        x -= x.min()
        if x.max() > 0:
            x /= x.max()
        return x


def invert_potential(
    potential: np.ndarray,
    num_frames: int = None,
) -> np.ndarray:
    """Invert the potential to get warped temporal positions.

    Given ``potential`` mapping frame_index -> progress, returns the original
    frame indices corresponding to uniformly-spaced progress values.  This is
    the basis for temporal re-parameterisation (ReTime).

    Paper Eq.: tau_k = S^{-1}( k / (T-1) )

    Args:
        potential: ``(N,)`` normalised potential in [0, 1].
        num_frames: Number of output frames (default: same as input).

    Returns:
        Warped frame indices of shape ``(num_frames,)``.
    """
    potential = np.asarray(potential).flatten()
    n = len(potential)
    if num_frames is None:
        num_frames = n

    p_min, p_max = potential.min(), potential.max()
    if p_max - p_min < 1e-8:
        return np.linspace(0, n - 1, num_frames)

    p_norm = (potential - p_min) / (p_max - p_min)
    target = np.linspace(0, 1, num_frames)
    return np.interp(target, p_norm, np.arange(n, dtype=np.float64))
