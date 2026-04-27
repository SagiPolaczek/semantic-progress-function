"""Piecewise-linear segmentation of potential curves via dynamic programming."""

from typing import Dict, List, Tuple

import numpy as np


class LineSegmentor:
    """Segment a 1-D curve into piecewise-linear pieces.

    Uses a penalty parameter to control the trade-off between fit accuracy
    and number of segments: higher penalty -> fewer (longer) segments.

    Args:
        penalty: Cost added per segment (default 0.02).
    """

    def __init__(self, penalty: float = 0.02):
        self.penalty = penalty
        self._x: np.ndarray = None
        self._y: np.ndarray = None
        self._error: np.ndarray = None
        self._coeffs: Dict[Tuple[int, int], Tuple[float, float]] = {}
        self._segments: List[Tuple[int, int]] = []

    @property
    def segments(self) -> List[Tuple[int, int]]:
        """List of ``(start_index, end_index)`` for each segment."""
        return self._segments

    @property
    def num_segments(self) -> int:
        return len(self._segments)

    def _precompute(self, y: np.ndarray):
        """Build the O(n^2) error table (called once per signal)."""
        self._y = np.asarray(y).flatten()
        self._x = np.arange(len(self._y))
        n = len(self._x)

        self._error = np.full((n, n), np.inf)
        self._coeffs = {}
        for i in range(n):
            self._error[i][i] = 0.0
            self._coeffs[(i, i)] = (0.0, float(self._y[i]))
            for j in range(i + 1, n):
                xi = self._x[i : j + 1].astype(float)
                yi = self._y[i : j + 1]
                A = np.vstack([xi, np.ones_like(xi)]).T
                m, c = np.linalg.lstsq(A, yi, rcond=None)[0]
                self._error[i][j] = float(np.sum((yi - (m * xi + c)) ** 2))
                self._coeffs[(i, j)] = (float(m), float(c))

    def _solve_dp(self):
        """Run the DP pass with the current penalty (cheap, reuses error table)."""
        n = len(self._x)
        opt = np.zeros(n)
        breaks = [-1] * n
        for j in range(n):
            opt[j] = self._error[0][j] + self.penalty
            breaks[j] = -1
            for i in range(j):
                cost = opt[i] + self._error[i + 1][j] + self.penalty
                if cost < opt[j]:
                    opt[j] = cost
                    breaks[j] = i

        self._segments = []
        j = n - 1
        while j >= 0:
            i = breaks[j]
            self._segments.append((0 if i == -1 else i + 1, j))
            j = i
        self._segments.reverse()

    def fit(self, y: np.ndarray) -> "LineSegmentor":
        """Fit piecewise-linear model to curve *y*."""
        self._precompute(y)
        self._solve_dp()
        return self

    def fit_k(self, y: np.ndarray, k: int) -> "LineSegmentor":
        """Fit exactly *k* segments via binary search on the penalty."""
        original_penalty = self.penalty

        self._precompute(y)

        self.penalty = 1e-9
        self._solve_dp()
        if self.num_segments == k:
            self.penalty = original_penalty
            return self
        if self.num_segments < k:
            # Signal too simple for k segments even at minimal penalty
            self.penalty = original_penalty
            return self

        while self.num_segments > k:
            prev = self.penalty
            self.penalty *= 10
            self._solve_dp()

        if self.num_segments == k:
            self.penalty = original_penalty
            return self

        lo, hi = prev, self.penalty
        for _ in range(100):
            if hi - lo < 1e-10 or self.num_segments == k:
                break
            mid = (lo + hi) / 2
            self.penalty = mid
            self._solve_dp()
            if self.num_segments > k:
                lo = mid
            else:
                hi = mid

        self.penalty = original_penalty
        return self

    def get_coefficients(self) -> List[Tuple[int, int, float, float]]:
        """Return ``(start, end, slope, intercept)`` for each segment."""
        return [(s, e, *self._coeffs[(s, e)]) for s, e in self._segments]
