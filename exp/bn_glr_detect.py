"""Detector wrapper for the bounded-nuisance GLR on split (steady, excursion) evidence."""
import numpy as np
from filters import bn_glr
from filters.common import T0

B0_GRID = tuple(np.round(np.geomspace(1.0, 120.0, 14), 4))
B1_GRID = (-1e9,) + tuple(np.round(np.geomspace(0.5, 60.0, 9), 4))  # -1e9: no nuisance condition (CUSUM-nested)


class BNGLRDetector:
    name = 'BN-GLR'

    def __init__(self, t0=T0, d_grid=(24, 36), drifts=(0.0, 1.0), b0_grid=B0_GRID, b1_grid=B1_GRID):
        self.t0, self.d_grid, self.drifts = t0, tuple(d_grid), tuple(drifts)
        self.b0_grid, self.b1_grid = tuple(b0_grid), tuple(b1_grid)

    def fit_kappa(self, *args):
        # A common evidence scale is absorbed by (b0, b1, drift): the statistic is linear in it.
        pass

    def points(self, e, c0):
        for d in self.d_grid:
            for drift in self.drifts:
                for b0 in self.b0_grid:
                    for b1 in self.b1_grid:
                        params = dict(d_max=d, drift=drift, b0=float(b0), b1=float(b1))
                        yield (params, *self.run(e, c0, params))

    def run(self, e, c0, params):
        return bn_glr.run(e[0], e[1], c0, params['b0'], params['b1'], params['d_max'], params['drift'], self.t0)
