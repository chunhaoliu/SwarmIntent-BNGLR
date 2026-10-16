"""GPU-batched drop-in for exp.bn_glr_detect.BNGLRDetector (same grid order, same outputs)."""
import numpy as np
import torch
from exp.bn_glr_detect import BNGLRDetector, B0_GRID, B1_GRID
from filters import bn_glr_torch


class BNGLRDetectorTorch(BNGLRDetector):
    def __init__(self, *args, device='cuda' if torch.cuda.is_available() else 'cpu', max_cells=6e8, **kw):
        super().__init__(*args, **kw)
        self.device, self.max_cells = device, max_cells

    def points(self, e, c0):
        B, T, K = e[0].shape
        grid = [(b0, b1) for b0 in self.b0_grid for b1 in self.b1_grid]
        chunk = max(1, int(self.max_cells // (B * T * K)))
        for d in self.d_grid:
            for drift in self.drifts:
                for s in range(0, len(grid), chunk):
                    part = grid[s:s + chunk]
                    C, A = bn_glr_torch.run_batch(e[0], e[1], c0, [g[0] for g in part], [g[1] for g in part],
                                                  d, drift, self.t0, self.device)
                    for i, (b0, b1) in enumerate(part):
                        yield dict(d_max=d, drift=drift, b0=float(b0), b1=float(b1)), C[i], A[i]

    def run(self, e, c0, params):
        C, A = bn_glr_torch.run_batch(e[0], e[1], c0, [params['b0']], [params['b1']], params['d_max'],
                                      params['drift'], self.t0, self.device)
        return C[0], A[0]
