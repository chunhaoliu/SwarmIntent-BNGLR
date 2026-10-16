import unittest
import numpy as np
import torch
from filters import bn_glr
from exp.bn_glr_detect import BNGLRDetector
from exp.bn_glr_detect_torch import BNGLRDetectorTorch


class BNGLRTorchTest(unittest.TestCase):
    def test_batched_matches_numpy(self):
        rng = np.random.default_rng(0)
        B, T, K, t0 = 40, 50, 6, 8
        es, ee = rng.normal(size=(B, T, K)), rng.normal(size=(B, T, K)) * 2
        es[:20, 25:, 3] += 1.5; ee[20:, 15:30, :] += 1.0
        c0 = es[:, :t0 + 1].sum(1).argmax(-1)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        det_t = BNGLRDetectorTorch(t0=t0, d_grid=(5, 12), drifts=(0.0, 0.7), b0_grid=(1.0, 4.0, 9.0),
                                   b1_grid=(-1e9, 0.5, 3.0), device=device, max_cells=B * T * K * 4)
        det_n = BNGLRDetector(t0=t0, d_grid=(5, 12), drifts=(0.0, 0.7), b0_grid=(1.0, 4.0, 9.0), b1_grid=(-1e9, 0.5, 3.0))
        got = list(det_t.points((es, ee), c0)); ref = list(det_n.points((es, ee), c0))
        self.assertEqual(len(got), len(ref))
        for (p1, C1, A1), (p2, C2, A2) in zip(got, ref):
            self.assertEqual(p1, p2)
            np.testing.assert_array_equal(A1, A2)
            np.testing.assert_array_equal(C1, C2)
        C, A = det_t.run((es, ee), c0, got[7][0])
        np.testing.assert_array_equal(A, got[7][2])


if __name__ == '__main__':
    unittest.main()
