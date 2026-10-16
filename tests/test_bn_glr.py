import unittest
import numpy as np
from filters import bn_glr, cusum


def brute_first_alarm(es, ee, c, b0, b1, D, drift, t0):
    """First alarm of one flight by direct enumeration (no restarts needed before it)."""
    T, K = es.shape
    for t in range(t0 + 1, T):
        best = (-np.inf, None)
        for k in range(t0 + 1, t + 1):
            for j in range(K):
                if j == c:
                    continue
                w = sum(es[i, j] - es[i, c] - drift for i in range(k, t + 1))
                seg = 0.0
                for a in range(k, t + 1):
                    for b in range(a, min(t, a + D - 1) + 1):
                        seg = max(seg, sum(ee[i, c] - es[i, c] for i in range(a, b + 1)))
                lam = w - seg
                if w >= b0 and lam >= b1 and lam > best[0]:
                    best = (lam, j)
        if best[1] is not None:
            return t, best[1]
    return None, None


class BNGLRTest(unittest.TestCase):
    def test_matches_enumeration(self):
        rng = np.random.default_rng(0)
        for trial in range(20):
            B, T, K, t0 = 1, 14, 3, 2
            es, ee = rng.normal(size=(B, T, K)), rng.normal(size=(B, T, K))
            es[:, 6:, 1] += 1.5
            b0, b1, D, drift = rng.uniform(0, 3), rng.uniform(-1, 2), int(rng.integers(1, 5)), rng.uniform(0, .5)
            C, alarm = bn_glr.run(es, ee, np.array([0]), b0, b1, D, drift, t0)
            t, j = brute_first_alarm(es[0], ee[0], 0, b0, b1, D, drift, t0)
            got = np.flatnonzero(alarm[0])
            if t is None:
                self.assertEqual(got.size, 0)
            else:
                self.assertEqual(got[0], t)
                self.assertEqual(C[0, t], j)

    def test_without_nuisance_term_matches_cusum(self):
        rng = np.random.default_rng(1)
        B, T, K, t0 = 64, 60, 5, 10
        e = rng.normal(size=(B, T, K)); e[:30, 30:, 2] += 1.0
        c0 = e[:, :t0 + 1].sum(1).argmax(-1)
        for drift, tau in [(0.0, 4.0), (0.5, 2.5), (1.0, 6.0)]:
            C1, a1 = cusum.run(e, c0, drift, tau, t0)
            C2, a2 = bn_glr.run(e, e, c0, tau + 1e-9, -np.inf, 5, drift, t0)
            np.testing.assert_array_equal(a1, a2)
            np.testing.assert_array_equal(C1, C2)


if __name__ == '__main__':
    unittest.main()
