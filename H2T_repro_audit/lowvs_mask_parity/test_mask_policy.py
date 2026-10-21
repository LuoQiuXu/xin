"""CPU-only scientific invariants: python -m unittest discover -s ... -p 'test_*.py'"""
import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mask_policy import ARMS, CHANNELS, TAKE, choose_masks


def fixture(b=7, seed=213):
    rng = np.random.default_rng(seed)
    scores = rng.random((b, CHANNELS), dtype=np.float64)
    gumbel = rng.gumbel(size=(b, CHANNELS))
    fallback = np.zeros(b, dtype=np.uint8)
    fallback[-1] = 1
    pool = np.argsort(scores, axis=1, kind="stable")[:, :48]
    uniform = np.argsort(-gumbel, axis=1, kind="stable")[:, :TAKE]
    allowed = np.zeros((b, CHANNELS), dtype=bool)
    np.put_along_axis(allowed, pool, True, axis=1)
    allowed[fallback != 0] = True
    lowvs = np.argsort(-np.where(allowed, gumbel, -np.inf), axis=1, kind="stable")[:, :TAKE]
    trace = {"scores":scores, "fallback":fallback,
             "masks":{"Uniform-PerSample":uniform.astype(np.uint8),
                      "Low-VS":lowvs.astype(np.uint8)}}
    return trace, gumbel


class TestPolicy(unittest.TestCase):
    def test_all_arms_shapes_and_distinct_channels(self):
        for b in (1, 7, 95, 128):
            trace, g = fixture(b)
            for arm in ARMS:
                picked = choose_masks(arm, trace, g)
                self.assertEqual(picked.shape, (b, TAKE))
                self.assertTrue(np.all(np.diff(np.sort(picked, axis=1), axis=1) > 0))
                self.assertTrue(np.all(picked < CHANNELS))
                if arm.endswith("Shared"):
                    self.assertTrue(np.all(picked == picked[0]))

    def test_uniform_shared_equals_first_uniform_draw(self):
        trace, g = fixture()
        wanted = trace["masks"]["Uniform-PerSample"][0]
        actual = choose_masks("U-Shared", trace, g)
        self.assertTrue(np.array_equal(actual[0], wanted))

    def test_individual_replays_reviewed_policy(self):
        trace, g = fixture()
        for arm, key in (("U-Individual","Uniform-PerSample"),("VS-Individual","Low-VS")):
            self.assertTrue(np.array_equal(choose_masks(arm, trace, g), trace["masks"][key]))

    def test_shared_vs_pool_and_shared_noise(self):
        trace, g = fixture()
        pick = choose_masks("VS-Shared", trace, g)
        rank = np.argsort(trace["scores"][trace["fallback"]==0].mean(0), kind="stable")[:48]
        self.assertTrue(set(pick[0]).issubset(set(rank)))
        self.assertTrue(np.array_equal(pick, np.broadcast_to(pick[0],pick.shape)))
        self.assertEqual(pick.dtype, np.uint8)

    def test_shared_fallback_when_all_invalid(self):
        trace, g = fixture()
        trace["fallback"][:] = 1
        self.assertTrue(np.array_equal(choose_masks("VS-Shared", trace, g),
                                       choose_masks("U-Shared", trace, g)))

    def test_does_not_modify_inputs(self):
        trace, g = fixture()
        s, old_g = trace["scores"].copy(), g.copy()
        m = trace["masks"]["Uniform-PerSample"].copy()
        for arm in ARMS:
            choose_masks(arm, trace, g)
        self.assertTrue(np.array_equal(s,trace["scores"]))
        self.assertTrue(np.array_equal(g,old_g))
        self.assertTrue(np.array_equal(m, trace["masks"]["Uniform-PerSample"]))

    def test_invalid_inputs_fail_loudly(self):
        t, g = fixture()
        with self.assertRaises(ValueError): choose_masks("unknown", t, g)
        with self.assertRaises(ValueError): choose_masks("U-Shared", t, g[:, :-1])
        bad = g.copy(); bad[1,0] = np.nan
        with self.assertRaises(ValueError): choose_masks("VS-Shared", t, bad)


if __name__ == "__main__":
    unittest.main()
