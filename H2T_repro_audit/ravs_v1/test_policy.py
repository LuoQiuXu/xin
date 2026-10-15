"""Fast CPU-only checks, independent of checkpoint/data/CUDA: python test_policy.py."""
from __future__ import annotations
import numpy as np
from ravs_policy import ARMS, choose, self_test, receiver_hardness, frequency_need


def test_policy():
    self_test()
    rng = np.random.default_rng(906)
    batch = 4
    y = np.array([0, 99, 99, 99])
    counts = np.ones(100, dtype=np.int64) * 20
    counts[0] = 500
    counts[99] = 5
    z = rng.normal(size=(batch, 100))
    # Explicitly easy versus very hard, same tail class.
    z[1] = np.zeros(100); z[1, 99] = 100.
    z[2] = np.zeros(100); z[2, 99] = -100.
    z[3] = z[2].copy()
    noise = rng.gumbel(size=(batch, 64))
    s = rng.uniform(size=(batch, 64))
    order = np.argsort(s,axis=1,kind="stable")
    pool = np.zeros((batch,64),dtype=bool)
    np.put_along_axis(pool,order[:,:48],True,axis=1)
    uniform = np.argsort(-noise,axis=1,kind="stable")[:,:16].astype(np.uint8)
    low = np.argsort(-np.where(pool,noise,-np.inf),axis=1,kind="stable")[:,:16].astype(np.uint8)
    trace = {"scores":s,"fallback":np.array([0,0,0,1]),
             "masks":{"Uniform-PerSample":uniform,"Low-VS":low.copy()}}
    trace["masks"]["Low-VS"][3] = uniform[3]
    h = receiver_hardness(z, y)
    assert h[1] < h[2]
    f = frequency_need(y,counts)
    assert f[0] == 0 and f[1] == 1
    for arm in ARMS:
        mask, diag = choose(arm,trace,y,counts,z,noise)
        if arm == "No-Fusion":
            assert mask is None
            continue
        assert mask.shape == (batch, 16)
        assert len(np.unique(mask[1])) == 16
        if arm in ("RAVS-V1","Frequency-Only"):
            assert diag["candidate_pool_size"][0] == 64
            assert np.array_equal(mask[0],uniform[0])
            assert np.array_equal(mask[3],uniform[3])
        if arm == "RAVS-V1":
            assert diag["candidate_pool_size"][2] < diag["candidate_pool_size"][1]
        if arm == "Low-VS":
            assert np.array_equal(mask,trace["masks"]["Low-VS"])
    print("PASS: endpoints, frequency gate, margin ordering, fallback,16 distinct channels")


if __name__ == "__main__":
    test_policy()
