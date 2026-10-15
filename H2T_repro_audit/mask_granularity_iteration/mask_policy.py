"""Minimal controlled mask interventions. No learned channel scores or new hyperparameters."""
from __future__ import annotations
import numpy as np

ARMS = ("Uniform-PerSample", "Uniform-BatchShared", "Low-VS")
K = 16
CHANNELS = 64


def choose(arm: str, trace: dict) -> np.ndarray:
    """Return B x 16 channel IDs. Reuse the reviewed Uniform/Low-VS masks.

    BatchShared uses the first receiver's *uniform* draw, tiled across the
    minibatch. This has the exact original H2T 4D mask-sharing semantics,
    but deliberately uses the paired Gumbel stream rather than torch.randperm.
    """
    uniform = trace["masks"]["Uniform-PerSample"]
    if arm == "Uniform-PerSample":
        chosen = uniform
    elif arm == "Low-VS":
        chosen = trace["masks"]["Low-VS"]
    elif arm == "Uniform-BatchShared":
        chosen = np.repeat(uniform[:1], repeats=len(uniform), axis=0)
    else:
        raise ValueError(f"Unknown arm: {arm}")
    assert chosen.shape == (len(uniform), K)
    assert chosen.dtype == np.uint8
    assert np.all((chosen >= 0) & (chosen < CHANNELS))
    assert np.all(np.diff(np.sort(chosen, axis=1), axis=1) > 0)
    if arm == "Uniform-BatchShared":
        assert np.array_equal(chosen[0], uniform[0])
        assert np.array_equal(chosen, np.broadcast_to(chosen[:1], chosen.shape))
    return chosen


def self_test() -> None:
    rng = np.random.default_rng(724)
    for b in (1, 7, 95, 128):
        g = rng.gumbel(size=(b, CHANNELS))
        u = np.argsort(-g, axis=1, kind="stable")[:, :K].astype(np.uint8)
        l = np.argsort(-g[:, ::-1], axis=1, kind="stable")[:, :K].astype(np.uint8)
        fake = {"masks": {"Uniform-PerSample": u, "Low-VS": l}}
        assert np.array_equal(choose("Uniform-PerSample", fake), u)
        assert np.array_equal(choose("Low-VS", fake), l)
        batch = choose("Uniform-BatchShared", fake)
        assert np.array_equal(batch, np.tile(u[0], (b, 1)))
