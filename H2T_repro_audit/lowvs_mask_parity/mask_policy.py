"""Paired 2x2 channel masks: randomized vs Low-VS, shared vs individual.

The original H2T uses one random 16/64 mask per minibatch. Here U-Shared
matches that mask *distribution and sharing semantics* but uses a pre-recorded
Gumbel random stream, not the official torch.randperm RNG sequence.

Shared VS scores are the mean of valid receiver--donor channel scores within
the ACTUAL minibatch. This is a new, prespecified extension of Low-VS; no
claim is made that it is a module of the original H2T.
"""
from __future__ import annotations

import numpy as np

CHANNELS = 64
TAKE = 16
POOL = 48
ARMS = ("U-Shared", "VS-Shared", "U-Individual", "VS-Individual")


def _top_gumbel(gumbel: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    """Uniform size-TAKE subset of the allowed pool, using paired Gumbel keys."""
    if allowed.shape != gumbel.shape:
        raise ValueError("Mask pool and Gumbel keys must have identical shapes")
    if not np.isfinite(gumbel).all():
        raise ValueError("Non-finite Gumbel keys")
    if (allowed.sum(axis=1) < TAKE).any():
        raise ValueError("Candidate pool smaller than replacement count")
    values = np.where(allowed, gumbel, -np.inf)
    ids = np.argsort(-values, axis=1, kind="stable")[:, :TAKE]
    if ids.shape != (len(gumbel), TAKE):
        raise AssertionError("Invalid selected-mask dimensions")
    if not np.take_along_axis(allowed, ids, axis=1).all():
        raise AssertionError("Selected index outside the candidate pool")
    if np.any(np.diff(np.sort(ids, axis=1), axis=1) <= 0):
        raise AssertionError("Duplicate selected channel")
    return ids.astype(np.uint8)


def choose_masks(arm: str, trace: dict, noise: np.ndarray) -> np.ndarray:
    """Return Bx16 mask; alter neither trace nor the global random state.

    Shared Low-VS is defined by the within-batch mean of eligible scores.
    Donors with n<20 or degenerate channel scores are excluded from the
    shared score estimate. If none are eligible, it becomes a uniform mask.
    Individual arms reproduce the reviewed precomputed online policy masks.
    """
    if arm not in ARMS:
        raise ValueError(f"Unknown mask policy: {arm}")
    noise = np.asarray(noise, dtype=np.float64)
    scores = np.asarray(trace["scores"], dtype=np.float64)
    fallback = np.asarray(trace["fallback"])
    if scores.ndim != 2 or scores.shape[1] != CHANNELS:
        raise ValueError("scores must be [B, 64]")
    if noise.shape != scores.shape or fallback.shape != (len(scores),):
        raise ValueError("inconsistent scores/noise/fallback dimensions")
    if not np.isfinite(scores).all() or not np.isfinite(noise).all():
        raise ValueError("Non-finite score/noise")
    b = len(scores)
    if b == 0:
        raise ValueError("Empty minibatch")

    if arm == "U-Individual":
        out = np.asarray(trace["masks"]["Uniform-PerSample"]).copy()
    elif arm == "VS-Individual":
        out = np.asarray(trace["masks"]["Low-VS"]).copy()
    else:
        g = noise[:1]
        if arm == "VS-Shared" and np.any(fallback == 0):
            mean_scores = np.mean(scores[fallback == 0], axis=0)
            bottom = np.argsort(mean_scores, kind="stable")[:POOL]
            allowed = np.zeros((1, CHANNELS), dtype=bool)
            allowed[0, bottom] = True
        else:
            allowed = np.ones((1, CHANNELS), dtype=bool)
        ids = _top_gumbel(g, allowed)
        out = np.broadcast_to(ids, (b, TAKE)).copy()

    if out.shape != (b, TAKE):
        raise AssertionError("Incorrect mask shape")
    if np.any(np.diff(np.sort(out, axis=1), axis=1) <= 0):
        raise AssertionError("Duplicate mask channels")
    if not np.all((out >= 0) & (out < CHANNELS)):
        raise AssertionError("Out-of-range channel")
    if arm.endswith("Shared") and not np.array_equal(out, np.broadcast_to(out[:1], out.shape)):
        raise AssertionError("Batch-shared policy did not share the mask")
    if arm == "VS-Individual":
        uniform = np.asarray(trace["masks"]["Uniform-PerSample"])
        if not np.array_equal(out[fallback != 0], uniform[fallback != 0]):
            raise AssertionError("Individual fallback differs from Uniform")
    return out
