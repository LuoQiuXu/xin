"""RAVS-H2T V1: receiver-conditioned VS candidate pool; no new trainable weights.

This is an algorithm prototype, not a claim that hardness predicts augmentation utility.
All strategies select exactly 16 distinct whole feature-map channels. The original
regularized VS scores, prior and donor fallback are supplied by the reviewed
online_policy.py; they are never recomputed or retuned in this module.
"""
from __future__ import annotations
import numpy as np

ARMS = ("No-Fusion", "Uniform-PerSample", "Low-VS",
        "Frequency-Only", "RAVS-V1")
C = 64
REPLACE = 16
MIN_POOL = 48
MAX_POOL = 64
MAX_CLASS_N = 500
MIN_CLASS_N = 5
EPS = 1e-8


def receiver_hardness(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Detached clean-receiver margin normalized by per-sample logit std.

    Hardness = sigmoid(-(l_y - max_{j!=y} l_j)/(std(logits)+1e-8)).
    No validation scores, labels or reference-classifier predictions enter here.
    """
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    assert z.ndim == 2 and z.shape[1] == 100 and y.shape == (len(z),)
    assert np.isfinite(z).all() and ((y >= 0) & (y < 100)).all()
    positives = z[np.arange(len(z)), y]
    negatives = z.copy()
    negatives[np.arange(len(z)), y] = -np.inf
    margin = positives - negatives.max(axis=1)
    standardized = np.clip(margin / (z.std(axis=1) + EPS), -20., 20.)
    return 1. / (1. + np.exp(standardized))


def frequency_need(labels: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Class rarity in [0,1]: 0 at n=500 and 1 at n=5."""
    y = np.asarray(labels, dtype=np.int64)
    n = np.asarray(counts)[y].astype(np.float64)
    assert ((n >= MIN_CLASS_N) & (n <= MAX_CLASS_N)).all()
    return np.clip(np.log(MAX_CLASS_N / n) / np.log(MAX_CLASS_N / MIN_CLASS_N), 0., 1.)


def choose(
    arm: str,
    trace: dict,
    receiver_labels: np.ndarray,
    class_counts: np.ndarray,
    clean_logits: np.ndarray | None,
    gumbel: np.ndarray,
) -> tuple[np.ndarray | None, dict]:
    """Return (B,16) selected channels or None for No-Fusion, with diagnostics.

    Frequency-only holds hardness constant at 0.5. RAVS uses current
    classifier's clean receiver margin before loss.backward / optimizer.step.
    Uniform / Low-VS endpoints reuse archived online_policy masks *exactly*.
    """
    if arm not in ARMS:
        raise ValueError(arm)
    labels = np.asarray(receiver_labels, dtype=np.int64)
    noise = np.asarray(gumbel, dtype=np.float64)
    n = len(labels)
    assert noise.shape == (n, C) and np.isfinite(noise).all()
    assert trace["scores"].shape == (n, C)
    assert trace["fallback"].shape == (n,)
    uniform = trace["masks"]["Uniform-PerSample"]
    low = trace["masks"]["Low-VS"]
    assert uniform.shape == low.shape == (n, REPLACE)
    need = frequency_need(labels, class_counts)
    if clean_logits is None:
        hardness = np.full(n, np.nan)
    else:
        hardness = receiver_hardness(clean_logits, labels)

    # Predeclared rule: omega = rarity * (0.5+0.5*hardness).
    # Frequency-Only is exactly the same formula with hardness fixed at 0.5.
    if arm == "RAVS-V1":
        assert np.isfinite(hardness).all(), "RAVS requires current clean logits"
        strength = need * (0.5 + 0.5 * hardness)
    elif arm == "Frequency-Only":
        # Diagnostic h must describe the gate actually used, not unused online h.
        hardness = np.full(n, 0.5)
        strength = need * 0.75
    else:
        strength = np.full(n, float("nan"))
    sizes = np.full(n, MAX_POOL, dtype=np.int64)
    if arm == "Low-VS":
        sizes[:] = MIN_POOL
        sizes[np.asarray(trace["fallback"]) > 0] = MAX_POOL
    if arm in ("RAVS-V1", "Frequency-Only"):
        # Fixed half-up rounding, pool=64 at omega0 and pool=48 at omega1.
        sizes = np.floor(MAX_POOL - (MAX_POOL - MIN_POOL) * strength + 0.5).astype(np.int64)
        sizes = np.clip(sizes, MIN_POOL, MAX_POOL)
        sizes[np.asarray(trace["fallback"]) > 0] = MAX_POOL

    if arm == "No-Fusion":
        selection = None
    elif arm == "Uniform-PerSample":
        selection = uniform.copy()
    elif arm == "Low-VS":
        selection = low.copy()
    else:
        order = np.argsort(trace["scores"], axis=1, kind="stable")
        flags = np.arange(C)[None, :] < sizes[:, None]
        eligible = np.zeros((n, C), dtype=bool)
        np.put_along_axis(eligible, order, flags, axis=1)
        # Gumbel ranking yields uniform sampling without replacement within
        # each per-receiver candidate pool. Reuse the exact same noise stream.
        selection = np.argsort(-np.where(eligible, noise, -np.inf),
                               axis=1, kind="stable")[:, :REPLACE].astype(np.uint8)
        if np.any(trace["fallback"] > 0):
            assert np.array_equal(selection[trace["fallback"] > 0],
                                  uniform[trace["fallback"] > 0])
        assert np.take_along_axis(eligible, selection, axis=1).all()

    if selection is not None:
        assert selection.shape == (n, REPLACE)
        assert selection.dtype == np.uint8
        assert (selection < C).all()
        assert np.all(np.diff(np.sort(selection, axis=1), axis=1) > 0)
    return selection, {
        "frequency_need": need, "hardness": hardness,
        "omega": strength, "candidate_pool_size": sizes,
        "fallback": np.asarray(trace["fallback"]),
    }


def self_test() -> None:
    rng = np.random.default_rng(2036)
    n = 6
    z = rng.normal(size=(n, 100))
    labels = np.array([0, 1, 2, 3, 4, 5])
    counts = np.r_[500, 5, 100, 20, 500, 5]
    h = receiver_hardness(z, labels)
    assert ((h > 0.) & (h < 1.)).all()
    s = rng.uniform(size=(n, C))
    noise = rng.gumbel(size=(n, C))
    idx = np.argsort(s, axis=1, kind="stable")[:, :MIN_POOL]
    pool = np.zeros((n, C), dtype=bool)
    np.put_along_axis(pool, idx, True, axis=1)
    u = np.argsort(-noise, axis=1, kind="stable")[:, :REPLACE].astype(np.uint8)
    low = np.argsort(-np.where(pool, noise, -np.inf),
                     axis=1, kind="stable")[:, :REPLACE].astype(np.uint8)
    trace = {"scores": s, "masks": {"Uniform-PerSample":u,"Low-VS":low},
             "fallback":np.array([0,0,0,0,1,1], dtype=np.uint8)}
    trace["masks"]["Low-VS"][4:] = u[4:]   # historic donor-class fallback
    for arm in ARMS:
        selection, diag = choose(arm, trace, labels, counts, z, noise)
        assert diag["candidate_pool_size"].shape == (n,)
        if arm == "No-Fusion":
            assert selection is None
        else:
            assert selection.shape == (n, REPLACE)
        if arm in ("Frequency-Only", "RAVS-V1"):
            assert diag["candidate_pool_size"][0] == C    # n=500
            assert np.array_equal(selection[4:], u[4:])  # fallback
        if arm == "Uniform-PerSample":
            assert np.array_equal(selection, u)
        if arm == "Low-VS":
            assert np.array_equal(selection, trace["masks"]["Low-VS"])
    # With omega=0 (frequent class), RAVS exactly reduces to Uniform.
    selection, _ = choose("RAVS-V1", trace, labels, counts, z, noise)
    assert np.array_equal(selection[0], u[0])
