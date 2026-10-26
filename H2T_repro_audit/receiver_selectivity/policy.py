"""Pure selection logic: isolate receiver-class exposure in original H2T.

No new feature score, learnable gate, loss, attention or donor selection.
Only the binary decision to apply *the existing* H2T channel mask changes.
"""
from __future__ import annotations
import numpy as np

TAIL_START = 71
ARMS = ("DR-NoFusion", "Original-BatchShared", "TailOff", "RandomOffMatched")


def choose_active(labels: np.ndarray, arm: str, seed: int, epoch: int, batch: int) -> np.ndarray:
    """Return boolean flags for original H2T fusion, one per receiver.

    TailOff excludes tail receiver labels [71,99].
    RandomOffMatched excludes the exact same NUMBER of examples each batch,
    drawn without using labels. The random generator is independent from
    augmentation, donor sampling and shared channel-mask RNG streams.
    """
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1 or len(labels) == 0:
        raise ValueError("labels must be a nonempty rank-1 array")
    if np.any((labels < 0) | (labels >= 100)):
        raise ValueError("receiver labels outside [0,99]")
    if arm not in ARMS:
        raise ValueError(f"unknown ablation arm: {arm}")
    if arm == "DR-NoFusion":
        return np.zeros(len(labels), dtype=bool)
    if arm == "Original-BatchShared":
        return np.ones(len(labels), dtype=bool)
    n_tail = int(np.count_nonzero(labels >= TAIL_START))
    if arm == "TailOff":
        active = labels < TAIL_START
    else:
        # Generate exactly n_tail off-positions independently of receiver labels.
        # The same seed/epoch/batch always yields the same control, with no effect
        # on the historical NumPy, torch, augmentation, or donor random streams.
        ss = np.random.SeedSequence([20261026, int(seed), int(epoch), int(batch)])
        rng = np.random.default_rng(ss)
        off = rng.choice(len(labels), size=n_tail, replace=False)
        active = np.ones(len(labels), dtype=bool)
        active[off] = False
    assert int(active.sum()) == len(labels) - n_tail
    return active
