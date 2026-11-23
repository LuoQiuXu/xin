from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class NoiseResult:
    """Deterministic symmetric label-noise realization for a fixed LT subset."""

    clean_labels: np.ndarray
    noisy_labels: np.ndarray
    flipped_mask: np.ndarray
    requested_rate: float
    realized_rate: float
    seed: int
    num_classes: int

    @property
    def observed_class_counts(self) -> np.ndarray:
        return np.bincount(self.noisy_labels, minlength=self.num_classes).astype(np.int64)

    @property
    def clean_class_counts(self) -> np.ndarray:
        return np.bincount(self.clean_labels, minlength=self.num_classes).astype(np.int64)


def make_symmetric_noisy_labels(
    clean_labels: Iterable[int] | np.ndarray,
    noise_rate: float,
    seed: int,
    num_classes: int = 100,
) -> NoiseResult:
    """Apply instance-independent symmetric noise without ever retaining a flipped label.

    Exactly round(noise_rate * N) examples are selected without replacement.
    Each selected label is changed uniformly to one of the other K-1 classes.
    This is class-agnostic symmetric noise; it is not pair-flip, class-dependent,
    or instance-dependent noise. Input order is preserved.
    """
    labels = np.asarray(clean_labels, dtype=np.int64)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("clean_labels must be a non-empty 1-D array")
    if num_classes < 2:
        raise ValueError("num_classes must be >= 2")
    if np.any(labels < 0) or np.any(labels >= num_classes):
        raise ValueError("clean_labels contains an out-of-range class ID")
    if not 0.0 <= float(noise_rate) < 1.0:
        raise ValueError("noise_rate must be in [0, 1)")
    if int(seed) < 0:
        raise ValueError("seed must be non-negative")

    rng = np.random.RandomState(int(seed))
    noisy = labels.copy()
    n_flip = int(round(float(noise_rate) * labels.size))
    flipped = np.zeros(labels.size, dtype=bool)
    if n_flip:
        selected = rng.choice(labels.size, size=n_flip, replace=False)
        # Offset in [1,K-1] guarantees a different target while preserving
        # a uniform distribution over all alternative classes.
        offsets = rng.randint(1, num_classes, size=n_flip)
        noisy[selected] = (labels[selected] + offsets) % num_classes
        flipped[selected] = True

    realized = float(flipped.mean())
    if int(flipped.sum()) != n_flip:
        raise RuntimeError("Noise realization has an unexpected number of flips")
    if np.any(noisy[flipped] == labels[flipped]) or np.any(noisy[~flipped] != labels[~flipped]):
        raise RuntimeError("Symmetric noise integrity check failed")

    return NoiseResult(
        clean_labels=labels.copy(),
        noisy_labels=noisy,
        flipped_mask=flipped,
        requested_rate=float(noise_rate),
        realized_rate=realized,
        seed=int(seed),
        num_classes=int(num_classes),
    )


def validate_noise_result(result: NoiseResult, expected_clean_counts: np.ndarray | None = None) -> None:
    """Fail closed if a saved/generated corruption violates the LT/noise contract."""
    clean = np.asarray(result.clean_labels, dtype=np.int64)
    noisy = np.asarray(result.noisy_labels, dtype=np.int64)
    flipped = np.asarray(result.flipped_mask, dtype=bool)
    if clean.shape != noisy.shape or clean.shape != flipped.shape or clean.ndim != 1:
        raise ValueError("clean/noisy labels and flipped mask must have identical 1-D shapes")
    if np.any((clean < 0) | (clean >= result.num_classes)):
        raise ValueError("clean labels outside class range")
    if np.any((noisy < 0) | (noisy >= result.num_classes)):
        raise ValueError("noisy labels outside class range")
    if not np.array_equal(clean[~flipped], noisy[~flipped]):
        raise ValueError("Unflipped examples were modified")
    if np.any(clean[flipped] == noisy[flipped]):
        raise ValueError("A marked flipped example retained its original label")
    if int(flipped.sum()) != int(round(result.requested_rate * clean.size)):
        raise ValueError("Observed flip count differs from requested symmetric-noise rate")
    if expected_clean_counts is not None:
        expected = np.asarray(expected_clean_counts, dtype=np.int64)
        if not np.array_equal(np.bincount(clean, minlength=result.num_classes), expected):
            raise ValueError("Clean labels do not match the expected CIFAR-100-LT class counts")
