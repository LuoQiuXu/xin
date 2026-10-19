"""Pure policy functions for the Low-VS donor-affinity × margin-protection pilot.

This module does not train a model and never reads validation/test data. All
prototype inputs must be frozen Stage-I features from CIFAR-100-LT TRAIN.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

N_CLASSES = 100
N_CHANNELS = 64
N_REPLACE = 16
N_CANDIDATES = 48
HEAD_IDS = tuple(range(36))
TAIL_START = 71
N_FREQUENCY_STRATA = 6
TOP_CLASSES_PER_STRATUM = 2


def _counts_array(class_counts: Sequence[int]) -> np.ndarray:
    counts = np.asarray(class_counts, dtype=np.int64)
    if counts.shape != (N_CLASSES,):
        raise ValueError(f"class_counts must have shape ({N_CLASSES},), got {counts.shape}")
    if np.any(counts <= 0):
        raise ValueError("All CIFAR-100-LT classes must have positive training counts")
    return counts


def build_frequency_strata(
    class_counts: Sequence[int],
    head_ids: Sequence[int] = HEAD_IDS,
    n_strata: int = N_FREQUENCY_STRATA,
) -> tuple[np.ndarray, ...]:
    """Partition head classes into equal-sized count-rank strata.

    Classes are sorted by decreasing training count (class ID breaks ties).
    For CIFAR-100-LT IF=100, 36 head classes and 6 strata yield 6 classes per
    stratum. The random and affinity arms draw the stratum from the same
    count-mass distribution, so the marginal stratum prior is paired.
    """
    counts = _counts_array(class_counts)
    ids = np.asarray(tuple(int(c) for c in head_ids), dtype=np.int64)
    if ids.ndim != 1 or len(ids) == 0 or len(np.unique(ids)) != len(ids):
        raise ValueError("head_ids must be a non-empty unique sequence")
    if np.any(ids < 0) or np.any(ids >= N_CLASSES):
        raise ValueError("head_ids contains an invalid class ID")
    if not 1 <= n_strata <= len(ids):
        raise ValueError("n_strata must be between 1 and number of head classes")
    ids = np.asarray(sorted(ids.tolist(), key=lambda c: (-int(counts[c]), c)), dtype=np.int64)
    strata = tuple(np.asarray(chunk, dtype=np.int64) for chunk in np.array_split(ids, n_strata))
    if any(len(s) == 0 for s in strata):
        raise AssertionError("Empty frequency stratum")
    flat = np.concatenate(strata)
    if not np.array_equal(np.sort(flat), np.sort(ids)):
        raise AssertionError("Frequency strata do not partition head IDs")
    return strata


def build_head_prototypes(
    train_gap: np.ndarray,
    train_labels: Sequence[int],
    head_ids: Sequence[int] = HEAD_IDS,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute L2-normalized class means from clean TRAIN GAP features only."""
    x = np.asarray(train_gap, dtype=np.float32)
    y = np.asarray(train_labels, dtype=np.int64)
    ids = np.asarray(tuple(int(c) for c in head_ids), dtype=np.int64)
    if x.ndim != 2 or x.shape[1] != N_CHANNELS:
        raise ValueError(f"train_gap must have shape [N,{N_CHANNELS}]")
    if y.shape != (len(x),) or len(x) == 0:
        raise ValueError("train_labels must match train_gap rows")
    if not np.isfinite(x).all():
        raise ValueError("train_gap contains non-finite values")
    prototypes = []
    for class_id in ids:
        selected = x[y == class_id]
        if len(selected) == 0:
            raise ValueError(f"No TRAIN features for head class {class_id}")
        mean = selected.mean(axis=0, dtype=np.float64).astype(np.float32)
        norm = float(np.linalg.norm(mean))
        if not math.isfinite(norm) or norm <= 1e-12:
            raise ValueError(f"Degenerate prototype for class {class_id}")
        prototypes.append(mean / norm)
    result = np.stack(prototypes, axis=0).astype(np.float32)
    if result.shape != (len(ids), N_CHANNELS) or not np.isfinite(result).all():
        raise AssertionError("Invalid head prototype bank")
    return ids, result


def receiver_prototype_affinity(
    receiver_gap: np.ndarray,
    head_ids: Sequence[int],
    normalized_prototypes: np.ndarray,
) -> np.ndarray:
    """Cosine affinity of each frozen TRAIN receiver feature to each head prototype."""
    x = np.asarray(receiver_gap, dtype=np.float32)
    p = np.asarray(normalized_prototypes, dtype=np.float32)
    ids = np.asarray(tuple(int(c) for c in head_ids), dtype=np.int64)
    if x.ndim != 2 or x.shape[1] != N_CHANNELS:
        raise ValueError(f"receiver_gap must have shape [B,{N_CHANNELS}]")
    if p.shape != (len(ids), N_CHANNELS):
        raise ValueError("prototype shape does not match head_ids")
    if not np.isfinite(x).all() or not np.isfinite(p).all():
        raise ValueError("Affinity inputs must be finite")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x_norm = x / np.maximum(norms, 1e-12)
    affinity = x_norm @ p.T
    if not np.isfinite(affinity).all():
        raise ValueError("Non-finite prototype affinity")
    return affinity.astype(np.float32)


def choose_donor_classes(
    affinity: np.ndarray,
    head_ids: Sequence[int],
    class_counts: Sequence[int],
    strata: Sequence[Sequence[int]],
    u_stratum: Sequence[float],
    u_class: Sequence[float],
    policy: str,
    top_classes_per_stratum: int = TOP_CLASSES_PER_STRATUM,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Choose donor classes using paired random numbers and frequency strata.

    policy='random' samples count-proportionally within a sampled stratum.
    policy='high' samples from the top-affinity K classes in that same stratum.
    policy='bottom' is a diagnostic-only bottom-affinity counterpart.

    All policies receive the same u_stratum and therefore the same frequency
    stratum per receiver. u_class is also shared, providing common random
    numbers despite different within-stratum candidate sets.
    """
    if policy not in {"random", "high", "bottom"}:
        raise ValueError(f"Unknown donor policy: {policy}")
    counts = _counts_array(class_counts)
    ids = np.asarray(tuple(int(c) for c in head_ids), dtype=np.int64)
    scores = np.asarray(affinity, dtype=np.float64)
    us = np.asarray(u_stratum, dtype=np.float64)
    uc = np.asarray(u_class, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] != len(ids):
        raise ValueError("affinity must have shape [B,len(head_ids)]")
    b = scores.shape[0]
    if us.shape != (b,) or uc.shape != (b,):
        raise ValueError("u_stratum and u_class must have shape [B]")
    if not np.isfinite(scores).all():
        raise ValueError("Affinity scores must be finite")
    if np.any((us < 0) | (us >= 1)) or np.any((uc < 0) | (uc >= 1)):
        raise ValueError("Uniform random values must lie in [0,1)")
    strata_arr = tuple(np.asarray(s, dtype=np.int64) for s in strata)
    if not strata_arr or any(len(s) == 0 for s in strata_arr):
        raise ValueError("strata must be non-empty")
    all_strata = np.concatenate(strata_arr)
    if len(np.unique(all_strata)) != len(all_strata):
        raise ValueError("Frequency strata overlap")
    if set(all_strata.tolist()) != set(ids.tolist()):
        raise ValueError("Frequency strata must partition head_ids")
    id_to_col = {int(c): i for i, c in enumerate(ids)}
    masses = np.asarray([counts[s].sum() for s in strata_arr], dtype=np.float64)
    cdf = np.cumsum(masses / masses.sum())
    donor_classes = np.empty(b, dtype=np.int64)
    chosen_strata = np.empty(b, dtype=np.int64)
    selected_affinity = np.empty(b, dtype=np.float32)
    k = max(1, min(int(top_classes_per_stratum), min(len(s) for s in strata_arr)))

    for i in range(b):
        si = min(int(np.searchsorted(cdf, us[i], side="right")), len(strata_arr) - 1)
        pool = strata_arr[si]
        if policy != "random":
            cols = np.asarray([id_to_col[int(c)] for c in pool], dtype=np.int64)
            local_scores = scores[i, cols]
            order = np.argsort(local_scores, kind="stable")
            pool = pool[order[-k:]] if policy == "high" else pool[order[:k]]
        weights = counts[pool].astype(np.float64)
        weights /= weights.sum()
        ci = min(int(np.searchsorted(np.cumsum(weights), uc[i], side="right")), len(pool) - 1)
        chosen = int(pool[ci])
        donor_classes[i] = chosen
        chosen_strata[i] = si
        selected_affinity[i] = scores[i, id_to_col[chosen]]
    return donor_classes, chosen_strata, selected_affinity


def sample_donor_indices(
    donor_classes: Sequence[int],
    train_labels: Sequence[int],
    u_instance: Sequence[float],
) -> np.ndarray:
    """Sample an actual local TRAIN index uniformly within each chosen class."""
    y = np.asarray(train_labels, dtype=np.int64)
    classes = np.asarray(donor_classes, dtype=np.int64)
    u = np.asarray(u_instance, dtype=np.float64)
    if y.ndim != 1 or classes.ndim != 1 or u.shape != classes.shape:
        raise ValueError("Invalid donor-index input shapes")
    if np.any((u < 0) | (u >= 1)):
        raise ValueError("u_instance must lie in [0,1)")
    cache: dict[int, np.ndarray] = {}
    out = np.empty(len(classes), dtype=np.int64)
    for i, c in enumerate(classes):
        c = int(c)
        if c not in cache:
            cache[c] = np.flatnonzero(y == c)
            if len(cache[c]) == 0:
                raise ValueError(f"No TRAIN samples for donor class {c}")
        candidates = cache[c]
        j = min(int(u[i] * len(candidates)), len(candidates) - 1)
        out[i] = candidates[j]
    if not np.array_equal(y[out], classes):
        raise AssertionError("Sampled donor index label mismatch")
    return out


def channel_margin_delta(
    receiver_maps: torch.Tensor,
    donor_maps: torch.Tensor,
    target: torch.Tensor,
    classifier: torch.nn.Module,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-channel change in the receiver's fixed strongest-competitor margin.

    q_d = (W[y,d] - W[j*,d]) * (GAP(donor)_d - GAP(receiver)_d), where j*
    is the strongest non-target class on the clean receiver. Since the Stage-II
    classifier is linear after GAP, sum_d q_d equals the exact change in the
    y-vs-j* margin if every channel were replaced, holding j* fixed.
    """
    if receiver_maps.shape != donor_maps.shape or receiver_maps.ndim != 4:
        raise ValueError("receiver_maps and donor_maps must share [B,C,H,W] shape")
    if receiver_maps.shape[1] != N_CHANNELS:
        raise ValueError(f"Expected {N_CHANNELS} feature channels")
    fc = classifier.module.fc if hasattr(classifier, "module") else classifier.fc
    if fc.weight.ndim != 2 or fc.weight.shape[1] != N_CHANNELS:
        raise ValueError("Expected a linear classifier with 64 GAP inputs")
    y = target.to(device=receiver_maps.device, dtype=torch.long)
    if y.shape != (len(receiver_maps),):
        raise ValueError("target shape mismatch")
    with torch.no_grad():
        receiver_gap = receiver_maps.mean(dim=(-2, -1))
        donor_gap = donor_maps.mean(dim=(-2, -1))
        clean_logits = classifier(receiver_maps)
        rows = torch.arange(len(y), device=y.device)
        competitors = clean_logits.detach().clone()
        competitors[rows, y] = -torch.inf
        j_star = competitors.argmax(dim=1)
        w_delta = fc.weight[y] - fc.weight[j_star]
        q = w_delta * (donor_gap - receiver_gap)
        if not torch.isfinite(q).all():
            raise ValueError("Non-finite channel margin contribution")
    return q, j_star, clean_logits.detach()


def protected_lowvs_mask(
    bottom48_candidates: torch.Tensor,
    channel_delta_margin: torch.Tensor,
    gumbel_noise: torch.Tensor,
    n_protect: int = N_REPLACE,
    n_replace: int = N_REPLACE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exclude the 16 worst predicted margin-transfer channels within Bottom48.

    The original Low-VS Bottom48 candidate pool is immutable. The n_protect
    channels with the smallest q_d inside that pool are protected; n_replace
    channels are then sampled from the remaining candidates using the SAME
    Gumbel values as the random Low-VS arm. Thus A-vs-C and B-vs-D only differ
    in the final candidate exclusion, not the donor, VS score, Gumbel stream,
    replacement count, or non-tail policy.
    """
    candidates = torch.as_tensor(bottom48_candidates, dtype=torch.bool)
    q = torch.as_tensor(channel_delta_margin, device=candidates.device)
    noise = torch.as_tensor(gumbel_noise, device=candidates.device)
    if candidates.ndim != 2 or candidates.shape[1] != N_CHANNELS:
        raise ValueError("bottom48_candidates must have shape [B,64]")
    if q.shape != candidates.shape or noise.shape != candidates.shape:
        raise ValueError("q and gumbel_noise must match candidate-mask shape")
    if n_protect < 1 or n_replace < 1:
        raise ValueError("n_protect and n_replace must be positive")
    counts = candidates.sum(dim=1)
    if not torch.all(counts == N_CANDIDATES):
        raise ValueError(f"Every Bottom48 row must contain {N_CANDIDATES} candidates")
    if n_protect + n_replace > N_CANDIDATES:
        raise ValueError("Protection plus replacement exceeds candidate pool")
    if not torch.isfinite(q).all() or not torch.isfinite(noise).all():
        raise ValueError("q and Gumbel values must be finite")
    selected = torch.zeros_like(candidates)
    protected = torch.zeros_like(candidates)
    for i in range(len(candidates)):
        cand = torch.nonzero(candidates[i], as_tuple=False).flatten()
        q_order = torch.argsort(q[i, cand], stable=True)
        protected_idx = cand[q_order[:n_protect]]
        protected[i, protected_idx] = True
        keep = candidates[i].clone()
        keep[protected_idx] = False
        eligible = torch.nonzero(keep, as_tuple=False).flatten()
        noise_order = torch.argsort(noise[i, eligible], descending=True, stable=True)
        selected_idx = eligible[noise_order[:n_replace]]
        selected[i, selected_idx] = True
    if not torch.all(selected.sum(dim=1) == n_replace):
        raise AssertionError("Wrong replacement count")
    if torch.any(selected & protected) or torch.any(selected & ~candidates):
        raise AssertionError("Selected mask violates protection or Bottom48")
    return selected, protected
