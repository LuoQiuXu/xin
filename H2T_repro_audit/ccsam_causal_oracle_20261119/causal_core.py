"""Calibration-only masks and row-local optimizer-gradient interventions.

No loader, model update, or evaluation is performed at import time.  The
validated parent formula is imported rather than copied.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / "ccsam_stage2_controlled_cifar100_20261106"
for directory in (BASE, BASE / "ccmia"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from ccmia_core import (  # noqa: E402
    clean_weighted_ce_gradient,
    decompose_perturbation_gradient_by_source,
    project_source_gradients_to_fixed_margin,
)

ARMS = ("C2", "BASE_C3", "TARGETED", "RANDOM_MATCHED",
        "FREQUENCY_MATCHED", "ORACLE_SELECTION")
ALIASES = {"ORACLE_SUPPRESSION": "TARGETED"}
GRAD_ATOL, GRAD_RTOL = 2e-5, 2e-4  # unchanged validated CC-MIA limits
SPLIT_SEED = 20261119
BOOTSTRAP_SEED, BOOTSTRAP_REPLICATES = 20261120, 2000


def canonical_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def array_hash(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode())
    digest.update(str(value.dtype).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def deterministic_split(raw_ids, allocation_classes, seed=SPLIT_SEED, per_half=10):
    """Use pre-existing allocation metadata, never model scores, to split rows.

    A raw-ID sort makes membership invariant to a permutation of cache rows.
    Actual labels are validated separately, with Evaluation still sealed.
    """
    raw_ids = np.asarray(raw_ids, dtype=np.int64)
    classes = np.asarray(allocation_classes, dtype=np.int64)
    if raw_ids.ndim != 1 or classes.shape != raw_ids.shape:
        raise ValueError("One allocation class and original TRAIN ID per row required")
    if len(np.unique(raw_ids)) != len(raw_ids):
        raise ValueError("Duplicate original TRAIN IDs")
    calibration, evaluation = [], []
    for cls in np.unique(classes):
        rows = np.flatnonzero(classes == cls)
        if len(rows) != 2 * per_half:
            raise ValueError(f"Class {cls} has {len(rows)}, expected {2 * per_half}")
        rows = rows[np.argsort(raw_ids[rows], kind="stable")]
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(cls)]))
        rows = rows[rng.permutation(len(rows))]
        calibration.extend(rows[:per_half].tolist())
        evaluation.extend(rows[per_half:].tolist())
    calibration, evaluation = np.sort(calibration), np.sort(evaluation)
    if np.intersect1d(calibration, evaluation).size:
        raise RuntimeError("Calibration / Evaluation overlap")
    if not np.array_equal(np.sort(np.r_[calibration, evaluation]), np.arange(len(raw_ids))):
        raise RuntimeError("Split does not partition the original holdout")
    return {
        "split_seed": int(seed), "per_class_per_half": int(per_half),
        "calibration_positions": calibration.tolist(),
        "evaluation_positions": evaluation.tolist(),
        "calibration_raw_train_ids": raw_ids[calibration].tolist(),
        "evaluation_raw_train_ids": raw_ids[evaluation].tolist(),
        "allocation_classes": classes.tolist(),
        "calibration_positions_sha256": array_hash(calibration.astype(np.int64)),
        "evaluation_positions_sha256": array_hash(evaluation.astype(np.int64)),
        "calibration_raw_ids_sha256": array_hash(raw_ids[calibration]),
        "evaluation_raw_ids_sha256": array_hash(raw_ids[evaluation]),
        "interpretation": "TRAIN-derived held-out development data; class 0 absent",
    }


def eligible_pairs(train_counts, calibration_classes):
    counts = np.asarray(train_counts, dtype=np.int64)
    if counts.ndim != 1 or np.any(counts <= 0):
        raise ValueError("Positive sealed TRAIN counts required")
    targets = np.intersect1d(np.flatnonzero(counts > 100), calibration_classes)
    sources = np.flatnonzero(counts <= 100)
    return [(int(s), int(t)) for s in sources for t in targets]


def fixed_k(number_eligible):
    if number_eligible < 0:
        raise ValueError("Negative pair count")
    return (int(number_eligible) + 5) // 10  # exact half-up, no float rounding


def source_bin(count):
    # Predeclared TRAIN-only bins, including absent/synthetic classes in tests.
    return int(np.searchsorted([20, 50, 101], int(count), side="right"))


def target_bin(count):
    return int(np.searchsorted([200, 350], int(count), side="right"))


def matched_control(pairs, targeted, counts, rng_seed, exact_target):
    pairs, targeted = sorted(set(pairs)), sorted(set(targeted))
    if not set(targeted).issubset(pairs):
        raise ValueError("Targeted mask outside eligible family")
    def key(pair):
        s, t = pair
        return (source_bin(counts[s]), t if exact_target else target_bin(counts[t]))
    budget = Counter(key(pair) for pair in targeted)
    rng = np.random.default_rng(int(rng_seed))
    selected = []
    for stratum in sorted(budget):
        population = [p for p in pairs if key(p) == stratum]
        n = budget[stratum]
        if n > len(population):
            raise RuntimeError("Matched-control stratum has insufficient eligible pairs")
        selection = rng.choice(len(population), size=n, replace=False)
        selected.extend(population[int(i)] for i in selection)
    result = sorted(selected)
    if len(result) != len(targeted) or Counter(map(key, result)) != budget:
        raise RuntimeError("Matched mask does not meet exact stratum budgets")
    return result


def construct_masks(scores, pairs, counts, seed, k=None):
    pairs = sorted(set(tuple(map(int, p)) for p in pairs))
    if not pairs or not all(np.isfinite(float(scores[p])) for p in pairs):
        raise ValueError("All eligible calibration scores must be present and finite")
    k = fixed_k(len(pairs)) if k is None else int(k)
    if not 0 <= k <= len(pairs):
        raise ValueError("K outside eligible pair family")
    harmful = sorted(pairs, key=lambda p: (float(scores[p]), p[0], p[1]))[:k]
    beneficial = sorted(pairs, key=lambda p: (-float(scores[p]), p[0], p[1]))[:k]
    random_seed = int(seed) + 710001
    frequency_seed = int(seed) + 820001
    masks = {
        "BASE_C3": [], "TARGETED": sorted(harmful),
        "RANDOM_MATCHED": matched_control(pairs, harmful, counts, random_seed, True),
        "FREQUENCY_MATCHED": matched_control(pairs, harmful, counts, frequency_seed, False),
        "ORACLE_SELECTION": sorted(set(pairs) - set(beneficial)),
    }
    # ORACLE_SELECTION has a fixed retained K, not a suppressed K. Ineligible
    # target rows are untouched. There is no extra suppression-alias training.
    return {
        "seed": int(seed), "eligible_pairs": [list(p) for p in pairs],
        "K": k, "suppressed_pairs": {a: [list(p) for p in v] for a, v in masks.items()},
        "oracle_retained_pairs": [list(p) for p in sorted(beneficial)],
        "aliases": ALIASES, "random_seed": random_seed,
        "frequency_seed": frequency_seed,
        "selection_objective": "maximize sum of K calibration row-local direct margin contributions",
        "tie_break": "source ID, then target ID, ascending",
    }


def keep_mask(num_sources, num_targets, suppressed, device=None):
    result = torch.ones((num_sources, num_targets), dtype=torch.bool, device=device)
    seen = set()
    for source, target in suppressed:
        pair = (int(source), int(target))
        if pair in seen or not (0 <= pair[0] < num_sources and 0 <= pair[1] < num_targets):
            raise ValueError("Duplicate or out-of-range source->target mask")
        seen.add(pair)
        result[pair] = False
    return result


def apply_pair_mask(source_weight, source_bias, keep):
    if source_weight.ndim != 3 or source_bias.shape != source_weight.shape[:2]:
        raise ValueError("Expected [source,target,D] and [source,target]")
    if keep.shape != source_bias.shape or keep.dtype != torch.bool:
        raise ValueError("Boolean [source,target] keep mask required")
    if keep.device != source_weight.device or source_bias.device != source_weight.device:
        raise ValueError("Gradient and mask devices differ")
    if not torch.isfinite(source_weight).all() or not torch.isfinite(source_bias).all():
        raise RuntimeError("Nonfinite direct gradient")
    masked_w = source_weight.masked_fill(~keep.unsqueeze(-1), 0.0)
    masked_b = source_bias.masked_fill(~keep, 0.0)
    if torch.count_nonzero(masked_w[~keep]) or torch.count_nonzero(masked_b[~keep]):
        raise RuntimeError("Suppressed pair contribution is nonzero")
    if not torch.equal(masked_w[keep], source_weight[keep]) or not torch.equal(masked_b[keep], source_bias[keep]):
        raise RuntimeError("Mask changed a retained component")
    return masked_w, masked_b


def masked_optimizer_gradient(clean_w, clean_b, source_w, source_b, keep):
    """Keep the actual clean-autograd path, add retained analytic direct terms."""
    masked_w, masked_b = apply_pair_mask(source_w, source_b, keep)
    if clean_w.shape != source_w.shape[1:] or clean_b.shape != source_b.shape[1:]:
        raise ValueError("Clean gradient dimensions differ from target classifier")
    if not torch.isfinite(clean_w).all() or not torch.isfinite(clean_b).all():
        raise RuntimeError("Nonfinite clean gradient")
    return clean_w + masked_w.sum(0), clean_b + masked_b.sum(0)


def calibration_competitors(logits, labels, counts):
    """One non-Many class per represented Many class, chosen on CAL only.

    Maximize class-mean initial CAL logit. Applying this fixed class map to
    Evaluation never chooses an Evaluation-specific competing class.
    """
    labels = np.asarray(labels, dtype=np.int64)
    values = np.asarray(logits)
    counts = np.asarray(counts)
    if values.shape != (len(labels), len(counts)) or not np.isfinite(values).all():
        raise ValueError("Invalid calibration logits")
    nonmany = np.flatnonzero(counts <= 100)
    if len(nonmany) == 0:
        raise ValueError("No non-Many competitor")
    result = {}
    for cls in np.intersect1d(np.unique(labels), np.flatnonzero(counts > 100)):
        mean = values[labels == cls].mean(axis=0)
        result[int(cls)] = int(nonmany[np.argmax(mean[nonmany])])
    if not result:
        raise ValueError("No represented Many class")
    return result


def balance_diagnostics(mask_bundle, counts):
    targeted = [tuple(p) for p in mask_bundle["suppressed_pairs"]["TARGETED"]]
    rows = []
    for arm in ("TARGETED", "RANDOM_MATCHED", "FREQUENCY_MATCHED"):
        pairs = [tuple(p) for p in mask_bundle["suppressed_pairs"][arm]]
        for axis, position in (("source", 0), ("target", 1)):
            ref = np.asarray([counts[p[position]] for p in targeted], dtype=float)
            val = np.asarray([counts[p[position]] for p in pairs], dtype=float)
            pooled = np.sqrt((ref.var() + val.var()) / 2) if len(val) else 0.0
            difference = float(val.mean() - ref.mean()) if len(val) else 0.0
            smd = difference / pooled if pooled else (0.0 if difference == 0 else None)
            rows.append({"arm": arm, "axis": axis, "n": len(pairs),
                         "mean_train_count": float(val.mean()) if len(val) else None,
                         "standardized_mean_difference": smd})
        keys = Counter((source_bin(counts[s]), target_bin(counts[t])) for s, t in pairs)
        reference = Counter((source_bin(counts[s]), target_bin(counts[t])) for s, t in targeted)
        if keys != reference:
            raise RuntimeError("Control failed source x target TRAIN-frequency matching")
        if arm == "RANDOM_MATCHED":
            if Counter((source_bin(counts[s]), t) for s, t in pairs) != Counter((source_bin(counts[s]), t) for s, t in targeted):
                raise RuntimeError("RANDOM_MATCHED target-class counts differ")
        for (source, target), n in sorted(keys.items()):
            rows.append({"arm": arm, "axis": "joint_bin", "source_bin": source,
                         "target_bin": target, "n": n})
    return rows
