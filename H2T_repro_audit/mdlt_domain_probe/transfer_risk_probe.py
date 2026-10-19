"""Validation-only H2T feature-transfer risk probe.

Reuses the completed OfficeHome fixed-feature pilot. No model is trained, no test
features are loaded, and no hyperparameter search is performed.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import traceback

import numpy as np
import torch
import torch.nn.functional as F

from officehome_data import CSV_SHA256, DOMAINS, N_CLASSES, SEEDS, read_protocol, sha256, split_rows, write_csv, write_json
from officehome_domain_probe import HERE

CANDIDATES_PER_RECEIVER = 4
FEATURE_DIM = 512
REPLACED_CHANNELS = 128
BATCH_SIZE = 128
BOOTSTRAP_REPS = 500
ARMS = ("H2T-Global", "H2T-DomainMatched", "H2T-CrossDomain")
PREDICTORS = (
    "directional_predicted_delta_ce",
    "masked_relative_l2",
    "masked_cosine_distance",
    "donor_label_mismatch",
    "baseline_ce",
    "baseline_negative_margin",
    "domain_mismatch",
)


def average_ranks(values: np.ndarray) -> np.ndarray:
    """One-based average ranks with deterministic tie handling; no SciPy required."""
    x = np.asarray(values, dtype=np.float64).ravel()
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return ranks


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """AUROC for a score where larger means more likely harmful; None if undefined."""
    y = np.asarray(labels, dtype=bool).ravel()
    s = np.asarray(scores, dtype=np.float64).ravel()
    if len(y) != len(s) or len(y) == 0 or y.all() or (~y).all():
        return None
    if not np.isfinite(s).all() or np.unique(s).size < 2:
        return None
    ranks = average_ranks(s)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float | None:
    """Spearman correlation with average ranks and safe handling of constant arrays."""
    rx, ry = average_ranks(x), average_ranks(y)
    if len(rx) < 2 or np.std(rx) == 0 or np.std(ry) == 0:
        return None
    value = float(np.corrcoef(rx, ry)[0, 1])
    return value if math.isfinite(value) else None


def target_margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """True-label logit minus the strongest non-target logit."""
    row = torch.arange(len(labels), device=logits.device)
    target = logits[row, labels]
    other = logits.clone()
    other[row, labels] = -torch.inf
    return target - other.max(dim=1).values


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _assert_array(name: str, value: np.ndarray, shape_tail: tuple[int, ...] | None = None) -> None:
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains NaN/Inf")
    if shape_tail is not None and tuple(value.shape[1:]) != shape_tail:
        raise ValueError(f"{name} has shape {value.shape}, expected (*,{shape_tail})")


def load_seed_inputs(run_dir: Path, seed: int) -> dict:
    seed_dir = run_dir / f"seed{seed}"
    manifest = json.loads((seed_dir / "feature_manifest.json").read_text(encoding="utf-8"))
    loaded = {}
    for split in ("train", "val"):
        entry = manifest[split]
        path = (seed_dir / entry["path"]).resolve()
        if not path.is_relative_to(seed_dir.resolve()):
            raise ValueError(f"Unsafe feature path in manifest: {entry['path']}")
        if not path.is_file() or sha256(path) != entry["sha256"]:
            raise ValueError(f"{split} feature cache missing or SHA256 mismatch for seed {seed}")
        data = _load_npz(path)
        for key in ("features", "labels", "domains", "raw_indices"):
            if key not in data:
                raise ValueError(f"{path} lacks {key}")
        features = data["features"].astype(np.float32, copy=False)
        labels = data["labels"].astype(np.int64, copy=False)
        domains = data["domains"].astype(np.int64, copy=False)
        raw_indices = data["raw_indices"].astype(np.int64, copy=False)
        _assert_array(f"seed{seed}/{split}/features", features, (FEATURE_DIM,))
        if len(features) != len(labels) or len(features) != len(domains) or len(features) != len(raw_indices):
            raise ValueError(f"Inconsistent {split} cache lengths for seed {seed}")
        if ((labels < 0) | (labels >= N_CLASSES)).any() or ((domains < 0) | (domains >= len(DOMAINS))).any():
            raise ValueError(f"Invalid labels/domains in seed {seed} {split} cache")
        if len(np.unique(raw_indices)) != len(raw_indices):
            raise ValueError(f"Duplicate raw indices in seed {seed} {split} cache")
        loaded[split] = {
            "features": features, "labels": labels, "domains": domains, "raw_indices": raw_indices
        }

    plan_path = seed_dir / "output" / "sampling_plan.npz"
    plan_audit = json.loads((seed_dir / "sampling_plan_audit.json").read_text(encoding="utf-8"))
    if not plan_path.is_file() or sha256(plan_path) != plan_audit["plan_sha256"]:
        raise ValueError(f"Sampling plan missing or SHA256 mismatch for seed {seed}")
    plan = _load_npz(plan_path)
    required = ("global", "masks", "train_csv_indices")
    if any(key not in plan for key in required):
        raise ValueError(f"Sampling plan incomplete for seed {seed}")
    if not np.array_equal(plan["train_csv_indices"], loaded["train"]["raw_indices"]):
        raise ValueError(f"Sampling plan/train feature order mismatch for seed {seed}")
    if plan["global"].ndim != 2 or plan["masks"].ndim != 3:
        raise ValueError(f"Unexpected sampling plan dimensions for seed {seed}")

    stage1 = json.loads((seed_dir / "stage1_summary.json").read_text(encoding="utf-8"))
    checkpoint_path = seed_dir / "output" / "H2T-Global_epoch10.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing fixed epoch-10 H2T-Global classifier for seed {seed}")
    checkpoint_sha256 = sha256(checkpoint_path)
    pilot_state = json.loads((run_dir / "state_audit.json").read_text(encoding="utf-8"))
    state_record = next((r for r in pilot_state if r.get("seed") == seed and r.get("arm") == "H2T-Global"), None)
    if state_record is None or state_record.get("sha256") != checkpoint_sha256:
        raise ValueError(f"Epoch-10 H2T-Global checkpoint SHA256 mismatch for seed {seed}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("seed") != seed or checkpoint.get("arm") != "H2T-Global" or checkpoint.get("epoch") != 10:
        raise ValueError(f"Unexpected classifier checkpoint metadata for seed {seed}")
    if checkpoint.get("stage1_sha256") != stage1.get("best_sha256"):
        raise ValueError(f"Stage-I checkpoint provenance mismatch for seed {seed}")
    head = torch.nn.Linear(FEATURE_DIM, N_CLASSES)
    head.load_state_dict(checkpoint["classifier"])
    head.eval()
    return {"seed_dir": seed_dir, "train": loaded["train"], "val": loaded["val"],
            "plan": plan, "head": head, "checkpoint_sha256": checkpoint_sha256,
            "feature_sha256": {split: manifest[split]["sha256"] for split in ("train", "val")},
            "plan_sha256": plan_audit["plan_sha256"], "stage1_sha256": stage1["best_sha256"]}


def make_candidate_donors(inputs: dict, seed: int) -> dict[str, np.ndarray]:
    """Paired candidates: same receiver, donor class and mask across all three arms."""
    train, val, plan = inputs["train"], inputs["val"], inputs["plan"]
    n_val = len(val["labels"])
    needed = n_val * CANDIDATES_PER_RECEIVER
    global_plan = plan["global"]
    masks = plan["masks"]
    if global_plan.shape[0] < 1 or global_plan.shape[1] < needed:
        raise ValueError(f"Seed {seed}: global plan too short for {needed} candidates")
    n_batches = math.ceil(n_val / BATCH_SIZE)
    if masks.shape[0] < 1 or masks.shape[1] < n_batches:
        raise ValueError(f"Seed {seed}: shared-mask plan too short for validation batches")

    global_donors = global_plan[0, :needed].reshape(n_val, CANDIDATES_PER_RECEIVER).astype(np.int64)
    train_labels, train_domains = train["labels"], train["domains"]
    val_domains = val["domains"]
    donor_classes = train_labels[global_donors]
    if ((global_donors < 0) | (global_donors >= len(train_labels))).any():
        raise ValueError(f"Seed {seed}: donor index outside train feature cache")

    matched_rng = np.random.default_rng(np.random.SeedSequence([seed, 901]))
    cross_rng = np.random.default_rng(np.random.SeedSequence([seed, 902]))
    matched = np.empty_like(global_donors)
    cross = np.empty_like(global_donors)
    matched_fallback = np.zeros_like(global_donors, dtype=bool)
    cross_fallback = np.zeros_like(global_donors, dtype=bool)
    pool_cache: dict[tuple[int, int, str], np.ndarray] = {}

    def pool_for(cls: int, domain: int, mode: str) -> np.ndarray:
        key = (cls, domain, mode)
        if key not in pool_cache:
            if mode == "matched":
                mask = (train_labels == cls) & (train_domains == domain)
            else:
                mask = (train_labels == cls) & (train_domains != domain)
            pool_cache[key] = np.flatnonzero(mask)
        return pool_cache[key]

    for i in range(n_val):
        recipient_domain = int(val_domains[i])
        for j in range(CANDIDATES_PER_RECEIVER):
            donor_class = int(donor_classes[i, j])
            same_pool = pool_for(donor_class, recipient_domain, "matched")
            other_pool = pool_for(donor_class, recipient_domain, "cross")
            if len(same_pool):
                matched[i, j] = int(matched_rng.choice(same_pool))
            else:
                matched[i, j] = global_donors[i, j]
                matched_fallback[i, j] = True
            if len(other_pool):
                cross[i, j] = int(cross_rng.choice(other_pool))
            else:
                cross[i, j] = global_donors[i, j]
                cross_fallback[i, j] = True

    for candidate in (global_donors, matched, cross):
        if not np.array_equal(train_labels[candidate], donor_classes):
            raise AssertionError(f"Seed {seed}: donor class changed across paired arms")
    recipient_domain_grid = np.repeat(val_domains[:, None], CANDIDATES_PER_RECEIVER, axis=1)
    if not np.all(train_domains[matched[~matched_fallback]] == recipient_domain_grid[~matched_fallback]):
        raise AssertionError("Matched donor domain constraint failed")
    if not np.all(train_domains[cross[~cross_fallback]] != recipient_domain_grid[~cross_fallback]):
        raise AssertionError("Cross-domain donor constraint failed")
    return {
        "H2T-Global": global_donors,
        "H2T-DomainMatched": matched,
        "H2T-CrossDomain": cross,
        "H2T-DomainMatched_fallback": matched_fallback,
        "H2T-CrossDomain_fallback": cross_fallback,
        "donor_classes": donor_classes,
        "mask_plan": masks[0, :n_batches],
    }


def build_receiver_masks(mask_plan: np.ndarray, n_receivers: int) -> np.ndarray:
    masks = np.zeros((n_receivers, FEATURE_DIM), dtype=bool)
    for i in range(n_receivers):
        channels = mask_plan[i // BATCH_SIZE]
        if len(np.unique(channels)) != REPLACED_CHANNELS or channels.min() < 0 or channels.max() >= FEATURE_DIM:
            raise ValueError("Invalid channel mask in sampling plan")
        masks[i, channels] = True
    if not np.all(masks.sum(axis=1) == REPLACED_CHANNELS):
        raise AssertionError("Each receiver must replace exactly 128 channels")
    return masks


def evaluate_arm(inputs: dict, seed: int, arm: str, donor_indices: np.ndarray,
                 fallback: np.ndarray, receiver_masks: np.ndarray) -> dict:
    train, val, head = inputs["train"], inputs["val"], inputs["head"]
    n_val = len(val["labels"])
    k = CANDIDATES_PER_RECEIVER
    donor_flat = donor_indices.reshape(-1)
    receiver_flat = np.repeat(val["features"], k, axis=0)
    receiver_mask_flat = np.repeat(receiver_masks, k, axis=0)
    donor_flat_features = train["features"][donor_flat]
    delta = (donor_flat_features - receiver_flat) * receiver_mask_flat.astype(np.float32)
    fused = receiver_flat + delta

    x0 = torch.from_numpy(val["features"])
    y0 = torch.from_numpy(val["labels"])
    x1 = torch.from_numpy(fused)
    y1 = torch.from_numpy(np.repeat(val["labels"], k))
    with torch.no_grad():
        weight = head.weight.detach().float()
        bias = head.bias.detach().float() if head.bias is not None else None
        logits0 = F.linear(x0, weight, bias)
        logits1 = F.linear(x1, weight, bias)
        base_ce = F.cross_entropy(logits0, y0, reduction="none")
        fused_ce = F.cross_entropy(logits1, y1, reduction="none").reshape(n_val, k)
        base_margin = target_margin(logits0, y0)
        fused_margin = target_margin(logits1, y1).reshape(n_val, k)
        base_pred = logits0.argmax(dim=1).numpy()
        fused_pred = logits1.argmax(dim=1).numpy().reshape(n_val, k)

        probabilities = logits0.softmax(dim=1)
        probabilities[torch.arange(n_val), y0] -= 1.0
        feature_gradient = probabilities @ weight
        directional_delta_ce = (
            feature_gradient.repeat_interleave(k, dim=0) * torch.from_numpy(delta)
        ).sum(dim=1).numpy().reshape(n_val, k)

    receiver_masked = receiver_flat * receiver_mask_flat.astype(np.float32)
    donor_masked = donor_flat_features * receiver_mask_flat.astype(np.float32)
    delta_norm = np.linalg.norm(delta, axis=1)
    receiver_mask_norm = np.linalg.norm(receiver_masked, axis=1)
    donor_norm = np.linalg.norm(donor_masked, axis=1)
    cosine = np.sum(receiver_masked * donor_masked, axis=1) / (
        receiver_mask_norm * donor_norm + 1e-8
    )
    train_labels, train_domains = train["labels"], train["domains"]
    donor_labels = train_labels[donor_flat].reshape(n_val, k)
    donor_domains = train_domains[donor_flat].reshape(n_val, k)
    recipient_labels = np.repeat(val["labels"][:, None], k, axis=1)
    recipient_domains = np.repeat(val["domains"][:, None], k, axis=1)
    base_ce_np = base_ce.numpy()
    base_margin_np = base_margin.numpy()
    fused_ce_np = fused_ce.numpy()
    fused_margin_np = fused_margin.numpy()
    base_correct = (base_pred == val["labels"])[:, None]
    fused_correct = fused_pred == recipient_labels
    delta_ce = fused_ce_np - base_ce_np[:, None]
    delta_margin = fused_margin_np - base_margin_np[:, None]
    predictors = {
        "directional_predicted_delta_ce": directional_delta_ce,
        "masked_relative_l2": (delta_norm / (receiver_mask_norm + 1e-8)).reshape(n_val, k),
        "masked_cosine_distance": (1.0 - cosine).reshape(n_val, k),
        "donor_label_mismatch": (donor_labels != recipient_labels).astype(np.float64),
        "baseline_ce": np.repeat(base_ce_np[:, None], k, axis=1),
        "baseline_negative_margin": np.repeat((-base_margin_np)[:, None], k, axis=1),
        "domain_mismatch": (donor_domains != recipient_domains).astype(np.float64),
    }
    return {
        "delta_ce": delta_ce,
        "delta_margin": delta_margin,
        "harm": delta_ce > 1e-8,
        "margin_harm": delta_margin < -1e-8,
        "base_correct": np.broadcast_to(base_correct, (n_val, k)),
        "fused_correct": fused_correct,
        "flip_to_wrong": base_correct & (~fused_correct),
        "wrong_to_correct": (~base_correct) & fused_correct,
        "predictors": predictors,
        "donor_indices": donor_indices,
        "donor_labels": donor_labels,
        "donor_domains": donor_domains,
        "fallback": fallback,
    }


def summarize_arm(seed: int, arm: str, result: dict, val: dict) -> dict:
    n = result["delta_ce"].size
    return {
        "seed": seed, "arm": arm, "n_candidate_substitutions": n,
        "mean_delta_ce_nats": float(result["delta_ce"].mean()),
        "median_delta_ce_nats": float(np.median(result["delta_ce"])),
        "harm_rate_delta_ce_positive": float(result["harm"].mean()),
        "mean_delta_target_margin": float(result["delta_margin"].mean()),
        "margin_harm_rate": float(result["margin_harm"].mean()),
        "base_accuracy_on_repeated_candidates": float(result["base_correct"].mean()),
        "post_substitution_accuracy": float(result["fused_correct"].mean()),
        "flip_to_wrong_rate": float(result["flip_to_wrong"].mean()),
        "wrong_to_correct_rate": float(result["wrong_to_correct"].mean()),
        "donor_label_mismatch_rate": float(result["predictors"]["donor_label_mismatch"].mean()),
        "actual_domain_mismatch_rate": float(result["predictors"]["domain_mismatch"].mean()),
        "fallback_rate": float(result["fallback"].mean()),
        "mean_masked_relative_l2": float(result["predictors"]["masked_relative_l2"].mean()),
        "mean_masked_cosine_distance": float(result["predictors"]["masked_cosine_distance"].mean()),
        "validation_images": len(val["labels"]),
        "test_used": False,
    }


def predictor_metrics(seed: int, arm: str, result: dict) -> list[dict]:
    harm = result["harm"].ravel()
    outcome = result["delta_ce"].ravel()
    rows = []
    for predictor in PREDICTORS:
        score = result["predictors"][predictor].ravel()
        rows.append({
            "seed": seed, "arm": arm, "predictor": predictor, "n": len(score),
            "harm_prevalence": float(harm.mean()),
            "auroc_harm": binary_auc(harm, score),
            "spearman_with_delta_ce": spearman_rho(score, outcome),
            "score_unique_values": int(np.unique(score).size),
            "test_used": False,
        })
    return rows


def stratified_bootstrap_ci(image_values: np.ndarray, labels: np.ndarray, domains: np.ndarray,
                            seed: int, reps: int = BOOTSTRAP_REPS) -> tuple[float, float]:
    """Resample validation images within each fixed domain-class cell."""
    cells = []
    for d in range(len(DOMAINS)):
        for c in range(N_CLASSES):
            idx = np.flatnonzero((domains == d) & (labels == c))
            if len(idx) == 0:
                raise ValueError(f"Empty validation cell domain={DOMAINS[d]}, class={c}")
            cells.append(idx)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1701]))
    estimates = np.empty(reps, dtype=np.float64)
    for b in range(reps):
        sampled = np.concatenate([rng.choice(idx, size=len(idx), replace=True) for idx in cells])
        estimates[b] = float(image_values[sampled].mean())
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def build_candidate_rows(seed_outputs: dict, val_reference: dict) -> list[dict]:
    rows = []
    raw_indices = val_reference["raw_indices"]
    val_labels = val_reference["labels"]
    val_domains = val_reference["domains"]
    for seed in SEEDS:
        seed_data = seed_outputs[seed]
        for arm in ARMS:
            result = seed_data["arms"][arm]
            for i in range(len(val_labels)):
                for j in range(CANDIDATES_PER_RECEIVER):
                    donor_index = int(result["donor_indices"][i, j])
                    rows.append({
                        "seed": seed, "arm": arm, "val_position": i,
                        "val_raw_index": int(raw_indices[i]), "candidate_id": j,
                        "receiver_domain": DOMAINS[int(val_domains[i])],
                        "receiver_label": int(val_labels[i]),
                        "donor_raw_index": int(seed_data["train"]["raw_indices"][donor_index]),
                        "donor_domain": DOMAINS[int(result["donor_domains"][i, j])],
                        "donor_label": int(result["donor_labels"][i, j]),
                        "donor_label_mismatch": int(result["predictors"]["donor_label_mismatch"][i, j]),
                        "domain_mismatch": int(result["predictors"]["domain_mismatch"][i, j]),
                        "conditional_fallback": int(result["fallback"][i, j]),
                        "directional_predicted_delta_ce": float(result["predictors"]["directional_predicted_delta_ce"][i, j]),
                        "masked_relative_l2": float(result["predictors"]["masked_relative_l2"][i, j]),
                        "masked_cosine_distance": float(result["predictors"]["masked_cosine_distance"][i, j]),
                        "baseline_ce": float(result["predictors"]["baseline_ce"][i, j]),
                        "baseline_negative_margin": float(result["predictors"]["baseline_negative_margin"][i, j]),
                        "delta_ce_nats": float(result["delta_ce"][i, j]),
                        "harm_delta_ce_positive": int(result["harm"][i, j]),
                        "delta_target_margin": float(result["delta_margin"][i, j]),
                        "margin_harm": int(result["margin_harm"][i, j]),
                        "base_correct": int(result["base_correct"][i, j]),
                        "post_substitution_correct": int(result["fused_correct"][i, j]),
                        "flip_to_wrong": int(result["flip_to_wrong"][i, j]),
                        "wrong_to_correct": int(result["wrong_to_correct"][i, j]),
                    })
    return rows


def write_report(out: Path, arm_rows: list[dict], predictor_rows: list[dict],
                 gate: dict, paired_ci: dict) -> None:
    lines = [
        "# OfficeHome H2T transfer-risk probe",
        "",
        "## Purpose and boundaries",
        "",
        "This is a validation-only, fixed-feature mechanism probe. It reuses the completed pilot's",
        "cached train/validation GAP features, the saved epoch-10 H2T-Global linear classifier,",
        "and the existing donor-class/mask plan. It trains no model, reads no test feature cache,",
        "and performs no hyperparameter search.",
        "",
        "Each validation receiver gets four paired donor-class candidates. Across Global,",
        "DomainMatched and CrossDomain, the receiver, donor class and 128-channel batch-shared mask",
        "are held fixed; donor identity necessarily changes with the domain constraint.",
        "",
        "## Outcome and predictors",
        "",
        "- Primary outcome: actual change in receiver-label cross-entropy after substitution,",
        "  delta_CE = CE(fused) - CE(original); positive means harmful under the fixed classifier.",
        "- Secondary outcome: change in true-label logit margin.",
        "- Directional score: first-order Taylor estimate grad_f CE(original) dot masked(donor - receiver).",
        "- Non-directional baselines: masked relative L2, masked cosine distance, donor-label mismatch,",
        "  baseline CE / negative margin, and domain mismatch.",
        "- AUROC and Spearman are descriptive prediction diagnostics, not significance tests.",
        "",
        "## Per-arm outcome summary (mean over three seeds)",
        "",
        "| Arm | ΔCE (nats) | Harm rate | Δmargin | Flip-to-wrong | Wrong-to-correct | Domain mismatch |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        subset = [r for r in arm_rows if r["arm"] == arm]
        mean = lambda key: float(np.mean([r[key] for r in subset]))
        lines.append(
            f"| {arm} | {mean('mean_delta_ce_nats'):.5f} | {mean('harm_rate_delta_ce_positive'):.4f} "
            f"| {mean('mean_delta_target_margin'):.5f} | {mean('flip_to_wrong_rate'):.4f} "
            f"| {mean('wrong_to_correct_rate'):.4f} | {mean('actual_domain_mismatch_rate'):.4f} |"
        )
    lines += [
        "",
        "## Directional risk prediction",
        "",
        "AUROC asks whether the score ranks substitutions with positive ΔCE above substitutions with",
        "non-positive ΔCE. Higher scores always mean greater predicted risk. A domain-mismatch score",
        "has no within-arm variation when the arm enforces a domain constraint; its AUROC is reported",
        "as undefined rather than interpreted as evidence for or against domain mismatch.",
        "",
        "| Seed | Arm | Directional ΔCE AUROC | Relative-L2 AUROC | Cosine-distance AUROC | Label-mismatch AUROC | Domain-mismatch AUROC |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    lookup = {(r["seed"], r["arm"], r["predictor"]): r for r in predictor_rows}
    for seed in SEEDS:
        for arm in ARMS:
            vals = [lookup[(seed, arm, p)]["auroc_harm"] for p in (
                "directional_predicted_delta_ce", "masked_relative_l2", "masked_cosine_distance",
                "donor_label_mismatch", "domain_mismatch")]
            fmt = lambda x: "NA" if x is None else f"{x:.3f}"
            lines.append(f"| {seed} | {arm} | " + " | ".join(fmt(x) for x in vals) + " |")
    lines += [
        "",
        "## Paired CrossDomain − DomainMatched contrast",
        "",
        "Positive ΔCE difference means CrossDomain substitutions increase loss more on average.",
        "The bootstrap resamples validation images within each domain-class cell and averages each",
        "image's four candidates across the three fixed seed checkpoints. The interval is conditional",
        "on these trained checkpoints; it is not a confidence interval over future training seeds.",
        "",
        f"- Mean paired ΔCE difference: {paired_ci['delta_ce']['mean']:.6f} nats "
        f"(stratified image-bootstrap 95% interval {paired_ci['delta_ce']['ci95'][0]:.6f}, "
        f"{paired_ci['delta_ce']['ci95'][1]:.6f}).",
        f"- Mean paired Δmargin difference: {paired_ci['delta_margin']['mean']:.6f} "
        f"(95% interval {paired_ci['delta_margin']['ci95'][0]:.6f}, "
        f"{paired_ci['delta_margin']['ci95'][1]:.6f}).",
        f"- Harm-rate difference: {paired_ci['harm_rate']['mean']:.6f} "
        f"(95% interval {paired_ci['harm_rate']['ci95'][0]:.6f}, "
        f"{paired_ci['harm_rate']['ci95'][1]:.6f}).",
        "",
        "Per-seed paired contrasts are in paired_cross_minus_matched.csv; every candidate pair shares",
        "the same validation receiver, donor class and mask, but not the same donor image.",
        "",
        "## Predeclared exploratory go/no-go gate",
        "",
        "A minimal directional-risk module is worth prototyping only if, in at least two of three arms,",
        "the directional score has AUROC >= 0.60 and exceeds the masked-relative-L2 AUROC by >= 0.05",
        "in at least two of the three seeds. This is a practical screening gate, not a statistical",
        "significance threshold. If it fails, do not train a new module on the strength of this probe.",
        "",
        f"- Gate result: **{'PASS' if gate['pass'] else 'FAIL'}**.",
        f"- Arms meeting the gate: {', '.join(gate['passing_arms']) if gate['passing_arms'] else 'none'}.",
        "",
        "## Interpretation limits",
        "",
        "- This measures one-step substitution effects under the fixed epoch-10 H2T-Global linear head.",
        "  It does not establish end-to-end representation-learning effects or prove a new method works.",
        "- Donor domain and donor identity are not separable in the conditional arms; domain effects remain",
        "  a paired mechanism signal, not a perfectly isolated causal effect.",
        "- Validation labels are used to evaluate the supervised training-time risk signal. Do not tune",
        "  thresholds on this same validation set and then report these results as independent confirmation.",
        "- No test images/pixels, cached test features, test predictions, or test metrics were read; only pinned split metadata was parsed to verify train/validation row alignment.",
        "- If the gate passes, the next experiment should be one minimal risk-gated intervention with",
        "  thresholds fixed on validation, followed by independent confirmation. If it fails, stop this",
        "  branch rather than adding more heuristics.",
        "",
        "## Artifacts",
        "",
        "- candidate_level.csv: all receiver/candidate/arm outcomes and predictors.",
        "- per_seed_arm_summary.csv: substitution outcome summaries.",
        "- risk_predictor_summary.csv: AUROC and Spearman by seed, arm and score.",
        "- paired_cross_minus_matched.csv: paired domain-condition contrasts.",
        "- summary.json: provenance, gate and paired bootstrap intervals.",
        "",
    ]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run_probe(run_dir: Path, out: Path) -> dict:
    run_dir = run_dir.resolve()
    out = out.resolve()
    if not run_dir.is_relative_to(HERE.resolve()) or not out.is_relative_to((HERE / "results").resolve()):
        raise ValueError("Pilot input must be inside this experiment directory and output inside results/")
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {out}")
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if status.get("status") != "COMPLETE":
        raise ValueError(f"Pilot is not complete: {status.get('status')}")
    if config.get("csv_sha256") != CSV_SHA256:
        raise ValueError("Pilot split provenance mismatch")
    audit = json.loads((run_dir / "dataset_audit.json").read_text(encoding="utf-8"))
    if not audit.get("ready") or audit.get("csv_sha256") != CSV_SHA256:
        raise ValueError("Pilot dataset audit did not pass")
    if len(json.loads((run_dir / "state_audit.json").read_text(encoding="utf-8"))) != len(SEEDS) * 4:
        raise ValueError("Pilot state audit does not contain all 12 seed x arm records")

    # Metadata-only protocol validation: compare cached row order/labels/domains to the pinned CSV.
    protocol_rows = read_protocol(HERE / "protocol" / "OfficeHome.csv")
    expected_rows = {split: split_rows(protocol_rows, split) for split in ("train", "val")}
    for split in ("train", "val"):
        if not expected_rows[split]:
            raise ValueError(f"Pinned protocol has no {split} rows")
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    seed_outputs = {}
    arm_summaries, predictor_summaries = [], []
    val_reference = None
    source_hashes = {
        "officehome_data.py": sha256(HERE / "officehome_data.py"),
        "officehome_domain_probe.py": sha256(HERE / "officehome_domain_probe.py"),
        "transfer_risk_probe.py": sha256(HERE / "transfer_risk_probe.py"),
    }
    input_manifest = {
        "pilot_run_dir": str(run_dir),
        "pilot_config_sha256": sha256(run_dir / "config.json"),
        "pilot_status_sha256": sha256(run_dir / "status.json"),
        "csv_sha256": CSV_SHA256,
        "seeds": list(SEEDS),
        "candidates_per_receiver": CANDIDATES_PER_RECEIVER,
        "mask": "existing epoch-1 128-of-512 masks; shared across 128 validation receivers and all candidate arms",
        "donor_class_plan": "first 4 x N_val global donor draws from existing seed-specific Stage-II plan",
        "classifier": "fixed epoch-10 H2T-Global head per seed",
        "outcome": "actual delta cross-entropy; positive means harmful",
        "test_used": False,
        "source_sha256": source_hashes,
        "pinned_protocol_metadata_only_check": True,
    }
    write_json(out / "config.json", input_manifest)

    for seed in SEEDS:
        inputs = load_seed_inputs(run_dir, seed)
        for split in ("train", "val"):
            cache = inputs[split]
            protocol_split = expected_rows[split]
            if not np.array_equal(cache["raw_indices"], np.array([r.index for r in protocol_split], dtype=np.int64)):
                raise ValueError(f"Seed {seed} {split} cache raw-index order differs from the pinned protocol")
            if not np.array_equal(cache["labels"], np.array([r.label for r in protocol_split], dtype=np.int64)):
                raise ValueError(f"Seed {seed} {split} cache labels differ from the pinned protocol")
            if not np.array_equal(cache["domains"], np.array([r.domain for r in protocol_split], dtype=np.int64)):
                raise ValueError(f"Seed {seed} {split} cache domains differ from the pinned protocol")
        val = inputs["val"]
        if val_reference is None:
            val_reference = val
        else:
            for key in ("raw_indices", "labels", "domains"):
                if not np.array_equal(val_reference[key], val[key]):
                    raise ValueError(f"Validation {key} differs across seeds")
        donors = make_candidate_donors(inputs, seed)
        receiver_masks = build_receiver_masks(donors["mask_plan"], len(val["labels"]))
        arms = {}
        for arm in ARMS:
            fallback = (donors["H2T-DomainMatched_fallback"] if arm == "H2T-DomainMatched"
                        else donors["H2T-CrossDomain_fallback"] if arm == "H2T-CrossDomain"
                        else np.zeros_like(donors["H2T-Global"], dtype=bool))
            arms[arm] = evaluate_arm(inputs, seed, arm, donors[arm], fallback, receiver_masks)
            arm_summaries.append(summarize_arm(seed, arm, arms[arm], val))
            predictor_summaries.extend(predictor_metrics(seed, arm, arms[arm]))
        seed_outputs[seed] = {"arms": arms, "train": inputs["train"],
                              "stage1_sha256": inputs["stage1_sha256"],
                              "classifier_sha256": inputs["checkpoint_sha256"],
                              "feature_sha256": inputs["feature_sha256"],
                              "plan_sha256": inputs["plan_sha256"],
                              "donor_classes": donors["donor_classes"]}
        print(f"Risk probe seed={seed}: evaluated {len(val['labels']) * CANDIDATES_PER_RECEIVER} paired candidate sets", flush=True)

    candidate_rows = build_candidate_rows(seed_outputs, val_reference)
    write_csv(out / "candidate_level.csv", candidate_rows)
    write_csv(out / "per_seed_arm_summary.csv", arm_summaries)
    write_csv(out / "risk_predictor_summary.csv", predictor_summaries)

    paired_rows = []
    cross_ce_by_seed, cross_margin_by_seed, cross_harm_by_seed = [], [], []
    for seed in SEEDS:
        matched = seed_outputs[seed]["arms"]["H2T-DomainMatched"]
        cross = seed_outputs[seed]["arms"]["H2T-CrossDomain"]
        d_ce = cross["delta_ce"] - matched["delta_ce"]
        d_margin = cross["delta_margin"] - matched["delta_margin"]
        d_harm = cross["harm"].astype(np.float64) - matched["harm"].astype(np.float64)
        d_flip = cross["flip_to_wrong"].astype(np.float64) - matched["flip_to_wrong"].astype(np.float64)
        d_recover = cross["wrong_to_correct"].astype(np.float64) - matched["wrong_to_correct"].astype(np.float64)
        cross_ce_by_seed.append(d_ce)
        cross_margin_by_seed.append(d_margin)
        cross_harm_by_seed.append(d_harm)
        paired_rows.append({
            "seed": seed, "mean_cross_minus_matched_delta_ce_nats": float(d_ce.mean()),
            "mean_cross_minus_matched_delta_margin": float(d_margin.mean()),
            "cross_minus_matched_harm_rate": float(d_harm.mean()),
            "cross_minus_matched_flip_to_wrong_rate": float(d_flip.mean()),
            "cross_minus_matched_wrong_to_correct_rate": float(d_recover.mean()),
            "candidate_pairs": int(d_ce.size), "test_used": False,
        })
    write_csv(out / "paired_cross_minus_matched.csv", paired_rows)

    val_labels = val_reference["labels"]
    val_domains = val_reference["domains"]
    def per_image_mean(stack: list[np.ndarray]) -> np.ndarray:
        return np.mean(np.stack(stack, axis=0), axis=(0, 2))
    paired_values = {
        "delta_ce": per_image_mean(cross_ce_by_seed),
        "delta_margin": per_image_mean(cross_margin_by_seed),
        "harm_rate": per_image_mean(cross_harm_by_seed),
    }
    paired_ci = {}
    for key, values in paired_values.items():
        lo, hi = stratified_bootstrap_ci(values, val_labels, val_domains, seed=9100 + len(key))
        paired_ci[key] = {"mean": float(values.mean()), "ci95": [lo, hi],
                          "bootstrap_reps": BOOTSTRAP_REPS,
                          "interpretation": "conditional on fixed trained checkpoints"}
    predictor_lookup = {(r["seed"], r["arm"], r["predictor"]): r for r in predictor_summaries}
    gate_arms = {}
    for arm in ARMS:
        passed_seeds = []
        for seed in SEEDS:
            directional = predictor_lookup[(seed, arm, "directional_predicted_delta_ce")]["auroc_harm"]
            magnitude = predictor_lookup[(seed, arm, "masked_relative_l2")]["auroc_harm"]
            passed_seeds.append(
                directional is not None and magnitude is not None
                and directional >= 0.60 and directional - magnitude >= 0.05
            )
        gate_arms[arm] = {
            "pass": sum(passed_seeds) >= 2,
            "passing_seed_count": int(sum(passed_seeds)),
            "seed_pass_flags": dict(zip(map(str, SEEDS), passed_seeds)),
        }
    passing_arms = [arm for arm, item in gate_arms.items() if item["pass"]]
    gate = {
        "pass": len(passing_arms) >= 2,
        "criterion": "In >=2/3 arms, directional AUROC >=0.60 and >=0.05 above masked-relative-L2 AUROC in >=2/3 seeds",
        "arms": gate_arms,
        "passing_arms": passing_arms,
        "note": "Exploratory screening rule, not a significance threshold.",
    }
    summary = {
        "status": "COMPLETE",
        "test_used": False,
        "models_trained": 0,
        "candidate_sets_per_seed": int(len(val_labels) * CANDIDATES_PER_RECEIVER),
        "candidate_substitutions_total": int(len(val_labels) * CANDIDATES_PER_RECEIVER * len(SEEDS) * len(ARMS)),
        "seeds": list(SEEDS),
        "arms": list(ARMS),
        "paired_cross_minus_matched": paired_ci,
        "directional_risk_gate": gate,
        "source_sha256": source_hashes,
        "input_artifact_sha256": {
            str(seed): {
                "train_features": seed_outputs[seed]["feature_sha256"]["train"],
                "val_features": seed_outputs[seed]["feature_sha256"]["val"],
                "sampling_plan": seed_outputs[seed]["plan_sha256"],
                "h2t_global_epoch10_classifier": seed_outputs[seed]["classifier_sha256"],
                "stage1_best_checkpoint": seed_outputs[seed]["stage1_sha256"],
            } for seed in SEEDS
        },
        "interpretation_guardrail": "All intervals condition on fixed trained checkpoints; three seeds and repeated validation images are not independent image-level replicates.",
    }
    write_json(out / "summary.json", summary)
    write_report(out, arm_summaries, predictor_summaries, gate, paired_ci)
    write_json(out / "status.json", {"status": "COMPLETE", "test_used": False,
                                    "models_trained": 0, "candidate_substitutions_total": summary["candidate_substitutions_total"]})
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Completed pilot directory under this experiment's results/")
    parser.add_argument("--out", type=Path, required=True, help="New output directory under results/; never overwritten")
    args = parser.parse_args()
    try:
        summary = run_probe(args.run_dir, args.out)
        print(json.dumps({
            "status": summary["status"],
            "test_used": summary["test_used"],
            "candidate_substitutions_total": summary["candidate_substitutions_total"],
            "directional_risk_gate": summary["directional_risk_gate"],
            "output": str(args.out.resolve()),
        }, ensure_ascii=False, indent=2), flush=True)
        return 0
    except Exception as exc:
        out = args.out.resolve()
        if out.exists() and out.is_dir():
            write_json(out / "status.json", {"status": "FAILED", "test_used": False,
                                            "error": repr(exc), "traceback": traceback.format_exc()})
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
