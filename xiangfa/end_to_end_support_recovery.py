from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torch.utils.data import DataLoader

from lt_frequency_sweep_experiment import DEFAULT_RANKS, make_sweep_design
from lt_rank_swap_experiment import (
    exponential_counts,
    load_model,
    make_rank_swap_design,
    select_indices,
)
from support_dynamics_experiment import (
    NUM_CLASSES,
    ArrayCIFAR100,
    eval_transform,
    l2_normalize,
    make_nested_class_orders,
    orthogonal_procrustes,
    save_json,
    set_seed,
    write_csv,
)


@dataclass
class FeatureBundle:
    features: np.ndarray
    logits: np.ndarray
    labels: np.ndarray
    indices: np.ndarray


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="End-to-end similar-class support recovery feasibility test."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(r"D:\xiangmuxunlian\hafe_experiment\data"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "outputs" / "end_to_end_recovery",
    )
    parser.add_argument(
        "--rank-root",
        type=Path,
        default=project_dir / "outputs" / "lt_rank_swap",
    )
    parser.add_argument(
        "--sweep-root",
        type=Path,
        default=project_dir / "outputs" / "lt_frequency_sweep",
    )
    parser.add_argument(
        "--balanced-root",
        type=Path,
        default=project_dir / "outputs" / "balanced_neighbor_trajectory",
    )
    parser.add_argument("--class-a", type=int, default=47)
    parser.add_argument("--class-b", type=int, default=52)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    parser.add_argument("--head-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--head-epochs", type=int, default=100)
    parser.add_argument("--head-lr", type=float, default=0.1)
    parser.add_argument("--head-batch-size", type=int, default=2048)
    parser.add_argument("--balanced-per-class", type=int, default=500)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--subset-seed", type=int, default=20260729)
    parser.add_argument("--rank-seed", type=int, default=20260801)
    parser.add_argument("--imbalance-factor", type=float, default=100.0)
    parser.add_argument("--max-count", type=int, default=500)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def read_class_names(data_root: Path) -> list[str]:
    import pickle

    with (data_root / "cifar-100-python" / "meta").open("rb") as handle:
        payload = pickle.load(handle, encoding="latin1")
    return list(
        payload.get("fine_label_names", payload.get(b"fine_label_names"))
    )


@torch.no_grad()
def extract_bundle(
    model: nn.Module,
    raw_dataset,
    indices: np.ndarray | None,
    args: argparse.Namespace,
    device: torch.device,
) -> FeatureBundle:
    targets = np.asarray(raw_dataset.targets, dtype=np.int64)
    dataset = ArrayCIFAR100(
        raw_dataset.data,
        targets,
        indices,
        eval_transform(),
        return_index=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    features, logits, labels, global_indices = [], [], [], []
    model.eval()
    for images, current_labels, current_indices in loader:
        images = images.to(device, non_blocking=True)
        current_logits, current_features = model(images, return_features=True)
        features.append(current_features.float().cpu().numpy())
        logits.append(current_logits.float().cpu().numpy())
        labels.append(current_labels.numpy())
        global_indices.append(current_indices.numpy())
    return FeatureBundle(
        features=np.concatenate(features).astype(np.float32),
        logits=np.concatenate(logits).astype(np.float32),
        labels=np.concatenate(labels).astype(np.int64),
        indices=np.concatenate(global_indices).astype(np.int64),
    )


def endpoint_paths(seed: int, args: argparse.Namespace) -> dict[str, Path]:
    seed_dir = args.rank_root / f"seed_{seed}"
    return {
        "AB": seed_dir / "model_A_head_B_tail_final.pth",
        "BA": seed_dir / "model_B_head_A_tail_final.pth",
    }


def sweep_paths(seed: int, design: dict, args: argparse.Namespace) -> list[Path]:
    endpoints = endpoint_paths(seed, args)
    paths = []
    for state in design["states"]:
        if state["rank_a"] == 1:
            paths.append(endpoints["AB"])
        elif state["rank_a"] == 100:
            paths.append(endpoints["BA"])
        else:
            paths.append(
                args.sweep_root
                / f"seed_{seed}"
                / f"model_{state['name']}_final.pth"
            )
    return paths


def validate_checkpoints(
    seed: int,
    endpoint_design: dict,
    sweep_design: dict,
    args: argparse.Namespace,
) -> dict:
    checks = []
    paths = sweep_paths(seed, sweep_design, args)
    for path, state in zip(paths, sweep_design["states"]):
        if not path.exists():
            raise FileNotFoundError(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        compatible = (
            int(checkpoint.get("seed", -1)) == seed
            and np.array_equal(
                np.asarray(checkpoint.get("counts", []), dtype=np.int64),
                state["counts"],
            )
        )
        if not compatible:
            raise ValueError(f"Incompatible checkpoint: {path}")
        checks.append(
            {
                "path": str(path),
                "rank_a": state["rank_a"],
                "count_a": state["count_a"],
                "count_b": state["count_b"],
                "compatible": compatible,
            }
        )
    if not np.array_equal(
        endpoint_design["counts_ab"], sweep_design["states"][0]["counts"]
    ):
        raise AssertionError("AB endpoint design mismatch.")
    if not np.array_equal(
        endpoint_design["counts_ba"], sweep_design["states"][-1]["counts"]
    ):
        raise AssertionError("BA endpoint design mismatch.")
    return {"seed": seed, "checkpoints": checks}


def support_statistics(features: np.ndarray) -> dict:
    normalized = l2_normalize(features.astype(np.float64))
    prototype = normalized.mean(axis=0)
    prototype /= max(np.linalg.norm(prototype), 1e-12)
    cosines = np.clip(normalized @ prototype, -1.0, 1.0)
    angles = np.arccos(cosines)
    tangent = normalized - cosines[:, None] * prototype[None, :]
    covariance = tangent.T @ tangent / max(len(tangent) - 1, 1)
    covariance = (covariance + covariance.T) * 0.5
    trace = float(np.trace(covariance))
    return {
        "prototype": prototype,
        "covariance": covariance,
        "trace": trace,
        "angle_q50": float(np.quantile(angles, 0.50)),
        "angle_q80": float(np.quantile(angles, 0.80)),
        "angle_q95": float(np.quantile(angles, 0.95)),
    }


def fit_log_trend(shots: np.ndarray, values: np.ndarray) -> dict:
    scarcity = np.log(shots[0] / shots) / math.log(shots[0] / shots[-1])
    log_values = np.log(np.maximum(values, 1e-10))
    slope, intercept = np.polyfit(scarcity, log_values, 1)
    predicted = intercept + slope * scarcity
    denominator = float(np.sum((log_values - log_values.mean()) ** 2))
    r2 = 0.0 if denominator <= 1e-12 else 1.0 - float(
        np.sum((log_values - predicted) ** 2) / denominator
    )
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": r2,
        "head_to_tail_ratio": float(np.exp(-slope)),
        "fitted_values": np.exp(predicted),
    }


def extract_trajectory_statistics(
    seed: int,
    raw_train,
    class_orders: list[np.ndarray],
    design: dict,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    paths = sweep_paths(seed, design, args)
    state_stats = []
    for state, path in zip(design["states"], paths):
        model = load_model(path, device)
        row = {"rank_a": state["rank_a"]}
        for role, class_id, count_name in [
            ("A", args.class_a, "count_a"),
            ("B", args.class_b, "count_b"),
        ]:
            count = int(state[count_name])
            indices = class_orders[class_id][:count]
            bundle = extract_bundle(model, raw_train, indices, args, device)
            row[role] = support_statistics(bundle.features)
        state_stats.append(row)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    shots = np.asarray(design["a_counts_head_to_tail"], dtype=np.float64)
    result = {"shots": shots, "roles": {}}
    for role, source_indices in [
        ("A", list(range(len(state_stats)))),
        ("B", list(reversed(range(len(state_stats))))),
    ]:
        traces = np.asarray(
            [state_stats[index][role]["trace"] for index in source_indices]
        )
        q50 = np.asarray(
            [state_stats[index][role]["angle_q50"] for index in source_indices]
        )
        trace_fit = fit_log_trend(shots, traces)
        q50_fit = fit_log_trend(shots, q50)
        trace_ratio = trace_fit["head_to_tail_ratio"]
        radius_squared_ratio = q50_fit["head_to_tail_ratio"] ** 2
        combined_ratio = float(
            np.clip(math.sqrt(trace_ratio * radius_squared_ratio), 0.25, 4.0)
        )
        result["roles"][role] = {
            "trace": traces,
            "angle_q50": q50,
            "trace_fit": trace_fit,
            "angle_q50_fit": q50_fit,
            "predicted_head_to_tail_trace_ratio": combined_ratio,
        }
    return result


def load_balanced_pair_scores(args: argparse.Namespace) -> dict[int, dict[int, float]]:
    path = args.balanced_root / "seed_42" / "all_class_pairs.csv"
    scores: dict[int, dict[int, float]] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            target = int(row["target_class"])
            candidate = int(row["candidate_class"])
            scores.setdefault(target, {})[candidate] = float(row["prototype"])
    return scores


def choose_far_head(
    target: int,
    donor: int,
    counts: np.ndarray,
    scores: dict[int, dict[int, float]],
) -> int:
    candidates = [
        class_id
        for class_id in range(NUM_CLASSES)
        if class_id not in {target, donor} and int(counts[class_id]) >= 200
    ]
    if not candidates:
        raise ValueError("No far head candidate with at least 200 samples.")
    return min(candidates, key=lambda class_id: scores[target][class_id])


def aligned_counter_features(
    target_bundle: FeatureBundle,
    counter_bundle: FeatureBundle,
    background_classes: np.ndarray,
) -> tuple[np.ndarray, float]:
    target_mask = np.isin(target_bundle.labels, background_classes)
    counter_mask = np.isin(counter_bundle.labels, background_classes)
    target_order = np.argsort(target_bundle.indices[target_mask])
    counter_order = np.argsort(counter_bundle.indices[counter_mask])
    target_indices = target_bundle.indices[target_mask][target_order]
    counter_indices = counter_bundle.indices[counter_mask][counter_order]
    if not np.array_equal(target_indices, counter_indices):
        raise AssertionError("Background training anchors are not identical.")
    target_anchor = target_bundle.features[target_mask][target_order]
    counter_anchor = counter_bundle.features[counter_mask][counter_order]
    rotation = orthogonal_procrustes(counter_anchor, target_anchor)
    aligned = l2_normalize(counter_bundle.features.astype(np.float64) @ rotation)
    target_normalized = l2_normalize(target_anchor.astype(np.float64))
    counter_normalized = aligned[counter_mask][counter_order]
    alignment_cosine = float(
        np.mean(np.sum(target_normalized * counter_normalized, axis=1))
    )
    return aligned, alignment_cosine


def tangent_projected_shape(
    covariance: np.ndarray,
    target_prototype: np.ndarray,
) -> np.ndarray:
    dimension = len(target_prototype)
    projector = np.eye(dimension) - np.outer(target_prototype, target_prototype)
    projected = projector @ covariance @ projector
    projected = (projected + projected.T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(projected)
    projected = (eigenvectors * np.clip(eigenvalues, 0.0, None)) @ eigenvectors.T
    trace = float(np.trace(projected))
    if trace <= 1e-12:
        return projector / max(dimension - 1, 1)
    return projected / trace


def generate_features(
    prototype: np.ndarray,
    covariance: np.ndarray,
    count: int,
    seed: int,
) -> np.ndarray:
    generator = np.random.default_rng(seed)
    eigenvalues, eigenvectors = np.linalg.eigh((covariance + covariance.T) * 0.5)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    noise = generator.standard_normal((count, len(prototype)))
    noise = noise @ (eigenvectors * np.sqrt(eigenvalues)).T
    generated = prototype[None, :] + noise
    return l2_normalize(generated).astype(np.float32)


def balanced_feature_set(
    features: np.ndarray,
    labels: np.ndarray,
    target: int,
    per_class: int,
    seed: int,
    target_replacement: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    all_features, all_labels = [], []
    for class_id in range(NUM_CLASSES):
        current = features[labels == class_id]
        if class_id == target and target_replacement is not None:
            selected = target_replacement
            if len(selected) != per_class:
                raise AssertionError("Target replacement has wrong size.")
        else:
            selected_indices = generator.choice(
                len(current), size=per_class, replace=len(current) < per_class
            )
            selected = current[selected_indices]
        all_features.append(selected.astype(np.float32))
        all_labels.append(np.full(per_class, class_id, dtype=np.int64))
    return np.concatenate(all_features), np.concatenate(all_labels)


def train_linear_head(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    set_seed(seed)
    dimension = train_features.shape[1]
    classifier = nn.Linear(dimension, NUM_CLASSES).to(device)
    optimizer = torch.optim.SGD(
        classifier.parameters(),
        lr=args.head_lr,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=True,
    )
    x = torch.from_numpy(train_features).to(device)
    y = torch.from_numpy(train_labels).to(device)
    for epoch in range(args.head_epochs):
        progress = epoch / max(args.head_epochs - 1, 1)
        learning_rate = args.head_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        permutation = torch.randperm(len(x), device=device)
        for start in range(0, len(x), args.head_batch_size):
            batch_indices = permutation[start : start + args.head_batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(classifier(x[batch_indices]), y[batch_indices])
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite linear-head loss.")
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        logits = []
        test_tensor = torch.from_numpy(test_features).to(device)
        for start in range(0, len(test_tensor), args.eval_batch_size):
            logits.append(classifier(test_tensor[start : start + args.eval_batch_size]).cpu())
    return torch.cat(logits).argmax(dim=1).numpy()


def accuracy_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    target: int,
    donor: int,
    counts: np.ndarray,
) -> dict:
    per_class = np.asarray(
        [np.mean(predictions[labels == class_id] == class_id) for class_id in range(NUM_CLASSES)]
    )
    target_mask = labels == target
    return {
        "overall_accuracy": float(np.mean(predictions == labels)),
        "macro_accuracy": float(per_class.mean()),
        "target_accuracy": float(per_class[target]),
        "donor_accuracy": float(per_class[donor]),
        "other_99_accuracy": float(np.mean(predictions[~target_mask] == labels[~target_mask])),
        "target_confused_as_donor": float(np.mean(predictions[target_mask] == donor)),
        "many_accuracy": float(per_class[counts >= 100].mean()),
        "medium_accuracy": float(per_class[(counts >= 20) & (counts < 100)].mean()),
        "few_accuracy": float(per_class[counts < 20].mean()),
    }


def prepare_replacements(
    target: int,
    donor: int,
    far_class: int,
    target_bundle: FeatureBundle,
    aligned_counter: np.ndarray,
    counter_bundle: FeatureBundle,
    trajectory_ratio: float,
    per_class: int,
    augmentation_seed: int,
) -> tuple[dict[str, np.ndarray | None], dict]:
    target_features = l2_normalize(
        target_bundle.features[target_bundle.labels == target].astype(np.float64)
    )
    donor_features = l2_normalize(
        target_bundle.features[target_bundle.labels == donor].astype(np.float64)
    )
    far_features = l2_normalize(
        target_bundle.features[target_bundle.labels == far_class].astype(np.float64)
    )
    oracle_features = aligned_counter[counter_bundle.labels == target]
    target_stats = support_statistics(target_features)
    donor_stats = support_statistics(donor_features)
    far_stats = support_statistics(far_features)
    oracle_stats = support_statistics(oracle_features)
    prototype = target_stats["prototype"]
    dimension = len(prototype)
    projector = np.eye(dimension) - np.outer(prototype, prototype)
    isotropic_shape = projector / max(dimension - 1, 1)
    self_shape_raw = tangent_projected_shape(target_stats["covariance"], prototype)
    self_shape = 0.5 * self_shape_raw + 0.5 * isotropic_shape
    donor_shape = tangent_projected_shape(donor_stats["covariance"], prototype)
    far_shape = tangent_projected_shape(far_stats["covariance"], prototype)
    oracle_shape = tangent_projected_shape(oracle_stats["covariance"], prototype)
    predicted_trace = float(
        np.clip(target_stats["trace"] * trajectory_ratio, 1e-5, 1.0)
    )
    real_target = target_features.astype(np.float32)
    synthetic_count = per_class - len(real_target)

    def combined(shape: np.ndarray, trace: float, offset: int) -> np.ndarray:
        synthetic = generate_features(
            prototype,
            shape * trace,
            synthetic_count,
            augmentation_seed + offset,
        )
        return np.concatenate([real_target, synthetic]).astype(np.float32)

    replacements: dict[str, np.ndarray | None] = {
        "crt_repeat": None,
        "self_gaussian": combined(self_shape, target_stats["trace"], 11),
        "similar_no_trajectory": combined(donor_shape, target_stats["trace"], 23),
        "similar_isotropic": combined(isotropic_shape, predicted_trace, 37),
        "similar_mixed": combined(
            0.5 * self_shape + 0.5 * donor_shape, predicted_trace, 41
        ),
        "similar_full": combined(donor_shape, predicted_trace, 53),
        "far_full": combined(far_shape, predicted_trace, 67),
        "oracle_support": combined(
            oracle_shape, oracle_stats["trace"], 79
        ),
    }
    diagnostics = {
        "target_train_count": len(target_features),
        "donor_train_count": len(donor_features),
        "far_class": far_class,
        "far_train_count": len(far_features),
        "oracle_train_count": len(oracle_features),
        "oracle_head_trace": oracle_stats["trace"],
        "target_tail_trace": target_stats["trace"],
        "trajectory_head_to_tail_trace_ratio": trajectory_ratio,
        "predicted_head_trace": predicted_trace,
        "donor_head_trace": donor_stats["trace"],
        "far_head_trace": far_stats["trace"],
    }
    return replacements, diagnostics


def run_scenario(
    seed: int,
    scenario: str,
    target: int,
    donor: int,
    target_order: str,
    counter_order: str,
    endpoint_design: dict,
    endpoint_bundles: dict[str, dict[str, FeatureBundle]],
    trajectory: dict,
    pair_scores: dict[int, dict[int, float]],
    class_names: list[str],
    args: argparse.Namespace,
    device: torch.device,
    seed_dir: Path,
) -> tuple[list[dict], dict]:
    target_bundle = endpoint_bundles[target_order]["train"]
    counter_bundle = endpoint_bundles[counter_order]["train"]
    test_bundle = endpoint_bundles[target_order]["test"]
    counts = (
        endpoint_design["counts_ab"]
        if target_order == "AB"
        else endpoint_design["counts_ba"]
    )
    aligned_counter, alignment_cosine = aligned_counter_features(
        target_bundle,
        counter_bundle,
        endpoint_design["background_classes"],
    )
    normalized_train = l2_normalize(target_bundle.features.astype(np.float64)).astype(np.float32)
    normalized_test = l2_normalize(test_bundle.features.astype(np.float64)).astype(np.float32)
    far_class = choose_far_head(target, donor, counts, pair_scores)
    donor_role = "A" if donor == args.class_a else "B"
    trajectory_ratio = trajectory["roles"][donor_role][
        "predicted_head_to_tail_trace_ratio"
    ]
    scenario_dir = seed_dir / scenario
    scenario_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_rows = []
    result_rows = []

    original_predictions = test_bundle.logits.argmax(axis=1)
    original_metrics = accuracy_metrics(
        original_predictions, test_bundle.labels, target, donor, counts
    )
    result_rows.append(
        {
            "seed": seed,
            "scenario": scenario,
            "target": target,
            "target_name": class_names[target],
            "donor": donor,
            "donor_name": class_names[donor],
            "method": "original_ce",
            "head_seed": -1,
            **original_metrics,
        }
    )

    for head_seed in args.head_seeds:
        replacements, diagnostics = prepare_replacements(
            target,
            donor,
            far_class,
            target_bundle,
            aligned_counter,
            counter_bundle,
            trajectory_ratio,
            args.balanced_per_class,
            augmentation_seed=seed * 10000 + head_seed * 100,
        )
        diagnostic_rows.append(
            {
                "seed": seed,
                "scenario": scenario,
                "head_seed": head_seed,
                "target": target,
                "donor": donor,
                "donor_similarity": pair_scores[target][donor],
                "far_class": far_class,
                "far_class_name": class_names[far_class],
                "far_similarity": pair_scores[target][far_class],
                "alignment_cosine": alignment_cosine,
                **diagnostics,
            }
        )
        for method, replacement in replacements.items():
            train_features, train_labels = balanced_feature_set(
                normalized_train,
                target_bundle.labels,
                target,
                args.balanced_per_class,
                seed=seed * 100000 + head_seed * 1000,
                target_replacement=replacement,
            )
            predictions = train_linear_head(
                train_features,
                train_labels,
                normalized_test,
                args,
                seed=head_seed,
                device=device,
            )
            metrics = accuracy_metrics(
                predictions, test_bundle.labels, target, donor, counts
            )
            result_rows.append(
                {
                    "seed": seed,
                    "scenario": scenario,
                    "target": target,
                    "target_name": class_names[target],
                    "donor": donor,
                    "donor_name": class_names[donor],
                    "method": method,
                    "head_seed": head_seed,
                    **metrics,
                }
            )
            print(
                f"[seed {seed} {scenario} head {head_seed} {method}] "
                f"target={metrics['target_accuracy'] * 100:.1f}% "
                f"overall={metrics['overall_accuracy'] * 100:.2f}%",
                flush=True,
            )
    write_csv(diagnostic_rows, scenario_dir / "support_diagnostics.csv")
    write_csv(result_rows, scenario_dir / "classifier_results.csv")
    summary = {
        "seed": seed,
        "scenario": scenario,
        "target": {"id": target, "name": class_names[target]},
        "donor": {"id": donor, "name": class_names[donor]},
        "far": {"id": far_class, "name": class_names[far_class]},
        "donor_similarity": pair_scores[target][donor],
        "far_similarity": pair_scores[target][far_class],
        "alignment_cosine": alignment_cosine,
        "trajectory_ratio": trajectory_ratio,
    }
    return result_rows, summary


def extract_endpoints(
    seed: int,
    raw_train,
    raw_test,
    class_orders: list[np.ndarray],
    endpoint_design: dict,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, dict[str, FeatureBundle]]:
    result = {}
    for order, counts in [
        ("AB", endpoint_design["counts_ab"]),
        ("BA", endpoint_design["counts_ba"]),
    ]:
        path = endpoint_paths(seed, args)[order]
        model = load_model(path, device)
        train_indices = select_indices(class_orders, counts)
        print(f"[seed {seed}] extracting {order} train/test features", flush=True)
        result[order] = {
            "train": extract_bundle(model, raw_train, train_indices, args, device),
            "test": extract_bundle(model, raw_test, None, args, device),
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return result


def aggregate(
    rows: list[dict],
    scenario_summaries: list[dict],
    args: argparse.Namespace,
) -> dict:
    methods = sorted(set(row["method"] for row in rows))
    aggregate_rows = []
    for method in methods:
        current = [row for row in rows if row["method"] == method]
        aggregate_rows.append(
            {
                "method": method,
                "runs": len(current),
                "target_accuracy_mean": float(
                    np.mean([row["target_accuracy"] for row in current])
                ),
                "target_accuracy_std": float(
                    np.std([row["target_accuracy"] for row in current], ddof=1)
                    if len(current) > 1
                    else 0.0
                ),
                "overall_accuracy_mean": float(
                    np.mean([row["overall_accuracy"] for row in current])
                ),
                "other_99_accuracy_mean": float(
                    np.mean([row["other_99_accuracy"] for row in current])
                ),
                "target_confused_as_donor_mean": float(
                    np.mean([row["target_confused_as_donor"] for row in current])
                ),
            }
        )
    write_csv(rows, args.output_dir / "all_classifier_results.csv")
    write_csv(aggregate_rows, args.output_dir / "method_summary.csv")
    result = {"methods": aggregate_rows, "scenarios": scenario_summaries}
    save_json(result, args.output_dir / "summary.json")

    try:
        import matplotlib.pyplot as plt

        labels = [row["method"] for row in aggregate_rows]
        means = [row["target_accuracy_mean"] * 100 for row in aggregate_rows]
        errors = [row["target_accuracy_std"] * 100 for row in aggregate_rows]
        figure, axis = plt.subplots(figsize=(12, 6), constrained_layout=True)
        axis.bar(np.arange(len(labels)), means, yerr=errors, capsize=3)
        axis.set_xticks(np.arange(len(labels)), labels, rotation=35, ha="right")
        axis.set_ylabel("Target tail-class accuracy (%)")
        axis.set_title("End-to-end support recovery feasibility")
        axis.grid(axis="y", alpha=0.25)
        figure.savefig(args.output_dir / "target_accuracy_comparison.png", dpi=180)
        plt.close(figure)
    except ImportError:
        pass
    return result


def build_report(result: dict, args: argparse.Namespace) -> None:
    rows = result["methods"]
    by_method = {row["method"]: row for row in rows}
    lines = [
        "# 相似类支撑恢复端到端可行性实验",
        "",
        "- 平衡参考空间筛选：maple_tree（47）与oak_tree（52）互为最近原型邻居。",
        "- 双向任务：A尾/B头以及B尾/A头。",
        "- 支撑预测、空间对齐和特征生成仅使用训练数据；测试集只用于最终评价。",
        "",
        "| 方法 | 目标尾类准确率 | 标准差 | 整体准确率 | 其余99类准确率 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['target_accuracy_mean'] * 100:.2f}% | "
            f"{row['target_accuracy_std'] * 100:.2f}% | "
            f"{row['overall_accuracy_mean'] * 100:.2f}% | "
            f"{row['other_99_accuracy_mean'] * 100:.2f}% |"
        )
    lines.extend(["", "## 自动诊断", ""])
    if "oracle_support" in by_method and "crt_repeat" in by_method:
        oracle_gain = (
            by_method["oracle_support"]["target_accuracy_mean"]
            - by_method["crt_repeat"]["target_accuracy_mean"]
        )
        lines.append(f"- Oracle相对cRT提升：{oracle_gain * 100:.2f}个百分点。")
    if "similar_full" in by_method and "crt_repeat" in by_method:
        similar_gain = (
            by_method["similar_full"]["target_accuracy_mean"]
            - by_method["crt_repeat"]["target_accuracy_mean"]
        )
        lines.append(f"- 相似类完整支撑相对cRT提升：{similar_gain * 100:.2f}个百分点。")
    if "similar_full" in by_method and "far_full" in by_method:
        donor_gain = (
            by_method["similar_full"]["target_accuracy_mean"]
            - by_method["far_full"]["target_accuracy_mean"]
        )
        lines.append(f"- 相似类相对远距离类提升：{donor_gain * 100:.2f}个百分点。")
    (args.output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.seeds = args.seeds[:1]
        args.head_seeds = args.head_seeds[:1]
        args.head_epochs = max(args.head_epochs, 300)
        args.balanced_per_class = min(args.balanced_per_class, 100)
        args.num_workers = min(args.num_workers, 2)
        args.output_dir = args.output_dir / "smoke"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    class_names = read_class_names(args.data_root)
    raw_train = torchvision.datasets.CIFAR100(
        root=args.data_root, train=True, download=False
    )
    raw_test = torchvision.datasets.CIFAR100(
        root=args.data_root, train=False, download=False
    )
    train_targets = np.asarray(raw_train.targets, dtype=np.int64)
    class_orders = make_nested_class_orders(train_targets, args.subset_seed)
    curve = exponential_counts(args.max_count, args.imbalance_factor)
    endpoint_design = make_rank_swap_design(
        args.class_a, args.class_b, curve, args.rank_seed
    )
    sweep_namespace = argparse.Namespace(
        class_a=args.class_a,
        class_b=args.class_b,
        ranks=DEFAULT_RANKS,
        rank_seed=args.rank_seed,
    )
    sweep_design = make_sweep_design(sweep_namespace, curve)
    pair_scores = load_balanced_pair_scores(args)
    if max(pair_scores[args.class_a], key=pair_scores[args.class_a].get) != args.class_b:
        raise AssertionError("B is not A's nearest balanced prototype class.")
    if max(pair_scores[args.class_b], key=pair_scores[args.class_b].get) != args.class_a:
        raise AssertionError("A is not B's nearest balanced prototype class.")

    audit = {
        "no_test_statistics_for_recovery": True,
        "test_usage": "final classifier evaluation only",
        "class_a": {"id": args.class_a, "name": class_names[args.class_a]},
        "class_b": {"id": args.class_b, "name": class_names[args.class_b]},
        "balanced_similarity_a_b": pair_scores[args.class_a][args.class_b],
        "a_and_b_mutual_nearest_prototypes": True,
        "seeds": [],
        "arguments": vars(args),
    }
    all_rows, scenario_summaries = [], []
    for seed in args.seeds:
        seed_dir = args.output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_audit = validate_checkpoints(
            seed, endpoint_design, sweep_design, args
        )
        audit["seeds"].append(checkpoint_audit)
        print(f"[seed {seed}] extracting training-only trajectories", flush=True)
        trajectory = extract_trajectory_statistics(
            seed,
            raw_train,
            class_orders,
            sweep_design,
            args,
            device,
        )
        save_json(trajectory, seed_dir / "training_trajectory.json")
        endpoints = extract_endpoints(
            seed,
            raw_train,
            raw_test,
            class_orders,
            endpoint_design,
            args,
            device,
        )
        scenarios = [
            ("A_tail_from_B", args.class_a, args.class_b, "BA", "AB"),
            ("B_tail_from_A", args.class_b, args.class_a, "AB", "BA"),
        ]
        for scenario, target, donor, target_order, counter_order in scenarios:
            rows, summary = run_scenario(
                seed,
                scenario,
                target,
                donor,
                target_order,
                counter_order,
                endpoint_design,
                endpoints,
                trajectory,
                pair_scores,
                class_names,
                args,
                device,
                seed_dir,
            )
            all_rows.extend(rows)
            scenario_summaries.append(summary)
    save_json(audit, args.output_dir / "audit.json")
    result = aggregate(all_rows, scenario_summaries, args)
    build_report(result, args)
    print(f"Finished: {args.output_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
