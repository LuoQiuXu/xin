from __future__ import annotations

import argparse
import csv
import math
import pickle
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from scipy.stats import spearmanr, wilcoxon
from torch.utils.data import DataLoader, RandomSampler

from support_dynamics_experiment import (
    ANGLE_NAMES,
    DESCRIPTOR_INDEX,
    DESCRIPTOR_NAMES,
    NUM_CLASSES,
    SHAPE_NAMES,
    ArrayCIFAR100,
    ResNetCifar,
    classification_accuracy,
    cosine_learning_rate,
    describe_supports,
    eval_transform,
    extract_features,
    l2_normalize,
    make_nested_class_orders,
    normalize_vector,
    orthogonal_procrustes,
    save_json,
    set_seed,
    train_transform,
    write_csv,
)


PRIMARY_NAMES = [
    "angle_q50",
    "angle_q80",
    "angle_q95",
    "angle_std",
    "tangent_trace",
    "tangent_effective_rank",
    "tangent_eigshare_1",
    "tangent_eigshare_2",
    "tangent_eigshare_3",
    "tangent_eigshare_4",
]


def primary_shape_state(descriptors: np.ndarray) -> np.ndarray:
    state = np.zeros(
        (len(descriptors), len(PRIMARY_NAMES)), dtype=np.float64
    )
    direct_names = [
        "angle_q50",
        "angle_q80",
        "angle_q95",
        "angle_std",
        "tangent_trace",
        "tangent_effective_rank",
    ]
    for column, name in enumerate(direct_names):
        state[:, column] = descriptors[:, DESCRIPTOR_INDEX[name]]
    trace = np.maximum(
        descriptors[:, DESCRIPTOR_INDEX["tangent_trace"]], 1e-12
    )
    for rank in range(1, 5):
        state[:, 5 + rank] = (
            descriptors[:, DESCRIPTOR_INDEX[f"tangent_eig_{rank}"]]
            / trace
        )
    return state


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Test whether balanced-space similar CIFAR-100 class pairs share "
            "500->200->50->20->5 support trajectories under matched "
            "counterfactual A/B/C tail-role interventions."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(r"D:\xiangmuxunlian\hafe_experiment\data"),
    )
    parser.add_argument(
        "--similarity-csv",
        type=Path,
        default=(
            project_dir
            / "outputs"
            / "balanced_neighbor_trajectory"
            / "seed_42"
            / "all_class_pairs.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "outputs" / "role_swap_trajectory",
    )
    parser.add_argument(
        "--levels", type=int, nargs="+", default=[5, 20, 50, 200, 500]
    )
    parser.add_argument("--num-triplets", type=int, default=10)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42, 43, 44]
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--subset-seed", type=int, default=20260729)
    parser.add_argument(
        "--samples-per-epoch",
        type=int,
        default=50000,
        help=(
            "Uniform-with-replacement draws per epoch. This fixes optimizer "
            "steps while preserving each constructed dataset's class ratios."
        ),
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument(
        "--design-only",
        action="store_true",
        help="Validate and save all role/count datasets without training.",
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def read_class_names(data_root: Path) -> list[str]:
    with (data_root / "cifar-100-python" / "meta").open("rb") as handle:
        payload = pickle.load(handle, encoding="latin1")
    return list(
        payload.get("fine_label_names", payload.get(b"fine_label_names"))
    )


def read_pair_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Balanced reference similarity file is missing: {path}"
        )
    return list(csv.DictReader(path.open("r", encoding="utf-8-sig")))


def select_disjoint_triplets(
    rows: list[dict], num_triplets: int
) -> list[dict]:
    directed_by_target: dict[int, list[dict]] = {}
    for row in rows:
        directed_by_target.setdefault(int(row["target_class"]), []).append(row)
    similar_candidates = sorted(
        (
            row
            for row in rows
            if int(row["target_class"]) < int(row["candidate_class"])
        ),
        key=lambda row: float(row["knn"]),
        reverse=True,
    )
    used: set[int] = set()
    triplets = []
    for pair in similar_candidates:
        anchor = int(pair["target_class"])
        similar = int(pair["candidate_class"])
        if anchor in used or similar in used:
            continue
        far_candidates = [
            row
            for row in directed_by_target[anchor]
            if int(row["candidate_class"]) not in used | {anchor, similar}
            and int(row["same_coarse"]) == 0
        ]
        if not far_candidates:
            continue
        # Low balanced-space fusion is a negative semantic/structural control.
        far_row = min(
            far_candidates, key=lambda row: float(row["fusion"])
        )
        far = int(far_row["candidate_class"])
        used.update({anchor, similar, far})
        triplets.append(
            {
                "triplet_id": len(triplets),
                "anchor": anchor,
                "similar": similar,
                "far": far,
                "anchor_similar_knn": float(pair["knn"]),
                "anchor_similar_prototype": float(pair["prototype"]),
                "anchor_similar_confusion": float(pair["confusion"]),
                "anchor_similar_fusion": float(pair["fusion"]),
                "anchor_far_knn": float(far_row["knn"]),
                "anchor_far_prototype": float(far_row["prototype"]),
                "anchor_far_confusion": float(far_row["confusion"]),
                "anchor_far_fusion": float(far_row["fusion"]),
            }
        )
        if len(triplets) == num_triplets:
            break
    if len(triplets) != num_triplets:
        raise ValueError(
            f"Could construct only {len(triplets)} disjoint triplets; "
            f"requested {num_triplets}."
        )
    return triplets


def select_frequency_indices(
    class_orders: list[np.ndarray],
    tail_classes: set[int],
    count: int,
) -> np.ndarray:
    return np.concatenate(
        [
            order[: count if class_id in tail_classes else 500]
            for class_id, order in enumerate(class_orders)
        ]
    ).astype(np.int64)


def train_frequency_model(
    seed: int,
    role: str,
    count: int,
    train_dataset: ArrayCIFAR100,
    test_loader: DataLoader,
    seed_dir: Path,
    tail_classes: set[int],
    args: argparse.Namespace,
    device: torch.device,
) -> Path:
    path = seed_dir / f"model_role_{role}_count_{count:03d}_final.pth"
    expected_tails = sorted(tail_classes)
    expected_steps = math.ceil(args.samples_per_epoch / args.batch_size)
    if args.skip_train and path.exists():
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        compatible = (
            int(checkpoint.get("seed", -1)) == seed
            and checkpoint.get("role") == role
            and int(checkpoint.get("probe_count", -1)) == count
            and checkpoint.get("tail_classes") == expected_tails
            and int(checkpoint.get("epochs", -1)) == args.epochs
            and int(checkpoint.get("samples_per_epoch", -1))
            == args.samples_per_epoch
        )
        if compatible:
            print(
                f"[seed {seed} role={role} n={count}] "
                f"reusing {path.name}",
                flush=True,
            )
            return path
        raise ValueError(f"Incompatible checkpoint: {path}")

    set_seed(seed)
    model = ResNetCifar(depth=32).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and not args.no_amp
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = RandomSampler(
        train_dataset,
        replacement=True,
        num_samples=args.samples_per_epoch,
        generator=generator,
    )
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )
    history = []
    started = time.time()
    for epoch in range(args.epochs):
        lr = cosine_learning_rate(args.lr, epoch, args.epochs)
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        loss_sum = 0.0
        correct = 0
        seen = 0
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda" and not args.no_amp,
            ):
                logits = model(images)
                loss = F.cross_entropy(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * labels.numel()
            correct += int((logits.detach().argmax(dim=1) == labels).sum())
            seen += int(labels.numel())
        row = {
            "epoch": epoch + 1,
            "lr": lr,
            "loss": loss_sum / max(seen, 1),
            "train_accuracy": correct / max(seen, 1),
            "examples_drawn": seen,
            "steps": len(loader),
            "elapsed_seconds": time.time() - started,
        }
        history.append(row)
        print_every = max(1, args.epochs // 5)
        if epoch == 0 or (epoch + 1) % print_every == 0:
            print(
                f"[seed {seed} role={role} n={count}] "
                f"epoch {epoch + 1:03d}/{args.epochs}, "
                f"loss={row['loss']:.4f}, "
                f"train={row['train_accuracy'] * 100:.2f}%, "
                f"steps={row['steps']}/{expected_steps}, "
                f"elapsed={row['elapsed_seconds']:.1f}s",
                flush=True,
            )
    test_accuracy = classification_accuracy(model, test_loader, device)
    torch.save(
        {
            "model": model.state_dict(),
            "seed": seed,
            "role": role,
            "probe_count": count,
            "tail_classes": expected_tails,
            "epochs": args.epochs,
            "samples_per_epoch": args.samples_per_epoch,
            "test_accuracy": test_accuracy,
            "history": history,
        },
        path,
    )
    save_json(history, path.with_suffix(".history.json"))
    print(
        f"[seed {seed} role={role} n={count}] "
        f"test={test_accuracy * 100:.2f}% saved={path.name}",
        flush=True,
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return path


def load_model(path: Path, device: torch.device) -> ResNetCifar:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = ResNetCifar(depth=32)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval()


def evaluate_checkpoint(
    path: Path,
    dataset: ArrayCIFAR100,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    model = load_model(path, device)
    features, logits, labels, _ = extract_features(
        model, dataset, args, device
    )
    support = describe_supports(features, labels, logits)
    result = {
        "features": features.astype(np.float32),
        "labels": labels,
        "descriptors": support.descriptors,
        "accuracy": support.class_accuracy,
        "prototypes": support.prototypes,
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def cosine(first: np.ndarray, second: np.ndarray) -> float:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(first, second) / denominator)


def correlation(first: np.ndarray, second: np.ndarray) -> float:
    if np.std(first) <= 1e-12 or np.std(second) <= 1e-12:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


def safe_wilcoxon(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if np.allclose(values, 0.0):
        return {"statistic": 0.0, "pvalue": 1.0}
    result = wilcoxon(values)
    return {
        "statistic": float(result.statistic),
        "pvalue": float(result.pvalue),
    }


def exact_sign_flip_pvalue(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    observed = abs(float(values.mean()))
    if len(values) == 0:
        return 1.0
    means = []
    for mask in range(1 << len(values)):
        signs = np.asarray(
            [1.0 if mask & (1 << index) else -1.0
             for index in range(len(values))]
        )
        means.append(abs(float(np.mean(values * signs))))
    return float(np.mean(np.asarray(means) >= observed - 1e-12))


def bootstrap_mean_interval(
    values: np.ndarray,
    samples: int = 10000,
    seed: int = 20260730,
) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    indices = generator.integers(
        0, len(values), size=(samples, len(values))
    )
    means = values[indices].mean(axis=1)
    return [
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    ]


def build_class_trajectory(
    class_id: int,
    states: dict[int, dict],
    background_classes: np.ndarray,
    primary_head_scale: np.ndarray,
    full_head_scale: np.ndarray,
) -> dict:
    counts = sorted(states, reverse=True)
    primary_steps = []
    prototype_steps = []
    accuracy_steps = []
    increment_rows = []
    for high_count, low_count in zip(counts[:-1], counts[1:]):
        high = states[high_count]
        low = states[low_count]
        descriptor_delta = low["descriptors"] - high["descriptors"]
        background_delta = np.median(
            descriptor_delta[background_classes], axis=0
        )
        adjusted_full = (
            descriptor_delta[class_id] - background_delta
        ) / np.maximum(full_head_scale, 1e-8)
        primary_delta = (
            primary_shape_state(low["descriptors"])
            - primary_shape_state(high["descriptors"])
        )
        primary_background = np.median(
            primary_delta[background_classes], axis=0
        )
        adjusted_primary = (
            primary_delta[class_id] - primary_background
        ) / np.maximum(primary_head_scale, 1e-8)
        primary_steps.append(adjusted_primary)
        accuracy_delta = low["accuracy"] - high["accuracy"]
        adjusted_accuracy = float(
            accuracy_delta[class_id]
            - np.median(accuracy_delta[background_classes])
        )
        accuracy_steps.append(adjusted_accuracy)
        prototype_steps.append(
            low["aligned_prototypes"][class_id]
            - high["aligned_prototypes"][class_id]
        )
        row = {
            "class_id": class_id,
            "high_count": high_count,
            "low_count": low_count,
            "adjusted_accuracy_delta": adjusted_accuracy,
        }
        row.update(
            {
                f"adjusted_{name}": float(adjusted_full[index])
                for index, name in enumerate(DESCRIPTOR_NAMES)
            }
        )
        row.update(
            {
                f"primary_{name}": float(adjusted_primary[index])
                for index, name in enumerate(PRIMARY_NAMES)
            }
        )
        increment_rows.append(row)
    primary_array = np.asarray(primary_steps)
    prototype_array = np.asarray(prototype_steps)
    accuracy_array = np.asarray(accuracy_steps)
    return {
        "primary_steps": primary_array,
        "primary_vector": primary_array.reshape(-1),
        "prototype_steps": prototype_array,
        "prototype_vector": prototype_array.reshape(-1),
        "accuracy_vector": accuracy_array,
        "increment_rows": increment_rows,
    }


def compare_trajectories(first: dict, second: dict) -> dict:
    primary_step_cosines = [
        cosine(a, b)
        for a, b in zip(first["primary_steps"], second["primary_steps"])
    ]
    prototype_step_cosines = [
        cosine(a, b)
        for a, b in zip(
            first["prototype_steps"], second["prototype_steps"]
        )
    ]
    return {
        "primary_trajectory_cosine": cosine(
            first["primary_vector"], second["primary_vector"]
        ),
        "primary_trajectory_correlation": correlation(
            first["primary_vector"], second["primary_vector"]
        ),
        "primary_step_cosine_mean": float(
            np.mean(primary_step_cosines)
        ),
        "primary_endpoint_cosine": cosine(
            first["primary_steps"].sum(axis=0),
            second["primary_steps"].sum(axis=0),
        ),
        "primary_sign_agreement": float(
            np.mean(
                np.sign(first["primary_vector"])
                == np.sign(second["primary_vector"])
            )
        ),
        "prototype_trajectory_cosine": cosine(
            first["prototype_vector"], second["prototype_vector"]
        ),
        "prototype_step_cosine_mean": float(
            np.mean(prototype_step_cosines)
        ),
        "prototype_endpoint_cosine": cosine(
            first["prototype_steps"].sum(axis=0),
            second["prototype_steps"].sum(axis=0),
        ),
        "accuracy_trajectory_cosine": cosine(
            first["accuracy_vector"], second["accuracy_vector"]
        ),
        "accuracy_trajectory_correlation": correlation(
            first["accuracy_vector"], second["accuracy_vector"]
        ),
    }


def summarize_seed(
    seed: int,
    triplets: list[dict],
    trajectories: dict[int, dict],
    class_names: list[str],
    seed_dir: Path,
) -> dict:
    comparison_rows = []
    metric_names = None
    for triplet in triplets:
        anchor = triplet["anchor"]
        similar = triplet["similar"]
        far = triplet["far"]
        similar_result = compare_trajectories(
            trajectories[anchor], trajectories[similar]
        )
        far_result = compare_trajectories(
            trajectories[anchor], trajectories[far]
        )
        metric_names = [
            name
            for name, value in similar_result.items()
            if isinstance(value, (int, float))
        ]
        row = {
            "seed": seed,
            "triplet_id": triplet["triplet_id"],
            "anchor": anchor,
            "anchor_name": class_names[anchor],
            "similar": similar,
            "similar_name": class_names[similar],
            "far": far,
            "far_name": class_names[far],
        }
        for name in metric_names:
            row[f"similar_{name}"] = similar_result[name]
            row[f"far_{name}"] = far_result[name]
            row[f"delta_{name}"] = (
                similar_result[name] - far_result[name]
            )
        comparison_rows.append(row)
    write_csv(comparison_rows, seed_dir / "triplet_comparisons.csv")

    aggregate = {}
    for name in metric_names or []:
        similar_values = np.asarray(
            [row[f"similar_{name}"] for row in comparison_rows]
        )
        far_values = np.asarray(
            [row[f"far_{name}"] for row in comparison_rows]
        )
        delta = similar_values - far_values
        aggregate[name] = {
            "similar_mean": float(similar_values.mean()),
            "far_mean": float(far_values.mean()),
            "delta_mean": float(delta.mean()),
            "similar_better_fraction": float(np.mean(delta > 0.0)),
            "wilcoxon": safe_wilcoxon(delta),
        }
    summary = {
        "seed": seed,
        "num_triplets": len(triplets),
        "aggregate": aggregate,
    }
    save_json(summary, seed_dir / "summary.json")
    return summary


def run_seed(
    seed: int,
    raw_train,
    raw_test,
    class_orders: list[np.ndarray],
    triplets: list[dict],
    class_names: list[str],
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    seed_dir = args.output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    train_targets = np.asarray(raw_train.targets, dtype=np.int64)
    test_targets = np.asarray(raw_test.targets, dtype=np.int64)
    probe_classes = {
        class_id
        for triplet in triplets
        for class_id in (
            triplet["anchor"],
            triplet["similar"],
            triplet["far"],
        )
    }
    background_classes = np.asarray(
        sorted(set(range(NUM_CLASSES)) - probe_classes), dtype=np.int64
    )
    if len(probe_classes) != 3 * len(triplets):
        raise AssertionError("Triplets are not class-disjoint.")

    test_dataset = ArrayCIFAR100(
        raw_test.data,
        test_targets,
        None,
        eval_transform(),
        return_index=True,
    )
    test_loader = DataLoader(
        ArrayCIFAR100(
            raw_test.data, test_targets, None, eval_transform()
        ),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    role_fields = {"A": "anchor", "B": "similar", "C": "far"}
    model_specs: list[tuple[str, int, set[int]]] = [
        ("H", max(args.levels), set())
    ]
    for count in sorted(args.levels, reverse=True):
        if count == max(args.levels):
            continue
        for model_role, field in role_fields.items():
            model_specs.append(
                (
                    model_role,
                    count,
                    {triplet[field] for triplet in triplets},
                )
            )

    checkpoints: dict[tuple[str, int], Path] = {}
    design_rows = []
    for model_role, count, tail_classes in model_specs:
        indices = select_frequency_indices(
            class_orders, tail_classes, count
        )
        observed_counts = np.bincount(
            train_targets[indices], minlength=NUM_CLASSES
        )
        if not all(
            observed_counts[class_id]
            == (count if class_id in tail_classes else 500)
            for class_id in range(NUM_CLASSES)
        ):
            raise AssertionError("Constructed class counts are incorrect.")
        design_rows.append(
            {
                "seed": seed,
                "model_role": model_role,
                "count": count,
                "num_tail_classes": len(tail_classes),
                "num_probe_classes": len(probe_classes),
                "num_background_classes": len(background_classes),
                "dataset_size": len(indices),
                "tail_classes": " ".join(
                    str(class_id) for class_id in sorted(tail_classes)
                ),
                "samples_drawn_per_epoch": args.samples_per_epoch,
                "optimizer_steps_per_epoch": math.ceil(
                    args.samples_per_epoch / args.batch_size
                ),
            }
        )
        dataset = ArrayCIFAR100(
            raw_train.data,
            train_targets,
            indices,
            train_transform(),
        )
        checkpoints[(model_role, count)] = train_frequency_model(
            seed,
            model_role,
            count,
            dataset,
            test_loader,
            seed_dir,
            tail_classes,
            args,
            device,
        )
    write_csv(design_rows, seed_dir / "frequency_design.csv")

    print(f"[seed {seed}] extracting support trajectories", flush=True)
    raw_states = {
        key: evaluate_checkpoint(
            checkpoint, test_dataset, args, device
        )
        for key, checkpoint in checkpoints.items()
    }
    head_key = ("H", max(args.levels))
    head_state = raw_states[head_key]
    head_normalized = l2_normalize(
        head_state["features"].astype(np.float64)
    )
    states_by_model = {}
    alignment_rows = []
    for (model_role, count), state in raw_states.items():
        if (model_role, count) == head_key:
            aligned_features = head_normalized
            alignment_cosine = 1.0
        else:
            anchor_mask = np.isin(
                state["labels"], background_classes
            )
            rotation = orthogonal_procrustes(
                state["features"][anchor_mask],
                head_state["features"][anchor_mask],
            )
            aligned_features = l2_normalize(
                state["features"].astype(np.float64) @ rotation
            )
            alignment_cosine = float(
                np.mean(
                    np.sum(
                        aligned_features[anchor_mask]
                        * head_normalized[anchor_mask],
                        axis=1,
                    )
                )
            )
        aligned_prototypes = np.zeros(
            (NUM_CLASSES, aligned_features.shape[1]), dtype=np.float64
        )
        for class_id in range(NUM_CLASSES):
            aligned_prototypes[class_id] = normalize_vector(
                aligned_features[state["labels"] == class_id].mean(axis=0)
            )
        states_by_model[(model_role, count)] = {
            "descriptors": state["descriptors"],
            "accuracy": state["accuracy"],
            "aligned_prototypes": aligned_prototypes,
        }
        alignment_rows.append(
            {
                "seed": seed,
                "model_role": model_role,
                "count": count,
                "alignment_background_classes": len(background_classes),
                "same_image_cosine_after": alignment_cosine,
            }
        )
    write_csv(alignment_rows, seed_dir / "alignment.csv")

    state_rows = []
    class_states: dict[int, dict[int, dict]] = {}
    class_roles: dict[int, str] = {}
    for triplet in triplets:
        for model_role, field in role_fields.items():
            class_id = triplet[field]
            class_roles[class_id] = model_role
            class_states[class_id] = {
                max(args.levels): states_by_model[head_key]
            }
            for count in sorted(args.levels, reverse=True):
                if count == max(args.levels):
                    continue
                class_states[class_id][count] = states_by_model[
                    (model_role, count)
                ]
                state = class_states[class_id][count]

            for count in sorted(args.levels, reverse=True):
                state = class_states[class_id][count]
                row = {
                    "seed": seed,
                    "triplet_id": triplet["triplet_id"],
                    "role": field,
                    "model_role": (
                        "H" if count == max(args.levels) else model_role
                    ),
                    "class_id": class_id,
                    "class_name": class_names[class_id],
                    "count": count,
                    "accuracy": float(state["accuracy"][class_id]),
                }
                row.update(
                    {
                        name: float(
                            state["descriptors"][
                                class_id, DESCRIPTOR_INDEX[name]
                            ]
                        )
                        for name in DESCRIPTOR_NAMES
                    }
                )
                state_rows.append(row)
    write_csv(state_rows, seed_dir / "support_states.csv")

    full_head_scale = np.std(
        states_by_model[head_key]["descriptors"], axis=0, ddof=1
    )
    primary_head_scale = np.std(
        primary_shape_state(
            states_by_model[head_key]["descriptors"]
        ),
        axis=0,
        ddof=1,
    )
    trajectories = {}
    increment_rows = []
    for class_id in sorted(probe_classes):
        trajectory = build_class_trajectory(
            class_id,
            class_states[class_id],
            background_classes,
            primary_head_scale,
            full_head_scale,
        )
        trajectories[class_id] = trajectory
        for row in trajectory["increment_rows"]:
            row["seed"] = seed
            row["class_name"] = class_names[class_id]
            row["model_role"] = class_roles[class_id]
            increment_rows.append(row)
    write_csv(increment_rows, seed_dir / "adjusted_increments.csv")
    summary = summarize_seed(
        seed, triplets, trajectories, class_names, seed_dir
    )
    summary["alignment"] = alignment_rows
    summary["frequency_design"] = design_rows
    summary["primary_trajectory_vectors"] = {
        str(class_id): trajectory["primary_vector"]
        for class_id, trajectory in trajectories.items()
    }
    save_json(summary, seed_dir / "summary.json")
    del raw_states
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def aggregate_across_seeds(
    summaries: list[dict],
    triplets: list[dict],
    output_dir: Path,
) -> dict:
    all_rows = []
    for summary in summaries:
        path = (
            output_dir
            / f"seed_{summary['seed']}"
            / "triplet_comparisons.csv"
        )
        all_rows.extend(
            list(csv.DictReader(path.open("r", encoding="utf-8-sig")))
        )
    primary = "primary_trajectory_cosine"
    per_triplet = []
    for triplet in triplets:
        rows = [
            row
            for row in all_rows
            if int(row["triplet_id"]) == triplet["triplet_id"]
        ]
        similar = np.mean(
            [float(row[f"similar_{primary}"]) for row in rows]
        )
        far = np.mean([float(row[f"far_{primary}"]) for row in rows])
        per_triplet.append(
            {
                "triplet_id": triplet["triplet_id"],
                "similar_seed_mean": float(similar),
                "far_seed_mean": float(far),
                "delta_seed_mean": float(similar - far),
            }
        )
    write_csv(per_triplet, output_dir / "cross_seed_triplets.csv")
    delta = np.asarray(
        [row["delta_seed_mean"] for row in per_triplet]
    )
    seed_effects = [
        float(
            summary["aggregate"]["primary_trajectory_cosine"][
                "delta_mean"
            ]
        )
        for summary in summaries
    ]
    probe_ids = sorted(
        {
            class_id
            for triplet in triplets
            for class_id in (
                triplet["anchor"],
                triplet["similar"],
                triplet["far"],
            )
        }
    )
    same_class_seed_cosines = []
    random_cross_class_cosines = []
    for first_summary, second_summary in combinations(summaries, 2):
        first_vectors = first_summary["primary_trajectory_vectors"]
        second_vectors = second_summary["primary_trajectory_vectors"]
        for position, class_id in enumerate(probe_ids):
            first = np.asarray(first_vectors[str(class_id)])
            second_same = np.asarray(second_vectors[str(class_id)])
            second_random = np.asarray(
                second_vectors[str(probe_ids[(position + 7) % len(probe_ids)])]
            )
            same_class_seed_cosines.append(cosine(first, second_same))
            random_cross_class_cosines.append(cosine(first, second_random))
    h1 = {
        "same_class_cross_seed_cosine_mean": (
            float(np.mean(same_class_seed_cosines))
            if same_class_seed_cosines
            else None
        ),
        "random_cross_class_cross_seed_cosine_mean": (
            float(np.mean(random_cross_class_cosines))
            if random_cross_class_cosines
            else None
        ),
    }
    h1_supported = bool(
        same_class_seed_cosines
        and np.mean(same_class_seed_cosines)
        > np.mean(random_cross_class_cosines)
    )
    wilcoxon_result = safe_wilcoxon(delta)
    result = {
        "primary_metric": primary,
        "num_seeds": len(summaries),
        "num_triplets": len(triplets),
        "delta_mean": float(delta.mean()),
        "delta_bootstrap_95ci": bootstrap_mean_interval(delta),
        "similar_better_fraction": float(np.mean(delta > 0.0)),
        "wilcoxon": wilcoxon_result,
        "exact_sign_flip_pvalue": exact_sign_flip_pvalue(delta),
        "seed_effects": seed_effects,
        "all_seed_effect_directions_positive": bool(
            all(value > 0.0 for value in seed_effects)
        ),
        "h1_cross_seed_repeatability": h1,
        "h1_supported": h1_supported,
        "preliminary_supported": bool(
            delta.mean() > 0.05
            and np.mean(delta > 0.0) >= 0.70
            and wilcoxon_result["pvalue"] < 0.05
            and all(value > 0.0 for value in seed_effects)
            and h1_supported
        ),
    }
    save_json(result, output_dir / "aggregate.json")
    return result


def build_report_legacy(
    summaries: list[dict],
    aggregate: dict,
    triplets: list[dict],
    class_names: list[str],
    output_dir: Path,
) -> None:
    lines = [
        "# 相似类别成对频率轨迹实验",
        "",
        "平衡参考空间预先选择互不重叠的目标–相似–远类三元组。所有探针类在同一个模型中共同处于500、200、50、20或5个训练样本；其余类别始终保持500个样本。",
        "",
        "## 类别设计",
        "",
        "| 组 | A目标类 | B相似类 | C远类 | A–B KNN | A–C KNN |",
        "|---:|---|---|---|---:|---:|",
    ]
    for triplet in triplets:
        lines.append(
            f"| {triplet['triplet_id']} | "
            f"{class_names[triplet['anchor']]} | "
            f"{class_names[triplet['similar']]} | "
            f"{class_names[triplet['far']]} | "
            f"{triplet['anchor_similar_knn']:.4f} | "
            f"{triplet['anchor_far_knn']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 每个随机种子的主要结果",
            "",
            "| Seed | A–B轨迹余弦 | A–C轨迹余弦 | 差值 | A–B胜出比例 | Wilcoxon p |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for summary in summaries:
        row = summary["aggregate"]["primary_trajectory_cosine"]
        lines.append(
            f"| {summary['seed']} | {row['similar_mean']:.4f} | "
            f"{row['far_mean']:.4f} | {row['delta_mean']:+.4f} | "
            f"{row['similar_better_fraction'] * 100:.1f}% | "
            f"{row['wilcoxon']['pvalue']:.4g} |"
        )
    verdict = (
        "初步支持：平衡空间相似类的头尾支撑轨迹显著比远类更一致。"
        if aggregate["preliminary_supported"]
        else "暂未通过：相似类轨迹相对远类的优势未达到预设稳定性和显著性门槛。"
    )
    lines.extend(
        [
            "",
            "## 跨种子配对判定",
            "",
            f"**{verdict}**",
            "",
            f"- A–B相对A–C平均轨迹余弦优势：**{aggregate['delta_mean']:+.4f}**",
            f"- A–B胜出的三元组比例：**{aggregate['similar_better_fraction'] * 100:.1f}%**",
            f"- 对三元组先跨种子平均后的Wilcoxon p：**{aggregate['wilcoxon']['pvalue']:.4g}**",
            "",
            "预设首轮门槛：平均优势>0.05、至少70%三元组胜出、Wilcoxon p<0.05。主要轨迹只使用类内角分位数和切空间协方差结构；每一步先减去固定频率背景类的中位变化。",
            "",
            "该实验验证共享长尾环境中的成对轨迹规律。若通过，下一步再做单对独立干预消融，以排除多探针类共同变尾产生的交互影响。",
        ]
    )
    (output_dir / "report_legacy.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def build_report(
    summaries: list[dict],
    aggregate: dict,
    triplets: list[dict],
    class_names: list[str],
    output_dir: Path,
) -> None:
    lines = [
        "# 反事实头尾角色互换轨迹实验",
        "",
        "平衡参考空间在查看本实验结果前固定目标–相似–远类三元组。每个低频节点分别训练 A、B、C 三个角色模型：每组三元组只有对应角色类别变为尾类，另外两个类别及固定背景类保持 500-shot。同一频率下三个模型具有完全相同的频率直方图。",
        "",
        "## 类别设计",
        "",
        "| 组 | A目标类 | B相似类 | C远类 | A–B KNN | A–C KNN |",
        "|---:|---|---|---|---:|---:|",
    ]
    for triplet in triplets:
        lines.append(
            f"| {triplet['triplet_id']} | "
            f"{class_names[triplet['anchor']]} | "
            f"{class_names[triplet['similar']]} | "
            f"{class_names[triplet['far']]} | "
            f"{triplet['anchor_similar_knn']:.4f} | "
            f"{triplet['anchor_far_knn']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 每个随机种子的主要结果",
            "",
            "| Seed | A–B轨迹余弦 | A–C轨迹余弦 | 差值 | A–B胜出比例 | Wilcoxon p |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for summary in summaries:
        row = summary["aggregate"]["primary_trajectory_cosine"]
        lines.append(
            f"| {summary['seed']} | {row['similar_mean']:.4f} | "
            f"{row['far_mean']:.4f} | {row['delta_mean']:+.4f} | "
            f"{row['similar_better_fraction'] * 100:.1f}% | "
            f"{row['wilcoxon']['pvalue']:.4g} |"
        )
    verdict = (
        "初步支持：平衡空间相似类的支撑区域频率响应轨迹比远类更一致。"
        if aggregate["preliminary_supported"]
        else "暂未通过：相似类相对远类的轨迹优势未同时达到效应量、稳定性和显著性门槛。"
    )
    lines.extend(
        [
            "",
            "## 跨种子预注册判定",
            "",
            f"**{verdict}**",
            "",
            f"- A–B相对A–C的平均轨迹余弦优势：**{aggregate['delta_mean']:+.4f}**",
            f"- 三元组bootstrap 95% CI：**[{aggregate['delta_bootstrap_95ci'][0]:+.4f}, {aggregate['delta_bootstrap_95ci'][1]:+.4f}]**",
            f"- A–B胜出的三元组比例：**{aggregate['similar_better_fraction'] * 100:.1f}%**",
            f"- 三元组先跨种子平均后的Wilcoxon p：**{aggregate['wilcoxon']['pvalue']:.4g}**",
            f"- 精确配对符号置换p：**{aggregate['exact_sign_flip_pvalue']:.4g}**",
            f"- 所有随机种子效应方向为正：**{aggregate['all_seed_effect_directions_positive']}**",
            f"- 同类跨种子轨迹余弦：**{aggregate['h1_cross_seed_repeatability']['same_class_cross_seed_cosine_mean']}**",
            f"- 随机跨类跨种子轨迹余弦：**{aggregate['h1_cross_seed_repeatability']['random_cross_class_cross_seed_cosine_mean']}**",
            "",
            "主要轨迹由固定平衡测试集上的角分位数、切空间协方差迹、有效秩和前四个归一化特征值构成。每一步先减去固定背景类的中位变化，再按共享500-shot模型的类间尺度归一化。",
            "",
            "预设通过条件：平均优势>0.05、至少70%三元组胜出、Wilcoxon p<0.05，并且不同随机种子的平均效应方向一致。",
        ]
    )
    (output_dir / "report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    args.levels = sorted(set(args.levels))
    if args.levels != [5, 20, 50, 200, 500]:
        raise ValueError("Expected levels: 5 20 50 200 500.")
    if 3 * args.num_triplets >= NUM_CLASSES:
        raise ValueError("Triplets must leave fixed-frequency background classes.")
    if args.smoke:
        args.epochs = 1
        args.seeds = args.seeds[:1]
        args.output_dir = args.output_dir / "smoke"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    class_names = read_class_names(args.data_root)
    pair_rows = read_pair_rows(args.similarity_csv)
    triplets = select_disjoint_triplets(pair_rows, args.num_triplets)
    save_json(
        {
            "arguments": vars(args),
            "triplets": triplets,
            "triplet_names": [
                {
                    "triplet_id": triplet["triplet_id"],
                    "anchor": class_names[triplet["anchor"]],
                    "similar": class_names[triplet["similar"]],
                    "far": class_names[triplet["far"]],
                }
                for triplet in triplets
            ],
            "invariants": {
                "counterfactual_roles": ["A", "B", "C"],
                "one_tail_role_per_triplet_per_model": True,
                "matched_frequency_histogram_across_roles": True,
                "background_count": 500,
                "nested_subsets": True,
                "trained_from_scratch": True,
                "balanced_reference_weights_used_for_initialization": False,
                "same_initialization_within_seed": True,
                "fixed_samples_and_optimizer_steps_per_epoch": True,
                "support_measured_on_fixed_balanced_test_set": True,
                "primary_descriptors": PRIMARY_NAMES,
            },
        },
        args.output_dir / "design.json",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(
        f"Triplets={len(triplets)}, probe classes={3 * len(triplets)}, "
        f"background classes={NUM_CLASSES - 3 * len(triplets)}",
        flush=True,
    )

    raw_train = torchvision.datasets.CIFAR100(
        root=args.data_root, train=True, download=False
    )
    raw_test = torchvision.datasets.CIFAR100(
        root=args.data_root, train=False, download=False
    )
    class_orders = make_nested_class_orders(
        np.asarray(raw_train.targets, dtype=np.int64),
        subset_seed=args.subset_seed,
    )
    if args.design_only:
        train_targets = np.asarray(raw_train.targets, dtype=np.int64)
        role_fields = {"A": "anchor", "B": "similar", "C": "far"}
        probe_classes = {
            class_id
            for triplet in triplets
            for class_id in (
                triplet["anchor"],
                triplet["similar"],
                triplet["far"],
            )
        }
        rows = []
        specifications = [("H", max(args.levels), set())]
        for count in sorted(args.levels, reverse=True):
            if count == max(args.levels):
                continue
            for role, field in role_fields.items():
                specifications.append(
                    (
                        role,
                        count,
                        {triplet[field] for triplet in triplets},
                    )
                )
        for role, count, tail_classes in specifications:
            indices = select_frequency_indices(
                class_orders, tail_classes, count
            )
            observed = np.bincount(
                train_targets[indices], minlength=NUM_CLASSES
            )
            expected = np.asarray(
                [
                    count if class_id in tail_classes else 500
                    for class_id in range(NUM_CLASSES)
                ]
            )
            if not np.array_equal(observed, expected):
                raise AssertionError(
                    f"Count validation failed for role={role}, n={count}."
                )
            rows.append(
                {
                    "model_role": role,
                    "count": count,
                    "num_tail_classes": len(tail_classes),
                    "num_head_classes": NUM_CLASSES - len(tail_classes),
                    "dataset_size": len(indices),
                    "frequency_histogram": (
                        f"{len(tail_classes)}x{count} + "
                        f"{NUM_CLASSES - len(tail_classes)}x500"
                    ),
                    "probe_classes_are_disjoint": (
                        len(probe_classes) == 3 * len(triplets)
                    ),
                }
            )
        write_csv(rows, args.output_dir / "design_validation.csv")
        print(
            f"Design validated: {len(rows)} unique models; "
            f"{args.output_dir / 'design_validation.csv'}",
            flush=True,
        )
        return
    summaries = [
        run_seed(
            seed,
            raw_train,
            raw_test,
            class_orders,
            triplets,
            class_names,
            args,
            device,
        )
        for seed in args.seeds
    ]
    save_json(summaries, args.output_dir / "all_seed_summaries.json")
    aggregate = aggregate_across_seeds(
        summaries, triplets, args.output_dir
    )
    build_report(
        summaries, aggregate, triplets, class_names, args.output_dir
    )
    print(f"Finished: {args.output_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
