from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader

from lt_rank_swap_experiment import (
    PRIMARY_NAMES,
    evaluate_checkpoint,
    primary_shape_state,
    read_class_names,
    train_model,
)
from support_dynamics_experiment import (
    NUM_CLASSES,
    ArrayCIFAR100,
    eval_transform,
    make_latin_frequency_design,
    make_nested_class_orders,
    save_json,
    train_transform,
    write_csv,
)


STABLE_METRICS = ["angle_q50", "angle_q95", "tangent_trace"]


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Leakage-free all-class Latin frequency experiment for testing the "
            "ordered hypothesis similar-support > random-support > distant-support."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(r"D:\xiangmuxunlian\hafe_experiment\data"),
    )
    parser.add_argument(
        "--balanced-pairs",
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
        default=project_dir / "outputs" / "latin_donor_trajectory_controls",
    )
    parser.add_argument(
        "--levels", type=int, nargs="+", default=[4, 14, 56, 208, 400]
    )
    parser.add_argument("--train-pool-per-class", type=int, default=400)
    parser.add_argument("--calibration-per-class", type=int, default=100)
    parser.add_argument("--split-seed", type=int, default=20260731)
    parser.add_argument("--design-seed", type=int, default=20260802)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--samples-per-epoch", type=int, default=40000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--bootstrap-repeats", type=int, default=20000)
    parser.add_argument("--analysis-seed", type=int, default=20260803)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--design-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def checksum(indices: np.ndarray) -> str:
    return hashlib.sha256(
        np.asarray(indices, dtype=np.int64).tobytes()
    ).hexdigest()


def split_orders(
    targets: np.ndarray, args: argparse.Namespace
) -> tuple[list[np.ndarray], np.ndarray]:
    if args.train_pool_per_class + args.calibration_per_class > 500:
        raise ValueError("train pool + calibration must not exceed 500/class")
    full_orders = make_nested_class_orders(targets, args.split_seed)
    train_orders = [
        order[: args.train_pool_per_class].copy() for order in full_orders
    ]
    calibration = np.concatenate(
        [
            order[
                args.train_pool_per_class :
                args.train_pool_per_class + args.calibration_per_class
            ]
            for order in full_orders
        ]
    ).astype(np.int64)
    if set(np.concatenate(train_orders)).intersection(calibration.tolist()):
        raise AssertionError("training/calibration leakage")
    return train_orders, calibration


def select_indices(
    train_orders: list[np.ndarray], counts: np.ndarray
) -> np.ndarray:
    return np.concatenate(
        [
            train_orders[class_id][: int(counts[class_id])]
            for class_id in range(NUM_CLASSES)
        ]
    ).astype(np.int64)


def checkpoint_path(seed_dir: Path, environment: int) -> Path:
    return seed_dir / f"model_latin_env_{environment:02d}_final.pth"


def load_similarity(path: Path) -> np.ndarray:
    scores = np.full((NUM_CLASSES, NUM_CLASSES), np.nan, dtype=np.float64)
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            target = int(row["target_class"])
            candidate = int(row["candidate_class"])
            scores[target, candidate] = float(row["prototype"])
    np.fill_diagonal(scores, 1.0)
    if np.isnan(scores).any():
        raise ValueError(f"Incomplete balanced similarity matrix: {path}")
    # Prototype cosine should be symmetric. Averaging also protects against
    # harmless CSV round-off differences.
    return 0.5 * (scores + scores.T)


def cosine(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64).reshape(-1)
    second = np.asarray(second, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    return 0.0 if denominator <= 1e-12 else float(first @ second / denominator)


def rank_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    unique, inverse, counts = np.unique(
        values, return_inverse=True, return_counts=True
    )
    for index in np.flatnonzero(counts > 1):
        ranks[inverse == index] = ranks[inverse == index].mean()
    return ranks


def correlation(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if np.std(first) <= 1e-12 or np.std(second) <= 1e-12:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


def spearman(first: np.ndarray, second: np.ndarray) -> float:
    return correlation(rank_values(first), rank_values(second))


def build_trajectories(
    states: list[dict], counts_matrix: np.ndarray, levels: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    stable_indices = [PRIMARY_NAMES.index(name) for name in STABLE_METRICS]
    raw = np.asarray(
        [primary_shape_state(state["descriptors"])[:, stable_indices] for state in states],
        dtype=np.float64,
    )
    # All environments have exactly the same frequency histogram. Removing the
    # per-environment median reduces global coordinate drift without using a
    # target/donor label or the official test set.
    corrected = raw - np.median(raw, axis=1, keepdims=True)
    scale = np.std(corrected.reshape(-1, len(STABLE_METRICS)), axis=0, ddof=1)
    scale = np.maximum(scale, 1e-8)
    corrected /= scale.reshape(1, 1, -1)

    descending_levels = np.asarray(sorted(levels, reverse=True), dtype=np.int64)
    trajectories = np.zeros(
        (NUM_CLASSES, len(levels), len(STABLE_METRICS)), dtype=np.float64
    )
    environment_order = np.zeros((NUM_CLASSES, len(levels)), dtype=np.int64)
    for class_id in range(NUM_CLASSES):
        lookup = {
            int(counts_matrix[environment, class_id]): environment
            for environment in range(len(levels))
        }
        order = np.asarray([lookup[int(level)] for level in descending_levels])
        environment_order[class_id] = order
        curve = corrected[order, class_id]
        trajectories[class_id] = curve - curve[0]
    return trajectories, environment_order


def pair_rows(
    trajectories: np.ndarray,
    balanced_similarity: np.ndarray,
    class_names: list[str],
) -> list[dict]:
    rows = []
    endpoint = trajectories[:, -1]
    signs = np.sign(trajectories[:, 1:])
    for target in range(NUM_CLASSES):
        for donor in range(target + 1, NUM_CLASSES):
            first = trajectories[target, 1:]
            second = trajectories[donor, 1:]
            rows.append(
                {
                    "target_class": target,
                    "target_name": class_names[target],
                    "donor_class": donor,
                    "donor_name": class_names[donor],
                    "balanced_prototype_similarity": float(
                        balanced_similarity[target, donor]
                    ),
                    "trajectory_cosine": cosine(first, second),
                    "endpoint_cosine": cosine(endpoint[target], endpoint[donor]),
                    "sign_agreement": float(
                        np.mean(signs[target] == signs[donor])
                    ),
                    "q50_curve_correlation": correlation(
                        trajectories[target, :, 0], trajectories[donor, :, 0]
                    ),
                    "q95_curve_correlation": correlation(
                        trajectories[target, :, 1], trajectories[donor, :, 1]
                    ),
                    "trace_curve_correlation": correlation(
                        trajectories[target, :, 2], trajectories[donor, :, 2]
                    ),
                }
            )
    return rows


def donor_groups(similarity: np.ndarray, target: int, args: argparse.Namespace) -> dict:
    candidates = np.asarray(
        [class_id for class_id in range(NUM_CLASSES) if class_id != target],
        dtype=np.int64,
    )
    ordered = candidates[np.argsort(similarity[target, candidates])[::-1]]
    # Match group sizes. Random controls are sampled once from the non-extreme
    # pool with a pre-registered seed and then reused for every training seed.
    random_pool = ordered[args.top_k : -args.top_k]
    generator = np.random.default_rng(args.analysis_seed + target * 1009)
    random_donors = generator.choice(
        random_pool, size=args.top_k, replace=False
    )
    return {
        "similar": ordered[: args.top_k],
        "random": random_donors,
        "distant": ordered[-args.top_k :],
    }


def sign_flip_pvalue(values: np.ndarray, generator: np.random.Generator) -> float:
    values = np.asarray(values, dtype=np.float64)
    observed = float(np.mean(values))
    repeats = 50000
    null = np.empty(repeats, dtype=np.float64)
    for start in range(0, repeats, 1000):
        stop = min(start + 1000, repeats)
        signs = generator.choice([-1.0, 1.0], size=(stop - start, len(values)))
        null[start:stop] = np.mean(signs * values.reshape(1, -1), axis=1)
    return float((1 + np.sum(null >= observed)) / (repeats + 1))


def bootstrap_difference(
    values: np.ndarray,
    repeats: int,
    generator: np.random.Generator,
) -> dict:
    values = np.asarray(values, dtype=np.float64)
    estimates = np.empty(repeats, dtype=np.float64)
    for start in range(0, repeats, 1000):
        stop = min(start + 1000, repeats)
        indices = generator.integers(
            0, len(values), size=(stop - start, len(values))
        )
        estimates[start:stop] = values[indices].mean(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {
        "mean_difference": float(np.mean(values)),
        "bootstrap_95_ci": [float(low), float(high)],
        "fraction_positive": float(np.mean(values > 0)),
        "one_sided_sign_flip_p": sign_flip_pvalue(values, generator),
    }


def analyze_groups(
    trajectories: np.ndarray,
    similarity: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[dict], dict]:
    target_rows = []
    for target in range(NUM_CLASSES):
        groups = donor_groups(similarity, target, args)
        row = {"target_class": target}
        for group_name, donors in groups.items():
            scores = np.asarray(
                [
                    cosine(trajectories[target, 1:], trajectories[donor, 1:])
                    for donor in donors
                ],
                dtype=np.float64,
            )
            row[f"{group_name}_mean"] = float(scores.mean())
            row[f"{group_name}_std"] = float(scores.std(ddof=1))
            row[f"{group_name}_donors"] = " ".join(str(int(x)) for x in donors)
            row[f"{group_name}_similarity_mean"] = float(
                similarity[target, donors].mean()
            )
        row["strict_order"] = int(
            row["similar_mean"] > row["random_mean"] > row["distant_mean"]
        )
        target_rows.append(row)

    similar = np.asarray([row["similar_mean"] for row in target_rows])
    random = np.asarray([row["random_mean"] for row in target_rows])
    distant = np.asarray([row["distant_mean"] for row in target_rows])
    generator = np.random.default_rng(args.analysis_seed)
    upper = np.triu_indices(NUM_CLASSES, k=1)
    pair_similarities = similarity[upper]
    pair_trajectory = np.asarray(
        [
            cosine(trajectories[first, 1:], trajectories[second, 1:])
            for first, second in zip(*upper)
        ]
    )
    summary = {
        "pre_registered_primary_metrics": STABLE_METRICS,
        "category_means": {
            "similar_support": float(similar.mean()),
            "random_support": float(random.mean()),
            "distant_support": float(distant.mean()),
        },
        "strict_order_fraction_across_targets": float(
            np.mean((similar > random) & (random > distant))
        ),
        "contrasts": {
            "similar_minus_random": bootstrap_difference(
                similar - random, args.bootstrap_repeats, generator
            ),
            "random_minus_distant": bootstrap_difference(
                random - distant, args.bootstrap_repeats, generator
            ),
            "similar_minus_distant": bootstrap_difference(
                similar - distant, args.bootstrap_repeats, generator
            ),
        },
        "all_pair_spearman_balanced_similarity_vs_trajectory_similarity": spearman(
            pair_similarities, pair_trajectory
        ),
        "ordered_hypothesis_pass": bool(
            similar.mean() > random.mean() > distant.mean()
        ),
        "strong_pass": False,
    }
    summary["strong_pass"] = bool(
        summary["ordered_hypothesis_pass"]
        and summary["contrasts"]["similar_minus_random"]["bootstrap_95_ci"][0] > 0
        and summary["contrasts"]["random_minus_distant"]["bootstrap_95_ci"][0] > 0
        and summary["contrasts"]["similar_minus_random"]["one_sided_sign_flip_p"] < 0.025
        and summary["contrasts"]["random_minus_distant"]["one_sided_sign_flip_p"] < 0.025
    )
    return target_rows, summary


def make_plot(target_rows: list[dict], output: Path) -> None:
    values = [
        [row["similar_mean"] for row in target_rows],
        [row["random_mean"] for row in target_rows],
        [row["distant_mean"] for row in target_rows],
    ]
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    parts = ax.violinplot(values, showmeans=True, showextrema=False)
    colors = ["#2a9d8f", "#e9c46a", "#e76f51"]
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_alpha(0.75)
    ax.set_xticks([1, 2, 3], ["similar", "random", "distant"])
    ax.set_ylabel("Stable support trajectory cosine")
    ax.set_title("Donor proximity control (100 target classes)")
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def aggregate_seed_results(
    seed_rows: list[list[dict]], summaries: list[dict], args: argparse.Namespace
) -> dict:
    if not seed_rows:
        return {}
    groups = ["similar", "random", "distant"]
    values = {
        group: np.asarray(
            [
                [float(row[f"{group}_mean"]) for row in rows]
                for rows in seed_rows
            ],
            dtype=np.float64,
        )
        for group in groups
    }
    # Training seeds are repeated measurements of the same 100 target classes.
    # Average within target first, then bootstrap targets to avoid pretending
    # that seed x class observations are independent.
    target_means = {group: value.mean(axis=0) for group, value in values.items()}
    generator = np.random.default_rng(args.analysis_seed + 99173)
    similar = target_means["similar"]
    random = target_means["random"]
    distant = target_means["distant"]
    aggregate = {
        "seeds": [summary["seed"] for summary in summaries],
        "number_of_training_seeds": len(summaries),
        "category_means_after_averaging_seeds_within_target": {
            "similar_support": float(similar.mean()),
            "random_support": float(random.mean()),
            "distant_support": float(distant.mean()),
        },
        "per_seed_category_means": {
            group: [float(x) for x in value.mean(axis=1)]
            for group, value in values.items()
        },
        "strict_order_fraction_across_targets": float(
            np.mean((similar > random) & (random > distant))
        ),
        "contrasts": {
            "similar_minus_random": bootstrap_difference(
                similar - random, args.bootstrap_repeats, generator
            ),
            "random_minus_distant": bootstrap_difference(
                random - distant, args.bootstrap_repeats, generator
            ),
            "similar_minus_distant": bootstrap_difference(
                similar - distant, args.bootstrap_repeats, generator
            ),
        },
        "per_seed_all_pair_spearman": [
            summary[
                "all_pair_spearman_balanced_similarity_vs_trajectory_similarity"
            ]
            for summary in summaries
        ],
    }
    aggregate["ordered_hypothesis_pass"] = bool(
        similar.mean() > random.mean() > distant.mean()
    )
    aggregate["strong_pass"] = bool(
        aggregate["ordered_hypothesis_pass"]
        and aggregate["contrasts"]["similar_minus_random"]["bootstrap_95_ci"][0]
        > 0
        and aggregate["contrasts"]["random_minus_distant"]["bootstrap_95_ci"][0]
        > 0
        and aggregate["contrasts"]["similar_minus_random"][
            "one_sided_sign_flip_p"
        ]
        < 0.025
        and aggregate["contrasts"]["random_minus_distant"][
            "one_sided_sign_flip_p"
        ]
        < 0.025
    )
    return aggregate


def run_seed(
    seed: int,
    raw_train,
    train_orders: list[np.ndarray],
    calibration: np.ndarray,
    counts_matrix: np.ndarray,
    class_names: list[str],
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    seed_dir = args.output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    targets = np.asarray(raw_train.targets, dtype=np.int64)
    calibration_dataset = ArrayCIFAR100(
        raw_train.data, targets, calibration, eval_transform()
    )
    calibration_loader = DataLoader(
        calibration_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    checkpoints = []
    for environment, counts in enumerate(counts_matrix):
        path = checkpoint_path(seed_dir, environment)
        if args.skip_train and path.exists():
            saved = torch.load(path, map_location="cpu", weights_only=False)
            compatible = (
                int(saved.get("seed", -1)) == seed
                and int(saved.get("epochs", -1)) == args.epochs
                and int(saved.get("samples_per_epoch", -1)) == args.samples_per_epoch
                and int(saved.get("split_seed", -1)) == args.split_seed
                and int(saved.get("design_seed", -1)) == args.design_seed
                and np.array_equal(
                    np.asarray(saved.get("counts", []), dtype=np.int64), counts
                )
            )
            if not compatible:
                raise ValueError(f"Incompatible checkpoint: {path}")
            print(f"[seed {seed} env {environment}] reusing checkpoint", flush=True)
            checkpoints.append(path)
            continue

        indices = select_indices(train_orders, counts)
        dataset = ArrayCIFAR100(
            raw_train.data, targets, indices, train_transform()
        )
        trained = train_model(
            seed,
            f"latin_env_{environment:02d}",
            counts,
            dataset,
            calibration_loader,
            seed_dir,
            args,
            device,
        )
        saved = torch.load(trained, map_location="cpu", weights_only=False)
        saved.update(
            {
                "split_seed": args.split_seed,
                "design_seed": args.design_seed,
                "train_pool_per_class": args.train_pool_per_class,
                "calibration_per_class": args.calibration_per_class,
                "evaluation_split": "held_out_cifar100_train_calibration",
                "official_test_used": False,
            }
        )
        torch.save(saved, trained)
        checkpoints.append(trained)

    initial_checksums = [
        torch.load(path, map_location="cpu", weights_only=False)["initial_checksum"]
        for path in checkpoints
    ]
    if len(set(initial_checksums)) != 1:
        raise AssertionError("Latin environments did not share initialization")

    print(f"[seed {seed}] extracting calibration support states", flush=True)
    states = [
        evaluate_checkpoint(path, calibration_dataset, args, device)
        for path in checkpoints
    ]
    trajectories, environment_order = build_trajectories(
        states, counts_matrix, args.levels
    )
    np.save(seed_dir / "stable_trajectories.npy", trajectories)
    similarity = load_similarity(args.balanced_pairs)
    pairs = pair_rows(trajectories, similarity, class_names)
    target_rows, summary = analyze_groups(trajectories, similarity, args)
    write_csv(pairs, seed_dir / "all_pair_trajectory_similarity.csv")
    write_csv(target_rows, seed_dir / "target_group_comparisons.csv")
    make_plot(target_rows, seed_dir / "donor_group_trajectory_cosine.png")
    summary.update(
        {
            "seed": seed,
            "levels_head_to_tail": sorted(args.levels, reverse=True),
            "environment_order_by_class": environment_order,
            "calibration_accuracy_by_environment": [
                float(np.mean(state["accuracy"])) for state in states
            ],
        }
    )
    save_json(summary, seed_dir / "summary.json")
    return summary


def main() -> None:
    args = parse_args()
    args.levels = sorted(args.levels)
    if len(args.levels) != 5:
        raise ValueError("This controlled Latin design requires exactly five levels")
    if max(args.levels) > args.train_pool_per_class:
        raise ValueError("A frequency level exceeds the training pool")
    if args.smoke:
        args.epochs = 1
        args.samples_per_epoch = min(args.samples_per_epoch, 2048)
        args.num_workers = 0
        args.bootstrap_repeats = min(args.bootstrap_repeats, 1000)
        args.seeds = args.seeds[:1]
        args.output_dir = args.output_dir / "smoke"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_train = torchvision.datasets.CIFAR100(
        root=args.data_root, train=True, download=False
    )
    targets = np.asarray(raw_train.targets, dtype=np.int64)
    train_orders, calibration = split_orders(targets, args)
    counts_matrix, class_groups = make_latin_frequency_design(
        args.levels, args.design_seed
    )
    design_audit = {
        "arguments": vars(args),
        "counts_matrix": counts_matrix,
        "class_groups": class_groups,
        "environment_totals": counts_matrix.sum(axis=1),
        "identical_frequency_histograms": all(
            np.array_equal(np.sort(counts_matrix[0]), np.sort(row))
            for row in counts_matrix
        ),
        "every_class_visits_every_level": all(
            np.array_equal(np.sort(counts_matrix[:, class_id]), args.levels)
            for class_id in range(NUM_CLASSES)
        ),
        "train_pool_checksum": checksum(np.concatenate(train_orders)),
        "calibration_checksum": checksum(calibration),
        "train_calibration_overlap": 0,
        "official_test_loaded": False,
        "primary_hypothesis": "similar-support > random-support > distant-support",
        "primary_metrics": STABLE_METRICS,
        "similar_definition": f"top {args.top_k} balanced prototype neighbors",
        "random_definition": (
            f"{args.top_k} seeded random donors after excluding balanced-prototype "
            f"top {args.top_k} and bottom {args.top_k}; identical donor draw across "
            "training seeds"
        ),
        "distant_definition": f"bottom {args.top_k} balanced prototype neighbors",
    }
    save_json(design_audit, args.output_dir / "design_audit.json")
    print(
        f"Latin design: levels={args.levels}, models/seed={len(args.levels)}, "
        f"images/environment={int(counts_matrix[0].sum())}",
        flush=True,
    )
    print("Official CIFAR-100 test set is not loaded.", flush=True)
    if args.design_only:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    class_names = read_class_names(args.data_root)
    summaries = []
    seed_rows = []
    for seed in args.seeds:
        summaries.append(run_seed(
            seed,
            raw_train,
            train_orders,
            calibration,
            counts_matrix,
            class_names,
            args,
            device,
        ))
        rows_path = args.output_dir / f"seed_{seed}" / "target_group_comparisons.csv"
        with rows_path.open("r", newline="", encoding="utf-8-sig") as handle:
            seed_rows.append(list(csv.DictReader(handle)))
    save_json(summaries, args.output_dir / "all_seed_summaries.json")
    save_json(
        aggregate_seed_results(seed_rows, summaries, args),
        args.output_dir / "aggregate_hypothesis_test.json",
    )
    print(f"Finished: {args.output_dir / 'all_seed_summaries.json'}", flush=True)


if __name__ == "__main__":
    main()
