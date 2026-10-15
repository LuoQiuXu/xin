from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader

from lt_rank_swap_experiment import (
    PRIMARY_NAMES,
    cosine,
    evaluate_checkpoint,
    exponential_counts,
    indices_checksum,
    make_rank_swap_design,
    primary_shape_state,
    read_class_names,
    select_indices,
    train_model,
)
from support_dynamics_experiment import (
    DESCRIPTOR_INDEX,
    DESCRIPTOR_NAMES,
    NUM_CLASSES,
    ArrayCIFAR100,
    eval_transform,
    l2_normalize,
    make_nested_class_orders,
    normalize_vector,
    orthogonal_procrustes,
    save_json,
    train_transform,
    write_csv,
)


DEFAULT_RANKS = [1, 15, 29, 43, 58, 72, 86, 100]


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Eight-point counter-sweep of a similar CIFAR-100 class pair. "
            "The other 98 classes and their exact images remain fixed."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(r"D:\xiangmuxunlian\hafe_experiment\data"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "outputs" / "lt_frequency_sweep",
    )
    parser.add_argument(
        "--endpoint-root",
        type=Path,
        default=project_dir / "outputs" / "lt_rank_swap",
    )
    parser.add_argument("--class-a", type=int, default=47)
    parser.add_argument("--class-b", type=int, default=52)
    parser.add_argument("--ranks", type=int, nargs="+", default=DEFAULT_RANKS)
    parser.add_argument("--imbalance-factor", type=float, default=100.0)
    parser.add_argument("--max-count", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--samples-per-epoch", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--subset-seed", type=int, default=20260729)
    parser.add_argument("--rank-seed", type=int, default=20260801)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Reuse compatible intermediate checkpoints and train missing ones.",
    )
    parser.add_argument("--design-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def make_sweep_design(args: argparse.Namespace, curve: np.ndarray) -> dict:
    ranks = np.asarray(args.ranks, dtype=np.int64)
    if len(ranks) < 3:
        raise ValueError("At least three frequency ranks are required.")
    if not np.all(ranks[:-1] < ranks[1:]):
        raise ValueError("Ranks must be strictly increasing.")
    if int(ranks[0]) != 1 or int(ranks[-1]) != NUM_CLASSES:
        raise ValueError("The sweep must include ranks 1 and 100.")
    if not np.array_equal(ranks + ranks[::-1], np.full(len(ranks), 101)):
        raise ValueError("Ranks must be symmetric: r_k + r_(K-1-k) = 101.")

    endpoint_design = make_rank_swap_design(
        args.class_a, args.class_b, curve, args.rank_seed
    )
    background = endpoint_design["background_classes"]
    background_counts = endpoint_design["counts_ab"].copy()
    states = []
    for rank in ranks:
        mirror_rank = NUM_CLASSES + 1 - int(rank)
        counts = background_counts.copy()
        counts[args.class_a] = curve[int(rank) - 1]
        counts[args.class_b] = curve[mirror_rank - 1]
        states.append(
            {
                "rank_a": int(rank),
                "rank_b": mirror_rank,
                "count_a": int(counts[args.class_a]),
                "count_b": int(counts[args.class_b]),
                "counts": counts,
                "name": (
                    f"rank_{int(rank):03d}_"
                    f"A{int(counts[args.class_a])}_"
                    f"B{int(counts[args.class_b])}"
                ),
                "strict_if_histogram": bool(
                    np.array_equal(np.sort(counts), np.sort(curve))
                ),
            }
        )
    a_counts = np.asarray([state["count_a"] for state in states])
    b_counts = np.asarray([state["count_b"] for state in states])
    if not np.array_equal(a_counts, b_counts[::-1]):
        raise AssertionError("A and reversed-B do not share the same shot grid.")
    if not np.all(a_counts[:-1] >= a_counts[1:]):
        raise AssertionError("A counts must decrease from head to tail.")
    return {
        "ranks": ranks,
        "states": states,
        "background_classes": background,
        "background_counts": background_counts[background],
        "a_counts_head_to_tail": a_counts,
        "b_counts_tail_to_head": b_counts,
    }


def endpoint_path(
    seed: int, state: dict, args: argparse.Namespace
) -> Path | None:
    if state["rank_a"] == 1:
        return (
            args.endpoint_root
            / f"seed_{seed}"
            / "model_A_head_B_tail_final.pth"
        )
    if state["rank_a"] == NUM_CLASSES:
        return (
            args.endpoint_root
            / f"seed_{seed}"
            / "model_B_head_A_tail_final.pth"
        )
    return None


def validate_checkpoint(
    path: Path,
    seed: int,
    expected_counts: np.ndarray,
    args: argparse.Namespace,
) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required endpoint checkpoint is missing: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checks = {
        "seed": int(checkpoint.get("seed", -1)) == seed,
        "epochs": int(checkpoint.get("epochs", -1)) == args.epochs,
        "samples_per_epoch": (
            int(checkpoint.get("samples_per_epoch", -1))
            == args.samples_per_epoch
        ),
        "counts": np.array_equal(
            np.asarray(checkpoint.get("counts", []), dtype=np.int64),
            expected_counts,
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"Incompatible endpoint checkpoint {path}: {checks}")
    return checkpoint


def build_audit(
    args: argparse.Namespace,
    design: dict,
    class_orders: list[np.ndarray],
    class_names: list[str],
) -> dict:
    background = design["background_classes"]
    rows = []
    background_checksums = []
    selected_by_class = {args.class_a: [], args.class_b: []}
    for state in design["states"]:
        counts = state["counts"]
        indices = select_indices(class_orders, counts)
        background_indices = np.concatenate(
            [
                class_orders[class_id][: int(counts[class_id])]
                for class_id in background
            ]
        )
        background_checksums.append(indices_checksum(background_indices))
        selected_by_class[args.class_a].append(
            set(class_orders[args.class_a][: state["count_a"]].tolist())
        )
        selected_by_class[args.class_b].append(
            set(class_orders[args.class_b][: state["count_b"]].tolist())
        )
        rows.append(
            {
                "rank_a": state["rank_a"],
                "rank_b": state["rank_b"],
                "count_a": state["count_a"],
                "count_b": state["count_b"],
                "dataset_size": int(len(indices)),
                "realized_max_count": int(counts.max()),
                "realized_min_count": int(counts.min()),
                "realized_imbalance_factor": float(
                    counts.max() / counts.min()
                ),
                "strict_if_histogram": state["strict_if_histogram"],
                "background_checksum": background_checksums[-1],
            }
        )
    a_nested = all(
        selected_by_class[args.class_a][index + 1]
        <= selected_by_class[args.class_a][index]
        for index in range(len(rows) - 1)
    )
    b_nested = all(
        selected_by_class[args.class_b][index]
        <= selected_by_class[args.class_b][index + 1]
        for index in range(len(rows) - 1)
    )
    audit = {
        "class_a": {"id": args.class_a, "name": class_names[args.class_a]},
        "class_b": {"id": args.class_b, "name": class_names[args.class_b]},
        "frequency_source": "selected ranks from the IF=100 exponential curve",
        "intervention_design": (
            "A decreases while B increases; the other 98 class counts and "
            "exact selected images are fixed."
        ),
        "strict_full_if_histogram_at_intermediate_points": False,
        "constant_optimizer_draws_per_epoch": args.samples_per_epoch,
        "states": rows,
        "background_counts_identical_all_states": True,
        "background_indices_identical_all_states": (
            len(set(background_checksums)) == 1
        ),
        "a_subsets_nested_head_to_tail": a_nested,
        "b_subsets_nested_tail_to_head": b_nested,
        "same_head_to_tail_shot_grid": bool(
            np.array_equal(
                design["a_counts_head_to_tail"],
                design["b_counts_tail_to_head"][::-1],
            )
        ),
        "endpoint_models_reused_in_formal_run": not args.smoke,
        "arguments": vars(args),
    }
    required = [
        audit["background_counts_identical_all_states"],
        audit["background_indices_identical_all_states"],
        audit["a_subsets_nested_head_to_tail"],
        audit["b_subsets_nested_tail_to_head"],
        audit["same_head_to_tail_shot_grid"],
    ]
    if not all(required):
        raise AssertionError(f"Frequency-sweep audit failed: {audit}")
    write_csv(rows, args.output_dir / "frequency_states.csv")
    save_json(audit, args.output_dir / "design_audit.json")
    return audit


def rank_values(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    unique, inverse, counts = np.unique(
        values, return_inverse=True, return_counts=True
    )
    if np.any(counts > 1):
        for unique_index in range(len(unique)):
            mask = inverse == unique_index
            ranks[mask] = ranks[mask].mean()
    return ranks


def correlation(first: np.ndarray, second: np.ndarray) -> float:
    if np.std(first) <= 1e-12 or np.std(second) <= 1e-12:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


def spearman(first: np.ndarray, second: np.ndarray) -> float:
    return correlation(rank_values(first), rank_values(second))


def align_prototypes(
    states: list[dict],
    background: np.ndarray,
) -> tuple[np.ndarray, list[float]]:
    base_features = l2_normalize(states[0]["features"].astype(np.float64))
    labels = states[0]["labels"]
    anchor_mask = np.isin(labels, background)
    aligned_all = []
    alignment_cosines = []
    for state in states:
        rotation = orthogonal_procrustes(
            state["features"][anchor_mask],
            states[0]["features"][anchor_mask],
        )
        aligned = l2_normalize(
            state["features"].astype(np.float64) @ rotation
        )
        aligned_all.append(aligned)
        alignment_cosines.append(
            float(
                np.mean(
                    np.sum(
                        aligned[anchor_mask] * base_features[anchor_mask],
                        axis=1,
                    )
                )
            )
        )
    prototypes = np.zeros(
        (len(states), NUM_CLASSES, base_features.shape[1]), dtype=np.float64
    )
    for state_index, aligned in enumerate(aligned_all):
        for class_id in range(NUM_CLASSES):
            prototypes[state_index, class_id] = normalize_vector(
                aligned[labels == class_id].mean(axis=0)
            )
    return prototypes, alignment_cosines


def background_corrected_trajectory(
    values: np.ndarray,
    role_class: int,
    background: np.ndarray,
    source_indices: list[int],
    scale: np.ndarray,
) -> np.ndarray:
    baseline = source_indices[0]
    trajectory = np.zeros((len(source_indices), values.shape[-1]))
    for output_index, source_index in enumerate(source_indices):
        difference = values[source_index] - values[baseline]
        background_median = np.median(difference[background], axis=0)
        trajectory[output_index] = (
            difference[role_class] - background_median
        ) / scale
    return trajectory


def analyze_seed(
    seed: int,
    states: list[dict],
    design: dict,
    class_names: list[str],
    args: argparse.Namespace,
    seed_dir: Path,
) -> dict:
    primary = np.asarray(
        [primary_shape_state(state["descriptors"]) for state in states]
    )
    scale = np.std(primary.reshape(-1, primary.shape[-1]), axis=0, ddof=1)
    scale = np.maximum(scale, 1e-8)
    background = design["background_classes"]
    forward = list(range(len(states)))
    reverse = list(reversed(forward))
    trajectory_a = background_corrected_trajectory(
        primary, args.class_a, background, forward, scale
    )
    trajectory_b = background_corrected_trajectory(
        primary, args.class_b, background, reverse, scale
    )
    shots = design["a_counts_head_to_tail"].astype(np.int64)
    scarcity = np.log(args.max_count / shots) / math.log(
        args.imbalance_factor
    )

    response_rows = []
    state_cosines = []
    for point_index, shot in enumerate(shots):
        state_cosine = (
            None
            if point_index == 0
            else cosine(trajectory_a[point_index], trajectory_b[point_index])
        )
        if state_cosine is not None:
            state_cosines.append(state_cosine)
        for metric_index, metric in enumerate(PRIMARY_NAMES):
            response_rows.append(
                {
                    "seed": seed,
                    "point": point_index,
                    "shots": int(shot),
                    "scarcity": float(scarcity[point_index]),
                    "metric": metric,
                    "response_a": float(
                        trajectory_a[point_index, metric_index]
                    ),
                    "response_b": float(
                        trajectory_b[point_index, metric_index]
                    ),
                    "state_response_cosine": state_cosine,
                }
            )
    write_csv(response_rows, seed_dir / "trajectory_responses.csv")

    step_a = np.diff(trajectory_a, axis=0)
    step_b = np.diff(trajectory_b, axis=0)
    step_rows = []
    step_cosines = []
    for step_index in range(len(step_a)):
        step_cosine = cosine(step_a[step_index], step_b[step_index])
        step_cosines.append(step_cosine)
        step_rows.append(
            {
                "seed": seed,
                "from_shots": int(shots[step_index]),
                "to_shots": int(shots[step_index + 1]),
                "step_response_cosine": step_cosine,
                "step_norm_a": float(np.linalg.norm(step_a[step_index])),
                "step_norm_b": float(np.linalg.norm(step_b[step_index])),
            }
        )
    write_csv(step_rows, seed_dir / "trajectory_steps.csv")

    raw_rows = []
    for role, class_id, source_indices in [
        ("A", args.class_a, forward),
        ("B", args.class_b, reverse),
    ]:
        for point_index, source_index in enumerate(source_indices):
            state = states[source_index]
            row = {
                "seed": seed,
                "role": role,
                "class_id": class_id,
                "class_name": class_names[class_id],
                "point": point_index,
                "shots": int(shots[point_index]),
                "source_rank_a": design["states"][source_index]["rank_a"],
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
            raw_rows.append(row)
    write_csv(raw_rows, seed_dir / "trajectory_support_states.csv")

    metric_rows = []
    for metric_index, metric in enumerate(PRIMARY_NAMES):
        metric_rows.append(
            {
                "seed": seed,
                "metric": metric,
                "spearman_a_vs_scarcity": spearman(
                    scarcity, trajectory_a[:, metric_index]
                ),
                "spearman_b_vs_scarcity": spearman(
                    scarcity, trajectory_b[:, metric_index]
                ),
                "curve_correlation_a_b": correlation(
                    trajectory_a[:, metric_index],
                    trajectory_b[:, metric_index],
                ),
            }
        )
    write_csv(metric_rows, seed_dir / "metric_trends.csv")

    prototypes, alignment_cosines = align_prototypes(states, background)
    prototype_scale = np.ones(prototypes.shape[-1], dtype=np.float64)
    prototype_a = background_corrected_trajectory(
        prototypes, args.class_a, background, forward, prototype_scale
    )
    prototype_b = background_corrected_trajectory(
        prototypes, args.class_b, background, reverse, prototype_scale
    )
    prototype_state_cosines = [
        cosine(prototype_a[index], prototype_b[index])
        for index in range(1, len(states))
    ]
    prototype_step_cosines = [
        cosine(first, second)
        for first, second in zip(
            np.diff(prototype_a, axis=0), np.diff(prototype_b, axis=0)
        )
    ]

    summary = {
        "seed": seed,
        "shots_head_to_tail": shots,
        "trajectory_a": trajectory_a,
        "trajectory_b": trajectory_b,
        "flattened_trajectory_cosine": cosine(
            trajectory_a[1:].reshape(-1), trajectory_b[1:].reshape(-1)
        ),
        "flattened_trajectory_correlation": correlation(
            trajectory_a[1:].reshape(-1), trajectory_b[1:].reshape(-1)
        ),
        "state_response_cosines": state_cosines,
        "state_response_cosine_mean": float(np.mean(state_cosines)),
        "step_response_cosines": step_cosines,
        "step_response_cosine_mean": float(np.mean(step_cosines)),
        "metric_sign_agreement": float(
            np.mean(np.sign(trajectory_a[1:]) == np.sign(trajectory_b[1:]))
        ),
        "accuracy_a_head_to_tail": [
            float(states[index]["accuracy"][args.class_a])
            for index in forward
        ],
        "accuracy_b_head_to_tail": [
            float(states[index]["accuracy"][args.class_b])
            for index in reverse
        ],
        "alignment_cosines_to_rank1_reference": alignment_cosines,
        "prototype_state_cosines": prototype_state_cosines,
        "prototype_step_cosines": prototype_step_cosines,
    }
    save_json(summary, seed_dir / "summary.json")
    return summary


def run_seed(
    seed: int,
    raw_train,
    raw_test,
    class_orders: list[np.ndarray],
    design: dict,
    class_names: list[str],
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    seed_dir = args.output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    train_targets = np.asarray(raw_train.targets, dtype=np.int64)
    test_targets = np.asarray(raw_test.targets, dtype=np.int64)
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

    checkpoints = []
    for state in design["states"]:
        existing_endpoint = None if args.smoke else endpoint_path(
            seed, state, args
        )
        if existing_endpoint is not None:
            validate_checkpoint(
                existing_endpoint, seed, state["counts"], args
            )
            print(
                f"[seed {seed} {state['name']}] "
                f"reusing endpoint {existing_endpoint.name}",
                flush=True,
            )
            checkpoints.append(existing_endpoint)
            continue
        indices = select_indices(class_orders, state["counts"])
        dataset = ArrayCIFAR100(
            raw_train.data,
            train_targets,
            indices,
            train_transform(),
        )
        checkpoints.append(
            train_model(
                seed,
                state["name"],
                state["counts"],
                dataset,
                test_loader,
                seed_dir,
                args,
                device,
            )
        )

    initial_checksums = []
    for checkpoint_path in checkpoints:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        initial_checksums.append(checkpoint["initial_checksum"])
    if len(set(initial_checksums)) != 1:
        raise AssertionError(
            f"Seed {seed} sweep models do not share initialization."
        )

    print(f"[seed {seed}] extracting eight fixed-test supports", flush=True)
    states = [
        evaluate_checkpoint(path, test_dataset, args, device)
        for path in checkpoints
    ]
    return analyze_seed(
        seed,
        states,
        design,
        class_names,
        args,
        seed_dir,
    )


def make_plots(
    summaries: list[dict],
    args: argparse.Namespace,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; skipping plots.", flush=True)
        return

    shots = np.asarray(summaries[0]["shots_head_to_tail"])
    trajectories_a = np.asarray(
        [summary["trajectory_a"] for summary in summaries]
    )
    trajectories_b = np.asarray(
        [summary["trajectory_b"] for summary in summaries]
    )
    figure, axes = plt.subplots(2, 5, figsize=(18, 7), constrained_layout=True)
    for metric_index, (axis, metric) in enumerate(
        zip(axes.flat, PRIMARY_NAMES)
    ):
        for seed_index, summary in enumerate(summaries):
            axis.plot(
                shots,
                trajectories_a[seed_index, :, metric_index],
                color="#1f77b4",
                alpha=0.25,
                linewidth=1,
            )
            axis.plot(
                shots,
                trajectories_b[seed_index, :, metric_index],
                color="#d62728",
                alpha=0.25,
                linewidth=1,
            )
        axis.plot(
            shots,
            trajectories_a[:, :, metric_index].mean(axis=0),
            "-o",
            color="#1f77b4",
            label="A",
        )
        axis.plot(
            shots,
            trajectories_b[:, :, metric_index].mean(axis=0),
            "-s",
            color="#d62728",
            label="B",
        )
        axis.set_xscale("log")
        axis.invert_xaxis()
        axis.set_title(metric)
        axis.grid(alpha=0.2)
    axes.flat[0].legend()
    figure.supxlabel("Training shots: head to tail")
    figure.supylabel("Background-corrected standardized response")
    figure.savefig(args.output_dir / "support_trajectories.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for summary in summaries:
        axis.plot(
            shots,
            np.asarray(summary["accuracy_a_head_to_tail"]) * 100,
            color="#1f77b4",
            alpha=0.3,
        )
        axis.plot(
            shots,
            np.asarray(summary["accuracy_b_head_to_tail"]) * 100,
            color="#d62728",
            alpha=0.3,
        )
    axis.plot(
        shots,
        np.mean(
            [summary["accuracy_a_head_to_tail"] for summary in summaries],
            axis=0,
        )
        * 100,
        "-o",
        color="#1f77b4",
        label="A",
    )
    axis.plot(
        shots,
        np.mean(
            [summary["accuracy_b_head_to_tail"] for summary in summaries],
            axis=0,
        )
        * 100,
        "-s",
        color="#d62728",
        label="B",
    )
    axis.set_xscale("log")
    axis.invert_xaxis()
    axis.set_xlabel("Training shots: head to tail")
    axis.set_ylabel("Balanced-test class accuracy (%)")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.savefig(args.output_dir / "accuracy_trajectories.png", dpi=180)
    plt.close(figure)


def aggregate_results(
    summaries: list[dict], args: argparse.Namespace
) -> dict:
    flattened_cosines = [
        summary["flattened_trajectory_cosine"] for summary in summaries
    ]
    result = {
        "num_seeds": len(summaries),
        "seeds": [summary["seed"] for summary in summaries],
        "shots_head_to_tail": summaries[0]["shots_head_to_tail"],
        "flattened_trajectory_cosines": flattened_cosines,
        "flattened_trajectory_cosine_mean": float(
            np.mean(flattened_cosines)
        ),
        "state_response_cosine_mean": float(
            np.mean(
                [summary["state_response_cosine_mean"] for summary in summaries]
            )
        ),
        "step_response_cosine_mean": float(
            np.mean(
                [summary["step_response_cosine_mean"] for summary in summaries]
            )
        ),
        "metric_sign_agreement_mean": float(
            np.mean(
                [summary["metric_sign_agreement"] for summary in summaries]
            )
        ),
    }
    if len(summaries) >= 2:
        result["a_cross_seed_trajectory_cosine"] = cosine(
            np.asarray(summaries[0]["trajectory_a"]).reshape(-1),
            np.asarray(summaries[1]["trajectory_a"]).reshape(-1),
        )
        result["b_cross_seed_trajectory_cosine"] = cosine(
            np.asarray(summaries[0]["trajectory_b"]).reshape(-1),
            np.asarray(summaries[1]["trajectory_b"]).reshape(-1),
        )
    else:
        result["a_cross_seed_trajectory_cosine"] = None
        result["b_cross_seed_trajectory_cosine"] = None
    save_json(result, args.output_dir / "aggregate.json")
    save_json(summaries, args.output_dir / "all_seed_summaries.json")
    make_plots(summaries, args)
    return result


def build_report(
    aggregate: dict,
    summaries: list[dict],
    class_names: list[str],
    args: argparse.Namespace,
) -> None:
    lines = [
        "# A/B八点头尾频率轨迹实验",
        "",
        f"- A：{class_names[args.class_a]}（{args.class_a}）",
        f"- B：{class_names[args.class_b]}（{args.class_b}）",
        "- 样本节点：" + " → ".join(
            str(value) for value in aggregate["shots_head_to_tail"]
        ),
        "- 其余98类的样本数及具体图片在所有中间模型中保持不变。",
        "- 中间节点的A/B样本数来自IF=100指数曲线，但完整直方图不再严格等于标准CIFAR-100-LT。",
        "",
        "## 每个随机种子",
        "",
        "| Seed | 完整轨迹余弦 | 状态响应均值 | 相邻步骤均值 | 符号一致率 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary['seed']} | "
            f"{summary['flattened_trajectory_cosine']:.4f} | "
            f"{summary['state_response_cosine_mean']:.4f} | "
            f"{summary['step_response_cosine_mean']:.4f} | "
            f"{summary['metric_sign_agreement'] * 100:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## 汇总",
            "",
            f"- 完整轨迹余弦均值：**{aggregate['flattened_trajectory_cosine_mean']:.4f}**",
            f"- 状态响应余弦均值：**{aggregate['state_response_cosine_mean']:.4f}**",
            f"- 相邻步骤余弦均值：**{aggregate['step_response_cosine_mean']:.4f}**",
            f"- 指标符号一致率：**{aggregate['metric_sign_agreement_mean'] * 100:.1f}%**",
            f"- A跨种子轨迹余弦：**{aggregate['a_cross_seed_trajectory_cosine']}**",
            f"- B跨种子轨迹余弦：**{aggregate['b_cross_seed_trajectory_cosine']}**",
            "",
            "应结合每项指标的单调性、相邻频率步骤和远距离类别对照判断机制是否成立。",
        ]
    )
    (args.output_dir / "report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.epochs = 1
        args.seeds = args.seeds[:1]
        args.samples_per_epoch = min(args.samples_per_epoch, 2048)
        args.num_workers = min(args.num_workers, 2)
        args.output_dir = args.output_dir / "smoke"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    class_names = read_class_names(args.data_root)
    curve = exponential_counts(args.max_count, args.imbalance_factor)
    design = make_sweep_design(args, curve)

    raw_train = torchvision.datasets.CIFAR100(
        root=args.data_root, train=True, download=False
    )
    raw_test = torchvision.datasets.CIFAR100(
        root=args.data_root, train=False, download=False
    )
    train_targets = np.asarray(raw_train.targets, dtype=np.int64)
    class_orders = make_nested_class_orders(
        train_targets, subset_seed=args.subset_seed
    )
    build_audit(args, design, class_orders, class_names)
    print(
        f"Device candidate: {'cuda' if torch.cuda.is_available() else 'cpu'}",
        flush=True,
    )
    print(
        "A head-to-tail shots: "
        + " -> ".join(str(value) for value in design["a_counts_head_to_tail"]),
        flush=True,
    )
    print(
        f"New models required: "
        f"{len(args.seeds) * (len(design['states']) if args.smoke else len(design['states']) - 2)}",
        flush=True,
    )
    if args.design_only:
        print(f"Design audit written to {args.output_dir}", flush=True)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    summaries = [
        run_seed(
            seed,
            raw_train,
            raw_test,
            class_orders,
            design,
            class_names,
            args,
            device,
        )
        for seed in args.seeds
    ]
    aggregate = aggregate_results(summaries, args)
    build_report(aggregate, summaries, class_names, args)
    print(f"Finished: {args.output_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
