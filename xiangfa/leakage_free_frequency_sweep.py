from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader

from lt_frequency_sweep_experiment import (
    DEFAULT_RANKS,
    aggregate_results,
    analyze_seed,
    build_audit,
    build_report,
    make_plots,
    make_sweep_design,
)
from lt_rank_swap_experiment import (
    PRIMARY_NAMES,
    evaluate_checkpoint,
    exponential_counts,
    read_class_names,
    select_indices,
    train_model,
)
from support_dynamics_experiment import (
    ArrayCIFAR100,
    eval_transform,
    make_nested_class_orders,
    save_json,
    train_transform,
)


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Leakage-free eight-point A/B frequency counter-sweep. Models see "
            "only 400 CIFAR-100 train images/class; the remaining 100/class are "
            "used only to estimate trajectories. The official test set is never "
            "loaded by this stage."
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
        default=project_dir / "outputs" / "leakage_free_frequency_sweep",
    )
    parser.add_argument("--class-a", type=int, default=47)
    parser.add_argument("--class-b", type=int, default=52)
    parser.add_argument("--ranks", type=int, nargs="+", default=DEFAULT_RANKS)
    parser.add_argument("--imbalance-factor", type=float, default=100.0)
    parser.add_argument("--train-pool-per-class", type=int, default=400)
    parser.add_argument("--calibration-per-class", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--samples-per-epoch", type=int, default=40000)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--split-seed", type=int, default=20260731)
    parser.add_argument("--rank-seed", type=int, default=20260801)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--design-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def checksum(indices: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()


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
    train_union = set(np.concatenate(train_orders).tolist())
    if train_union.intersection(calibration.tolist()):
        raise AssertionError("training and calibration indices overlap")
    return train_orders, calibration


def checkpoint_path(seed_dir: Path, state: dict) -> Path:
    return seed_dir / f"model_{state['name']}_final.pth"


def cosine(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64).reshape(-1)
    second = np.asarray(second, dtype=np.float64).reshape(-1)
    return float(
        np.dot(first, second)
        / max(float(np.linalg.norm(first) * np.linalg.norm(second)), 1e-12)
    )


def stable_metric_analysis(summaries: list[dict]) -> dict:
    subsets = {
        "selected_recovery_support": [
            "angle_q50",
            "angle_q95",
            "tangent_trace",
        ],
        "all_radii_and_trace": [
            "angle_q50",
            "angle_q80",
            "angle_q95",
            "tangent_trace",
        ],
        "fine_unstable_details": [
            "angle_std",
            "tangent_eigshare_1",
            "tangent_eigshare_2",
            "tangent_eigshare_3",
            "tangent_eigshare_4",
        ],
    }
    seed_results = []
    for summary in summaries:
        first = np.asarray(summary["trajectory_a"], dtype=np.float64)[1:]
        second = np.asarray(summary["trajectory_b"], dtype=np.float64)[1:]
        per_metric = {}
        for index, name in enumerate(PRIMARY_NAMES):
            per_metric[name] = {
                "trajectory_cosine": cosine(first[:, index], second[:, index]),
                "sign_agreement": float(
                    np.mean(
                        np.sign(first[:, index]) == np.sign(second[:, index])
                    )
                ),
            }
        subset_results = {}
        for name, metric_names in subsets.items():
            indices = [PRIMARY_NAMES.index(metric) for metric in metric_names]
            subset_results[name] = {
                "metrics": metric_names,
                "trajectory_cosine": cosine(first[:, indices], second[:, indices]),
                "sign_agreement": float(
                    np.mean(
                        np.sign(first[:, indices])
                        == np.sign(second[:, indices])
                    )
                ),
            }
        seed_results.append(
            {
                "seed": summary["seed"],
                "per_metric": per_metric,
                "subsets": subset_results,
            }
        )
    return {
        "selection_rule": (
            "Use only coarse metrics with strong shared direction; retain fine "
            "spectral metrics for evaluation rather than recovery supervision."
        ),
        "seed_results": seed_results,
    }


def run_seed(
    seed: int,
    raw_train,
    train_orders: list[np.ndarray],
    calibration_indices: np.ndarray,
    design: dict,
    class_names: list[str],
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    seed_dir = args.output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    targets = np.asarray(raw_train.targets, dtype=np.int64)
    calibration_dataset = ArrayCIFAR100(
        raw_train.data,
        targets,
        calibration_indices,
        eval_transform(),
        return_index=True,
    )
    calibration_loader = DataLoader(
        ArrayCIFAR100(
            raw_train.data, targets, calibration_indices, eval_transform()
        ),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    checkpoints: list[Path] = []
    for state in design["states"]:
        path = checkpoint_path(seed_dir, state)
        if args.skip_train and path.exists():
            saved = torch.load(path, map_location="cpu", weights_only=False)
            compatible = (
                int(saved.get("seed", -1)) == seed
                and int(saved.get("epochs", -1)) == args.epochs
                and int(saved.get("samples_per_epoch", -1))
                == args.samples_per_epoch
                and np.array_equal(
                    np.asarray(saved.get("counts", []), dtype=np.int64),
                    state["counts"],
                )
                and int(saved.get("split_seed", args.split_seed))
                == args.split_seed
            )
            if not compatible:
                raise ValueError(f"Incompatible checkpoint: {path}")
            print(f"[seed {seed} {state['name']}] reusing checkpoint", flush=True)
            checkpoints.append(path)
            continue

        indices = select_indices(train_orders, state["counts"])
        dataset = ArrayCIFAR100(
            raw_train.data, targets, indices, train_transform()
        )
        trained = train_model(
            seed,
            state["name"],
            state["counts"],
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
                "train_pool_per_class": args.train_pool_per_class,
                "calibration_per_class": args.calibration_per_class,
                "evaluation_split": "held_out_cifar100_train_calibration",
            }
        )
        torch.save(saved, trained)
        checkpoints.append(trained)

    initial = []
    for path in checkpoints:
        saved = torch.load(path, map_location="cpu", weights_only=False)
        initial.append(saved["initial_checksum"])
    if len(set(initial)) != 1:
        raise AssertionError("All sweep states must start from identical weights")

    print(f"[seed {seed}] extracting held-out calibration supports", flush=True)
    states = [
        evaluate_checkpoint(path, calibration_dataset, args, device)
        for path in checkpoints
    ]
    return analyze_seed(
        seed, states, design, class_names, args, seed_dir
    )


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.epochs = 1
        args.seeds = args.seeds[:1]
        args.samples_per_epoch = min(args.samples_per_epoch, 2048)
        args.num_workers = min(args.num_workers, 2)
        args.output_dir = args.output_dir / "smoke"
    args.max_count = args.train_pool_per_class
    args.subset_seed = args.split_seed
    args.endpoint_root = Path("unused")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_train = torchvision.datasets.CIFAR100(
        root=args.data_root, train=True, download=False
    )
    targets = np.asarray(raw_train.targets, dtype=np.int64)
    train_orders, calibration_indices = split_orders(targets, args)
    curve = exponential_counts(args.max_count, args.imbalance_factor)
    design = make_sweep_design(args, curve)
    class_names = read_class_names(args.data_root)
    audit = build_audit(args, design, train_orders, class_names)
    split_audit = {
        "split_seed": args.split_seed,
        "train_pool_per_class": args.train_pool_per_class,
        "calibration_per_class": args.calibration_per_class,
        "train_pool_checksum": checksum(np.concatenate(train_orders)),
        "calibration_checksum": checksum(calibration_indices),
        "overlap_count": 0,
        "official_test_used": False,
        "counts_a": design["a_counts_head_to_tail"],
        "counts_b": design["b_counts_tail_to_head"],
        "design_audit": audit,
    }
    save_json(split_audit, args.output_dir / "split_audit.json")
    print(
        "A counts: "
        + " -> ".join(str(x) for x in design["a_counts_head_to_tail"]),
        flush=True,
    )
    print("Official CIFAR-100 test set is not loaded in this stage.", flush=True)
    if args.design_only:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    summaries = [
        run_seed(
            seed,
            raw_train,
            train_orders,
            calibration_indices,
            design,
            class_names,
            args,
            device,
        )
        for seed in args.seeds
    ]
    aggregate = aggregate_results(summaries, args)
    save_json(
        stable_metric_analysis(summaries),
        args.output_dir / "stable_metric_analysis.json",
    )
    make_plots(summaries, args)
    build_report(aggregate, summaries, class_names, args)
    print(f"Finished: {args.output_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
