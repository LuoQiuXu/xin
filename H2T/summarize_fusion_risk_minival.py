#!/usr/bin/env python
"""Aggregate repeated fixed-feature fusion-risk minival runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METHODS = ("pi_h2t_center", "recipient_boundary", "donor_conditioned")
BASELINES = METHODS[:2]


def parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("expected at least one seed")
    return seeds


def interpolate(curve: pd.DataFrame, budget: float, metric: str) -> float:
    ordered = curve.sort_values("mean_lambda")
    return float(np.interp(budget, ordered["mean_lambda"], ordered[metric]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate fusion-risk minival seeds")
    parser.add_argument(
        "--root", type=Path, default=Path("outputs/fusion_risk_minival")
    )
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds("42,43,44"))
    parser.add_argument("--run-pattern", default="cifar100_if100_seed{seed}_cap10_scaled")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/fusion_risk_minival/cifar100_if100_aggregate"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    prediction_frames: list[pd.DataFrame] = []
    matched_frames: list[pd.DataFrame] = []
    frontier_rows: list[dict[str, float | int | bool]] = []
    area_rows: list[dict[str, float | int | str]] = []

    for seed in args.seeds:
        run_dir = args.root / args.run_pattern.format(seed=seed)
        prediction = pd.read_csv(run_dir / "prediction.csv")
        prediction["seed"] = seed
        prediction_frames.append(prediction)

        matched = pd.read_csv(run_dir / "matched_control.csv")
        matched["seed"] = seed
        maximum = float(matched["target_mean_lambda"].max())
        matched["budget_fraction"] = (matched["target_mean_lambda"] / maximum).round(2)
        matched_frames.append(matched)

        control = pd.read_csv(run_dir / "control_curve.csv")
        donor_curve = control[control["method"] == "donor_conditioned"].copy()
        baseline_curves = {
            method: control[control["method"] == method] for method in BASELINES
        }
        common_maximum = min(
            float(curve["mean_lambda"].max()) for curve in baseline_curves.values()
        )
        for _, donor_row in donor_curve.iterrows():
            budget = float(donor_row["mean_lambda"])
            if budget > common_maximum + 1e-12:
                continue
            reference_flip = {
                method: interpolate(curve, budget, "harmful_flip_rate")
                for method, curve in baseline_curves.items()
            }
            reference_accuracy = {
                method: interpolate(curve, budget, "probe_accuracy")
                for method, curve in baseline_curves.items()
            }
            reference_loss = {
                method: interpolate(curve, budget, "mean_probe_loss_delta")
                for method, curve in baseline_curves.items()
            }
            donor_flip = float(donor_row["harmful_flip_rate"])
            donor_accuracy = float(donor_row["probe_accuracy"])
            donor_loss = float(donor_row["mean_probe_loss_delta"])
            frontier_rows.append(
                {
                    "seed": seed,
                    "gamma": float(donor_row["aggressiveness"]),
                    "mean_lambda": budget,
                    "donor_flip": donor_flip,
                    "best_baseline_flip": min(reference_flip.values()),
                    "flip_gain": min(reference_flip.values()) - donor_flip,
                    "donor_accuracy": donor_accuracy,
                    "best_baseline_accuracy": max(reference_accuracy.values()),
                    "accuracy_gain": donor_accuracy - max(reference_accuracy.values()),
                    "donor_loss_delta": donor_loss,
                    "best_baseline_loss_delta": min(reference_loss.values()),
                    "loss_gain": min(reference_loss.values()) - donor_loss,
                    "passes_flip_and_accuracy": bool(
                        donor_flip <= min(reference_flip.values()) + 1e-12
                        and donor_accuracy >= max(reference_accuracy.values()) - 1e-12
                    ),
                }
            )

        with (run_dir / "summary.json").open(encoding="utf-8") as handle:
            summary = json.load(handle)
        for method, value in summary["normalized_harmful_flip_area"].items():
            area_rows.append({"seed": seed, "method": method, "area": value})

    prediction_all = pd.concat(prediction_frames, ignore_index=True)
    prediction_flip = prediction_all[
        prediction_all["harm_definition"] == "probe_correct_to_wrong"
    ]
    prediction_summary = (
        prediction_flip.groupby(["candidate_lambda", "method"])[
            ["roc_auc", "average_precision", "harm_prevalence"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    prediction_summary.columns = [
        "_".join(item).rstrip("_") if isinstance(item, tuple) else item
        for item in prediction_summary.columns
    ]

    matched_all = pd.concat(matched_frames, ignore_index=True)
    control_summary = (
        matched_all.groupby(["budget_fraction", "method"])[
            ["harmful_flip_rate", "probe_accuracy", "mean_probe_loss_delta"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    control_summary.columns = [
        "_".join(item).rstrip("_") if isinstance(item, tuple) else item
        for item in control_summary.columns
    ]

    frontier = pd.DataFrame(frontier_rows)
    frontier_summary = (
        frontier.groupby("gamma")[["flip_gain", "accuracy_gain", "loss_gain"]]
        .agg(["mean", "std", "min"])
        .reset_index()
    )
    frontier_summary.columns = [
        "_".join(item).rstrip("_") if isinstance(item, tuple) else item
        for item in frontier_summary.columns
    ]
    stable_gamma = (
        frontier.groupby("gamma")["passes_flip_and_accuracy"].all()
    )
    passing_gamma = stable_gamma[stable_gamma].index.to_numpy(dtype=float)
    maximum_stable_gamma = float(passing_gamma.max()) if passing_gamma.size else None

    area = pd.DataFrame(area_rows)
    area_summary = area.groupby("method")["area"].agg(["mean", "std"]).reset_index()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_summary.to_csv(args.output_dir / "prediction_aggregate.csv", index=False)
    control_summary.to_csv(args.output_dir / "control_aggregate.csv", index=False)
    frontier.to_csv(args.output_dir / "policy_frontier_by_seed.csv", index=False)
    frontier_summary.to_csv(args.output_dir / "policy_frontier_aggregate.csv", index=False)
    area_summary.to_csv(args.output_dir / "control_area_aggregate.csv", index=False)
    aggregate = {
        "seeds": args.seeds,
        "maximum_gamma_passing_flip_and_accuracy_for_every_seed": maximum_stable_gamma,
        "note": (
            "lambda_critical is a boundary-hitting upper bound.  Use gamma < 1 "
            "to keep a strict interior margin."
        ),
    }
    with (args.output_dir / "aggregate_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2)

    print(prediction_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nControl area (lower is better)")
    print(area_summary.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print(f"\nmaximum gamma passing flip+accuracy for every seed: {maximum_stable_gamma}")
    print(f"wrote aggregate outputs to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
