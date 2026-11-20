"""Offline Logit Adjustment and cumulative classifier-row geometry diagnostics."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import traceback

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
PARENT_ROOT = REPO_ROOT / "H2T_repro_audit" / "ccsam_causal_oracle_20261119"
BASE_ROOT = REPO_ROOT / "H2T_repro_audit" / "ccsam_stage2_controlled_cifar100_20261106"
for _path in (PARENT_ROOT, BASE_ROOT, BASE_ROOT / "ccmia"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from isolated_data import file_sha
from oracle_runtime import load_context, make_models, read_json
from run_causal_oracle import _verify_postseal_data
from ccsam_common import guard_pass, state_hash

SEEDS = (123, 124, 125)
ARMS = ("C2", "BASE_C3", "TARGETED", "ORACLE_SELECTION")
TAUS = np.round(np.linspace(-1.5, 1.5, 61), 10)
EXPECTED_GROUP_COUNTS = {"Many": 340, "Medium": 450, "Few": 200}
LOGIT_ATOL, LOGIT_RTOL = 2e-6, 2e-6
GEOMETRY_ATOL, GEOMETRY_RTOL = 1e-9, 1e-8


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload) -> None:
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError(f"Refusing output outside new experiment directory: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def _write_csv(path: Path, rows, fields=None) -> None:
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError(f"Refusing output outside new experiment directory: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fields = fields or (list(rows[0]) if rows else [])
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _code_hashes():
    names = ("run_logit_geometry_diagnostics.py", "test_logit_geometry_diagnostics.py",
             "LOGIT_GEOMETRY_PROTOCOL.md", "CODEX_LOGIT_GEOMETRY_20261120.txt")
    result = {}
    for name in names:
        path = ROOT / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing required diagnostic source: {path}")
        result[name] = _sha(path)
    return result


def class_group(count: int) -> str:
    return "Many" if count > 100 else "Medium" if count >= 20 else "Few"


def logit_adjustment(logits: np.ndarray, priors: np.ndarray, tau: float) -> np.ndarray:
    logits, priors = np.asarray(logits), np.asarray(priors, dtype=np.float64)
    if logits.ndim != 2 or priors.shape != (logits.shape[1],):
        raise ValueError("Expected logits [N,C] and priors [C]")
    if not np.isfinite(logits).all() or not np.isfinite(priors).all():
        raise ValueError("Nonfinite logits or priors")
    if np.any(priors <= 0) or not np.isclose(priors.sum(), 1.0, atol=1e-10):
        raise ValueError("Priors must be positive and sum to one")
    if not np.isfinite(float(tau)):
        raise ValueError("tau must be finite")
    return logits.astype(np.float64) - float(tau) * np.log(priors)[None, :]


def compute_metrics(logits: np.ndarray, labels: np.ndarray, counts: np.ndarray) -> dict:
    logits, labels, counts = np.asarray(logits), np.asarray(labels, dtype=np.int64), np.asarray(counts, dtype=np.int64)
    if logits.shape != (len(labels), len(counts)) or labels.ndim != 1 or len(labels) == 0:
        raise ValueError("Malformed logits/labels/counts")
    if not np.isfinite(logits).all() or np.any(labels < 0) or np.any(labels >= len(counts)):
        raise ValueError("Nonfinite logits or invalid labels")
    correct, sample_counts = logits.argmax(axis=1) == labels, counts[labels]
    masks = {"Many": sample_counts > 100,
             "Medium": (sample_counts >= 20) & (sample_counts <= 100),
             "Few": sample_counts < 20}
    values = {"overall_accuracy": float(correct.mean() * 100.0),
              "macro_accuracy": float(np.mean([correct[labels == c].mean()
                                               for c in np.unique(labels)]) * 100.0),
              "n_examples": int(len(labels))}
    for group, mask in masks.items():
        if not mask.any():
            raise ValueError(f"No {group} examples in the diagnostic partition")
        values[group.lower() + "_accuracy"] = float(correct[mask].mean() * 100.0)
        values["n_" + group.lower()] = int(mask.sum())
    if abs(values["overall_accuracy"] - values["macro_accuracy"]) > 1e-10:
        raise RuntimeError("Expected class-balanced split: Overall and Macro must agree")
    return values


def select_tau(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("No Calibration sweep rows")
    return sorted(rows, key=lambda row: (-float(row["overall_accuracy"]),
                                         abs(float(row["tau"])),
                                         float(row["tau"])))[0]


def decompose_vector(delta: np.ndarray, reference: np.ndarray) -> dict:
    delta, reference = np.asarray(delta, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    if delta.ndim != 1 or reference.shape != delta.shape:
        raise ValueError("delta/reference must be same-length vectors")
    if not np.isfinite(delta).all() or not np.isfinite(reference).all():
        raise ValueError("Nonfinite geometry vector")
    ref_norm = float(np.linalg.norm(reference))
    if ref_norm <= 1e-12:
        raise ValueError("Cannot decompose against a zero/near-zero reference row")
    coefficient = float(np.dot(delta, reference) / np.dot(reference, reference))
    radial, tangent = coefficient * reference, delta - coefficient * reference
    residual = float(np.max(np.abs(radial + tangent - delta)))
    orthogonality = float(np.dot(radial, tangent))
    if not np.allclose(radial + tangent, delta, atol=GEOMETRY_ATOL, rtol=GEOMETRY_RTOL):
        raise RuntimeError("Radial+tangential does not reconstruct displacement")
    if abs(orthogonality) > GEOMETRY_ATOL + GEOMETRY_RTOL * max(
            1.0, float(np.linalg.norm(radial) * np.linalg.norm(tangent))):
        raise RuntimeError("Radial and tangential components are not orthogonal")
    new_row = reference + delta
    new_norm = float(np.linalg.norm(new_row))
    cosine = float(np.dot(reference, new_row) / (ref_norm * new_norm)) if new_norm > 1e-12 else -1.0
    total_norm, radial_signed = float(np.linalg.norm(delta)), float(np.dot(delta, reference) / ref_norm)
    tangent_norm = float(np.linalg.norm(tangent))
    radial_fraction, tangent_fraction = ((radial_signed / total_norm) ** 2,
                                         (tangent_norm / total_norm) ** 2) if total_norm > 1e-15 else (0.0, 0.0)
    return {
        "radial_vector": radial, "tangential_vector": tangent,
        "radial_signed_first_order_norm_change": radial_signed,
        "radial_component_norm": float(np.linalg.norm(radial)),
        "tangential_component_norm": tangent_norm,
        "total_displacement_norm": total_norm,
        "exact_norm_change": new_norm - ref_norm,
        "angle_change_degrees": float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))),
        "radial_energy_fraction": float(radial_fraction),
        "tangential_energy_fraction": float(tangent_fraction),
        "orthogonality_dot": orthogonality, "reconstruction_max_abs": residual,
    }


def is_dominated(point: dict, candidates: list[dict], x_key="many_accuracy",
                 y_key="few_accuracy", eps=1e-12) -> bool:
    x, y = float(point[x_key]), float(point[y_key])
    return any(float(q[x_key]) >= x - eps and float(q[y_key]) >= y - eps
               and (float(q[x_key]) > x + eps or float(q[y_key]) > y + eps)
               for q in candidates)


def pareto_frontier(rows: list[dict]) -> list[dict]:
    unique = {}
    for row in rows:
        unique.setdefault((float(row["many_accuracy"]), float(row["few_accuracy"])), row)
    points = list(unique.values())
    return [point for point in points if not is_dominated(point, points)]


def _load_classifier(context, seed: int, arm: str, status: dict):
    info = status["checkpoints"].get(arm)
    if not info:
        raise RuntimeError(f"Missing {arm} checkpoint metadata for seed {seed}")
    relative = Path(str(info["path"]).replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError("Checkpoint path escapes parent experiment")
    checkpoint_path = PARENT_ROOT / relative
    if not checkpoint_path.is_file() or file_sha(checkpoint_path) != info["sha256"]:
        raise RuntimeError(f"Checkpoint SHA mismatch: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (payload.get("seed") != seed or payload.get("arm") != arm
            or payload.get("epoch") != 30 or payload.get("official_TEST_used") is not False):
        raise RuntimeError(f"Checkpoint provenance mismatch: seed={seed}, arm={arm}")
    backbone, head, _ = make_models(context, seed, torch.device("cpu"))
    del backbone
    head.load_state_dict(payload["state_dict_classifier"], strict=True)
    actual_hash = state_hash(head)
    if actual_hash != info["classifier_state_sha256"] or actual_hash != payload.get("final_classifier_sha256"):
        raise RuntimeError(f"Classifier tensor hash mismatch: seed={seed}, arm={arm}")
    weight = head.module.fc.weight.detach().cpu().numpy().astype(np.float64, copy=True)
    bias = head.module.fc.bias.detach().cpu().numpy().astype(np.float64, copy=True)
    if weight.shape != (100, 64) or bias.shape != (100,):
        raise RuntimeError("Expected a 100 x 64 linear classifier and 100-D bias")
    del head, payload
    return weight, bias


def _linear_logits(features: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(np.asarray(features, dtype=np.float32).copy())
    w = torch.from_numpy(np.asarray(weight, dtype=np.float32).copy())
    b = torch.from_numpy(np.asarray(bias, dtype=np.float32).copy())
    with torch.no_grad():
        chunks = [F.linear(x[i:i + 512], w, b).cpu().numpy() for i in range(0, len(x), 512)]
    result = np.concatenate(chunks, axis=0)
    if not np.isfinite(result).all():
        raise RuntimeError("Nonfinite recomputed logits")
    return result


def _verify_parent_and_load_data():
    completion = read_json(PARENT_ROOT / "formal_completed.json")
    if (completion.get("state") != "completed" or completion.get("seeds") != list(SEEDS)
            or completion.get("epochs") != 30 or completion.get("actual_optimizer_steps") != 91800
            or completion.get("official_TEST_used") is not False):
        raise RuntimeError("Parent formal completion ledger is incomplete or invalid")
    context = load_context(read_json(PARENT_ROOT / "split_manifest.json"))
    context["vault"].verify_seal()
    eval_x, eval_y = _verify_postseal_data(context)
    cal_x, cal_y = context["cal_x"], context["cal_y"]
    counts = np.asarray(context["counts"], dtype=np.int64)
    expected_labels = np.r_[0, np.repeat(10, 99)]
    if (cal_x.shape != (990, 64) or eval_x.shape != (990, 64)
            or not np.array_equal(np.bincount(cal_y, minlength=100), expected_labels)
            or not np.array_equal(np.bincount(eval_y, minlength=100), expected_labels)):
        raise RuntimeError("Calibration/Evaluation shape or stratified label allocation changed")
    group_counts = {
        "Many": int(np.count_nonzero(counts[cal_y] > 100)),
        "Medium": int(np.count_nonzero((counts[cal_y] >= 20) & (counts[cal_y] <= 100))),
        "Few": int(np.count_nonzero(counts[cal_y] < 20)),
    }
    if group_counts != EXPECTED_GROUP_COUNTS:
        raise RuntimeError(f"Unexpected Calibration group counts: {group_counts}")
    eval_groups = {
        "Many": int(np.count_nonzero(counts[eval_y] > 100)),
        "Medium": int(np.count_nonzero((counts[eval_y] >= 20) & (counts[eval_y] <= 100))),
        "Few": int(np.count_nonzero(counts[eval_y] < 20)),
    }
    if eval_groups != EXPECTED_GROUP_COUNTS:
        raise RuntimeError(f"Unexpected Evaluation group counts: {eval_groups}")
    return context, cal_x, cal_y, eval_x, eval_y, counts


def _load_seed_models_and_logits(context, seed, eval_y, cal_x, cal_y, eval_x):
    status = read_json(PARENT_ROOT / "results" / f"seed_{seed}" / "status.json")
    expected_steps = {arm: 5100 for arm in
                      ("C2", "BASE_C3", "TARGETED", "RANDOM_MATCHED", "FREQUENCY_MATCHED", "ORACLE_SELECTION")}
    if (status.get("state") != "completed" or status.get("epochs_completed") != 30
            or status.get("actual_optimizer_steps") != expected_steps
            or status.get("official_TEST_used") is not False):
        raise RuntimeError(f"Invalid parent status for seed {seed}")
    guard = status.get("test_guard", {})
    if (guard.get("official_TEST_used") is not False
            or guard.get("official_test_file_open_attempts") != 0
            or guard.get("CIFAR100_train_false_attempts") != 0 or guard.get("blocked_attempts")):
        raise RuntimeError(f"Parent TEST guard was not clean for seed {seed}")
    logits_path = PARENT_ROOT / "results" / f"seed_{seed}" / "final_logits.npz"
    if not logits_path.is_file() or file_sha(logits_path) != status["final_logits_sha256"]:
        raise RuntimeError(f"Saved final logits SHA mismatch for seed {seed}")
    saved = np.load(logits_path, allow_pickle=False)
    if not np.array_equal(saved["evaluation_labels"], eval_y):
        raise RuntimeError("Saved logits labels differ from sealed Evaluation labels")
    models, logits = {}, {}
    for arm in ARMS:
        weight, bias = _load_classifier(context, seed, arm, status)
        cal_logits = _linear_logits(cal_x, weight, bias)
        eval_logits = _linear_logits(eval_x, weight, bias)
        if not np.allclose(eval_logits, saved[arm], atol=LOGIT_ATOL, rtol=LOGIT_RTOL):
            raise RuntimeError(f"Checkpoint-to-saved-logit parity failed: seed={seed}, arm={arm}")
        models[arm] = {"weight": weight, "bias": bias}
        logits[arm] = {"calibration": cal_logits, "evaluation": eval_logits}
    saved.close()
    return status, models, logits


def _sweep_seed(seed, logits, cal_y, eval_y, counts):
    priors = counts.astype(np.float64) / float(counts.sum())
    rows, lookup = [], {}
    for partition, labels in (("calibration", cal_y), ("evaluation", eval_y)):
        for arm in ARMS:
            arm_rows = []
            for tau in TAUS:
                metrics = compute_metrics(logit_adjustment(logits[arm][partition], priors, float(tau)),
                                          labels, counts)
                row = {"seed": seed, "arm": arm, "partition": partition, "tau": float(tau), **metrics}
                rows.append(row)
                arm_rows.append(row)
            lookup[(partition, arm)] = arm_rows
    selected = []
    for arm in ARMS:
        cal_best = select_tau(lookup[("calibration", arm)])
        tau_star = float(cal_best["tau"])
        eval_locked = next(r for r in lookup[("evaluation", arm)] if abs(float(r["tau"]) - tau_star) < 1e-12)
        eval_peak = select_tau(lookup[("evaluation", arm)])
        selected.append({
            "seed": seed, "arm": arm, "tau_star_calibration": tau_star,
            "calibration_peak_overall": cal_best["overall_accuracy"],
            "calibration_peak_many": cal_best["many_accuracy"],
            "calibration_peak_medium": cal_best["medium_accuracy"],
            "calibration_peak_few": cal_best["few_accuracy"],
            "evaluation_at_locked_tau_overall": eval_locked["overall_accuracy"],
            "evaluation_at_locked_tau_many": eval_locked["many_accuracy"],
            "evaluation_at_locked_tau_medium": eval_locked["medium_accuracy"],
            "evaluation_at_locked_tau_few": eval_locked["few_accuracy"],
            "evaluation_only_peak_tau_descriptive": float(eval_peak["tau"]),
            "evaluation_only_peak_overall_descriptive": eval_peak["overall_accuracy"],
            "evaluation_only_peak_used_for_selection": False,
        })
    return rows, selected


def _frontier_summary(seed, curve_rows):
    output = []
    for partition in ("calibration", "evaluation"):
        by_arm = {arm: [r for r in curve_rows if r["seed"] == seed
                        and r["partition"] == partition and r["arm"] == arm] for arm in ARMS}
        c2_front, c3_front = pareto_frontier(by_arm["C2"]), pareto_frontier(by_arm["BASE_C3"])
        output.append({
            "seed": seed, "partition": partition, "comparison": "BASE_C3_vs_C2_frontier",
            "C2_frontier_points": len(c2_front), "BASE_C3_frontier_points": len(c3_front),
            "fraction_C2_frontier_dominated_by_BASE_C3":
                sum(is_dominated(p, c3_front) for p in c2_front) / max(1, len(c2_front)),
            "fraction_BASE_C3_frontier_dominated_by_C2":
                sum(is_dominated(p, c2_front) for p in c3_front) / max(1, len(c3_front)),
        })
        base_curve = by_arm["BASE_C3"]
        for arm in ("TARGETED", "ORACLE_SELECTION"):
            point = next(r for r in by_arm[arm] if abs(float(r["tau"])) < 1e-12)
            candidates = [r for r in base_curve if r["many_accuracy"] >= point["many_accuracy"] - 1e-12]
            best_few = max((r["few_accuracy"] for r in candidates), default=None)
            output.append({
                "seed": seed, "partition": partition, "comparison": arm + "_tau0_vs_BASE_C3_curve",
                "intervention_many_at_tau0": point["many_accuracy"],
                "intervention_few_at_tau0": point["few_accuracy"],
                "dominated_by_any_BASE_C3_tau": is_dominated(point, base_curve),
                "base_c3_points_with_many_at_least_intervention": len(candidates),
                "best_BASE_C3_few_at_equal_or_higher_many": best_few,
                "intervention_few_minus_best_BASE_C3_few":
                    point["few_accuracy"] - best_few if best_few is not None else None,
                "intervention_outside_BASE_C3_many_range": not bool(candidates),
            })
    return output


def _plot_curves(curve_rows, selected_rows, partition, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    for arm in ARMS:
        arm_rows = [r for r in curve_rows if r["partition"] == partition and r["arm"] == arm]
        means = []
        for tau in TAUS:
            points = [r for r in arm_rows if abs(float(r["tau"]) - float(tau)) < 1e-12]
            means.append({"tau": float(tau),
                          "many_accuracy": float(np.mean([r["many_accuracy"] for r in points])),
                          "few_accuracy": float(np.mean([r["few_accuracy"] for r in points]))})
        line, = ax.plot([r["many_accuracy"] for r in means], [r["few_accuracy"] for r in means],
                        marker="o", markersize=2.5, linewidth=1.4, label=arm)
        zero = next(r for r in means if abs(r["tau"]) < 1e-12)
        ax.scatter([zero["many_accuracy"]], [zero["few_accuracy"]], marker="*", s=100,
                   color=line.get_color())
        for chosen in selected_rows:
            if chosen["arm"] != arm:
                continue
            seed_point = next(r for r in arm_rows if r["seed"] == chosen["seed"]
                              and abs(float(r["tau"]) - float(chosen["tau_star_calibration"])) < 1e-12)
            ax.scatter([seed_point["many_accuracy"]], [seed_point["few_accuracy"]],
                       marker="D", s=38, color=line.get_color())
    ax.set_xlabel("Many accuracy (%)")
    ax.set_ylabel("Few accuracy (%)")
    ax.set_title(f"Many–Few Pareto diagnostic ({partition}; star=tau 0, diamond=Calibration-selected tau)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"pareto_many_few_{partition}.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    for arm in ARMS:
        arm_rows = [r for r in curve_rows if r["partition"] == partition and r["arm"] == arm]
        means, sds = [], []
        for tau in TAUS:
            points = [r for r in arm_rows if abs(float(r["tau"]) - float(tau)) < 1e-12]
            values = [r["overall_accuracy"] for r in points]
            means.append(float(np.mean(values)))
            sds.append(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)
        line, = ax.plot(TAUS, means, marker="o", markersize=2.5, linewidth=1.4, label=arm)
        ax.fill_between(TAUS, np.asarray(means) - np.asarray(sds), np.asarray(means) + np.asarray(sds),
                        alpha=0.12, color=line.get_color())
        for chosen in selected_rows:
            if chosen["arm"] != arm:
                continue
            locked = next(r for r in arm_rows if r["seed"] == chosen["seed"]
                          and abs(float(r["tau"]) - float(chosen["tau_star_calibration"])) < 1e-12)
            ax.scatter([locked["tau"]], [locked["overall_accuracy"]], marker="D", s=38,
                       color=line.get_color())
    ax.axvline(0.0, linestyle="--", linewidth=0.8)
    ax.set_xlabel("tau")
    ax.set_ylabel("Overall accuracy (%)")
    ax.set_title(f"Overall accuracy vs prior-adjustment tau ({partition}; mean ± sample SD)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"overall_vs_tau_{partition}.png", dpi=180)
    plt.close(fig)


def _margin_shifts(features, labels, target, competitor, geom, target_bias,
                   competitor_delta, competitor_bias):
    selected = labels == target
    if not selected.any():
        raise RuntimeError(f"No examples for Many target class {target}")
    x = np.asarray(features[selected], dtype=np.float64)
    radial_shift = float(np.mean(x @ geom["radial_vector"]))
    tangent_shift = float(np.mean(x @ geom["tangential_vector"]))
    bias_shift = float(target_bias)
    target_total = radial_shift + tangent_shift + bias_shift
    competitor_shift = float(np.mean(x @ competitor_delta) + competitor_bias)
    return {
        "target_radial_logit_shift": radial_shift,
        "target_tangential_logit_shift": tangent_shift,
        "target_bias_logit_shift": bias_shift,
        "target_row_total_logit_shift": target_total,
        "competitor_row_logit_shift": competitor_shift,
        "full_fixed_margin_shift": target_total - competitor_shift,
        "target_examples": int(selected.sum()), "competitor_class": int(competitor),
    }


def _geometry_seed(seed, models, cal_x, cal_y, eval_x, eval_y, counts):
    component_path = PARENT_ROOT / "calibration" / f"seed_{seed}" / "source_components_epoch30.npz"
    if not component_path.is_file():
        raise FileNotFoundError(f"Missing parent source components: {component_path}")
    with np.load(component_path, allow_pickle=False) as components:
        source_weight = np.asarray(components["weight"], dtype=np.float64)
        source_bias = np.asarray(components["bias"], dtype=np.float64)
    if source_weight.shape != (100, 100, 64) or source_bias.shape != (100, 100):
        raise RuntimeError("Unexpected source_components_epoch30.npz shapes")
    if not np.isfinite(source_weight).all() or not np.isfinite(source_bias).all():
        raise RuntimeError("Nonfinite source-conditioned parameter displacement")
    many_sources = np.flatnonzero(counts > 100)
    if np.count_nonzero(source_weight[many_sources]) or np.count_nonzero(source_bias[many_sources]):
        raise RuntimeError("Many input labels must have exactly zero direct perturbation contribution")
    target_ids = np.intersect1d(np.flatnonzero(counts > 100), np.unique(cal_y))
    source_ids = np.flatnonzero(counts <= 100)
    if len(target_ids) != 34 or len(source_ids) != 65 or len(target_ids) * len(source_ids) != 2210:
        raise RuntimeError("Unexpected eligible source-target family")
    competitor_map = read_json(PARENT_ROOT / "calibration" / f"seed_{seed}" / "competitors.json")
    W2, b2 = models["C2"]["weight"], models["C2"]["bias"]
    W3, b3 = models["BASE_C3"]["weight"], models["BASE_C3"]["bias"]
    pair_rows, target_rows = [], []

    for target in target_ids:
        reference = W2[target]
        direct_total = source_weight[:, target, :].sum(axis=0)
        observed = W3[target] - W2[target]
        remainder = observed - direct_total
        direct_geom = decompose_vector(direct_total, reference)
        observed_geom = decompose_vector(observed, reference)
        remainder_geom = decompose_vector(remainder, reference)
        direct_bias = float(source_bias[:, target].sum())
        observed_bias = float(b3[target] - b2[target])
        remainder_bias = observed_bias - direct_bias
        target_rows.append({
            "seed": seed, "target_class": int(target), "target_train_count": int(counts[target]),
            "target_group": class_group(int(counts[target])), "reference": "C2_final_classifier_row",
            "direct_radial_signed": direct_geom["radial_signed_first_order_norm_change"],
            "direct_tangential_norm": direct_geom["tangential_component_norm"],
            "direct_total_displacement_norm": direct_geom["total_displacement_norm"],
            "direct_exact_norm_change": direct_geom["exact_norm_change"],
            "direct_bias_change": direct_bias,
            "observed_radial_signed": observed_geom["radial_signed_first_order_norm_change"],
            "observed_tangential_norm": observed_geom["tangential_component_norm"],
            "observed_total_displacement_norm": observed_geom["total_displacement_norm"],
            "observed_exact_norm_change": observed_geom["exact_norm_change"],
            "observed_bias_change": observed_bias,
            "non_direct_remainder_radial_signed": remainder_geom["radial_signed_first_order_norm_change"],
            "non_direct_remainder_tangential_norm": remainder_geom["tangential_component_norm"],
            "non_direct_remainder_norm": remainder_geom["total_displacement_norm"],
            "non_direct_remainder_bias": remainder_bias,
            "direct_reconstruction_max_abs": direct_geom["reconstruction_max_abs"],
            "direct_orthogonality_dot": direct_geom["orthogonality_dot"],
        })
        for source in source_ids:
            geom = decompose_vector(source_weight[source, target], reference)
            delta_bias = float(source_bias[source, target])
            competitor = int(competitor_map[str(int(target))])
            if counts[competitor] > 100:
                raise RuntimeError("Frozen competitor must be a non-Many classifier row")
            competitor_delta = source_weight[source, competitor]
            competitor_bias = float(source_bias[source, competitor])
            cal = _margin_shifts(cal_x, cal_y, int(target), competitor, geom, delta_bias,
                                 competitor_delta, competitor_bias)
            ev = _margin_shifts(eval_x, eval_y, int(target), competitor, geom, delta_bias,
                                competitor_delta, competitor_bias)
            pair_rows.append({
                "seed": seed, "source_class": int(source),
                "source_train_count": int(counts[source]), "source_group": class_group(int(counts[source])),
                "target_many_class": int(target), "target_train_count": int(counts[target]),
                "target_group": class_group(int(counts[target])), "reference": "C2_final_classifier_row",
                "radial_signed_first_order_norm_change": geom["radial_signed_first_order_norm_change"],
                "radial_component_norm": geom["radial_component_norm"],
                "tangential_component_norm": geom["tangential_component_norm"],
                "total_parameter_displacement_norm": geom["total_displacement_norm"],
                "exact_classifier_row_norm_change": geom["exact_norm_change"],
                "classifier_row_angle_change_degrees": geom["angle_change_degrees"],
                "radial_energy_fraction": geom["radial_energy_fraction"],
                "tangential_energy_fraction": geom["tangential_energy_fraction"],
                "bias_parameter_displacement": delta_bias,
                "calibration_target_radial_logit_shift": cal["target_radial_logit_shift"],
                "calibration_target_tangential_logit_shift": cal["target_tangential_logit_shift"],
                "calibration_target_bias_logit_shift": cal["target_bias_logit_shift"],
                "calibration_competitor_logit_shift": cal["competitor_row_logit_shift"],
                "calibration_full_fixed_margin_shift": cal["full_fixed_margin_shift"],
                "evaluation_target_radial_logit_shift": ev["target_radial_logit_shift"],
                "evaluation_target_tangential_logit_shift": ev["target_tangential_logit_shift"],
                "evaluation_target_bias_logit_shift": ev["target_bias_logit_shift"],
                "evaluation_competitor_logit_shift": ev["competitor_row_logit_shift"],
                "evaluation_full_fixed_margin_shift": ev["full_fixed_margin_shift"],
                "competitor_class": competitor,
            })
    return pair_rows, target_rows


def _render_report(curves, selected, pareto, target_rows, seed_status):
    lines = [
        "# CC-MIA Logit Adjustment 与分类器几何离线诊断", "",
        "## 实验边界", "",
        "本报告只分析已完成的 Stage-II 线性分类器，没有重新训练或执行任何 optimizer update。Calibration/Evaluation 均来自官方 TRAIN 的派生开发集；Evaluation 不是独立测试集。官方 CIFAR-100 TEST 未访问。",
        "",
        "几何分析使用 CC-MIA 保存的、经过 SGD momentum 传播的累计来源条件参数位移，不是逐步原始梯度。径向/切向分解以 C2 最终分类器行为固定参考方向。", "",
        "## Logit Adjustment", "",
        "使用原始 TRAIN 类别先验，固定扫描 tau=-1.50 到 1.50（步长0.05）。tau 仅在 Calibration 上按 Overall 最大值选择，Evaluation 报告锁定 tau 的结果；Evaluation 自身峰值仅作描述。", "",
        "| Seed | Arm | tau* (Calibration) | Cal Overall | Eval Overall at locked tau | Eval Many | Eval Medium | Eval Few | Eval-only peak (descriptive) |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selected:
        lines.append(
            f"| {row['seed']} | {row['arm']} | {row['tau_star_calibration']:.2f} | "
            f"{row['calibration_peak_overall']:.3f} | {row['evaluation_at_locked_tau_overall']:.3f} | "
            f"{row['evaluation_at_locked_tau_many']:.3f} | {row['evaluation_at_locked_tau_medium']:.3f} | "
            f"{row['evaluation_at_locked_tau_few']:.3f} | {row['evaluation_only_peak_overall_descriptive']:.3f} "
            f"(tau={row['evaluation_only_peak_tau_descriptive']:.2f}) |"
        )
    lines += ["", "Overall 与 Macro 在该开发划分中相同，因为每个已表示类别恰好有10个样本；两者不是独立收益。", "",
              "### Pareto 与后处理可解释性", "",
              "| Seed | Partition | Comparison | C2 frontier dominated by C3 | C3 frontier dominated by C2 | Intervention tau=0 dominated by C3 curve | Few gap vs best C3 with >= Many |",
              "|---:|---|---|---:|---:|---:|---:|"]
    for row in pareto:
        if row["comparison"] == "BASE_C3_vs_C2_frontier":
            lines.append(
                f"| {row['seed']} | {row['partition']} | C3 vs C2 frontier | "
                f"{row['fraction_C2_frontier_dominated_by_BASE_C3']:.3f} | "
                f"{row['fraction_BASE_C3_frontier_dominated_by_C2']:.3f} | — | — |")
        else:
            gap = row.get("intervention_few_minus_best_BASE_C3_few")
            lines.append(
                f"| {row['seed']} | {row['partition']} | {row['comparison']} | — | — | "
                f"{row['dominated_by_any_BASE_C3_tau']} | "
                f"{('%.3f' % gap) if gap is not None else 'outside Many range'} |")
    lines += ["", "Pareto 支配只检验扫描的一维先验偏置族。某个点不被该曲线支配，不能证明所有类别偏置调整都无效，也不能单独证明几何机制具有因果性。", "",
              "## Radial / tangential / bias attribution", "",
              "| Seed | Many target rows | Mean direct radial signed | Mean direct tangential norm | Mean observed radial signed | Mean observed tangential norm | Mean non-direct radial signed | Mean non-direct tangential norm |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for seed in SEEDS:
        rows = [r for r in target_rows if r["seed"] == seed]
        lines.append(
            f"| {seed} | {len(rows)} | {np.mean([r['direct_radial_signed'] for r in rows]):.6f} | "
            f"{np.mean([r['direct_tangential_norm'] for r in rows]):.6f} | "
            f"{np.mean([r['observed_radial_signed'] for r in rows]):.6f} | "
            f"{np.mean([r['observed_tangential_norm'] for r in rows]):.6f} | "
            f"{np.mean([r['non_direct_remainder_radial_signed'] for r in rows]):.6f} | "
            f"{np.mean([r['non_direct_remainder_tangential_norm'] for r in rows]):.6f} |")
    lines += ["", "Bias 位移单独保存在 pair/target CSV 中，没有混入径向或切向向量。非直接余项是 observed minus direct 的合并余项，不能单独解释为 state feedback。", "",
              "## 结果文件", "",
              "- logit_adjustment_curve.csv: 每个 seed/model/partition/tau 的完整指标。",
              "- logit_adjustment_selected.csv: Calibration 选择的 tau 及 Evaluation 锁定结果。",
              "- pareto_comparison.csv: 前沿支配比例和干预点相对 BASE_C3 曲线的位置。",
              "- geometry_source_target_pairs.csv: 2,210 个 source→Many-target pair 的位移、偏置及 margin 投影。",
              "- geometry_target_rows.csv: Many 目标行的 direct / observed / non-direct 几何分解。",
              "- geometry_summary.json: 协议、来源、边界和 test guard。", "",
              "## 局限", "",
              "1. 三个固定 Stage-II seeds 和 TRAIN-derived 开发划分不支持总体显著性或独立测试泛化结论。",
              "2. tau sweep 只覆盖单一 log-prior bias correction family；不能代表任意类别偏置或校准方法。",
              "3. source_components_epoch30 是沿原 BASE_C3 轨迹积分的参数位移归因；其径向/切向分解依赖固定参考方向，不等同于每一步梯度投影。",
              "4. C2 最终行作为参考方向是明确的比较基准；更换参考方向可能改变径向/切向数值。",
              "5. 离线几何归因本身不能证明对径向或切向分量实施投影会改善 Few/Overall。任何干预必须另行提出可证伪假设并预注册。", "",
              "## 完成状态", "",
              f"三个 seed 的 checkpoint / logits parity 和 parent artifact checks 均通过：{all(x['evaluation_logits_recomputed_and_verified'] for x in seed_status)}。本阶段未训练模型，optimizer updates=0。"]
    return "\n".join(lines) + "\n"


def run_analysis():
    if (ROOT / "analysis").exists() or (ROOT / "diagnostics_started.json").exists():
        raise FileExistsError("Refusing to overwrite any prior diagnostic output")
    _write_json(ROOT / "diagnostics_started.json", {
        "branch_expected": "experiment/ccmia-logit-geometry-diagnostics-20261120",
        "parent_branch": "experiment/ccmia-causal-oracle-20261119",
        "training_performed": False, "optimizer_updates": 0, "official_TEST_used": False,
        "code_sha256": _code_hashes(),
    })
    context, cal_x, cal_y, eval_x, eval_y, counts = _verify_parent_and_load_data()
    curve_rows, selected_rows, pareto_rows = [], [], []
    geometry_pairs, geometry_targets, seed_status = [], [], []
    for seed in SEEDS:
        status, models, logits = _load_seed_models_and_logits(context, seed, eval_y, cal_x, cal_y, eval_x)
        curves, selected = _sweep_seed(seed, logits, cal_y, eval_y, counts)
        curve_rows.extend(curves)
        selected_rows.extend(selected)
        pareto_rows.extend(_frontier_summary(seed, curves))
        pairs, targets = _geometry_seed(seed, models, cal_x, cal_y, eval_x, eval_y, counts)
        geometry_pairs.extend(pairs)
        geometry_targets.extend(targets)
        seed_status.append({"seed": seed, "state": status["state"],
                            "checkpoint_arms_verified": list(ARMS),
                            "evaluation_logits_recomputed_and_verified": True,
                            "geometry_pairs": len(pairs), "geometry_target_rows": len(targets)})
    output = ROOT / "analysis"
    output.mkdir(exist_ok=False)
    _write_csv(output / "logit_adjustment_curve.csv", curve_rows)
    _write_csv(output / "logit_adjustment_selected.csv", selected_rows)
    _write_csv(output / "pareto_comparison.csv", pareto_rows)
    _write_csv(output / "geometry_source_target_pairs.csv", geometry_pairs)
    _write_csv(output / "geometry_target_rows.csv", geometry_targets)
    for partition in ("calibration", "evaluation"):
        _plot_curves(curve_rows, selected_rows, partition, output)
    guard = guard_pass()
    if (guard.get("official_TEST_used") is not False
            or guard.get("official_test_file_open_attempts", 0) != 0
            or guard.get("CIFAR100_train_false_attempts", 0) != 0
            or guard.get("blocked_attempts")):
        raise RuntimeError("Official TEST guard was not clean at diagnostic completion")
    summary = {
        "completed": True, "seeds": list(SEEDS), "arms": list(ARMS),
        "tau_grid": [float(x) for x in TAUS], "train_prior_source": "sealed original TRAIN class counts",
        "calibration_examples": int(len(cal_y)), "evaluation_examples": int(len(eval_y)),
        "group_sample_counts": EXPECTED_GROUP_COUNTS,
        "overall_equals_macro_because": "10 examples per each of 99 represented classes in both partitions",
        "geometry_semantics": "source-conditioned cumulative parameter displacements propagated through SGD momentum; not instantaneous gradients",
        "geometry_reference": "fixed C2 final classifier row for each Many target",
        "non_direct_remainder_semantics": "observed BASE_C3-C2 parameter delta minus summed direct source contribution; includes state feedback, weight decay and writeback residual",
        "official_TEST_used": False, "test_guard": guard, "seed_status": seed_status,
        "code_sha256": _code_hashes(),
    }
    _write_json(output / "geometry_summary.json", summary)
    with (output / "DIAGNOSTIC_REPORT_ZH.md").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(_render_report(curve_rows, selected_rows, pareto_rows, geometry_targets, seed_status))
    _write_json(output / "diagnostics_completed.json", {
        "state": "completed", "training_performed": False, "optimizer_updates": 0,
        "official_TEST_used": False, "seeds": list(SEEDS), "arms": list(ARMS),
        "tau_points": len(TAUS), "curve_rows": len(curve_rows),
        "geometry_pair_rows": len(geometry_pairs), "geometry_target_rows": len(geometry_targets),
        "test_guard": guard, "code_sha256": _code_hashes(),
    })
    print("OFFLINE DIAGNOSTICS COMPLETE: no training, no optimizer updates, official TEST untouched", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("analyze",), required=True)
    args = parser.parse_args()
    try:
        if args.phase == "analyze":
            run_analysis()
    except BaseException as error:
        failure = ROOT / "validation" / "diagnostic_failure.json"
        failure.parent.mkdir(parents=True, exist_ok=True)
        if not failure.exists():
            with failure.open("x", encoding="utf-8", newline="\n") as handle:
                json.dump({"phase": args.phase, "exception": repr(error),
                           "traceback": traceback.format_exc(),
                           "training_performed": False, "official_TEST_used": False},
                          handle, ensure_ascii=False, indent=2)
                handle.write("\n")
        raise


if __name__ == "__main__":
    main()
