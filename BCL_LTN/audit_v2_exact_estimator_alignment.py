"""Exact-V2 Estimator Alignment Audit & Online EM Trajectory Analyzer.

Bridges the gap between the Stage-1 audit (which ran estimate_transition_and_prior)
and the actual online v2 trainer (which runs estimate_anchored_transition_and_prior):

Part A (5-regime Stage-1 aligned benchmark):
    Runs estimate_anchored_transition_and_prior with exact v2 training defaults
    (confidence_power=2.0, shrinkage_mass=10.0, prior_pseudocount=1.0,
    max_iter=200, tolerance=1e-6) on the identical frozen TRAIN logits and
    five noise regimes (sym02, sym04, sym06, asym04_coarse_cycle, cifar100n_real).
    Strictly seals each estimate before RegimeTruthGate opens truth/.

Part B (Online v2 Sym-0.4 EM trajectory audit):
    Exports full 100-class pi and 100x100 T diagnostics across all 36 saved
    online snapshots (Epochs 20..195, highlighting 20, 50, 100, 150, 195),
    including Few Spearman rank correlation, Few CV, tail mass ratio, and
    alignment with Epoch-200 classifier weight norms and prototype cosines.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from BCL_AUDIT.cifar100lt.models import BCLCIFARModel
from BCL_LTN.audit_v2_transition_estimator import (
    K,
    N,
    probability_metrics,
    read_json,
    sha256,
    softmax_numpy,
    transition_and_prior_metrics,
    write_json,
)
from BCL_LTN.make_noisy_lt import group_class_ids
from BCL_LTN.ntc_bcl_v2_losses import (
    estimate_anchored_transition_and_prior,
    measure_prototype_and_classifier_health,
)
from BCL_LTN.train_noisy_bcl import OfficialTestAudit, safe_torch_load
from BCL_LTN.train_ntc_bcl_v2 import RegimeTruthGate


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE1_DATA_ROOT = ROOT / "BCL_LTN" / "outputs" / "ntc_bcl_v2_stage1_data_20261126"
DEFAULT_STAGE1_DIR = ROOT / "BCL_LTN" / "outputs" / "ntc_bcl_v2_stage1_20261126"
DEFAULT_LOGITS_PATH = DEFAULT_STAGE1_DIR / "train_logits.npy"
DEFAULT_V2_RUN_DIR = ROOT / "BCL_LTN" / "outputs" / "NTC_BCL_V2_sym04_seed42_20261126"
DEFAULT_DATA_ROOT = ROOT / "H2T" / "data" / "cifar100"
DEFAULT_OUTPUT_DIR = ROOT / "BCL_LTN" / "outputs" / "ntc_bcl_v2_exact_alignment_20261126"
REGIMES = ("sym02", "sym04", "sym06", "asym04_coarse_cycle", "cifar100n_real")
MILESTONE_EPOCHS = (20, 50, 100, 150, 195)


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Compute 1-based fractional average ranks for tie-safe Spearman correlation."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("values must be a non-empty 1-D vector")
    n = arr.size
    order = np.argsort(arr, kind="mergesort")
    sorted_vals = arr[order]
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_vals[j] == sorted_vals[i]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def spearman_rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Exact tie-aware Spearman rank correlation without scipy dependency."""
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or a.size < 2:
        raise ValueError("x and y must be matching 1-D vectors with length >= 2")
    rx = average_ranks(a)
    ry = average_ranks(b)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(np.sqrt(np.sum(rx * rx) * np.sum(ry * ry)))
    if denom <= 1e-15:
        return 0.0
    return float(np.sum(rx * ry) / denom)


def coefficient_of_variation(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    if mean <= 1e-15:
        return 0.0
    return float(arr.std(ddof=0) / mean)


def compute_detailed_prior_and_group_metrics(
    pi_hat: np.ndarray,
    t_hat: np.ndarray,
    pi_true: np.ndarray,
    t_ref: np.ndarray,
    t_emp: np.ndarray,
    clean_counts: np.ndarray,
    groups: Dict[str, np.ndarray],
) -> Dict[str, object]:
    """Compute global + Many/Medium/Few prior recovery, Spearman, CV, and T errors."""
    pi_eps = 1e-12
    row_tv_ref = 0.5 * np.abs(t_hat - t_ref).sum(axis=1)
    row_tv_emp = 0.5 * np.abs(t_hat - t_emp).sum(axis=1)
    diag_err_ref = np.abs(np.diag(t_hat) - np.diag(t_ref))
    diag_err_emp = np.abs(np.diag(t_hat) - np.diag(t_emp))

    group_metrics = {}
    for name, ids in groups.items():
        pi_t_g = pi_true[ids]
        pi_h_g = pi_hat[ids]
        pi_sum_t = float(pi_t_g.sum())
        pi_sum_h = float(pi_h_g.sum())
        pi_kl_g = float(np.sum(pi_t_g * np.log((pi_t_g + pi_eps) / (pi_h_g + pi_eps))))
        t_true_noise_g = float(
            1.0 - np.dot(pi_t_g, np.diag(t_emp)[ids]) / max(pi_sum_t, pi_eps)
        )
        t_est_noise_g = float(
            1.0 - np.dot(pi_h_g, np.diag(t_hat)[ids]) / max(pi_sum_h, pi_eps)
        )
        ratios = pi_h_g / np.maximum(pi_t_g, pi_eps)
        group_metrics[name] = {
            "classes": int(ids.size),
            "prior_mass_true": pi_sum_t,
            "prior_mass_estimated": pi_sum_h,
            "prior_mass_ratio": pi_sum_h / max(pi_sum_t, pi_eps),
            "prior_mae": float(np.abs(pi_h_g - pi_t_g).mean()),
            "prior_l1": float(np.abs(pi_h_g - pi_t_g).sum()),
            "prior_relative_l1": float(np.abs(pi_h_g - pi_t_g).sum() / max(pi_sum_t, pi_eps)),
            "prior_kl_component": pi_kl_g,
            "spearman_prior": spearman_rank_correlation(pi_t_g, pi_h_g),
            "cv_true": coefficient_of_variation(pi_t_g),
            "cv_estimated": coefficient_of_variation(pi_h_g),
            "classes_under_0_5x_true": int(np.sum(ratios < 0.5)),
            "classes_over_2_0x_true": int(np.sum(ratios > 2.0)),
            "t_diag_mae_ref": float(diag_err_ref[ids].mean()),
            "t_diag_mae_emp": float(diag_err_emp[ids].mean()),
            "row_tv_mean_ref": float(row_tv_ref[ids].mean()),
            "row_tv_mean_emp": float(row_tv_emp[ids].mean()),
            "group_noise_true_emp": t_true_noise_g,
            "group_noise_estimated": t_est_noise_g,
        }

    return {
        "overall_spearman_prior": spearman_rank_correlation(pi_true, pi_hat),
        "global_t_mae_emp": float(np.abs(t_hat - t_emp).mean()),
        "global_t_frob_emp": float(np.linalg.norm(t_hat - t_emp)),
        "global_t_diag_mae_emp": float(diag_err_emp.mean()),
        "global_row_tv_mean_emp": float(row_tv_emp.mean()),
        "global_row_tv_mean_ref": float(row_tv_ref.mean()),
        "groups": group_metrics,
    }


def load_regime_truth_locked(
    regime_dir: Path, truth_gate: RegimeTruthGate, observed: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    truth_dir = regime_dir / "truth"
    with truth_gate.offline_only():
        t_manifest = read_json(truth_dir / "truth_manifest.json")
        clean_path = truth_dir / "clean_labels.npy"
        noisy_path = truth_dir / "is_noisy.npy"
        emp_path = truth_dir / "true_transition.npy"
        theo_path = truth_dir / "theoretical_transition.npy"
        if sha256(clean_path) != t_manifest["clean_labels_sha256"]:
            raise RuntimeError("clean_labels SHA256 mismatch")
        if sha256(noisy_path) != t_manifest["is_noisy_sha256"]:
            raise RuntimeError("is_noisy SHA256 mismatch")
        if sha256(emp_path) != t_manifest["true_transition_sha256"]:
            raise RuntimeError("true_transition SHA256 mismatch")
        clean = np.load(clean_path, allow_pickle=False)
        is_noisy = np.load(noisy_path, allow_pickle=False)
        t_emp = np.load(emp_path, allow_pickle=False)
        ref_kind = t_manifest.get("transition_reference", "theoretical")
        if ref_kind == "theoretical" and theo_path.is_file():
            t_ref = np.load(theo_path, allow_pickle=False)
        else:
            t_ref = t_emp
    if not np.array_equal(is_noisy, clean != observed):
        raise RuntimeError("is_noisy mismatch with clean != observed")
    return clean, t_emp, t_ref, is_noisy, t_manifest


def run_stage1_exact_v2_regime(
    regime: str,
    stage1_data_root: Path,
    stage1_dir: Path,
    logits: np.ndarray,
    logits_sha: str,
    out_root: Path,
    *,
    confidence_power: float = 2.0,
    shrinkage_mass: float = 10.0,
    prior_pseudocount: float = 1.0,
    max_iter: int = 200,
    tolerance: float = 1e-6,
) -> Dict[str, object]:
    """Run seal-before-truth Exact-V2 anchored EM benchmark on one regime."""
    regime_dir = stage1_data_root / regime
    manifest = read_json(regime_dir / "manifest.json")
    obs_path = regime_dir / manifest["observed_labels_file"]
    if sha256(obs_path) != manifest["observed_labels_sha256"]:
        raise RuntimeError(f"Observed labels SHA256 mismatch for {regime}")
    observed = np.load(obs_path, allow_pickle=False)

    truth_gate = RegimeTruthGate(regime_dir)
    truth_gate.install()

    out_dir = out_root / "stage1_aligned" / regime
    out_dir.mkdir(parents=True, exist_ok=True)

    probs_np = softmax_numpy(logits)
    probs_t = torch.from_numpy(probs_np)
    obs_t = torch.from_numpy(observed).long()

    est = estimate_anchored_transition_and_prior(
        probs_t,
        obs_t,
        confidence_power=confidence_power,
        shrinkage_mass=shrinkage_mass,
        prior_pseudocount=prior_pseudocount,
        max_iter=max_iter,
        tolerance=tolerance,
    )
    pi_hat = est.clean_prior.cpu().numpy()
    t_hat = est.transition.cpu().numpy()
    gamma_hat = est.posterior_clean.cpu().numpy()
    anchor_sup = est.anchor_support.cpu().numpy()
    eff_sup = est.effective_support.cpu().numpy()

    bundle_path = out_dir / "estimate_bundle.npz"
    with bundle_path.open("xb") as bf:
        np.savez_compressed(
            bf,
            clean_prior=pi_hat,
            transition=t_hat,
            posterior_clean=gamma_hat,
            anchor_support=anchor_sup,
            effective_support=eff_sup,
        )

    estimator_only = {
        "regime": regime,
        "estimator": "estimate_anchored_transition_and_prior",
        "logits_sha256": logits_sha,
        "observed_labels_sha256": manifest["observed_labels_sha256"],
        "hyperparameters": {
            "confidence_power": confidence_power,
            "shrinkage_mass": shrinkage_mass,
            "prior_pseudocount": prior_pseudocount,
            "max_iter": max_iter,
            "tolerance": tolerance,
        },
        "iterations": int(est.iterations),
        "converged": bool(est.converged),
        "final_delta": float(est.final_delta),
        "estimated_noise_rate": float(est.estimated_noise_rate),
        "global_diagonal_retention": float(est.global_diagonal_retention.item()),
        "prior_min": float(pi_hat.min()),
        "prior_max": float(pi_hat.max()),
        "truth_opens_before_seal": truth_gate.opens,
    }
    write_json(out_dir / "estimator_only.json", estimator_only)
    write_json(
        out_dir / "seal.json",
        {
            "state": "immutable_estimator_outputs",
            "truth_loaded_before_seal": False,
            "estimate_bundle_sha256": sha256(bundle_path),
            "estimator_only_sha256": sha256(out_dir / "estimator_only.json"),
        },
    )

    if truth_gate.opens or truth_gate.denied:
        raise RuntimeError(f"Truth gate opened before seal in {regime}")

    # Offline truth evaluation
    clean, t_emp, t_ref, _, t_manifest = load_regime_truth_locked(
        regime_dir, truth_gate, observed
    )
    clean_counts = np.bincount(clean, minlength=K)
    pi_true = clean_counts.astype(np.float64) / float(N)
    groups = group_class_ids(clean_counts)

    ref_kind = t_manifest.get("transition_reference", "theoretical")
    base_tp = transition_and_prior_metrics(
        pi_hat, t_hat, clean, observed, t_ref, t_emp, ref_kind
    )
    detailed = compute_detailed_prior_and_group_metrics(
        pi_hat, t_hat, pi_true, t_ref, t_emp, clean_counts, groups
    )
    raw_q_acc = probability_metrics(probs_np, clean, clean_counts)
    gamma_acc = probability_metrics(gamma_hat, clean, clean_counts)

    # Compare against prior Stage-1 offline_evaluation.json if present
    old_stage1_path = stage1_dir / regime / "offline_evaluation.json"
    old_stage1_summary = None
    if old_stage1_path.is_file():
        old_eval = read_json(old_stage1_path)
        old_tp = old_eval["transition_and_prior_metrics"]["balanced_posterior"]
        old_est = old_eval["estimator_only"]["balanced_posterior_estimate"]
        old_few = old_tp["frequency_groups"]["Few"]
        old_stage1_summary = {
            "converged": old_est["converged"],
            "iterations": old_est["iterations"],
            "final_delta": old_est["final_delta"],
            "estimated_noise_rate": old_tp["estimated_noise_rate"],
            "t_mae": old_tp["transition_reference"]["mae"],
            "t_diag_mae": old_tp["transition_reference"]["diagonal_mae"],
            "prior_kl": old_tp["clean_prior"]["kl_true_to_estimated"],
            "prior_min": old_tp["clean_prior"]["minimum_estimated"],
            "few_prior_mass_estimated": old_few["prior_mass_estimated"],
        }

    eval_payload = {
        "regime": regime,
        "estimator": "estimate_anchored_transition_and_prior",
        "convergence": {
            "iterations": int(est.iterations),
            "converged": bool(est.converged),
            "final_delta": float(est.final_delta),
        },
        "transition_and_prior": base_tp,
        "detailed_group_metrics": detailed,
        "probability_accuracy": {
            "raw_balanced_q": {
                g: float(raw_q_acc[g]["accuracy_percent"])
                for g in ("Overall", "Many", "Medium", "Few")
            },
            "anchored_em_gamma": {
                g: float(gamma_acc[g]["accuracy_percent"])
                for g in ("Overall", "Many", "Medium", "Few")
            },
        },
        "prior_stage1_joint_em_comparison": old_stage1_summary,
    }
    write_json(out_dir / "offline_evaluation.json", eval_payload)

    # 100-class CSV
    raw_pred = probs_np.argmax(axis=1)
    gamma_pred = gamma_hat.argmax(axis=1)
    row_tv_ref = 0.5 * np.abs(t_hat - t_ref).sum(axis=1)
    row_tv_emp = 0.5 * np.abs(t_hat - t_emp).sum(axis=1)
    with (out_dir / "per_class_metrics.csv").open("w", newline="", encoding="utf-8") as cf:
        fields = [
            "class_id",
            "clean_count",
            "frequency_group",
            "pi_true",
            "pi_estimated",
            "pi_ratio",
            "T_diag_reference",
            "T_diag_empirical",
            "T_diag_estimated",
            "T_row_tv_reference",
            "T_row_tv_empirical",
            "anchor_support",
            "effective_support",
            "raw_balanced_q_acc_pct",
            "anchored_em_gamma_acc_pct",
        ]
        writer = csv.DictWriter(cf, fieldnames=fields)
        writer.writeheader()
        for cid in range(K):
            grp = (
                "Many"
                if clean_counts[cid] > 100
                else ("Medium" if clean_counts[cid] >= 20 else "Few")
            )
            mask_c = clean == cid
            n_c = int(mask_c.sum())
            writer.writerow(
                {
                    "class_id": cid,
                    "clean_count": int(clean_counts[cid]),
                    "frequency_group": grp,
                    "pi_true": float(pi_true[cid]),
                    "pi_estimated": float(pi_hat[cid]),
                    "pi_ratio": float(pi_hat[cid] / max(pi_true[cid], 1e-12)),
                    "T_diag_reference": float(t_ref[cid, cid]),
                    "T_diag_empirical": float(t_emp[cid, cid]),
                    "T_diag_estimated": float(t_hat[cid, cid]),
                    "T_row_tv_reference": float(row_tv_ref[cid]),
                    "T_row_tv_empirical": float(row_tv_emp[cid]),
                    "anchor_support": float(anchor_sup[cid]),
                    "effective_support": float(eff_sup[cid]),
                    "raw_balanced_q_acc_pct": 100.0 * float((raw_pred[mask_c] == cid).mean())
                    if n_c > 0
                    else 0.0,
                    "anchored_em_gamma_acc_pct": 100.0 * float((gamma_pred[mask_c] == cid).mean())
                    if n_c > 0
                    else 0.0,
                }
            )
    return eval_payload, (clean, t_emp, t_ref)


def run_online_v2_trajectory_audit(
    v2_run_dir: Path,
    sym04_truth: Tuple[np.ndarray, np.ndarray, np.ndarray],
    out_root: Path,
) -> Dict[str, object]:
    """Export per-group and per-class EM snapshots from completed v2 sym04 run."""
    clean, t_emp, t_ref = sym04_truth

    clean_counts = np.bincount(clean, minlength=K)
    pi_true = clean_counts.astype(np.float64) / float(N)
    groups = group_class_ids(clean_counts)

    # Load Epoch-200 student norms, prototype cosines, and TEST per-class accuracy
    ckpt_200 = safe_torch_load(v2_run_dir / "checkpoints" / "epoch_200.pth", torch.device("cpu"))
    model = BCLCIFARModel().eval()
    model.load_state_dict(ckpt_200["model_state"], strict=True)
    proto_health = measure_prototype_and_classifier_health(
        model.classifier.weight, model.prototype_mlp
    )
    w_norm_200 = np.asarray(proto_health["classifier_weight_norm"], dtype=np.float64)
    dir_cos_200 = np.asarray(
        proto_health["dir_per_class_max_offdiag_cosine"], dtype=np.float64
    )
    ft = read_json(v2_run_dir / "final_test_metrics.json")
    student_test_acc = np.asarray(ft["student"]["per_class_correct"], dtype=np.int64)

    snap_dir = v2_run_dir / "em_snapshots"
    snap_paths = sorted(snap_dir.glob("epoch_*.npz"))
    if not snap_paths:
        raise FileNotFoundError(f"No EM snapshots found in {snap_dir}")

    traj_dir = out_root / "online_v2_sym04_em_trajectory"
    traj_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, object]] = []
    milestone_records: Dict[str, object] = {}
    milestone_csv_rows: List[Dict[str, object]] = []

    for sp in snap_paths:
        epoch = int(sp.stem.split("_")[1])
        with np.load(sp, allow_pickle=False) as arr:
            raw_pi = arr["raw_em_prior"].astype(np.float64)
            raw_t = arr["raw_em_transition"].astype(np.float64)
            bld_pi = arr["blended_prior"].astype(np.float64)
            bld_t = arr["blended_transition"].astype(np.float64)

        raw_det = compute_detailed_prior_and_group_metrics(
            raw_pi, raw_t, pi_true, t_ref, t_emp, clean_counts, groups
        )
        bld_det = compute_detailed_prior_and_group_metrics(
            bld_pi, bld_t, pi_true, t_ref, t_emp, clean_counts, groups
        )
        pi_eps = 1e-12
        bld_kl = float(np.sum(pi_true * np.log((pi_true + pi_eps) / (bld_pi + pi_eps))))
        raw_kl = float(np.sum(pi_true * np.log((pi_true + pi_eps) / (raw_pi + pi_eps))))

        row = {
            "epoch": epoch,
            "raw_noise_est": float(1.0 - np.dot(raw_pi, np.diag(raw_t))),
            "blended_noise_est": float(1.0 - np.dot(bld_pi, np.diag(bld_t))),
            "raw_T_mae_emp": raw_det["global_t_mae_emp"],
            "blended_T_mae_emp": bld_det["global_t_mae_emp"],
            "raw_T_diag_mae_emp": raw_det["global_t_diag_mae_emp"],
            "blended_T_diag_mae_emp": bld_det["global_t_diag_mae_emp"],
            "raw_Row_TV_emp": raw_det["global_row_tv_mean_emp"],
            "blended_Row_TV_emp": bld_det["global_row_tv_mean_emp"],
            "raw_pi_kl": raw_kl,
            "blended_pi_kl": bld_kl,
            "raw_overall_spearman": raw_det["overall_spearman_prior"],
            "blended_overall_spearman": bld_det["overall_spearman_prior"],
            "raw_many_pi_ratio": raw_det["groups"]["Many"]["prior_mass_ratio"],
            "blended_many_pi_ratio": bld_det["groups"]["Many"]["prior_mass_ratio"],
            "raw_med_pi_ratio": raw_det["groups"]["Medium"]["prior_mass_ratio"],
            "blended_med_pi_ratio": bld_det["groups"]["Medium"]["prior_mass_ratio"],
            "raw_few_pi_ratio": raw_det["groups"]["Few"]["prior_mass_ratio"],
            "blended_few_pi_ratio": bld_det["groups"]["Few"]["prior_mass_ratio"],
            "raw_few_spearman": raw_det["groups"]["Few"]["spearman_prior"],
            "blended_few_spearman": bld_det["groups"]["Few"]["spearman_prior"],
            "raw_few_cv": raw_det["groups"]["Few"]["cv_estimated"],
            "blended_few_cv": bld_det["groups"]["Few"]["cv_estimated"],
            "few_cv_true": bld_det["groups"]["Few"]["cv_true"],
            "blended_few_under_0_5x": bld_det["groups"]["Few"]["classes_under_0_5x_true"],
            "blended_few_over_2_0x": bld_det["groups"]["Few"]["classes_over_2_0x_true"],
            "blended_many_diag_mae": bld_det["groups"]["Many"]["t_diag_mae_emp"],
            "blended_med_diag_mae": bld_det["groups"]["Medium"]["t_diag_mae_emp"],
            "blended_few_diag_mae": bld_det["groups"]["Few"]["t_diag_mae_emp"],
        }
        summary_rows.append(row)

        if epoch in MILESTONE_EPOCHS:
            bld_row_tv = 0.5 * np.abs(bld_t - t_emp).sum(axis=1)
            raw_row_tv = 0.5 * np.abs(raw_t - t_emp).sum(axis=1)
            milestone_records[f"epoch_{epoch:03d}"] = {
                "summary": row,
                "raw_prior_100": raw_pi.tolist(),
                "blended_prior_100": bld_pi.tolist(),
                "raw_T_diag_100": np.diag(raw_t).tolist(),
                "blended_T_diag_100": np.diag(bld_t).tolist(),
                "blended_T_100x100": bld_t.tolist(),
            }
            for cid in range(K):
                grp = (
                    "Many"
                    if clean_counts[cid] > 100
                    else ("Medium" if clean_counts[cid] >= 20 else "Few")
                )
                milestone_csv_rows.append(
                    {
                        "epoch": epoch,
                        "class_id": cid,
                        "clean_count": int(clean_counts[cid]),
                        "frequency_group": grp,
                        "pi_true": float(pi_true[cid]),
                        "raw_pi_hat": float(raw_pi[cid]),
                        "raw_pi_ratio": float(raw_pi[cid] / max(pi_true[cid], 1e-12)),
                        "blended_pi_hat": float(bld_pi[cid]),
                        "blended_pi_ratio": float(bld_pi[cid] / max(pi_true[cid], 1e-12)),
                        "T_diag_true_emp": float(t_emp[cid, cid]),
                        "raw_T_diag": float(raw_t[cid, cid]),
                        "blended_T_diag": float(bld_t[cid, cid]),
                        "raw_row_tv_emp": float(raw_row_tv[cid]),
                        "blended_row_tv_emp": float(bld_row_tv[cid]),
                        "ep200_classifier_norm": float(w_norm_200[cid]),
                        "ep200_dir_max_cos": float(dir_cos_200[cid]),
                        "ep200_student_test_acc_pct": int(student_test_acc[cid]),
                    }
                )

    with (traj_dir / "trajectory_summary.csv").open("w", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    write_json(traj_dir / "trajectory_summary.json", {"rows": summary_rows})

    with (traj_dir / "milestone_per_class_ep_20_50_100_150_195.csv").open(
        "w", newline="", encoding="utf-8"
    ) as cf:
        writer = csv.DictWriter(cf, fieldnames=list(milestone_csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(milestone_csv_rows)
    write_json(traj_dir / "milestone_snapshots_ep_20_50_100_150_195.json", milestone_records)

    return {
        "snapshots_count": len(summary_rows),
        "milestone_rows": [r for r in summary_rows if r["epoch"] in MILESTONE_EPOCHS],
    }


def write_alignment_report(
    out_root: Path,
    stage1_results: Dict[str, Dict[str, object]],
    online_summary: Dict[str, object],
    logits_sha: str,
) -> None:
    lines = [
        "# Exact-V2 Estimator Alignment Audit Report",
        "",
        "- Part A runs the **exact v2 online estimator** (`estimate_anchored_transition_and_prior`, `confidence_power=2.0`, `shrinkage_mass=10.0`, `prior_pseudocount=1.0`, `max_iter=200`, `tolerance=1e-6`) on the exact frozen Stage-1 `train_logits.npy` (`sha256: " + logits_sha + "`) across all five regimes (`sym02`, `sym04`, `sym06`, `asym04_coarse_cycle`, `cifar100n_real`).",
        "- Part B extracts the full 100-class and Many/Medium/Few trajectory from the completed online `v2 sym04` run's 36 `em_snapshots` (`Epoch 20..195`).",
        "",
        "## Part A: 5-Regime Comparison (Old Stage-1 Joint EM vs Exact-V2 Anchored EM on Frozen v1 Teacher)",
        "",
        "| Regime | Realized Noise | Est Noise (Old -> Exact-V2) | Converged (Old -> V2) | T Diag MAE (Old -> V2) | Prior KL (Old -> V2) | Few Mass Ratio (Old -> V2) | V2 Few Spearman | V2 Few CV (true 0.411) |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for reg in REGIMES:
        res = stage1_results[reg]
        tp = res["transition_and_prior"]
        det = res["detailed_group_metrics"]
        few = det["groups"]["Few"]
        old = res["prior_stage1_joint_em_comparison"] or {}
        old_n = f"{old['estimated_noise_rate']:.4f}" if old else "n/a"
        old_c = str(old["converged"]) if old else "n/a"
        old_d = f"{old['t_diag_mae']:.4f}" if old else "n/a"
        old_kl = f"{old['prior_kl']:.4f}" if old else "n/a"
        old_fr = f"{old['few_prior_mass_estimated'] / few['prior_mass_true']:.2f}x" if old else "n/a"
        lines.append(
            f"| {reg} | {tp['realized_noise_rate']:.4f} | "
            f"{old_n} -> **{tp['estimated_noise_rate']:.4f}** | "
            f"{old_c} -> **{res['convergence']['converged']}** ({res['convergence']['iterations']} it) | "
            f"{old_d} -> **{tp['transition_reference']['diagonal_mae']:.4f}** | "
            f"{old_kl} -> **{tp['clean_prior']['kl_true_to_estimated']:.4f}** | "
            f"{old_fr} -> **{few['prior_mass_ratio']:.2f}x** | "
            f"{few['spearman_prior']:.3f} | {few['cv_estimated']:.3f} |"
        )

    lines.extend([
        "",
        "## Part B: Online v2 Sym-0.4 EMA-Teacher EM Trajectory (Realized Noise = 0.4037, True Few Mass = 2.81%, Few CV = 0.411)",
        "",
        "| Epoch | Raw Noise | Blended Noise | Blended T Diag MAE | Blended Row TV | Blended Prior KL | Many/Med/Few Mass Ratio | Few Spearman (Raw / Bld) | Few CV (Raw / Bld) | Few <0.5x / >2x |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |",
    ])
    for r in online_summary["milestone_rows"]:
        lines.append(
            f"| {r['epoch']} | {r['raw_noise_est']:.4f} | {r['blended_noise_est']:.4f} | "
            f"{r['blended_T_diag_mae_emp']:.4f} | {r['blended_Row_TV_emp']:.4f} | "
            f"{r['blended_pi_kl']:.4f} | "
            f"{r['blended_many_pi_ratio']:.2f}x / {r['blended_med_pi_ratio']:.2f}x / {r['blended_few_pi_ratio']:.2f}x | "
            f"{r['raw_few_spearman']:.3f} / {r['blended_few_spearman']:.3f} | "
            f"{r['raw_few_cv']:.3f} / {r['blended_few_cv']:.3f} | "
            f"{r['blended_few_under_0_5x']} / {r['blended_few_over_2_0x']} |"
        )
    (out_root / "EXACT_V2_ALIGNMENT_AUDIT_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Exact-V2 Estimator Alignment Audit")
    parser.add_argument("--stage1-data-root", type=Path, default=DEFAULT_STAGE1_DATA_ROOT)
    parser.add_argument("--stage1-dir", type=Path, default=DEFAULT_STAGE1_DIR)
    parser.add_argument("--logits", type=Path, default=DEFAULT_LOGITS_PATH)
    parser.add_argument("--v2-run-dir", type=Path, default=DEFAULT_V2_RUN_DIR)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {args.output_dir}")
    test_gate = OfficialTestAudit(args.data_root)

    logits_sha = sha256(args.logits)
    logits = np.load(args.logits, allow_pickle=False)
    if logits.shape != (N, K):
        raise RuntimeError(f"Unexpected TRAIN logits shape: {logits.shape}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage1_results = {}
    sym04_truth = None
    for reg in REGIMES:
        eval_payload, truth_tuple = run_stage1_exact_v2_regime(
            reg, args.stage1_data_root, args.stage1_dir, logits, logits_sha, args.output_dir
        )
        stage1_results[reg] = eval_payload
        if reg == "sym04":
            sym04_truth = truth_tuple
    assert sym04_truth is not None
    online_summary = run_online_v2_trajectory_audit(
        args.v2_run_dir, sym04_truth, args.output_dir
    )
    write_alignment_report(args.output_dir, stage1_results, online_summary, logits_sha)
    if test_gate.opens or test_gate.denied:
        raise RuntimeError(f"Official TEST accessed during estimator audit: {test_gate.opens}")


if __name__ == "__main__":
    main()
