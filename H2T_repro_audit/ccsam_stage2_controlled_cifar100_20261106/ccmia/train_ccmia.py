from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

BASE_ROOT = Path(__file__).resolve().parents[1]
CCMIA_ROOT = Path(__file__).resolve().parent
if str(BASE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASE_ROOT))

from ccsam_common import (  # noqa: E402
    BATCH_SIZE,
    FEATURE_DIM,
    MOMENTUM,
    NUM_CLASSES,
    STAGE2_EPOCHS,
    STAGE2_LR,
    WEIGHT_DECAY,
    augment_batch,
    build_models,
    build_transforms,
    check_backbone_unchanged,
    class_weights_from_counts,
    clean_compatible_gap,
    guard_pass,
    hash_arrays,
    install_test_guard,
    metrics_from_predictions,
    read_json,
    resolve_checkpoint,
    seed_everything,
    sha256_file,
    state_hash,
    validate_class_aware_plan,
    write_json,
)
from train_ccsam import C3_ARM, compute_arm_loss, load_prepared  # noqa: E402
from ccmia_core import (  # noqa: E402
    ParameterWritebackRoundingLedger,
    SGDInfluenceTracker,
    clean_weighted_ce_gradient,
    decompose_perturbation_gradient_by_source,
    max_abs_and_relative_error,
    project_gradient_to_fixed_margin,
    project_parameter_delta_to_fixed_margin,
    project_source_gradients_to_fixed_margin,
    project_source_parameter_delta_to_fixed_margin,
)

SEEDS = (123, 124, 125)
C2_ARM = "C2_WEIGHTED_CE_NO_PERTURB"
TRACE_FIELDS = (
    "epoch",
    "batch",
    "batch_size",
    "indices_labels_sha256",
    "augmented_images_sha256",
    "gap_features_sha256",
)
MANY_THRESHOLD_EXCLUSIVE = 100
MEDIUM_MIN_INCLUSIVE = 20
MOMENTUM_RECONSTRUCTION_REL_TOL = 5e-3
MOMENTUM_RECONSTRUCTION_ABS_TOL = 2e-6
GRADIENT_DECOMPOSITION_ATOL = 2e-5
GRADIENT_DECOMPOSITION_RTOL = 2e-4

CODE_PATHS = (
    CCMIA_ROOT / "ccmia_core.py",
    CCMIA_ROOT / "train_ccmia.py",
    CCMIA_ROOT / "analyze_ccmia.py",
    CCMIA_ROOT / "test_ccmia_core.py",
    CCMIA_ROOT / "CCMIA_PROTOCOL.md",
    BASE_ROOT / "train_ccsam.py",
    BASE_ROOT / "ccsam_common.py",
    BASE_ROOT / "experiment_protocol.json",
    BASE_ROOT / "c3_selective" / "C3_PROTOCOL.md",
    BASE_ROOT / "tpr_lowrank" / "TPR_PROTOCOL.md",
)


def read_csv_rows(path: Path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, fieldnames, rows, append=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        if not append or path.stat().st_size == 0:
            writer.writeheader()
        writer.writerows(rows)


def code_hashes():
    result = {}
    for path in CODE_PATHS:
        if not path.is_file():
            raise FileNotFoundError(f"Required source/protocol file is missing: {path}")
        result[str(path.relative_to(BASE_ROOT))] = sha256_file(path)
    return result


def prepare_context():
    manifest_path = BASE_ROOT / "prepared" / "prepared_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "Missing prepared/prepared_manifest.json. Do not regenerate or replace prepared inputs in this audit."
        )
    preliminary = read_json(manifest_path)
    if preliminary.get("official_TRAIN_pickle") is None and preliminary.get("official_train_pickle") is None:
        raise RuntimeError("Prepared manifest lacks the official TRAIN pickle provenance.")
    train_pickle = preliminary.get("official_train_pickle", preliminary.get("official_TRAIN_pickle"))
    data_root = Path(train_pickle).expanduser().resolve().parent.parent
    install_test_guard(data_root)

    manifest, train_images, train_labels, validation_gap, validation_labels = load_prepared()
    counts = np.asarray(manifest.get("train_class_counts"), dtype=np.int64)
    if counts.shape != (NUM_CLASSES,) or np.any(counts <= 0):
        raise RuntimeError("Prepared manifest must contain 100 positive TRAIN class counts.")
    actual_counts = np.bincount(np.asarray(train_labels, dtype=np.int64), minlength=NUM_CLASSES)
    if not np.array_equal(actual_counts, counts):
        raise RuntimeError("Prepared TRAIN labels disagree with the sealed class-count vector.")
    if train_images.shape != (10847, 32, 32, 3) or len(train_labels) != 10847:
        raise RuntimeError("Unexpected prepared CIFAR-100-LT TRAIN image/label shape.")
    if validation_gap.shape != (1980, FEATURE_DIM) or validation_labels.shape != (1980,):
        raise RuntimeError("Unexpected TRAIN-derived holdout feature/label shape.")
    if manifest.get("official_TEST_used") is not False:
        raise RuntimeError("Prepared manifest does not certify official TEST-free preparation.")
    if int(manifest.get("official_TEST_images_or_labels_loaded", -1)) != 0:
        raise RuntimeError("Prepared manifest does not certify that official TEST images/labels were never loaded.")
    if int(manifest.get("CIFAR100_train_false_calls", -1)) != 0:
        raise RuntimeError("Prepared manifest records a forbidden CIFAR100(train=False) call.")
    if manifest.get("checkpoint_retrained") is not False:
        raise RuntimeError("Prepared manifest does not certify use of the sealed Clean Stage-I checkpoint.")

    validation_counts = np.bincount(np.asarray(validation_labels, dtype=np.int64), minlength=NUM_CLASSES)
    sealed_validation_counts = np.asarray(manifest.get("validation_class_counts"), dtype=np.int64)
    if sealed_validation_counts.shape != (NUM_CLASSES,) or not np.array_equal(validation_counts, sealed_validation_counts):
        raise RuntimeError("Prepared validation labels disagree with the sealed per-class holdout counts.")
    if validation_counts[0] != 0 or not np.all(validation_counts[1:] == 20):
        raise RuntimeError("Unexpected TRAIN-derived validation split; expected class 0 absent and 20 samples in classes 1-99.")

    train_pickle = Path(manifest["official_train_pickle"]).expanduser().resolve()
    if not train_pickle.is_file() or sha256_file(train_pickle) != manifest.get("official_train_pickle_sha256"):
        raise RuntimeError("Official TRAIN pickle is missing or its SHA256 differs from the prepared manifest.")
    split_path = Path(manifest["split_path"]).expanduser().resolve()
    if not split_path.is_file() or sha256_file(split_path) != manifest.get("split_sha256"):
        raise RuntimeError("Prepared split-index file is missing or its SHA256 differs from the manifest.")
    checkpoint_path = resolve_checkpoint(manifest["checkpoint_path"])
    if sha256_file(checkpoint_path) != manifest.get("clean_stage1_sha256"):
        raise RuntimeError("Clean Stage-I checkpoint file SHA256 differs from the sealed manifest.")

    train_transform, _, transform_sha = build_transforms()
    if transform_sha != manifest.get("transform_source_sha256"):
        raise RuntimeError("The audited augmentation transform changed after preparation.")
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "train_images": train_images,
        "train_labels": np.asarray(train_labels, dtype=np.int64),
        "validation_gap": np.asarray(validation_gap, dtype=np.float32),
        "validation_labels": np.asarray(validation_labels, dtype=np.int64),
        "counts": counts,
        "train_transform": train_transform,
        "data_root": data_root,
    }


def validate_inputs(context):
    manifest = context["manifest"]
    train_labels = context["train_labels"]
    count = len(train_labels)
    batches_per_epoch = math.ceil(count / BATCH_SIZE)
    expected_trace_rows = STAGE2_EPOCHS * batches_per_epoch
    seed_records = {}

    for seed in SEEDS:
        plan_path = BASE_ROOT / "prepared" / f"sampling_plan_seed{seed}.npy"
        if not plan_path.is_file():
            raise FileNotFoundError(f"Missing sealed sampling plan: {plan_path}")
        plan = np.load(plan_path, mmap_mode="r", allow_pickle=False)
        if plan.shape != (STAGE2_EPOCHS, count) or plan.dtype != np.int32:
            raise RuntimeError(f"Unexpected sampling plan shape/dtype for seed {seed}: {plan.shape}/{plan.dtype}")
        plan_audit = validate_class_aware_plan(train_labels, seed, plan)
        if plan_audit.get("matches_official_sequence") is not True:
            raise RuntimeError(f"Sampling plan does not match the pinned official sampler for seed {seed}.")

        c0_dir = BASE_ROOT / "results" / f"seed_{seed}" / "C0_BALANCED_CE"
        trace_path = c0_dir / "batch_trace.csv"
        status_path = c0_dir / "status.json"
        if not trace_path.is_file() or not status_path.is_file():
            raise FileNotFoundError(
                f"Missing completed C0 reference trace/status for seed {seed}; no substitute trace will be generated."
            )
        status = read_json(status_path)
        if status.get("state") != "completed" or status.get("official_TEST_used") is not False:
            raise RuntimeError(f"C0 reference status is incomplete or not TEST-free for seed {seed}.")
        trace_rows = read_csv_rows(trace_path)
        if len(trace_rows) != expected_trace_rows:
            raise RuntimeError(
                f"C0 reference trace length mismatch for seed {seed}: {len(trace_rows)} != {expected_trace_rows}."
            )
        for index, row in enumerate(trace_rows):
            for key in TRACE_FIELDS:
                if key not in row or row[key] == "":
                    raise RuntimeError(f"C0 trace missing {key} at row {index}, seed {seed}.")
        seed_records[str(seed)] = {
            "sampling_plan_sha256": hash_arrays(plan),
            "sampling_plan_indices": int(plan.size),
            "official_sampler_matches": True,
            "official_sampler_source_sha256": plan_audit["official_source_sha256"],
            "c0_trace_sha256": sha256_file(trace_path),
            "c0_trace_rows": len(trace_rows),
            "c0_status_state": status.get("state"),
        }

    if manifest.get("clean_stage1_sha256") != "7ced9d07f71335357c08623573f038acd0dfdcdeadb55d95d57b332a14f83a88":
        raise RuntimeError("Clean Stage-I checkpoint hash differs from the sealed CC-SAM protocol.")
    if manifest.get("validation_baseline_overall_matches_sealed_clean_protocol") is not True:
        raise RuntimeError("Prepared TRAIN-derived holdout failed the Clean Stage-I baseline reproduction gate.")
    return {
        "passed": True,
        "dataset": "CIFAR-100-LT IF=100",
        "backbone": "frozen ResNet-32",
        "feature": "64-D Clean-compatible GAP",
        "train_samples": int(len(train_labels)),
        "validation_samples": int(len(context["validation_labels"])),
        "train_class_counts": context["counts"].tolist(),
        "many_definition": "original TRAIN count > 100",
        "medium_definition": "20 <= original TRAIN count <= 100",
        "few_definition": "original TRAIN count < 20",
        "official_TEST_used": False,
        "prepared_manifest_sha256": sha256_file(context["manifest_path"]),
        "clean_stage1_sha256": manifest["clean_stage1_sha256"],
        "transform_source_sha256": manifest["transform_source_sha256"],
        "seeds": seed_records,
        "code_sha256": code_hashes(),
    }


def require_test_guard():
    guard = guard_pass()
    if guard.get("official_test_file_open_attempts") != 0 or guard.get("CIFAR100_train_false_attempts") != 0:
        raise RuntimeError("Official TEST access guard did not remain clean.")
    return guard


@torch.no_grad()
def classifier_logits(classifier, features, device):
    classifier.eval()
    chunks = []
    fc = classifier.module.fc
    for start in range(0, len(features), 512):
        x = torch.from_numpy(np.array(features[start:start + 512], copy=True)).to(device)
        logits = fc(x)
        if logits.shape != (len(x), NUM_CLASSES) or not torch.isfinite(logits).all():
            raise RuntimeError("Malformed/non-finite validation logits.")
        chunks.append(logits.detach())
    classifier.train()
    return torch.cat(chunks, dim=0)


def choose_fixed_competitors(initial_logits, validation_labels, counts, device):
    labels = torch.as_tensor(validation_labels, dtype=torch.long, device=device)
    count_tensor = torch.as_tensor(counts, dtype=torch.long, device=device)
    nonmany_ids = torch.nonzero(count_tensor <= MANY_THRESHOLD_EXCLUSIVE, as_tuple=False).flatten()
    many_mask = count_tensor[labels] > MANY_THRESHOLD_EXCLUSIVE
    if nonmany_ids.numel() == 0 or int(many_mask.sum().item()) == 0:
        raise RuntimeError("Fixed-margin audit requires both Many and non-Many validation classes.")
    q_local = initial_logits[:, nonmany_ids].argmax(dim=1)
    competitors = nonmany_ids[q_local]
    if bool((count_tensor[competitors] > MANY_THRESHOLD_EXCLUSIVE).any().item()):
        raise RuntimeError("Initial fixed competitor selection leaked a Many class.")
    return competitors, many_mask, nonmany_ids


def margin_diagnostics(logits, labels, counts, fixed_competitors):
    device = logits.device
    labels_t = torch.as_tensor(labels, dtype=torch.long, device=device)
    counts_t = torch.as_tensor(counts, dtype=torch.long, device=device)
    fixed_q = fixed_competitors.to(device=device, dtype=torch.long)
    many_mask = counts_t[labels_t] > MANY_THRESHOLD_EXCLUSIVE
    if int(many_mask.sum().item()) == 0:
        raise RuntimeError("No Many validation samples are available.")
    row = torch.arange(len(labels_t), device=device)
    true_logits = logits[row, labels_t]
    fixed_margin = true_logits - logits[row, fixed_q]
    nonmany_ids = torch.nonzero(counts_t <= MANY_THRESHOLD_EXCLUSIVE, as_tuple=False).flatten()
    dynamic_q = nonmany_ids[logits[:, nonmany_ids].argmax(dim=1)]
    dynamic_margin = true_logits - logits[row, dynamic_q]
    many_fixed = fixed_margin[many_mask]
    many_dynamic = dynamic_margin[many_mask]
    predictions = logits.argmax(dim=1).cpu().numpy()
    accuracy = metrics_from_predictions(predictions, np.asarray(labels, dtype=np.int64), np.asarray(counts, dtype=np.int64))
    return {
        "accuracy": accuracy,
        "many_samples": int(many_mask.sum().item()),
        "fixed_margin_mean": float(many_fixed.mean().item()),
        "fixed_margin_median": float(many_fixed.median().item()),
        "fixed_margin_p10": float(torch.quantile(many_fixed, 0.10).item()),
        "fixed_margin_negative_fraction": float((many_fixed < 0).float().mean().item()),
        "dynamic_margin_mean": float(many_dynamic.mean().item()),
        "dynamic_margin_median": float(many_dynamic.median().item()),
        "dynamic_margin_p10": float(torch.quantile(many_dynamic, 0.10).item()),
        "dynamic_margin_negative_fraction": float((many_dynamic < 0).float().mean().item()),
        "dynamic_competitor_changed_fraction": float((dynamic_q[many_mask] != fixed_q[many_mask]).float().mean().item()),
        "per_class": {
            int(class_id): {
                "n": int((many_mask & (labels_t == int(class_id))).sum().item()),
                "fixed_mean": float(fixed_margin[many_mask & (labels_t == int(class_id))].mean().item()),
                "fixed_median": float(fixed_margin[many_mask & (labels_t == int(class_id))].median().item()),
                "fixed_negative_fraction": float(
                    (fixed_margin[many_mask & (labels_t == int(class_id))] < 0).float().mean().item()
                ),
                "dynamic_mean": float(dynamic_margin[many_mask & (labels_t == int(class_id))].mean().item()),
            }
            for class_id in torch.unique(labels_t[many_mask]).tolist()
        },
    }


def append_evaluation_rows(seed, epoch, arm, diagnostics, metric_rows, per_class_rows):
    accuracy = diagnostics["accuracy"]
    metric_rows.append({
        "seed": seed,
        "epoch": epoch,
        "arm": arm,
        "overall": accuracy["Overall"],
        "macro": accuracy["Macro"],
        "many_accuracy": accuracy["Many"],
        "mediumN_accuracy": accuracy["MediumN"],
        "few_accuracy": accuracy["Few"],
        "many_samples": diagnostics["many_samples"],
        "fixed_margin_mean": diagnostics["fixed_margin_mean"],
        "fixed_margin_median": diagnostics["fixed_margin_median"],
        "fixed_margin_p10": diagnostics["fixed_margin_p10"],
        "fixed_margin_negative_fraction": diagnostics["fixed_margin_negative_fraction"],
        "dynamic_margin_mean": diagnostics["dynamic_margin_mean"],
        "dynamic_margin_median": diagnostics["dynamic_margin_median"],
        "dynamic_margin_p10": diagnostics["dynamic_margin_p10"],
        "dynamic_margin_negative_fraction": diagnostics["dynamic_margin_negative_fraction"],
        "dynamic_competitor_changed_fraction": diagnostics["dynamic_competitor_changed_fraction"],
    })
    for class_id, values in diagnostics["per_class"].items():
        per_class_rows.append({
            "seed": seed,
            "epoch": epoch,
            "arm": arm,
            "target_many_class": class_id,
            "validation_samples": values["n"],
            "fixed_margin_mean": values["fixed_mean"],
            "fixed_margin_median": values["fixed_median"],
            "fixed_margin_negative_fraction": values["fixed_negative_fraction"],
            "dynamic_margin_mean": values["dynamic_mean"],
        })


def compute_paired_gradients(classifier2, classifier3, gap, labels, class_weights, counts, epoch):
    fc2 = classifier2.module.fc
    fc3 = classifier3.module.fc
    loss2_pack = compute_arm_loss(C2_ARM, gap, labels, fc2, class_weights, epoch)
    loss3_pack = compute_arm_loss(
        C3_ARM, gap, labels, fc3, class_weights, epoch, train_counts=counts
    )
    loss2, logits2, clean2, adv2, delta2, pstats2, alpha2 = loss2_pack
    loss3, logits3, clean3, adv3, delta3, pstats3, alpha3 = loss3_pack
    if abs(float(alpha2) - float(alpha3)) > 1e-12:
        raise RuntimeError("C2 and C3 alpha schedules diverged.")
    if not torch.equal(delta2, torch.zeros_like(delta2)):
        raise RuntimeError("C2 unexpectedly generated a feature perturbation.")
    many_input = torch.as_tensor(counts, dtype=torch.long, device=labels.device)[labels] > MANY_THRESHOLD_EXCLUSIVE
    if not torch.equal(delta3[many_input], torch.zeros_like(delta3[many_input])):
        raise RuntimeError("C3 Many input perturbation is not exactly zero.")

    with torch.no_grad():
        logits_adv3 = fc3(gap.detach() + delta3.detach())
        source_grad_w, source_grad_b = decompose_perturbation_gradient_by_source(
            gap.detach(), labels, delta3.detach(), logits3.detach(), logits_adv3.detach(),
            class_weights, float(alpha3)
        )
        clean2_w, clean2_b = clean_weighted_ce_gradient(
            gap.detach(), labels, logits2.detach(), class_weights, float(alpha2)
        )
        clean3_w, clean3_b = clean_weighted_ce_gradient(
            gap.detach(), labels, logits3.detach(), class_weights, float(alpha3)
        )

    classifier2.zero_grad(set_to_none=True)
    classifier3.zero_grad(set_to_none=True)
    loss2.backward()
    loss3.backward()
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in fc2.parameters()):
        raise RuntimeError("C2 classifier has missing or non-finite gradients.")
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in fc3.parameters()):
        raise RuntimeError("C3 classifier has missing or non-finite gradients.")

    grad2_w, grad2_b = fc2.weight.grad.detach().clone(), fc2.bias.grad.detach().clone()
    grad3_w, grad3_b = fc3.weight.grad.detach().clone(), fc3.bias.grad.detach().clone()
    direct_w, direct_b = source_grad_w.sum(dim=0), source_grad_b.sum(dim=0)
    state_w, state_b = clean3_w - clean2_w, clean3_b - clean2_b
    total_w, total_b = grad3_w - grad2_w, grad3_b - grad2_b

    checks = {
        "c2_clean_gradient_max_abs": max(
            float((grad2_w - clean2_w).abs().max().item()),
            float((grad2_b - clean2_b).abs().max().item()),
        ),
        "direct_source_sum_max_abs": max(
            float(((grad3_w - clean3_w) - direct_w).abs().max().item()),
            float(((grad3_b - clean3_b) - direct_b).abs().max().item()),
        ),
        "total_gradient_decomposition_max_abs": max(
            float((total_w - direct_w - state_w).abs().max().item()),
            float((total_b - direct_b - state_b).abs().max().item()),
        ),
    }
    if not torch.allclose(grad2_w, clean2_w, rtol=GRADIENT_DECOMPOSITION_RTOL, atol=GRADIENT_DECOMPOSITION_ATOL):
        raise RuntimeError(f"C2 analytic gradient check failed: {checks}")
    if not torch.allclose(grad2_b, clean2_b, rtol=GRADIENT_DECOMPOSITION_RTOL, atol=GRADIENT_DECOMPOSITION_ATOL):
        raise RuntimeError(f"C2 analytic bias check failed: {checks}")
    if not torch.allclose(grad3_w - clean3_w, direct_w, rtol=GRADIENT_DECOMPOSITION_RTOL, atol=GRADIENT_DECOMPOSITION_ATOL):
        raise RuntimeError(f"Direct source gradient decomposition failed: {checks}")
    if not torch.allclose(grad3_b - clean3_b, direct_b, rtol=GRADIENT_DECOMPOSITION_RTOL, atol=GRADIENT_DECOMPOSITION_ATOL):
        raise RuntimeError(f"Direct source bias decomposition failed: {checks}")
    if not torch.allclose(total_w, direct_w + state_w, rtol=GRADIENT_DECOMPOSITION_RTOL, atol=GRADIENT_DECOMPOSITION_ATOL):
        raise RuntimeError(f"Total weight-gradient decomposition failed: {checks}")
    if not torch.allclose(total_b, direct_b + state_b, rtol=GRADIENT_DECOMPOSITION_RTOL, atol=GRADIENT_DECOMPOSITION_ATOL):
        raise RuntimeError(f"Total bias-gradient decomposition failed: {checks}")

    many_source_ids = torch.as_tensor(counts, dtype=torch.long, device=labels.device) > MANY_THRESHOLD_EXCLUSIVE
    if torch.count_nonzero(source_grad_w[many_source_ids]).item() != 0 or torch.count_nonzero(source_grad_b[many_source_ids]).item() != 0:
        raise RuntimeError("Many source classes unexpectedly contribute direct perturbation gradients.")

    return {
        "loss2": loss2,
        "loss3": loss3,
        "logits2": logits2.detach(),
        "logits3": logits3.detach(),
        "clean2": clean2.detach(),
        "adv3": adv3.detach(),
        "delta3": delta3.detach(),
        "perturb_stats": pstats3,
        "alpha": float(alpha3),
        "source_grad_weight": source_grad_w,
        "source_grad_bias": source_grad_b,
        "direct_grad_weight": direct_w,
        "direct_grad_bias": direct_b,
        "state_grad_weight": state_w,
        "state_grad_bias": state_b,
        "total_grad_weight": total_w,
        "total_grad_bias": total_b,
        "grad2_weight": grad2_w,
        "grad2_bias": grad2_b,
        "grad3_weight": grad3_w,
        "grad3_bias": grad3_b,
        "checks": checks,
        "loss2_value": float(loss2.detach().item()),
        "loss3_value": float(loss3.detach().item()),
    }


def rounding_tensor_record(value, include_values):
    flat_index = value.abs().argmax()
    index = [int(axis) for axis in torch.unravel_index(flat_index, value.shape)]
    record = {
        "shape": list(value.shape),
        "max_abs": value.abs().max().item(),
        "rms": value.square().mean().sqrt().item(),
        "max_abs_index": index,
        "value_at_max_abs": value.flatten()[flat_index].item(),
    }
    if include_values:
        record["values"] = value.detach().cpu().tolist()
    return record


def writeback_rounding_diagnostics(ledger2, ledger3, include_values):
    report = {"steps": ledger2.steps}
    for name in ("weight", "bias"):
        values = {}
        for arm, ledger in (("C2", ledger2), ("C3", ledger3)):
            step = ledger.last_step[name]
            residual, bound = step["residual"], step["bound"]
            denominator = torch.where(bound > 0, bound, torch.ones_like(bound))
            values[arm] = {
                **rounding_tensor_record(residual, include_values),
                "bound_max_abs": bound.abs().max().item(),
                "maximum_residual_over_bound": (residual.abs() / denominator).max().item(),
                "replay_bitwise_equal": step["replay_bitwise_equal"],
                "learning_rate": step["learning_rate"],
                "foreach": step["foreach"],
                "nesterov": step["nesterov"],
                "momentum_buffer_initialized": step["momentum_buffer_initialized"],
                "before_sha256": hash_arrays(step["before"].cpu().numpy()),
                "after_sha256": hash_arrays(step["after"].cpu().numpy()),
                "update_direction_sha256": hash_arrays(step["update_direction"].cpu().numpy()),
            }
        values["step_delta_R"] = rounding_tensor_record(
            ledger3.last_step[name]["residual"] - ledger2.last_step[name]["residual"], include_values
        )
        values["cumulative_delta_R"] = rounding_tensor_record(
            ledger3.cumulative[name] - ledger2.cumulative[name], include_values
        )
        report[name] = values
    return report


def reconstruction_diagnostics(tracker, classifier2, classifier3, optimizer2, optimizer3,
                               writeback_ledgers=None, include_rounding_values=False):
    fc2 = classifier2.module.fc
    fc3 = classifier3.module.fc
    components = tracker.parameter_components()
    buffer_components = tracker.buffer_components()

    actual_weight = fc3.weight.detach().double() - fc2.weight.detach().double()
    actual_bias = fc3.bias.detach().double() - fc2.bias.detach().double()
    predicted_weight = (
        components["direct_weight"] + components["state_weight"] + components["decay_weight"]
    )
    predicted_bias = (
        components["direct_bias"] + components["state_bias"] + components["decay_bias"]
    )
    raw_weight_abs, raw_weight_rel = max_abs_and_relative_error(
        fc3.weight.detach() - fc2.weight.detach(), predicted_weight
    )
    raw_bias_abs, raw_bias_rel = max_abs_and_relative_error(fc3.bias.detach() - fc2.bias.detach(), predicted_bias)
    if writeback_ledgers is not None:
        ledger2, ledger3 = writeback_ledgers
        if ledger2 is ledger3 or ledger2.steps == 0 or ledger2.steps != ledger3.steps:
            raise RuntimeError("Writeback ledgers must contain matching independent paired steps")
        for name in ("weight", "bias"):
            for ledger, fc in ((ledger2, fc2), (ledger3, fc3)):
                if not torch.equal(ledger.last_step[name]["after"], getattr(fc, name).detach()):
                    raise RuntimeError("Writeback ledger does not belong to the current classifier state")
        predicted_weight = predicted_weight + ledger3.cumulative["weight"] - ledger2.cumulative["weight"]
        predicted_bias = predicted_bias + ledger3.cumulative["bias"] - ledger2.cumulative["bias"]
    weight_abs, weight_rel = max_abs_and_relative_error(actual_weight, predicted_weight)
    bias_abs, bias_rel = max_abs_and_relative_error(actual_bias, predicted_bias)

    actual_buffer_weight = (
        optimizer3.state[fc3.weight]["momentum_buffer"].double()
        - optimizer2.state[fc2.weight]["momentum_buffer"].double()
    )
    actual_buffer_bias = (
        optimizer3.state[fc3.bias]["momentum_buffer"].double()
        - optimizer2.state[fc2.bias]["momentum_buffer"].double()
    )
    predicted_buffer_weight = (
        buffer_components["direct_weight"] + buffer_components["state_weight"] + buffer_components["decay_weight"]
    )
    predicted_buffer_bias = (
        buffer_components["direct_bias"] + buffer_components["state_bias"] + buffer_components["decay_bias"]
    )
    for value in (actual_weight, actual_bias, predicted_weight, predicted_bias,
                  actual_buffer_weight, actual_buffer_bias, predicted_buffer_weight, predicted_buffer_bias):
        if not torch.isfinite(value).all():
            raise RuntimeError("Nonfinite parameter or momentum reconstruction")
    buffer_weight_abs, buffer_weight_rel = max_abs_and_relative_error(
        actual_buffer_weight, predicted_buffer_weight
    )
    buffer_bias_abs, buffer_bias_rel = max_abs_and_relative_error(
        actual_buffer_bias, predicted_buffer_bias
    )
    parameter_max_rel = max(weight_rel, bias_rel)
    buffer_max_rel = max(buffer_weight_rel, buffer_bias_rel)
    max_rel = max(parameter_max_rel, buffer_max_rel)
    max_abs = max(weight_abs, bias_abs, buffer_weight_abs, buffer_bias_abs)
    if max_rel > MOMENTUM_RECONSTRUCTION_REL_TOL or max_abs > MOMENTUM_RECONSTRUCTION_ABS_TOL:
        raise RuntimeError(
            "SGD momentum/weight-decay attribution failed reconstruction: "
            f"weight_rel={weight_rel:.6g}, bias_rel={bias_rel:.6g}, "
            f"buffer_weight_rel={buffer_weight_rel:.6g}, buffer_bias_rel={buffer_bias_rel:.6g}, max_abs={max_abs:.6g}"
        )
    return {
        "parameter_weight_max_abs": weight_abs,
        "parameter_weight_relative": weight_rel,
        "parameter_bias_max_abs": bias_abs,
        "parameter_bias_relative": bias_rel,
        "parameter_max_relative": parameter_max_rel,
        "momentum_buffer_weight_max_abs": buffer_weight_abs,
        "momentum_buffer_weight_relative": buffer_weight_rel,
        "momentum_buffer_bias_max_abs": buffer_bias_abs,
        "momentum_buffer_bias_relative": buffer_bias_rel,
        "momentum_buffer_max_relative": buffer_max_rel,
        "max_relative_reconstruction_error": max_rel,
        "max_absolute_reconstruction_error": max_abs,
        "raw_parameter_weight_max_abs": raw_weight_abs,
        "raw_parameter_weight_relative": raw_weight_rel,
        "raw_parameter_bias_max_abs": raw_bias_abs,
        "raw_parameter_bias_relative": raw_bias_rel,
        "parameter_writeback_rounding": (
            writeback_rounding_diagnostics(*writeback_ledgers, include_values=include_rounding_values)
            if writeback_ledgers is not None else None
        ),
    }


def make_trace_row(epoch, batch_index, indices, target_np, images_cpu, gap):
    return {
        "epoch": str(epoch + 1),
        "batch": str(batch_index),
        "batch_size": str(len(indices)),
        "indices_labels_sha256": hash_arrays(indices, target_np),
        "augmented_images_sha256": hash_arrays(images_cpu.numpy()),
        "gap_features_sha256": hash_arrays(gap.detach().cpu().numpy()),
    }


def check_trace_row(actual, expected, seed, epoch, batch_index):
    mismatch = [key for key in TRACE_FIELDS if actual.get(key) != expected.get(key)]
    if mismatch:
        raise RuntimeError(
            f"Paired C0 trace mismatch seed={seed}, epoch={epoch + 1}, batch={batch_index}: {mismatch}"
        )


def fixed_competitor_csv(path, validation_labels, fixed_competitors, many_mask):
    rows = []
    q = fixed_competitors.detach().cpu().numpy()
    labels = np.asarray(validation_labels, dtype=np.int64)
    mask = many_mask.detach().cpu().numpy().astype(bool)
    for index in np.flatnonzero(mask):
        rows.append({
            "validation_index": int(index),
            "true_many_class": int(labels[index]),
            "initial_fixed_nonmany_competitor": int(q[index]),
        })
    write_csv_rows(path, ("validation_index", "true_many_class", "initial_fixed_nonmany_competitor"), rows)


def evaluate_epoch(seed, epoch, classifier2, classifier3, context, fixed_q, many_mask, out_dir,
                   metric_rows, per_class_rows):
    val_features = context["validation_gap"]
    val_labels = context["validation_labels"]
    counts = context["counts"]
    device = next(classifier2.parameters()).device
    logits2 = classifier_logits(classifier2, val_features, device)
    logits3 = classifier_logits(classifier3, val_features, device)
    diag2 = margin_diagnostics(logits2, val_labels, counts, fixed_q)
    diag3 = margin_diagnostics(logits3, val_labels, counts, fixed_q)
    append_evaluation_rows(seed, epoch, "C2", diag2, metric_rows, per_class_rows)
    append_evaluation_rows(seed, epoch, "C3", diag3, metric_rows, per_class_rows)
    return logits2, logits3, diag2, diag3


def run_seed_smoke(seed, context, device):
    seed_everything(seed, deterministic=True)
    checkpoint = resolve_checkpoint(context["manifest"]["checkpoint_path"])
    backbone, classifier3, _ = build_models(checkpoint, device)
    classifier2 = copy.deepcopy(classifier3).to(device)
    if state_hash(classifier2) != state_hash(classifier3):
        raise RuntimeError("Smoke C2/C3 initial classifier states differ.")
    frozen_backbone = {key: value.detach().cpu().clone() for key, value in backbone.state_dict().items()}
    optimizer2 = torch.optim.SGD(
        classifier2.module.fc.parameters(), lr=STAGE2_LR, momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY, nesterov=False
    )
    optimizer3 = torch.optim.SGD(
        classifier3.module.fc.parameters(), lr=STAGE2_LR, momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY, nesterov=False
    )
    counts = context["counts"]
    weights = class_weights_from_counts(counts, device)
    plan = np.load(BASE_ROOT / "prepared" / f"sampling_plan_seed{seed}.npy", mmap_mode="r", allow_pickle=False)
    generator = torch.Generator(device="cpu").manual_seed(seed + 300009)
    reference = read_csv_rows(
        BASE_ROOT / "results" / f"seed_{seed}" / "C0_BALANCED_CE" / "batch_trace.csv"
    )
    val_features = torch.from_numpy(context["validation_gap"]).to(device)
    val_labels = torch.as_tensor(context["validation_labels"], dtype=torch.long, device=device)
    with torch.no_grad():
        initial_logits = classifier3.module.fc(val_features)
    fixed_q, many_mask, _ = choose_fixed_competitors(initial_logits, context["validation_labels"], counts, device)
    tracker = SGDInfluenceTracker(NUM_CLASSES, FEATURE_DIM, device, MOMENTUM, STAGE2_LR, WEIGHT_DECAY)

    initial_classifier_sha = state_hash(classifier3)
    writeback2, writeback3 = ParameterWritebackRoundingLedger(), ParameterWritebackRoundingLedger()
    checked = []
    for batch_index in range(2):
        indices = np.asarray(plan[0, batch_index * BATCH_SIZE:min((batch_index + 1) * BATCH_SIZE, len(context["train_labels"]))], dtype=np.int64)
        target_np = np.asarray(context["train_labels"][indices], dtype=np.int64)
        labels = torch.from_numpy(target_np).to(device)
        images_cpu = augment_batch(context["train_images"], indices, context["train_transform"], generator)
        with torch.no_grad():
            gap = clean_compatible_gap(backbone(images_cpu.to(device, non_blocking=True)))
        trace = make_trace_row(0, batch_index, indices, target_np, images_cpu, gap)
        check_trace_row(trace, reference[batch_index], seed, 0, batch_index)

        pack = compute_paired_gradients(classifier2, classifier3, gap, labels, weights, counts, 0)
        source_pressure = project_source_gradients_to_fixed_margin(
            pack["source_grad_weight"], pack["source_grad_bias"],
            val_features, val_labels, fixed_q, many_mask, STAGE2_LR
        )
        direct_pressure = project_gradient_to_fixed_margin(
            pack["direct_grad_weight"], pack["direct_grad_bias"],
            val_features, val_labels, fixed_q, many_mask, STAGE2_LR
        )
        margin_additivity_error = float(
            (source_pressure["margin_step"].sum(dim=0) - direct_pressure["margin_step"]).abs().max().item()
        )
        if margin_additivity_error > 5e-5:
            raise RuntimeError(f"Smoke source-margin projection is not additive: {margin_additivity_error}")
        tracker.advance(
            pack["source_grad_weight"], pack["source_grad_bias"],
            pack["state_grad_weight"], pack["state_grad_bias"],
            classifier3.module.fc.weight.detach() - classifier2.module.fc.weight.detach(),
            classifier3.module.fc.bias.detach() - classifier2.module.fc.bias.detach(),
        )
        writeback2.capture(classifier2.module.fc.named_parameters(), optimizer2)
        writeback3.capture(classifier3.module.fc.named_parameters(), optimizer3)
        optimizer2.step()
        optimizer3.step()
        writeback2.record()
        writeback3.record()
        reconstruction = reconstruction_diagnostics(
            tracker, classifier2, classifier3, optimizer2, optimizer3,
            writeback_ledgers=(writeback2, writeback3), include_rounding_values=True,
        )
        checked.append({
            "batch": batch_index,
            "batch_trace_matches_C0": True,
            "alpha": pack["alpha"],
            "c2_loss": pack["loss2_value"],
            "c3_loss": pack["loss3_value"],
            "mean_delta_l2": float(torch.linalg.vector_norm(pack["delta3"], dim=1).mean().item()),
            "source_margin_additivity_max_abs": margin_additivity_error,
            "direct_fixed_many_margin_step_mean": float(direct_pressure["margin_step"].mean().item()),
            "gradient_checks": pack["checks"],
            "sgd_reconstruction": reconstruction,
        })

    check_backbone_unchanged(backbone, frozen_backbone)
    guard = require_test_guard()
    return {
        "seed": seed,
        "initial_classifier_sha256": initial_classifier_sha,
        "two_real_batches_passed": True,
        "batches": checked,
        "backbone_and_bn_bitwise_unchanged": True,
        "official_TEST_used": False,
        "test_guard": guard,
    }


def validation_directory(directory=None):
    if directory is None:
        return CCMIA_ROOT
    resolved = (CCMIA_ROOT / directory).resolve()
    if not resolved.is_relative_to((CCMIA_ROOT / "diagnostics").resolve()):
        raise ValueError("Independent validation reports must remain under ccmia/diagnostics/")
    return resolved


def run_preflight(validation_dir=None):
    context = prepare_context()
    audit = validate_inputs(context)
    guard = require_test_guard()
    report = {
        **audit,
        "completed_at": __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds"),
        "test_guard": guard,
        "source_branch_note": "CC-MIA is isolated from C0-C4 and TPR production code; no protected source file is modified.",
    }
    path = validation_directory(validation_dir) / "preflight_report.json"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    write_json(path, report)
    print("CC-MIA PREFLIGHT PASS")
    return report


def run_smoke(validation_dir=None):
    context = prepare_context()
    audit = validate_inputs(context)
    report_root = validation_directory(validation_dir)
    report_path = report_root / "smoke_report.json"
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite {report_path}")
    preflight_path = report_root / "preflight_report.json"
    if not preflight_path.is_file():
        raise RuntimeError("Run --phase preflight first.")
    preflight = read_json(preflight_path)
    if preflight.get("passed") is not True or preflight.get("prepared_manifest_sha256") != audit["prepared_manifest_sha256"]:
        raise RuntimeError("Preflight report is missing, failed or belongs to different prepared inputs.")
    if preflight.get("code_sha256") != audit["code_sha256"]:
        raise RuntimeError("Code changed after preflight; rerun preflight before Smoke.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is intentionally disabled.")
    device = torch.device("cuda:0")
    seed_reports = [run_seed_smoke(seed, context, device) for seed in SEEDS]
    guard = require_test_guard()
    report = {
        "passed": True,
        "completed_at": __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds"),
        "purpose": "two real paired C2/C3 batches per seed; optimizer updates discarded",
        "formal_training_started": False,
        "prepared_manifest_sha256": audit["prepared_manifest_sha256"],
        "code_sha256": audit["code_sha256"],
        "seeds": seed_reports,
        "test_guard": guard,
        "official_TEST_used": False,
    }
    write_json(report_path, report)
    print("CC-MIA SMOKE PASS: exact source-gradient decomposition, paired C0 traces, SGD momentum/weight-decay reconstruction.")
    print("No formal training was started.")
    return report


def source_group_name(class_id, counts):
    n = int(counts[int(class_id)])
    if n > MANY_THRESHOLD_EXCLUSIVE:
        return "Many"
    if n >= MEDIUM_MIN_INCLUSIVE:
        return "MediumN"
    return "Few"


def initialize_epoch_accumulators(device):
    return {
        "source_margin_sum": torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.float64, device=device),
        "source_margin_negative_fraction_sum": torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.float64, device=device),
        "source_target_logit_sum": torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.float64, device=device),
        "source_competitor_logit_sum": torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.float64, device=device),
        "source_target_steps": torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.float64, device=device),
        "source_presence_batches": torch.zeros(NUM_CLASSES, dtype=torch.float64, device=device),
        "source_sample_sum": torch.zeros(NUM_CLASSES, dtype=torch.float64, device=device),
        "group_norm_sum": torch.zeros(NUM_CLASSES, 3, dtype=torch.float64, device=device),
        "group_norm_present_sum": torch.zeros(NUM_CLASSES, 3, dtype=torch.float64, device=device),
        "group_norm_present_batches": torch.zeros(NUM_CLASSES, 3, dtype=torch.float64, device=device),
        "step_count": 0,
    }


def accumulate_source_diagnostics(acc, source_grad_w, source_grad_b, labels, source_pressure, counts):
    batch_counts = torch.bincount(labels, minlength=NUM_CLASSES).to(torch.float64)
    acc["source_sample_sum"] += batch_counts
    present = batch_counts > 0
    acc["source_presence_batches"] += present.to(torch.float64)

    target_labels = source_pressure["target_labels"]
    margin = source_pressure["margin_step"].to(torch.float64)
    target_step = source_pressure["target_logit_step"].to(torch.float64)
    competitor_step = source_pressure["competitor_logit_step"].to(torch.float64)
    scatter_index = target_labels.unsqueeze(0).expand(NUM_CLASSES, -1)
    class_count = torch.bincount(target_labels, minlength=NUM_CLASSES).to(torch.float64)
    valid_class = class_count > 0
    denom = class_count.clamp_min(1.0).unsqueeze(0)
    acc["source_margin_sum"] += torch.zeros_like(acc["source_margin_sum"]).scatter_add_(1, scatter_index, margin) / denom
    acc["source_margin_negative_fraction_sum"] += (
        torch.zeros_like(acc["source_margin_negative_fraction_sum"]).scatter_add_(
            1, scatter_index, (margin < 0).to(torch.float64)
        ) / denom
    )
    acc["source_target_logit_sum"] += (
        torch.zeros_like(acc["source_target_logit_sum"]).scatter_add_(1, scatter_index, target_step) / denom
    )
    acc["source_competitor_logit_sum"] += (
        torch.zeros_like(acc["source_competitor_logit_sum"]).scatter_add_(1, scatter_index, competitor_step) / denom
    )
    acc["source_target_steps"] += valid_class.to(torch.float64).unsqueeze(0).expand(NUM_CLASSES, -1)

    source_w = source_grad_w.detach().to(torch.float64)
    source_b = source_grad_b.detach().to(torch.float64)
    group_ids = (
        torch.as_tensor(np.flatnonzero(counts > MANY_THRESHOLD_EXCLUSIVE), device=source_grad_w.device),
        torch.as_tensor(np.flatnonzero((counts >= MEDIUM_MIN_INCLUSIVE) & (counts <= MANY_THRESHOLD_EXCLUSIVE)), device=source_grad_w.device),
        torch.as_tensor(np.flatnonzero(counts < MEDIUM_MIN_INCLUSIVE), device=source_grad_w.device),
    )
    for group_index, ids in enumerate(group_ids):
        if ids.numel() == 0:
            continue
        norm = torch.sqrt(
            source_w.index_select(1, ids).square().sum(dim=(1, 2))
            + source_b.index_select(1, ids).square().sum(dim=1)
        )
        acc["group_norm_sum"][:, group_index] += norm
        acc["group_norm_present_sum"][:, group_index] += norm * present.to(torch.float64)
        acc["group_norm_present_batches"][:, group_index] += present.to(torch.float64)
    acc["step_count"] += 1


def source_group_gradient_rows(seed, epoch, acc, counts):
    rows = []
    source_ids = np.flatnonzero(counts <= MANY_THRESHOLD_EXCLUSIVE)
    group_names = ("Many", "MediumN", "Few")
    steps = max(int(acc["step_count"]), 1)
    norm_all = acc["group_norm_sum"].detach().cpu().numpy()
    norm_present = acc["group_norm_present_sum"].detach().cpu().numpy()
    present_group = acc["group_norm_present_batches"].detach().cpu().numpy()
    present_source = acc["source_presence_batches"].detach().cpu().numpy()
    for source_id in source_ids:
        for group_index, group_name in enumerate(group_names):
            present_steps = float(present_group[source_id, group_index])
            rows.append({
                "seed": seed,
                "epoch": epoch,
                "source_class": int(source_id),
                "source_train_count": int(counts[source_id]),
                "source_group": source_group_name(source_id, counts),
                "target_classifier_group": group_name,
                "mean_batch_gradient_frobenius_all_steps": float(norm_all[source_id, group_index] / steps),
                "mean_batch_gradient_frobenius_when_source_present": (
                    float(norm_present[source_id, group_index] / present_steps)
                    if present_steps > 0 else 0.0
                ),
                "source_presence_batches": present_steps,
                "total_batches": steps,
                "source_batch_presence_rate": float(present_source[source_id] / steps),
            })
    return rows


def decomposition_epoch_rows(seed, epoch, tracker, classifier2, classifier3, context, val_features_t,
                             val_labels_t, fixed_q, many_mask, epoch_acc, writeback_ledgers=None):
    fc2 = classifier2.module.fc
    fc3 = classifier3.module.fc
    components = tracker.parameter_components()
    actual_w = fc3.weight.detach().double() - fc2.weight.detach().double()
    actual_b = fc3.bias.detach().double() - fc2.bias.detach().double()
    rounding_w, rounding_b = torch.zeros_like(actual_w), torch.zeros_like(actual_b)
    if writeback_ledgers is not None:
        ledger2, ledger3 = writeback_ledgers
        rounding_w = ledger3.cumulative["weight"] - ledger2.cumulative["weight"]
        rounding_b = ledger3.cumulative["bias"] - ledger2.cumulative["bias"]

    direct_source = project_source_parameter_delta_to_fixed_margin(
        tracker.direct_parameter_weight, tracker.direct_parameter_bias,
        val_features_t, val_labels_t, fixed_q, many_mask
    )
    state = project_parameter_delta_to_fixed_margin(
        components["state_weight"], components["state_bias"],
        val_features_t, val_labels_t, fixed_q, many_mask
    )
    decay = project_parameter_delta_to_fixed_margin(
        components["decay_weight"], components["decay_bias"],
        val_features_t, val_labels_t, fixed_q, many_mask
    )
    actual = project_parameter_delta_to_fixed_margin(
        actual_w, actual_b, val_features_t, val_labels_t, fixed_q, many_mask
    )
    writeback = project_parameter_delta_to_fixed_margin(
        rounding_w, rounding_b, val_features_t, val_labels_t, fixed_q, many_mask
    )
    direct_margin = direct_source["margin_delta"].sum(dim=0)
    predicted_margin = direct_margin + state["margin_delta"] + decay["margin_delta"] + writeback["margin_delta"]
    residual_margin = actual["margin_delta"] - predicted_margin

    direct_target = direct_source["target_logit_delta"].sum(dim=0)
    direct_comp = direct_source["competitor_logit_delta"].sum(dim=0)
    actual_target = actual["target_logit_delta"]
    actual_comp = actual["competitor_logit_delta"]
    row = {
        "seed": seed,
        "epoch": epoch,
        "many_validation_samples": int(actual["margin_delta"].numel()),
        "actual_c3_minus_c2_fixed_margin_mean": float(actual["margin_delta"].mean().item()),
        "direct_perturbation_component_mean": float(direct_margin.mean().item()),
        "state_feedback_component_mean": float(state["margin_delta"].mean().item()),
        "weight_decay_component_mean": float(decay["margin_delta"].mean().item()),
        "parameter_writeback_rounding_component_mean": float(writeback["margin_delta"].mean().item()),
        "reconstruction_residual_component_mean": float(residual_margin.mean().item()),
        "actual_target_logit_delta_mean": float(actual_target.mean().item()),
        "actual_competitor_logit_delta_mean": float(actual_comp.mean().item()),
        "direct_target_logit_component_mean": float(direct_target.mean().item()),
        "direct_competitor_logit_component_mean": float(direct_comp.mean().item()),
        "direct_negative_fraction": float((direct_margin < 0).float().mean().item()),
        "state_negative_fraction": float((state["margin_delta"] < 0).float().mean().item()),
        "actual_negative_fraction": float((actual["margin_delta"] < 0).float().mean().item()),
        "margin_reconstruction_max_abs": float(residual_margin.abs().max().item()),
        "margin_reconstruction_rmse": float(torch.sqrt(torch.mean(residual_margin.square())).item()),
        "parameter_reconstruction_relative": max(
            max_abs_and_relative_error(
                actual_w, components["direct_weight"] + components["state_weight"] + components["decay_weight"] + rounding_w
            )[1],
            max_abs_and_relative_error(
                actual_b, components["direct_bias"] + components["state_bias"] + components["decay_bias"] + rounding_b
            )[1],
        ),
    }
    source_rows = []
    target_labels = direct_source["target_labels"].detach().cpu().numpy()
    source_margin = direct_source["margin_delta"].detach().cpu().numpy()
    source_target = direct_source["target_logit_delta"].detach().cpu().numpy()
    source_comp = direct_source["competitor_logit_delta"].detach().cpu().numpy()
    counts = context["counts"]
    source_ids = np.flatnonzero(counts <= MANY_THRESHOLD_EXCLUSIVE)
    step_margin = epoch_acc["source_margin_sum"].detach().cpu().numpy()
    step_negative = epoch_acc["source_margin_negative_fraction_sum"].detach().cpu().numpy()
    step_target = epoch_acc["source_target_logit_sum"].detach().cpu().numpy()
    step_competitor = epoch_acc["source_competitor_logit_sum"].detach().cpu().numpy()
    step_counts = epoch_acc["source_target_steps"].detach().cpu().numpy()
    source_presence = epoch_acc["source_presence_batches"].detach().cpu().numpy()
    source_sample_sum = epoch_acc["source_sample_sum"].detach().cpu().numpy()
    total_steps = max(int(epoch_acc["step_count"]), 1)
    for source_id in source_ids:
        for target_class in np.unique(target_labels):
            mask = target_labels == int(target_class)
            n_steps = max(float(step_counts[source_id, int(target_class)]), 1.0)
            source_rows.append({
                "seed": seed,
                "epoch": epoch,
                "source_class": int(source_id),
                "source_train_count": int(counts[source_id]),
                "source_group": source_group_name(source_id, counts),
                "target_many_class": int(target_class),
                "target_validation_samples": int(mask.sum()),
                "cumulative_direct_margin_delta_mean": float(source_margin[source_id, mask].mean()),
                "cumulative_direct_margin_negative_fraction": float((source_margin[source_id, mask] < 0).mean()),
                "cumulative_direct_target_logit_delta_mean": float(source_target[source_id, mask].mean()),
                "cumulative_direct_competitor_logit_delta_mean": float(source_comp[source_id, mask].mean()),
                "mean_direct_gradient_step_margin": float(step_margin[source_id, int(target_class)] / n_steps),
                "mean_direct_gradient_step_negative_fraction": float(step_negative[source_id, int(target_class)] / n_steps),
                "mean_direct_gradient_step_target_logit": float(step_target[source_id, int(target_class)] / n_steps),
                "mean_direct_gradient_step_competitor_logit": float(step_competitor[source_id, int(target_class)] / n_steps),
                "source_batch_presence_rate": float(source_presence[source_id] / total_steps),
                "mean_source_examples_per_batch": float(source_sample_sum[source_id] / total_steps),
            })
    return row, source_rows


def train_seed(seed, context, device):
    out_dir = CCMIA_ROOT / "results" / f"seed_{seed}"
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing CC-MIA run directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=False)
    status_path = out_dir / "status.json"

    status = {
        "state": "running",
        "seed": int(seed),
        "started_at": __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds"),
        "official_TEST_used": False,
        "clean_stage1_sha256": context["manifest"]["clean_stage1_sha256"],
        "prepared_manifest_sha256": sha256_file(context["manifest_path"]),
        "epochs_completed": 0,
        "paired_batch_steps_completed": 0,
    }
    write_json(status_path, status)

    try:
        seed_everything(seed, deterministic=True)
        checkpoint = resolve_checkpoint(context["manifest"]["checkpoint_path"])
        backbone, classifier3, _ = build_models(checkpoint, device)
        classifier2 = copy.deepcopy(classifier3).to(device)
        if state_hash(classifier2) != state_hash(classifier3):
            raise RuntimeError("Paired C2/C3 initial classifier hashes differ.")
        initial_classifier_sha = state_hash(classifier3)
        initial_backbone_sha = state_hash(backbone)
        frozen_backbone = {key: value.detach().cpu().clone() for key, value in backbone.state_dict().items()}

        optimizer2 = torch.optim.SGD(
            classifier2.module.fc.parameters(), lr=STAGE2_LR, momentum=MOMENTUM,
            weight_decay=WEIGHT_DECAY, nesterov=False
        )
        optimizer3 = torch.optim.SGD(
            classifier3.module.fc.parameters(), lr=STAGE2_LR, momentum=MOMENTUM,
            weight_decay=WEIGHT_DECAY, nesterov=False
        )
        if optimizer2.state or optimizer3.state:
            raise RuntimeError("Paired SGD optimizers must start with empty momentum buffers.")

        counts = context["counts"]
        class_weights = class_weights_from_counts(counts, device)
        plan = np.load(BASE_ROOT / "prepared" / f"sampling_plan_seed{seed}.npy", mmap_mode="r", allow_pickle=False)
        reference_path = BASE_ROOT / "results" / f"seed_{seed}" / "C0_BALANCED_CE" / "batch_trace.csv"
        reference_rows = read_csv_rows(reference_path)
        expected_steps = STAGE2_EPOCHS * math.ceil(len(context["train_labels"]) / BATCH_SIZE)
        if len(reference_rows) != expected_steps:
            raise RuntimeError(f"Paired C0 trace length changed after preflight for seed {seed}.")

        augmentation_generator = torch.Generator(device="cpu").manual_seed(seed + 300009)
        val_features_t = torch.from_numpy(context["validation_gap"]).to(device)
        val_labels_t = torch.as_tensor(context["validation_labels"], dtype=torch.long, device=device)
        with torch.no_grad():
            initial_val_logits = classifier3.module.fc(val_features_t)
        fixed_q, many_mask, nonmany_ids = choose_fixed_competitors(
            initial_val_logits, context["validation_labels"], counts, device
        )
        fixed_competitor_csv(
            out_dir / "fixed_competitors.csv", context["validation_labels"], fixed_q, many_mask
        )

        tracker = SGDInfluenceTracker(
            NUM_CLASSES, FEATURE_DIM, device, MOMENTUM, STAGE2_LR, WEIGHT_DECAY
        )
        writeback2, writeback3 = ParameterWritebackRoundingLedger(), ParameterWritebackRoundingLedger()
        metric_rows = []
        per_class_rows = []
        initial_logits2, initial_logits3, initial_diag2, initial_diag3 = evaluate_epoch(
            seed, 0, classifier2, classifier3, context, fixed_q, many_mask, out_dir,
            metric_rows, per_class_rows
        )
        if not torch.equal(initial_logits2, initial_logits3):
            raise RuntimeError("C2/C3 initial validation logits are not bitwise identical.")

        metric_fields = (
            "seed", "epoch", "arm", "overall", "macro", "many_accuracy", "mediumN_accuracy",
            "few_accuracy", "many_samples", "fixed_margin_mean", "fixed_margin_median",
            "fixed_margin_p10", "fixed_margin_negative_fraction", "dynamic_margin_mean",
            "dynamic_margin_median", "dynamic_margin_p10", "dynamic_margin_negative_fraction",
            "dynamic_competitor_changed_fraction",
        )
        per_class_fields = (
            "seed", "epoch", "arm", "target_many_class", "validation_samples",
            "fixed_margin_mean", "fixed_margin_median", "fixed_margin_negative_fraction",
            "dynamic_margin_mean",
        )
        step_fields = (
            "seed", "epoch", "batch", "batch_size", "alpha", "learning_rate",
            "c2_loss", "c3_loss", "mean_feature_delta_l2",
            "direct_gradient_margin_step_mean", "direct_gradient_margin_step_median",
            "direct_gradient_margin_step_negative_fraction", "direct_target_logit_step_mean",
            "direct_competitor_logit_step_mean", "state_feedback_margin_step_mean",
            "state_feedback_margin_step_negative_fraction", "total_gradient_margin_step_mean",
            "total_gradient_margin_step_negative_fraction", "direct_gradient_weight_norm",
            "state_feedback_gradient_weight_norm", "total_gradient_weight_norm",
            "source_additivity_max_abs", "gradient_decomposition_max_abs",
            "parameter_reconstruction_relative", "momentum_buffer_reconstruction_relative",
            "raw_parameter_weight_relative", "raw_parameter_bias_relative",
            "writeback_weight_delta_R_max_abs", "writeback_bias_delta_R_max_abs",
        )
        group_fields = (
            "seed", "epoch", "source_class", "source_train_count", "source_group",
            "target_classifier_group", "mean_batch_gradient_frobenius_all_steps",
            "mean_batch_gradient_frobenius_when_source_present", "source_presence_batches",
            "total_batches", "source_batch_presence_rate",
        )
        source_fields = (
            "seed", "epoch", "source_class", "source_train_count", "source_group",
            "target_many_class", "target_validation_samples", "cumulative_direct_margin_delta_mean",
            "cumulative_direct_margin_negative_fraction", "cumulative_direct_target_logit_delta_mean",
            "cumulative_direct_competitor_logit_delta_mean",
            "mean_direct_gradient_step_margin",
            "mean_direct_gradient_step_negative_fraction",
            "mean_direct_gradient_step_target_logit",
            "mean_direct_gradient_step_competitor_logit",
            "source_batch_presence_rate",
            "mean_source_examples_per_batch",
        )
        decomposition_fields = (
            "seed", "epoch", "many_validation_samples", "actual_c3_minus_c2_fixed_margin_mean",
            "direct_perturbation_component_mean", "state_feedback_component_mean",
            "weight_decay_component_mean", "reconstruction_residual_component_mean",
            "parameter_writeback_rounding_component_mean",
            "actual_target_logit_delta_mean", "actual_competitor_logit_delta_mean",
            "direct_target_logit_component_mean", "direct_competitor_logit_component_mean",
            "direct_negative_fraction", "state_negative_fraction", "actual_negative_fraction",
            "margin_reconstruction_max_abs", "margin_reconstruction_rmse",
            "parameter_reconstruction_relative",
        )

        metrics_path = out_dir / "margin_metrics.csv"
        per_class_path = out_dir / "per_class_margin.csv"
        step_path = out_dir / "step_influence.csv"
        group_path = out_dir / "source_target_group_gradient.csv"
        source_path = out_dir / "source_margin_influence.csv"
        decomposition_path = out_dir / "margin_decomposition.csv"
        trace_path = out_dir / "batch_trace.csv"

        write_csv_rows(metrics_path, metric_fields, metric_rows)
        write_csv_rows(per_class_path, per_class_fields, per_class_rows)

        with trace_path.open("w", encoding="utf-8", newline="") as trace_file, \
             step_path.open("w", encoding="utf-8", newline="") as step_file:
            trace_writer = csv.DictWriter(trace_file, fieldnames=TRACE_FIELDS)
            step_writer = csv.DictWriter(step_file, fieldnames=step_fields)
            trace_writer.writeheader()
            step_writer.writeheader()
            global_batch = 0

            for epoch in range(STAGE2_EPOCHS):
                classifier2.train()
                classifier3.train()
                epoch_acc = initialize_epoch_accumulators(device)
                epoch_loss2 = 0.0
                epoch_loss3 = 0.0
                epoch_correct2 = 0
                epoch_correct3 = 0
                epoch_samples = 0
                epoch_max_reconstruction = 0.0
                epoch_max_gradient_error = 0.0
                epoch_max_source_error = 0.0

                for batch_index, start in enumerate(range(0, len(context["train_labels"]), BATCH_SIZE)):
                    end = min(start + BATCH_SIZE, len(context["train_labels"]))
                    indices = np.asarray(plan[epoch, start:end], dtype=np.int64)
                    target_np = np.asarray(context["train_labels"][indices], dtype=np.int64)
                    labels = torch.from_numpy(target_np).to(device)
                    images_cpu = augment_batch(
                        context["train_images"], indices, context["train_transform"], augmentation_generator
                    )
                    with torch.no_grad():
                        gap = clean_compatible_gap(backbone(images_cpu.to(device, non_blocking=True)))
                    if gap.shape != (end - start, FEATURE_DIM) or not torch.isfinite(gap).all():
                        raise RuntimeError("Malformed/non-finite frozen-backbone GAP batch.")

                    trace_row = make_trace_row(epoch, batch_index, indices, target_np, images_cpu, gap)
                    check_trace_row(trace_row, reference_rows[global_batch], seed, epoch, batch_index)
                    trace_writer.writerow(trace_row)

                    pack = compute_paired_gradients(
                        classifier2, classifier3, gap, labels, class_weights, counts, epoch
                    )
                    source_pressure = project_source_gradients_to_fixed_margin(
                        pack["source_grad_weight"], pack["source_grad_bias"],
                        val_features_t, val_labels_t, fixed_q, many_mask, STAGE2_LR
                    )
                    direct_pressure = project_gradient_to_fixed_margin(
                        pack["direct_grad_weight"], pack["direct_grad_bias"],
                        val_features_t, val_labels_t, fixed_q, many_mask, STAGE2_LR
                    )
                    state_pressure = project_gradient_to_fixed_margin(
                        pack["state_grad_weight"], pack["state_grad_bias"],
                        val_features_t, val_labels_t, fixed_q, many_mask, STAGE2_LR
                    )
                    total_pressure = project_gradient_to_fixed_margin(
                        pack["total_grad_weight"], pack["total_grad_bias"],
                        val_features_t, val_labels_t, fixed_q, many_mask, STAGE2_LR
                    )
                    source_sum_error = float(
                        (source_pressure["margin_step"].sum(dim=0) - direct_pressure["margin_step"]).abs().max().item()
                    )
                    if source_sum_error > 5e-5:
                        raise RuntimeError(f"Source-class margin contributions are not additive: {source_sum_error}")
                    gradient_error = max(pack["checks"].values())
                    epoch_max_gradient_error = max(epoch_max_gradient_error, gradient_error)
                    epoch_max_source_error = max(epoch_max_source_error, source_sum_error)

                    accumulate_source_diagnostics(
                        epoch_acc, pack["source_grad_weight"], pack["source_grad_bias"],
                        labels, source_pressure, counts
                    )

                    before_dw = classifier3.module.fc.weight.detach() - classifier2.module.fc.weight.detach()
                    before_db = classifier3.module.fc.bias.detach() - classifier2.module.fc.bias.detach()
                    tracker.advance(
                        pack["source_grad_weight"], pack["source_grad_bias"],
                        pack["state_grad_weight"], pack["state_grad_bias"],
                        before_dw, before_db
                    )
                    writeback2.capture(classifier2.module.fc.named_parameters(), optimizer2)
                    writeback3.capture(classifier3.module.fc.named_parameters(), optimizer3)
                    optimizer2.step()
                    optimizer3.step()
                    writeback2.record()
                    writeback3.record()
                    reconstruction = reconstruction_diagnostics(
                        tracker, classifier2, classifier3, optimizer2, optimizer3,
                        writeback_ledgers=(writeback2, writeback3),
                    )
                    epoch_max_reconstruction = max(
                        epoch_max_reconstruction, reconstruction["max_relative_reconstruction_error"]
                    )

                    step_writer.writerow({
                        "seed": seed,
                        "epoch": epoch + 1,
                        "batch": batch_index,
                        "batch_size": len(indices),
                        "alpha": pack["alpha"],
                        "learning_rate": STAGE2_LR,
                        "c2_loss": pack["loss2_value"],
                        "c3_loss": pack["loss3_value"],
                        "mean_feature_delta_l2": float(torch.linalg.vector_norm(pack["delta3"], dim=1).mean().item()),
                        "direct_gradient_margin_step_mean": float(direct_pressure["margin_step"].mean().item()),
                        "direct_gradient_margin_step_median": float(direct_pressure["margin_step"].median().item()),
                        "direct_gradient_margin_step_negative_fraction": float((direct_pressure["margin_step"] < 0).float().mean().item()),
                        "direct_target_logit_step_mean": float(direct_pressure["target_logit_step"].mean().item()),
                        "direct_competitor_logit_step_mean": float(direct_pressure["competitor_logit_step"].mean().item()),
                        "state_feedback_margin_step_mean": float(state_pressure["margin_step"].mean().item()),
                        "state_feedback_margin_step_negative_fraction": float((state_pressure["margin_step"] < 0).float().mean().item()),
                        "total_gradient_margin_step_mean": float(total_pressure["margin_step"].mean().item()),
                        "total_gradient_margin_step_negative_fraction": float((total_pressure["margin_step"] < 0).float().mean().item()),
                        "direct_gradient_weight_norm": float(torch.linalg.vector_norm(pack["direct_grad_weight"]).item()),
                        "state_feedback_gradient_weight_norm": float(torch.linalg.vector_norm(pack["state_grad_weight"]).item()),
                        "total_gradient_weight_norm": float(torch.linalg.vector_norm(pack["total_grad_weight"]).item()),
                        "source_additivity_max_abs": source_sum_error,
                        "gradient_decomposition_max_abs": gradient_error,
                        "parameter_reconstruction_relative": reconstruction["parameter_max_relative"],
                        "momentum_buffer_reconstruction_relative": reconstruction["momentum_buffer_max_relative"],
                        "raw_parameter_weight_relative": reconstruction["raw_parameter_weight_relative"],
                        "raw_parameter_bias_relative": reconstruction["raw_parameter_bias_relative"],
                        "writeback_weight_delta_R_max_abs": reconstruction["parameter_writeback_rounding"]["weight"]["step_delta_R"]["max_abs"],
                        "writeback_bias_delta_R_max_abs": reconstruction["parameter_writeback_rounding"]["bias"]["step_delta_R"]["max_abs"],
                    })

                    batch_size = end - start
                    epoch_loss2 += pack["loss2_value"] * batch_size
                    epoch_loss3 += pack["loss3_value"] * batch_size
                    epoch_correct2 += int((pack["logits2"].argmax(1) == labels).sum().item())
                    epoch_correct3 += int((pack["logits3"].argmax(1) == labels).sum().item())
                    epoch_samples += batch_size
                    global_batch += 1

                if epoch_samples != len(context["train_labels"]):
                    raise RuntimeError(f"Epoch sample count mismatch: {epoch_samples}.")
                check_backbone_unchanged(backbone, frozen_backbone)

                _, _, diag2, diag3 = evaluate_epoch(
                    seed, epoch + 1, classifier2, classifier3, context, fixed_q, many_mask,
                    out_dir, metric_rows, per_class_rows
                )
                write_csv_rows(metrics_path, metric_fields, metric_rows)
                write_csv_rows(per_class_path, per_class_fields, per_class_rows)

                # Refresh aggregate files at epoch boundaries so partial runs retain a readable record.
                group_rows = source_group_gradient_rows(seed, epoch + 1, epoch_acc, counts)
                write_csv_rows(group_path, group_fields, group_rows, append=True)
                decomp_row, cumulative_source_rows = decomposition_epoch_rows(
                    seed, epoch + 1, tracker, classifier2, classifier3, context,
                    val_features_t, val_labels_t, fixed_q, many_mask, epoch_acc,
                    writeback_ledgers=(writeback2, writeback3),
                )
                write_csv_rows(source_path, source_fields, cumulative_source_rows, append=True)
                write_csv_rows(decomposition_path, decomposition_fields, [decomp_row], append=True)

                status["epochs_completed"] = epoch + 1
                status["paired_batch_steps_completed"] = global_batch
                status["last_epoch"] = {
                    "epoch": epoch + 1,
                    "c2_train_loss": epoch_loss2 / epoch_samples,
                    "c3_train_loss": epoch_loss3 / epoch_samples,
                    "c2_train_accuracy": 100.0 * epoch_correct2 / epoch_samples,
                    "c3_train_accuracy": 100.0 * epoch_correct3 / epoch_samples,
                    "c2_validation_many_accuracy": diag2["accuracy"]["Many"],
                    "c3_validation_many_accuracy": diag3["accuracy"]["Many"],
                    "c2_validation_few_accuracy": diag2["accuracy"]["Few"],
                    "c3_validation_few_accuracy": diag3["accuracy"]["Few"],
                    "c2_fixed_many_margin": diag2["fixed_margin_mean"],
                    "c3_fixed_many_margin": diag3["fixed_margin_mean"],
                    "actual_c3_minus_c2_fixed_margin": decomp_row["actual_c3_minus_c2_fixed_margin_mean"],
                    "direct_perturbation_margin_component": decomp_row["direct_perturbation_component_mean"],
                    "state_feedback_margin_component": decomp_row["state_feedback_component_mean"],
                    "weight_decay_margin_component": decomp_row["weight_decay_component_mean"],
                    "parameter_writeback_rounding_margin_component": decomp_row["parameter_writeback_rounding_component_mean"],
                    "max_gradient_decomposition_error": epoch_max_gradient_error,
                    "max_source_margin_additivity_error": epoch_max_source_error,
                    "max_sgd_reconstruction_relative": epoch_max_reconstruction,
                }
                write_json(status_path, status)
                trace_file.flush()
                step_file.flush()
                print(
                    f"CC-MIA seed={seed} epoch={epoch + 1:02d}/{STAGE2_EPOCHS} "
                    f"Many C2/C3={diag2['accuracy']['Many']:.3f}/{diag3['accuracy']['Many']:.3f} "
                    f"margin delta={decomp_row['actual_c3_minus_c2_fixed_margin_mean']:+.6f} "
                    f"direct={decomp_row['direct_perturbation_component_mean']:+.6f} "
                    f"state={decomp_row['state_feedback_component_mean']:+.6f}",
                    flush=True,
                )

        check_backbone_unchanged(backbone, frozen_backbone)
        if state_hash(backbone) != initial_backbone_sha:
            raise RuntimeError("Frozen backbone hash changed after paired formal training.")
        if global_batch != expected_steps:
            raise RuntimeError(f"Unexpected paired step count: {global_batch} != {expected_steps}")

        checkpoint2 = out_dir / "final_classifier_C2.pth"
        checkpoint3 = out_dir / "final_classifier_C3.pth"
        torch.save({
            "seed": int(seed), "arm": "C2", "epoch": STAGE2_EPOCHS,
            "state_dict_classifier": classifier2.state_dict(),
            "official_TEST_used": False,
            "initial_classifier_sha256": initial_classifier_sha,
            "final_classifier_sha256": state_hash(classifier2),
        }, checkpoint2)
        torch.save({
            "seed": int(seed), "arm": "C3", "epoch": STAGE2_EPOCHS,
            "state_dict_classifier": classifier3.state_dict(),
            "official_TEST_used": False,
            "initial_classifier_sha256": initial_classifier_sha,
            "final_classifier_sha256": state_hash(classifier3),
        }, checkpoint3)

        status.update({
            "state": "completed",
            "completed_at": __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds"),
            "initial_classifier_sha256": initial_classifier_sha,
            "final_c2_classifier_sha256": state_hash(classifier2),
            "final_c3_classifier_sha256": state_hash(classifier3),
            "final_backbone_sha256": state_hash(backbone),
            "backbone_and_bn_bitwise_unchanged": True,
            "paired_C0_trace_batches_matched": global_batch,
            "optimizer": {
                "type": "SGD", "lr": STAGE2_LR, "momentum": MOMENTUM,
                "weight_decay": WEIGHT_DECAY, "nesterov": False,
            },
            "official_TEST_used": False,
            "checkpoints": {
                "C2": {"path": checkpoint2.name, "sha256": sha256_file(checkpoint2)},
                "C3": {"path": checkpoint3.name, "sha256": sha256_file(checkpoint3)},
            },
        })
        write_json(status_path, status)
        return status
    except Exception as exc:
        status["state"] = "failed"
        status["failed_at"] = __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds")
        status["error"] = repr(exc)
        status["traceback"] = traceback.format_exc()
        write_json(status_path, status)
        raise


def run_formal(validation_dir=None):
    context = prepare_context()
    audit = validate_inputs(context)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is intentionally disabled.")
    report_root = validation_directory(validation_dir)
    preflight_path = report_root / "preflight_report.json"
    smoke_path = report_root / "smoke_report.json"
    if not preflight_path.is_file() or not smoke_path.is_file():
        raise RuntimeError("Formal run requires successful preflight and Smoke reports.")
    preflight = read_json(preflight_path)
    smoke = read_json(smoke_path)
    if preflight.get("passed") is not True or smoke.get("passed") is not True:
        raise RuntimeError("Preflight or Smoke did not pass.")
    if preflight.get("prepared_manifest_sha256") != audit["prepared_manifest_sha256"]:
        raise RuntimeError("Preflight uses different prepared inputs.")
    if smoke.get("prepared_manifest_sha256") != audit["prepared_manifest_sha256"]:
        raise RuntimeError("Smoke uses different prepared inputs.")
    if preflight.get("code_sha256") != audit["code_sha256"] or smoke.get("code_sha256") != audit["code_sha256"]:
        raise RuntimeError("Code changed after preflight/Smoke; rerun both gates.")
    if (CCMIA_ROOT / "formal_completed.json").exists() or (CCMIA_ROOT / "results").exists():
        raise FileExistsError("Formal outputs already exist; refusing overwrite, resume or silent restart.")

    device = torch.device("cuda:0")
    results = []
    for seed in SEEDS:
        results.append(train_seed(seed, context, device))
    guard = require_test_guard()
    completion = {
        "state": "completed",
        "completed_at": __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset": "CIFAR-100-LT IF=100",
        "seeds": list(SEEDS),
        "paired_arms": ["C2_WEIGHTED_CE_NO_PERTURB", "C3_NO_MANY_PERTURB"],
        "epochs": STAGE2_EPOCHS,
        "paired_steps": sum(int(item["paired_batch_steps_completed"]) for item in results),
        "official_TEST_used": False,
        "test_guard": guard,
        "prepared_manifest_sha256": audit["prepared_manifest_sha256"],
        "code_sha256": audit["code_sha256"],
        "seed_status": results,
    }
    write_json(CCMIA_ROOT / "formal_completed.json", completion)
    print("CC-MIA FORMAL TRAINING COMPLETED: paired C2/C3, 3 seeds x 30 epochs.")
    return completion


def main():
    parser = argparse.ArgumentParser(
        description="CC-MIA: paired C2/C3 source-conditioned cross-class margin influence audit."
    )
    parser.add_argument("--phase", choices=("preflight", "smoke", "formal"), required=True)
    parser.add_argument(
        "--validation-dir", type=Path,
        help="Use independent preflight/Smoke reports under ccmia/diagnostics/.",
    )
    args = parser.parse_args()
    if args.phase == "preflight":
        run_preflight(args.validation_dir)
    elif args.phase == "smoke":
        run_smoke(args.validation_dir)
    else:
        run_formal(args.validation_dir)


if __name__ == "__main__":
    main()
