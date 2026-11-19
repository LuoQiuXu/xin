"""Finite CC-MIA causal/oracle experiment; each phase is an exclusive run."""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import sys
import traceback

import numpy as np
import torch

from causal_core import (
    ALIASES, ARMS, ROOT, array_hash, balance_diagnostics, calibration_competitors,
    canonical_hash, construct_masks, eligible_pairs, fixed_k, keep_mask,
)
from isolated_data import file_sha, seal_masks
from oracle_runtime import (
    BASE, BRANCH, PARENT, SEEDS, BATCH_SIZE, FEATURE_DIM, MOMENTUM, NUM_CLASSES,
    NBATCH, NSTEPS, NTRAIN, STAGE2_EPOCHS, STAGE2_LR, WEIGHT_DECAY,
    ParameterWritebackRoundingLedger, SGDInfluenceTracker, check_backbone_unchanged,
    class_weights_from_counts, code_hashes, compute_paired_gradients, cuda_device,
    environment, git, guard_pass, inspect_parent, intervention_step, load_context,
    make_models, new_csv, new_json, now, optimizer_for, read_csv, read_json,
    real_batches, reconstruction_diagnostics, require_cpu, state_hash,
    validate_class_aware_plan,
)
from ccmia_core import project_source_parameter_delta_to_fixed_margin
from train_ccsam import verify_analytic_gradient


def require_preflight():
    require_cpu()
    report = read_json(ROOT / "preflight_report.json")
    if report.get("passed") is not True or report.get("code_sha256") != code_hashes():
        raise RuntimeError("Preflight missing, failed, or stale")
    if git("branch", "--show-current") != BRANCH:
        raise RuntimeError("Wrong branch")
    for filename, expected in report["immutable_inputs"].items():
        if file_sha(filename) != expected:
            raise RuntimeError(f"Immutable input changed after preflight: {filename}")
    return report


def run_preflight():
    require_cpu()
    if (ROOT / "preflight_report.json").exists():
        raise FileExistsError("Existing preflight must not be overwritten")
    audit = inspect_parent()
    context = load_context()
    inputs = dict(audit["reused_source_and_artifact_sha256"])
    for path in (context["manifest_path"], context["checkpoint"],
                 Path(context["manifest"]["split_path"]),
                 context["prepared"] / "train_images.npy", context["prepared"] / "train_labels.npy"):
        inputs[str(path)] = file_sha(path)
    plans = {}
    for seed in SEEDS:
        path = context["prepared"] / f"sampling_plan_seed{seed}.npy"
        plan = np.load(path, allow_pickle=False)
        if plan.shape != (30, NTRAIN) or plan.dtype != np.int32:
            raise RuntimeError("Sampling plan shape / dtype changed")
        record = validate_class_aware_plan(context["train_labels"], seed, plan)
        plans[str(seed)] = record
        inputs[str(path)] = file_sha(path)
        c0 = BASE / "results" / f"seed_{seed}" / "C0_BALANCED_CE"
        rows = read_csv(c0 / "batch_trace.csv")
        if len(rows) != NSTEPS or read_json(c0 / "status.json").get("state") != "completed":
            raise RuntimeError("Original C0 trace incomplete")
        for filename in ("batch_trace.csv", "status.json"):
            inputs[str(c0 / filename)] = file_sha(c0 / filename)
    pairs = eligible_pairs(context["counts"], np.unique(context["cal_y"]))
    if len(pairs) != 2210 or fixed_k(len(pairs)) != 221:
        raise RuntimeError("Unexpected eligible family / K")
    new_json(ROOT / "split_manifest.json", context["split"])
    new_json(ROOT / "source_audit.json", audit)
    new_json(ROOT / "sampling_audit.json", plans)
    for filename in ("split_manifest.json", "source_audit.json", "sampling_audit.json"):
        inputs[str(ROOT / filename)] = file_sha(ROOT / filename)
    report = {
        "passed": True, "completed_at": now(), "branch": BRANCH,
        "parent_commit": audit["parent_commit"], "head": git("rev-parse", "HEAD"),
        "code_sha256": code_hashes(), "immutable_inputs": inputs,
        "prepared_manifest_sha256": file_sha(context["manifest_path"]),
        "clean_stage1_sha256": file_sha(context["checkpoint"]),
        "calibration_features_sha256": array_hash(context["cal_x"]),
        "calibration_labels_sha256": array_hash(context["cal_y"]),
        "split_sha256": canonical_hash(context["split"]),
        "calibration_examples": 990, "evaluation_examples": 990,
        "evaluation_labels_features_or_metrics_read": False,
        "deferred_until_mask_seal": ["Evaluation label validation", "complete holdout array SHA256"],
        "eligible_pairs": len(pairs), "K": fixed_k(len(pairs)),
        "reference_scope": "complete parent BASE_C3 trajectory, 30 epochs per seed",
        "AMP": False, "formal_training_started": False,
        "test_guard": guard_pass(), "holdout_access_log": context["vault"].events,
        "environment": environment(), "official_TEST_used": False,
    }
    new_json(ROOT / "preflight_report.json", report)
    print("PREFLIGHT PASS: parent, TRAIN, all plans, isolated Calibration; Evaluation remains locked", flush=True)
    return report


def _group(count):
    return "Many" if count > 100 else "Medium" if count >= 20 else "Few"


def _competitor_tensors(context, head, device):
    x = torch.from_numpy(context["cal_x"]).to(device)
    y = torch.from_numpy(context["cal_y"]).to(device)
    with torch.no_grad():
        logits = head.module.fc(x)
    mapping = calibration_competitors(logits.cpu().numpy(), context["cal_y"], context["counts"])
    many = torch.as_tensor(context["counts"], device=device)[y] > 100
    q = torch.as_tensor([mapping.get(int(label), 0) for label in context["cal_y"]], device=device)
    if int(many.sum()) != 340:
        raise RuntimeError("Expected 340 Calibration Many examples")
    return mapping, x, y, q, many


def _source_score_rows(seed, epoch, source_w, source_b, x, y, q, many, counts):
    projection = project_source_parameter_delta_to_fixed_margin(
        source_w, source_b, x, y, q, many)
    targets = projection["target_labels"].detach().cpu().numpy()
    values = {k: projection[k].detach().cpu().numpy() for k in
              ("target_logit_delta", "competitor_logit_delta", "margin_delta")}
    rows = []
    for source, target in eligible_pairs(counts, np.unique(targets)):
        selected = targets == target
        rows.append({
            "seed": seed, "epoch": epoch, "source_class": source,
            "source_train_count": int(counts[source]), "source_group": _group(counts[source]),
            "target_many_class": target, "target_train_count": int(counts[target]),
            "target_group": _group(counts[target]), "calibration_target_examples": int(selected.sum()),
            "row_local_direct_fixed_margin_delta": float(values["target_logit_delta"][source, selected].mean()),
            "full_source_fixed_margin_delta": float(values["margin_delta"][source, selected].mean()),
            "unmodified_competitor_logit_delta": float(values["competitor_logit_delta"][source, selected].mean()),
        })
    return rows


def _baseline_probe(context, device, destination):
    records = []
    for seed in SEEDS:
        backbone, head3, frozen = make_models(context, seed, device)
        head2 = copy.deepcopy(head3)
        opt2, opt3 = optimizer_for(head2), optimizer_for(head3)
        ledger2, ledger3 = ParameterWritebackRoundingLedger(), ParameterWritebackRoundingLedger()
        weights = class_weights_from_counts(context["counts"], device)
        iterator = real_batches(context, seed, backbone, device, epochs=1)
        epoch, batch, _, _, gap, labels, trace = next(iterator)
        analytic = verify_analytic_gradient(gap, labels, head3.module.fc, weights)
        pack = compute_paired_gradients(head2, head3, gap, labels, weights, context["counts"], 0)
        tracker = SGDInfluenceTracker(100, 64, device, MOMENTUM, STAGE2_LR, WEIGHT_DECAY)
        tracker.advance(pack["source_grad_weight"], pack["source_grad_bias"],
                        pack["state_grad_weight"], pack["state_grad_bias"],
                        head3.module.fc.weight.detach() - head2.module.fc.weight.detach(),
                        head3.module.fc.bias.detach() - head2.module.fc.bias.detach())
        ledger2.capture(head2.module.fc.named_parameters(), opt2)
        ledger3.capture(head3.module.fc.named_parameters(), opt3)
        opt2.step()
        opt3.step()
        ledger2.record()
        ledger3.record()
        reconstruction = reconstruction_diagnostics(tracker, head2, head3, opt2, opt3,
                                                     writeback_ledgers=(ledger2, ledger3))
        check_backbone_unchanged(backbone, frozen)
        records.append({"seed": seed, "real_batches": 1, "trace": trace,
                        "original_feature_gradient_check": analytic,
                        "gradient_checks": pack["checks"], "reconstruction": reconstruction,
                        "backbone_bn_bitwise_unchanged": True})
    report = {"passed": True, "purpose": "real baseline parity check before Calibration reference replay",
              "seeds": records, "formal_intervention_training_started": False,
              "evaluation_read": False, "test_guard": guard_pass()}
    new_json(destination / "baseline_probe.json", report)


def _calibrate_seed(context, seed, device, destination):
    folder = destination / f"seed_{seed}"
    folder.mkdir(exist_ok=False)
    cache_dir = ROOT / "cache" / f"seed_{seed}"
    cache_dir.mkdir(parents=True, exist_ok=False)
    cache_path = cache_dir / "paired_train_gap.npy"
    if cache_path.exists():
        raise FileExistsError(cache_path)
    cache = np.lib.format.open_memmap(cache_path, mode="w+", dtype=np.float32, shape=(30, NTRAIN, 64))
    backbone, head3, frozen = make_models(context, seed, device)
    head2 = copy.deepcopy(head3)
    opt2, opt3 = optimizer_for(head2), optimizer_for(head3)
    ledger2, ledger3 = ParameterWritebackRoundingLedger(), ParameterWritebackRoundingLedger()
    weights = class_weights_from_counts(context["counts"], device)
    tracker = SGDInfluenceTracker(100, 64, device, MOMENTUM, STAGE2_LR, WEIGHT_DECAY)
    mapping, cal_x, cal_y, q, many = _competitor_tensors(context, head3, device)
    new_json(folder / "competitors.json", {str(k): v for k, v in mapping.items()})
    influence_path = folder / "influence_epochs.csv"
    last_rows = None
    max_gradient, max_reconstruction = 0.0, 0.0
    trace_rows = []
    steps = 0
    with influence_path.open("x", encoding="utf-8", newline="") as handle:
        writer = None
        for epoch, batch, start, stop, gap, labels, trace in real_batches(context, seed, backbone, device):
            pack = compute_paired_gradients(head2, head3, gap, labels, weights, context["counts"], epoch)
            tracker.advance(pack["source_grad_weight"], pack["source_grad_bias"],
                            pack["state_grad_weight"], pack["state_grad_bias"],
                            head3.module.fc.weight.detach() - head2.module.fc.weight.detach(),
                            head3.module.fc.bias.detach() - head2.module.fc.bias.detach())
            ledger2.capture(head2.module.fc.named_parameters(), opt2)
            ledger3.capture(head3.module.fc.named_parameters(), opt3)
            opt2.step()
            opt3.step()
            ledger2.record()
            ledger3.record()
            reconstruction = reconstruction_diagnostics(tracker, head2, head3, opt2, opt3,
                                                         writeback_ledgers=(ledger2, ledger3))
            max_gradient = max(max_gradient, *pack["checks"].values())
            max_reconstruction = max(max_reconstruction, reconstruction["max_relative_reconstruction_error"])
            cache[epoch, start:stop] = gap.detach().cpu().numpy()
            trace_rows.append(trace)
            steps += 1
            if batch == NBATCH - 1:
                check_backbone_unchanged(backbone, frozen)
                last_rows = _source_score_rows(seed, epoch + 1, tracker.direct_parameter_weight,
                                               tracker.direct_parameter_bias, cal_x, cal_y, q, many, context["counts"])
                if writer is None:
                    writer = csv.DictWriter(handle, fieldnames=list(last_rows[0]))
                    writer.writeheader()
                writer.writerows(last_rows)
                handle.flush()
                cache.flush()
                print(f"CALIBRATION seed={seed} epoch={epoch + 1}/30 steps={steps} "
                      f"gradient_max={max_gradient:.3g} reconstruction_rel={max_reconstruction:.3g}", flush=True)
    if steps != NSTEPS:
        raise RuntimeError("Calibration reference replay incomplete")
    parent_status = read_json(PARENT / "results" / f"seed_{seed}" / "status.json")
    states = {"C2": state_hash(head2), "C3": state_hash(head3)}
    for arm, actual in states.items():
        if actual != parent_status[f"final_{arm.lower()}_classifier_sha256"]:
            raise RuntimeError(f"Calibration replay final {arm} does not exactly match parent: {actual}")
    del cache
    new_csv(cache_dir / "batch_trace.csv", trace_rows)
    components = folder / "source_components_epoch30.npz"
    if components.exists():
        raise FileExistsError(components)
    np.savez_compressed(components,
                        weight=tracker.direct_parameter_weight.detach().cpu().numpy(),
                        bias=tracker.direct_parameter_bias.detach().cpu().numpy())
    new_csv(folder / "pair_scores_epoch30.csv", last_rows)
    pairs = eligible_pairs(context["counts"], np.unique(context["cal_y"]))
    scores = {(row["source_class"], row["target_many_class"]): row["row_local_direct_fixed_margin_delta"]
              for row in last_rows}
    bundle = construct_masks(scores, pairs, context["counts"], seed)
    new_json(folder / "mask_bundle.json", bundle)
    balance = balance_diagnostics(bundle, context["counts"])
    new_json(folder / "matched_control_balance.json", balance)
    masks = {a: set(map(tuple, ps)) for a, ps in bundle["suppressed_pairs"].items()}
    mask_rows = []
    for source, target in pairs:
        row = {"source_class": source, "source_train_count": int(context["counts"][source]),
               "target_many_class": target, "target_train_count": int(context["counts"][target]),
               "calibration_score": scores[(source, target)]}
        row.update({arm + "_suppressed": int((source, target) in selected) for arm, selected in masks.items()})
        row["ORACLE_SUPPRESSION_suppressed"] = row["TARGETED_suppressed"]
        mask_rows.append(row)
    new_csv(folder / "mask_table.csv", mask_rows)
    report = {
        "passed": True, "seed": seed, "epochs_replayed": 30, "paired_steps_replayed": steps,
        "actual_optimizer_steps": {"C2": ledger2.steps, "BASE_C3": ledger3.steps},
        "c0_trace_batches_matched": len(trace_rows), "final_classifier_sha256": states,
        "final_states_bitwise_match_parent": True, "backbone_bn_bitwise_unchanged": True,
        "calibration_only": True, "calibration_features_sha256": array_hash(context["cal_x"]),
        "calibration_labels_sha256": array_hash(context["cal_y"]),
        "influence_epoch_rows": 30 * len(pairs), "influence_epochs_sha256": file_sha(influence_path),
        "source_target_frequency_checked": True, "eligible_pairs": len(pairs), "K": bundle["K"],
        "mask_bundle_sha256": file_sha(folder / "mask_bundle.json"),
        "oracle_suppression_alias": "TARGETED", "cache_sha256": file_sha(cache_path),
        "cache_trace_sha256": file_sha(cache_dir / "batch_trace.csv"),
        "max_gradient_decomposition_abs": max_gradient,
        "max_sgd_reconstruction_relative": max_reconstruction,
    }
    new_json(folder / "calibration_report.json", report)
    return report


def run_calibration():
    preflight = require_preflight()
    context = load_context(read_json(ROOT / "split_manifest.json"))
    device = cuda_device()
    destination = ROOT / "calibration"
    destination.mkdir(exist_ok=False)
    _baseline_probe(context, device, destination)
    records = [_calibrate_seed(context, seed, device, destination) for seed in SEEDS]
    dependencies = dict(preflight["immutable_inputs"])
    for name in code_hashes():
        dependencies[str(ROOT / name)] = file_sha(ROOT / name)
    for path in sorted(destination.rglob("*")):
        if path.is_file():
            dependencies[str(path)] = file_sha(path)
    for path in sorted((ROOT / "cache").rglob("*")):
        if path.is_file():
            dependencies[str(path)] = file_sha(path)
    events = context["vault"].events
    if any(event["partition"] != "calibration" for event in events):
        raise RuntimeError("Unexpected pre-seal holdout access")
    payload = {
        "sealed_at": now(), "seeds": list(SEEDS), "split_sha256": canonical_hash(context["split"]),
        "dependencies": dependencies, "evaluation_unopened_before_seal": True,
        "calibration_only": True, "eligible_pairs": 2210, "K": 221,
        "oracle_suppression_alias": ALIASES,
        "practical_effect_flags": {"many_gain_pp": 2.0, "fixed_margin_gain": 0.10,
                                   "few_change_floor_pp": -2.0, "consistent_seeds": 2},
        "gate_admission": "HUMAN_REVIEW_ONLY",
        "holdout_access_log_before_seal": events,
    }
    seal = seal_masks(ROOT / "mask_seal.json", payload)
    new_json(ROOT / "calibration_completed.json", {
        "passed": True, "seeds": records, "mask_seal_sha256": seal["payload_sha256"],
        "code_sha256": code_hashes(), "official_TEST_used": False, "test_guard": guard_pass(),
        "formal_intervention_training_started": False, "environment": environment(),
    })
    print("CALIBRATION PASS: all 30-epoch replay states match parent; all masks SEALED; no Evaluation read", flush=True)


def _load_masks(seed, device):
    bundle = read_json(ROOT / "calibration" / f"seed_{seed}" / "mask_bundle.json")
    return bundle, {arm: keep_mask(100, 100, suppressed, device)
                    for arm, suppressed in bundle["suppressed_pairs"].items()}


def _verify_postseal_data(context):
    context["vault"].verify_seal()
    for name in ("validation_gap_features.npy", "validation_labels.npy", "validation_images.npy"):
        actual = context["vault"].sealed_integrity_hash(name)
        if actual != context["manifest"]["prepared_array_actions"][name]["sha256"]:
            raise RuntimeError(f"Sealed holdout cache SHA mismatch after mask sealing: {name}")
    x, y = context["vault"].read("evaluation")
    if x.shape != (990, 64) or not np.array_equal(np.bincount(y, minlength=100), np.r_[0, np.repeat(10, 99)]):
        raise RuntimeError("Evaluation size / label allocation mismatch")
    return x, y


def run_smoke():
    preflight = require_preflight()
    if (ROOT / "smoke_report.json").exists():
        raise FileExistsError("Refusing to overwrite Smoke")
    completion = read_json(ROOT / "calibration_completed.json")
    if completion.get("passed") is not True or completion.get("code_sha256") != code_hashes():
        raise RuntimeError("Complete sealed Calibration is required before production Smoke")
    context = load_context(read_json(ROOT / "split_manifest.json"))
    context["vault"].verify_seal()
    _verify_postseal_data(context)
    device = cuda_device()
    records = []
    for seed in SEEDS:
        backbone, initial, frozen = make_models(context, seed, device)
        heads = {arm: copy.deepcopy(initial) for arm in ARMS}
        optimizers = {arm: optimizer_for(head) for arm, head in heads.items()}
        ledgers = {arm: ParameterWritebackRoundingLedger() for arm in ARMS}
        bundle, masks = _load_masks(seed, device)
        weights = class_weights_from_counts(context["counts"], device)
        cache = np.load(ROOT / "cache" / f"seed_{seed}" / "paired_train_gap.npy", mmap_mode="r")
        iterator = real_batches(context, seed, backbone, device, epochs=1)
        epoch, batch, start, stop, gap, labels, trace = next(iterator)
        if array_hash(gap.detach().cpu().numpy()) != array_hash(np.asarray(cache[0, start:stop])):
            raise RuntimeError("Fresh real-data Smoke differs from sealed Calibration feature cache")
        # Independent, original parent baseline on the same current state.
        reference2, reference3 = copy.deepcopy(initial), copy.deepcopy(initial)
        original = compute_paired_gradients(reference2, reference3, gap, labels, weights, context["counts"], 0)
        opt_ref2, opt_ref3 = optimizer_for(reference2), optimizer_for(reference3)
        opt_ref2.step()
        opt_ref3.step()
        arm_reports = {}
        for arm in ARMS:
            keep = masks.get(arm, keep_mask(100, 100, [], device))
            check = intervention_step(heads[arm], arm, gap, labels, weights, context["counts"], 0,
                                      keep, optimizers[arm], ledgers[arm])
            if check["updated_parameter_tensors"] != 2:
                raise RuntimeError(f"Smoke {arm} did not actually update weight and bias")
            arm_reports[arm] = {**check, "optimizer_steps": ledgers[arm].steps,
                                "trace_sha256": canonical_hash(trace)}
        if state_hash(heads["C2"]) != state_hash(reference2) or state_hash(heads["BASE_C3"]) != state_hash(reference3):
            raise RuntimeError("BASE_C3/C2 production path does not match parent autograd update bitwise")
        check_backbone_unchanged(backbone, frozen)
        records.append({"seed": seed, "real_batches_per_arm": 1, "arms": arm_reports,
                        "K": bundle["K"], "trace": trace, "parent_baseline_bitwise_parity": True,
                        "backbone_bn_bitwise_unchanged": True,
                        "original_gradient_checks": original["checks"]})
        print(f"SMOKE seed={seed} PASS all six actual arms, C0 trace and parent baseline exact", flush=True)
    report = {"passed": True, "completed_at": now(), "code_sha256": code_hashes(),
              "prepared_manifest_sha256": preflight["prepared_manifest_sha256"],
              "mask_seal_sha256": file_sha(ROOT / "mask_seal.json"),
              "seeds": records, "aliases": ALIASES, "test_guard": guard_pass(),
              "holdout_access_log": context["vault"].events, "official_TEST_used": False,
              "environment": environment(), "formal_training_started": False}
    new_json(ROOT / "smoke_report.json", report)
    print("PRODUCTION SMOKE PASS: formal is now eligible; no formal output created by Smoke", flush=True)


def metrics(logits, labels, counts, competitors):
    logits = np.asarray(logits)
    labels = np.asarray(labels, dtype=np.int64)
    if logits.shape != (len(labels), 100) or not np.isfinite(logits).all():
        raise RuntimeError("Nonfinite or malformed metric logits")
    correct = logits.argmax(1) == labels
    many, few = counts[labels] > 100, counts[labels] < 20
    q = np.array([competitors.get(str(int(y)), 0) for y in labels], dtype=np.int64)
    margin = logits[np.arange(len(labels)), labels] - logits[np.arange(len(labels)), q]
    per_class = []
    for cls in sorted(set(labels[many])):
        selected = labels == cls
        per_class.append({"class_id": int(cls), "n": int(selected.sum()),
                          "accuracy": float(correct[selected].mean() * 100),
                          "fixed_margin": float(margin[selected].mean())})
    values = {"overall_accuracy": float(correct.mean() * 100),
              "macro_accuracy": float(np.mean([correct[labels == c].mean() for c in np.unique(labels)]) * 100),
              "many_accuracy": float(correct[many].mean() * 100),
              "few_accuracy": float(correct[few].mean() * 100),
              "fixed_many_margin": float(margin[many].mean())}
    return values, per_class


def _logits(head, features, device):
    with torch.no_grad():
        chunks = [head.module.fc(torch.from_numpy(features[i:i + 512].copy()).to(device)).cpu().numpy()
                  for i in range(0, len(features), 512)]
    result = np.concatenate(chunks)
    if not np.isfinite(result).all():
        raise RuntimeError("Nonfinite evaluation logits")
    return result


def _formal_seed(context, seed, device, destination, eval_x, eval_y):
    folder = destination / f"seed_{seed}"
    folder.mkdir(exist_ok=False)
    backbone, initial, frozen = make_models(context, seed, device)
    heads = {arm: copy.deepcopy(initial) for arm in ARMS}
    optimizers = {arm: optimizer_for(head) for arm, head in heads.items()}
    ledgers = {arm: ParameterWritebackRoundingLedger() for arm in ARMS}
    bundle, masks = _load_masks(seed, device)
    weights = class_weights_from_counts(context["counts"], device)
    cache_path = ROOT / "cache" / f"seed_{seed}" / "paired_train_gap.npy"
    cache = np.load(cache_path, mmap_mode="r", allow_pickle=False)
    plan = np.load(context["prepared"] / f"sampling_plan_seed{seed}.npy", mmap_mode="r", allow_pickle=False)
    traces = read_csv(ROOT / "cache" / f"seed_{seed}" / "batch_trace.csv")
    c0_traces = read_csv(BASE / "results" / f"seed_{seed}" / "C0_BALANCED_CE" / "batch_trace.csv")
    if len(traces) != NSTEPS or traces != c0_traces:
        raise RuntimeError("Sealed feature-cache input trace is not the complete original C0 trace")
    competitors = read_json(ROOT / "calibration" / f"seed_{seed}" / "competitors.json")
    metric_rows, per_class_rows, final_logits = [], [], {}
    max_errors = {arm: 0.0 for arm in ARMS}
    start_at = now()
    with (folder / "batch_trace.csv").open("x", encoding="utf-8", newline="") as trace_handle, \
         (folder / "gradient_trace.csv").open("x", encoding="utf-8", newline="") as gradient_handle, \
         (folder / "execution.jsonl").open("x", encoding="utf-8", newline="\n") as journal:
        trace_writer = csv.DictWriter(trace_handle, fieldnames=list(traces[0]))
        trace_writer.writeheader()
        gradient_writer = csv.DictWriter(gradient_handle, fieldnames=["seed", "epoch", "batch", "arm", "loss",
            "unmasked_weight_max_abs", "unmasked_bias_max_abs", "masked_weight_max_abs", "masked_bias_max_abs",
            "weight_update_max_abs", "bias_update_max_abs", "writeback_replay_bitwise_equal"])
        gradient_writer.writeheader()
        for epoch in range(30):
            losses, correct, n = {a: 0.0 for a in ARMS}, {a: 0 for a in ARMS}, 0
            for batch in range(NBATCH):
                start, stop = batch * BATCH_SIZE, min((batch + 1) * BATCH_SIZE, NTRAIN)
                indices = np.asarray(plan[epoch, start:stop], dtype=np.int64)
                labels_np = np.asarray(context["train_labels"][indices], dtype=np.int64)
                labels = torch.from_numpy(labels_np).to(device)
                gap_np = np.asarray(cache[epoch, start:stop])
                gap = torch.from_numpy(gap_np.copy()).to(device)
                expected = traces[epoch * NBATCH + batch]
                if (array_hash(gap_np) != expected["gap_features_sha256"]
                        or _indices_labels_hash(indices, labels_np) != expected["indices_labels_sha256"]):
                    raise RuntimeError("Formal cache / index-label hash mismatch")
                trace_writer.writerow(expected)
                for arm in ARMS:
                    keep = masks.get(arm, keep_mask(100, 100, [], device))
                    check = intervention_step(heads[arm], arm, gap, labels, weights, context["counts"], epoch,
                                              keep, optimizers[arm], ledgers[arm])
                    max_errors[arm] = max(max_errors[arm], *check["gradient_checks"].values())
                    losses[arm] += check["loss"] * len(labels_np)
                    correct[arm] += check["correct"]
                    gradient_writer.writerow({"seed": seed, "epoch": epoch + 1, "batch": batch, "arm": arm,
                        "loss": check["loss"], **check["gradient_checks"],
                        "weight_update_max_abs": check["parameter_update_max_abs"]["weight"],
                        "bias_update_max_abs": check["parameter_update_max_abs"]["bias"],
                        "writeback_replay_bitwise_equal": True})
                n += len(labels_np)
            if n != NTRAIN:
                raise RuntimeError("Wrong formal epoch length")
            check_backbone_unchanged(backbone, frozen)
            for arm in ARMS:
                for partition, x, y in (("calibration", context["cal_x"], context["cal_y"]),
                                        ("evaluation", eval_x, eval_y)):
                    logits = _logits(heads[arm], x, device)
                    values, pc = metrics(logits, y, context["counts"], competitors)
                    row = {"seed": seed, "epoch": epoch + 1, "arm": arm, "partition": partition,
                           "train_loss": losses[arm] / n, "train_accuracy": correct[arm] / n * 100, **values}
                    metric_rows.append(row)
                    if partition == "evaluation":
                        per_class_rows.extend({"seed": seed, "epoch": epoch + 1, "arm": arm, **p} for p in pc)
                        if epoch == 29:
                            final_logits[arm] = logits
                if ledgers[arm].steps != (epoch + 1) * NBATCH:
                    raise RuntimeError("Optimizer updates differ across paired arms")
            record = {"seed": seed, "epoch_completed": epoch + 1, "steps_per_arm": (epoch + 1) * NBATCH,
                      "time": now(), "official_TEST_used": False}
            journal.write(json.dumps(record) + "\n")
            journal.flush()
            trace_handle.flush()
            gradient_handle.flush()
            # Write each epoch once, preserving partial-run evidence on failure.
            new_csv(folder / f"epoch_{epoch + 1:02d}_metrics.csv", metric_rows[-2 * len(ARMS):])
            print(f"FORMAL seed={seed} epoch={epoch + 1}/30 actual_steps_per_arm={(epoch + 1) * NBATCH} "
                  f"max_gradient_error={max(max_errors.values()):.3g}", flush=True)
    parent_status = read_json(PARENT / "results" / f"seed_{seed}" / "status.json")
    for arm, parent_arm in (("C2", "c2"), ("BASE_C3", "c3")):
        if state_hash(heads[arm]) != parent_status[f"final_{parent_arm}_classifier_sha256"]:
            raise RuntimeError(f"Formal {arm} final state differs from original parent")
    new_csv(folder / "epoch_metrics.csv", metric_rows)
    new_csv(folder / "per_class_many_epoch_metrics.csv", per_class_rows)
    checkpoints = {}
    for arm in ARMS:
        path = folder / f"final_classifier_{arm}.pth"
        if path.exists():
            raise FileExistsError(path)
        torch.save({"state_dict_classifier": heads[arm].state_dict(), "seed": seed, "arm": arm, "epoch": 30,
                    "official_TEST_used": False, "initial_classifier_sha256": state_hash(initial),
                    "final_classifier_sha256": state_hash(heads[arm]),
                    "mask_seal_sha256": file_sha(ROOT / "mask_seal.json")}, path)
        checkpoints[arm] = {"path": str(path.relative_to(ROOT)), "sha256": file_sha(path),
                            "classifier_state_sha256": state_hash(heads[arm]), "bytes": path.stat().st_size}
    logits_path = folder / "final_logits.npz"
    if logits_path.exists():
        raise FileExistsError(logits_path)
    np.savez_compressed(logits_path, **final_logits, evaluation_labels=eval_y)
    report = {"state": "completed", "seed": seed, "started_at": start_at, "completed_at": now(),
              "epochs_completed": 30, "paired_batches": NSTEPS,
              "actual_optimizer_steps": {a: ledgers[a].steps for a in ARMS}, "checkpoints": checkpoints,
              "C2_BASE_C3_final_states_match_parent": True, "paired_trace_identity": True,
              "cache_sha256": file_sha(cache_path), "aliases": ALIASES,
              "final_logits_sha256": file_sha(logits_path), "max_gradient_errors": max_errors,
              "backbone_bn_bitwise_unchanged": True, "official_TEST_used": False,
              "test_guard": guard_pass(), "mask_seal_sha256": file_sha(ROOT / "mask_seal.json")}
    new_json(folder / "status.json", report)
    return report


def _indices_labels_hash(indices, labels):
    # Exact parent hash_arrays convention, including both shapes and dtypes.
    from ccsam_common import hash_arrays
    return hash_arrays(indices, labels)


def run_formal():
    require_preflight()
    smoke = read_json(ROOT / "smoke_report.json")
    if (smoke.get("passed") is not True or smoke.get("code_sha256") != code_hashes()
            or smoke.get("mask_seal_sha256") != file_sha(ROOT / "mask_seal.json")
            or smoke.get("formal_training_started") is not False):
        raise RuntimeError("Production Smoke admission gate failed")
    context = load_context(read_json(ROOT / "split_manifest.json"))
    eval_x, eval_y = _verify_postseal_data(context)
    device = cuda_device()
    destination = ROOT / "results"
    destination.mkdir(exist_ok=False)  # no resume / overwrite
    new_json(destination / "FORMAL_STARTED.json", {"started_at": now(), "code_sha256": code_hashes(),
        "smoke_sha256": file_sha(ROOT / "smoke_report.json"), "mask_seal_sha256": file_sha(ROOT / "mask_seal.json"),
        "seeds": list(SEEDS), "arms": list(ARMS), "epochs": 30, "official_TEST_used": False})
    statuses = [_formal_seed(context, seed, device, destination, eval_x, eval_y) for seed in SEEDS]
    complete = {"state": "completed", "completed_at": now(), "seeds": list(SEEDS), "epochs": 30,
                "actual_arms": list(ARMS), "aliases": ALIASES, "paired_batches": 15300,
                "actual_optimizer_steps": 3 * len(ARMS) * NSTEPS,
                "seed_status": statuses, "code_sha256": code_hashes(),
                "mask_seal_sha256": file_sha(ROOT / "mask_seal.json"),
                "official_TEST_used": False, "test_guard": guard_pass(), "environment": environment()}
    new_json(ROOT / "formal_completed.json", complete)
    print("FORMAL COMPLETE: 3 seeds x 30 epochs, six paired actual arms; Oracle Suppression aliases TARGETED", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("preflight", "calibrate", "smoke", "formal", "analyze"))
    args = parser.parse_args()
    try:
        if args.phase == "analyze":
            from analyze_causal_oracle import run_analysis
            run_analysis()
        else:
            {"preflight": run_preflight, "calibrate": run_calibration,
             "smoke": run_smoke, "formal": run_formal}[args.phase]()
    except BaseException as error:
        failure = ROOT / "validation" / (args.phase + "_FAILURE.json")
        if not failure.exists():
            new_json(failure, {"phase": args.phase, "failed_at": now(), "exception": repr(error),
                "traceback": traceback.format_exc(), "formal_completed": (ROOT / "formal_completed.json").exists(),
                "official_TEST_used": False})
        raise


if __name__ == "__main__":
    main()
