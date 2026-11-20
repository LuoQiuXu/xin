from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
BASE = ROOT.parent / "ccsam_stage2_controlled_cifar100_20261106"
ORACLE_DIR = ROOT.parent / "ccsam_causal_oracle_20261119"
CURRENT_BRANCH = "experiment/tail-preserving-margin-aware-gate-20261120"
PINNED_PARENT_HEAD = "bb8536273effd8a478812509a5389c6c01c154ae"
EXPECTED_ELIGIBLE_PAIRS = 2210
EXPECTED_MANY_TARGETS = 34
EXPECTED_NONMANY_SOURCES = 65
HASHED_DOCS = (
    ROOT / "README.md",
    ROOT / "TPMAG_PROTOCOL.md",
    ROOT / "CODEX_TPMAG_EXECUTE_20261120.txt",
)

for directory in (ORACLE_DIR, BASE, BASE / "ccmia"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import oracle_runtime as OR  # noqa: E402
from causal_core import calibration_competitors, canonical_hash  # noqa: E402
from ccmia_core import (  # noqa: E402
    ParameterWritebackRoundingLedger,
    decompose_perturbation_gradient_by_source,
)
from ccsam_common import class_weights_from_counts  # noqa: E402
from train_ccsam import C3_ARM, compute_arm_loss  # noqa: E402
from train_ccmia import C2_ARM  # noqa: E402
from gate_core import (  # noqa: E402
    apply_tail_preserving_gate,
    build_margin_reference,
    compute_tail_preserving_gate,
)


SEEDS = OR.SEEDS
METRIC_KEYS = (
    "overall_accuracy",
    "macro_accuracy",
    "many_accuracy",
    "medium_accuracy",
    "few_accuracy",
    "fixed_many_margin",
)
BASELINE_METRIC_KEYS = (
    "overall_accuracy",
    "macro_accuracy",
    "many_accuracy",
    "few_accuracy",
    "fixed_many_margin",
)
BASELINE_ARMS = (
    "C2",
    "BASE_C3",
    "TARGETED",
    "RANDOM_MATCHED",
    "FREQUENCY_MATCHED",
    "ORACLE_SELECTION",
)


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json_exclusive(path, payload):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("JSON output outside TP-MAG experiment directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def write_json_atomic(path, payload):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("JSON output outside TP-MAG experiment directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    if temporary.exists():
        raise FileExistsError(temporary)
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def file_sha(path):
    return OR.file_sha(Path(path))


def code_hashes():
    paths = sorted(ROOT.glob("*.py")) + list(HASHED_DOCS)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing code/protocol files: " + ", ".join(missing))
    return {
        str(path.relative_to(ROOT)).replace("\\", "/"): file_sha(path)
        for path in paths
    }


def git(*args):
    result = subprocess.run(
        ["git", *args],
        cwd=REPO,
        text=True,
        capture_output=True,
        encoding="utf-8",
        check=True,
    )
    return result.stdout.strip()


def require_branch():
    current = git("branch", "--show-current")
    if current != CURRENT_BRANCH:
        raise RuntimeError(f"Wrong branch: {current}; expected {CURRENT_BRANCH}")
    git("merge-base", "--is-ancestor", PINNED_PARENT_HEAD, "HEAD")


def require_cpu():
    receipt_path = ROOT / "validation" / "CPU_VALIDATION.json"
    if not receipt_path.is_file():
        raise RuntimeError("CPU validation receipt missing; run validate_cpu.py first")
    receipt = read_json(receipt_path)
    if receipt.get("passed") is not True or receipt.get("code_sha256") != code_hashes():
        raise RuntimeError("CPU validation receipt failed or is stale")


def require_preflight():
    require_cpu()
    require_branch()
    path = ROOT / "validation" / "preflight_report.json"
    if not path.is_file():
        raise RuntimeError("TP-MAG preflight report missing")
    report = read_json(path)
    if report.get("passed") is not True or report.get("code_sha256") != code_hashes():
        raise RuntimeError("TP-MAG preflight failed or is stale")
    for filename, expected in report.get("immutable_inputs", {}).items():
        if file_sha(filename) != expected:
            raise RuntimeError(f"Immutable preflight input changed: {filename}")
    return report


def cuda_device():
    require_cpu()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; CPU fallback is prohibited")
    torch.cuda.init()
    if torch.cuda.device_count() < 1:
        raise RuntimeError("No CUDA device is available")
    torch.cuda.set_device(0)
    torch.cuda.synchronize(0)
    return torch.device("cuda:0")


def assert_test_guard_clean():
    guard = OR.guard_pass()
    if (
        guard.get("official_test_file_open_attempts") != 0
        or guard.get("CIFAR100_train_false_attempts") != 0
        or guard.get("official_TEST_used") is not False
    ):
        raise RuntimeError(f"Official TEST access guard is not clean: {guard}")
    return guard


def tensor_reference(context, initial_head, device):
    cal_x = torch.from_numpy(np.asarray(context["cal_x"], dtype=np.float32).copy()).to(device)
    cal_y = torch.from_numpy(np.asarray(context["cal_y"], dtype=np.int64).copy()).to(device)
    with torch.no_grad():
        initial_logits = initial_head.module.fc(cal_x)
    counts = torch.as_tensor(context["counts"], dtype=torch.long, device=device)
    reference = build_margin_reference(cal_x, cal_y, initial_logits, counts)
    expected_competitors = calibration_competitors(
        initial_logits.detach().cpu().numpy(),
        cal_y.detach().cpu().numpy(),
        np.asarray(context["counts"], dtype=np.int64),
    )
    actual_competitors = {
        int(target): int(reference["competitor_ids"][int(target)].item())
        for target in reference["target_ids"].detach().cpu().tolist()
    }
    if actual_competitors != expected_competitors:
        raise RuntimeError("TP-MAG frozen competitor map differs from parent implementation")
    pair_count = int(reference["eligible"].sum().item())
    if pair_count != EXPECTED_ELIGIBLE_PAIRS:
        raise RuntimeError(f"Unexpected eligible pair count: {pair_count}")
    if int(reference["target_ids"].numel()) != EXPECTED_MANY_TARGETS:
        raise RuntimeError("Unexpected represented Many target count")
    if int(reference["source_ids"].numel()) != EXPECTED_NONMANY_SOURCES:
        raise RuntimeError("Unexpected represented non-Many source count")
    return reference, cal_x, cal_y, initial_logits


def reference_record(reference, cal_x, cal_y, initial_logits):
    def digest(tensor):
        array = tensor.detach().cpu().contiguous().numpy()
        return hashlib.sha256(array.tobytes(order="C")).hexdigest()

    return {
        "eligible_pairs": int(reference["eligible"].sum().item()),
        "source_ids": reference["source_ids"].detach().cpu().tolist(),
        "target_ids": reference["target_ids"].detach().cpu().tolist(),
        "competitor_ids": reference["competitor_ids"].detach().cpu().tolist(),
        "calibration_features_sha256": digest(cal_x),
        "calibration_labels_sha256": digest(cal_y),
        "initial_logits_sha256": digest(initial_logits),
        "head_mean_features_sha256": digest(reference["head_mean_features"]),
        "tail_weighted_features_sha256": digest(reference["tail_weighted_features"]),
        "tail_probability_mean_sha256": digest(reference["tail_probability_mean"]),
    }


def run_preflight():
    require_cpu()
    require_branch()
    report_path = ROOT / "validation" / "preflight_report.json"
    if report_path.exists():
        raise FileExistsError("Preflight report already exists; refusing overwrite")

    context = OR.load_context()
    parent_complete = read_json(OR.ROOT / "formal_completed.json")
    if parent_complete.get("state") != "completed":
        raise RuntimeError("Completed Causal/Oracle parent experiment is required")
    if parent_complete.get("official_TEST_used") is not False:
        raise RuntimeError("Causal/Oracle parent does not certify TEST-free execution")
    if parent_complete.get("actual_optimizer_steps") != 91800:
        raise RuntimeError("Causal/Oracle parent optimizer-update ledger is invalid")
    final_validation_path = OR.ROOT / "validation" / "FINAL_ARTIFACT_VALIDATION.json"
    final_validation = read_json(final_validation_path)
    if (
        final_validation.get("passed") is not True
        or final_validation.get("experimental_code_still_matches_CPU_receipt") is not True
        or final_validation.get("all_masks_verified_against_seal") is not True
        or final_validation.get("official_TEST_used") is not False
        or final_validation.get("actual_optimizer_updates_from_observed_gradient_rows") != 91800
        or set(final_validation.get("phase_exit_codes", {})) != {
            "CPU", "preflight", "calibrate", "smoke", "formal", "analyze"
        }
        or any(int(code) != 0 for code in final_validation.get("phase_exit_codes", {}).values())
    ):
        raise RuntimeError("Causal/Oracle final artifact validation did not pass")
    if parent_complete.get("seeds") != list(SEEDS):
        raise RuntimeError("Parent seed set differs from TP-MAG protocol")
    if not set(BASELINE_ARMS).issubset(set(parent_complete.get("actual_arms", []))):
        raise RuntimeError("Parent baseline arms are incomplete")

    immutable = {}
    def add_input(path):
        path = Path(path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        immutable[str(path)] = file_sha(path)

    for path in (
        context["manifest_path"],
        context["checkpoint"],
        Path(context["manifest"]["split_path"]),
        context["prepared"] / "train_images.npy",
        context["prepared"] / "train_labels.npy",
        OR.ROOT / "formal_completed.json",
        OR.ROOT / "mask_seal.json",
        OR.ROOT / "ARTIFACT_SHA256.csv",
        OR.ROOT / "validation" / "FINAL_ARTIFACT_VALIDATION.json",
        BASE / "ccmia" / "formal_completed.json",
        BASE / "ccsam_common.py",
        BASE / "train_ccsam.py",
        BASE / "ccmia" / "ccmia_core.py",
        BASE / "ccmia" / "train_ccmia.py",
        ORACLE_DIR / "causal_core.py",
        ORACLE_DIR / "isolated_data.py",
        ORACLE_DIR / "oracle_runtime.py",
    ):
        add_input(path)

    reference_records = {}
    for seed in SEEDS:
        plan_path = context["prepared"] / f"sampling_plan_seed{seed}.npy"
        c0_dir = BASE / "results" / f"seed_{seed}" / "C0_BALANCED_CE"
        oracle_dir = OR.ROOT / "results" / f"seed_{seed}"
        parent_ccmia_status = BASE / "ccmia" / "results" / f"seed_{seed}" / "status.json"
        oracle_status = oracle_dir / "status.json"
        baseline_metrics = oracle_dir / "epoch_metrics.csv"
        for path in (plan_path, c0_dir / "batch_trace.csv", c0_dir / "status.json",
                     parent_ccmia_status, oracle_status, baseline_metrics):
            add_input(path)

        c0_status = read_json(c0_dir / "status.json")
        ccmia_status = read_json(parent_ccmia_status)
        oracle_seed_status = read_json(oracle_status)
        if c0_status.get("state") != "completed":
            raise RuntimeError(f"C0 reference trace is incomplete for seed {seed}")
        if ccmia_status.get("state") != "completed":
            raise RuntimeError(f"Original CC-MIA parent is incomplete for seed {seed}")
        if oracle_seed_status.get("state") != "completed":
            raise RuntimeError(f"Causal/Oracle parent is incomplete for seed {seed}")
        if oracle_seed_status.get("official_TEST_used") is not False:
            raise RuntimeError(f"Parent TEST provenance failed for seed {seed}")
        if (
            oracle_seed_status.get("epochs_completed") != OR.STAGE2_EPOCHS
            or oracle_seed_status.get("backbone_bn_bitwise_unchanged") is not True
            or set(oracle_seed_status.get("actual_optimizer_steps", {}).values()) != {OR.NSTEPS}
        ):
            raise RuntimeError(f"Parent seed {seed} optimizer/backbone ledger is invalid")

        plan = np.load(plan_path, mmap_mode="r", allow_pickle=False)
        if plan.shape != (OR.STAGE2_EPOCHS, OR.NTRAIN) or plan.dtype != np.int32:
            raise RuntimeError(f"Sampling plan shape/dtype mismatch for seed {seed}")
        plan_audit = OR.validate_class_aware_plan(context["train_labels"], seed, plan)
        if plan_audit.get("matches_official_sequence") is not True:
            raise RuntimeError(f"Official sampler parity failed for seed {seed}")

        backbone, initial_head, frozen = OR.make_models(context, seed, torch.device("cpu"))
        reference, cal_x, cal_y, initial_logits = tensor_reference(
            context, initial_head, torch.device("cpu")
        )
        reference_records[str(seed)] = reference_record(
            reference, cal_x, cal_y, initial_logits
        )
        OR.check_backbone_unchanged(backbone, frozen)
        del backbone, initial_head

    events = context["vault"].events
    if any(event.get("partition") != "calibration" for event in events):
        raise RuntimeError("Preflight accessed a non-Calibration holdout partition")

    report = {
        "passed": True,
        "completed_at": now(),
        "branch": CURRENT_BRANCH,
        "pinned_parent_head": PINNED_PARENT_HEAD,
        "head": git("rev-parse", "HEAD"),
        "code_sha256": code_hashes(),
        "immutable_inputs": immutable,
        "prepared_manifest_sha256": file_sha(context["manifest_path"]),
        "clean_stage1_sha256": file_sha(context["checkpoint"]),
        "parent_formal_completed_sha256": file_sha(OR.ROOT / "formal_completed.json"),
        "parent_final_artifact_validation_sha256": file_sha(
            OR.ROOT / "validation" / "FINAL_ARTIFACT_VALIDATION.json"
        ),
        "mask_seal_sha256": file_sha(OR.ROOT / "mask_seal.json"),
        "calibration_examples": int(len(context["cal_y"])),
        "calibration_group_counts": {
            "Many": int(np.count_nonzero(context["counts"][context["cal_y"]] > 100)),
            "Medium": int(np.count_nonzero(
                (context["counts"][context["cal_y"]] >= 20)
                & (context["counts"][context["cal_y"]] <= 100)
            )),
            "Few": int(np.count_nonzero(context["counts"][context["cal_y"]] < 20)),
        },
        "evaluation_features_labels_metrics_read": False,
        "evaluation_unopened": True,
        "reference_records_by_seed": reference_records,
        "official_TEST_used": False,
        "test_guard": assert_test_guard_clean(),
        "holdout_access_log": events,
        "environment": OR.environment(),
    }
    write_json_exclusive(report_path, report)
    print("TP-MAG PREFLIGHT PASS: Calibration-only reference verified; Evaluation remains unopened")


def compute_tpmag_batch(head, gap, labels, weights, counts, epoch, reference):
    fc = head.module.fc
    c3_pack = compute_arm_loss(
        C3_ARM, gap, labels, fc, weights, epoch, train_counts=counts
    )
    c2_pack = compute_arm_loss(C2_ARM, gap, labels, fc, weights, epoch)
    loss3, logits_clean, _, _, delta, perturb_stats, alpha = c3_pack
    loss2 = c2_pack[0]

    original_grad = torch.autograd.grad(loss3, (fc.weight, fc.bias))
    clean_grad = torch.autograd.grad(loss2, (fc.weight, fc.bias))
    with torch.no_grad():
        logits_adv = fc(gap.detach() + delta.detach())
        source_w, source_b = decompose_perturbation_gradient_by_source(
            gap.detach(),
            labels,
            delta.detach(),
            logits_clean.detach(),
            logits_adv.detach(),
            weights,
            float(alpha),
        )

    direct_w_error = float(
        (original_grad[0] - clean_grad[0] - source_w.sum(dim=0)).abs().max().item()
    )
    direct_b_error = float(
        (original_grad[1] - clean_grad[1] - source_b.sum(dim=0)).abs().max().item()
    )
    if not torch.allclose(
        original_grad[0] - clean_grad[0], source_w.sum(dim=0),
        rtol=OR.GRAD_RTOL, atol=OR.GRAD_ATOL
    ):
        raise RuntimeError(f"TP-MAG direct weight decomposition failed: {direct_w_error}")
    if not torch.allclose(
        original_grad[1] - clean_grad[1], source_b.sum(dim=0),
        rtol=OR.GRAD_RTOL, atol=OR.GRAD_ATOL
    ):
        raise RuntimeError(f"TP-MAG direct bias decomposition failed: {direct_b_error}")

    many_sources = torch.as_tensor(counts, device=labels.device)[labels] > 100
    if torch.count_nonzero(delta[many_sources]).item() != 0:
        raise RuntimeError("C3 unexpectedly perturbed a Many input")
    many_ids = torch.as_tensor(counts, device=labels.device) > 100
    if torch.count_nonzero(source_w[many_ids]).item() != 0:
        raise RuntimeError("Many source has a nonzero direct weight gradient")
    if torch.count_nonzero(source_b[many_ids]).item() != 0:
        raise RuntimeError("Many source has a nonzero direct bias gradient")

    gate, gate_diagnostics = compute_tail_preserving_gate(
        source_w, source_b, reference
    )
    grad_w, grad_b, gated_w, gated_b = apply_tail_preserving_gate(
        clean_grad[0], clean_grad[1], source_w, source_b, gate
    )
    if grad_w.dtype != torch.float32 or grad_b.dtype != torch.float32:
        raise RuntimeError("TP-MAG optimizer gradients must remain FP32")
    if not torch.isfinite(grad_w).all() or not torch.isfinite(grad_b).all():
        raise RuntimeError("Non-finite TP-MAG optimizer gradient")
    if not torch.equal(
        gate[~reference["eligible"]],
        torch.ones_like(gate[~reference["eligible"]]),
    ):
        raise RuntimeError("An ineligible source-target component changed")

    return {
        "loss": loss3,
        "clean_logits": logits_clean.detach(),
        "alpha": float(alpha),
        "perturb_stats": perturb_stats,
        "source_weight": source_w,
        "source_bias": source_b,
        "gate": gate,
        "gate_diagnostics": gate_diagnostics,
        "gated_weight": gated_w,
        "gated_bias": gated_b,
        "optimizer_grad_weight": grad_w.detach(),
        "optimizer_grad_bias": grad_b.detach(),
        "gradient_checks": {
            "direct_weight_max_abs": direct_w_error,
            "direct_bias_max_abs": direct_b_error,
        },
    }


def gate_batch_statistics(pack, reference):
    eligible = reference["eligible"]
    active = pack["gate_diagnostics"]["active_pairs"]
    gate_values = pack["gate"][active].detach().to(torch.float64)
    benefit = pack["gate_diagnostics"]["head_benefit"][active]
    tail_shift = pack["gate_diagnostics"]["tail_margin_shift"][active]
    tail_risk = pack["gate_diagnostics"]["tail_risk"][active]
    original_sq = (
        pack["source_weight"][eligible].detach().to(torch.float64).square().sum()
        + pack["source_bias"][eligible].detach().to(torch.float64).square().sum()
    )
    gated_sq = (
        pack["gated_weight"][eligible].detach().to(torch.float64).square().sum()
        + pack["gated_bias"][eligible].detach().to(torch.float64).square().sum()
    )
    retained = (
        torch.ones((), dtype=torch.float64, device=original_sq.device)
        if float(original_sq.item()) == 0.0
        else torch.sqrt(gated_sq / original_sq)
    )
    if gate_values.numel() == 0:
        mean_gate, median_gate = 1.0, 1.0
        zero_fraction, one_fraction = 0.0, 1.0
        mean_benefit, mean_tail_shift, mean_tail_risk = 0.0, 0.0, 0.0
    else:
        mean_gate = float(gate_values.mean().item())
        median_gate = float(gate_values.median().item())
        zero_fraction = float((gate_values == 0).to(torch.float64).mean().item())
        one_fraction = float((gate_values >= 1.0 - 1e-7).to(torch.float64).mean().item())
        mean_benefit = float(benefit.mean().item())
        mean_tail_shift = float(tail_shift.mean().item())
        mean_tail_risk = float(tail_risk.mean().item())
    return {
        "mean_gate": mean_gate,
        "median_gate": median_gate,
        "zero_gate_fraction": zero_fraction,
        "one_gate_fraction": one_fraction,
        "mean_head_benefit": mean_benefit,
        "mean_tail_margin_shift": mean_tail_shift,
        "mean_tail_risk": mean_tail_risk,
        "retained_direct_gradient_norm_ratio": float(retained.item()),
        "eligible_pair_count": int(eligible.sum().item()),
        "active_pair_count": int(active.sum().item()),
    }


def run_smoke():
    preflight = require_preflight()
    report_path = ROOT / "validation" / "smoke_report.json"
    if report_path.exists():
        raise FileExistsError("Smoke report already exists; refusing overwrite")
    context = OR.load_context()
    device = cuda_device()
    records = []

    for seed in SEEDS:
        backbone, head, frozen = OR.make_models(context, seed, device)
        reference, _, _, _ = tensor_reference(context, head, device)
        weights = class_weights_from_counts(context["counts"], device)
        optimizer = OR.optimizer_for(head)
        ledger = ParameterWritebackRoundingLedger()
        iterator = OR.real_batches(context, seed, backbone, device, epochs=1)
        epoch, batch, start, stop, gap, labels, trace = next(iterator)
        pack = compute_tpmag_batch(
            head, gap, labels, weights, context["counts"], epoch, reference
        )
        optimizer.zero_grad(set_to_none=True)
        fc = head.module.fc
        fc.weight.grad = pack["optimizer_grad_weight"].clone()
        fc.bias.grad = pack["optimizer_grad_bias"].clone()
        ledger.capture(fc.named_parameters(), optimizer)
        optimizer.step()
        ledger.record()
        OR.check_backbone_unchanged(backbone, frozen)
        if ledger.steps != 1:
            raise RuntimeError(f"Smoke optimizer step count mismatch for seed {seed}")
        if not all(torch.isfinite(p).all() for p in fc.parameters()):
            raise RuntimeError(f"Non-finite Smoke model for seed {seed}")
        records.append({
            "seed": int(seed),
            "epoch": int(epoch + 1),
            "batch": int(batch),
            "batch_size": int(stop - start),
            "optimizer_steps": int(ledger.steps),
            "trace_sha256": canonical_hash(trace),
            "gate_statistics": gate_batch_statistics(pack, reference),
            "gradient_checks": pack["gradient_checks"],
            "writeback_replay_bitwise_equal": True,
            "backbone_bn_bitwise_unchanged": True,
        })
        del backbone, head, optimizer, ledger

    if any(event.get("partition") != "calibration" for event in context["vault"].events):
        raise RuntimeError("Smoke accessed Evaluation before formal training")
    report = {
        "passed": True,
        "completed_at": now(),
        "code_sha256": code_hashes(),
        "preflight_sha256": file_sha(ROOT / "validation" / "preflight_report.json"),
        "seeds": records,
        "real_data_one_batch_per_seed": True,
        "evaluation_unopened": True,
        "formal_training_started": False,
        "official_TEST_used": False,
        "test_guard": assert_test_guard_clean(),
        "holdout_access_log": context["vault"].events,
        "environment": OR.environment(),
    }
    write_json_exclusive(report_path, report)
    print("TP-MAG SMOKE PASS: one real batch/update per seed; no Evaluation read")


def write_csv_header(path, fields):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("CSV output outside TP-MAG experiment directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("x", encoding="utf-8", newline="")
    writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
    writer.writeheader()
    handle.flush()
    return handle, writer


def train_seed(context, seed, device, destination):
    folder = destination / f"seed_{seed}"
    folder.mkdir(exist_ok=False)
    backbone, head, frozen = OR.make_models(context, seed, device)
    initial_hash = OR.state_hash(head)
    reference, _, _, _ = tensor_reference(context, head, device)
    weights = class_weights_from_counts(context["counts"], device)
    optimizer = OR.optimizer_for(head)
    ledger = ParameterWritebackRoundingLedger()
    status_path = folder / "status.json"
    status = {
        "state": "running",
        "seed": int(seed),
        "started_at": now(),
        "epochs_completed": 0,
        "optimizer_steps": 0,
        "initial_classifier_sha256": initial_hash,
        "official_TEST_used": False,
    }
    write_json_exclusive(status_path, status)

    gate_fields = (
        "seed", "epoch", "batch", "batch_size", "mean_gate", "median_gate",
        "zero_gate_fraction", "one_gate_fraction", "mean_head_benefit",
        "mean_tail_margin_shift", "mean_tail_risk",
        "retained_direct_gradient_norm_ratio", "eligible_pair_count", "active_pair_count",
        "direct_weight_decomposition_max_abs", "direct_bias_decomposition_max_abs",
        "trace_sha256",
    )
    epoch_fields = (
        "seed", "epoch", "optimizer_steps", "train_loss", "train_accuracy",
        "mean_gate", "median_gate", "zero_gate_fraction", "one_gate_fraction",
        "mean_head_benefit", "mean_tail_margin_shift", "mean_tail_risk",
        "retained_direct_gradient_norm_ratio", "active_pair_count",
        "max_gradient_decomposition_error",
        "official_TEST_used",
    )
    gate_handle, gate_writer = write_csv_header(folder / "gate_trace.csv", gate_fields)
    epoch_handle, epoch_writer = write_csv_header(folder / "epoch_metrics.csv", epoch_fields)

    try:
        steps = 0
        for epoch, batch, start, stop, gap, labels, trace in OR.real_batches(
            context, seed, backbone, device, epochs=OR.STAGE2_EPOCHS
        ):
            if batch == 0:
                epoch_loss_sum = 0.0
                epoch_correct_sum = 0
                epoch_sample_count = 0
                epoch_gate_rows = []

            pack = compute_tpmag_batch(
                head, gap, labels, weights, context["counts"], epoch, reference
            )
            stats = gate_batch_statistics(pack, reference)
            optimizer.zero_grad(set_to_none=True)
            fc = head.module.fc
            fc.weight.grad = pack["optimizer_grad_weight"].clone()
            fc.bias.grad = pack["optimizer_grad_bias"].clone()
            ledger.capture(fc.named_parameters(), optimizer)
            optimizer.step()
            ledger.record()
            steps += 1

            batch_size = int(stop - start)
            epoch_loss_sum += float(pack["loss"].detach().item()) * batch_size
            epoch_correct_sum += int(
                (pack["clean_logits"].argmax(dim=1) == labels).sum().item()
            )
            epoch_sample_count += batch_size

            gate_row = {
                "seed": seed, "epoch": epoch + 1, "batch": batch,
                "batch_size": batch_size, **stats,
                "direct_weight_decomposition_max_abs": pack["gradient_checks"]["direct_weight_max_abs"],
                "direct_bias_decomposition_max_abs": pack["gradient_checks"]["direct_bias_max_abs"],
                "trace_sha256": canonical_hash(trace),
            }
            gate_writer.writerow(gate_row)
            gate_handle.flush()
            epoch_gate_rows.append(gate_row)

            if batch == OR.NBATCH - 1:
                if epoch_sample_count != OR.NTRAIN or len(epoch_gate_rows) != OR.NBATCH:
                    raise RuntimeError("Epoch sample/batch count mismatch")
                numeric = lambda key: [float(r[key]) for r in epoch_gate_rows]
                max_error = max(
                    max(float(r["direct_weight_decomposition_max_abs"]),
                        float(r["direct_bias_decomposition_max_abs"]))
                    for r in epoch_gate_rows
                )
                epoch_row = {
                    "seed": seed,
                    "epoch": epoch + 1,
                    "optimizer_steps": steps,
                    "train_loss": epoch_loss_sum / epoch_sample_count,
                    "train_accuracy": epoch_correct_sum / epoch_sample_count * 100.0,
                    "mean_gate": float(np.mean(numeric("mean_gate"))),
                    "median_gate": float(np.mean(numeric("median_gate"))),
                    "zero_gate_fraction": float(np.mean(numeric("zero_gate_fraction"))),
                    "one_gate_fraction": float(np.mean(numeric("one_gate_fraction"))),
                    "mean_head_benefit": float(np.mean(numeric("mean_head_benefit"))),
                    "mean_tail_margin_shift": float(np.mean(numeric("mean_tail_margin_shift"))),
                    "mean_tail_risk": float(np.mean(numeric("mean_tail_risk"))),
                    "retained_direct_gradient_norm_ratio": float(
                        np.mean(numeric("retained_direct_gradient_norm_ratio"))
                    ),
                    "active_pair_count": float(np.mean(numeric("active_pair_count"))),
                    "max_gradient_decomposition_error": max_error,
                    "official_TEST_used": False,
                }
                epoch_writer.writerow(epoch_row)
                epoch_handle.flush()
                OR.check_backbone_unchanged(backbone, frozen)
                status.update({
                    "epochs_completed": epoch + 1,
                    "optimizer_steps": steps,
                    "last_epoch": epoch_row,
                })
                write_json_atomic(status_path, status)
                print(
                    f"TPMAG seed={seed} epoch={epoch + 1:02d}/{OR.STAGE2_EPOCHS} "
                    f"steps={steps} gate={epoch_row['mean_gate']:.4f} "
                    f"retained={epoch_row['retained_direct_gradient_norm_ratio']:.4f}",
                    flush=True,
                )

        if steps != OR.NSTEPS or ledger.steps != OR.NSTEPS:
            raise RuntimeError(
                f"TP-MAG optimizer steps mismatch: loop={steps}, ledger={ledger.steps}, expected={OR.NSTEPS}"
            )
        OR.check_backbone_unchanged(backbone, frozen)
        for state in optimizer.state.values():
            for value in state.values():
                if torch.is_tensor(value) and not torch.isfinite(value).all():
                    raise RuntimeError("Non-finite SGD momentum/optimizer state")

        checkpoint = folder / "final_classifier_TPMAG.pth"
        if checkpoint.exists():
            raise FileExistsError(checkpoint)
        torch.save({
            "state_dict_classifier": head.state_dict(),
            "seed": int(seed),
            "arm": "TPMAG",
            "epoch": OR.STAGE2_EPOCHS,
            "official_TEST_used": False,
            "initial_classifier_sha256": initial_hash,
            "final_classifier_sha256": OR.state_hash(head),
            "gate": "B/(B+R+1e-12)",
            "reference_source": "sealed Calibration only",
        }, checkpoint)
        status.update({
            "state": "completed",
            "completed_at": now(),
            "epochs_completed": OR.STAGE2_EPOCHS,
            "optimizer_steps": steps,
            "final_classifier_sha256": OR.state_hash(head),
            "checkpoint": checkpoint.name,
            "checkpoint_sha256": file_sha(checkpoint),
            "gate_trace_sha256": file_sha(folder / "gate_trace.csv"),
            "epoch_metrics_sha256": file_sha(folder / "epoch_metrics.csv"),
            "backbone_bn_bitwise_unchanged": True,
            "optimizer": {
                "type": "SGD",
                "lr": OR.STAGE2_LR,
                "momentum": OR.MOMENTUM,
                "weight_decay": OR.WEIGHT_DECAY,
                "nesterov": False,
                "AMP": False,
            },
            "official_TEST_used": False,
        })
        write_json_atomic(status_path, status)
        return status
    finally:
        gate_handle.close()
        epoch_handle.close()


def logits_for(head, features, device):
    chunks = []
    head.eval()
    with torch.no_grad():
        for start in range(0, len(features), 512):
            batch = torch.from_numpy(np.asarray(features[start:start + 512], dtype=np.float32).copy()).to(device)
            logits = head.module.fc(batch)
            if logits.shape != (len(batch), 100) or not torch.isfinite(logits).all():
                raise RuntimeError("Malformed or non-finite TP-MAG evaluation logits")
            chunks.append(logits.detach().cpu().numpy())
    head.train()
    return np.concatenate(chunks, axis=0)


def evaluate_partition(logits, labels, counts, competitor_ids):
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.asarray(counts, dtype=np.int64)
    if logits.shape != (len(labels), 100) or not np.isfinite(logits).all():
        raise RuntimeError("Malformed metric logits")
    predictions = logits.argmax(axis=1)
    correct = predictions == labels
    classes = np.unique(labels)
    per_class = {}
    for class_id in classes.tolist():
        selected = labels == int(class_id)
        per_class[int(class_id)] = float(correct[selected].mean() * 100.0)

    many_mask = counts[labels] > 100
    medium_mask = (counts[labels] >= 20) & (counts[labels] <= 100)
    few_mask = counts[labels] < 20
    if not many_mask.any() or not medium_mask.any() or not few_mask.any():
        raise RuntimeError("Evaluation partition lacks one of Many/Medium/Few groups")
    many_labels = labels[many_mask]
    q = competitor_ids[many_labels]
    if np.any(q < 0) or np.any(counts[q] > 100):
        raise RuntimeError("Frozen fixed Many competitors are invalid")
    rows = np.arange(len(labels))[many_mask]
    fixed_margin = logits[rows, many_labels] - logits[rows, q]
    class_means = lambda mask: float(np.mean([
        per_class[int(c)] for c in np.unique(labels[mask])
    ]))
    values = {
        "overall_accuracy": float(correct.mean() * 100.0),
        "macro_accuracy": float(np.mean(list(per_class.values()))),
        "many_accuracy": class_means(many_mask),
        "medium_accuracy": class_means(medium_mask),
        "few_accuracy": class_means(few_mask),
        "fixed_many_margin": float(fixed_margin.mean()),
        "samples": int(len(labels)),
        "many_samples": int(many_mask.sum()),
        "medium_samples": int(medium_mask.sum()),
        "few_samples": int(few_mask.sum()),
    }
    per_class_rows = [
        {"class_id": int(class_id), "n": int(np.count_nonzero(labels == class_id)),
         "accuracy": per_class[int(class_id)],
         "group": "Many" if counts[class_id] > 100 else
                  "Medium" if counts[class_id] >= 20 else "Few"}
        for class_id in classes.tolist()
    ]
    return values, per_class_rows


def read_parent_baselines(seed):
    path = OR.ROOT / "results" / f"seed_{seed}" / "epoch_metrics.csv"
    rows = OR.read_csv(path)
    result = {}
    for arm in BASELINE_ARMS:
        selected = [
            row for row in rows
            if int(row["epoch"]) == OR.STAGE2_EPOCHS
            and row["partition"] == "evaluation"
            and row["arm"] == arm
        ]
        if len(selected) != 1:
            raise RuntimeError(f"Expected exactly one final Evaluation row for seed {seed}, arm {arm}")
        row = selected[0]
        result[arm] = {key: float(row[key]) for key in BASELINE_METRIC_KEYS}
    return result


def run_formal():
    preflight = require_preflight()
    smoke_path = ROOT / "validation" / "smoke_report.json"
    if not smoke_path.is_file():
        raise RuntimeError("Production Smoke is required before formal training")
    smoke = read_json(smoke_path)
    if smoke.get("passed") is not True or smoke.get("code_sha256") != code_hashes():
        raise RuntimeError("Production Smoke failed or is stale")
    if smoke.get("formal_training_started") is not False:
        raise RuntimeError("Smoke receipt does not certify that formal training had not started")

    context = OR.load_context()
    device = cuda_device()
    destination = ROOT / "results"
    destination.mkdir(exist_ok=False)
    write_json_exclusive(destination / "FORMAL_STARTED.json", {
        "started_at": now(),
        "code_sha256": code_hashes(),
        "preflight_sha256": file_sha(ROOT / "validation" / "preflight_report.json"),
        "smoke_sha256": file_sha(smoke_path),
        "seeds": list(SEEDS),
        "arm": "TPMAG",
        "epochs": OR.STAGE2_EPOCHS,
        "optimizer_steps_per_seed": OR.NSTEPS,
        "official_TEST_used": False,
    })

    statuses = []
    for seed in SEEDS:
        statuses.append(train_seed(context, seed, device, destination))

    # Evaluation is opened only after all three formal TP-MAG seed runs finish.
    eval_x, eval_y = context["vault"].read("evaluation")
    if eval_x.shape != (990, 64) or len(eval_y) != 990:
        raise RuntimeError("Sealed Evaluation partition shape mismatch")
    if not np.array_equal(np.bincount(eval_y, minlength=100), np.r_[0, np.repeat(10, 99)]):
        raise RuntimeError("Sealed Evaluation class allocation mismatch")

    final_rows = []
    per_class_rows = []
    baseline_rows = []
    for seed in SEEDS:
        backbone, head, frozen = OR.make_models(context, seed, device)
        reference, cal_x, cal_y, initial_logits = tensor_reference(context, head, device)
        competitor_ids = reference["competitor_ids"].detach().cpu().numpy().astype(np.int64)
        checkpoint = destination / f"seed_{seed}" / "final_classifier_TPMAG.pth"
        payload = torch.load(checkpoint, map_location=device)
        head.load_state_dict(payload["state_dict_classifier"], strict=True)
        if OR.state_hash(head) != payload["final_classifier_sha256"]:
            raise RuntimeError(f"TP-MAG checkpoint state hash mismatch for seed {seed}")

        for partition, x, y in (
            ("calibration", context["cal_x"], context["cal_y"]),
            ("evaluation", eval_x, eval_y),
        ):
            logits = logits_for(head, x, device)
            values, class_rows = evaluate_partition(
                logits, y, context["counts"], competitor_ids
            )
            final_rows.append({
                "seed": seed, "arm": "TPMAG", "partition": partition, **values
            })
            if partition == "evaluation":
                per_class_rows.extend({
                    "seed": seed, "arm": "TPMAG", **row
                } for row in class_rows)

        baseline = read_parent_baselines(seed)
        for arm, values in baseline.items():
            baseline_rows.append({
                "seed": seed, "arm": arm, "partition": "evaluation", **values
            })
        OR.check_backbone_unchanged(backbone, frozen)
        del backbone, head

    if any(event.get("partition") not in ("calibration", "evaluation")
           for event in context["vault"].events):
        raise RuntimeError("Unexpected holdout access event")
    guard = assert_test_guard_clean()
    if guard.get("official_test_file_open_attempts") != 0 or guard.get("CIFAR100_train_false_attempts") != 0:
        raise RuntimeError("Official TEST access guard did not remain clean")

    metric_path = destination / "final_metrics.csv"
    class_path = destination / "per_class_evaluation_metrics.csv"
    comparison_path = destination / "paired_comparison.csv"
    write_csv_exclusive(metric_path, final_rows)
    write_csv_exclusive(class_path, per_class_rows)
    write_csv_exclusive(destination / "parent_baseline_metrics.csv", baseline_rows)

    comparison_rows = []
    for seed in SEEDS:
        tpmag = next(
            row for row in final_rows
            if row["seed"] == seed and row["partition"] == "evaluation"
        )
        for baseline in [row for row in baseline_rows if row["seed"] == seed]:
            for metric in BASELINE_METRIC_KEYS:
                comparison_rows.append({
                    "seed": seed,
                    "comparison": f"TPMAG_minus_{baseline['arm']}",
                    "baseline_arm": baseline["arm"],
                    "metric": metric,
                    "tpmag_value": float(tpmag[metric]),
                    "baseline_value": float(baseline[metric]),
                    "delta": float(tpmag[metric]) - float(baseline[metric]),
                })
    write_csv_exclusive(comparison_path, comparison_rows)

    completion = {
        "state": "completed",
        "completed_at": now(),
        "seeds": list(SEEDS),
        "arm": "TPMAG",
        "epochs": OR.STAGE2_EPOCHS,
        "optimizer_steps_per_seed": OR.NSTEPS,
        "actual_optimizer_steps": sum(int(x["optimizer_steps"]) for x in statuses),
        "seed_status": statuses,
        "code_sha256": code_hashes(),
        "preflight_sha256": file_sha(ROOT / "validation" / "preflight_report.json"),
        "smoke_sha256": file_sha(smoke_path),
        "evaluation_read_after_all_seed_training": True,
        "calibration_examples": int(len(context["cal_y"])),
        "evaluation_examples": int(len(eval_y)),
        "holdout_access_log": context["vault"].events,
        "official_TEST_used": False,
        "test_guard": guard,
        "environment": OR.environment(),
    }
    write_json_exclusive(ROOT / "formal_completed.json", completion)
    print("TP-MAG FORMAL COMPLETE: 3 seeds x 30 epochs; Evaluation opened after all training")


def write_csv_exclusive(path, rows):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("CSV output outside TP-MAG experiment directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fields = list(rows[0].keys())
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def run_analysis():
    completion_path = ROOT / "formal_completed.json"
    if not completion_path.is_file():
        raise RuntimeError("TP-MAG formal completion is required before analysis")
    completion = read_json(completion_path)
    if completion.get("state") != "completed" or completion.get("code_sha256") != code_hashes():
        raise RuntimeError("TP-MAG formal completion is invalid or stale")
    analysis_dir = ROOT / "analysis"
    analysis_dir.mkdir(exist_ok=False)

    comparisons = OR.read_csv(ROOT / "results" / "paired_comparison.csv")
    deltas = {}
    for row in comparisons:
        if row["metric"] not in ("overall_accuracy", "many_accuracy", "few_accuracy", "fixed_many_margin"):
            continue
        key = (int(row["seed"]), row["baseline_arm"], row["metric"])
        deltas[key] = float(row["delta"])

    def values(arm, metric):
        return np.asarray([deltas[(seed, arm, metric)] for seed in SEEDS], dtype=np.float64)

    base_overall = values("BASE_C3", "overall_accuracy")
    base_many = values("BASE_C3", "many_accuracy")
    base_margin = values("BASE_C3", "fixed_many_margin")
    base_few = values("BASE_C3", "few_accuracy")
    targeted_overall = values("TARGETED", "overall_accuracy")
    targeted_many = values("TARGETED", "many_accuracy")
    targeted_few = values("TARGETED", "few_accuracy")

    mean_many = float(base_many.mean())
    mean_margin = float(base_margin.mean())
    mean_few = float(base_few.mean())
    consistent = int(np.count_nonzero((base_many > 0) & (base_margin > 0)))
    head_pass = mean_many >= 2.0 and mean_margin >= 0.10 and consistent >= 2
    tail_pass = mean_few >= -2.0
    targeted_pareto_pass = (
        float(targeted_many.mean()) >= 0.0
        and float(targeted_few.mean()) >= 0.0
        and (
            float(targeted_many.mean()) > 0.0
            or float(targeted_few.mean()) > 0.0
        )
    )
    go = bool(head_pass and tail_pass and targeted_pareto_pass)

    def summary(arr):
        return {
            "mean": float(arr.mean()),
            "sample_sd": float(arr.std(ddof=1)),
            "per_seed": [float(x) for x in arr.tolist()],
        }

    decision = {
        "decision": "GO" if go else "NO_GO",
        "head_efficacy_requirement": {
            "mean_many_gain_pp_vs_BASE_C3": summary(base_many),
            "mean_fixed_many_margin_gain_vs_BASE_C3": summary(base_margin),
            "positive_both_in_at_least_two_seeds": consistent,
            "required_many_gain_pp": 2.0,
            "required_fixed_margin_gain": 0.10,
            "passed": bool(head_pass),
        },
        "overall_context": {
            "mean_overall_accuracy_delta_pp_vs_BASE_C3": summary(base_overall),
            "mean_overall_accuracy_delta_pp_vs_TARGETED": summary(targeted_overall),
        },
        "tail_floor_requirement": {
            "mean_few_change_pp_vs_BASE_C3": summary(base_few),
            "required_floor_pp": -2.0,
            "passed": bool(tail_pass),
        },
        "hard_suppression_tradeoff_requirement": {
            "mean_many_delta_vs_TARGETED_pp": summary(targeted_many),
            "mean_few_delta_vs_TARGETED_pp": summary(targeted_few),
            "required_weak_pareto_dominance": True,
            "passed": bool(targeted_pareto_pass),
        },
        "go": go,
        "stop_rule": (
            "GO: proceed only to external-dataset validation; do not add another mechanism."
            if go else
            "NO-GO: stop this Gate line. No epsilon, threshold, coefficient, loss or reference tuning is permitted."
        ),
        "official_TEST_used": False,
        "evaluation_scope": "TRAIN-derived development partition; not an independent test set",
    }
    write_json_exclusive(analysis_dir / "decision.json", decision)

    lines = [
        "# TP-MAG Final Development Report",
        "",
        f"Decision: **{decision['decision']}**",
        "",
        "All metrics below are on the sealed TRAIN-derived Evaluation partition. This is not an independent test set.",
        "Official CIFAR-100 TEST access: false.",
        "",
        "## Paired differences versus BASE_C3",
        "",
        "| Seed | Overall delta (pp) | Many accuracy delta (pp) | Fixed Many margin delta | Few accuracy delta (pp) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for index, seed in enumerate(SEEDS):
        lines.append(
            f"| {seed} | {base_overall[index]:+.3f} | {base_many[index]:+.3f} | {base_margin[index]:+.6f} | {base_few[index]:+.3f} |"
        )
    lines.extend([
        "",
        "## Paired differences versus TARGETED",
        "",
        "| Seed | Overall delta (pp) | Many accuracy delta (pp) | Few accuracy delta (pp) |",
        "|---:|---:|---:|---:|",
    ])
    for index, seed in enumerate(SEEDS):
        lines.append(
            f"| {seed} | {targeted_overall[index]:+.3f} | {targeted_many[index]:+.3f} | {targeted_few[index]:+.3f} |"
        )
    lines.extend([
        "",
        "## Frozen decision criteria",
        "",
        f"- Head efficacy: {'PASS' if head_pass else 'FAIL'}; mean Many gain {mean_many:+.3f} pp, mean fixed-margin gain {mean_margin:+.6f}, both positive in {consistent}/3 seeds.",
        f"- Tail floor: {'PASS' if tail_pass else 'FAIL'}; mean Few change {mean_few:+.3f} pp.",
        f"- Pareto comparison against TARGETED: {'PASS' if targeted_pareto_pass else 'FAIL'}; TP-MAG must be no worse on both mean Many and mean Few accuracy and strictly better on at least one.",
        "",
        "## Stop decision",
        "",
        decision["stop_rule"],
        "",
        "No threshold, epsilon, coefficient, loss or reference tuning was performed. The Gate was trained once with the frozen protocol.",
    ])
    report_path = analysis_dir / "TPMAG_FINAL_REPORT.md"
    with report_path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"TP-MAG analysis complete: {decision['decision']}")


def main():
    parser = argparse.ArgumentParser(description="Finite TP-MAG method experiment")
    parser.add_argument(
        "--phase",
        required=True,
        choices=("preflight", "smoke", "formal", "analyze"),
    )
    args = parser.parse_args()
    try:
        if args.phase == "preflight":
            run_preflight()
        elif args.phase == "smoke":
            run_smoke()
        elif args.phase == "formal":
            run_formal()
        else:
            run_analysis()
    except BaseException as error:
        failure = ROOT / "validation" / f"{args.phase.upper()}_FAILURE.json"
        if not failure.exists():
            write_json_exclusive(failure, {
                "phase": args.phase,
                "failed_at": now(),
                "exception": repr(error),
                "traceback": traceback.format_exc(),
                "official_TEST_used": False,
            })
        raise


if __name__ == "__main__":
    main()
