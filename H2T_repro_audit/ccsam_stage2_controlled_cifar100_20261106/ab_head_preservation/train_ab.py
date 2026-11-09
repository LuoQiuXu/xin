"""C4: full CC-SAM feature perturbation plus Many-only head-preservation distillation.

This runner is intentionally isolated from the sealed C0-C3 experiment code.
It trains only C4, reuses the per-seed C2 classifier as a frozen teacher, and
never accesses the official CIFAR-100 TEST split.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from ccsam_common import (  # noqa: E402
    ROOT, NUM_CLASSES, FEATURE_DIM, BATCH_SIZE, STAGE2_EPOCHS, STAGE2_LR,
    MOMENTUM, WEIGHT_DECAY, SEEDS, EXPECTED_CLEAN_SHA256,
    sha256_file, hash_arrays, state_hash, write_json, read_json,
    install_test_guard, guard_pass, resolve_checkpoint, build_transforms,
    build_models, seed_everything, augment_batch, class_weights_from_counts,
    alpha_for_epoch, check_backbone_unchanged, clean_compatible_gap,
    metrics_from_predictions,
)
from train_ccsam import (  # noqa: E402
    compute_arm_loss, read_trace, verify_analytic_gradient,
)
from train_c3 import baseline_file_snapshot  # noqa: E402


AB_DIR = Path(__file__).resolve().parent
C3_DIR = ROOT / "c3_selective"
C4_ARM = "C4_CCSAM_HEAD_PRESERVE"
TEACHER_ARM = "C2_WEIGHTED_CE_NO_PERTURB"
DISTILL_TEMPERATURE = 2.0
HEAD_PRESERVE_LAMBDA = 1.0
MANY_COUNT_THRESHOLD = 100
TRACE_FIELDS = (
    "epoch", "batch", "batch_size", "indices_labels_sha256",
    "augmented_images_sha256", "gap_features_sha256",
)
SOURCE_FILES = (
    "train_ccsam.py",
    "ccsam_common.py",
    "train_c3.py",
    "experiment_protocol.json",
    "formal_launch_20261106_01/ARTIFACT_SHA256.csv",
    "prepared/prepared_manifest.json",
    "c3_selective/formal_completed.json",
    "c3_selective/c3_summary.json",
    "c3_selective/C3_RESULTS.md",
    "c3_selective/ARTIFACT_SHA256.csv",
)


def stamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def exclusive_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as fp:
        json.dump(value, fp, ensure_ascii=False, indent=2, allow_nan=False)
        fp.write("\n")


def exclusive_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def head_preservation_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    train_counts,
    temperature: float = DISTILL_TEMPERATURE,
):
    """KL(teacher || student), averaged over Many inputs only.

    The softmax target spans all 100 output classes. The input mask is based on
    the sealed TRAIN count and uses the strict original Many rule n_c > 100.
    Teacher logits are detached even if a caller accidentally enables autograd.
    """
    if temperature <= 0:
        raise ValueError("Distillation temperature must be positive")
    if student_logits.ndim != 2 or student_logits.shape[1] != NUM_CLASSES:
        raise ValueError("Student logits must have shape [B,100]")
    if teacher_logits.shape != student_logits.shape:
        raise ValueError("Teacher and student logits must have identical shapes")
    if labels.shape != (student_logits.shape[0],) or labels.dtype != torch.long:
        raise ValueError("Labels must be int64 with one value per logit row")
    if student_logits.device != teacher_logits.device or labels.device != student_logits.device:
        raise ValueError("Student, teacher and labels must share a device")
    counts = torch.as_tensor(train_counts, dtype=torch.long, device=labels.device)
    if counts.shape != (NUM_CLASSES,) or bool((counts <= 0).any().item()):
        raise ValueError("Expected positive sealed TRAIN counts for all 100 classes")
    if bool(((labels < 0) | (labels >= NUM_CLASSES)).any().item()):
        raise ValueError("Label outside [0,99]")
    if not torch.isfinite(student_logits).all() or not torch.isfinite(teacher_logits).all():
        raise RuntimeError("Non-finite logits passed to head-preservation distillation")

    many_mask = counts[labels] > MANY_COUNT_THRESHOLD
    many_count = int(many_mask.sum().item())
    if many_count == 0:
        # Keep a valid zero-valued path so total-loss backward remains defined.
        return student_logits.sum() * 0.0, many_mask, 0

    student_log_prob = F.log_softmax(
        student_logits[many_mask] / float(temperature), dim=1
    )
    with torch.no_grad():
        teacher_prob = F.softmax(
            teacher_logits[many_mask].detach() / float(temperature), dim=1
        )
    loss = F.kl_div(
        student_log_prob, teacher_prob, reduction="batchmean"
    ) * (float(temperature) ** 2)
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite Many-only head-preservation KL")
    return loss, many_mask, many_count


def c3_artifact_snapshot() -> dict[str, str]:
    """Verify the complete archived C3 artifact index, including local-only weights."""
    marker_path = C3_DIR / "formal_completed.json"
    summary_path = C3_DIR / "c3_summary.json"
    index_path = C3_DIR / "ARTIFACT_SHA256.csv"
    if not marker_path.is_file() or not summary_path.is_file() or not index_path.is_file():
        raise FileNotFoundError("The completed C3 archive and checksum index are required")
    marker = read_json(marker_path)
    summary = read_json(summary_path)
    if (
        marker.get("passed") is not True
        or marker.get("arm") != "C3_NO_MANY_PERTURB"
        or marker.get("seeds") != list(SEEDS)
        or marker.get("epochs_per_seed") != STAGE2_EPOCHS
        or marker.get("official_TEST_used") is not False
        or marker.get("original_nine_runs_unchanged") is not True
        or summary.get("official_TEST_used") is not False
        or summary.get("original_nine_runs_untouched") is not True
    ):
        raise RuntimeError("C3 completion or TEST-free provenance is not valid")

    snapshots: dict[str, str] = {}
    with index_path.open("r", encoding="utf-8", newline="") as fp:
        rows = list(csv.DictReader(fp))
    if not rows:
        raise RuntimeError("C3 artifact checksum index is empty")
    for row in rows:
        relative = row["path"].replace("\\", "/")
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing archived C3 artifact: {path}")
        actual = sha256_file(path)
        if actual != row["sha256"]:
            raise RuntimeError(f"C3 archived artifact checksum mismatch: {relative}")
        snapshots[relative] = actual
    return snapshots


def source_snapshot() -> dict[str, str]:
    result = {}
    for relative in SOURCE_FILES:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing protected source/protocol input: {path}")
        result[relative] = sha256_file(path)
    script_path = AB_DIR / "train_ab.py"
    protocol_path = AB_DIR / "AB_PROTOCOL.md"
    for path in (script_path, protocol_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing C4 source/protocol file: {path}")
        result[str(path.relative_to(ROOT)).replace("\\", "/")] = sha256_file(path)
    return result


def collect_preflight(manifest: dict) -> dict:
    if manifest.get("clean_stage1_sha256") != EXPECTED_CLEAN_SHA256:
        raise RuntimeError("Prepared arrays do not reference the sealed Clean Stage I checkpoint")
    if manifest.get("official_TEST_used") is not False:
        raise RuntimeError("Prepared manifest does not certify TEST-free preparation")
    if manifest.get("validation_baseline_overall_matches_sealed_clean_protocol") is not True:
        raise RuntimeError("Prepared TRAIN-holdout baseline verification is not sealed")

    checkpoint = resolve_checkpoint(manifest.get("checkpoint_path"))
    if sha256_file(checkpoint) != EXPECTED_CLEAN_SHA256:
        raise RuntimeError("Clean Stage I checkpoint checksum mismatch")

    prepared_required = (
        "train_images.npy", "train_labels.npy", "validation_images.npy",
        "validation_gap_features.npy", "validation_labels.npy",
        "sampling_plan_seed123.npy", "sampling_plan_seed124.npy",
        "sampling_plan_seed125.npy",
    )
    for name in prepared_required:
        if not (ROOT / "prepared" / name).is_file():
            raise FileNotFoundError(f"Missing sealed prepared input: {name}")

    counts = np.asarray(manifest.get("train_class_counts"), dtype=np.int64)
    if counts.shape != (NUM_CLASSES,) or np.any(counts <= 0):
        raise RuntimeError("Invalid sealed 100-class TRAIN counts")
    if not np.any(counts > MANY_COUNT_THRESHOLD) or not np.any(counts <= MANY_COUNT_THRESHOLD):
        raise RuntimeError("C4 requires both Many and non-Many classes")

    # This verifies all nine original C0/C1/C2 runs against the independent
    # published checksum index, including local-only traces and checkpoints.
    baseline = baseline_file_snapshot()
    c3 = c3_artifact_snapshot()

    teacher_hashes = {}
    for seed in SEEDS:
        teacher_dir = ROOT / "results" / f"seed_{seed}" / TEACHER_ARM
        status_path = teacher_dir / "status.json"
        checkpoint_path = teacher_dir / "final_classifier.pth"
        if not status_path.is_file() or not checkpoint_path.is_file():
            raise FileNotFoundError(f"Missing local C2 teacher for seed {seed}")
        status = read_json(status_path)
        if (
            status.get("state") != "completed"
            or status.get("seed") != seed
            or status.get("arm") != TEACHER_ARM
            or status.get("final_epoch") != STAGE2_EPOCHS
            or status.get("official_TEST_used") is not False
        ):
            raise RuntimeError(f"Unsealed C2 teacher status for seed {seed}")
        actual = sha256_file(checkpoint_path)
        if actual != status.get("final_classifier_sha256"):
            raise RuntimeError(f"C2 teacher checkpoint/status SHA mismatch for seed {seed}")
        teacher_hashes[str(seed)] = actual

        plan = np.load(ROOT / "prepared" / f"sampling_plan_seed{seed}.npy",
                       mmap_mode="r", allow_pickle=False)
        if plan.shape != (STAGE2_EPOCHS, 10847) or plan.dtype != np.int32:
            raise RuntimeError(f"Invalid sealed sampling plan for seed {seed}")

    source = source_snapshot()
    return {
        "checked_at": stamp(),
        "passed": True,
        "clean_stage1_sha256": EXPECTED_CLEAN_SHA256,
        "stage2_seeds": list(SEEDS),
        "epochs_per_seed": STAGE2_EPOCHS,
        "train_samples": 10847,
        "validation_samples": 1980,
        "many_rule": "sealed TRAIN class count strictly greater than 100",
        "many_class_ids": np.flatnonzero(counts > MANY_COUNT_THRESHOLD).tolist(),
        "nonmany_class_ids": np.flatnonzero(counts <= MANY_COUNT_THRESHOLD).tolist(),
        "teacher_arm": TEACHER_ARM,
        "teacher_checkpoint_sha256": teacher_hashes,
        "distillation_temperature": DISTILL_TEMPERATURE,
        "head_preservation_lambda": HEAD_PRESERVE_LAMBDA,
        "original_c0_c1_c2_artifact_sha256": baseline,
        "c3_artifact_sha256": c3,
        "source_and_protocol_sha256": source,
        "official_TEST_used": False,
    }


def compare_protected_inputs(previous: dict) -> None:
    current_baseline = baseline_file_snapshot()
    if current_baseline != previous["original_c0_c1_c2_artifact_sha256"]:
        changed = sorted(
            key for key in set(current_baseline) | set(previous["original_c0_c1_c2_artifact_sha256"])
            if current_baseline.get(key) != previous["original_c0_c1_c2_artifact_sha256"].get(key)
        )
        raise RuntimeError(f"Original C0/C1/C2 artifacts changed: {changed}")

    current_c3 = c3_artifact_snapshot()
    if current_c3 != previous["c3_artifact_sha256"]:
        changed = sorted(
            key for key in set(current_c3) | set(previous["c3_artifact_sha256"])
            if current_c3.get(key) != previous["c3_artifact_sha256"].get(key)
        )
        raise RuntimeError(f"Archived C3 artifacts changed: {changed}")

    current_source = source_snapshot()
    if current_source != previous["source_and_protocol_sha256"]:
        changed = sorted(
            key for key in set(current_source) | set(previous["source_and_protocol_sha256"])
            if current_source.get(key) != previous["source_and_protocol_sha256"].get(key)
        )
        raise RuntimeError(f"Protected source/protocol changed during C4: {changed}")


def load_teacher(seed: int, student_classifier, device: torch.device):
    """Load the matching seed's final C2 classifier as a frozen linear teacher."""
    teacher_dir = ROOT / "results" / f"seed_{seed}" / TEACHER_ARM
    status = read_json(teacher_dir / "status.json")
    checkpoint_path = teacher_dir / "final_classifier.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if (
        checkpoint.get("epoch") != STAGE2_EPOCHS
        or checkpoint.get("seed") != seed
        or checkpoint.get("arm") != TEACHER_ARM
        or checkpoint.get("clean_stage1_sha256") != EXPECTED_CLEAN_SHA256
        or checkpoint.get("official_TEST_used") is not False
        or checkpoint.get("backbone_retrained") is not False
    ):
        raise RuntimeError(f"Invalid C2 teacher checkpoint metadata for seed {seed}")

    student_initial_hash = state_hash(student_classifier)
    if checkpoint.get("initial_classifier_sha256") != student_initial_hash:
        raise RuntimeError(
            f"C2 teacher and C4 student do not share the sealed Clean classifier initialization "
            f"for seed {seed}"
        )

    state = checkpoint.get("state_dict_classifier")
    if not isinstance(state, dict) or "fc.weight" not in state or "fc.bias" not in state:
        raise RuntimeError(f"Unexpected C2 classifier state dictionary for seed {seed}")
    if tuple(state["fc.weight"].shape) != (NUM_CLASSES, FEATURE_DIM):
        raise RuntimeError("C2 teacher weight shape is not [100,64]")
    if tuple(state["fc.bias"].shape) != (NUM_CLASSES,):
        raise RuntimeError("C2 teacher bias shape is not [100]")

    # Constructing nn.Linear consumes RNG. Fork/restore it so C4's training RNG
    # state is not perturbed by teacher instantiation.
    with torch.random.fork_rng(devices=[device.index] if device.type == "cuda" else []):
        teacher = nn.Linear(FEATURE_DIM, NUM_CLASSES).to(device)
    teacher.load_state_dict({
        "weight": state["fc.weight"],
        "bias": state["fc.bias"],
    }, strict=True)
    teacher.eval()
    teacher.requires_grad_(False)

    teacher_sha = sha256_file(checkpoint_path)
    if teacher_sha != status.get("final_classifier_sha256"):
        raise RuntimeError(f"C2 teacher checkpoint changed after preflight for seed {seed}")
    return teacher, teacher_sha, student_initial_hash


def evaluate_student_teacher(student_classifier, teacher, validation_gap, validation_labels,
                             train_counts, device):
    student_classifier.eval()
    teacher.eval()
    student_predictions = []
    teacher_predictions = []
    many_student_logits = []
    many_teacher_logits = []
    many_labels = []

    with torch.no_grad():
        for start in range(0, len(validation_labels), 512):
            end = min(start + 512, len(validation_labels))
            features = torch.from_numpy(
                np.array(validation_gap[start:end], dtype=np.float32, copy=True)
            ).to(device)
            student_logits = student_classifier.module.fc(features)
            teacher_logits = teacher(features)
            if (
                student_logits.shape != (end - start, NUM_CLASSES)
                or teacher_logits.shape != student_logits.shape
                or not torch.isfinite(student_logits).all()
                or not torch.isfinite(teacher_logits).all()
            ):
                raise RuntimeError("Malformed/non-finite C4 validation logits")
            student_predictions.append(student_logits.argmax(dim=1).cpu().numpy())
            teacher_predictions.append(teacher_logits.argmax(dim=1).cpu().numpy())

            labels_chunk = torch.as_tensor(
                np.asarray(validation_labels[start:end], dtype=np.int64), device=device
            )
            many_mask = torch.as_tensor(
                train_counts, dtype=torch.long, device=device
            )[labels_chunk] > MANY_COUNT_THRESHOLD
            if bool(many_mask.any().item()):
                many_student_logits.append(student_logits[many_mask].cpu())
                many_teacher_logits.append(teacher_logits[many_mask].cpu())
                many_labels.append(labels_chunk[many_mask].cpu())

    student_pred = np.concatenate(student_predictions)
    teacher_pred = np.concatenate(teacher_predictions)
    student_metrics = metrics_from_predictions(student_pred, validation_labels, train_counts)
    teacher_metrics = metrics_from_predictions(teacher_pred, validation_labels, train_counts)

    if not many_student_logits:
        raise RuntimeError("The TRAIN-holdout validation partition has no observed Many examples")
    s_logits = torch.cat(many_student_logits, dim=0).to(device)
    t_logits = torch.cat(many_teacher_logits, dim=0).to(device)
    m_labels = torch.cat(many_labels, dim=0).to(device)
    kd, many_mask, many_count = head_preservation_kl(
        s_logits, t_logits, m_labels, train_counts, DISTILL_TEMPERATURE
    )
    if many_count != len(m_labels) or not bool(many_mask.all().item()):
        raise RuntimeError("Validation Many-only distillation mask is inconsistent")
    agreement = float(
        (s_logits.argmax(dim=1) == t_logits.argmax(dim=1)).float().mean().cpu()
    )

    student_classifier.train()
    teacher.eval()
    return student_metrics, teacher_metrics, {
        "observed_many_validation_samples": int(len(m_labels)),
        "teacher_student_many_kl": float(kd.detach().cpu()),
        "teacher_student_many_top1_agreement": agreement,
        "teacher_validation_many_accuracy": float(teacher_metrics["Many"]),
        "student_validation_many_accuracy": float(student_metrics["Many"]),
    }


def smoke_one_seed(seed, manifest, train_images, train_labels, train_transform, device):
    checkpoint = resolve_checkpoint(manifest["checkpoint_path"])
    seed_everything(seed, deterministic=True)
    backbone, student, _ = build_models(checkpoint, device)
    frozen_backbone = {
        key: value.detach().cpu().clone()
        for key, value in backbone.state_dict().items()
    }
    initial_student_hash = state_hash(student)
    teacher, teacher_sha, _ = load_teacher(seed, student, device)
    teacher_initial_hash = state_hash(teacher)

    counts = np.asarray(manifest["train_class_counts"], dtype=np.int64)
    weights = class_weights_from_counts(counts, device)
    plan = np.load(ROOT / "prepared" / f"sampling_plan_seed{seed}.npy",
                   mmap_mode="r", allow_pickle=False)
    c1_trace_path = ROOT / "results" / f"seed_{seed}" / "C1_CCSAM_FEATURE_GRAD" / "batch_trace.csv"
    reference = read_trace(c1_trace_path)
    if len(reference) != STAGE2_EPOCHS * 170:
        raise RuntimeError(f"C1 paired trace length mismatch for seed {seed}")

    generator = torch.Generator(device="cpu").manual_seed(seed + 300009)
    batch_reports = []
    optimizer = None
    for batch_id in (0, 1):
        start = batch_id * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(train_labels))
        indices = np.asarray(plan[0, start:end], dtype=np.int64)
        target_np = np.asarray(train_labels[indices], dtype=np.int64)
        target = torch.as_tensor(target_np, dtype=torch.long, device=device)
        images_cpu = augment_batch(train_images, indices, train_transform, generator)
        with torch.no_grad():
            gap = clean_compatible_gap(backbone(images_cpu.to(device)))
        trace_row = {
            "epoch": "1",
            "batch": str(batch_id),
            "batch_size": str(end - start),
            "indices_labels_sha256": hash_arrays(indices, target_np),
            "augmented_images_sha256": hash_arrays(images_cpu.numpy()),
            "gap_features_sha256": hash_arrays(gap.detach().cpu().numpy()),
        }
        expected = reference[batch_id]
        if any(trace_row[key] != expected[key] for key in TRACE_FIELDS):
            raise RuntimeError(f"C4 smoke inputs differ from original C1 at seed={seed}, batch={batch_id}")

        c1_loss, logits_clean, clean_losses, adv_ce, delta, pstats, alpha = compute_arm_loss(
            "C1_CCSAM_FEATURE_GRAD", gap, target, student.module.fc, weights, 0
        )
        if batch_id == 0:
            gradient_check = verify_analytic_gradient(
                gap, target, student.module.fc, weights
            )
        else:
            gradient_check = None

        with torch.no_grad():
            teacher_logits = teacher(gap.detach())
        kd, many_mask, many_count = head_preservation_kl(
            logits_clean, teacher_logits, target, counts, DISTILL_TEMPERATURE
        )
        total_loss = c1_loss + HEAD_PRESERVE_LAMBDA * kd
        if not torch.isfinite(total_loss):
            raise RuntimeError("C4 smoke produced a non-finite total loss")
        if batch_id == 1 and (many_count <= 0 or many_count >= len(target)):
            raise RuntimeError("Second real smoke batch must contain both Many and non-Many examples")

        report = {
            "batch": batch_id,
            "paired_c1_trace_exact": True,
            "many_samples": many_count,
            "nonmany_samples": int((~many_mask).sum().item()),
            "c1_base_loss": float(c1_loss.detach().cpu()),
            "head_kd_loss": float(kd.detach().cpu()),
            "total_loss": float(total_loss.detach().cpu()),
            "mean_delta_l2": float(pstats["mean_delta_l2_per_sample"]),
            "temperature": DISTILL_TEMPERATURE,
            "lambda": HEAD_PRESERVE_LAMBDA,
            "analytic_gradient_check": gradient_check,
        }
        batch_reports.append(report)

        if batch_id == 1:
            optimizer = torch.optim.SGD(
                student.module.fc.parameters(),
                lr=STAGE2_LR,
                momentum=MOMENTUM,
                weight_decay=WEIGHT_DECAY,
                nesterov=False,
            )
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            if any(
                parameter.grad is None or not torch.isfinite(parameter.grad).all()
                for parameter in student.module.fc.parameters()
            ):
                raise RuntimeError("C4 student classifier gradient missing/non-finite")
            if any(parameter.grad is not None for parameter in backbone.parameters()):
                raise RuntimeError("Frozen backbone received a C4 smoke gradient")
            if any(parameter.grad is not None for parameter in teacher.parameters()):
                raise RuntimeError("Frozen C2 teacher received a C4 smoke gradient")
            optimizer.step()
            check_backbone_unchanged(backbone, frozen_backbone)
            if state_hash(student) == initial_student_hash:
                raise RuntimeError("C4 smoke did not update the student classifier")
            if state_hash(teacher) != teacher_initial_hash:
                raise RuntimeError("C2 teacher changed during C4 smoke")

    result = {
        "seed": seed,
        "teacher_checkpoint_sha256": teacher_sha,
        "paired_with_original_c1_batches": True,
        "batches": batch_reports,
        "disposable_student_update": True,
        "student_updated": True,
        "teacher_frozen": True,
        "backbone_and_bn_bitwise_unchanged": True,
    }
    del optimizer, backbone, student, teacher
    torch.cuda.empty_cache()
    return result


def smoke_all(manifest, train_images, train_labels, train_transform, device, preflight):
    path = AB_DIR / "smoke_report.json"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing C4 smoke report: {path}")
    reports = [
        smoke_one_seed(seed, manifest, train_images, train_labels, train_transform, device)
        for seed in SEEDS
    ]
    compare_protected_inputs(preflight)
    report = {
        "passed": True,
        "completed_at": stamp(),
        "arm": C4_ARM,
        "seeds": list(SEEDS),
        "batches_per_seed": 2,
        "formal_training_started": False,
        "only_disposable_student_updates": True,
        "teacher_temperature": DISTILL_TEMPERATURE,
        "head_preservation_lambda": HEAD_PRESERVE_LAMBDA,
        "checks": reports,
        "source_sha256": sha256_file(AB_DIR / "train_ab.py"),
        "protocol_sha256": sha256_file(AB_DIR / "AB_PROTOCOL.md"),
        "preflight_report_sha256": sha256_file(AB_DIR / "preflight_report.json"),
        "prepared_manifest_sha256": sha256_file(ROOT / "prepared" / "prepared_manifest.json"),
        "test_guard": guard_pass(),
        "official_TEST_used": False,
    }
    if report["test_guard"].get("blocked_attempts") or report["test_guard"].get("official_TEST_used"):
        raise RuntimeError("Official TEST guard violation during C4 smoke")
    exclusive_json(path, report)
    return report


def train_one_seed(seed, manifest, train_images, train_labels, validation_gap,
                   validation_labels, train_transform, device, teacher_sha_expected):
    out = AB_DIR / "results" / f"seed_{seed}" / C4_ARM
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing C4 run: {out}")
    out.mkdir(parents=True, exist_ok=False)

    seed_everything(seed, deterministic=True)
    checkpoint = resolve_checkpoint(manifest["checkpoint_path"])
    backbone, student, _ = build_models(checkpoint, device)
    frozen_backbone = {
        key: value.detach().cpu().clone()
        for key, value in backbone.state_dict().items()
    }
    initial_backbone_hash = state_hash(backbone)
    initial_student_hash = state_hash(student)

    teacher, teacher_sha, teacher_initial_student_hash = load_teacher(seed, student, device)
    if teacher_sha != teacher_sha_expected or teacher_initial_student_hash != initial_student_hash:
        raise RuntimeError(f"C2 teacher changed or initialization mismatch for seed {seed}")
    teacher_initial_hash = state_hash(teacher)

    optimizer = torch.optim.SGD(
        student.module.fc.parameters(),
        lr=STAGE2_LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        nesterov=False,
    )
    if optimizer.state:
        raise RuntimeError("C4 optimizer must start with empty momentum state")

    counts = np.asarray(manifest["train_class_counts"], dtype=np.int64)
    weights = class_weights_from_counts(counts, device)
    plan = np.load(ROOT / "prepared" / f"sampling_plan_seed{seed}.npy",
                   mmap_mode="r", allow_pickle=False)
    if plan.shape != (STAGE2_EPOCHS, len(train_labels)):
        raise RuntimeError(f"Unexpected sealed sampling plan shape for seed {seed}")

    c1_trace_path = ROOT / "results" / f"seed_{seed}" / "C1_CCSAM_FEATURE_GRAD" / "batch_trace.csv"
    reference_rows = read_trace(c1_trace_path)
    expected_batches = STAGE2_EPOCHS * ((len(train_labels) + BATCH_SIZE - 1) // BATCH_SIZE)
    if len(reference_rows) != expected_batches:
        raise RuntimeError(f"Original C1 batch trace length mismatch for seed {seed}")

    augmentation_generator = torch.Generator(device="cpu").manual_seed(seed + 300009)
    trace_path = out / "batch_trace.csv"
    metrics_path = out / "metrics.csv"
    metric_fields = [
        "epoch", "alpha", "learning_rate", "train_total_loss", "train_c1_base_loss",
        "train_head_kd_loss", "head_kd_lambda", "distill_temperature",
        "train_clean_accuracy", "validation_Overall", "validation_Macro",
        "validation_Head", "validation_OfficialMedium", "validation_Tail",
        "validation_Many", "validation_MediumN", "validation_Few",
        "teacher_validation_Many", "teacher_student_Many_KL",
        "teacher_student_Many_top1_agreement",
        "mean_delta_l2_per_sample", "mean_relative_delta_l2",
        "mean_class_gradient_norm_before_normalization", "mean_adv_minus_clean_ce",
        "zero_gradient_classes", "steps", "samples", "many_samples_for_kd",
    ]
    status = {
        "state": "running",
        "seed": int(seed),
        "arm": C4_ARM,
        "started_at": stamp(),
        "clean_stage1_sha256": manifest["clean_stage1_sha256"],
        "initial_backbone_sha256": initial_backbone_hash,
        "initial_classifier_sha256": initial_student_hash,
        "teacher_arm": TEACHER_ARM,
        "teacher_checkpoint_sha256": teacher_sha,
        "teacher_frozen": True,
        "teacher_temperature": DISTILL_TEMPERATURE,
        "head_preservation_lambda": HEAD_PRESERVE_LAMBDA,
        "many_count_rule": "TRAIN count strictly greater than 100",
        "optimizer": {
            "type": "SGD", "lr": STAGE2_LR, "momentum": MOMENTUM,
            "weight_decay": WEIGHT_DECAY, "nesterov": False,
        },
        "backbone_frozen": True,
        "bn_eval_and_buffers_frozen": True,
        "official_TEST_used": False,
        "epochs": [],
    }
    write_json(out / "status.json", status)

    global_batch = 0
    with trace_path.open("x", encoding="utf-8", newline="") as trace_file, \
         metrics_path.open("x", encoding="utf-8", newline="") as metric_file:
        trace_writer = csv.DictWriter(trace_file, fieldnames=list(TRACE_FIELDS))
        metric_writer = csv.DictWriter(metric_file, fieldnames=metric_fields)
        trace_writer.writeheader()
        metric_writer.writeheader()

        for epoch in range(STAGE2_EPOCHS):
            student.train()
            total_loss_sum = 0.0
            c1_loss_sum = 0.0
            kd_loss_sum = 0.0
            many_seen = 0
            correct = 0
            seen = 0
            delta_sum = 0.0
            relative_delta_sum = 0.0
            grad_norm_sum = 0.0
            adv_ce_delta_sum = 0.0
            zero_gradient_classes = 0
            epoch_steps = 0
            alpha_value = alpha_for_epoch(epoch)

            for batch_index, start in enumerate(range(0, len(train_labels), BATCH_SIZE)):
                end = min(start + BATCH_SIZE, len(train_labels))
                indices = np.asarray(plan[epoch, start:end], dtype=np.int64)
                target_np = np.asarray(train_labels[indices], dtype=np.int64)
                target = torch.as_tensor(target_np, dtype=torch.long, device=device)
                images_cpu = augment_batch(
                    train_images, indices, train_transform, augmentation_generator
                )
                with torch.no_grad():
                    feature_map = backbone(images_cpu.to(device, non_blocking=True))
                    gap = clean_compatible_gap(feature_map)
                if gap.shape != (end - start, FEATURE_DIM) or not torch.isfinite(gap).all():
                    raise RuntimeError("Malformed or non-finite C4 GAP features")

                trace_row = {
                    "epoch": str(epoch + 1),
                    "batch": str(batch_index),
                    "batch_size": str(end - start),
                    "indices_labels_sha256": hash_arrays(indices, target_np),
                    "augmented_images_sha256": hash_arrays(images_cpu.numpy()),
                    "gap_features_sha256": hash_arrays(gap.detach().cpu().numpy()),
                }
                expected = reference_rows[global_batch]
                if any(trace_row[key] != expected[key] for key in TRACE_FIELDS):
                    raise RuntimeError(
                        f"C4/C1 paired input mismatch: seed={seed}, epoch={epoch + 1}, batch={batch_index}"
                    )
                trace_writer.writerow(trace_row)

                c1_loss, logits_clean, clean_losses, adv_ce, delta, pstats, alpha_value = compute_arm_loss(
                    "C1_CCSAM_FEATURE_GRAD", gap, target, student.module.fc, weights, epoch
                )
                with torch.no_grad():
                    teacher_logits = teacher(gap.detach())
                kd_loss, many_mask, many_count = head_preservation_kl(
                    logits_clean, teacher_logits, target, counts, DISTILL_TEMPERATURE
                )
                total_loss = c1_loss + HEAD_PRESERVE_LAMBDA * kd_loss
                if not torch.isfinite(total_loss):
                    raise RuntimeError("Non-finite C4 total loss")

                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                if any(
                    parameter.grad is None or not torch.isfinite(parameter.grad).all()
                    for parameter in student.module.fc.parameters()
                ):
                    raise RuntimeError(f"Missing/non-finite C4 student gradient at seed={seed}")
                if any(parameter.grad is not None for parameter in backbone.parameters()):
                    raise RuntimeError("Frozen backbone unexpectedly received a gradient")
                if any(parameter.grad is not None for parameter in teacher.parameters()):
                    raise RuntimeError("Frozen C2 teacher unexpectedly received a gradient")
                optimizer.step()

                batch_size = end - start
                total_loss_sum += float(total_loss.detach().cpu()) * batch_size
                c1_loss_sum += float(c1_loss.detach().cpu()) * batch_size
                kd_loss_sum += float(kd_loss.detach().cpu()) * many_count
                many_seen += many_count
                correct += int((logits_clean.detach().argmax(dim=1) == target).sum().item())
                seen += batch_size
                delta_sum += pstats["mean_delta_l2_per_sample"] * batch_size
                relative_delta_sum += pstats["mean_relative_delta_l2"] * batch_size
                grad_norm_sum += pstats["mean_class_gradient_norm_before_normalization"] * batch_size
                if "mean_unweighted_adv_minus_clean_ce" in pstats:
                    adv_ce_delta_sum += pstats["mean_unweighted_adv_minus_clean_ce"] * batch_size
                zero_gradient_classes += int(pstats["zero_gradient_classes"])
                epoch_steps += 1
                global_batch += 1

            if seen != len(train_labels) or epoch_steps != 170:
                raise RuntimeError(f"C4 epoch accounting mismatch: samples={seen}, steps={epoch_steps}")
            if many_seen <= 0:
                raise RuntimeError(f"C4 head-preservation KD was inactive for the entire epoch: seed={seed}, epoch={epoch + 1}")
            if delta_sum <= 0:
                raise RuntimeError(f"C4 original C1 feature perturbation was inactive: seed={seed}, epoch={epoch + 1}")
            check_backbone_unchanged(backbone, frozen_backbone)
            if state_hash(teacher) != teacher_initial_hash:
                raise RuntimeError("Frozen C2 teacher parameters changed during formal training")

            student_metrics, teacher_metrics, teacher_diagnostics = evaluate_student_teacher(
                student, teacher, validation_gap, validation_labels, counts, device
            )
            row = {
                "epoch": epoch + 1,
                "alpha": float(alpha_value),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "train_total_loss": total_loss_sum / seen,
                "train_c1_base_loss": c1_loss_sum / seen,
                "train_head_kd_loss": kd_loss_sum / max(many_seen, 1),
                "head_kd_lambda": HEAD_PRESERVE_LAMBDA,
                "distill_temperature": DISTILL_TEMPERATURE,
                "train_clean_accuracy": 100.0 * correct / seen,
                "validation_Overall": student_metrics["Overall"],
                "validation_Macro": student_metrics["Macro"],
                "validation_Head": student_metrics["Head"],
                "validation_OfficialMedium": student_metrics["OfficialMedium"],
                "validation_Tail": student_metrics["Tail"],
                "validation_Many": student_metrics["Many"],
                "validation_MediumN": student_metrics["MediumN"],
                "validation_Few": student_metrics["Few"],
                "teacher_validation_Many": teacher_metrics["Many"],
                "teacher_student_Many_KL": teacher_diagnostics["teacher_student_many_kl"],
                "teacher_student_Many_top1_agreement": teacher_diagnostics["teacher_student_many_top1_agreement"],
                "mean_delta_l2_per_sample": delta_sum / seen,
                "mean_relative_delta_l2": relative_delta_sum / seen,
                "mean_class_gradient_norm_before_normalization": grad_norm_sum / seen,
                "mean_adv_minus_clean_ce": adv_ce_delta_sum / seen,
                "zero_gradient_classes": zero_gradient_classes,
                "steps": epoch_steps,
                "samples": seen,
                "many_samples_for_kd": many_seen,
            }
            metric_writer.writerow(row)
            metric_file.flush()
            trace_file.flush()

            status["epochs"].append({
                "epoch": epoch + 1,
                "metrics": row,
                "validation": student_metrics,
                "teacher_validation": teacher_metrics,
                "teacher_diagnostics": teacher_diagnostics,
                "frozen_backbone_and_bn_bitwise_unchanged": True,
                "teacher_bitwise_unchanged": True,
            })
            write_json(out / "status.json", status)
            print(
                f"seed={seed} C4 epoch={epoch + 1}/{STAGE2_EPOCHS} "
                f"loss={row['train_total_loss']:.5f} kd={row['train_head_kd_loss']:.5f} "
                f"overall={row['validation_Overall']:.3f} many={row['validation_Many']:.3f} "
                f"few={row['validation_Few']:.3f}",
                flush=True,
            )

    check_backbone_unchanged(backbone, frozen_backbone)
    if state_hash(teacher) != teacher_initial_hash:
        raise RuntimeError("C2 teacher changed before C4 checkpoint save")
    final_metrics = status["epochs"][-1]["validation"]
    payload = {
        "epoch": STAGE2_EPOCHS,
        "seed": int(seed),
        "arm": C4_ARM,
        "state_dict_classifier": {
            key: value.detach().cpu().clone()
            for key, value in student.module.state_dict().items()
        },
        "clean_stage1_sha256": manifest["clean_stage1_sha256"],
        "initial_classifier_sha256": initial_student_hash,
        "teacher_arm": TEACHER_ARM,
        "teacher_checkpoint_sha256": teacher_sha,
        "teacher_temperature": DISTILL_TEMPERATURE,
        "head_preservation_lambda": HEAD_PRESERVE_LAMBDA,
        "final_validation": final_metrics,
        "primary_epoch_fixed": True,
        "official_TEST_used": False,
        "backbone_retrained": False,
    }
    torch.save(payload, out / "final_classifier.pth")
    status["state"] = "completed"
    status["completed_at"] = stamp()
    status["final_epoch"] = STAGE2_EPOCHS
    status["final_validation"] = final_metrics
    status["final_classifier_sha256"] = sha256_file(out / "final_classifier.pth")
    status["official_TEST_used"] = False
    status["teacher_bitwise_unchanged"] = True
    write_json(out / "status.json", status)

    del backbone, student, teacher, optimizer
    torch.cuda.empty_cache()
    return status


def main():
    parser = argparse.ArgumentParser(description="C4: CC-SAM plus Many-only head-preservation distillation")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--smoke-only", action="store_true")
    mode.add_argument("--formal", action="store_true")
    args = parser.parse_args()

    manifest_path = ROOT / "prepared" / "prepared_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Sealed prepared manifest is missing")
    preliminary = read_json(manifest_path)
    if preliminary.get("official_TEST_used") is not False:
        raise RuntimeError("Prepared manifest does not certify TEST-free preparation")
    data_root = Path(preliminary["official_train_pickle"]).resolve().parent.parent
    install_test_guard(data_root)

    manifest, train_images, train_labels, validation_gap, validation_labels = __import__(
        "train_ccsam"
    ).load_prepared()
    train_transform, _, transform_sha = build_transforms()
    if transform_sha != manifest["transform_source_sha256"]:
        raise RuntimeError("Audited CIFAR augmentation source changed after preparation")

    preflight = collect_preflight(manifest)
    preflight_path = AB_DIR / "preflight_report.json"

    if args.preflight:
        if preflight_path.exists():
            raise FileExistsError("C4 preflight report already exists; do not overwrite")
        exclusive_json(preflight_path, preflight)
        print("C4 PREFLIGHT PASS; original C0-C3 sealed and all three C2 teachers available.")
        return

    if not preflight_path.is_file():
        raise FileNotFoundError("Run --preflight first")
    saved_preflight = read_json(preflight_path)
    saved_comparable = {key: value for key, value in saved_preflight.items() if key != "checked_at"}
    current_comparable = {key: value for key, value in preflight.items() if key != "checked_at"}
    if saved_comparable != current_comparable:
        raise RuntimeError("C4 preflight snapshot changed; stop rather than silently refresh inputs")

    if args.smoke_only:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; no CPU fallback")
        if (AB_DIR / "smoke_report.json").exists():
            raise FileExistsError("C4 smoke report already exists; never overwrite or rerun")
        smoke_all(manifest, train_images, train_labels, train_transform, torch.device("cuda:0"), preflight)
        print("C4 SMOKE PASS for all three seeds; formal training has not started.")
        return

    smoke_path = AB_DIR / "smoke_report.json"
    if not smoke_path.is_file():
        raise FileNotFoundError("Run --smoke-only successfully before formal training")
    smoke = read_json(smoke_path)
    expected_teacher_hashes = preflight["teacher_checkpoint_sha256"]
    smoke_teacher_hashes = {
        str(item["seed"]): item["teacher_checkpoint_sha256"]
        for item in smoke.get("checks", [])
    }
    if (
        smoke.get("passed") is not True
        or smoke.get("arm") != C4_ARM
        or smoke.get("source_sha256") != sha256_file(AB_DIR / "train_ab.py")
        or smoke.get("protocol_sha256") != sha256_file(AB_DIR / "AB_PROTOCOL.md")
        or smoke.get("prepared_manifest_sha256") != sha256_file(manifest_path)
        or smoke.get("preflight_report_sha256") != sha256_file(preflight_path)
        or len(smoke.get("checks", [])) != len(SEEDS)
        or [item.get("seed") for item in smoke.get("checks", [])] != list(SEEDS)
        or any(item.get("paired_with_original_c1_batches") is not True
               or item.get("disposable_student_update") is not True
               or item.get("teacher_frozen") is not True
               or item.get("backbone_and_bn_bitwise_unchanged") is not True
               for item in smoke.get("checks", []))
        or smoke_teacher_hashes != expected_teacher_hashes
        or smoke.get("official_TEST_used") is not False
    ):
        raise RuntimeError("C4 smoke report/source/teacher inputs are invalid")
    if (AB_DIR / "formal_completed.json").exists():
        raise FileExistsError("C4 formal completion marker already exists")
    for seed in SEEDS:
        destination = AB_DIR / "results" / f"seed_{seed}" / C4_ARM
        if destination.exists():
            raise FileExistsError(f"C4 output already exists: {destination}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no CPU fallback")
    compare_protected_inputs(preflight)

    completed = []
    for seed in SEEDS:
        status = train_one_seed(
            seed, manifest, train_images, train_labels, validation_gap,
            validation_labels, train_transform, torch.device("cuda:0"),
            expected_teacher_hashes[str(seed)],
        )
        if status.get("state") != "completed" or status.get("final_epoch") != STAGE2_EPOCHS:
            raise RuntimeError(f"C4 did not complete 30 epochs for seed {seed}")
        completed.append({
            "seed": seed,
            "epoch": status["final_epoch"],
            "final_classifier_sha256": status["final_classifier_sha256"],
            "teacher_checkpoint_sha256": status["teacher_checkpoint_sha256"],
        })
        compare_protected_inputs(preflight)

    guard = guard_pass()
    if guard.get("official_TEST_used") or guard.get("blocked_attempts"):
        raise RuntimeError("Official TEST guard violation during C4 formal training")
    compare_protected_inputs(preflight)
    result = {
        "passed": True,
        "arm": C4_ARM,
        "completed_at": stamp(),
        "seeds": list(SEEDS),
        "epochs_per_seed": STAGE2_EPOCHS,
        "new_epochs": len(SEEDS) * STAGE2_EPOCHS,
        "optimizer_steps": len(SEEDS) * STAGE2_EPOCHS * 170,
        "only_c4_new_training": True,
        "original_c0_c1_c2_and_c3_unchanged": True,
        "runs": completed,
        "teacher_temperature": DISTILL_TEMPERATURE,
        "head_preservation_lambda": HEAD_PRESERVE_LAMBDA,
        "official_TEST_used": False,
        "test_guard": guard,
        "preflight_report_sha256": sha256_file(preflight_path),
        "smoke_report_sha256": sha256_file(smoke_path),
        "source_sha256": sha256_file(AB_DIR / "train_ab.py"),
        "protocol_sha256": sha256_file(AB_DIR / "AB_PROTOCOL.md"),
    }
    exclusive_json(AB_DIR / "formal_completed.json", result)
    print("C4 FORMAL TRAINING COMPLETED: 3 seeds x 30 epochs; C0-C3 remain unchanged.")


if __name__ == "__main__":
    main()
