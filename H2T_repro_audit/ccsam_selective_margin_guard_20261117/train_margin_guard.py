"""Train and evaluate the fixed D3 selective margin-guard extension.

This script reuses the completed CC-SAM D0/D1/D2 cross-dataset run. It never
re-trains Stage I, re-extracts features, regenerates sampling plans, or opens
official TEST data before the dedicated evaluate phase.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BASE_DIR = HERE.parent / "ccsam_cross_dataset_formal_20261112"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from cross_dataset_core import class_weights_from_counts, compute_method_loss
from cross_dataset_models import assert_encoder_frozen, freeze_encoder, state_dict_sha256
from experiment_common import (
    OUTPUT_ROOT, REPO as EXPERIMENT_REPO, METRICS, cuda_device, make_sgd, now,
    read_json, record_run, require_complete, run_log, run_path, save_status,
    sha256_file, write_json,
)
from extract_features import open_cache
from margin_guard_core import selective_many_tail_margin_guard
from prepare_experiment import PREFLIGHT, open_plan
from train_stage1 import CHECKPOINT, finite_gradients, load_stage1, stage1_status
from train_stage2 import (
    D0, D1, D2, HEAD, assert_teacher_frozen, head_status, load_head, new_head,
    pairing_binding, save_head,
)

SMG_CONFIG_PATH = HERE / "smg_run_protocol.json"
SMG_CONFIG = json.loads(SMG_CONFIG_PATH.read_text(encoding="utf-8"))
BASELINE_ATTEMPT = SMG_CONFIG["baseline_attempt"]
EXTENSION_NAME = SMG_CONFIG["extension_attempt"]
EXTENSION_ROOT = HERE / "runs" / EXTENSION_NAME
PREFLIGHT_OUT = EXTENSION_ROOT / "preflight"
SMOKE_OUT = EXTENSION_ROOT / "smoke"
FORMAL_OUT = EXTENSION_ROOT / "formal"
FINAL_OUT = EXTENSION_ROOT / "final_evaluation"
ANALYSIS_OUT = EXTENSION_ROOT / "analysis"
PROTECTED_FILE = PREFLIGHT_OUT / "protected_artifacts.json"
BASELINE_ARMS = (D0, D1, D2)
NEW_ARM = "D3"
SMG_SOURCE_FILES = (
    HERE / "train_margin_guard.py",
    HERE / "margin_guard_core.py",
    SMG_CONFIG_PATH,
)
GUARD = SMG_CONFIG["selective_margin_guard"]
MANY_THRESHOLD = int(GUARD["many_threshold"])
RETAINED_FRACTION = float(GUARD["retained_margin_fraction"])
GUARD_EPS = float(GUARD["epsilon"])
GUARD_COEFFICIENT = float(GUARD["coefficient"])


def smg_source_hashes() -> dict[str, str]:
    return {
        path.relative_to(REPO).as_posix(): sha256_file(path)
        for path in SMG_SOURCE_FILES
    }


def require_runtime_contract(exp) -> None:
    if EXPERIMENT_REPO.resolve() != REPO.resolve():
        raise RuntimeError("The CC-SAM extension and base experiment are not in the same repository")
    if OUTPUT_ROOT.name != BASELINE_ATTEMPT:
        raise RuntimeError(
            f"Set CCSAM_CROSS_DATASET_ATTEMPT={BASELINE_ATTEMPT!r}; "
            f"observed output root {OUTPUT_ROOT}"
        )
    if tuple(exp.arms) != ("D0", "D1", "D2", "D3"):
        raise RuntimeError(f"Expected frozen arms D0/D1/D2/D3, observed {exp.arms}")
    if exp.dirty_sources:
        raise RuntimeError(
            "Base experiment has uncommitted source changes; commit or restore them before SMG:\n"
            + "\n".join(exp.dirty_sources)
        )
    if not SMG_CONFIG_PATH.is_file():
        raise FileNotFoundError(SMG_CONFIG_PATH)


def load_existing_prepared() -> dict:
    """Read the accepted old preflight without applying its obsolete source-hash gate."""
    status = require_complete(
        PREFLIGHT, phase="preflight", expected={"official_test_access": False}
    )
    manifest_path = PREFLIGHT / "prepared_manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("official_test_access") is not False:
        raise RuntimeError("The baseline prepared manifest does not certify TEST isolation")
    if manifest.get("pinned_sampler_sha256") != SMG_CONFIG["official_sampler_sha256"]:
        raise RuntimeError("The baseline manifest uses a different pinned sampler")
    if set(manifest.get("datasets", {})) != {"imagenet_lt", "places_lt"}:
        raise RuntimeError("The baseline manifest does not contain exactly both target datasets")
    if "prepared_manifest.json" not in status.get("artifacts", {}):
        raise RuntimeError("The baseline preflight does not hash prepared_manifest.json")
    return manifest


def _protect_status(protected: dict[str, str], directory: Path, status: dict) -> None:
    for relative, digest in status.get("artifacts", {}).items():
        path = directory / relative
        protected[path.relative_to(REPO).as_posix()] = digest
    status_path = directory / "status.json"
    protected[status_path.relative_to(REPO).as_posix()] = sha256_file(status_path)


def _read_trace_hashes(directory: Path, epochs: int) -> list[str]:
    path = directory / "sampling_trace.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != epochs:
        raise RuntimeError(f"Expected {epochs} sampling trace rows in {path}, found {len(rows)}")
    hashes = []
    for index, row in enumerate(rows):
        if int(row["epoch"]) != index + 1:
            raise RuntimeError(f"Nonsequential epoch in sampling trace: {path}")
        hashes.append(row["indices_sha256"])
    return hashes


def _validate_job(exp, prepared: dict, name: str, seed: int, device: torch.device):
    """Validate and load one paired baseline job; no TEST object is constructed."""
    seal = prepared["datasets"][name]
    stage1 = stage1_status(exp, name, seed, seal)
    encoder, initial_classifier, _ = load_stage1(exp, name, seed, seal, stage1)
    encoder = encoder.to(device)
    encoder_snapshot = freeze_encoder(encoder)
    initial_state = {
        key: value.detach().cpu().clone()
        for key, value in initial_classifier.state_dict().items()
    }

    features, labels, cache = open_cache(exp, name, seed, seal, stage1)
    plan, plan_record = open_plan(seal, seed, exp.protocol["stage2"]["epochs"])
    binding = pairing_binding(exp, name, seed, seal, stage1, cache, plan_record)
    expected_trace = plan_record["per_epoch_indices_sha256"]
    baseline_statuses = {}
    teacher = None

    for arm in BASELINE_ARMS:
        directory = run_path("stage2", name, seed, arm)
        status = head_status(directory, binding, arm, exp.protocol["stage2"]["epochs"])
        if status.get("epoch_indices_sha256") != expected_trace:
            raise RuntimeError(f"{arm} did not consume the sealed plan for {name} seed {seed}")
        if _read_trace_hashes(directory, exp.protocol["stage2"]["epochs"]) != expected_trace:
            raise RuntimeError(f"{arm} sampling_trace.csv differs from the sealed plan")
        if status.get("frozen_encoder_and_bn_unchanged") is not True:
            raise RuntimeError(f"{arm} does not certify a frozen encoder/BN state")
        if arm == D2:
            if (
                status.get("teacher_checkpoint_sha256") != baseline_statuses[D0]["artifacts"][HEAD]
                or status.get("teacher_frozen_and_unchanged") is not True
            ):
                raise RuntimeError(f"D2 teacher provenance mismatch for {name} seed {seed}")

        loaded = load_head(
            directory, status, binding, arm, exp.protocol["stage2"]["epochs"],
            device, frozen=(arm == D0),
        )
        if arm == D0:
            teacher = loaded
        else:
            del loaded
        baseline_statuses[arm] = status

    if teacher is None:
        raise RuntimeError("Matching D0 teacher was not loaded")
    teacher_state_sha = baseline_statuses[D0]["classifier_state_sha256"]
    assert_teacher_frozen(teacher, teacher_state_sha)
    assert_encoder_frozen(encoder, encoder_snapshot)

    counts = np.asarray(seal["train_class_counts"], dtype=np.int64)
    return {
        "seal": seal,
        "stage1": stage1,
        "encoder": encoder,
        "encoder_snapshot": encoder_snapshot,
        "initial_state": initial_state,
        "features": features,
        "labels": labels,
        "cache": cache,
        "plan": plan,
        "plan_record": plan_record,
        "binding": binding,
        "counts": counts,
        "teacher": teacher,
        "teacher_state_sha": teacher_state_sha,
        "teacher_checkpoint_sha": baseline_statuses[D0]["artifacts"][HEAD],
        "baseline_statuses": baseline_statuses,
    }


def _collect_protected_artifacts(exp, prepared: dict) -> dict[str, str]:
    protected: dict[str, str] = {}
    baseline_preflight = require_complete(
        PREFLIGHT, phase="preflight", expected={"official_test_access": False}
    )
    _protect_status(protected, PREFLIGHT, baseline_preflight)

    for name, seed in exp.jobs():
        job = _validate_job(exp, prepared, name, seed, torch.device("cpu"))
        stage1_dir = run_path("stage1", name, seed)
        cache_dir = run_path("features", name, seed)
        _protect_status(protected, stage1_dir, job["stage1"])
        _protect_status(protected, cache_dir, job["cache"])
        for arm in BASELINE_ARMS:
            _protect_status(
                protected, run_path("stage2", name, seed, arm),
                job["baseline_statuses"][arm],
            )
        del job
    return protected


def verify_protected_artifacts(protected: dict[str, str]) -> None:
    for relative, expected_sha in protected.items():
        path = REPO / relative
        if not path.is_file():
            raise RuntimeError(f"Protected baseline artifact disappeared: {relative}")
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"Protected baseline artifact changed: {relative}; "
                f"expected {expected_sha}, observed {actual_sha}"
            )


def combine_d3_objective(
    d1_loss: torch.Tensor,
    student_clean_logits: torch.Tensor,
    teacher_clean_logits: torch.Tensor,
    labels: torch.Tensor,
    train_counts,
):
    """Add only the fixed selective margin guard to the unchanged D1 objective."""
    guard_loss, diagnostics = selective_many_tail_margin_guard(
        student_clean_logits,
        teacher_clean_logits,
        labels,
        train_counts,
        many_threshold=MANY_THRESHOLD,
        retained_fraction=RETAINED_FRACTION,
        eps=GUARD_EPS,
    )
    total_loss = d1_loss + GUARD_COEFFICIENT * guard_loss
    if not bool(torch.isfinite(total_loss)):
        raise RuntimeError("Nonfinite D3 total loss")
    return total_loss, guard_loss, diagnostics


def _d3_batch_loss(job: dict, student, x, y, weights, counts_tensor, epoch_index: int):
    d1_loss, d1_diagnostics = compute_method_loss(
        "D1", x, y, student, weights, epoch_index, counts_tensor, teacher=None
    )
    student_clean_logits = student(x.detach())
    with torch.no_grad():
        teacher_clean_logits = job["teacher"](x.detach())
    total_loss, guard_loss, guard_diagnostics = combine_d3_objective(
        d1_loss, student_clean_logits, teacher_clean_logits, y, counts_tensor
    )
    return total_loss, d1_loss, guard_loss, d1_diagnostics, guard_diagnostics


def _require_certificate(exp, directory: Path, phase: str) -> dict:
    status = require_complete(
        directory, phase=phase, expected={"official_test_access": False}
    )
    config = read_json(directory / "config.json")
    if config.get("git_head") != exp.head:
        raise RuntimeError(f"{phase} certificate belongs to a different Git HEAD")
    if config.get("source_sha256") != exp.source_hashes:
        raise RuntimeError(f"{phase} certificate has different base source hashes")
    if config.get("smg_source_sha256") != smg_source_hashes():
        raise RuntimeError(f"{phase} certificate has different SMG source hashes")
    return status


def run_preflight(exp) -> None:
    if PREFLIGHT_OUT.exists():
        raise FileExistsError(f"Refusing to overwrite SMG preflight: {PREFLIGHT_OUT}")
    if FINAL_OUT.exists() or ANALYSIS_OUT.exists() or SMOKE_OUT.exists() or FORMAL_OUT.exists():
        raise FileExistsError("A later SMG phase already exists; refusing an implicit restart")
    for name, seed in exp.jobs():
        output = run_path("stage2", name, seed, NEW_ARM)
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite an existing D3 run: {output}")

    prepared = load_existing_prepared()
    config = exp.config(
        "smg-preflight",
        smg_source_sha256=smg_source_hashes(),
        smg_protocol=SMG_CONFIG,
        baseline_attempt=BASELINE_ATTEMPT,
    )
    with record_run(PREFLIGHT_OUT, config) as status:
        protected = _collect_protected_artifacts(exp, prepared)
        write_json(PROTECTED_FILE, protected)
        status.update(
            baseline_attempt=BASELINE_ATTEMPT,
            baseline_stage1_verified=len(list(exp.jobs())),
            baseline_feature_caches_verified=len(list(exp.jobs())),
            baseline_stage2_verified=len(list(exp.jobs())) * len(BASELINE_ARMS),
            d3_output_directories_absent=True,
            protected_artifact_count=len(protected),
            prepared_manifest_sha256=sha256_file(PREFLIGHT / "prepared_manifest.json"),
            smg_source_sha256=smg_source_hashes(),
        )
        run_log(PREFLIGHT_OUT, "All six Stage-I/cache/plan bindings and 18 D0/D1/D2 checkpoints verified")
        run_log(PREFLIGHT_OUT, f"Protected baseline file count: {len(protected)}")


def _require_smg_phase(exp, directory: Path, phase: str) -> dict:
    return _require_certificate(exp, directory, phase)


def run_smoke(exp, device: torch.device) -> None:
    _require_smg_phase(exp, PREFLIGHT_OUT, "smg-preflight")
    if SMOKE_OUT.exists():
        raise FileExistsError(f"Refusing to overwrite SMG Smoke: {SMOKE_OUT}")
    prepared = load_existing_prepared()
    config = exp.config(
        "smg-smoke", smg_source_sha256=smg_source_hashes(),
        smg_protocol=SMG_CONFIG, disposable_batches_per_job=2,
    )
    settings = exp.protocol["stage2"]
    with record_run(SMOKE_OUT, config, device=device) as status:
        job_records = []
        for name, seed in exp.jobs():
            job = _validate_job(exp, prepared, name, seed, device)
            student = new_head(exp.datasets[name]["num_classes"], job["initial_state"], device)
            weights = class_weights_from_counts(job["counts"], device)
            counts_tensor = torch.as_tensor(job["counts"], dtype=torch.long, device=device)
            optimizer = make_sgd(student.parameters(), settings)
            teacher_before = state_dict_sha256(job["teacher"])
            batch_records = []

            for batch_index in range(2):
                start = batch_index * settings["batch_size"]
                stop = min(start + settings["batch_size"], len(job["labels"]))
                indices = np.asarray(job["plan"][0, start:stop], dtype=np.int32)
                if len(indices) == 0:
                    raise RuntimeError(f"Missing disposable TRAIN batch for {name} seed {seed}")
                x = torch.from_numpy(np.asarray(job["features"][indices], dtype=np.float32)).to(device)
                y = torch.from_numpy(np.asarray(job["labels"][indices], dtype=np.int64)).to(device)
                optimizer.zero_grad(set_to_none=True)
                total, d1_loss, guard_loss, _, guard_diag = _d3_batch_loss(
                    job, student, x, y, weights, counts_tensor, 0
                )
                total.backward()
                if not finite_gradients(student):
                    raise RuntimeError(f"Nonfinite D3 Smoke gradient: {name} seed {seed}")
                optimizer.step()
                if any(not bool(torch.isfinite(p).all()) for p in student.parameters()):
                    raise RuntimeError(f"Nonfinite D3 Smoke parameter: {name} seed {seed}")
                assert_teacher_frozen(job["teacher"], job["teacher_state_sha"])
                assert_encoder_frozen(job["encoder"], job["encoder_snapshot"])
                batch_records.append({
                    "batch": batch_index + 1,
                    "indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
                    "samples": len(indices),
                    "d1_loss": float(d1_loss.detach().cpu()),
                    "guard_loss": float(guard_loss.detach().cpu()),
                    "total_loss": float(total.detach().cpu()),
                    "eligible_many_inputs": guard_diag["eligible_many_inputs"],
                    "active_guard_inputs": guard_diag["active_guard_inputs"],
                })

            if state_dict_sha256(job["teacher"]) != teacher_before:
                raise RuntimeError(f"D0 teacher changed during Smoke: {name} seed {seed}")
            job_records.append({
                "dataset": name, "seed": seed,
                "plan_sha256": job["plan_record"]["sha256"],
                "d1_epoch1_indices_sha256": job["baseline_statuses"][D1]["epoch_indices_sha256"][0],
                "batches": batch_records,
                "teacher_frozen": True,
                "encoder_and_bn_frozen": True,
            })
            run_log(SMOKE_OUT, f"{name} seed {seed}: two paired real TRAIN batches passed")
            del job, student, optimizer
        status.update(
            jobs=job_records, jobs_completed=len(job_records),
            disposable_batches_completed=sum(len(item["batches"]) for item in job_records),
            official_test_access=False,
        )


def _write_epoch_row(writer, row: dict) -> None:
    writer.writerow(row)


def train_d3_job(exp, prepared: dict, name: str, seed: int, device: torch.device) -> None:
    job = _validate_job(exp, prepared, name, seed, device)
    directory = run_path("stage2", name, seed, NEW_ARM)
    if directory.exists():
        raise FileExistsError(f"Refusing to overwrite or resume D3 output: {directory}")

    settings = exp.protocol["stage2"]
    epochs = int(settings["epochs"])
    batch_size = int(settings["batch_size"])
    features, labels, plan = job["features"], job["labels"], job["plan"]
    binding = job["binding"]
    teacher = job["teacher"]
    teacher_state_sha = job["teacher_state_sha"]
    teacher_checkpoint_sha = job["teacher_checkpoint_sha"]
    teacher_before = state_dict_sha256(teacher)
    student = new_head(exp.datasets[name]["num_classes"], job["initial_state"], device)
    if state_dict_sha256(student) != binding["initial_classifier_sha256"]:
        raise RuntimeError("D3 did not start from the original Stage-I classifier")
    optimizer = make_sgd(student.parameters(), settings)
    weights = class_weights_from_counts(job["counts"], device)
    counts_tensor = torch.as_tensor(job["counts"], dtype=torch.long, device=device)
    d1_trace = job["baseline_statuses"][D1]["epoch_indices_sha256"]

    config = exp.config(
        "stage2", **binding, arm=NEW_ARM,
        teacher_checkpoint_sha256=teacher_checkpoint_sha,
        train_class_counts=job["counts"].tolist(),
        smg_source_sha256=smg_source_hashes(),
        smg_protocol=SMG_CONFIG,
    )
    with record_run(directory, config, device=device) as status:
        status.update(
            binding, arm=NEW_ARM, epochs_completed=0, optimizer_steps=0,
            epoch_indices_sha256=[], teacher_checkpoint_sha256=teacher_checkpoint_sha,
            initial_classifier_sha256=binding["initial_classifier_sha256"],
            official_test_access=False,
        )
        metrics_path = directory / "epoch_metrics.csv"
        trace_path = directory / "sampling_trace.csv"
        metric_fields = [
            "epoch", "alpha", "d1_loss", "guard_loss", "total_loss", "delta_mean_l2",
            "many_inputs", "eligible_many_inputs", "active_guard_inputs",
            "eligible_fraction_of_many", "active_fraction_of_eligible",
            "teacher_margin_mean", "student_margin_mean",
            "student_to_teacher_margin_ratio_mean", "optimizer_steps", "samples",
        ]
        with metrics_path.open("x", encoding="utf-8", newline="") as metrics_handle, \
             trace_path.open("x", encoding="utf-8", newline="") as trace_handle:
            metric_writer = csv.DictWriter(metrics_handle, fieldnames=metric_fields)
            trace_writer = csv.DictWriter(
                trace_handle, fieldnames=["epoch", "steps", "samples", "indices_sha256"]
            )
            metric_writer.writeheader()
            trace_writer.writeheader()

            for epoch_index, planned in enumerate(plan):
                digest = hashlib.sha256()
                samples = steps = 0
                many_total = eligible_total = active_total = 0
                d1_sum = total_sum = delta_sum = guard_sum = 0.0
                teacher_margin_sum = student_margin_sum = ratio_sum = 0.0
                status["current_epoch"] = epoch_index + 1

                for start in range(0, len(labels), batch_size):
                    status["current_batch"] = steps + 1
                    indices = np.asarray(planned[start:start + batch_size], dtype=np.int32)
                    digest.update(indices.tobytes())
                    x = torch.from_numpy(np.asarray(features[indices], dtype=np.float32)).to(
                        device, non_blocking=True
                    )
                    y = torch.from_numpy(np.asarray(labels[indices], dtype=np.int64)).to(
                        device, non_blocking=True
                    )

                    optimizer.zero_grad(set_to_none=True)
                    total, d1_loss, guard_loss, d1_diag, guard_diag = _d3_batch_loss(
                        job, student, x, y, weights, counts_tensor, epoch_index
                    )
                    total.backward()
                    if not finite_gradients(student):
                        raise RuntimeError(
                            f"D3 received missing/nonfinite classifier gradients: {name} seed {seed}"
                        )
                    optimizer.step()
                    if any(not bool(torch.isfinite(parameter).all()) for parameter in student.parameters()):
                        raise RuntimeError(f"D3 produced nonfinite classifier parameters: {name} seed {seed}")

                    size = len(indices)
                    samples += size
                    steps += 1
                    status["optimizer_steps"] += 1
                    many = int(guard_diag["many_inputs"])
                    eligible = int(guard_diag["eligible_many_inputs"])
                    active = int(guard_diag["active_guard_inputs"])
                    many_total += many
                    eligible_total += eligible
                    active_total += active
                    d1_sum += float(d1_loss.detach().cpu()) * size
                    total_sum += float(total.detach().cpu()) * size
                    delta_sum += float(d1_diag["delta_mean_l2"]) * size
                    guard_sum += float(guard_loss.detach().cpu()) * eligible
                    teacher_margin_sum += float(guard_diag["teacher_margin_mean"]) * eligible
                    student_margin_sum += float(guard_diag["student_margin_mean"]) * eligible
                    ratio_sum += float(guard_diag["student_to_teacher_margin_ratio_mean"]) * eligible

                    assert_teacher_frozen(teacher, teacher_state_sha)
                    assert_encoder_frozen(job["encoder"], job["encoder_snapshot"])

                trace_hash = digest.hexdigest()
                expected_hash = job["plan_record"]["per_epoch_indices_sha256"][epoch_index]
                if trace_hash != expected_hash or trace_hash != d1_trace[epoch_index]:
                    raise RuntimeError(
                        f"D3 sampling sequence differs from the sealed plan/D1: {name} seed {seed} "
                        f"epoch {epoch_index + 1}"
                    )
                if samples != len(labels):
                    raise RuntimeError(f"D3 consumed {samples} of {len(labels)} samples")

                eligible_denominator = max(eligible_total, 1)
                many_denominator = max(many_total, 1)
                row = {
                    "epoch": epoch_index + 1,
                    "alpha": min((epoch_index + 1) / 30.0, 0.3),
                    "d1_loss": d1_sum / samples,
                    "guard_loss": guard_sum / eligible_denominator,
                    "total_loss": total_sum / samples,
                    "delta_mean_l2": delta_sum / samples,
                    "many_inputs": many_total,
                    "eligible_many_inputs": eligible_total,
                    "active_guard_inputs": active_total,
                    "eligible_fraction_of_many": eligible_total / many_denominator,
                    "active_fraction_of_eligible": active_total / eligible_denominator,
                    "teacher_margin_mean": teacher_margin_sum / eligible_denominator,
                    "student_margin_mean": student_margin_sum / eligible_denominator,
                    "student_to_teacher_margin_ratio_mean": ratio_sum / eligible_denominator,
                    "optimizer_steps": steps,
                    "samples": samples,
                }
                _write_epoch_row(metric_writer, row)
                trace_writer.writerow({
                    "epoch": epoch_index + 1, "steps": steps, "samples": samples,
                    "indices_sha256": trace_hash,
                })
                metrics_handle.flush()
                trace_handle.flush()
                status["epoch_indices_sha256"].append(trace_hash)
                status.update(
                    epochs_completed=epoch_index + 1,
                    current_epoch=epoch_index + 1,
                    last_epoch_metrics=row,
                )
                save_status(directory, status)
                run_log(
                    directory,
                    f"{name} seed {seed} D3 epoch {epoch_index + 1}/{epochs} "
                    f"D1={row['d1_loss']:.6f} SMG={row['guard_loss']:.6f} "
                    f"Many-active={active_total}/{eligible_total}",
                )

        assert_teacher_frozen(teacher, teacher_state_sha)
        if state_dict_sha256(teacher) != teacher_before:
            raise RuntimeError("D0 teacher changed during D3 formal training")
        assert_encoder_frozen(job["encoder"], job["encoder_snapshot"])
        state_sha = save_head(
            directory / HEAD, student, binding, NEW_ARM, epochs, teacher_checkpoint_sha
        )
        status.update(
            final_epoch=epochs,
            epochs_completed=epochs,
            classifier_state_sha256=state_sha,
            teacher_checkpoint_sha256=teacher_checkpoint_sha,
            frozen_encoder_and_bn_unchanged=True,
            teacher_frozen_and_unchanged=True,
            teacher_classifier_state_sha256=teacher_state_sha,
            official_test_access=False,
        )


def verify_all_d3(exp, prepared: dict) -> int:
    verified = 0
    for name, seed in exp.jobs():
        seal = prepared["datasets"][name]
        stage1 = stage1_status(exp, name, seed, seal)
        _, _, cache = open_cache(exp, name, seed, seal, stage1)
        _, plan_record = open_plan(seal, seed, exp.protocol["stage2"]["epochs"])
        binding = pairing_binding(
            exp, name, seed, seal, stage1, cache, plan_record
        )
        d0 = head_status(run_path("stage2", name, seed, D0), binding, D0, exp.protocol["stage2"]["epochs"])
        d3 = head_status(run_path("stage2", name, seed, NEW_ARM), binding, NEW_ARM, exp.protocol["stage2"]["epochs"])
        if d3.get("teacher_checkpoint_sha256") != d0["artifacts"][HEAD]:
            raise RuntimeError(f"D3 teacher checkpoint mismatch: {name} seed {seed}")
        if d3.get("teacher_classifier_state_sha256") != d0["classifier_state_sha256"]:
            raise RuntimeError(f"D3 teacher classifier-state mismatch: {name} seed {seed}")
        if d3.get("teacher_frozen_and_unchanged") is not True:
            raise RuntimeError(f"D3 did not certify a frozen teacher: {name} seed {seed}")
        if d3.get("epoch_indices_sha256") != plan_record["per_epoch_indices_sha256"]:
            raise RuntimeError(f"D3 sampling hashes differ from the sealed plan: {name} seed {seed}")
        if d3.get("frozen_encoder_and_bn_unchanged") is not True:
            raise RuntimeError(f"D3 did not certify a frozen encoder: {name} seed {seed}")
        verified += 1
    return verified


def run_formal(exp, device: torch.device) -> None:
    _require_smg_phase(exp, PREFLIGHT_OUT, "smg-preflight")
    _require_smg_phase(exp, SMOKE_OUT, "smg-smoke")
    if FORMAL_OUT.exists():
        raise FileExistsError(f"Refusing to overwrite SMG formal record: {FORMAL_OUT}")
    prepared = load_existing_prepared()
    protected = read_json(PROTECTED_FILE)
    verify_protected_artifacts(protected)

    config = exp.config(
        "smg-formal", smg_source_sha256=smg_source_hashes(),
        smg_protocol=SMG_CONFIG, new_stage2_runs=6,
    )
    with record_run(FORMAL_OUT, config, device=device) as status:
        completed = []
        for name, seed in exp.jobs():
            train_d3_job(exp, prepared, name, seed, device)
            completed.append({"dataset": name, "seed": seed, "arm": NEW_ARM})
            status.update(
                jobs_completed=len(completed), completed_jobs=completed,
                official_test_access=False,
            )
            save_status(FORMAL_OUT, status)
        verified = verify_all_d3(exp, prepared)
        verify_protected_artifacts(protected)
        write_json(FORMAL_OUT / "protected_artifacts_after.json", {
            "unchanged": True, "verified_file_count": len(protected),
            "protected_artifacts": protected,
        })
        status.update(
            jobs_completed=verified,
            stage1_reused=6,
            feature_caches_reused=6,
            baseline_stage2_reused=18,
            d3_stage2_completed=verified,
            stage2_total_verified=24,
            protected_baseline_unchanged=True,
            official_test_access=False,
        )
        run_log(FORMAL_OUT, "Six D3 runs completed; all 24 D0-D3 classifiers and protected baselines verified")


def _require_formal_complete(exp) -> dict:
    status = _require_smg_phase(exp, FORMAL_OUT, "smg-formal")
    if status.get("d3_stage2_completed") != 6 or status.get("protected_baseline_unchanged") is not True:
        raise RuntimeError("SMG formal certificate is incomplete")
    return status


def run_evaluate(exp, device: torch.device) -> None:
    _require_formal_complete(exp)
    prepared = load_existing_prepared()
    protected = read_json(PROTECTED_FILE)
    verify_protected_artifacts(protected)
    if FINAL_OUT.exists():
        raise FileExistsError(f"Refusing to overwrite official-test evaluation: {FINAL_OUT}")

    import evaluate_final as evaluator

    evaluator.FINAL = FINAL_OUT
    evaluator.evaluate(exp, prepared, device)
    final_status = require_complete(
        FINAL_OUT, phase="evaluate-final",
        expected={
            "official_test_access": True,
            "stage1_verified": 6,
            "stage2_verified": 24,
        },
    )
    for name, spec in exp.datasets.items():
        if final_status.get("test_splits", {}).get(name, {}).get("rows") != spec["test_images"]:
            raise RuntimeError(f"Final TEST row count mismatch for {name}")
    verify_protected_artifacts(protected)
    write_json(EXTENSION_ROOT / "protected_artifacts_after_evaluation.json", {
        "unchanged": True, "verified_file_count": len(protected),
        "protected_artifacts": protected,
    })


def _paired_values(dataset_summary: dict, contrast: str, metric: str) -> list[float]:
    return [
        float(row[metric]) for row in dataset_summary["paired_per_seed"]
        if row["contrast"] == contrast
    ]


def _guard_diagnostics(exp) -> dict:
    result = {}
    for name, seed in exp.jobs():
        path = run_path("stage2", name, seed, NEW_ARM) / "epoch_metrics.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        result[f"{name}/seed_{seed}"] = {
            "many_inputs": sum(int(row["many_inputs"]) for row in rows),
            "eligible_many_inputs": sum(int(row["eligible_many_inputs"]) for row in rows),
            "active_guard_inputs": sum(int(row["active_guard_inputs"]) for row in rows),
        }
    return result


def build_smg_report(exp, summary: dict) -> tuple[str, dict]:
    datasets = summary["datasets"]
    gates = {}
    sections = [
        "# CC-SAM D3: Selective Many-to-Tail Margin Guard results",
        "",
        "D3 is the unchanged D1 objective plus the fixed SMG penalty. Results use fixed epoch-30 classifiers and three paired seeds. No statistical-significance claim is made.",
        "",
        "## Paired mean differences (percentage points)",
        "",
        "| Dataset | Contrast | Overall | Many | Medium | Few |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, data in datasets.items():
        for contrast in ("D3-D0", "D3-D1", "D3-D2"):
            stats = data["paired_mean_sample_sd"][contrast]
            sections.append(
                f"| {name} | {contrast} | "
                + " | ".join(f"{stats[m]['mean']:+.3f} ± {stats[m]['sample_sd']:.3f}" for m in METRICS)
                + " |"
            )

        d3_d1_many = _paired_values(data, "D3-D1", "Many")
        d3_d0_few = _paired_values(data, "D3-D0", "Few")
        d3_d2_many = _paired_values(data, "D3-D2", "Many")
        d3_d0_overall = _paired_values(data, "D3-D0", "Overall")
        gates[name] = {
            "many_improves_over_d1_mean_and_at_least_2_of_3_seeds": (
                bool(d3_d1_many) and statistics.mean(d3_d1_many) > 0
                and sum(value > 0 for value in d3_d1_many) >= 2
            ),
            "few_gain_over_d0_in_at_least_2_of_3_seeds": (
                len(d3_d0_few) == 3 and sum(value > 0 for value in d3_d0_few) >= 2
            ),
            "many_recovery_over_d2_mean": bool(d3_d2_many) and statistics.mean(d3_d2_many) > 0,
            "overall_not_below_d0_by_more_than_0_2pp": (
                bool(d3_d0_overall) and statistics.mean(d3_d0_overall) >= -0.2
            ),
        }

    diagnostics = _guard_diagnostics(exp)
    totals = {
        key: sum(value[key] for value in diagnostics.values())
        for key in ("many_inputs", "eligible_many_inputs", "active_guard_inputs")
    }
    diagnostics_summary = {
        **totals,
        "eligible_fraction_of_many": totals["eligible_many_inputs"] / max(totals["many_inputs"], 1),
        "active_fraction_of_eligible": totals["active_guard_inputs"] / max(totals["eligible_many_inputs"], 1),
        "guard_active_on_measurable_subset": (
            totals["eligible_many_inputs"] > 0 and totals["active_guard_inputs"] > 0
        ),
    }
    gates["guard_diagnostics"] = diagnostics_summary

    sections.extend([
        "",
        "## Predeclared decision gates",
        "",
        "| Dataset | Many improves over D1 | Few gain over D0 retained | Many recovery over D2 | Overall floor |",
        "|---|---|---|---|---|",
    ])
    for name in datasets:
        g = gates[name]
        sections.append(
            f"| {name} | {'PASS' if g['many_improves_over_d1_mean_and_at_least_2_of_3_seeds'] else 'FAIL'} "
            f"| {'PASS' if g['few_gain_over_d0_in_at_least_2_of_3_seeds'] else 'FAIL'} "
            f"| {'PASS' if g['many_recovery_over_d2_mean'] else 'FAIL'} "
            f"| {'PASS' if g['overall_not_below_d0_by_more_than_0_2pp'] else 'FAIL'} |"
        )
    sections.extend([
        "",
        "## Guard activation",
        "",
        f"- Many inputs: {diagnostics_summary['many_inputs']:,}",
        f"- Eligible Many inputs: {diagnostics_summary['eligible_many_inputs']:,}",
        f"- Active guard inputs: {diagnostics_summary['active_guard_inputs']:,}",
        f"- Eligible / Many: {diagnostics_summary['eligible_fraction_of_many']:.4%}",
        f"- Active / eligible: {diagnostics_summary['active_fraction_of_eligible']:.4%}",
        f"- Guard active on a measurable subset: {diagnostics_summary['guard_active_on_measurable_subset']}",
        "",
        "A failed gate is a negative experimental result, not a reason to tune SMG on the same official TEST split. Any next experiment must be justified using the current evidence and a newly declared protocol.",
    ])
    return "\n".join(sections) + "\n", gates


def run_analyze(exp) -> None:
    _require_formal_complete(exp)
    final_status = require_complete(
        FINAL_OUT, phase="evaluate-final",
        expected={"official_test_access": True, "stage2_verified": 24},
    )
    if final_status.get("test_splits", {}).keys() != exp.datasets.keys():
        raise RuntimeError("Final evaluation does not contain both official test splits")
    if ANALYSIS_OUT.exists():
        raise FileExistsError(f"Refusing to overwrite SMG analysis: {ANALYSIS_OUT}")

    prepared = load_existing_prepared()
    import analyze_results as analyzer

    analyzer.FINAL = FINAL_OUT
    analyzer.ANALYSIS = ANALYSIS_OUT
    analyzer.analyze(exp, prepared)

    summary = read_json(ANALYSIS_OUT / "cross_dataset_summary.json")
    report, gates = build_smg_report(exp, summary)
    report_path = EXTENSION_ROOT / "SMG_RESULTS.md"
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite SMG result report: {report_path}")
    report_path.write_text(report, encoding="utf-8", newline="\n")
    write_json(EXTENSION_ROOT / "decision_gates.json", gates)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        required=True,
        choices=("preflight", "smoke", "formal", "evaluate", "analyze"),
    )
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    from experiment_common import load_experiment

    exp = load_experiment(args.workers)
    require_runtime_contract(exp)

    if args.phase == "preflight":
        run_preflight(exp)
    elif args.phase == "smoke":
        run_smoke(exp, cuda_device())
    elif args.phase == "formal":
        run_formal(exp, cuda_device())
    elif args.phase == "evaluate":
        run_evaluate(exp, cuda_device())
    elif args.phase == "analyze":
        run_analyze(exp)

    print(f"SMG phase {args.phase} completed at {now()}", flush=True)


if __name__ == "__main__":
    main()
