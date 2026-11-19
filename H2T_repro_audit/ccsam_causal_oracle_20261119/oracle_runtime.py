"""Read-only parent reuse, provenance, exclusive outputs, and numerical steps."""
from __future__ import annotations

import copy
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from causal_core import (
    ROOT, BASE, GRAD_ATOL, GRAD_RTOL, array_hash, canonical_hash,
    decompose_perturbation_gradient_by_source, deterministic_split,
    masked_optimizer_gradient,
)
from isolated_data import HoldoutVault, file_sha
from ccsam_common import (  # noqa: E402
    BATCH_SIZE, FEATURE_DIM, MOMENTUM, NUM_CLASSES, STAGE2_EPOCHS, STAGE2_LR,
    WEIGHT_DECAY, augment_batch, build_models, build_transforms,
    check_backbone_unchanged, class_weights_from_counts, clean_compatible_gap,
    guard_pass, install_test_guard, resolve_checkpoint, seed_everything,
    state_hash, validate_class_aware_plan,
)
from train_ccsam import C3_ARM, compute_arm_loss  # noqa: E402
from ccmia_core import ParameterWritebackRoundingLedger, SGDInfluenceTracker  # noqa: E402
from train_ccmia import (  # noqa: E402
    C2_ARM, CODE_PATHS, TRACE_FIELDS, check_trace_row, compute_paired_gradients,
    make_trace_row, reconstruction_diagnostics,
)

REPO = ROOT.parents[1]
PARENT = BASE / "ccmia"
BRANCH = "experiment/ccmia-causal-oracle-20261119"
START_HEAD = "398200c263fff2c6130bd1d12428b559af5c4a68"
PARENT_HEAD = "63ae0ec84ff6b6579d682aec230811effc7a0ae7"
SEEDS = (123, 124, 125)
NTRAIN = 10847
NBATCH = math.ceil(NTRAIN / BATCH_SIZE)
NSTEPS = NBATCH * STAGE2_EPOCHS


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def new_json(path, payload):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"Output outside experiment: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def new_csv(path, rows, fields=None):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("CSV output outside experiment")
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fields = fields or (list(rows[0]) if rows else [])
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def git(*args):
    result = subprocess.run(["git", *args], cwd=REPO, text=True,
                            capture_output=True, encoding="utf-8", check=True)
    return result.stdout.strip()


def code_hashes():
    paths = sorted(ROOT.glob("*.py")) + [ROOT / "CAUSAL_ORACLE_PROTOCOL.md",
                                       ROOT / "CODEX_CAUSAL_ORACLE_20261119.txt"]
    return {str(path.relative_to(ROOT)): file_sha(path) for path in paths}


def require_cpu():
    receipt = read_json(ROOT / "validation" / "CPU_VALIDATION.json")
    if receipt.get("passed") is not True or receipt.get("code_sha256") != code_hashes():
        raise RuntimeError("CPU syntax / real unit-test receipt missing, failed, or stale")
    return receipt


def environment():
    # CPU preflight does not initialize CUDA. Query device details only in the
    # GPU phase, after the CPU gate.
    result = {"python": sys.executable, "python_version": sys.version,
              "numpy": np.__version__, "torch": torch.__version__,
              "torch_cuda": torch.version.cuda, "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES"),
              "cudnn_benchmark": torch.backends.cudnn.benchmark,
              "cudnn_deterministic": torch.backends.cudnn.deterministic,
              "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
              "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
              "AMP": False, "model_dtype": "torch.float32"}
    if torch.cuda.is_initialized():
        result.update(device_index=torch.cuda.current_device(),
                      device_name=torch.cuda.get_device_name(torch.cuda.current_device()),
                      cudnn_version=torch.backends.cudnn.version())
    return result


def cuda_device():
    require_cpu()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no CPU fallback")
    torch.cuda.init()
    if torch.cuda.device_count() < 1:
        raise RuntimeError("No valid CUDA device")
    torch.cuda.set_device(0)
    torch.cuda.synchronize(0)
    return torch.device("cuda:0")


def _lf_hash(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def inspect_parent():
    if git("branch", "--show-current") != BRANCH:
        raise RuntimeError("Wrong experiment branch")
    git("merge-base", "--is-ancestor", START_HEAD, "HEAD")
    git("merge-base", "--is-ancestor", PARENT_HEAD, "HEAD")
    scope = str(ROOT.relative_to(REPO)).replace("\\", "/") + "/"
    for name in git("diff", "--name-only", "HEAD").splitlines():
        if not name.replace("\\", "/").startswith(scope):
            raise RuntimeError(f"Tracked change outside authorized scope: {name}")
    complete = read_json(PARENT / "formal_completed.json")
    if (complete.get("state") != "completed" or complete.get("paired_steps") != 15300
            or complete.get("epochs") != 30 or complete.get("seeds") != list(SEEDS)
            or complete.get("official_TEST_used") is not False):
        raise RuntimeError("Parent CC-MIA completion ledger is invalid")
    hashes = {}
    for path in CODE_PATHS:
        actual = file_sha(path)
        key = str(path.relative_to(BASE))
        expected = complete["code_sha256"].get(key)
        if actual != expected:
            raise RuntimeError(f"Parent source SHA differs from completed run: {path}: {actual} != {expected}")
        hashes[str(path)] = actual
    artifact_records = []
    manifest = PARENT / "ARTIFACT_SHA256.csv"
    for row in read_csv(manifest):
        path = PARENT / row["path"]
        if not path.is_file():
            raise FileNotFoundError(f"Parent artifact missing: {path}")
        raw, normalized = file_sha(path), None
        if raw != row["sha256_raw"]:
            normalized = _lf_hash(path)
            if not row.get("sha256_lf_normalized") or normalized != row["sha256_lf_normalized"]:
                raise RuntimeError(f"Parent artifact differs from raw and declared LF SHA: {path}")
        artifact_records.append({"path": str(path), "actual_sha256": raw,
                                 "historical_sha256": row["sha256_raw"],
                                 "declared_LF_only_conversion": normalized is not None})
        hashes[str(path)] = raw
    for path in (PARENT / "formal_completed.json", manifest):
        hashes[str(path)] = file_sha(path)
    # Transitive, local frozen model and transform implementation fingerprints.
    runtime = BASE.parent / "clean_vs_stability_runs" / "clean_s1_seed42_s2_124_125_20261014" / "runtime"
    for path in sorted(runtime.rglob("*.py")):
        hashes[str(path)] = file_sha(path)
    hashes[str(BASE / "official_reference" / "Samplers.py")] = file_sha(BASE / "official_reference" / "Samplers.py")
    schemas = {}
    for path in sorted((PARENT / "results" / "seed_123").glob("*.csv")):
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            columns = next(reader)
            n = sum(1 for _ in reader)
        schemas[path.name] = {"columns": columns, "rows": n, "bytes": path.stat().st_size}
    for seed in SEEDS:
        folder = PARENT / "results" / f"seed_{seed}"
        status = read_json(folder / "status.json")
        if (status.get("state") != "completed" or status.get("epochs_completed") != 30
                or status.get("paired_batch_steps_completed") != NSTEPS
                or status.get("backbone_and_bn_bitwise_unchanged") is not True):
            raise RuntimeError(f"Parent seed {seed} incomplete")
        for arm in ("C2", "C3"):
            item = status["checkpoints"][arm]
            path = folder / item["path"]
            if file_sha(path) != item["sha256"]:
                raise RuntimeError(f"Parent checkpoint SHA mismatch: {path}")
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            if checkpoint.get("epoch") != 30 or checkpoint.get("official_TEST_used") is not False:
                raise RuntimeError("Parent checkpoint provenance invalid")
            if checkpoint.get("seed") != seed:
                raise RuntimeError("Parent checkpoint seed invalid")
    return {"passed": True, "parent_commit": PARENT_HEAD, "start_head": START_HEAD,
            "current_head": git("rev-parse", "HEAD"), "branch": BRANCH,
            "reused_source_and_artifact_sha256": hashes,
            "parent_artifact_checks": artifact_records, "actual_csv_schemas": schemas,
            "parent_epochs": 30, "parent_AMP": False,
            "historical_rankings_used_for_mask_selection": False}


def load_context(split=None):
    prepared = BASE / "prepared"
    manifest_path = prepared / "prepared_manifest.json"
    manifest = read_json(manifest_path)
    parent = read_json(PARENT / "formal_completed.json")
    if file_sha(manifest_path) != parent["prepared_manifest_sha256"]:
        raise RuntimeError("Prepared manifest changed since parent completion")
    data_root = Path(manifest["official_train_pickle"]).parent.parent
    install_test_guard(data_root)
    if (manifest.get("official_TEST_used") is not False
            or manifest.get("official_TEST_images_or_labels_loaded") != 0
            or manifest.get("CIFAR100_train_false_calls") != 0
            or manifest.get("checkpoint_retrained") is not False):
        raise RuntimeError("Prepared provenance is not sealed / TEST-free")
    checkpoint = resolve_checkpoint(manifest["checkpoint_path"])
    split_path = Path(manifest["split_path"])
    if file_sha(split_path) != manifest["split_sha256"]:
        raise RuntimeError("Original TRAIN/holdout IDs changed")
    if file_sha(manifest["official_train_pickle"]) != manifest["official_train_pickle_sha256"]:
        raise RuntimeError("Official TRAIN pickle integrity failure")
    with np.load(split_path, allow_pickle=False) as original:
        train_raw = original["train_raw_indices"].astype(np.int64)
        holdout_raw = original["validation_raw_indices"].astype(np.int64)
    if len(train_raw) != NTRAIN or len(holdout_raw) != 1980 or np.intersect1d(train_raw, holdout_raw).size:
        raise RuntimeError("Original raw-ID overlap / size mismatch")
    if split is None:
        # Pre-existing cache allocation, not a read of Evaluation labels.
        # CAL validates it now; EVAL validates it only after all masks seal.
        allocation = np.repeat(np.arange(1, 100, dtype=np.int64), 20)
        split = deterministic_split(holdout_raw, allocation)
    if len(split["calibration_positions"]) != 990 or len(split["evaluation_positions"]) != 990:
        raise RuntimeError("Expected 990/990 development partition")
    vault = HoldoutVault(prepared, split, ROOT / "mask_seal.json")
    counts = np.asarray(manifest["train_class_counts"], dtype=np.int64)
    if counts.shape != (100,) or np.any(counts <= 0):
        raise RuntimeError("Bad sealed TRAIN counts")
    for name in ("train_images.npy", "train_labels.npy"):
        if file_sha(prepared / name) != manifest["prepared_array_actions"][name]["sha256"]:
            raise RuntimeError(f"Prepared TRAIN integrity failure: {name}")
    images = np.load(prepared / "train_images.npy", mmap_mode="r", allow_pickle=False)
    labels = np.load(prepared / "train_labels.npy", allow_pickle=False)
    if images.shape != (NTRAIN, 32, 32, 3) or images.dtype != np.uint8:
        raise RuntimeError("Unexpected TRAIN image array")
    if labels.shape != (NTRAIN,) or labels.dtype != np.int64:
        raise RuntimeError("Unexpected TRAIN label array")
    if not np.array_equal(np.bincount(labels, minlength=100), counts):
        raise RuntimeError("TRAIN counts changed")
    transform, _, transform_sha = build_transforms()
    if transform_sha != manifest["transform_source_sha256"]:
        raise RuntimeError("Transform source changed")
    cal_x, cal_y = vault.read("calibration")
    if cal_x.shape != (990, 64) or not np.array_equal(np.bincount(cal_y, minlength=100), np.r_[0, np.repeat(10, 99)]):
        raise RuntimeError("Calibration shape / allocation invalid")
    return {"manifest": manifest, "manifest_path": manifest_path, "prepared": prepared,
            "counts": counts, "train_images": images, "train_labels": labels,
            "train_transform": transform, "checkpoint": checkpoint,
            "split": split, "vault": vault, "cal_x": cal_x, "cal_y": cal_y}


def make_models(context, seed, device):
    seed_everything(seed, deterministic=True)
    backbone, head, _ = build_models(context["checkpoint"], device)
    frozen = {k: v.detach().cpu().clone() for k, v in backbone.state_dict().items()}
    expected = read_json(PARENT / "results" / f"seed_{seed}" / "status.json")["initial_classifier_sha256"]
    if state_hash(head) != expected:
        raise RuntimeError("Initial classifier differs from parent")
    return backbone, head, frozen


def optimizer_for(head):
    return torch.optim.SGD(head.module.fc.parameters(), lr=STAGE2_LR,
                           momentum=MOMENTUM, weight_decay=WEIGHT_DECAY, nesterov=False)


def real_batches(context, seed, backbone, device, epochs=30):
    plan = np.load(context["prepared"] / f"sampling_plan_seed{seed}.npy", mmap_mode="r", allow_pickle=False)
    reference = read_csv(BASE / "results" / f"seed_{seed}" / "C0_BALANCED_CE" / "batch_trace.csv")
    generator = torch.Generator(device="cpu").manual_seed(seed + 300009)
    for epoch in range(epochs):
        for batch in range(NBATCH):
            start, stop = batch * BATCH_SIZE, min((batch + 1) * BATCH_SIZE, NTRAIN)
            indices = np.asarray(plan[epoch, start:stop], dtype=np.int64)
            labels_np = np.asarray(context["train_labels"][indices], dtype=np.int64)
            images = augment_batch(context["train_images"], indices, context["train_transform"], generator)
            with torch.no_grad():
                gap = clean_compatible_gap(backbone(images.to(device)))
            row = make_trace_row(epoch, batch, indices, labels_np, images, gap)
            check_trace_row(row, reference[epoch * NBATCH + batch], seed, epoch, batch)
            yield epoch, batch, start, stop, gap, torch.from_numpy(labels_np).to(device), row


def _check_close(actual, reference, name):
    if not torch.isfinite(actual).all() or not torch.isfinite(reference).all():
        raise RuntimeError(f"Nonfinite {name}")
    error = float((actual - reference).abs().max().item())
    if not torch.allclose(actual, reference, rtol=GRAD_RTOL, atol=GRAD_ATOL):
        raise RuntimeError(f"Gradient reconstruction failed ({name}), max_abs={error}")
    return error


def intervention_step(head, arm, gap, labels, weights, counts, epoch, keep, optimizer, ledger):
    fc = head.module.fc
    parent_arm = C2_ARM if arm == "C2" else C3_ARM
    result = compute_arm_loss(parent_arm, gap, labels, fc, weights, epoch,
                              train_counts=counts if arm != "C2" else None)
    loss, clean_logits, _, _, delta, _, alpha = result
    head.zero_grad(set_to_none=True)
    g_original = torch.autograd.grad(loss, (fc.weight, fc.bias))
    errors = {"unmasked_weight_max_abs": 0.0, "unmasked_bias_max_abs": 0.0,
              "masked_weight_max_abs": 0.0, "masked_bias_max_abs": 0.0}
    if arm == "C2":
        gradients = g_original
    else:
        many = torch.as_tensor(counts, device=labels.device)[labels] > 100
        if torch.count_nonzero(delta[many]):
            raise RuntimeError("Many-input feature perturbation is nonzero")
        with torch.no_grad():
            adv_logits = fc(gap.detach() + delta)
            source_w, source_b = decompose_perturbation_gradient_by_source(
                gap, labels, delta, clean_logits.detach(), adv_logits, weights, alpha)
        clean_loss = compute_arm_loss(C2_ARM, gap, labels, fc, weights, epoch)[0]
        clean_g = torch.autograd.grad(clean_loss, (fc.weight, fc.bias))
        original_prediction = (clean_g[0] + source_w.sum(0), clean_g[1] + source_b.sum(0))
        errors["unmasked_weight_max_abs"] = _check_close(original_prediction[0], g_original[0], "original C3 weight")
        errors["unmasked_bias_max_abs"] = _check_close(original_prediction[1], g_original[1], "original C3 bias")
        if arm == "BASE_C3" or bool(keep.all().item()):
            # Preserve the original baseline (and K=0 identity) bitwise.
            gradients = g_original
        else:
            gradients = masked_optimizer_gradient(*clean_g, source_w, source_b, keep)
            removed_w = source_w.masked_fill(keep.unsqueeze(-1), 0).sum(0)
            removed_b = source_b.masked_fill(keep, 0).sum(0)
            errors["masked_weight_max_abs"] = _check_close(gradients[0], g_original[0] - removed_w, "masked C3 weight")
            errors["masked_bias_max_abs"] = _check_close(gradients[1], g_original[1] - removed_b, "masked C3 bias")
        if torch.count_nonzero(source_w[torch.as_tensor(counts, device=labels.device) > 100]):
            raise RuntimeError("Many source unexpectedly has direct gradient")
    before = {n: p.detach().clone() for n, p in fc.named_parameters()}
    for parameter, gradient in zip((fc.weight, fc.bias), gradients):
        if gradient.dtype != torch.float32 or not torch.isfinite(gradient).all():
            raise RuntimeError("Missing / nonfinite / non-FP32 optimizer gradient")
        parameter.grad = gradient.detach().clone()
    ledger.capture(fc.named_parameters(), optimizer)
    optimizer.step()
    ledger.record()  # independent bitwise replay, buffer check, rounding bound
    if not all(torch.isfinite(p).all() for p in fc.parameters()):
        raise RuntimeError("Nonfinite updated classifier")
    differences = {n: float((p.detach() - before[n]).abs().max().item()) for n, p in fc.named_parameters()}
    return {"loss": float(loss.detach().item()), "alpha": float(alpha),
            "gradient_checks": errors, "writeback_replay_bitwise_equal": True,
            "parameter_update_max_abs": differences,
            "updated_parameter_tensors": sum(v > 0 for v in differences.values()),
            "correct": int((clean_logits.detach().argmax(1) == labels).sum().item()),
            "examples": len(labels)}
