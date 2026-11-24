"""Paired Seed-42 NTC-BCL training from the verified baseline initialization."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import random
import subprocess
import sys
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from BCL_AUDIT.cifar100lt.data import build_cifar_transforms, cifar100lt_class_counts
from BCL_AUDIT.cifar100lt.losses import BalancedContrastiveLoss
from BCL_AUDIT.cifar100lt.models import BCLCIFARModel
from BCL_AUDIT.cifar100lt.train import (
    BASE_LR, BATCH_SIZE, CHECKPOINT_EPOCHS, EPOCHS, LAMBDA_LC, LR_GAMMA, LR_MILESTONES,
    MOMENTUM, MU_BCL, TEMPERATURE, WARMUP_EPOCHS, WEIGHT_DECAY, classification_logits_v1,
    compose_group_loss, epoch_seed_for, file_sha256, first_batch_views_sha256,
    learning_rate_for_epoch, load_initial_model_state, seed_worker, set_seed, state_dict_sha256,
)
from BCL_LTN.audit_epoch160_margin_reliability import load_cifar100_train_pixels_only
from BCL_LTN.make_noisy_lt import group_class_ids
from BCL_LTN.noise_robust_losses import (
    ForwardCorrectedLogitCompensation, NoiseAwareSoftBalancedContrastiveLoss,
    estimate_clean_class_counts, soft_targets_from_three_view_logits,
)
from BCL_LTN.train_noisy_bcl import (
    FIXED_INDICES_SHA256, FIXED_TRAIN_SOURCE_SHA256, NOISE_ARRAYS, OfficialTestAudit,
    TrainDiagnosticDataset, json_dump, safe_torch_load, write_train_diagnostics_csv,
)


ROOT = Path(__file__).resolve().parents[1]
BRANCH = "research/bcl-ltn-root-correction-20261124"
DATA_ROOT = ROOT / "H2T/data/cifar100"
NOISE = ROOT / "BCL_LTN/data"
INDICES = ROOT / "BCL_AUDIT/outputs/data_audit_recheck_20261122_01/cifar100lt_if100_train_indices_seed0.npy"
BASE_ROOT = ROOT / "BCL_LTN/outputs/BCL_sym04_seed42"
BASE_RUN = BASE_ROOT / "BCL_sym04_seed42"
OUTPUT_ROOT = ROOT / "BCL_LTN/outputs/NTC_BCL_sym04_seed42_20261124"
SMOKE_ROOT = OUTPUT_ROOT.with_name(OUTPUT_ROOT.name + "_smoke")
VALIDATION_ROOT = OUTPUT_ROOT.with_name(OUTPUT_ROOT.name + "_validation")
ARM_B, ARM_C = ARMS = ("NTC_LC_HARD_BCL", "NTC_FULL_SOFT_BCL")
N, K, SEED = 10_847, 100, 42
ETA, EPSILON, Q_TEMPERATURE, INSTANCE_EPOCHS = 0.4, 1.0, 1.0, 20
STEPS_PER_EPOCH = (N + BATCH_SIZE - 1) // BATCH_SIZE
MANIFEST_SHA = "0903d755db1e490cdf7a507f38a2c05ccdb32c7ecbd01d029487ea95205bd5e8"
NOISY_SHA = "691b8c8bf5758811d54c33c213e3bf2db12f586aa3d857f37da309a55d03554f"
INITIAL_STATE_SHA = "11c52fe9e2b082b0c36ef025f19ae2d963dbc19c456e0d5aa17aac14fffdf440"
SOURCE_FILES = (
    "BCL_LTN/CODEX_NTC_BCL_ROOT_FIX.txt", "BCL_LTN/train_ntc_bcl.py", "BCL_LTN/noise_robust_losses.py",
    "BCL_LTN/tests/test_noise_robust_losses.py", "BCL_LTN/train_noisy_bcl.py",
    "BCL_LTN/make_noisy_lt.py", "BCL_LTN/audit_epoch160_margin_reliability.py",
    *(f"BCL_AUDIT/cifar100lt/{name}.py" for name in ("data", "losses", "models", "train")),
)
PROTOCOL = {
    "epochs": EPOCHS, "batch_size": BATCH_SIZE, "seed": SEED, "num_workers": 0, "precision": "fp32",
    "amp": False, "tf32": False, "gradient_accumulation": False, "optimizer": "SGD", "lr": BASE_LR,
    "momentum": MOMENTUM, "weight_decay": WEIGHT_DECAY, "lr_warmup_epochs": WARMUP_EPOCHS,
    "lr_milestones": list(LR_MILESTONES), "lr_gamma": LR_GAMMA, "lambda_lc": LAMBDA_LC, "mu_bcl": MU_BCL,
    "contrastive_temperature": TEMPERATURE, "q_temperature": Q_TEMPERATURE, "noise_rate": ETA,
    "prior_epsilon": EPSILON, "instance_warmup_epochs_C": INSTANCE_EPOCHS,
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def checked_file(path, expected):
    if file_sha256(path) != expected:
        raise RuntimeError(f"SHA256 mismatch: {path}")


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


class TruthReadGate:
    """Block sample truth files until all TRAIN checkpoint predictions have been sealed."""

    def __init__(self):
        self.targets = {
            os.path.normcase(str((NOISE / NOISE_ARRAYS[key]).resolve()))
            for key in ("clean_labels", "is_noisy")
        }
        self.opens = self.denied = 0
        self.allowed = False
        sys.addaudithook(self.audit)

    def audit(self, event, args):
        if event != "open" or not isinstance(args[0], (str, bytes)):
            return
        path = os.path.normcase(os.path.abspath(os.fsdecode(args[0])))
        if path in self.targets:
            if not self.allowed:
                self.denied += 1
                raise RuntimeError("TRAIN truth was opened before prediction freeze")
            self.opens += 1

    @contextmanager
    def offline_only(self):
        if self.opens or self.denied:
            raise RuntimeError("Unclean sample-truth access history")
        self.allowed = True
        try:
            yield
        finally:
            self.allowed = False


class ObservedTrainDataset(Dataset):
    """Original three transforms on fixed TRAIN pixels with an observed-only target."""

    def __init__(self, pixels, indices, observed, transforms):
        self.pixels, self.indices, self.observed = pixels, indices, observed
        self.v1, self.sim, self.eval_transform = transforms

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = int(self.indices[position])
        image = Image.fromarray(self.pixels[index])
        return (self.v1(image), self.sim(image), self.sim(image)), int(self.observed[position]), index


def observed_inputs():
    manifest_path = NOISE / "noise_manifest.json"
    observed_path = NOISE / NOISE_ARRAYS["noisy_labels"]
    checked_file(manifest_path, MANIFEST_SHA)
    checked_file(observed_path, NOISY_SHA)
    checked_file(INDICES, FIXED_INDICES_SHA256)
    for name, digest in FIXED_TRAIN_SOURCE_SHA256.items():
        checked_file(DATA_ROOT / "cifar-100-python" / name, digest)
    manifest = read_json(manifest_path)
    expected = {
        "seed": SEED, "noise_rate_requested": ETA, "num_train_samples": N, "num_classes": K,
        "indices_sha256": FIXED_INDICES_SHA256, "noisy_labels_file": observed_path.name,
        "noisy_labels_sha256": NOISY_SHA, "train_source_sha256": FIXED_TRAIN_SOURCE_SHA256,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Noise manifest does not match the locked data protocol")
    indices = np.load(INDICES, allow_pickle=False)
    observed = np.load(observed_path, allow_pickle=False)
    if (indices.shape != (N,) or indices.dtype != np.int64 or np.unique(indices).size != N
            or np.any((indices < 0) | (indices >= 50_000)) or observed.shape != (N,)
            or observed.dtype != np.int64 or np.any((observed < 0) | (observed >= K))):
        raise RuntimeError("Invalid fixed indices or observed labels")
    counts = np.bincount(observed, minlength=K)
    if counts.tolist() != manifest["observed_noisy_class_counts"]:
        raise RuntimeError("Observed counts and noise manifest differ")
    dataset = ObservedTrainDataset(load_cifar100_train_pixels_only(DATA_ROOT), indices, observed, build_cifar_transforms())
    return dataset, counts, manifest


def verified_initialization():
    summary = read_json(BASE_RUN / "training_summary.json")
    environment = read_json(BASE_ROOT / "environment.json")
    expected = summary["initial_model_state_sha256"]
    if (expected != INITIAL_STATE_SHA or environment.get("seed") != SEED
            or environment.get("initial_model_state_sha256", expected) != expected
            or summary.get("epochs_completed") != EPOCHS
            or summary.get("optimizer_updates") != EPOCHS * STEPS_PER_EPOCH):
        raise RuntimeError("Historical initialization or training receipt mismatch")
    init_file = BASE_ROOT / "init_seed42.pth"
    if init_file.is_file():
        state = load_initial_model_state(init_file, expected_seed=SEED, expected_sha256=expected)
        source = {"source": str(init_file), "file_sha256": file_sha256(init_file)}
    else:
        set_seed(SEED)
        state = BCLCIFARModel().state_dict()
        source = {"source": "set_seed(42) plus BCLCIFARModel"}
    if state_dict_sha256(state) != expected:
        raise RuntimeError("The Seed-42 initialization differs from the historical baseline")
    return state, {**source, "model_state_sha256": expected}, summary


def prior_record(counts):
    raw, corrected = estimate_clean_class_counts(counts, noise_rate=ETA, epsilon=EPSILON)
    clean_prior = corrected / N
    receipt = {
        "eta": ETA, "epsilon": EPSILON, "observed_count": counts.tolist(), "inverse_raw_count": raw.tolist(),
        "corrected_count": corrected.tolist(), "observed_prior_report_only": (counts / N).tolist(),
        "corrected_prior": clean_prior.tolist(), "corrected_count_sum": float(corrected.sum()),
        "classes_clipped_before_normalization": int((raw < EPSILON).sum()),
        "prior_sha256": array_hash(clean_prior.astype("<f8")), "hash_encoding": "little-endian float64 in class-ID order",
    }
    return corrected, receipt


def train_loader(dataset):
    generator = torch.Generator()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=0,
                        pin_memory=True, generator=generator, worker_init_fn=seed_worker)
    return loader, generator


def make_training_objects(state, arm, counts, device):
    set_seed(SEED)
    model = BCLCIFARModel()
    model.load_state_dict(state, strict=True)
    if state_dict_sha256(model.state_dict()) != INITIAL_STATE_SHA:
        raise RuntimeError(f"{arm}: restored initial model SHA256 mismatch")
    model.to(device).train()
    correction = ForwardCorrectedLogitCompensation(counts, noise_rate=ETA).to(device)
    # Original hard BCL uses the shape of these counts to obtain K. Its class occurrences use the current observed batch.
    contrastive = (BalancedContrastiveLoss(counts, temperature=TEMPERATURE) if arm == ARM_B else
                   NoiseAwareSoftBalancedContrastiveLoss(K, temperature=TEMPERATURE)).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=BASE_LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    return model, correction, contrastive, optimizer


def bcl_mode(arm, epoch_index):
    if arm == ARM_B:
        return "hard_observed"
    if arm == ARM_C:
        return "instance_warmup" if epoch_index < INSTANCE_EPOCHS else "soft_semantic"
    raise ValueError(f"Unknown NTC-BCL arm: {arm}")


def three_view_objective(model, arm, epoch_index, views, observed, correction, contrastive):
    """One backbone forward in v1/v2/v3 order, followed by the locked arm objective."""
    mode = bcl_mode(arm, epoch_index)
    batch_size = observed.shape[0]
    features, raw, projected, prototypes = model(torch.cat(views, dim=0), compute_contrastive=mode != "instance_warmup")
    if mode == "instance_warmup":
        projected = model.sample_mlp(features)
    v1 = classification_logits_v1(raw, batch_size)
    _, v2, v3 = projected.split(batch_size, dim=0)
    two_views = torch.stack((v2, v3), dim=1)
    if mode == "hard_observed":
        bcl = contrastive(prototypes, two_views, observed)
    elif mode == "instance_warmup":
        bcl = contrastive(None, two_views, semantic_enabled=False)
    else:
        q = soft_targets_from_three_view_logits(raw, batch_size, temperature=Q_TEMPERATURE)
        bcl = contrastive(prototypes, two_views, q)
    return correction(v1, observed), bcl, raw


def watch_training_calls(model, optimizer):
    counts = Counter({"backbone": 0, "sample_mlp": 0, "prototype_mlp": 0, "optimizer": 0})

    def tally(name):
        def hook(*_args):
            counts[name] += 1
        return hook

    for name in ("backbone", "sample_mlp", "prototype_mlp"):
        getattr(model, name).register_forward_hook(tally(name))
    optimizer.register_step_post_hook(tally("optimizer"))
    return counts


def sgd_step(model, arm, epoch_index, views, observed, correction, contrastive, optimizer, device, *, smoke=False):
    mode = bcl_mode(arm, epoch_index)
    expected_unused = mode == "instance_warmup"
    observed = observed.to(device, non_blocking=True)
    views = [view.to(device, non_blocking=True) for view in views]
    if (len(views) != 3 or any(view.shape != (len(observed), 3, 32, 32) or view.dtype != torch.float32 for view in views)
            or observed.dtype != torch.long):
        raise RuntimeError("The three-view FP32 TRAIN batch is invalid")
    optimizer.zero_grad(set_to_none=True)
    lc, bcl, raw = three_view_objective(model, arm, epoch_index, views, observed, correction, contrastive)
    total = compose_group_loss("BCL", lc, bcl)
    metrics = torch.stack((lc.detach(), bcl.detach(), total.detach()))
    if metrics.dtype != torch.float32 or raw.dtype != torch.float32 or not torch.isfinite(metrics).all().item():
        raise RuntimeError(f"Non-finite or non-FP32 loss in {arm}, epoch index {epoch_index}")
    total.backward()
    parameters = dict(model.named_parameters())
    expected_none = {name for name in parameters if expected_unused and name.startswith("prototype_mlp.")}
    actual_none = {name for name, parameter in parameters.items() if parameter.grad is None}
    if actual_none != expected_none:
        raise RuntimeError(f"Gradient connectivity mismatch in {mode}: expected {sorted(expected_none)}, got {sorted(actual_none)}")
    active = {name: p for name, p in parameters.items() if name not in expected_none}
    if not torch.isfinite(torch.cat([p.grad.detach().flatten() for p in active.values()])).all().item():
        bad = [name for name, p in active.items() if not torch.isfinite(p.grad).all().item()]
        raise RuntimeError(f"Non-finite active gradients in {arm}, epoch index {epoch_index}: {bad}")
    if smoke:
        before = {name: p.detach().clone() for name, p in parameters.items()}
    optimizer.step()
    result = {
        **dict(zip(("loss_lc", "loss_bcl", "loss_total"), metrics.cpu().tolist())),
        "observed_correct": int((raw[:len(observed)].detach().argmax(1) == observed).sum().item()),
    }
    if smoke:
        if not torch.isfinite(torch.cat([p.detach().flatten() for p in parameters.values()])).all().item():
            raise RuntimeError("Non-finite parameters after Smoke SGD")
        changed = [name for name, p in parameters.items() if not torch.equal(p.detach(), before[name])]
        if not changed or not set(changed).issubset(active):
            raise RuntimeError("No active parameter updated or a cold-start prototype changed")
        for prefix in ("backbone.", "classifier.", "sample_mlp.", *(("prototype_mlp.",) if not expected_unused else ())):
            if not any(name.startswith(prefix) for name in changed):
                raise RuntimeError(f"No Smoke update in the active branch {prefix}")
        result.update({
            "mode": mode, "backbone_input_shape": [3 * len(observed), 3, 32, 32],
            "active_gradient_tensors": len(active), "expected_grad_none": sorted(actual_none),
            "unexpected_missing_gradients": 0, "nonfinite_gradients": 0, "changed_parameter_tensors": len(changed),
            "max_abs_parameter_change": max(float((parameters[name].detach() - before[name]).abs().max()) for name in changed),
            "raw_three_view_logits_sha256": array_hash(raw.detach().cpu().numpy()),
        })
    return result


def seed_training_epoch(epoch_index, generator, optimizer):
    epoch_seed = epoch_seed_for(SEED, epoch_index)
    set_seed(epoch_seed)
    generator.manual_seed(epoch_seed)
    lr = learning_rate_for_epoch(epoch_index)
    for group in optimizer.param_groups:
        group["lr"] = lr
    return epoch_seed, lr


def batch_identity(views, observed, indices):
    return {"augmented_views_sha256": first_batch_views_sha256(views),
            "source_indices_sha256": array_hash(indices.numpy()),
            "observed_labels_sha256": array_hash(observed.numpy())}


def run_smoke(output, dataset, init, corrected, provenance, device, test_gate, truth_gate):
    records = {}
    for arm in ARMS:
        torch.cuda.empty_cache()
        model, lc, bcl, optimizer = make_training_objects(init, arm, corrected, device)
        calls = watch_training_calls(model, optimizer)
        loader, generator = train_loader(dataset)
        epoch_seed, lr = seed_training_epoch(0, generator, optimizer)
        views, observed, indices = next(iter(loader))
        if len(observed) != BATCH_SIZE or indices.unique().numel() != BATCH_SIZE:
            raise RuntimeError("Smoke did not receive a full unique 256-sample batch")
        index_labels = dict(zip(dataset.indices.tolist(), dataset.observed.tolist()))
        if observed.tolist() != [index_labels[index] for index in indices.tolist()]:
            raise RuntimeError("Smoke observed labels do not match the fixed TRAIN indices")
        before = state_dict_sha256(model.prototype_mlp.state_dict())
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        step = sgd_step(model, arm, 0, views, observed, lc, bcl, optimizer, device, smoke=True)
        torch.cuda.synchronize(device)
        seconds = time.perf_counter() - started
        unchanged = state_dict_sha256(model.prototype_mlp.state_dict()) == before
        expected_calls = {"backbone": 1, "sample_mlp": 1, "prototype_mlp": int(arm == ARM_B), "optimizer": 1}
        if dict(calls) != expected_calls or (arm == ARM_C and not unchanged):
            raise RuntimeError(f"Smoke control flow or cold-start state mismatch: {arm}, {calls}")
        if test_gate.opens or test_gate.denied or truth_gate.opens or truth_gate.denied:
            raise RuntimeError("Smoke isolation gate failed")
        record = {
            "arm": arm, "passed": True, **step, "identity": batch_identity(views, observed, indices),
            "epoch_index": 0, "epoch_seed": epoch_seed, "lr": lr, "actual_calls": dict(calls),
            "prototype_state_unchanged": unchanged, "initial_model_state_sha256": INITIAL_STATE_SHA,
            "official_test_opens": 0, "truth_array_opens": 0, "formal_training_started": False,
            "seconds": seconds, "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }
        json_dump(output / arm / "smoke_receipt.json", record)
        records[arm] = record
        print(f"[SMOKE] {arm} {step['mode']} updates=1 LC={step['loss_lc']:.6f} BCL={step['loss_bcl']:.6f}", flush=True)
        del model, lc, bcl, optimizer
    a, b = (records[arm] for arm in ARMS)
    for key in ("identity", "epoch_seed", "lr", "initial_model_state_sha256", "raw_three_view_logits_sha256", "loss_lc"):
        if a[key] != b[key]:
            raise RuntimeError(f"GPU Smoke pairing differs for {key}")
    json_dump(output / "smoke_completed.json", {"passed": True, "provenance": provenance, "arms": records})


def save_checkpoint(path, model, optimizer, generator, arm, epoch, updates, provenance):
    state = model.state_dict()
    state_sha = state_dict_sha256(state)
    with path.open("xb") as stream:
        torch.save({
            "arm": arm, "epoch": epoch, "seed": SEED, "model_state": state, "model_state_sha256": state_sha,
            "optimizer_state": optimizer.state_dict(), "optimizer_updates": updates,
            "rng_state": {"python": random.getstate(), "numpy": np.random.get_state(),
                          "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()},
            "loader_generator_state": generator.get_state(), "provenance": provenance,
        }, stream)
    return {"epoch": epoch, "path": str(path), "sha256": file_sha256(path), "model_state_sha256": state_sha}


def train_arm(output, dataset, counts, init, corrected, arm, provenance, device, smoke, baseline_rows, paired_rows):
    torch.cuda.empty_cache()
    model, lc, bcl, optimizer = make_training_objects(init, arm, corrected, device)
    calls = watch_training_calls(model, optimizer)
    loader, generator = train_loader(dataset)
    directory = output / arm
    checkpoints_dir = directory / "checkpoints"
    checkpoints_dir.mkdir(parents=True)
    sorted_indices = np.sort(dataset.indices)
    targets_at_index = np.full(50_000, -1, dtype=np.int64)
    targets_at_index[dataset.indices] = dataset.observed
    prototype_initial_hash = state_dict_sha256(model.prototype_mlp.state_dict())
    warmup = None
    checkpoints, rows = [], []
    started = time.perf_counter()
    with (directory / "history.jsonl").open("x", encoding="utf-8") as log:
        for epoch_index in range(EPOCHS):
            epoch = epoch_index + 1
            epoch_seed, lr = seed_training_epoch(epoch_index, generator, optimizer)
            mode = bcl_mode(arm, epoch_index)
            previous = calls.copy()
            all_indices, label_counts = [], np.zeros(K, dtype=np.int64)
            augmentation_hash = hashlib.sha256()
            loss_sums = np.zeros(3, dtype=np.float64)
            seen = correct = 0
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            epoch_start = time.perf_counter()
            for batch_index, (views, observed, indices) in enumerate(loader):
                if batch_index == 0:
                    first_batch = batch_identity(views, observed, indices)
                    if epoch_index == 0 and first_batch != smoke["identity"]:
                        raise RuntimeError(f"{arm}: first formal batch differs from Smoke")
                source_np = indices.numpy()
                observed_np = observed.numpy()
                if not np.array_equal(observed_np, targets_at_index[source_np]):
                    raise RuntimeError("Observed labels do not align with source indices")
                label_counts += np.bincount(observed_np, minlength=K)
                all_indices.append(source_np.copy())
                for view in views:
                    augmentation_hash.update(memoryview(view.contiguous().numpy()).cast("B"))
                step = sgd_step(model, arm, epoch_index, views, observed, lc, bcl, optimizer, device)
                loss_sums += np.asarray([step["loss_lc"], step["loss_bcl"], step["loss_total"]]) * len(observed)
                correct += step["observed_correct"]
                seen += len(observed)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - epoch_start
            order = np.concatenate(all_indices)
            actual_calls = {key: value - previous[key] for key, value in calls.items()}
            expected_calls = {"backbone": STEPS_PER_EPOCH, "sample_mlp": STEPS_PER_EPOCH,
                              "prototype_mlp": 0 if mode == "instance_warmup" else STEPS_PER_EPOCH,
                              "optimizer": STEPS_PER_EPOCH}
            if (seen != N or not np.array_equal(np.sort(order), sorted_indices)
                    or not np.array_equal(label_counts, counts) or actual_calls != expected_calls):
                raise RuntimeError(f"{arm} epoch {epoch}: coverage or runtime call count mismatch")
            row = {
                "epoch": epoch, "epoch_index": epoch_index, "epoch_seed": epoch_seed, "lr": lr, "bcl_mode": mode,
                **dict(zip(("loss_lc", "loss_bcl", "loss_total"), (loss_sums / seen).tolist())),
                "observed_correct": correct, "observed_accuracy_percent": 100.0 * correct / seen,
                "samples": seen, "optimizer_updates": actual_calls["optimizer"], "actual_calls": actual_calls,
                "observed_noisy_class_counts": label_counts.tolist(), "sample_order_sha256": array_hash(order),
                "all_three_view_augmentations_sha256": augmentation_hash.hexdigest(), "first_batch": first_batch,
                "epoch_seconds": elapsed, "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            }
            base = baseline_rows[epoch_index]
            for key in ("epoch", "epoch_seed", "lr", "sample_order_sha256"):
                if row[key] != base[key]:
                    raise RuntimeError(f"{arm} epoch {epoch}: historical data/schedule mismatch for {key}")
            if paired_rows is not None:
                for key in ("sample_order_sha256", "all_three_view_augmentations_sha256", "first_batch",
                            "observed_noisy_class_counts", "epoch_seed", "lr"):
                    if row[key] != paired_rows[epoch_index][key]:
                        raise RuntimeError(f"{arm} epoch {epoch}: arm pairing failed for {key}")
            if arm == ARM_C and epoch == INSTANCE_EPOCHS:
                proto_end = state_dict_sha256(model.prototype_mlp.state_dict())
                if proto_end != prototype_initial_hash or any(p in optimizer.state for p in model.prototype_mlp.parameters()):
                    raise RuntimeError("C instance cold-start changed prototype parameters, BN, or optimizer state")
                warmup = {"completed_epochs": epoch, "prototype_mlp_calls": calls["prototype_mlp"],
                          "optimizer_updates": calls["optimizer"], "prototype_state_sha256_before": prototype_initial_hash,
                          "prototype_state_sha256_after": proto_end, "prototype_optimizer_state_entries": 0}
                json_dump(directory / "instance_warmup_completed.json", warmup)
            if epoch in CHECKPOINT_EPOCHS:
                entry = save_checkpoint(checkpoints_dir / f"epoch_{epoch:03d}.pth", model, optimizer, generator,
                                        arm, epoch, calls["optimizer"], provenance)
                row["checkpoint"] = entry
                checkpoints.append(entry)
            rows.append(row)
            log.write(json.dumps(row, allow_nan=False) + "\n")
            log.flush()
            print(f"[{arm}] epoch {epoch:03d}/{EPOCHS} mode={mode} lr={lr:.5g} updates={calls['optimizer']} "
                  f"LC={row['loss_lc']:.6f} BCL={row['loss_bcl']:.6f} total={row['loss_total']:.6f} "
                  f"observed_acc={row['observed_accuracy_percent']:.2f}% seconds={elapsed:.1f}", flush=True)
    if (len(rows) != EPOCHS or calls["optimizer"] != EPOCHS * STEPS_PER_EPOCH
            or tuple(entry["epoch"] for entry in checkpoints) != CHECKPOINT_EPOCHS
            or (arm == ARM_C and warmup is None)):
        raise RuntimeError(f"{arm} did not complete the fixed training schedule")
    result = {
        "arm": arm, "epochs_completed": len(rows), "actual_optimizer_updates": calls["optimizer"],
        "checkpoints": checkpoints, "last_epoch": rows[-1], "total_training_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": max(row["peak_allocated_bytes"] for row in rows),
        "peak_reserved_bytes": max(row["peak_reserved_bytes"] for row in rows), "actual_calls": dict(calls),
        "instance_warmup": warmup, "initial_state_sha256": INITIAL_STATE_SHA,
    }
    json_dump(directory / "training_completed.json", result)
    return result, rows


def restore_saved_model(model, entry, arm, provenance):
    path = Path(entry["path"])
    checked_file(path, entry["sha256"])
    checkpoint = safe_torch_load(path, torch.device("cpu"))
    if (checkpoint.get("arm") != arm or checkpoint.get("epoch") != entry["epoch"]
            or checkpoint.get("optimizer_updates") != entry["epoch"] * STEPS_PER_EPOCH
            or checkpoint.get("provenance") != provenance or not checkpoint.get("optimizer_state", {}).get("state")
            or checkpoint.get("model_state_sha256") != entry["model_state_sha256"]):
        raise RuntimeError("Saved checkpoint has incorrect training provenance")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    if state_dict_sha256(model.state_dict()) != entry["model_state_sha256"]:
        raise RuntimeError("Restored checkpoint model SHA256 mismatch")


@torch.inference_mode()
def predict_raw(model, dataset, device):
    model.eval()
    before = state_dict_sha256(model.state_dict())
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    predicted = np.empty(len(dataset), dtype=np.int64)
    start = 0
    for images, positions in loader:
        stop = start + len(positions)
        _, logits, _, _ = model(images.to(device, non_blocking=True), compute_contrastive=False)
        if (not np.array_equal(positions.numpy(), np.arange(start, stop))
                or logits.shape != (len(positions), K) or logits.dtype != torch.float32
                or not torch.isfinite(logits).all().item()):
            raise RuntimeError("Unexpected eval positions or non-finite raw logits")
        predicted[start:stop] = logits.argmax(1).cpu().numpy()
        start = stop
    if start != len(dataset) or state_dict_sha256(model.state_dict()) != before:
        raise RuntimeError("Incomplete evaluation or changed model/BN state")
    return predicted


def freeze_train_predictions(output, model, dataset, results, provenance, device):
    evaluation = TrainDiagnosticDataset(dataset.pixels, dataset.indices, dataset.eval_transform)
    inventory = {}
    for arm in ARMS:
        directory = output / arm / "train_predictions"
        directory.mkdir()
        inventory[arm] = []
        for entry in results[arm]["checkpoints"]:
            restore_saved_model(model, entry, arm, provenance)
            predicted = predict_raw(model, evaluation, device)
            path = directory / f"epoch_{entry['epoch']:03d}.npz"
            with path.open("xb") as stream:
                np.savez_compressed(stream, original_train_index=dataset.indices, predicted_class=predicted)
            inventory[arm].append({**entry, "prediction_path": str(path), "prediction_sha256": file_sha256(path)})
            print(f"[TRAIN PREDICTIONS] {arm} epoch {entry['epoch']:03d} samples={len(predicted)}", flush=True)
    json_dump(output / "train_predictions_frozen.json", inventory)
    return inventory


def offline_train_diagnostics(output, inventory, dataset, manifest, truth_gate):
    arrays = {}
    with truth_gate.offline_only():
        for key in ("clean_labels", "is_noisy"):
            path = NOISE / NOISE_ARRAYS[key]
            if manifest[f"{key}_file"] != path.name:
                raise RuntimeError("Historical diagnostic label filename mismatch")
            checked_file(path, manifest[f"{key}_sha256"])
            arrays[key] = np.load(path, allow_pickle=False)
    clean, noisy = arrays["clean_labels"], arrays["is_noisy"]
    clean_counts = cifar100lt_class_counts()
    if (clean.shape != (N,) or clean.dtype != np.int64 or np.any((clean < 0) | (clean >= K))
            or noisy.shape != (N,) or noisy.dtype != np.bool_ or not np.array_equal(noisy, clean != dataset.observed)
            or not np.array_equal(np.bincount(clean, minlength=K), clean_counts)
            or int(noisy.sum()) != manifest["num_flipped"]):
        raise RuntimeError("Frozen-prediction diagnostic truth failed integrity checks")
    groups = {"Overall": np.arange(K), **group_class_ids(clean_counts)}
    rows = {}
    for arm in ARMS:
        rows[arm] = []
        for entry in inventory[arm]:
            path = Path(entry["prediction_path"])
            checked_file(path, entry["prediction_sha256"])
            with np.load(path, allow_pickle=False) as saved:
                prediction = saved["predicted_class"]
                if not np.array_equal(saved["original_train_index"], dataset.indices) or prediction.shape != (N,):
                    raise RuntimeError("Frozen TRAIN prediction join mismatches fixed indices")
            row = {
                "epoch": entry["epoch"], "samples": N, "model_state_unchanged": True,
                "model_state_sha256": entry["model_state_sha256"], "prediction_sha256": entry["prediction_sha256"],
                "clean_subset_accuracy": {}, "noisy_subset_wrong_label_memorization_rate": {},
            }
            for name, classes in groups.items():
                source = np.isin(clean, classes)
                good, flipped = source & ~noisy, source & noisy
                nc, nn = int(good.sum()), int(flipped.sum())
                correct = int((prediction[good] == clean[good]).sum())
                memorized = int((prediction[flipped] == dataset.observed[flipped]).sum())
                row["clean_subset_accuracy"][name] = {
                    "correct": correct, "total": nc, "accuracy_percent": 100.0 * correct / nc,
                }
                row["noisy_subset_wrong_label_memorization_rate"][name] = {
                    "predicted_as_corrupted_observed_label": memorized, "total_noisy_examples": nn,
                    "memorization_rate_percent": 100.0 * memorized / nn,
                }
            json_dump(output / arm / "train_diagnostics" / f"epoch_{entry['epoch']:03d}.json", row)
            rows[arm].append(row)
        write_train_diagnostics_csv(output / arm / "train_diagnostics.csv", rows[arm])
    return rows


def official_test_one_open():
    # Read the official test pickle once. Validate the same bytes before decoding them.
    filename, expected_md5 = torchvision.datasets.CIFAR100.test_list[0]
    if filename != "test":
        raise RuntimeError("Unexpected CIFAR-100 TEST filename")
    payload = (DATA_ROOT / "cifar-100-python" / filename).read_bytes()
    if hashlib.md5(payload).hexdigest() != expected_md5:
        raise RuntimeError("Official CIFAR-100 TEST checksum mismatch")
    data = pickle.loads(payload, encoding="bytes")
    pixels = np.asarray(data.get(b"data", data.get("data")))
    targets = np.asarray(data.get(b"fine_labels", data.get("fine_labels")), dtype=np.int64)
    if (pixels.shape != (10_000, 3072) or pixels.dtype != np.uint8 or targets.shape != (10_000,)
            or np.any((targets < 0) | (targets >= K)) or not np.all(np.bincount(targets, minlength=K) == 100)):
        raise RuntimeError("Unexpected balanced CIFAR-100 TEST contents")
    pixels = pixels.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).copy()
    return pixels, targets, hashlib.sha256(payload).hexdigest()


def evaluate_final_test(output, model, dataset, results, diag, provenance, device, gate):
    for arm in ARMS:
        if (results[arm]["epochs_completed"] != EPOCHS or results[arm]["actual_optimizer_updates"] != EPOCHS * STEPS_PER_EPOCH
                or tuple(row["epoch"] for row in diag[arm]) != CHECKPOINT_EPOCHS
                or results[arm]["checkpoints"][-1]["model_state_sha256"] != diag[arm][-1]["model_state_sha256"]):
            raise RuntimeError("Both complete training arms and their TRAIN diagnostics are required for TEST")
    if gate.opens or gate.denied:
        raise RuntimeError("TEST isolation gate already has an access event")
    json_dump(output / "final_test_gate.json", {"allowed": True, "arms": list(ARMS),
              "epochs_per_arm": EPOCHS, "train_diagnostic_epochs": list(CHECKPOINT_EPOCHS), "official_test_opens_before": 0})
    print("[FINAL TEST GATE] Both arms and all ten TRAIN checkpoint diagnostics are complete.", flush=True)
    with gate.final_evaluation():
        pixels, targets, test_sha = official_test_one_open()
        test = TrainDiagnosticDataset(pixels, np.arange(len(targets)), dataset.eval_transform)
        results_test = {}
        groups = group_class_ids(cifar100lt_class_counts())
        for arm in ARMS:
            entry = results[arm]["checkpoints"][-1]
            restore_saved_model(model, entry, arm, provenance)
            predicted = predict_raw(model, test, device)
            correct = np.bincount(targets[predicted == targets], minlength=K)
            metrics = {"Overall": 100.0 * int(correct.sum()) / len(targets),
                       **{name: float(correct[ids].mean()) for name, ids in groups.items()}}
            result = {
                "checkpoint_sha256": entry["sha256"], "model_state_sha256": entry["model_state_sha256"],
                "metrics": metrics, "num_test_samples": len(targets), "per_class_correct": correct.tolist(),
                "evaluation": "raw logits; one fixed epoch-200 forward; macro group class accuracy",
                "official_test_evaluation_passes": 1, "test_file_sha256": test_sha,
            }
            json_dump(output / arm / "final_test_metrics.json", result)
            results_test[arm] = result
            print(f"[FINAL TEST] {arm} {metrics}", flush=True)
    if gate.opens != 1 or gate.denied:
        raise RuntimeError(f"Expected exactly one official TEST file open, observed {gate.opens}")
    return results_test


def git_provenance():
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

    branch = git("branch", "--show-current")
    if branch != BRANCH:
        raise RuntimeError(f"Expected branch {BRANCH}, got {branch}")
    return {"branch": branch, "head": git("rev-parse", "HEAD")}


def source_hashes():
    return {relative: file_sha256(ROOT / relative) for relative in SOURCE_FILES}


def check_protected(inventory):
    expected = inventory["protected_sha256"]
    for relative, digest in expected.items():
        checked_file(ROOT / relative, digest)
    roots = (ROOT / "BCL_AUDIT", BASE_ROOT)
    actual = {path.relative_to(ROOT).as_posix() for directory in roots for path in directory.rglob("*") if path.is_file()}
    prefixes = tuple(directory.relative_to(ROOT).as_posix() + "/" for directory in roots)
    if actual != {relative for relative in expected if relative.startswith(prefixes)}:
        raise RuntimeError("Protected historical directory file set changed")
    for relative, prior in inventory["truth_files_stat_only"].items():
        stat = (ROOT / relative).stat()
        if stat.st_size != prior["bytes"] or stat.st_mtime_ns != prior["mtime_ns"]:
            raise RuntimeError("Protected sample truth file metadata changed")
    return len(expected)


def verify_smoke(smoke_path, provenance):
    smoke = read_json(smoke_path)
    records = smoke["arms"]
    if smoke.get("passed") is not True or smoke.get("provenance") != provenance or list(records) != list(ARMS):
        raise RuntimeError("Formal launch requires the current run's paired GPU Smoke")
    for arm, record in records.items():
        if (read_json(smoke_path.parent / arm / "smoke_receipt.json") != record
                or record["passed"] is not True or record["mode"] != bcl_mode(arm, 0)
                or record["actual_calls"] != {"backbone": 1, "sample_mlp": 1, "prototype_mlp": int(arm == ARM_B), "optimizer": 1}
                or record["backbone_input_shape"] != [3 * BATCH_SIZE, 3, 32, 32]
                or record["unexpected_missing_gradients"] or record["nonfinite_gradients"]
                or record["official_test_opens"] or record["truth_array_opens"]
                or record["initial_model_state_sha256"] != INITIAL_STATE_SHA
                or record["formal_training_started"] is not False or record["changed_parameter_tensors"] <= 0
                or (arm == ARM_C and record["prototype_state_unchanged"] is not True)):
            raise RuntimeError(f"Invalid GPU Smoke receipt for {arm}")
    for field in ("identity", "raw_three_view_logits_sha256", "initial_model_state_sha256", "loss_lc"):
        if records[ARM_B][field] != records[ARM_C][field]:
            raise RuntimeError(f"Matched Smoke field differs: {field}")
    return records


def baseline_comparison(summary):
    metrics_path = BASE_RUN / "final_test_metrics.json"
    archived = read_json(metrics_path)
    metrics = {name: archived["overall" if name == "Overall" else name] for name in ("Overall", "Many", "Medium", "Few")}
    if (archived["checkpoint_sha256"] != summary["checkpoint_sha256"]["epoch_200.pth"]
            or archived["official_test_eval_passes"] != 1
            or not np.allclose(list(metrics.values()), (32.22, 50.42857142857143, 35.65714285714286, 6.966666666666667),
                               rtol=0, atol=1e-10)):
        raise RuntimeError("Historical BCL final evaluation differs from the locked baseline")
    with (BASE_RUN / "history.jsonl").open(encoding="utf-8") as stream:
        history = [json.loads(line) for line in stream if line.strip()]
    if tuple(row["epoch"] for row in history) != tuple(range(1, EPOCHS + 1)):
        raise RuntimeError("Historical BCL data order receipt is incomplete")
    return history, {
        "test_metrics": metrics, "training_diagnostics": summary["checkpoint_diagnostics"],
        "last_epoch": history[-1], "epochs_completed": summary["epochs_completed"],
        "actual_optimizer_updates": summary["optimizer_updates"], "total_training_seconds": summary["total_training_seconds"],
        "peak_allocated_bytes": summary["max_peak_cuda_memory_bytes"], "peak_reserved_bytes": summary["max_peak_cuda_reserved_bytes"],
        "metrics_path": str(metrics_path), "test_metrics_sha256": file_sha256(metrics_path), "historical": True,
    }


def make_report(result):
    arms = result["comparison"]
    groups = ("Overall", "Many", "Medium", "Few")
    prior = result["prior"]
    report = [
        "# NTC-BCL · CIFAR-100-LT Seed 42", "",
        "B、C 两组从原始 Seed 42 初始化各完成 200 轮。A 使用已归档的带噪 BCL 结果。",
        "使用已知 0.4 的对称噪声率以及均匀非对角转移。结论适用于这一个合成噪声协议和种子。",
        "",
        "## 第 200 轮官方 TEST", "",
        "| 组别 | Overall | Many | Medium | Few |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, arm in arms.items():
        report.append(f"| {name} | " + " | ".join(f"{arm['test_metrics'][group]:.3f}%" for group in groups) + " |")
    report.extend([
        "",
        "| 对照 | Δ Overall (pp) | Δ Many (pp) | Δ Medium (pp) | Δ Few (pp) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for name, values in result["test_deltas_pp"].items():
        report.append(f"| {name} | " + " | ".join(f"{values[group]:+.3f}" for group in groups) + " |")
    report.extend([
        "", "## TRAIN 检查点诊断", "",
        "各列按原始 clean-source 分组，顺序为 Overall / Many / Medium / Few。模型预测写盘后才合并真值。",
        "",
        "| 组别 | Epoch | Clean subset accuracy | 错误观测标签记忆率 |",
        "| --- | ---: | --- | --- |",
    ])
    for name, arm in arms.items():
        for diag in arm["training_diagnostics"]:
            clean, wrong = diag["clean_subset_accuracy"], diag["noisy_subset_wrong_label_memorization_rate"]
            clean_values = " / ".join(f"{clean[g]['accuracy_percent']:.2f}%" for g in groups)
            wrong_values = " / ".join(f"{wrong[g]['memorization_rate_percent']:.2f}%" for g in groups)
            report.append(f"| {name} | {diag['epoch']} | {clean_values} | {wrong_values} |")
    report.extend([
        "", "## 最后一轮损失与资源", "",
        "| 组别 | LC | BCL | 加权总损失 | 实际 SGD 更新 | 训练秒数 | 峰值分配 GiB | 峰值保留 GiB |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for name, arm in arms.items():
        row = arm["last_epoch"]
        report.append(f"| {name} | {row['loss_lc']:.6f} | {row['loss_bcl']:.6f} | {row['loss_total']:.6f} | "
                      f"{arm['actual_optimizer_updates']} | {arm['total_training_seconds']:.1f} | "
                      f"{arm['peak_allocated_bytes'] / 2**30:.3f} | {arm['peak_reserved_bytes'] / 2**30:.3f} |")
    lc = result["test_deltas_pp"]["B−A"]
    soft = result["test_deltas_pp"]["C−B"]
    report.extend([
        "", "## 机制证据与局限", "",
        f"- B−A 的 Overall 和 Few 变化分别为 {lc['Overall']:+.3f} pp 和 {lc['Few']:+.3f} pp。"
        "这个对照测量了反演先验与前向校正的联合影响。",
        f"- C−B 的 Overall 和 Few 变化分别为 {soft['Overall']:+.3f} pp 和 {soft['Few']:+.3f} pp。"
        "C 的对照包含 20 轮实例级冷启动及随后 180 轮软语义关系，无法将这两项的贡献分开。",
        "- C 的实例冷启动中，prototype_mlp 调用与其动量缓冲创建次数均为 0。"
        "其参数与 BN 缓冲区逐字节保持一致。随后每步从同次三视图前向的原始 logits 构造分离梯度的 q。",
        "- C 的 prototype 正关系使用 q 和连续置信度。观测标签仍用于 B、C 的前向校正 LC，"
        "以及 B 的原始硬标签 BCL。三组对照无法单独识别先验反演、前向校正和软语义的交互效应。",
        f"- 先验反演在计数空间以 epsilon={EPSILON:.1f} 截断 {prior['classes_clipped_before_normalization']} 个类别，"
        f"归一化总数为 {prior['corrected_count_sum']:.9f}。估计计数范围 "
        f"{min(prior['corrected_count']):.6f}–{max(prior['corrected_count']):.6f}。",
        f"- 估计先验 SHA256 {prior['prior_sha256']}",
        f"- Seed 42 初始化状态 SHA256 {result['provenance']['initialization']['model_state_sha256']}",
        "- B、C 的 200 轮采样次序和全部三视图增强逐轮指纹一致。两组训练及 10 份 TRAIN 诊断完成后，"
        "同一个官方 TEST pickle 打开 1 次，两个最终模型各评估 1 次。",
        f"- {result['protected_files_verified']} 个历史文件的 SHA256 与执行前一致。"
        "样本真值文件在预测冻结后的离线阶段校验。",
        "",
    ])
    return "\n".join(report)


def run(args):
    start = time.perf_counter()
    git = git_provenance()
    validation = VALIDATION_ROOT
    current_source = source_hashes()
    tests = read_json(validation / "pytest_receipt.json")
    if tests.get("exit_code") != 0 or tests.get("source_sha256") != current_source:
        raise RuntimeError("A passed CPU pytest receipt for this exact source is required")
    checked_file(validation / "pytest_verified.log", tests["pytest_log_sha256"])
    protection = read_json(validation / "protected_before.json")
    if {"branch": protection["branch"], "head": protection["head"]} != git:
        raise RuntimeError("The current checkout differs from the initial protection receipt")
    check_protected(protection)
    output = args.output_root or (SMOKE_ROOT if args.phase == "smoke" else OUTPUT_ROOT)
    output = output.resolve()
    output.relative_to(ROOT / "BCL_LTN/outputs")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")

    test_gate, truth_gate = OfficialTestAudit(DATA_ROOT), TruthReadGate()
    if not torch.cuda.is_available():
        raise RuntimeError("RTX 5070 Laptop CUDA is required")
    device = torch.device("cuda:0")
    torch.cuda.init()
    torch.cuda.set_device(device)
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision("highest")
    set_seed(SEED)
    gpu = torch.cuda.get_device_name(device)
    if ("RTX 5070 Laptop" not in gpu or torch.is_autocast_enabled("cuda")
            or torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32):
        raise RuntimeError("The fixed RTX 5070 Laptop FP32 protocol is unavailable")
    dataset, observed_counts, manifest = observed_inputs()
    init, initialization, archive_summary = verified_initialization()
    corrected, prior = prior_record(observed_counts)
    provenance = {
        "git": git, "source_sha256": current_source, "protocol": PROTOCOL, "initialization": initialization,
        "indices_sha256": FIXED_INDICES_SHA256, "manifest_sha256": MANIFEST_SHA, "observed_labels_sha256": NOISY_SHA,
        "train_source_sha256": FIXED_TRAIN_SOURCE_SHA256, "prior_sha256": prior["prior_sha256"], "gpu": gpu,
        "python": sys.executable, "torch": str(torch.__version__), "torchvision": torchvision.__version__,
        "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "device": str(device),
        "pytest_log_sha256": tests["pytest_log_sha256"],
    }
    smoke_path = args.smoke_root.resolve() / "smoke_completed.json"
    smoke = verify_smoke(smoke_path, provenance) if args.phase == "formal" else None
    output.mkdir()
    json_dump(output / "environment.json", provenance)
    json_dump(output / "corrected_prior.json", prior)
    transition_path = output / "transition_matrix.npy"
    with transition_path.open("xb") as stream:
        np.save(stream, ForwardCorrectedLogitCompensation(corrected, noise_rate=ETA).transition.numpy(), allow_pickle=False)
    if args.phase == "smoke":
        run_smoke(output, dataset, init, corrected, provenance, device, test_gate, truth_gate)
        check_protected(protection)
        return

    baseline_rows, archive = baseline_comparison(archive_summary)
    results, paired_rows = {}, None
    for arm in ARMS:
        results[arm], rows = train_arm(output, dataset, observed_counts, init, corrected, arm, provenance, device,
                                       smoke[arm], baseline_rows, paired_rows)
        paired_rows = rows
    if test_gate.opens or test_gate.denied or truth_gate.opens or truth_gate.denied:
        raise RuntimeError("Data isolation failed during training")
    json_dump(output / "all_training_completed.json", {
        "state": "completed", "arms": {arm: {"epochs": results[arm]["epochs_completed"],
                                            "optimizer_updates": results[arm]["actual_optimizer_updates"]} for arm in ARMS},
        "paired_augmentation_fingerprints": True, "truth_open_count": 0, "official_test_open_count": 0,
    })
    torch.cuda.empty_cache()
    model = BCLCIFARModel().to(device)
    predictions = freeze_train_predictions(output, model, dataset, results, provenance, device)
    if truth_gate.opens or truth_gate.denied or test_gate.opens or test_gate.denied:
        raise RuntimeError("Data isolation failed before prediction freeze")
    diagnostics = offline_train_diagnostics(output, predictions, dataset, manifest, truth_gate)
    protected_count = check_protected(protection)
    if source_hashes() != current_source or git_provenance() != git:
        raise RuntimeError("Executed source or branch changed before final evaluation")
    test = evaluate_final_test(output, model, dataset, results, diagnostics, provenance, device, test_gate)
    groups = ("Overall", "Many", "Medium", "Few")
    comparison = {"A · BCL_sym04": archive}
    for key, arm in zip(("B", "C"), ARMS):
        comparison[f"{key} · {arm}"] = {
            **results[arm], "training_diagnostics": diagnostics[arm], "test_metrics": test[arm]["metrics"],
        }
    deltas = {
        name: {group: after[group] - before[group] for group in groups}
        for name, after, before in (
            ("B−A", test[ARM_B]["metrics"], archive["test_metrics"]),
            ("C−A", test[ARM_C]["metrics"], archive["test_metrics"]),
            ("C−B", test[ARM_C]["metrics"], test[ARM_B]["metrics"]),
        )
    }
    result = {
        "state": "completed", "provenance": provenance, "prior": prior, "comparison": comparison,
        "test_deltas_pp": deltas, "official_test_opens_before_gate": 0, "official_test_file_opens": test_gate.opens,
        "official_test_denied": test_gate.denied, "official_test_evaluation_passes_per_new_arm": 1,
        "truth_array_opens_before_train_prediction_freeze": 0, "truth_array_opens_after_freeze": truth_gate.opens,
        "protected_files_verified": protected_count, "matched_train_order_and_all_augmentations": True,
        "matched_historical_train_order": True, "smoke_receipt_sha256": file_sha256(smoke_path),
        "transition_sha256": file_sha256(transition_path), "execution_seconds": time.perf_counter() - start,
    }
    json_dump(output / "comparison_summary.json", result)
    with (output / "NTC_BCL_ROOT_FIX_REPORT.md").open("x", encoding="utf-8") as stream:
        stream.write(make_report(result))
    print(json.dumps({"state": "completed", "TEST": {arm: test[arm]["metrics"] for arm in ARMS},
                      "execution_seconds": result["execution_seconds"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fixed NTC-BCL Seed-42 smoke and paired formal training")
    parser.add_argument("--phase", choices=("smoke", "formal"), required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--smoke-root", type=Path, default=SMOKE_ROOT)
    run(parser.parse_args())
