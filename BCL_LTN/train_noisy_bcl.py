from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import hashlib
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from BCL_AUDIT.cifar100lt.data import (
    CIFAR100LT, CIFAR100BalancedTest, cifar100lt_class_counts,
)
from BCL_AUDIT.cifar100lt.losses import BalancedContrastiveLoss, LogitCompensatedCE
from BCL_AUDIT.cifar100lt.models import BCLCIFARModel
from BCL_AUDIT.cifar100lt.train import (
    BATCH_SIZE, BASE_LR, CHECKPOINT_EPOCHS, EPOCHS, LAMBDA_LC, LR_GAMMA,
    LR_MILESTONES, MOMENTUM, MU_BCL, TEMPERATURE, WEIGHT_DECAY, WARMUP_EPOCHS,
    classification_logits_v1, compose_group_loss, epoch_seed_for, file_sha256,
    first_batch_views_sha256, learning_rate_for_epoch, load_initial_model_state,
    seed_worker, set_seed, state_dict_sha256,
)
from BCL_LTN.make_noisy_lt import group_class_ids


FIXED_INDICES_SHA256 = "be73a9441289dbadec4190c8c30a1bfc551434f17b33f6dc0ceea7fb77340ac7"
FIXED_TRAIN_SOURCE_SHA256 = {
    "train": "735e79b04f092ca3d2e6d07f368c0a7d70d48c48d28865950cc24454cf45129b",
    "meta": "a5d4786345c961390f865e93b434dbd5c6904ce880667e0cb888c97d449f28b9",
}
NOISE_ARRAYS = {
    "noisy_labels": "cifar100lt_if100_sym04_seed42.npy",
    "clean_labels": "cifar100lt_if100_clean_seed42.npy",
    "is_noisy": "cifar100lt_if100_is_noisy_seed42.npy",
}
DEFAULT_FORMAL_SMOKE_REPORT = Path("BCL_LTN/outputs/BCL_sym04_seed42_smoke/smoke_report.json")


class OfficialTestAudit:
    """Count and block official TEST pickle opens outside the final evaluation."""

    def __init__(self, data_root: Path):
        self.path = os.path.normcase(str(data_root / "cifar-100-python" / "test"))
        self.opens = self.denied = 0
        self.final_started = self.allowed = False
        sys.addaudithook(self.audit_open)

    def audit_open(self, event, args):
        if event != "open" or not isinstance(args[0], (str, bytes)):
            return
        path = os.fsdecode(args[0])
        if os.path.basename(path).lower() != "test" or os.path.normcase(os.path.abspath(path)) != self.path:
            return
        if not self.allowed:
            self.denied += 1
            raise RuntimeError(f"Official CIFAR-100 TEST opened outside final evaluation: {path}")
        self.opens += 1

    @contextmanager
    def final_evaluation(self):
        if self.final_started or self.denied or self.opens:
            raise RuntimeError("Official TEST evaluation is single-use and requires a clear TRAIN-only audit")
        self.final_started = self.allowed = True
        try:
            yield
        finally:
            self.allowed = False


def load_verified_noise(data_root, indices_path, noise_dir):
    """Use the fixed TRAIN inputs and verify the saved noise artifacts without drawing labels."""
    actual_index_hash = file_sha256(indices_path)
    if actual_index_hash != FIXED_INDICES_SHA256:
        raise RuntimeError(f"Fixed IF=100 index SHA256 mismatch: {indices_path}")
    manifest_path = noise_dir / "noise_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("seed") != 42 or manifest.get("noise_rate_requested") != 0.4
            or manifest.get("indices_sha256") != actual_index_hash):
        raise RuntimeError("The noise manifest does not match the locked indices, seed or noise rate")
    train_source = data_root / "cifar-100-python"
    # Older manifests omit TRAIN/meta hashes. The fixed source SHA256 values still apply.
    if (manifest.get("train_source_sha256", FIXED_TRAIN_SOURCE_SHA256) != FIXED_TRAIN_SOURCE_SHA256
            or any(file_sha256(train_source / name) != expected for name, expected in FIXED_TRAIN_SOURCE_SHA256.items())):
        raise RuntimeError("CIFAR-100 TRAIN or meta SHA256 does not match the locked source")
    arrays = {}
    for key, name in NOISE_ARRAYS.items():
        path = noise_dir / name
        if manifest.get(f"{key}_file") != name or manifest.get(f"{key}_sha256") != file_sha256(path):
            raise RuntimeError(f"Noise manifest file/hash mismatch: {path}")
        arrays[key] = np.load(path, allow_pickle=False)
    noisy, clean, mask = (arrays[key] for key in NOISE_ARRAYS)
    if (noisy.shape != (10_847,) or clean.shape != noisy.shape or mask.shape != noisy.shape
            or noisy.dtype != np.int64 or clean.dtype != np.int64 or mask.dtype != np.bool_
            or np.any((noisy < 0) | (noisy >= 100) | (clean < 0) | (clean >= 100))):
        raise RuntimeError("Invalid saved noisy labels, clean labels or flip mask")
    if not np.array_equal(noisy != clean, mask):
        raise RuntimeError("Saved flip mask does not match the exact changed-label positions")
    base = CIFAR100LT(root=data_root, mode="bcl", indices_file=indices_path, split_seed=0, download=False)
    if not np.array_equal(base.targets[base.indices], clean):
        raise RuntimeError("Saved clean labels do not align with the fixed TRAIN indices")
    clean_counts = cifar100lt_class_counts()
    observed_counts = np.bincount(noisy, minlength=100)
    flips = int(mask.sum())
    if (not np.array_equal(np.bincount(clean, minlength=100), clean_counts)
            or manifest.get("clean_class_counts") != clean_counts.tolist()
            or manifest.get("observed_noisy_class_counts") != observed_counts.tolist()
            or np.any(observed_counts <= 0)
            or manifest.get("num_train_samples") != len(mask)
            or manifest.get("num_flipped") != flips
            or manifest.get("num_unflipped") != len(mask) - flips
            or manifest.get("noise_rate_realized") != flips / len(mask)
            or abs(flips / len(mask) - 0.4) >= 0.02):
        raise RuntimeError("Noise manifest class counts, flip totals or realized noise rate are inconsistent")
    return base, manifest, noisy, clean, mask


def safe_torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


class NoisyBCLDataset(Dataset):
    """Training-only adapter: exposes noisy labels and source indices, never clean labels."""
    def __init__(self, base: CIFAR100LT):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, position: int):
        views, observed_label = self.base[position]
        source_index = int(self.base.indices[position])
        return views, int(observed_label), source_index


class TrainDiagnosticDataset(Dataset):
    """Separate eval-only view; never used by the optimizer or loss functions."""
    def __init__(self, raw_data: np.ndarray, indices: np.ndarray, transform):
        self.raw_data = raw_data
        self.indices = np.asarray(indices, dtype=np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position: int):
        source_index = int(self.indices[position])
        image = Image.fromarray(self.raw_data[source_index])
        return self.transform(image), int(position)


def json_dump(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    paths = ("BCL_LTN/CODEX_RUN.txt", "BCL_LTN/make_noisy_lt.py", "BCL_LTN/train_noisy_bcl.py",
             *(f"BCL_AUDIT/cifar100lt/{name}.py" for name in ("data", "losses", "models", "train")))
    return {name: file_sha256(root / name) for name in paths}


def bcl_sgd_step(model, optimizer, criterion_lc, criterion_bcl, views, observed_labels,
                 device, *, check_smoke=False):
    observed_labels = observed_labels.to(device, non_blocking=True)
    batch_size = observed_labels.shape[0]
    all_views = torch.cat([view.to(device, non_blocking=True) for view in views], dim=0)
    _, all_logits, embeddings, prototypes = model(all_views, compute_contrastive=True)
    logits = classification_logits_v1(all_logits, batch_size)
    two_views = torch.stack((embeddings[batch_size:2 * batch_size], embeddings[2 * batch_size:]), dim=1)
    loss_lc = criterion_lc(logits, observed_labels)
    loss_bcl = criterion_bcl(prototypes, two_views, observed_labels)
    loss = compose_group_loss("BCL", loss_lc, loss_bcl)
    metrics = {"loss_lc": float(loss_lc.detach()), "loss_bcl": float(loss_bcl.detach()),
               "loss_total": float(loss.detach())}
    if not all(np.isfinite(value) for value in metrics.values()):
        raise RuntimeError(f"Nonfinite BCL objective: {metrics}")

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if check_smoke:
        if (all_views.shape != (3 * BATCH_SIZE, 3, 32, 32)
                or any(t.dtype != torch.float32 for t in (all_views, all_logits, embeddings, prototypes, loss))):
            raise RuntimeError("GPU Smoke must use genuine FP32 tensors and a 768-image backbone input")
        parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
        missing = [name for name, p in parameters.items() if p.grad is None]
        nonfinite = [name for name, p in parameters.items()
                     if p.grad is not None and not bool(torch.isfinite(p.grad).all())]
        if missing or nonfinite:
            raise RuntimeError(f"BCL Smoke gradient gate: missing={missing}; nonfinite={nonfinite}")
        before = {name: p.detach().clone() for name, p in parameters.items()}
    optimizer.step()
    if check_smoke:
        if not all(bool(torch.isfinite(p).all()) for p in parameters.values()):
            raise RuntimeError("Nonfinite BCL Smoke parameters after SGD")
        changed = [name for name, p in parameters.items() if not torch.equal(p.detach(), before[name])]
        if not changed:
            raise RuntimeError("BCL Smoke SGD step did not change any parameter")
        metrics.update({"backbone_input_shape": list(all_views.shape), "gradient_tensors": len(parameters),
                        "missing_gradients": 0, "nonfinite_gradients": 0,
                        "optimizer_updates": 1, "changed_parameter_tensors": len(changed),
                        "max_abs_parameter_change": max(
                            float((parameters[name].detach() - before[name]).abs().max()) for name in changed)})
    return metrics


def verify_smoke_receipt(path: Path, environment: dict) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required passed Smoke report is missing: {path}")
    smoke = json.loads(path.read_text(encoding="utf-8"))
    for key in ("source_sha256", "indices_sha256", "noise_manifest_sha256", "noisy_labels_sha256",
                "gpu", "torch", "cuda", "cudnn", "batch_size", "precision", "amp", "tf32"):
        if smoke.get(key) != environment[key]:
            raise RuntimeError(f"Smoke-to-formal provenance mismatch: {key}")
    if (smoke.get("passed") is not True or smoke.get("optimizer_updates") != 1
            or smoke.get("backbone_input_shape") != [3 * BATCH_SIZE, 3, 32, 32]
            or smoke.get("missing_gradients") != 0 or smoke.get("nonfinite_gradients") != 0
            or not smoke.get("changed_parameter_tensors", 0) > 0
            or smoke.get("official_test_opens") != 0 or smoke.get("official_test_denied") != 0):
        raise RuntimeError("A passed, TEST-free FP32 GPU Smoke is required before formal training")
    return smoke


@torch.inference_mode()
def evaluate_train_checkpoint(model, base_dataset, indices, clean_labels,
                              noisy_labels, is_noisy, class_counts, device,
                              num_workers, batch_size, epoch, run_dir):
    """Offline train-only diagnostic; clean truth is read only after predictions."""
    model.eval()
    state_hash = state_dict_sha256(model.state_dict())
    diagnostic_set = TrainDiagnosticDataset(
        base_dataset.data, indices, base_dataset.transform_eval
    )
    loader = DataLoader(
        diagnostic_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )
    predictions = np.empty(len(indices), dtype=np.int64)
    seen = np.zeros(len(indices), dtype=np.bool_)
    for images, positions in loader:
        images = images.to(device, non_blocking=True)
        _, logits, _, _ = model(images, compute_contrastive=False)
        if not bool(torch.isfinite(logits).all()):
            raise RuntimeError(f"Nonfinite TRAIN diagnostic logits at epoch {epoch}")
        locations = positions.numpy()
        if seen[locations].any() or len(np.unique(locations)) != len(locations):
            raise RuntimeError(f"Duplicate TRAIN diagnostic positions at epoch {epoch}")
        predictions[locations] = logits.argmax(dim=1).cpu().numpy()
        seen[locations] = True
    if not seen.all() or state_dict_sha256(model.state_dict()) != state_hash:
        raise RuntimeError(f"Incomplete TRAIN diagnostic or changed eval model state at epoch {epoch}")

    groups = {"Overall": np.arange(100), **group_class_ids(class_counts)}

    result = {
        "epoch": int(epoch),
        "split": "fixed 10,847-image TRAIN subset; eval transform; model.eval(); no optimizer updates",
        "samples": int(seen.sum()),
        "model_state_unchanged": True,
        "model_state_sha256": state_hash,
        "clean_subset_accuracy": {},
        "noisy_subset_wrong_label_memorization_rate": {},
    }
    for name, class_ids in groups.items():
        sample_group = np.isin(clean_labels, class_ids)
        clean_mask = sample_group & (~is_noisy)
        noisy_mask = sample_group & is_noisy
        clean_n = int(clean_mask.sum())
        noisy_n = int(noisy_mask.sum())
        clean_correct = int((predictions[clean_mask] == clean_labels[clean_mask]).sum())
        noisy_memorized = int((predictions[noisy_mask] == noisy_labels[noisy_mask]).sum())
        result["clean_subset_accuracy"][name] = {
            "correct": clean_correct,
            "total": clean_n,
            "accuracy_percent": 100.0 * clean_correct / max(clean_n, 1),
        }
        result["noisy_subset_wrong_label_memorization_rate"][name] = {
            "predicted_as_corrupted_observed_label": noisy_memorized,
            "total_noisy_examples": noisy_n,
            "memorization_rate_percent": 100.0 * noisy_memorized / max(noisy_n, 1),
        }

    # Persist prediction-free aggregates only. Clean truth and mask are not used
    # by any training object and are never passed to a criterion or optimizer.
    json_dump(run_dir / "train_diagnostics" / f"epoch_{epoch:03d}.json", result)
    return result


def write_train_diagnostics_csv(path: Path, diagnostics):
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("epoch", "clean_source_group", "clean_correct", "clean_total", "clean_accuracy_pct",
                         "wrong_label_memorized", "noisy_total", "wrong_label_memorization_pct"))
        for row in diagnostics:
            for group, clean in row["clean_subset_accuracy"].items():
                noisy = row["noisy_subset_wrong_label_memorization_rate"][group]
                writer.writerow((row["epoch"], group, clean["correct"], clean["total"], clean["accuracy_percent"],
                                 noisy["predicted_as_corrupted_observed_label"], noisy["total_noisy_examples"],
                                 noisy["memorization_rate_percent"]))


def train(args):
    data_root = args.data_root.expanduser().resolve()
    indices_path = args.indices_file.expanduser().resolve()
    noise_dir = args.noise_dir.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not indices_path.is_file():
        raise FileNotFoundError(indices_path)
    if not data_root.is_dir():
        raise FileNotFoundError(data_root)
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite output root: {output_root}")
    smoke_report_path = args.smoke_report.expanduser().resolve()
    if args.smoke_only:
        reserved = {output_root, output_root / "environment.json", output_root / "class_counts_comparison.csv"}
        if smoke_report_path.exists() or smoke_report_path in reserved:
            raise FileExistsError(f"Smoke report requires a fresh output file: {smoke_report_path}")
    report_path = output_root.parent / "BCL_sym04_baseline_report.md"
    if report_path.exists():
        report_path = output_root / report_path.name
    manifest_path = noise_dir / "noise_manifest.json"
    test_audit = OfficialTestAudit(data_root)

    # Validate the serialized artifacts before replacing the selected in-memory targets.
    base, manifest, noisy_labels, clean_labels, is_noisy = load_verified_noise(data_root, indices_path, noise_dir)
    manifest_sha256 = file_sha256(manifest_path)
    indices = base.indices.astype(np.int64, copy=True)
    noisy_full_targets = base.targets.copy()
    noisy_full_targets[indices] = noisy_labels
    base.targets = noisy_full_targets
    train_set = NoisyBCLDataset(base)

    observed_counts = np.bincount(noisy_labels, minlength=100).astype(np.int64)
    clean_counts = np.asarray(manifest["clean_class_counts"], dtype=np.int64)
    if np.any(observed_counts <= 0) or observed_counts.sum() != len(train_set):
        raise RuntimeError("Observed noisy class counts invalid")
    device = torch.device(args.device or "cuda:0")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the fixed FP32 BCL-LTN protocol")
    device = torch.device("cuda", device.index if device.index is not None else 0)
    torch.cuda.init()
    torch.cuda.set_device(device)
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision("highest")
    set_seed(42)
    amp = torch.is_autocast_enabled("cuda")
    tf32 = torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32
    if amp or tf32 or args.num_workers != 0:
        raise RuntimeError("The run requires AMP/TF32 disabled and num_workers=0")
    free_memory, total_memory = torch.cuda.mem_get_info(device)

    import torchvision
    environment = {
        "python": sys.version, "python_executable": sys.executable, "platform": platform.platform(),
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "device": str(device),
        "gpu": torch.cuda.get_device_name(device), "gpu_total_memory_bytes": int(total_memory),
        "gpu_free_before_model_bytes": int(free_memory), "amp": bool(amp), "tf32": bool(tf32),
        "seed": 42, "precision": "fp32", "batch_size": BATCH_SIZE,
        "epochs": EPOCHS, "indices_sha256": manifest["indices_sha256"],
        "noisy_labels_sha256": manifest["noisy_labels_sha256"],
        "noise_manifest_sha256": manifest_sha256,
        "train_source_sha256": FIXED_TRAIN_SOURCE_SHA256,
        "source_sha256": source_hashes(),
    }
    smoke = None if args.smoke_only else verify_smoke_receipt(smoke_report_path, environment)
    if smoke is not None:
        environment["smoke_report_sha256"] = file_sha256(smoke_report_path)
    output_root.mkdir(parents=True, exist_ok=False)
    json_dump(output_root / "environment.json", environment)
    with (output_root / "class_counts_comparison.csv").open("x", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "clean_count", "observed_noisy_count"])
        writer.writerows((i, int(clean_counts[i]), int(observed_counts[i])) for i in range(100))

    # One initialization generated exactly as the original BCL entry point.
    set_seed(42)
    template = BCLCIFARModel()
    initial_state = {k: v.detach().cpu().clone() for k, v in template.state_dict().items()}
    del template
    initial_hash = state_dict_sha256(initial_state)
    if smoke is not None:
        if smoke.get("initial_model_state_sha256") != initial_hash:
            raise RuntimeError("The formal initial state differs from the validated Smoke state")
        initial_path = output_root / "init_seed42.pth"
        with initial_path.open("xb") as stream:
            torch.save({"model_state": initial_state, "seed": 42,
                        "model_state_sha256": initial_hash}, stream)
        initial_state = load_initial_model_state(initial_path, expected_seed=42, expected_sha256=initial_hash)

    model = BCLCIFARModel()
    model.load_state_dict(initial_state, strict=True)
    if state_dict_sha256(model.state_dict()) != initial_hash:
        raise RuntimeError("Initial model state hash mismatch")
    model = model.to(device)

    criterion_lc = LogitCompensatedCE(observed_counts).to(device)
    criterion_bcl = BalancedContrastiveLoss(
        class_counts=observed_counts, temperature=TEMPERATURE
    ).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=BASE_LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
    )

    generator = torch.Generator()
    generator.manual_seed(42)
    loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"), drop_last=False,
        generator=generator, worker_init_fn=seed_worker,
    )
    if args.smoke_only:
        epoch_seed = epoch_seed_for(42, 0)
        set_seed(epoch_seed)
        generator.manual_seed(epoch_seed)
        model.train()
        for group in optimizer.param_groups:
            group["lr"] = learning_rate_for_epoch(0)
        batch = next(iter(loader))
        if len(batch) != 3:
            raise RuntimeError("Training dataset must emit exactly views, noisy labels and source indices")
        views, labels, sources = batch
        if (len(views) != 3 or any(view.shape != (BATCH_SIZE, 3, 32, 32) for view in views)
                or labels.shape != (BATCH_SIZE,) or sources.unique().numel() != BATCH_SIZE
                or not np.isin(sources.numpy(), indices).all()
                or not np.array_equal(labels.numpy(), base.targets[sources.numpy()])):
            raise RuntimeError("Real Smoke batch does not match the noisy three-view training contract")
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        smoke_step = bcl_sgd_step(model, optimizer, criterion_lc, criterion_bcl, views, labels,
                                  device, check_smoke=True)
        torch.cuda.synchronize(device)
        receipt = {
            **environment, **smoke_step, "passed": True, "train_batch_tuple_fields": 3,
            "view_input_shapes": [list(view.shape) for view in views],
            "epoch_seed": epoch_seed, "lr": learning_rate_for_epoch(0),
            "initial_model_state_sha256": initial_hash,
            "augmented_views_sha256": first_batch_views_sha256(views),
            "source_indices_sha256": hashlib.sha256(sources.numpy().tobytes()).hexdigest(),
            "noisy_batch_labels_sha256": hashlib.sha256(labels.numpy().tobytes()).hexdigest(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "step_seconds": time.perf_counter() - started,
            "official_test_opens": test_audit.opens, "official_test_denied": test_audit.denied,
            "formal_training_started": False,
        }
        json_dump(smoke_report_path, receipt)
        print(json.dumps(receipt, indent=2))
        return receipt

    run_dir = output_root / "BCL_sym04_seed42"
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite run directory: {run_dir}")
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    history_path = run_dir / "history.jsonl"
    diagnostics = []
    total_start = time.perf_counter()
    max_peak_memory = 0
    max_reserved_memory = 0
    total_updates = 0
    expected_indices = np.sort(indices)

    for epoch_index in range(EPOCHS):
        epoch = epoch_index + 1
        epoch_seed = epoch_seed_for(42, epoch_index)
        set_seed(epoch_seed)
        generator.manual_seed(epoch_seed)
        lr = learning_rate_for_epoch(epoch_index)
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        loss_lc_sum = loss_bcl_sum = loss_sum = 0.0
        seen = 0
        updates = 0
        order_hash = hashlib.sha256()
        seen_indices = []
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        epoch_start = time.perf_counter()

        for batch_index, (views, observed_labels, source_indices) in enumerate(loader):
            source_np = source_indices.numpy().astype(np.int64, copy=False)
            if epoch_index == 0 and batch_index == 0:
                if (first_batch_views_sha256(views) != smoke["augmented_views_sha256"]
                        or hashlib.sha256(source_np.tobytes()).hexdigest() != smoke["source_indices_sha256"]
                        or hashlib.sha256(observed_labels.numpy().tobytes()).hexdigest() != smoke["noisy_batch_labels_sha256"]):
                    raise RuntimeError("Formal epoch-1 first batch differs from the validated GPU Smoke batch")
            order_hash.update(source_np.tobytes())
            seen_indices.extend(source_np.tolist())
            bsz = int(observed_labels.shape[0])
            step = bcl_sgd_step(model, optimizer, criterion_lc, criterion_bcl, views, observed_labels, device)
            loss_lc_sum += step["loss_lc"] * bsz
            loss_bcl_sum += step["loss_bcl"] * bsz
            loss_sum += step["loss_total"] * bsz
            seen += bsz
            updates += 1

        torch.cuda.synchronize(device)
        peak_memory = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        max_peak_memory = max(max_peak_memory, peak_memory)
        max_reserved_memory = max(max_reserved_memory, peak_reserved)
        elapsed = time.perf_counter() - epoch_start
        observed_indices = np.asarray(seen_indices, dtype=np.int64)
        if (seen != 10_847 or observed_indices.size != 10_847 or updates != len(loader)
                or np.unique(observed_indices).size != 10_847
                or not np.array_equal(np.sort(observed_indices), expected_indices)):
            raise RuntimeError(f"Epoch {epoch}: sampler failed fixed-index coverage audit")
        total_updates += updates

        row = {
            "epoch": epoch, "epoch_seed": epoch_seed, "lr": lr,
            "loss_lc": loss_lc_sum / seen, "loss_bcl": loss_bcl_sum / seen,
            "loss_total": loss_sum / seen, "samples": seen,
            "sample_order_sha256": order_hash.hexdigest(),
            "epoch_seconds": elapsed, "peak_cuda_memory_bytes": peak_memory,
            "peak_cuda_reserved_bytes": peak_reserved, "optimizer_updates": updates,
            "observed_noisy_class_counts": observed_counts.tolist(),
            "clean_labels_accessed_by_train_loop": False,
            "is_noisy_accessed_by_train_loop": False,
        }

        if epoch in CHECKPOINT_EPOCHS:
            ckpt_name = (f"epoch_{epoch:03d}_pre_decay.pth"
                         if epoch in (160, 180) else f"epoch_{epoch:03d}.pth")
            ckpt_path = checkpoint_dir / ckpt_name
            checkpoint = {
                "epoch": epoch, "seed": 42, "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "model_state_sha256": state_dict_sha256(model.state_dict()),
                "rng_state": {"python": random.getstate(), "numpy": np.random.get_state(),
                              "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()},
                "data_loader_generator_state": generator.get_state(),
                "class_counts_for_lc_and_bcl": observed_counts.tolist(),
                "initial_model_state_sha256": initial_hash,
                "indices_sha256": manifest["indices_sha256"], "noise_manifest_sha256": manifest_sha256,
                "source_sha256": environment["source_sha256"], "optimizer_updates": total_updates,
                "protocol": {
                    "epochs": EPOCHS, "batch_size": BATCH_SIZE, "precision": "fp32",
                    "lambda_lc": LAMBDA_LC, "mu_bcl": MU_BCL,
                    "temperature": TEMPERATURE, "lr": BASE_LR,
                    "momentum": MOMENTUM, "weight_decay": WEIGHT_DECAY,
                    "warmup_epochs": WARMUP_EPOCHS, "lr_milestones": LR_MILESTONES, "lr_gamma": LR_GAMMA,
                    "current_lr": lr, "epoch_seed": epoch_seed, "amp": False, "tf32": False,
                },
            }
            with ckpt_path.open("xb") as stream:
                torch.save(checkpoint, stream)
            row["checkpoint"] = str(ckpt_path)
            row["checkpoint_sha256"] = file_sha256(ckpt_path)
            diag = evaluate_train_checkpoint(
                model, base, indices, clean_labels, noisy_labels, is_noisy,
                clean_counts, device, args.num_workers, BATCH_SIZE, epoch, run_dir,
            )
            diagnostics.append(diag)
            row["train_diagnostics_file"] = str(
                run_dir / "train_diagnostics" / f"epoch_{epoch:03d}.json"
            )
            if diag["model_state_sha256"] != checkpoint["model_state_sha256"]:
                raise RuntimeError(f"TRAIN diagnostic model does not match epoch-{epoch} checkpoint")
            del checkpoint

        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[BCL_sym04] epoch {epoch:03d}/{EPOCHS} lr={lr:.5g} "
              f"LC={row['loss_lc']:.4f} BCL={row['loss_bcl']:.4f} "
              f"total={row['loss_total']:.4f} time={elapsed:.1f}s "
              f"peak_allocated={peak_memory / (1024**3):.2f}GiB reserved={peak_reserved / (1024**3):.2f}GiB")

    total_seconds = time.perf_counter() - total_start
    final_checkpoint = checkpoint_dir / "epoch_200.pth"
    if not final_checkpoint.is_file():
        raise RuntimeError("Missing final epoch-200 checkpoint")
    if (tuple(d["epoch"] for d in diagnostics) != CHECKPOINT_EPOCHS or test_audit.opens
            or test_audit.denied or total_updates != EPOCHS * len(loader)):
        raise RuntimeError("Formal optimizer, TRAIN diagnostic or TEST isolation gate did not pass")
    write_train_diagnostics_csv(run_dir / "train_diagnostics.csv", diagnostics)
    training_summary = {
        "run": "BCL_sym04_seed42", "epochs_completed": EPOCHS,
        "total_training_seconds": total_seconds,
        "max_peak_cuda_memory_bytes": max_peak_memory,
        "max_peak_cuda_reserved_bytes": max_reserved_memory,
        "optimizer_updates": total_updates, "official_test_opens_before_gate": test_audit.opens,
        "initial_model_state_sha256": initial_hash,
        "observed_noisy_class_counts": observed_counts.tolist(),
        "checkpoint_diagnostics": diagnostics,
        "checkpoint_paths": [str(p) for p in sorted(checkpoint_dir.glob("*.pth"))],
        "checkpoint_sha256": {p.name: file_sha256(p) for p in sorted(checkpoint_dir.glob("*.pth"))},
    }
    json_dump(run_dir / "training_summary.json", training_summary)

    checkpoint = safe_torch_load(final_checkpoint, torch.device("cpu"))
    if checkpoint["epoch"] != EPOCHS or checkpoint["model_state_sha256"] != diagnostics[-1]["model_state_sha256"]:
        raise RuntimeError("Epoch-200 checkpoint differs from its TRAIN diagnostic")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    if state_dict_sha256(model.state_dict()) != checkpoint["model_state_sha256"]:
        raise RuntimeError("Reloaded epoch-200 checkpoint state SHA256 differs")
    del checkpoint
    model.eval()
    test_labels, test_predictions = [], []
    # Authorize the only TEST pass after all five TRAIN diagnostics have been persisted.
    with test_audit.final_evaluation(), torch.inference_mode():
        test_set = CIFAR100BalancedTest(root=data_root, download=False)
        test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=True)
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            _, logits, _, _ = model(images, compute_contrastive=False)
            if not bool(torch.isfinite(logits).all()):
                raise RuntimeError("Nonfinite final TEST logits")
            test_labels.append(labels.numpy())
            test_predictions.append(logits.argmax(dim=1).cpu().numpy())
    y = np.concatenate(test_labels).astype(np.int64)
    pred = np.concatenate(test_predictions).astype(np.int64)
    if len(y) != 10_000 or not np.array_equal(np.bincount(y, minlength=100), np.full(100, 100)):
        raise RuntimeError("Official balanced TEST class counts do not match CIFAR-100")
    class_acc = np.asarray([
        np.mean(pred[y == c] == y[y == c]) for c in range(100)
    ], dtype=np.float64)
    groups = group_class_ids(clean_counts)
    test_metrics = {
        "split": "balanced CIFAR-100 TEST; one evaluation after epoch 200 and five TRAIN diagnostics",
        "official_test_eval_passes": 1, "official_test_opens_before_gate": 0,
        "official_test_file_opens_during_eval": test_audit.opens,
        "checkpoint_sha256": training_summary["checkpoint_sha256"][final_checkpoint.name],
        "evaluation": "raw logits; macro-average per-class accuracy for Many/Medium/Few",
        "overall": 100.0 * float(np.mean(pred == y)),
        "Many": 100.0 * float(np.mean(class_acc[groups["Many"]])),
        "Medium": 100.0 * float(np.mean(class_acc[groups["Medium"]])),
        "Few": 100.0 * float(np.mean(class_acc[groups["Few"]])),
        "class_accuracy": class_acc.tolist(),
        "num_test_samples": int(len(y)),
    }
    json_dump(run_dir / "final_test_metrics.json", test_metrics)
    np.savez_compressed(run_dir / "final_test_predictions.npz",
                        labels=y, predictions=pred, class_accuracy=class_acc)

    report_lines = [
        "# BCL_sym04_seed42 — symmetric-noise baseline report", "",
        "## Protocol",
        "- CIFAR-100-LT IF=100; fixed 10,847-image TRAIN subset; no resampling.",
        "- Independent Bernoulli(0.4) symmetric label noise, seed 42; replacement uniformly among other 99 classes.",
        "- Original BCL ResNet-32, three views, FP32, batch 256, 200 epochs.",
        "- Objective: 2.0 × LC + 0.6 × BCL; LC prior and BCL class-count argument use observed noisy-label counts.",
        "- SGD 0.15, momentum 0.9, weight decay 0.0005, 5-epoch warmup, decay ×0.1 after epochs 160 and 180.",
        "- One balanced TEST evaluation after all 200 epochs and the five TRAIN diagnostics.", "",
        "## Noise realization",
        f"- Requested rate: {manifest['noise_rate_requested']:.4f}",
        f"- Realized rate: {manifest['noise_rate_realized']:.6f}",
        f"- Flipped: {manifest['num_flipped']} / {manifest['num_train_samples']}",
        f"- Index SHA256: {manifest['indices_sha256']}",
        f"- TRAIN and meta source SHA256: {json.dumps(manifest['train_source_sha256'])}",
        f"- Noise manifest SHA256: {manifest_sha256}",
        f"- Noisy-label SHA256: {manifest['noisy_labels_sha256']}",
        f"- Full clean and observed noisy class counts: {output_root / 'class_counts_comparison.csv'}",
        f"- Clean-label SHA256 (offline only): {manifest['clean_labels_sha256']}",
        f"- Mask SHA256 (offline only): {manifest['is_noisy_sha256']}", "",
        "| Clean-defined group | Classes | Clean source samples | Flipped source samples | Source flip rate | Observed destination labels |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group, values in manifest["many_medium_few"].items():
        source = values["clean_samples"]
        flips = values["clean_source_samples_flipped"]
        destination = values["observed_noisy_label_samples"]
        report_lines.append(f"| {group} | {values['num_classes']} | {source} | {flips} | {100 * flips / source:.2f}% | {destination} |")
    report_lines += [
        "",
        "## Five-checkpoint TRAIN diagnostics",
        f"Numerators and denominators by original clean source group are in {run_dir / 'train_diagnostics.csv'}.",
        "",
        "| Epoch | Clean overall | Clean Many | Clean Medium | Clean Few | Memorization overall | Memorization Many | Memorization Medium | Memorization Few |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for d in diagnostics:
        clean = d["clean_subset_accuracy"]
        noisy = d["noisy_subset_wrong_label_memorization_rate"]
        report_lines.append(
            f"| {d['epoch']} | "
            + " | ".join(f"{values['accuracy_percent']:.2f}%" for values in clean.values())
            + " | " + " | ".join(f"{values['memorization_rate_percent']:.2f}%" for values in noisy.values())
            + " |"
        )
    report_lines += [
        "", "## Final TEST (raw logits)", "",
        "| Metric | Noisy BCL | Clean BCL reference | Delta (pp) |",
        "|---|---:|---:|---:|",
    ]
    refs = {"Overall": 50.76, "Many": 66.54, "Medium": 49.40, "Few": 31.20}
    keys = {"Overall": "overall", "Many": "Many", "Medium": "Medium", "Few": "Few"}
    for label, key in keys.items():
        value = test_metrics[key]
        report_lines.append(f"| {label} | {value:.2f}% | {refs[label]:.2f}% | {value-refs[label]:+.2f} |")
    report_lines += [
        "", "## Runtime and provenance",
        f"- Training seconds: {total_seconds:.1f}",
        f"- Max CUDA allocated: {max_peak_memory / (1024**3):.3f} GiB",
        f"- Max CUDA reserved: {max_reserved_memory / (1024**3):.3f} GiB",
        f"- Actual training optimizer updates: {total_updates}",
        f"- Initial model state SHA256: {initial_hash}",
        f"- Environment: {json.dumps(environment, ensure_ascii=False)}",
        f"- Checkpoints: {', '.join(training_summary['checkpoint_paths'])}",
        f"- Checkpoint file SHA256: {json.dumps(training_summary['checkpoint_sha256'])}",
        "", "## Leakage and split audit",
        "- Training batches expose only three views, observed noisy labels, and original source indices.",
        "- Saved clean labels and the mask are read for input integrity checks and separate eval-mode TRAIN diagnostics.",
        "- Both loss functions and the LC prior use observed noisy labels/counts only.",
        f"- TEST pickle opens before the final gate: {training_summary['official_test_opens_before_gate']}. "
        f"Final TEST evaluator passes: 1. TEST pickle opens during evaluation (integrity check and load): {test_audit.opens}.",
        "- Training completed with the fixed FP32 batch and no OOM, retries, or protocol deviations.",
        "- This is synthetic instance-independent symmetric noise; it does not establish robustness to real-world annotation noise.",
    ]
    with report_path.open("x", encoding="utf-8") as stream:
        stream.write("\n".join(report_lines) + "\n")
    print(f"Report: {report_path}")
    print(json.dumps(test_metrics, indent=2, ensure_ascii=False))
    return training_summary, test_metrics


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train original BCL on CIFAR-100-LT with symmetric label noise")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--indices-file", type=Path, required=True)
    p.add_argument("--noise-dir", type=Path, default=Path("BCL_LTN/data"))
    p.add_argument("--output-root", type=Path, default=Path("BCL_LTN/outputs/BCL_sym04_seed42"))
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--smoke-only", action="store_true", help="Run exactly one FP32 GPU optimizer step using TRAIN.")
    p.add_argument("--smoke-report", type=Path,
                   help="Smoke JSON output in smoke-only mode, or a passed Smoke JSON input for formal training.")
    args = p.parse_args(argv)
    if args.smoke_report is None:
        args.smoke_report = args.output_root / "smoke_report.json" if args.smoke_only else DEFAULT_FORMAL_SMOKE_REPORT
    return args


def main():
    train(parse_args())


if __name__ == "__main__":
    main()
