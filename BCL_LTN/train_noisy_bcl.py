from __future__ import annotations

import argparse
import hashlib
import json
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
    classification_logits_v1, epoch_seed_for, file_sha256, first_batch_views_sha256,
    seed_worker, set_seed, state_dict_sha256,
)
from BCL_LTN.make_noisy_lt import group_class_ids


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


def lr_for_epoch(epoch_index: int) -> float:
    if epoch_index < WARMUP_EPOCHS:
        return BASE_LR * float(epoch_index + 1) / float(WARMUP_EPOCHS)
    if epoch_index >= LR_MILESTONES[1]:
        return BASE_LR * (LR_GAMMA ** 2)
    if epoch_index >= LR_MILESTONES[0]:
        return BASE_LR * LR_GAMMA
    return BASE_LR


def json_dump(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


@torch.inference_mode()
def evaluate_train_checkpoint(model, base_dataset, indices, clean_labels,
                              noisy_labels, is_noisy, class_counts, device,
                              num_workers, batch_size, epoch, run_dir):
    """Offline train-only diagnostic; clean truth is read only after predictions."""
    model.eval()
    diagnostic_set = TrainDiagnosticDataset(
        base_dataset.data, indices, base_dataset.transform_eval
    )
    loader = DataLoader(
        diagnostic_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )
    predictions = np.empty(len(indices), dtype=np.int64)
    for images, positions in loader:
        images = images.to(device, non_blocking=True)
        _, logits, _, _ = model(images, compute_contrastive=False)
        predictions[positions.numpy()] = logits.argmax(dim=1).cpu().numpy()

    groups = group_class_ids(class_counts)
    source_group = np.empty(100, dtype=object)
    for name, class_ids in groups.items():
        source_group[class_ids] = name

    result = {
        "epoch": int(epoch),
        "split": "fixed 10,847-image TRAIN subset; eval transform; model.eval(); no optimizer updates",
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


def train(args):
    data_root = args.data_root.expanduser().resolve()
    indices_path = args.indices_file.expanduser().resolve()
    noise_dir = args.noise_dir.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not indices_path.is_file():
        raise FileNotFoundError(indices_path)
    if not data_root.is_dir():
        raise FileNotFoundError(data_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    noisy_path = noise_dir / "cifar100lt_if100_sym04_seed42.npy"
    clean_path = noise_dir / "cifar100lt_if100_clean_seed42.npy"
    mask_path = noise_dir / "cifar100lt_if100_is_noisy_seed42.npy"
    manifest_path = noise_dir / "noise_manifest.json"
    for path in (noisy_path, clean_path, mask_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"Run make_noisy_lt.py first; missing {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("seed") != 42 or manifest.get("noise_rate_requested") != 0.4:
        raise RuntimeError("Noise manifest does not match the locked NR=0.4 seed=42 protocol")
    if manifest.get("indices_sha256") != file_sha256(indices_path):
        raise RuntimeError("Noise manifest was generated from a different fixed-index file")
    noisy_labels = np.load(noisy_path).astype(np.int64, copy=False)
    clean_labels = np.load(clean_path).astype(np.int64, copy=False)
    is_noisy = np.load(mask_path).astype(bool, copy=False)
    if not (noisy_labels.shape == clean_labels.shape == is_noisy.shape == (10_847,)):
        raise RuntimeError("Noise arrays must each contain exactly 10,847 entries")
    if np.any(noisy_labels[is_noisy] == clean_labels[is_noisy]):
        raise RuntimeError("Noise mask/labels integrity failure")
    if np.any(noisy_labels[~is_noisy] != clean_labels[~is_noisy]):
        raise RuntimeError("Unflipped examples changed labels")

    # The original BCL dataset loader and transforms are reused. Only the
    # in-memory target vector of this training dataset instance is replaced.
    base = CIFAR100LT(root=data_root, mode="bcl", indices_file=indices_path,
                      split_seed=0, download=False)
    indices = base.indices.astype(np.int64, copy=True)
    if not np.array_equal(base.targets[indices], clean_labels):
        raise RuntimeError("Saved clean labels do not align with fixed source indices")
    if not np.array_equal(indices, np.load(indices_path).astype(np.int64)):
        raise RuntimeError("Fixed index order changed")
    noisy_full_targets = base.targets.copy()
    noisy_full_targets[indices] = noisy_labels
    base.targets = noisy_full_targets
    train_set = NoisyBCLDataset(base)

    observed_counts = np.bincount(noisy_labels, minlength=100).astype(np.int64)
    clean_counts = cifar100lt_class_counts()
    if np.any(observed_counts <= 0) or observed_counts.sum() != len(train_set):
        raise RuntimeError("Observed noisy class counts invalid")
    if not np.array_equal(clean_counts, np.bincount(clean_labels, minlength=100)):
        raise RuntimeError("Clean IF=100 labels differ from canonical class-count vector")

    import csv
    with (output_root / "class_counts_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "clean_count", "observed_noisy_count"])
        writer.writerows((i, int(clean_counts[i]), int(observed_counts[i])) for i in range(100))

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    import torchvision
    environment = {
        "python": sys.version, "platform": platform.platform(),
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "cuda": torch.version.cuda, "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "seed": 42, "precision": "fp32", "batch_size": BATCH_SIZE,
        "epochs": EPOCHS, "indices_sha256": file_sha256(indices_path),
        "noisy_labels_sha256": file_sha256(noisy_path),
    }
    json_dump(output_root / "environment.json", environment)

    # One initialization generated exactly as the original BCL entry point.
    set_seed(42)
    template = BCLCIFARModel()
    initial_state = {k: v.detach().cpu().clone() for k, v in template.state_dict().items()}
    del template
    initial_hash = state_dict_sha256(initial_state)
    initial_path = output_root / "init_seed42.pth"
    torch.save({"model_state": initial_state, "seed": 42,
                "model_state_sha256": initial_hash}, initial_path)

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
    run_dir = output_root / "BCL_sym04_seed42"
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite run directory: {run_dir}")
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    history_path = run_dir / "history.jsonl"
    diagnostics = []
    total_start = time.perf_counter()
    max_peak_memory = 0

    for epoch_index in range(EPOCHS):
        epoch = epoch_index + 1
        epoch_seed = epoch_seed_for(42, epoch_index)
        set_seed(epoch_seed)
        generator.manual_seed(epoch_seed)
        lr = lr_for_epoch(epoch_index)
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        loss_lc_sum = loss_bcl_sum = loss_sum = 0.0
        seen = 0
        order_hash = hashlib.sha256()
        seen_indices = []
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        epoch_start = time.perf_counter()

        for batch_index, (views, observed_labels, source_indices) in enumerate(loader):
            source_np = source_indices.numpy().astype(np.int64, copy=False)
            order_hash.update(source_np.tobytes())
            seen_indices.extend(source_np.tolist())
            observed_labels = observed_labels.to(device, non_blocking=True)
            bsz = int(observed_labels.shape[0])
            v1, v2, v3 = [x.to(device, non_blocking=True) for x in views]
            all_views = torch.cat([v1, v2, v3], dim=0)
            _, all_logits, all_embeddings, prototypes = model(
                all_views, compute_contrastive=True
            )
            logits = classification_logits_v1(all_logits, bsz)
            z2 = all_embeddings[bsz:2 * bsz]
            z3 = all_embeddings[2 * bsz:3 * bsz]
            two_view_features = torch.stack([z2, z3], dim=1)

            # Only observed noisy labels enter either objective.
            loss_lc = criterion_lc(logits, observed_labels)
            loss_bcl = criterion_bcl(prototypes, two_view_features, observed_labels)
            loss = LAMBDA_LC * loss_lc + MU_BCL * loss_bcl

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_lc_sum += float(loss_lc.detach()) * bsz
            loss_bcl_sum += float(loss_bcl.detach()) * bsz
            loss_sum += float(loss.detach()) * bsz
            seen += bsz

        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_memory = int(torch.cuda.max_memory_allocated(device))
        else:
            peak_memory = 0
        max_peak_memory = max(max_peak_memory, peak_memory)
        elapsed = time.perf_counter() - epoch_start
        observed_indices = np.asarray(seen_indices, dtype=np.int64)
        if (seen != 10_847 or observed_indices.size != 10_847
                or np.unique(observed_indices).size != 10_847
                or not np.array_equal(np.sort(observed_indices), np.sort(indices))):
            raise RuntimeError(f"Epoch {epoch}: sampler failed fixed-index coverage audit")

        row = {
            "epoch": epoch, "epoch_seed": epoch_seed, "lr": lr,
            "loss_lc": loss_lc_sum / seen, "loss_bcl": loss_bcl_sum / seen,
            "loss_total": loss_sum / seen, "samples": seen,
            "sample_order_sha256": order_hash.hexdigest(),
            "epoch_seconds": elapsed, "peak_cuda_memory_bytes": peak_memory,
            "observed_noisy_class_counts": observed_counts.tolist(),
            "clean_labels_accessed_by_train_loop": False,
            "is_noisy_accessed_by_train_loop": False,
        }

        if epoch in CHECKPOINT_EPOCHS:
            ckpt_name = (f"epoch_{epoch:03d}_pre_decay.pth"
                         if epoch in (160, 180) else f"epoch_{epoch:03d}.pth")
            ckpt_path = checkpoint_dir / ckpt_name
            torch.save({
                "epoch": epoch, "seed": 42, "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "class_counts_for_lc_and_bcl": observed_counts.tolist(),
                "initial_model_state_sha256": initial_hash,
                "noise_manifest_sha256": file_sha256(manifest_path),
                "protocol": {
                    "epochs": EPOCHS, "batch_size": BATCH_SIZE, "precision": "fp32",
                    "lambda_lc": LAMBDA_LC, "mu_bcl": MU_BCL,
                    "temperature": TEMPERATURE, "lr": BASE_LR,
                    "momentum": MOMENTUM, "weight_decay": WEIGHT_DECAY,
                },
            }, ckpt_path)
            row["checkpoint"] = str(ckpt_path)
            diag = evaluate_train_checkpoint(
                model, base, indices, clean_labels, noisy_labels, is_noisy,
                clean_counts, device, args.num_workers, BATCH_SIZE, epoch, run_dir,
            )
            diagnostics.append(diag)
            row["train_diagnostics_file"] = str(
                run_dir / "train_diagnostics" / f"epoch_{epoch:03d}.json"
            )

        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[BCL_sym04] epoch {epoch:03d}/{EPOCHS} lr={lr:.5g} "
              f"LC={row['loss_lc']:.4f} BCL={row['loss_bcl']:.4f} "
              f"total={row['loss_total']:.4f} time={elapsed:.1f}s "
              f"peak={peak_memory / (1024**3):.2f}GiB")

    total_seconds = time.perf_counter() - total_start
    final_checkpoint = checkpoint_dir / "epoch_200.pth"
    if not final_checkpoint.is_file():
        raise RuntimeError("Missing final epoch-200 checkpoint")
    training_summary = {
        "run": "BCL_sym04_seed42", "epochs_completed": EPOCHS,
        "total_training_seconds": total_seconds,
        "max_peak_cuda_memory_bytes": max_peak_memory,
        "initial_model_state_sha256": initial_hash,
        "observed_noisy_class_counts": observed_counts.tolist(),
        "checkpoint_diagnostics": diagnostics,
        "checkpoint_paths": [str(p) for p in sorted(checkpoint_dir.glob("*.pth"))],
    }
    json_dump(run_dir / "training_summary.json", training_summary)

    # TEST is instantiated once, only after the complete 200-epoch run.
    test_set = CIFAR100BalancedTest(root=data_root, download=False)
    test_loader = DataLoader(
        test_set, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    model.eval()
    test_labels, test_predictions = [], []
    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            _, logits, _, _ = model(images, compute_contrastive=False)
            test_labels.append(labels.numpy())
            test_predictions.append(logits.argmax(dim=1).cpu().numpy())
    y = np.concatenate(test_labels).astype(np.int64)
    pred = np.concatenate(test_predictions).astype(np.int64)
    class_acc = np.asarray([
        np.mean(pred[y == c] == y[y == c]) for c in range(100)
    ], dtype=np.float64)
    groups = group_class_ids(clean_counts)
    test_metrics = {
        "split": "balanced CIFAR-100 TEST; opened once after epoch 200",
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
        "- TEST opened once after all 200 epochs and train-only checkpoint diagnostics.", "",
        "## Noise realization",
        f"- Requested rate: {manifest['noise_rate_requested']:.4f}",
        f"- Realized rate: {manifest['noise_rate_realized']:.6f}",
        f"- Flipped: {manifest['num_flipped']} / {manifest['num_train_samples']}",
        f"- Index SHA256: {manifest['indices_sha256']}",
        f"- Noisy-label SHA256: {manifest['noisy_labels_sha256']}",
        "- Full clean vs. observed noisy per-class counts: class_counts_comparison.csv in the run directory.",
        f"- Clean-label SHA256 (offline only): {manifest['clean_labels_sha256']}",
        f"- Mask SHA256 (offline only): {manifest['is_noisy_sha256']}", "",
        "## Five-checkpoint TRAIN diagnostics",
        "| Epoch | Clean subset | Many | Medium | Few | Noisy wrong-label memorization | Many | Medium | Few |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for d in diagnostics:
        clean = d["clean_subset_accuracy"]
        noisy = d["noisy_subset_wrong_label_memorization_rate"]
        total_clean = sum(v["total"] for v in clean.values())
        total_correct = sum(v["correct"] for v in clean.values())
        total_noisy = sum(v["total_noisy_examples"] for v in noisy.values())
        total_mem = sum(v["predicted_as_corrupted_observed_label"] for v in noisy.values())
        report_lines.append(
            f"| {d['epoch']} | {100*total_correct/max(total_clean,1):.2f}% | "
            f"{clean['Many']['accuracy_percent']:.2f}% | {clean['Medium']['accuracy_percent']:.2f}% | "
            f"{clean['Few']['accuracy_percent']:.2f}% | {100*total_mem/max(total_noisy,1):.2f}% | "
            f"{noisy['Many']['memorization_rate_percent']:.2f}% | "
            f"{noisy['Medium']['memorization_rate_percent']:.2f}% | "
            f"{noisy['Few']['memorization_rate_percent']:.2f}% |"
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
        f"- Initial model state SHA256: {initial_hash}",
        f"- Environment: {json.dumps(environment, ensure_ascii=False)}",
        f"- Checkpoints: {', '.join(training_summary['checkpoint_paths'])}",
        "", "## Leakage and split audit",
        "- Training batches expose only three views, observed noisy labels, and original source indices.",
        "- Clean labels and is_noisy mask are read only by the separate post-epoch eval diagnostic function.",
        "- Both loss functions and the LC prior use observed noisy labels/counts only.",
        "- TEST is instantiated once after epoch 200; no TEST access during label generation, training or checkpoint diagnostics.",
        "- This is synthetic instance-independent symmetric noise; it does not establish robustness to real-world annotation noise.",
    ]
    report_path = output_root.parent / "BCL_sym04_baseline_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Report: {report_path}")
    print(json.dumps(test_metrics, indent=2, ensure_ascii=False))
    return training_summary, test_metrics


def main():
    p = argparse.ArgumentParser(description="Train original BCL on CIFAR-100-LT with symmetric label noise")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--indices-file", type=Path, required=True)
    p.add_argument("--noise-dir", type=Path, default=Path("BCL_LTN/data"))
    p.add_argument("--output-root", type=Path, default=Path("BCL_LTN/outputs/BCL_sym04_seed42"))
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
