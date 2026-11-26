"""Train one observed-label BCL student with a genuine EMA teacher for Stage 1.

This script is intentionally TEST-free. It reads only fixed TRAIN pixels,
raw TRAIN IDs, observed_labels.npy and its non-oracle manifest. It never opens
the regime's truth/ directory. The EMA teacher is updated after every student
optimizer step; BatchNorm buffers are copied from the student.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from BCL_AUDIT.cifar100lt.data import build_cifar_transforms
from BCL_AUDIT.cifar100lt.losses import BalancedContrastiveLoss, LogitCompensatedCE
from BCL_AUDIT.cifar100lt.models import BCLCIFARModel
from BCL_AUDIT.cifar100lt.train import (
    BASE_LR, BATCH_SIZE, EPOCHS, LR_GAMMA, LR_MILESTONES, MOMENTUM, WEIGHT_DECAY,
    epoch_seed_for, file_sha256, learning_rate_for_epoch, seed_worker, set_seed,
    state_dict_sha256,
)
from BCL_LTN.audit_epoch160_margin_reliability import load_cifar100_train_pixels_only
from BCL_LTN.noise_robust_losses import EMATeacher
from BCL_LTN.train_noisy_bcl import bcl_sgd_step


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "H2T" / "data" / "cifar100"
DEFAULT_INDICES = ROOT / "BCL_AUDIT" / "outputs" / "data_audit_recheck_20261122_01" / "cifar100lt_if100_train_indices_seed0.npy"
N, K, SEED = 10_847, 100, 42


class ObservedOnlyThreeViewDataset(Dataset):
    def __init__(self, pixels, indices, observed, transforms):
        self.pixels = pixels
        self.indices = indices
        self.observed = observed
        self.v1, self.sim, self.eval_transform = transforms

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        raw_id = int(self.indices[position])
        image = Image.fromarray(self.pixels[raw_id])
        return (self.v1(image), self.sim(image), self.sim(image)), int(self.observed[position]), raw_id


class EvalPixelsDataset(Dataset):
    def __init__(self, pixels, indices, eval_transform):
        self.pixels = pixels
        self.indices = indices
        self.transform = eval_transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        raw_id = int(self.indices[position])
        return self.transform(Image.fromarray(self.pixels[raw_id])), position


def install_truth_read_guard(regime_dir: Path) -> None:
    truth_root = (regime_dir / "truth").resolve()
    def audit(event, args):
        if event != "open" or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if path == truth_root or truth_root in path.parents:
            raise PermissionError(f"Training is forbidden from opening offline truth: {path}")
        if path.name in {"clean_labels.npy", "is_noisy.npy", "true_transition.npy", "truth_manifest.json"}:
            raise PermissionError(f"Training is forbidden from opening offline truth file: {path}")
    sys.addaudithook(audit)


def write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


@torch.inference_mode()
def extract_teacher_logits(teacher: torch.nn.Module, pixels, indices, eval_transform, device):
    teacher.eval()
    dataset = EvalPixelsDataset(pixels, indices, eval_transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    logits_out = np.empty((len(dataset), K), dtype=np.float32)
    seen = np.zeros(len(dataset), dtype=np.bool_)
    for images, positions in loader:
        positions_np = positions.numpy()
        if seen[positions_np].any():
            raise RuntimeError("Duplicate TRAIN position while extracting EMA logits")
        _, logits, _, _ = teacher(images.to(device, non_blocking=True), compute_contrastive=False)
        if logits.shape != (len(positions_np), K) or not bool(torch.isfinite(logits).all()):
            raise RuntimeError("Invalid EMA teacher logits")
        logits_out[positions_np] = logits.cpu().numpy()
        seen[positions_np] = True
    if not seen.all():
        raise RuntimeError("EMA teacher logits do not cover the fixed TRAIN subset")
    return logits_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--indices-file", type=Path, default=DEFAULT_INDICES)
    parser.add_argument("--observed-labels", type=Path, required=True)
    parser.add_argument("--regime-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()

    if not 0.0 <= args.ema_decay < 1.0:
        raise ValueError("ema-decay must be in [0, 1)")
    for path in (args.indices_file, args.observed_labels, args.regime_manifest,
                 args.data_root / "cifar-100-python" / "train"):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {args.output_dir}")

    manifest = json.loads(args.regime_manifest.read_text(encoding="utf-8"))
    indices = np.load(args.indices_file, allow_pickle=False)
    observed = np.load(args.observed_labels, allow_pickle=False)
    if (indices.shape != (N,) or indices.dtype != np.int64 or np.unique(indices).size != N
            or observed.shape != (N,) or observed.dtype != np.int64
            or np.any((observed < 0) | (observed >= K))):
        raise RuntimeError("Invalid fixed TRAIN IDs or observed labels")
    if (manifest.get("num_samples") != N or manifest.get("num_classes") != K
            or manifest.get("observed_labels_sha256") != file_sha256(args.observed_labels)
            or manifest.get("observed_class_counts") != np.bincount(observed, minlength=K).tolist()):
        raise RuntimeError("Observed-label file does not match its non-oracle manifest")
    if manifest.get("training_access_policy", "").find("observed_labels.npy only") < 0:
        raise RuntimeError("Missing observed-only training access policy")

    install_truth_read_guard(args.observed_labels.parent)
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Stage-1 EMA teacher run")
    torch.cuda.init()
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if torch.is_autocast_enabled("cuda"):
        raise RuntimeError("AMP must remain disabled for the locked FP32 protocol")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    pixels = load_cifar100_train_pixels_only(args.data_root)
    transforms = build_cifar_transforms()
    dataset = ObservedOnlyThreeViewDataset(pixels, indices, observed, transforms)
    counts = np.bincount(observed, minlength=K).astype(np.int64)
    generator = torch.Generator()
    generator.manual_seed(SEED)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=0,
        pin_memory=True, generator=generator, worker_init_fn=seed_worker,
    )

    set_seed(SEED)
    student = BCLCIFARModel().to(device)
    teacher = EMATeacher(student, decay=args.ema_decay)
    criterion_lc = LogitCompensatedCE(counts).to(device)
    criterion_bcl = BalancedContrastiveLoss(counts).to(device)
    optimizer = torch.optim.SGD(
        student.parameters(), lr=BASE_LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
    )

    environment = {
        "regime": manifest.get("regime"),
        "seed": SEED,
        "epochs": 1 if args.smoke_only else EPOCHS,
        "batch_size": BATCH_SIZE,
        "precision": "fp32",
        "amp": False,
        "tf32": False,
        "num_workers": 0,
        "optimizer": "SGD",
        "base_lr": BASE_LR,
        "momentum": MOMENTUM,
        "weight_decay": WEIGHT_DECAY,
        "lr_milestones": list(LR_MILESTONES),
        "lr_gamma": LR_GAMMA,
        "ema_decay": args.ema_decay,
        "ema_buffer_policy": "copy_student_buffers_each_update",
        "observed_labels_sha256": file_sha256(args.observed_labels),
        "indices_file_sha256": file_sha256(args.indices_file),
        "regime_manifest_sha256": file_sha256(args.regime_manifest),
        "clean_truth_loaded": False,
        "official_test_opened": False,
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    write_json(args.output_dir / "environment.json", environment)

    if args.smoke_only:
        epoch_seed = epoch_seed_for(SEED, 0)
        set_seed(epoch_seed)
        generator.manual_seed(epoch_seed)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate_for_epoch(0)
        student.train()
        views, labels, raw_ids = next(iter(loader))
        before = state_dict_sha256(teacher.model.state_dict())
        metrics = bcl_sgd_step(
            student, optimizer, criterion_lc, criterion_bcl, views, labels, device, check_smoke=True
        )
        teacher.update(student)
        after = state_dict_sha256(teacher.model.state_dict())
        receipt = {
            **environment,
            **metrics,
            "passed": True,
            "epoch_seed": epoch_seed,
            "raw_ids_count": int(len(raw_ids)),
            "ema_updates": teacher.num_updates,
            "ema_state_changed": before != after,
            "formal_training_started": False,
        }
        write_json(args.output_dir / "smoke_receipt.json", receipt)
        print(json.dumps(receipt, indent=2, ensure_ascii=False))
        return

    history_path = args.output_dir / "history.jsonl"
    total_updates = 0
    started = time.perf_counter()
    with history_path.open("x", encoding="utf-8") as history:
        for epoch_index in range(EPOCHS):
            epoch = epoch_index + 1
            epoch_seed = epoch_seed_for(SEED, epoch_index)
            set_seed(epoch_seed)
            generator.manual_seed(epoch_seed)
            lr = learning_rate_for_epoch(epoch_index)
            for group in optimizer.param_groups:
                group["lr"] = lr
            student.train()
            seen = 0
            sums = np.zeros(3, dtype=np.float64)
            for views, labels, _raw_ids in loader:
                step = bcl_sgd_step(student, optimizer, criterion_lc, criterion_bcl, views, labels, device)
                teacher.update(student)
                sums += np.asarray(
                    [step["loss_lc"], step["loss_bcl"], step["loss_total"]], dtype=np.float64
                ) * len(labels)
                seen += len(labels)
                total_updates += 1
            if seen != N:
                raise RuntimeError(f"Epoch {epoch}: observed {seen} samples, expected {N}")
            row = {
                "epoch": epoch,
                "epoch_seed": epoch_seed,
                "lr": lr,
                "samples": seen,
                "optimizer_updates": total_updates,
                "ema_updates": teacher.num_updates,
                "loss_lc": float(sums[0] / seen),
                "loss_bcl": float(sums[1] / seen),
                "loss_total": float(sums[2] / seen),
                "clean_truth_loaded": False,
            }
            history.write(json.dumps(row, allow_nan=False) + "\n")
            history.flush()
            print(
                f"[EMA-BCL] epoch {epoch:03d}/{EPOCHS} lr={lr:.5g} "
                f"LC={row['loss_lc']:.5f} BCL={row['loss_bcl']:.5f} "
                f"total={row['loss_total']:.5f} EMA-updates={teacher.num_updates}",
                flush=True,
            )

    if total_updates != EPOCHS * len(loader) or teacher.num_updates != total_updates:
        raise RuntimeError("EMA teacher update count does not match optimizer updates")

    student.eval()
    teacher.model.eval()
    ema_logits = extract_teacher_logits(teacher.model, pixels, indices, transforms[2], device)
    logits_path = args.output_dir / "ema_train_logits.npy"
    np.save(logits_path, ema_logits, allow_pickle=False)
    student_sha = state_dict_sha256(student.state_dict())
    teacher_sha = state_dict_sha256(teacher.model.state_dict())
    checkpoint_path = args.output_dir / "ema_teacher_checkpoint.pth"
    with checkpoint_path.open("xb") as stream:
        torch.save({
            "epoch": EPOCHS,
            "seed": SEED,
            "student_model_state": student.state_dict(),
            "ema_model_state": teacher.model.state_dict(),
            "student_state_sha256": student_sha,
            "ema_state_sha256": teacher_sha,
            "ema_decay": teacher.decay,
            "ema_updates": teacher.num_updates,
            "ema_buffer_policy": "copy_student_buffers_each_update",
            "observed_labels_sha256": environment["observed_labels_sha256"],
            "indices_file_sha256": environment["indices_file_sha256"],
            "regime_manifest_sha256": environment["regime_manifest_sha256"],
            "protocol": environment,
        }, stream)
    summary = {
        **environment,
        "epochs_completed": EPOCHS,
        "optimizer_updates": total_updates,
        "ema_updates": teacher.num_updates,
        "student_state_sha256": student_sha,
        "ema_state_sha256": teacher_sha,
        "ema_logits_sha256": file_sha256(logits_path),
        "ema_logits_shape": list(ema_logits.shape),
        "ema_checkpoint_sha256": file_sha256(checkpoint_path),
        "total_training_seconds": time.perf_counter() - started,
        "clean_truth_loaded": False,
        "official_test_opened": False,
    }
    write_json(args.output_dir / "training_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
