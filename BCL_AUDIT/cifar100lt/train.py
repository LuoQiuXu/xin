from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import CIFAR100BalancedTest, CIFAR100LT, cifar100lt_class_counts
from .losses import BalancedContrastiveLoss, LogitCompensatedCE
from .models import BCLCIFARModel


EPOCHS = 200
BATCH_SIZE = 256
TRAIN_SEED = 42
BASE_LR = 0.15
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9
WARMUP_EPOCHS = 5
LR_MILESTONES = (160, 180)
LR_GAMMA = 0.1
LAMBDA_LC = 2.0
MU_BCL = 0.6
TEMPERATURE = 0.1
CHECKPOINT_EPOCHS = (40, 100, 160, 180, 200)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def learning_rate_for_epoch(epoch_index: int) -> float:
    """0-based epoch index; decay starts after 160 and 180 completed epochs."""
    if epoch_index < WARMUP_EPOCHS:
        return BASE_LR * float(epoch_index + 1) / float(WARMUP_EPOCHS)
    if epoch_index >= LR_MILESTONES[1]:
        return BASE_LR * (LR_GAMMA ** 2)
    if epoch_index >= LR_MILESTONES[0]:
        return BASE_LR * LR_GAMMA
    return BASE_LR


def make_train_loader(
    data_root: Path,
    mode: str,
    indices_file: Path,
    num_workers: int,
    seed: int,
):
    dataset = CIFAR100LT(
        root=data_root,
        mode=mode,
        indices_file=indices_file,
        split_seed=0,
        download=True,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        generator=generator,
        worker_init_fn=seed_worker,
    )
    return dataset, loader


def _save_checkpoint(
    path: Path,
    model: BCLCIFARModel,
    optimizer: torch.optim.Optimizer,
    epoch_completed: int,
    group: str,
    seed: int,
    class_counts: np.ndarray,
    last_lr: float,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch_completed,
            "group": group,
            "seed": seed,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "class_counts": class_counts.tolist(),
            "last_lr": last_lr,
            "protocol": {
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "precision": "fp32",
                "lambda_lc": LAMBDA_LC,
                "mu_bcl": MU_BCL,
                "temperature": TEMPERATURE,
                "base_lr": BASE_LR,
                "weight_decay": WEIGHT_DECAY,
                "momentum": MOMENTUM,
            },
        },
        path,
    )


def train_group(
    group: str,
    data_root: Path,
    output_root: Path,
    indices_file: Path,
    initial_state: dict,
    num_workers: int,
    device: torch.device,
):
    if group not in {"LC_only", "BCL"}:
        raise ValueError(group)
    set_seed(TRAIN_SEED)

    mode = "lc" if group == "LC_only" else "bcl"
    dataset, loader = make_train_loader(
        data_root=data_root,
        mode=mode,
        indices_file=indices_file,
        num_workers=num_workers,
        seed=TRAIN_SEED,
    )
    class_counts = cifar100lt_class_counts()

    model = BCLCIFARModel()
    model.load_state_dict(initial_state, strict=True)
    model = model.to(device)

    criterion_lc = LogitCompensatedCE(class_counts).to(device)
    criterion_bcl = BalancedContrastiveLoss(
        class_counts=class_counts,
        temperature=TEMPERATURE,
    ).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=BASE_LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )

    run_dir = output_root / f"{group}_seed{TRAIN_SEED}"
    checkpoint_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    history_path = run_dir / "history.jsonl"
    if history_path.exists():
        history_path.unlink()

    epoch_summaries = []
    total_start = time.perf_counter()

    for epoch_index in range(EPOCHS):
        lr = learning_rate_for_epoch(epoch_index)
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = lr

        model.train()
        loss_lc_sum = 0.0
        loss_bcl_sum = 0.0
        loss_total_sum = 0.0
        correct = 0
        seen = 0

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        epoch_start = time.perf_counter()

        for images, labels in loader:
            labels = labels.to(device, non_blocking=True)
            batch_size = int(labels.shape[0])

            if group == "LC_only":
                images = images.to(device, non_blocking=True)
                _, logits, _, _ = model(images, compute_contrastive=False)
                loss_lc = criterion_lc(logits, labels)
                loss_bcl = logits.new_zeros(())
                loss = LAMBDA_LC * loss_lc
            else:
                view1, view2, view3 = [
                    view.to(device, non_blocking=True) for view in images
                ]
                # Preserve BCL's one-forward three-view BatchNorm semantics.
                all_views = torch.cat([view1, view2, view3], dim=0)
                _, all_logits, all_embeddings, class_prototypes = model(
                    all_views,
                    compute_contrastive=True,
                )
                logits = all_logits[:batch_size]
                z2 = all_embeddings[batch_size : 2 * batch_size]
                z3 = all_embeddings[2 * batch_size : 3 * batch_size]
                two_view_features = torch.stack([z2, z3], dim=1)

                loss_lc = criterion_lc(logits, labels)
                loss_bcl = criterion_bcl(
                    class_prototypes,
                    two_view_features,
                    labels,
                )
                loss = LAMBDA_LC * loss_lc + MU_BCL * loss_bcl

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            loss_lc_sum += float(loss_lc.detach().item()) * batch_size
            loss_bcl_sum += float(loss_bcl.detach().item()) * batch_size
            loss_total_sum += float(loss.detach().item()) * batch_size
            correct += int((logits.detach().argmax(dim=1) == labels).sum().item())
            seen += batch_size

        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_memory_bytes = int(torch.cuda.max_memory_allocated(device))
        else:
            peak_memory_bytes = 0
        epoch_seconds = time.perf_counter() - epoch_start

        summary = {
            "epoch": epoch_index + 1,
            "lr": lr,
            "loss_lc": loss_lc_sum / max(seen, 1),
            "loss_bcl": loss_bcl_sum / max(seen, 1),
            "loss_total": loss_total_sum / max(seen, 1),
            "train_top1_raw_logits": 100.0 * correct / max(seen, 1),
            "epoch_seconds": epoch_seconds,
            "peak_cuda_memory_bytes": peak_memory_bytes,
            "num_train_samples": seen,
        }
        human_epoch = epoch_index + 1
        if human_epoch in CHECKPOINT_EPOCHS:
            if human_epoch == 160:
                milestone = "epoch_160_pre_decay"
            elif human_epoch == 180:
                milestone = "epoch_180_pre_decay"
            else:
                milestone = f"epoch_{human_epoch:03d}"
            checkpoint_path = checkpoint_dir / f"{milestone}.pth"
            _save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                epoch_completed=human_epoch,
                group=group,
                seed=TRAIN_SEED,
                class_counts=class_counts,
                last_lr=lr,
            )
            summary["checkpoint"] = str(checkpoint_path)

        epoch_summaries.append(summary)
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")

        print(
            f"[{group}] epoch {human_epoch:03d}/{EPOCHS} "
            f"lr={lr:.6g} lc={summary['loss_lc']:.4f} "
            f"bcl={summary['loss_bcl']:.4f} "
            f"top1={summary['train_top1_raw_logits']:.2f} "
            f"time={epoch_seconds:.1f}s "
            f"peak_cuda={peak_memory_bytes / (1024**3):.2f}GiB"
        )

    total_seconds = time.perf_counter() - total_start
    result = {
        "group": group,
        "seed": TRAIN_SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "precision": "fp32",
        "total_training_seconds": total_seconds,
        "mean_epoch_seconds": float(np.mean([x["epoch_seconds"] for x in epoch_summaries])),
        "max_epoch_peak_cuda_memory_bytes": max(
            [x["peak_cuda_memory_bytes"] for x in epoch_summaries], default=0
        ),
        "final_epoch": epoch_summaries[-1],
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_paths": [
            item["checkpoint"] for item in epoch_summaries if "checkpoint" in item
        ],
    }
    with (run_dir / "training_summary.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return checkpoint_dir / "epoch_200.pth", result


def _safe_torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


@torch.inference_mode()
def evaluate_final_checkpoint(
    group: str,
    checkpoint_path: Path,
    data_root: Path,
    output_root: Path,
    num_workers: int,
    device: torch.device,
):
    # This function is called only after BOTH training groups have finished.
    test_set = CIFAR100BalancedTest(root=data_root, download=True)
    loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    checkpoint = _safe_torch_load(checkpoint_path, device)
    model = BCLCIFARModel().to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()

    all_labels = []
    all_predictions = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        _, logits, _, _ = model(images, compute_contrastive=False)
        predictions = logits.argmax(dim=1)
        all_labels.append(labels.cpu().numpy())
        all_predictions.append(predictions.cpu().numpy())

    labels_np = np.concatenate(all_labels).astype(np.int64)
    predictions_np = np.concatenate(all_predictions).astype(np.int64)
    class_counts = cifar100lt_class_counts()

    class_accuracy = np.zeros(100, dtype=np.float64)
    for class_id in range(100):
        mask = labels_np == class_id
        class_accuracy[class_id] = float(
            np.mean(predictions_np[mask] == labels_np[mask])
        )

    many_ids = np.where(class_counts > 100)[0]
    medium_ids = np.where((class_counts >= 20) & (class_counts <= 100))[0]
    few_ids = np.where(class_counts < 20)[0]

    confusion = np.bincount(
        labels_np * 100 + predictions_np,
        minlength=100 * 100,
    ).reshape(100, 100)

    metrics = {
        "group": group,
        "seed": TRAIN_SEED,
        "checkpoint": str(checkpoint_path),
        "test_split": "balanced CIFAR-100 test split; evaluated after both training runs",
        "overall_top1": 100.0 * float(np.mean(predictions_np == labels_np)),
        "many_top1_macro_class": 100.0 * float(np.mean(class_accuracy[many_ids])),
        "medium_top1_macro_class": 100.0 * float(np.mean(class_accuracy[medium_ids])),
        "few_top1_macro_class": 100.0 * float(np.mean(class_accuracy[few_ids])),
        "num_many_classes": int(len(many_ids)),
        "num_medium_classes": int(len(medium_ids)),
        "num_few_classes": int(len(few_ids)),
    }

    eval_dir = output_root / f"{group}_seed{TRAIN_SEED}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        eval_dir / "final_test_predictions.npz",
        labels=labels_np,
        predictions=predictions_np,
        class_accuracy=class_accuracy,
    )
    np.save(eval_dir / "final_test_confusion_matrix.npy", confusion)
    with (eval_dir / "final_test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(
        f"[FINAL TEST] {group}: overall={metrics['overall_top1']:.2f}, "
        f"many={metrics['many_top1_macro_class']:.2f}, "
        f"medium={metrics['medium_top1_macro_class']:.2f}, "
        f"few={metrics['few_top1_macro_class']:.2f}"
    )
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="One-seed, 200-epoch FP32 LC-only vs BCL CIFAR-100-LT pilot"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("BCL_AUDIT/outputs"))
    parser.add_argument(
        "--indices-file",
        type=Path,
        default=Path(
            "BCL_AUDIT/outputs/data_audit/cifar100lt_if100_train_indices_seed0.npy"
        ),
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    indices_file = args.indices_file.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    indices_file.parent.mkdir(parents=True, exist_ok=True)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")

    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    try:
        import torchvision
        print(f"torchvision: {torchvision.__version__}")
    except Exception:
        pass
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(f"CUDA runtime: {torch.version.cuda}")

    # Create a single initial state, then reload it for both independent runs.
    set_seed(TRAIN_SEED)
    template_model = BCLCIFARModel()
    initial_state = {
        key: value.detach().cpu().clone()
        for key, value in template_model.state_dict().items()
    }
    del template_model
    initial_state_path = output_root / f"initial_model_seed{TRAIN_SEED}.pth"
    torch.save({"model_state": initial_state, "seed": TRAIN_SEED}, initial_state_path)

    # Run the two groups sequentially. No test dataset is instantiated here.
    lc_checkpoint, lc_summary = train_group(
        group="LC_only",
        data_root=data_root,
        output_root=output_root,
        indices_file=indices_file,
        initial_state=initial_state,
        num_workers=args.num_workers,
        device=device,
    )
    bcl_checkpoint, bcl_summary = train_group(
        group="BCL",
        data_root=data_root,
        output_root=output_root,
        indices_file=indices_file,
        initial_state=initial_state,
        num_workers=args.num_workers,
        device=device,
    )

    # Only now instantiate the balanced test dataset for final reporting.
    lc_metrics = evaluate_final_checkpoint(
        "LC_only", lc_checkpoint, data_root, output_root, args.num_workers, device
    )
    bcl_metrics = evaluate_final_checkpoint(
        "BCL", bcl_checkpoint, data_root, output_root, args.num_workers, device
    )

    pair_summary = {
        "seed": TRAIN_SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "precision": "fp32",
        "initial_model_checkpoint": str(initial_state_path),
        "indices_file": str(indices_file),
        "LC_only_training": lc_summary,
        "BCL_training": bcl_summary,
        "LC_only_final_test": lc_metrics,
        "BCL_final_test": bcl_metrics,
        "known_caveat": (
            "BCL uses one concatenated 3-view backbone forward, so BatchNorm sees "
            "v1+v2+v3; LC-only uses only v1 in this first pilot."
        ),
    }
    with (output_root / "paired_pilot_summary.json").open("w", encoding="utf-8") as f:
        json.dump(pair_summary, f, indent=2, ensure_ascii=False)
    print(f"Paired pilot summary: {output_root / 'paired_pilot_summary.json'}")


if __name__ == "__main__":
    main()
