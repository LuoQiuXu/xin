#!/usr/bin/env python
"""Self-contained Stage-I trainer for the offline fusion-risk pilot.

This reproduces the released CIFAR-100-LT IF=100 Stage-I recipe (ResNet-32,
cross entropy, image mixup, 200 epochs, five-epoch warm-up, LR drops after
epochs 160 and 180) without depending on the repository's old yacs utilities.
The checkpoint schema is compatible with ``export_cifar100_minival_features.py``.
"""

from __future__ import annotations

import argparse
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models import resnet_cifar


def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def long_tailed_indices(
    targets: np.ndarray,
    imbalance_factor: float,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    num_classes = 100
    image_max = targets.size / num_classes
    counts = np.asarray(
        [
            int(image_max * imbalance_factor ** (class_id / (num_classes - 1.0)))
            for class_id in range(num_classes)
        ],
        dtype=np.int64,
    )
    rng = np.random.RandomState(seed)
    selected: list[np.ndarray] = []
    for class_id, count in enumerate(counts):
        indices = np.flatnonzero(targets == class_id)
        rng.shuffle(indices)
        selected.append(indices[:count])
    return np.concatenate(selected), counts


def learning_rate(epoch: int, base_lr: float) -> float:
    one_based_epoch = epoch + 1
    if one_based_epoch <= 5:
        return base_lr * one_based_epoch / 5.0
    if one_based_epoch > 180:
        return base_lr * 0.01
    if one_based_epoch > 160:
        return base_lr * 0.1
    return base_lr


def mixup(
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    permutation = torch.randperm(images.size(0), device=images.device)
    mixed = lam * images + (1.0 - lam) * images[permutation]
    return mixed, labels, labels[permutation], lam


def top1_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float((logits.argmax(dim=1) == labels).float().mean().item())


def train_epoch(
    backbone: nn.Module,
    classifier: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    alpha: float,
    amp_enabled: bool,
    print_freq: int,
    max_batches: int,
) -> tuple[float, float]:
    backbone.train()
    classifier.train()
    criterion = nn.CrossEntropyLoss()
    loss_sum = 0.0
    accuracy_sum = 0.0
    sample_count = 0
    started = time.time()

    for batch_index, (images, labels) in enumerate(loader):
        if max_batches > 0 and batch_index >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        images, labels_a, labels_b, lam = mixup(images, labels, alpha)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            features = backbone(images)
            logits = classifier(features)
            loss = lam * criterion(logits, labels_a) + (1.0 - lam) * criterion(
                logits, labels_b
            )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.size(0)
        loss_sum += float(loss.item()) * batch_size
        accuracy_sum += top1_accuracy(logits.detach(), labels) * batch_size
        sample_count += batch_size
        if batch_index % print_freq == 0:
            print(
                f"  batch {batch_index:4d}/{len(loader):4d} "
                f"loss={loss_sum/sample_count:.4f} "
                f"acc={accuracy_sum/sample_count:.4f} "
                f"elapsed={time.time()-started:.1f}s",
                flush=True,
            )
    return loss_sum / sample_count, accuracy_sum / sample_count


@torch.inference_mode()
def evaluate(
    backbone: nn.Module,
    classifier: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    backbone.eval()
    classifier.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    loss_sum = 0.0
    correct = 0
    sample_count = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = classifier(backbone(images))
        loss_sum += float(criterion(logits, labels).item())
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        sample_count += labels.size(0)
    return loss_sum / sample_count, correct / sample_count


def save_checkpoint(
    output_dir: Path,
    epoch: int,
    backbone: nn.Module,
    classifier: nn.Module,
    optimizer: torch.optim.Optimizer,
    best_accuracy: float,
    is_best: bool,
) -> None:
    state = {
        "epoch": epoch + 1,
        "state_dict_model": backbone.state_dict(),
        "state_dict_classifier": classifier.state_dict(),
        "state_dict_optimizer": optimizer.state_dict(),
        "best_acc1": best_accuracy,
        "its_ece": 0.0,
    }
    current_path = output_dir / "current.pth.tar"
    torch.save(state, current_path)
    if is_best:
        shutil.copyfile(current_path, output_dir / "model_best.pth.tar")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train CIFAR-100-LT Stage-I for minival")
    parser.add_argument("--data-root", type=Path, default=Path("data/cifar100"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/stage1_minival"))
    parser.add_argument("--imbalance-factor", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--mixup-alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--print-freq", type=int, default=40)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    amp_enabled = bool(args.amp and device.type == "cuda")
    set_seed(args.seed, args.deterministic)

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010),
            ),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010),
            ),
        ]
    )
    full_train = datasets.CIFAR100(
        root=args.data_root,
        train=True,
        transform=train_transform,
        download=args.download,
    )
    test_dataset = datasets.CIFAR100(
        root=args.data_root,
        train=False,
        transform=eval_transform,
        download=args.download,
    )
    lt_indices, class_counts = long_tailed_indices(
        np.asarray(full_train.targets), args.imbalance_factor, seed=0
    )
    train_loader = DataLoader(
        Subset(full_train, lt_indices.tolist()),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=max(args.batch_size, 256),
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    backbone = resnet_cifar.resnet32_fe().to(device)
    classifier = resnet_cifar.Classifier(feat_in=64, num_classes=100).to(device)
    optimizer = torch.optim.SGD(
        list(backbone.parameters()) + list(classifier.parameters()),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    start_epoch = 0
    best_accuracy = 0.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        backbone.load_state_dict(checkpoint["state_dict_model"])
        classifier.load_state_dict(checkpoint["state_dict_classifier"])
        if "state_dict_optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["state_dict_optimizer"])
        start_epoch = int(checkpoint.get("epoch", 0))
        best_accuracy = float(checkpoint.get("best_acc1", 0.0))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"device={device}, amp={amp_enabled}, LT samples={lt_indices.size}, "
        f"class count={class_counts.max()}->{class_counts.min()}, epochs={args.epochs}",
        flush=True,
    )
    for epoch in range(start_epoch, args.epochs):
        lr = learning_rate(epoch, args.lr)
        for group in optimizer.param_groups:
            group["lr"] = lr
        print(f"epoch {epoch+1}/{args.epochs}, lr={lr:.5f}", flush=True)
        train_loss, train_acc = train_epoch(
            backbone,
            classifier,
            train_loader,
            optimizer,
            scaler,
            device,
            args.mixup_alpha,
            amp_enabled,
            args.print_freq,
            args.max_batches,
        )
        test_loss, test_acc = evaluate(backbone, classifier, test_loader, device)
        previous_best = best_accuracy
        best_accuracy = max(best_accuracy, test_acc)
        save_checkpoint(
            args.output_dir,
            epoch,
            backbone,
            classifier,
            optimizer,
            best_accuracy,
            test_acc >= previous_best,
        )
        print(
            f"epoch {epoch+1}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"test_loss={test_loss:.4f}, test_acc={test_acc:.4f}, "
            f"best={best_accuracy:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
