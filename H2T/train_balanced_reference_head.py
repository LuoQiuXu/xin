"""Train a class-balanced cRT reference head on frozen Stage-I features.

The feature archive and original CE head must come from the same Stage-I
checkpoint.  Only a newly initialized linear classifier is trained.  Test
metrics are reported for diagnostics, but the saved reference is always the
last epoch; no test-set model selection is performed.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from models import resnet_cifar


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Balanced cRT reference head on frozen features"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--lr-factor", type=float, default=0.2)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def balanced_schedule(
    labels: np.ndarray,
    epochs: int,
    samples_per_epoch: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    class_indices = {class_id: np.flatnonzero(labels == class_id) for class_id in classes}
    result = np.empty(epochs * samples_per_epoch, dtype=np.int64)
    for epoch in range(epochs):
        start = epoch * samples_per_epoch
        chosen_classes = rng.integers(0, classes.size, size=samples_per_epoch)
        epoch_indices = np.empty(samples_per_epoch, dtype=np.int64)
        for class_id in classes:
            mask = chosen_classes == class_id
            count = int(mask.sum())
            if count:
                epoch_indices[mask] = rng.choice(
                    class_indices[int(class_id)],
                    size=count,
                    replace=True,
                )
        result[start : start + samples_per_epoch] = epoch_indices
    return result


@torch.no_grad()
def evaluate(
    classifier: resnet_cifar.Classifier,
    features: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
) -> dict[str, Any]:
    classifier.eval()
    correct = torch.zeros(100, device=features.device)
    count = torch.zeros(100, device=features.device)
    loss_sum = 0.0
    for start in range(0, labels.numel(), batch_size):
        stop = min(start + batch_size, labels.numel())
        logits = classifier.fc(features[start:stop])
        target = labels[start:stop]
        loss_sum += float(F.cross_entropy(logits, target, reduction="sum"))
        prediction = logits.argmax(dim=1)
        count.scatter_add_(0, target, torch.ones_like(target, dtype=torch.float32))
        correct.scatter_add_(0, target, (prediction == target).float())
    class_accuracy = correct / torch.clamp(count, min=1.0)
    return {
        "loss": loss_sum / labels.numel(),
        "overall": float(correct.sum() / count.sum()),
        "many": float(class_accuracy[:36].mean()),
        "medium": float(class_accuracy[36:71].mean()),
        "few": float(class_accuracy[71:].mean()),
        "balanced_accuracy": float(class_accuracy.mean()),
        "class_accuracy": class_accuracy.cpu().tolist(),
    }


def compact(metrics: dict[str, Any]) -> dict[str, float]:
    return {key: value for key, value in metrics.items() if key != "class_accuracy"}


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    archive_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    archive = np.load(archive_path, allow_pickle=False)
    train_features_np = archive["train_features"].astype(np.float32, copy=False)
    train_labels_np = archive["train_labels"].astype(np.int64, copy=False)
    test_features_np = archive["test_features"].astype(np.float32, copy=False)
    test_labels_np = archive["test_labels"].astype(np.int64, copy=False)
    source_weight = archive["classifier_weight"].astype(np.float32, copy=False)
    source_bias = archive["classifier_bias"].astype(np.float32, copy=False)

    train_features = torch.from_numpy(train_features_np).to(device)
    train_labels = torch.from_numpy(train_labels_np).to(device)
    test_features = torch.from_numpy(test_features_np).to(device)
    test_labels = torch.from_numpy(test_labels_np).to(device)

    source_classifier = resnet_cifar.Classifier(feat_in=64, num_classes=100).to(device)
    source_classifier.fc.weight.data.copy_(torch.from_numpy(source_weight).to(device))
    source_classifier.fc.bias.data.copy_(torch.from_numpy(source_bias).to(device))
    source_train = evaluate(
        source_classifier,
        train_features,
        train_labels,
        args.batch_size,
    )
    source_test = evaluate(
        source_classifier,
        test_features,
        test_labels,
        args.batch_size,
    )

    # cRT: discard the imbalanced CE classifier and retrain a new linear head
    # with class-balanced recipient sampling while keeping all features fixed.
    seed_everything(args.seed)
    classifier = resnet_cifar.Classifier(feat_in=64, num_classes=100).to(device)
    optimizer = torch.optim.SGD(
        classifier.parameters(),
        lr=args.lr * args.lr_factor,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    samples_per_epoch = train_labels.numel()
    schedule = balanced_schedule(
        train_labels_np,
        args.epochs,
        samples_per_epoch,
        args.seed + 50_021,
    )
    history: list[dict[str, Any]] = []

    print(
        f"device={device}, train={tuple(train_features.shape)}, "
        f"test={tuple(test_features.shape)}"
    )
    print("source train", compact(source_train))
    print("source test ", compact(source_test))

    for epoch in range(args.epochs):
        classifier.train()
        cosine = 0.5 * (1.0 + math.cos(epoch / args.epochs * math.pi))
        learning_rate = args.lr_factor * args.lr * cosine
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

        epoch_indices = schedule[
            epoch * samples_per_epoch : (epoch + 1) * samples_per_epoch
        ]
        loss_sum = 0.0
        correct = 0
        for start in range(0, samples_per_epoch, args.batch_size):
            stop = min(start + args.batch_size, samples_per_epoch)
            indices = torch.from_numpy(epoch_indices[start:stop]).to(device)
            target = train_labels[indices]
            logits = classifier.fc(train_features[indices])
            loss = F.cross_entropy(logits, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * target.numel()
            correct += int((logits.argmax(dim=1) == target).sum())

        train_metrics = evaluate(
            classifier,
            train_features,
            train_labels,
            args.batch_size,
        )
        test_metrics = evaluate(
            classifier,
            test_features,
            test_labels,
            args.batch_size,
        )
        row = {
            "epoch": epoch + 1,
            "lr": learning_rate,
            "balanced_stream_loss": loss_sum / samples_per_epoch,
            "balanced_stream_accuracy": correct / samples_per_epoch,
            "train": train_metrics,
            "test": test_metrics,
        }
        history.append(row)
        print(
            f"epoch {epoch + 1:02d}/{args.epochs}: "
            f"balanced_stream={row['balanced_stream_accuracy']:.4f}, "
            f"test={test_metrics['overall']:.4f}, "
            f"M/Md/F={test_metrics['many']:.4f}/"
            f"{test_metrics['medium']:.4f}/{test_metrics['few']:.4f}, "
            f"balanced={test_metrics['balanced_accuracy']:.4f}"
        )

    saved = {
        "type": "class_balanced_crt_reference",
        "feature_archive": str(archive_path),
        "seed": args.seed,
        "epoch": args.epochs,
        "state_dict": classifier.state_dict(),
        "reference_weight": classifier.fc.weight.detach().cpu(),
        "reference_bias": classifier.fc.bias.detach().cpu(),
        "source_weight": torch.from_numpy(source_weight.copy()),
        "source_bias": torch.from_numpy(source_bias.copy()),
        "source_train": source_train,
        "source_test": source_test,
        "history": history,
        "config": vars(args),
        "selection": "last_epoch_only",
    }
    torch.save(saved, output_path)
    summary_path = output_path.with_suffix(".json")
    summary = {
        "type": saved["type"],
        "feature_archive": str(archive_path),
        "checkpoint": str(output_path),
        "selection": saved["selection"],
        "source_train": compact(source_train),
        "source_test": compact(source_test),
        "last_train": compact(history[-1]["train"]),
        "last_test": compact(history[-1]["test"]),
        "history": [
            {
                "epoch": row["epoch"],
                "lr": row["lr"],
                "balanced_stream_loss": row["balanced_stream_loss"],
                "balanced_stream_accuracy": row["balanced_stream_accuracy"],
                "train": compact(row["train"]),
                "test": compact(row["test"]),
            }
            for row in history
        ],
        "config": vars(args),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {output_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
