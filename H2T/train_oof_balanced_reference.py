"""Build recipient-OOF balanced reference heads and calibrated reliability.

The backbone features are frozen.  Each sample is assigned to one of five
class-stratified folds.  Head k is class-balanced cRT trained without fold k.
For samples in fold k, temperature T_k is fitted from the OOF logits/labels of
the other four folds only.  The saved reliability is the continuous true-class
probability p_T(y|f), never a correctness hard gate.
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
    parser = argparse.ArgumentParser(description="Recipient-OOF balanced references")
    parser.add_argument("--features", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--folds", type=int, default=5)
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


def stratified_folds(labels: np.ndarray, fold_count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    result = np.empty(labels.size, dtype=np.int64)
    for class_id in np.unique(labels):
        indices = np.flatnonzero(labels == class_id).copy()
        rng.shuffle(indices)
        result[indices] = np.arange(indices.size, dtype=np.int64) % fold_count
    return result


def balanced_schedule(
    labels: np.ndarray,
    allowed: np.ndarray,
    epochs: int,
    samples_per_epoch: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    class_indices = {
        int(class_id): allowed[labels[allowed] == class_id] for class_id in classes
    }
    if any(indices.size == 0 for indices in class_indices.values()):
        raise ValueError("every class must retain at least one OOF training sample")
    schedule = np.empty((epochs, samples_per_epoch), dtype=np.int64)
    for epoch in range(epochs):
        chosen_classes = rng.integers(0, classes.size, size=samples_per_epoch)
        for class_id in classes:
            mask = chosen_classes == class_id
            if mask.any():
                schedule[epoch, mask] = rng.choice(
                    class_indices[int(class_id)], int(mask.sum()), replace=True
                )
    return schedule


@torch.no_grad()
def logits_in_batches(
    classifier: resnet_cifar.Classifier,
    features: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    classifier.eval()
    return torch.cat(
        [classifier.fc(features[start : start + batch_size])
         for start in range(0, features.shape[0], batch_size)],
        dim=0,
    )


def train_head(
    features: torch.Tensor,
    labels: torch.Tensor,
    labels_np: np.ndarray,
    allowed: np.ndarray,
    args: argparse.Namespace,
    fold: int,
) -> tuple[resnet_cifar.Classifier, list[dict[str, float]]]:
    seed_everything(args.seed + 1009 * (fold + 1))
    classifier = resnet_cifar.Classifier(feat_in=64, num_classes=100).to(features.device)
    optimizer = torch.optim.SGD(
        classifier.parameters(),
        lr=args.lr * args.lr_factor,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    samples_per_epoch = labels.numel()
    schedule = balanced_schedule(
        labels_np,
        allowed,
        args.epochs,
        samples_per_epoch,
        args.seed + 50021 + fold,
    )
    history: list[dict[str, float]] = []
    for epoch in range(args.epochs):
        classifier.train()
        lr = args.lr_factor * args.lr * 0.5 * (
            1.0 + math.cos(epoch / args.epochs * math.pi)
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        loss_sum = 0.0
        correct = 0
        epoch_indices = schedule[epoch]
        for start in range(0, epoch_indices.size, args.batch_size):
            index = torch.from_numpy(epoch_indices[start : start + args.batch_size]).to(
                features.device
            )
            target = labels[index]
            logits = classifier.fc(features[index])
            loss = F.cross_entropy(logits, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * target.numel()
            correct += int((logits.argmax(1) == target).sum())
        history.append(
            {
                "epoch": epoch + 1,
                "lr": lr,
                "balanced_loss": loss_sum / samples_per_epoch,
                "balanced_accuracy": correct / samples_per_epoch,
            }
        )
        print(
            f"fold {fold} epoch {epoch + 1:02d}/{args.epochs}: "
            f"balanced_acc={history[-1]['balanced_accuracy']:.4f}"
        )
    return classifier, history


def class_balanced_nll(
    logits: torch.Tensor,
    target: torch.Tensor,
    log_temperature: torch.Tensor,
) -> torch.Tensor:
    counts = torch.bincount(target, minlength=logits.shape[1]).float().clamp_min(1.0)
    weights = counts.sum() / (counts * counts.numel())
    per_sample = F.cross_entropy(
        logits / log_temperature.exp().clamp(0.05, 20.0), target, reduction="none"
    )
    return (per_sample * weights[target]).sum() / weights[target].sum()


def fit_temperature(logits: torch.Tensor, target: torch.Tensor) -> float:
    log_temperature = torch.zeros((), device=logits.device, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_temperature], lr=0.1, max_iter=100, line_search_fn="strong_wolfe"
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        loss = class_balanced_nll(logits, target, log_temperature)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 20.0))


def group_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    correct = prediction.eq(target)
    class_correct = torch.zeros(100, device=target.device)
    class_count = torch.zeros(100, device=target.device)
    class_correct.scatter_add_(0, target, correct.float())
    class_count.scatter_add_(0, target, torch.ones_like(target, dtype=torch.float32))
    accuracy = class_correct / class_count.clamp_min(1.0)
    return {
        "overall": float(correct.float().mean()),
        "many": float(accuracy[:36].mean()),
        "medium": float(accuracy[36:71].mean()),
        "few": float(accuracy[71:].mean()),
        "balanced": float(accuracy.mean()),
    }


def reliability_summary(q: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    groups = torch.where(labels < 36, 0, torch.where(labels < 71, 1, 2))
    result: dict[str, Any] = {}
    for name, group in (("all", None), ("many", 0), ("medium", 1), ("few", 2)):
        values = q if group is None else q[groups == group]
        quantiles = torch.quantile(values, torch.tensor([0.1, 0.5, 0.9], device=q.device))
        result[name] = {
            "mean": float(values.mean()),
            "p10": float(quantiles[0]),
            "p50": float(quantiles[1]),
            "p90": float(quantiles[2]),
        }
    return result


def main() -> None:
    args = parse_args()
    if args.folds < 2:
        raise ValueError("--folds must be at least 2")
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    source = Path(args.features).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    archive = np.load(source, allow_pickle=False)
    features_np = archive["train_features"].astype(np.float32, copy=False)
    labels_np = archive["train_labels"].astype(np.int64, copy=False)
    features = torch.from_numpy(features_np).to(device)
    labels = torch.from_numpy(labels_np).to(device)
    folds_np = stratified_folds(labels_np, args.folds, args.seed + 17011)
    folds = torch.from_numpy(folds_np).to(device)

    weights: list[torch.Tensor] = []
    biases: list[torch.Tensor] = []
    histories: list[list[dict[str, float]]] = []
    oof_logits = torch.empty(labels.numel(), 100, device=device)
    for fold in range(args.folds):
        allowed = np.flatnonzero(folds_np != fold)
        classifier, history = train_head(
            features, labels, labels_np, allowed, args, fold
        )
        logits = logits_in_batches(classifier, features, args.batch_size)
        held_out = folds == fold
        oof_logits[held_out] = logits[held_out]
        weights.append(classifier.fc.weight.detach().cpu().clone())
        biases.append(classifier.fc.bias.detach().cpu().clone())
        histories.append(history)

    temperatures: list[float] = []
    q = torch.empty(labels.numel(), device=device)
    calibrated_oof_logits = torch.empty_like(oof_logits)
    for fold in range(args.folds):
        calibration = folds != fold
        temperature = fit_temperature(oof_logits[calibration], labels[calibration])
        temperatures.append(temperature)
        held_out = folds == fold
        calibrated_oof_logits[held_out] = oof_logits[held_out] / temperature
        q[held_out] = F.softmax(calibrated_oof_logits[held_out], dim=1).gather(
            1, labels[held_out, None]
        ).squeeze(1)
        print(f"fold {fold}: cross-fitted temperature={temperature:.4f}")

    metrics = group_metrics(calibrated_oof_logits.argmax(1), labels)
    reliability = reliability_summary(q, labels)
    saved: dict[str, Any] = {
        "type": "recipient_oof_balanced_reference",
        "feature_archive": str(source),
        "seed": args.seed,
        "fold_count": args.folds,
        "fold_ids": torch.from_numpy(folds_np),
        "reference_weight": torch.stack(weights),
        "reference_bias": torch.stack(biases),
        "temperature": torch.tensor(temperatures, dtype=torch.float32),
        "q_true_class": q.detach().cpu(),
        "oof_logits": oof_logits.detach().cpu(),
        "calibrated_oof_logits": calibrated_oof_logits.detach().cpu(),
        "train_labels": torch.from_numpy(labels_np.copy()),
        "train_selected_indices": torch.from_numpy(
            archive["train_selected_indices"].astype(np.int64, copy=True)
        ),
        "metrics": metrics,
        "reliability": reliability,
        "history": histories,
        "config": vars(args),
    }
    torch.save(saved, output)
    summary = {
        key: value
        for key, value in saved.items()
        if key not in {
            "fold_ids", "reference_weight", "reference_bias", "temperature",
            "q_true_class", "oof_logits", "calibrated_oof_logits", "train_labels",
            "train_selected_indices", "history",
        }
    }
    summary["temperature"] = temperatures
    summary["fold_sizes"] = [int((folds == fold).sum()) for fold in range(args.folds)]
    summary["metrics"] = metrics
    summary["reliability"] = reliability
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("OOF metrics", metrics)
    print("reliability", reliability)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
