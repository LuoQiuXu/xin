from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

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
    # Strict FP32 for this pilot: disable TF32 shortcuts as well as AMP.
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def epoch_seed_for(seed: int, epoch_index: int) -> int:
    """Stable, group-independent seed for each zero-based training epoch."""
    return int(seed) * 100_000 + int(epoch_index)


def state_dict_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    """Hash tensor names, dtypes, shapes and bytes in a deterministic order."""
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def classification_logits_v1(all_logits: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Return only v1 logits from a concatenated [v1,v2,v3] forward."""
    if all_logits.ndim != 2 or all_logits.shape[0] != 3 * int(batch_size):
        raise ValueError(
            "Expected exactly 3B logits from the matched three-view backbone, "
            f"got {tuple(all_logits.shape)} for B={batch_size}"
        )
    return all_logits[:batch_size]


def compose_group_loss(
    group: str,
    loss_lc: torch.Tensor,
    loss_bcl: torch.Tensor,
) -> torch.Tensor:
    """Keep the paired objective weights exact: LC-only=2.0*LC; BCL=2.0*LC+0.6*BCL."""
    if group == "LC_only":
        return LAMBDA_LC * loss_lc
    if group == "BCL":
        return LAMBDA_LC * loss_lc + MU_BCL * loss_bcl
    raise ValueError(f"Unknown training group: {group}")


def first_batch_views_sha256(views) -> str:
    digest = hashlib.sha256()
    for view in views:
        array = view.detach().cpu().contiguous().numpy()
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


class IndexedCIFAR100LTDataset(Dataset):
    """Expose original TRAIN indices for a paired per-epoch sample-order audit."""

    def __init__(self, dataset: CIFAR100LT):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        views, label = self.dataset[index]
        return views, label, int(self.dataset.indices[index])


def verify_paired_data_streams(output_root: Path) -> dict:
    """Require exact epoch seeds, sample orders and first-batch view fingerprints."""
    histories = {}
    for group in ("LC_only", "BCL"):
        path = output_root / f"{group}_seed{TRAIN_SEED}" / "history.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Missing training history for paired audit: {path}")
        with path.open("r", encoding="utf-8") as stream:
            histories[group] = [json.loads(line) for line in stream if line.strip()]

    if len(histories["LC_only"]) != EPOCHS or len(histories["BCL"]) != EPOCHS:
        raise RuntimeError(
            "Paired data-stream audit requires complete histories for both 200-epoch runs"
        )

    mismatches = []
    fields = ("epoch", "epoch_seed", "sample_order_sha256", "first_batch_views_sha256")
    for left, right in zip(histories["LC_only"], histories["BCL"]):
        for field in fields:
            if left.get(field) != right.get(field):
                mismatches.append({
                    "epoch": left.get("epoch"),
                    "field": field,
                    "LC_only": left.get(field),
                    "BCL": right.get(field),
                })
    report = {
        "seed": TRAIN_SEED,
        "epochs_compared": min(len(histories["LC_only"]), len(histories["BCL"])),
        "comparison_fields": list(fields),
        "paired_data_streams_identical": not mismatches,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "note": (
            "The full epoch sample order is hashed. The first augmented minibatch "
            "per epoch is byte-hashed as an augmentation sentinel; both groups use "
            "the same epoch seed, worker seeding and transform pipeline."
        ),
    }
    report_path = output_root / "paired_data_stream_audit.json"
    with report_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
    if mismatches:
        raise RuntimeError(
            f"LC-only/BCL data streams diverged; see {report_path}. "
            "Stop before TEST evaluation."
        )
    return report


class BatchNormAudit:
    """Capture first-minibatch per-view inputs and checkpoint running stats.

    Backbone BatchNorm2d sees concatenated [v1,v2,v3] in both paired groups.
    The BCL sample-projection BatchNorm1d is recorded separately because that
    projection head is only executed by the BCL objective.
    """

    def __init__(self, model: nn.Module):
        self.modules = {}
        self.handles = []
        self.active = False
        self.first_batch_inputs = {}
        for name, module in model.named_modules():
            is_backbone_bn = name.startswith("backbone.") and isinstance(module, nn.BatchNorm2d)
            is_sample_bn = name == "sample_mlp.net.1" and isinstance(module, nn.BatchNorm1d)
            if is_backbone_bn or is_sample_bn:
                self.modules[name] = module
                self.handles.append(module.register_forward_pre_hook(self._make_hook(name)))

    def _make_hook(self, name: str):
        def hook(module, inputs):
            if not self.active or name in self.first_batch_inputs:
                return
            x = inputs[0].detach().float()
            if x.ndim not in (2, 4) or x.shape[0] % 3 != 0:
                self.first_batch_inputs[name] = {"view_split_available": False}
                return
            batch_size = x.shape[0] // 3
            dims = (0,) if x.ndim == 2 else (0, 2, 3)
            mixed_mean = x.mean(dim=dims)
            mixed_var = x.var(dim=dims, unbiased=False)
            view_stats = {}
            for view_index, chunk in enumerate(x.split(batch_size, dim=0), start=1):
                view_mean = chunk.mean(dim=dims)
                view_var = chunk.var(dim=dims, unbiased=False)
                view_stats[f"v{view_index}"] = {
                    "input_mean_channel_average": float(view_mean.mean().cpu()),
                    "input_var_channel_average": float(view_var.mean().cpu()),
                    "mean_abs_channel_difference_from_concat": float(
                        (view_mean - mixed_mean).abs().mean().cpu()
                    ),
                    "var_abs_channel_difference_from_concat": float(
                        (view_var - mixed_var).abs().mean().cpu()
                    ),
                }
            self.first_batch_inputs[name] = {
                "view_split_available": True,
                "batch_size_per_view": int(batch_size),
                "concat_input_mean_channel_average": float(mixed_mean.mean().cpu()),
                "concat_input_var_channel_average": float(mixed_var.mean().cpu()),
                "views": view_stats,
            }
        return hook

    def arm(self):
        self.first_batch_inputs = {}
        self.active = True

    def finish_batch(self):
        self.active = False

    def snapshot(self, group: str, epoch: int):
        running_stats = {}
        for name, module in self.modules.items():
            mean = module.running_mean.detach().float().cpu()
            var = module.running_var.detach().float().cpu()
            running_stats[name] = {
                "num_batches_tracked": int(module.num_batches_tracked.detach().cpu().item()),
                "running_mean_abs_average": float(mean.abs().mean()),
                "running_mean_channel_std": float(mean.std(unbiased=False)),
                "running_var_average": float(var.mean()),
                "running_var_channel_std": float(var.std(unbiased=False)),
            }
        return {
            "group": group,
            "epoch_completed": int(epoch),
            "view_order": ["v1_classification", "v2_contrastive", "v3_contrastive"],
            "backbone_bn_input": "concatenated 3B batch; train-mode BatchNorm2d",
            "sample_mlp_bn1d": "BCL-only; applied to all 3B backbone features",
            "first_minibatch_per_view_input_stats": self.first_batch_inputs,
            "running_stats_after_epoch": running_stats,
        }

    def close(self):
        self.active = False
        for handle in self.handles:
            handle.remove()
        self.handles = []


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
    if not indices_file.is_file():
        raise FileNotFoundError(
            f"Required preflight-generated IF=100 indices are missing: {indices_file}. "
            "Training must not silently regenerate a different split."
        )
    dataset = CIFAR100LT(
        root=data_root,
        mode=mode,
        indices_file=indices_file,
        split_seed=0,
        download=False,
    )
    indexed_dataset = IndexedCIFAR100LTDataset(dataset)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        indexed_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        generator=generator,
        worker_init_fn=seed_worker,
    )
    return dataset, loader, generator


def _save_checkpoint(
    path: Path,
    model: BCLCIFARModel,
    optimizer: torch.optim.Optimizer,
    epoch_completed: int,
    group: str,
    seed: int,
    class_counts: np.ndarray,
    last_lr: float,
    data_loader_generator: torch.Generator,
    initial_model_state_sha256: str,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch_completed,
            "group": group,
            "seed": seed,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "data_loader_generator_state": data_loader_generator.get_state(),
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            "class_counts": class_counts.tolist(),
            "initial_model_state_sha256": initial_model_state_sha256,
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
    initial_state_path: Path,
    initial_state_sha256: str,
    num_workers: int,
    device: torch.device,
):
    if group not in {"LC_only", "BCL"}:
        raise ValueError(group)
    set_seed(TRAIN_SEED)

    mode = "lc" if group == "LC_only" else "bcl"
    dataset, loader, loader_generator = make_train_loader(
        data_root=data_root,
        mode=mode,
        indices_file=indices_file,
        num_workers=num_workers,
        seed=TRAIN_SEED,
    )
    class_counts = cifar100lt_class_counts()

    # Each group independently reloads the exact saved init_seed42.pth file.
    initial_state = load_initial_model_state(
        initial_state_path,
        expected_seed=TRAIN_SEED,
        expected_sha256=initial_state_sha256,
    )
    model = BCLCIFARModel()
    model.load_state_dict(initial_state, strict=True)
    if state_dict_sha256(model.state_dict()) != initial_state_sha256:
        raise RuntimeError("Loaded initial model state does not match init_seed42.pth hash")
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
        raise FileExistsError(f"Refusing to overwrite existing history: {history_path}")

    epoch_summaries = []
    bn_audit = BatchNormAudit(model)
    bn_audit_path = run_dir / "batchnorm_audit.jsonl"
    if bn_audit_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing BatchNorm audit: {bn_audit_path}")
    total_start = time.perf_counter()

    for epoch_index in range(EPOCHS):
        human_epoch = epoch_index + 1
        epoch_seed = epoch_seed_for(TRAIN_SEED, epoch_index)
        # Reset augmentation / sampler RNGs identically for this epoch in both groups.
        set_seed(epoch_seed)
        loader_generator.manual_seed(epoch_seed)
        if human_epoch in CHECKPOINT_EPOCHS:
            bn_audit.arm()
        lr = learning_rate_for_epoch(epoch_index)
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = lr

        model.train()
        loss_lc_sum = 0.0
        loss_bcl_sum = 0.0
        loss_total_sum = 0.0
        correct = 0
        seen = 0
        sample_order_digest = hashlib.sha256()
        first_views_digest = None

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        epoch_start = time.perf_counter()

        for batch_index, (images, labels, source_indices) in enumerate(loader):
            sample_order_digest.update(
                source_indices.detach().cpu().numpy().astype(np.int64, copy=False).tobytes()
            )
            if batch_index == 0:
                first_views_digest = first_batch_views_sha256(images)
            labels = labels.to(device, non_blocking=True)
            batch_size = int(labels.shape[0])

            view1, view2, view3 = [
                view.to(device, non_blocking=True) for view in images
            ]
            # Both groups concatenate the same three views through the shared
            # backbone. LC-only uses only v1 logits; BCL additionally consumes
            # v2/v3 projections and classifier-derived prototypes.
            all_views = torch.cat([view1, view2, view3], dim=0)

            if group == "LC_only":
                _, all_logits, _, _ = model(
                    all_views,
                    compute_contrastive=False,
                )
                logits = classification_logits_v1(all_logits, batch_size)
                loss_lc = criterion_lc(logits, labels)
                loss_bcl = logits.new_zeros(())
                loss = compose_group_loss(group, loss_lc, loss_bcl)
            else:
                _, all_logits, all_embeddings, class_prototypes = model(
                    all_views,
                    compute_contrastive=True,
                )
                logits = classification_logits_v1(all_logits, batch_size)
                z2 = all_embeddings[batch_size : 2 * batch_size]
                z3 = all_embeddings[2 * batch_size : 3 * batch_size]
                two_view_features = torch.stack([z2, z3], dim=1)

                loss_lc = criterion_lc(logits, labels)
                loss_bcl = criterion_bcl(
                    class_prototypes,
                    two_view_features,
                    labels,
                )
                loss = compose_group_loss(group, loss_lc, loss_bcl)

            bn_audit.finish_batch()
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

        if first_views_digest is None:
            raise RuntimeError("DataLoader produced no batches in this epoch")
        summary = {
            "epoch": epoch_index + 1,
            "epoch_seed": epoch_seed,
            "sample_order_sha256": sample_order_digest.hexdigest(),
            "first_batch_views_sha256": first_views_digest,
            "lr": lr,
            "loss_lc": loss_lc_sum / max(seen, 1),
            "loss_bcl": loss_bcl_sum / max(seen, 1),
            "loss_total": loss_total_sum / max(seen, 1),
            "train_top1_raw_logits": 100.0 * correct / max(seen, 1),
            "epoch_seconds": epoch_seconds,
            "peak_cuda_memory_bytes": peak_memory_bytes,
            "num_train_samples": seen,
        }
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
                data_loader_generator=loader_generator,
                initial_model_state_sha256=initial_state_sha256,
            )
            summary["checkpoint"] = str(checkpoint_path)
            bn_record = bn_audit.snapshot(group, human_epoch)
            with bn_audit_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(bn_record, ensure_ascii=False) + "\n")
            summary["batchnorm_audit_epoch"] = human_epoch

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

    bn_audit.close()
    total_seconds = time.perf_counter() - total_start
    result = {
        "group": group,
        "seed": TRAIN_SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "precision": "fp32",
        "initial_model_state_sha256": initial_state_sha256,
        "total_training_seconds": total_seconds,
        "mean_epoch_seconds": float(np.mean([x["epoch_seconds"] for x in epoch_summaries])),
        "max_epoch_peak_cuda_memory_bytes": max(
            [x["peak_cuda_memory_bytes"] for x in epoch_summaries], default=0
        ),
        "final_epoch": epoch_summaries[-1],
        "checkpoint_dir": str(checkpoint_dir),
        "batchnorm_audit_file": str(bn_audit_path),
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


def load_initial_model_state(
    path: Path,
    expected_seed: int,
    expected_sha256: str,
) -> dict[str, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing paired initial checkpoint: {path}")
    checkpoint = _safe_torch_load(path, torch.device("cpu"))
    if int(checkpoint.get("seed", -1)) != int(expected_seed):
        raise RuntimeError("init_seed42.pth seed does not match the paired experiment seed")
    state = checkpoint.get("model_state")
    if not isinstance(state, dict):
        raise RuntimeError("init_seed42.pth does not contain a model_state dictionary")
    actual_sha256 = state_dict_sha256(state)
    recorded_sha256 = checkpoint.get("model_state_sha256")
    if recorded_sha256 != actual_sha256 or actual_sha256 != expected_sha256:
        raise RuntimeError("init_seed42.pth model-state SHA-256 verification failed")
    return state


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
    all_logits = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        _, logits, _, _ = model(images, compute_contrastive=False)
        predictions = logits.argmax(dim=1)
        all_labels.append(labels.cpu().numpy())
        all_predictions.append(predictions.cpu().numpy())
        all_logits.append(logits.detach().cpu().numpy())

    labels_np = np.concatenate(all_labels).astype(np.int64)
    predictions_np = np.concatenate(all_predictions).astype(np.int64)
    logits_np = np.concatenate(all_logits).astype(np.float32)
    class_counts = cifar100lt_class_counts()
    log_prior_np = np.log(class_counts.astype(np.float32) / float(class_counts.sum()))
    adjusted_logits_np = logits_np + log_prior_np[None, :]

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
        logits=logits_np,
        adjusted_logits=adjusted_logits_np,
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
    if not indices_file.is_file():
        raise FileNotFoundError(
            f"Required preflight-generated IF=100 indices are missing: {indices_file}. "
            "Do not regenerate or download the split in the training entry point."
        )
    output_root.mkdir(parents=True, exist_ok=True)

    existing_run_artifacts = [
        output_root / "environment.json",
        output_root / f"initial_model_seed{TRAIN_SEED}.pth",
        output_root / f"LC_only_seed{TRAIN_SEED}",
        output_root / f"BCL_seed{TRAIN_SEED}",
    ]
    existing_run_artifacts = [path for path in existing_run_artifacts if path.exists()]
    if existing_run_artifacts:
        raise RuntimeError(
            "Refusing to overwrite an existing paired pilot. Choose a fresh "
            "--output-root or move the previous run outputs first: "
            + ", ".join(str(path) for path in existing_run_artifacts)
        )

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")

    import torchvision

    environment = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device": str(device),
        "num_workers": args.num_workers,
        "seed": TRAIN_SEED,
        "precision": "fp32",
    }
    if device.type == "cuda":
        free_memory, total_memory = torch.cuda.mem_get_info(device)
        environment.update({
            "gpu_name": torch.cuda.get_device_name(device),
            "gpu_total_memory_bytes": int(total_memory),
            "gpu_free_memory_before_training_bytes": int(free_memory),
        })
    with (output_root / "environment.json").open("w", encoding="utf-8") as f:
        json.dump(environment, f, indent=2, ensure_ascii=False)

    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    print(f"torchvision: {torchvision.__version__}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(f"CUDA runtime: {torch.version.cuda}")
        print(f"GPU free/total: {free_memory}/{total_memory} bytes")

    # Create a single initial state, then reload it for both independent runs.
    set_seed(TRAIN_SEED)
    template_model = BCLCIFARModel()
    initial_state = {
        key: value.detach().cpu().clone()
        for key, value in template_model.state_dict().items()
    }
    del template_model
    initial_state_path = output_root / f"initial_model_seed{TRAIN_SEED}.pth"
    initial_state_sha256 = state_dict_sha256(initial_state)
    torch.save(
        {
            "model_state": initial_state,
            "seed": TRAIN_SEED,
            "model_state_sha256": initial_state_sha256,
        },
        initial_state_path,
    )

    # Run the two groups sequentially. No test dataset is instantiated here.
    lc_checkpoint, lc_summary = train_group(
        group="LC_only",
        data_root=data_root,
        output_root=output_root,
        indices_file=indices_file,
        initial_state_path=initial_state_path,
        initial_state_sha256=initial_state_sha256,
        num_workers=args.num_workers,
        device=device,
    )
    bcl_checkpoint, bcl_summary = train_group(
        group="BCL",
        data_root=data_root,
        output_root=output_root,
        indices_file=indices_file,
        initial_state_path=initial_state_path,
        initial_state_sha256=initial_state_sha256,
        num_workers=args.num_workers,
        device=device,
    )

    # Hard stop before TEST: paired sampler order and augmented-view sentinels
    # must match exactly for all 200 epochs.
    paired_data_stream_audit = verify_paired_data_streams(output_root)

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
        "initial_model_state_sha256": initial_state_sha256,
        "indices_file": str(indices_file),
        "paired_data_stream_audit": paired_data_stream_audit,
        "LC_only_training": lc_summary,
        "BCL_training": bcl_summary,
        "LC_only_final_test": lc_metrics,
        "BCL_final_test": bcl_metrics,
        "matched_view_protocol": (
            "Both LC-only and BCL concatenate v1+v2+v3 through the shared backbone; "
            "both therefore expose backbone BatchNorm2d to the same three-view input "
            "distribution. LC-only applies LC to v1 logits only. BCL additionally runs "
            "the sample projection MLP on all 3B features and applies BCL to v2/v3."
        ),
        "known_bn_caveat": (
            "BCL sample-projection BatchNorm1d also mixes all three views; this MLP "
            "is BCL-only. Train-mode BatchNorm can couple gradients across view rows. "
            "The training-time BatchNorm audit records per-view inputs and running "
            "statistics at epochs 40/100/160/180/200."
        ),
    }
    with (output_root / "paired_pilot_summary.json").open("w", encoding="utf-8") as f:
        json.dump(pair_summary, f, indent=2, ensure_ascii=False)
    print(f"Paired pilot summary: {output_root / 'paired_pilot_summary.json'}")


if __name__ == "__main__":
    main()
