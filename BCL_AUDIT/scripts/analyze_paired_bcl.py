from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from BCL_AUDIT.cifar100lt.data import CIFAR100TrainOnly, build_cifar_transforms, cifar100lt_class_counts
from BCL_AUDIT.cifar100lt.losses import BalancedContrastiveLoss, LogitCompensatedCE
from BCL_AUDIT.cifar100lt.models import BCLCIFARModel
from BCL_AUDIT.cifar100lt.train import LAMBDA_LC, MU_BCL, state_dict_sha256
from BCL_AUDIT.scripts.phase2_decision_audit import (
    load_and_validate_dino_topology,
    run_phase2_decision_audit,
)


CHECKPOINTS = [
    (40, "epoch_040.pth"),
    (100, "epoch_100.pth"),
    (160, "epoch_160_pre_decay.pth"),
    (180, "epoch_180_pre_decay.pth"),
    (200, "epoch_200.pth"),
]


def safe_torch_load(path: Path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def set_probe_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_static_data_audit(data_audit_dir: Path, indices_file: Path) -> dict[str, Any]:
    """Cross-check the analyzer's static audit directory against the exact indices file."""
    required = (
        "class_counts.csv",
        "superclass_counts.csv",
        "head_tail_pair_topology.csv",
        "data_audit_summary.json",
        "cifar100lt_if100_train_indices_seed0.npy",
    )
    missing = [name for name in required if not (data_audit_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Static TRAIN-only data-audit artifacts are missing: " + ", ".join(missing)
        )
    if not indices_file.is_file():
        raise FileNotFoundError(f"Missing fixed IF=100 indices: {indices_file}")

    static_indices_path = data_audit_dir / "cifar100lt_if100_train_indices_seed0.npy"
    indices_hash = file_sha256(indices_file)
    if file_sha256(static_indices_path) != indices_hash:
        raise RuntimeError(
            "The analyzer indices_file is not byte-identical to the static audit's saved indices"
        )

    with (data_audit_dir / "data_audit_summary.json").open(encoding="utf-8") as stream:
        summary = json.load(stream)
    expected_counts = cifar100lt_class_counts()
    if (
        summary.get("test_split_loaded") is not False
        or "training split" not in str(summary.get("source_split", ""))
        or summary.get("split_seed") != 0
        or abs(float(summary.get("imbalance_factor", -1.0)) - 0.01) > 1e-12
        or not np.array_equal(
            np.asarray(summary.get("class_counts", []), dtype=np.int64), expected_counts
        )
        or int(summary.get("long_tail_train_size", -1)) != 10_847
    ):
        raise RuntimeError("Static data-audit summary does not certify the fixed TRAIN-only IF=100 split")

    class_rows = read_csv(data_audit_dir / "class_counts.csv")
    if len(class_rows) != 100 or {int(row["fine_id"]) for row in class_rows} != set(range(100)):
        raise RuntimeError("class_counts.csv must contain each official fine class exactly once")
    observed_counts = np.asarray(
        [int(next(row["train_count"] for row in class_rows if int(row["fine_id"]) == class_id))
         for class_id in range(100)],
        dtype=np.int64,
    )
    if not np.array_equal(observed_counts, expected_counts):
        raise RuntimeError("class_counts.csv does not match the fixed CIFAR-100-LT IF=100 vector")

    return {
        "indices_file_sha256": indices_hash,
        "num_lt_train_samples": int(expected_counts.sum()),
        "test_split_loaded": False,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _pair_assignments(
    rows: list[dict[str, str]], path: Path
) -> dict[tuple[int, int], tuple[int, int, bool]]:
    assignments = {}
    for row_number, row in enumerate(rows, start=2):
        try:
            identity = (int(row["head_fine_id"]), int(row["tail_fine_id"]))
            relation = str(row["same_superclass"]).strip().lower()
            if relation not in {"true", "false", "1", "0", "yes", "no", "y", "n"}:
                raise ValueError("Missing or invalid same_superclass flag")
            assignment = (
                int(row["head_coarse_id"]),
                int(row["tail_coarse_id"]),
                parse_bool(relation),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Invalid pair identity or superclass assignment in {path.name}, row {row_number}"
            ) from exc
        if identity in assignments:
            raise RuntimeError(f"Duplicate pair identity {identity} in {path.name}")
        assignments[identity] = assignment
    return assignments


def load_pair_topology(data_audit_dir: Path) -> list[dict[str, str]]:
    static_path = data_audit_dir / "head_tail_pair_topology.csv"
    static_rows = read_csv(static_path)
    if not static_rows:
        raise RuntimeError("head_tail_pair_topology.csv is empty; run the static topology audit first")

    dino_path = data_audit_dir / "head_tail_pair_topology_dino.csv"
    if not (dino_path.exists() or dino_path.is_symlink()):
        return static_rows

    dino_rows = read_csv(dino_path)
    if len(dino_rows) != len(static_rows):
        raise RuntimeError("DINO sidecar row count differs from the static pair table")
    static = _pair_assignments(static_rows, static_path)
    dino = _pair_assignments(dino_rows, dino_path)
    if dino.keys() != static.keys():
        raise RuntimeError("DINO sidecar pair identities differ from the static pair table")
    if dino != static:
        raise RuntimeError("DINO sidecar superclass assignments differ from the static pair table")
    return dino_rows


def load_or_create_probe_cache(
    data_root: Path,
    indices_file: Path,
    cache_path: Path,
    per_class: int,
    seed: int,
):
    """Create/reuse a fixed, globally shuffled multi-view probe from LT TRAIN only."""
    if not indices_file.is_file():
        raise FileNotFoundError(f"Missing fixed CIFAR-100-LT indices: {indices_file}")

    selected_lt_indices = np.load(indices_file).astype(np.int64)
    if selected_lt_indices.ndim != 1:
        raise RuntimeError("The LT index file must be one-dimensional")
    if np.unique(selected_lt_indices).size != selected_lt_indices.size:
        raise RuntimeError("The LT index file contains duplicate source indices")

    train_set = CIFAR100TrainOnly(
        root=str(data_root),
        train=True,
        download=False,
        transform=None,
    )
    source_labels = np.asarray(train_set.targets, dtype=np.int64)
    if np.any(selected_lt_indices < 0) or np.any(selected_lt_indices >= len(source_labels)):
        raise RuntimeError("The LT index file contains out-of-range source indices")
    expected_lt_counts = cifar100lt_class_counts()
    observed_lt_counts = np.bincount(source_labels[selected_lt_indices], minlength=100)
    if not np.array_equal(observed_lt_counts, expected_lt_counts):
        raise RuntimeError("The fixed indices file does not match CIFAR-100-LT IF=100")
    indices_file_sha256 = file_sha256(indices_file)
    expected_probe_order = "per-class fixed-count selection followed by seeded global shuffle"
    expected_count = 100 * per_class

    if cache_path.is_file():
        cache = safe_torch_load(cache_path, map_location="cpu")
        required_keys = {"view1", "view2", "view3", "view_eval", "labels", "source_indices", "metadata"}
        if not required_keys.issubset(cache):
            raise RuntimeError("Probe cache is incomplete; use a fresh Phase-2 output root")
        labels = cache["labels"].detach().cpu().numpy().astype(np.int64, copy=False)
        source_indices = cache["source_indices"].detach().cpu().numpy().astype(np.int64, copy=False)
        metadata = cache.get("metadata", {})
        if labels.shape != (expected_count,) or source_indices.shape != (expected_count,):
            raise RuntimeError("Probe cache has an unexpected sample count; use a fresh Phase-2 output root")
        if np.unique(source_indices).size != expected_count:
            raise RuntimeError("Probe cache contains duplicate source indices")
        if not np.isin(source_indices, selected_lt_indices).all():
            raise RuntimeError("Probe cache contains images outside the fixed LT TRAIN subset")
        if not np.array_equal(labels, source_labels[source_indices]):
            raise RuntimeError("Probe cache labels do not match the original TRAIN source indices")
        if not np.array_equal(np.bincount(labels, minlength=100), np.full(100, per_class)):
            raise RuntimeError("Probe cache does not contain the requested fixed count per class")
        expected_metadata = {
            "test_split_loaded": False,
            "per_class": per_class,
            "probe_seed": seed,
            "probe_order": expected_probe_order,
            "indices_file_sha256": indices_file_sha256,
        }
        for key, expected_value in expected_metadata.items():
            if metadata.get(key) != expected_value:
                raise RuntimeError(
                    f"Probe cache metadata mismatch for {key}; use a fresh Phase-2 output root"
                )
        if "training split" not in str(metadata.get("source_split", "")):
            raise RuntimeError("Probe cache does not certify the original TRAIN split")
        for key in ("view1", "view2", "view3", "view_eval"):
            tensor = cache[key]
            if not torch.is_tensor(tensor) or tensor.shape != (expected_count, 3, 32, 32):
                raise RuntimeError(f"Probe cache tensor {key} has an invalid shape")
        return cache

    rng = np.random.RandomState(seed)
    probe_indices = []
    for class_id in range(100):
        pool = selected_lt_indices[source_labels[selected_lt_indices] == class_id]
        if pool.size < per_class:
            raise RuntimeError(
                f"Class {class_id} has only {pool.size} LT training images; "
                f"cannot sample {per_class} probes"
            )
        probe_indices.extend(
            rng.choice(pool, size=per_class, replace=False).astype(np.int64).tolist()
        )
    probe_indices = np.asarray(probe_indices, dtype=np.int64)
    # Keep the class quota fixed but mix class identities across diagnostic batches.
    probe_indices = probe_indices[rng.permutation(probe_indices.size)]

    transform_v1, transform_sim, transform_eval = build_cifar_transforms()
    set_probe_seed(seed)
    v1_list, v2_list, v3_list, eval_list = [], [], [], []
    labels = source_labels[probe_indices].astype(np.int64)
    for source_index in probe_indices.tolist():
        image = Image.fromarray(train_set.data[source_index])
        v1_list.append(transform_v1(image))
        v2_list.append(transform_sim(image))
        v3_list.append(transform_sim(image))
        eval_list.append(transform_eval(image))

    cache = {
        "view1": torch.stack(v1_list).cpu(),
        "view2": torch.stack(v2_list).cpu(),
        "view3": torch.stack(v3_list).cpu(),
        "view_eval": torch.stack(eval_list).cpu(),
        "labels": torch.from_numpy(labels).long(),
        "source_indices": torch.from_numpy(probe_indices).long(),
        "metadata": {
            "source_split": "original CIFAR-100 training split, restricted to fixed IF=100 LT indices",
            "test_split_loaded": False,
            "per_class": per_class,
            "probe_seed": seed,
            "probe_order": expected_probe_order,
            "indices_file_sha256": indices_file_sha256,
            "class_order": "official fine-label IDs 0..99",
        },
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_path)
    return cache


@torch.no_grad()
def extract_backbone_features(
    model: BCLCIFARModel,
    images: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    chunks = []
    for start in range(0, images.shape[0], batch_size):
        batch = images[start : start + batch_size].to(device, non_blocking=True)
        chunks.append(model.backbone(batch).detach())
    return torch.cat(chunks, dim=0)


def margins_from_logits(logits: torch.Tensor, labels: torch.Tensor):
    labels = labels.to(device=logits.device, dtype=torch.long)
    true_logits = logits.gather(1, labels[:, None]).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, labels[:, None], float("-inf"))
    return true_logits - masked.max(dim=1).values


def load_model_checkpoint(
    path: Path,
    device: torch.device,
    expected_group: str,
    expected_epoch: int,
    expected_initial_sha256: str,
):
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    checkpoint = safe_torch_load(path, map_location="cpu")
    expected = {
        "group": expected_group,
        "seed": 42,
        "epoch": expected_epoch,
        "initial_model_state_sha256": expected_initial_sha256,
    }
    for key, expected_value in expected.items():
        if checkpoint.get(key) != expected_value:
            raise RuntimeError(
                f"Checkpoint provenance mismatch for {path.name}: "
                f"{key}={checkpoint.get(key)!r}, expected {expected_value!r}"
            )
    model = BCLCIFARModel()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device)
    model.eval()
    return model


def compute_probe_margin_rows(
    model: BCLCIFARModel,
    group: str,
    epoch: int,
    probe: dict,
    class_meta: dict[int, dict[str, str]],
    criterion_lc: LogitCompensatedCE,
    device: torch.device,
    batch_size: int,
):
    features = extract_backbone_features(
        model, probe["view_eval"], device=device, batch_size=batch_size
    )
    labels = probe["labels"].to(device)
    logits = model.classifier(features)
    adjusted_logits = criterion_lc.adjusted_logits(logits)
    raw_margin = margins_from_logits(logits, labels)
    adjusted_margin = margins_from_logits(adjusted_logits, labels)
    predictions = logits.argmax(dim=1)

    rows = []
    for class_id in range(100):
        mask = labels == class_id
        meta = class_meta[class_id]
        rows.append(
            {
                "group": group,
                "source_split": "LT_train_probe",
                "view_type": "deterministic_eval_view",
                "epoch": epoch,
                "class_id": class_id,
                "coarse_id": meta["coarse_id"],
                "shot_group": meta["shot_group"],
                "probe_count": int(mask.sum().item()),
                "raw_top1": 100.0 * float((predictions[mask] == labels[mask]).float().mean().cpu()),
                "raw_margin_mean": float(raw_margin[mask].mean().cpu()),
                "raw_margin_median": float(raw_margin[mask].median().cpu()),
                "adjusted_margin_mean": float(adjusted_margin[mask].mean().cpu()),
                "adjusted_margin_median": float(adjusted_margin[mask].median().cpu()),
            }
        )
    return rows


def safe_rowwise_cosine(a: torch.Tensor, b: torch.Tensor):
    numerator = (a * b).sum(dim=-1)
    denominator = a.norm(dim=-1) * b.norm(dim=-1)
    return numerator / denominator.clamp_min(1e-12)


def compute_bcl_gradient_rows(
    model: BCLCIFARModel,
    epoch: int,
    probe: dict,
    class_meta: dict[int, dict[str, str]],
    criterion_lc: LogitCompensatedCE,
    criterion_bcl: BalancedContrastiveLoss,
    device: torch.device,
    batch_size: int,
):
    """Compare LC and BCL gradients at W_y and the same v2 probe feature f(x).

    The model is in eval mode: BN running statistics are fixed, so this measures
    the local loss-gradient interaction without train-mode cross-example BN
    gradient coupling. The latter is audited separately during training.
    """
    labels = probe["labels"].to(device)
    features_v1 = extract_backbone_features(
        model, probe["view1"], device=device, batch_size=batch_size
    )
    features_v2 = extract_backbone_features(
        model, probe["view2"], device=device, batch_size=batch_size
    ).detach().requires_grad_(True)
    features_v3 = extract_backbone_features(
        model, probe["view3"], device=device, batch_size=batch_size
    )

    count = labels.shape[0]
    all_features = torch.cat(
        [features_v1.detach(), features_v2, features_v3.detach()], dim=0
    )
    all_embeddings = model.sample_mlp(all_features)
    z2 = all_embeddings[count : 2 * count]
    z3 = all_embeddings[2 * count : 3 * count]
    two_view_embeddings = torch.stack([z2, z3], dim=1)
    prototypes = model.prototype_mlp(model.classifier.weight)

    # LC and BCL both differentiate through exactly the same f2 tensor.
    logits_same_feature = model.classifier(features_v2)
    loss_lc = criterion_lc(logits_same_feature, labels)
    loss_bcl = criterion_bcl(prototypes, two_view_embeddings, labels)

    grad_lc_w, grad_lc_f = torch.autograd.grad(
        loss_lc,
        [model.classifier.weight, features_v2],
        retain_graph=True,
    )
    grad_bcl_w, grad_bcl_f = torch.autograd.grad(
        loss_bcl,
        [model.classifier.weight, features_v2],
    )

    w_cos = safe_rowwise_cosine(grad_lc_w, grad_bcl_w).detach().cpu().numpy()
    w_lc_norm = grad_lc_w.norm(dim=1).detach().cpu().numpy()
    w_bcl_norm = grad_bcl_w.norm(dim=1).detach().cpu().numpy()
    w_lc_weighted_norm = LAMBDA_LC * w_lc_norm
    w_bcl_weighted_norm = MU_BCL * w_bcl_norm

    f_cos = safe_rowwise_cosine(grad_lc_f, grad_bcl_f).detach().cpu().numpy()
    f_lc_norm = grad_lc_f.norm(dim=1).detach().cpu().numpy()
    f_bcl_norm = grad_bcl_f.norm(dim=1).detach().cpu().numpy()
    f_lc_weighted_norm = LAMBDA_LC * f_lc_norm
    f_bcl_weighted_norm = MU_BCL * f_bcl_norm
    labels_np = labels.detach().cpu().numpy()

    rows = []
    # Emit only classes actually represented in the probe. The production
    # five-per-class cache covers all 100 classes; smaller unit fixtures need
    # not generate misleading empty-class NaN rows.
    for class_id in sorted(np.unique(labels_np).tolist()):
        class_mask = labels_np == class_id
        meta = class_meta[class_id]
        rows.append(
            {
                "probe_mode": "common_input_eval",
                "source_split": "LT_train_probe",
                "bn_mode": "eval",
                "epoch": epoch,
                "class_id": class_id,
                "coarse_id": meta["coarse_id"],
                "shot_group": meta["shot_group"],
                "lc_objective_weight": float(LAMBDA_LC),
                "bcl_objective_weight": float(MU_BCL),
                "W_y_gradient_cosine": float(w_cos[class_id]),
                "W_y_conflict_negative_cosine": bool(w_cos[class_id] < 0),
                "W_y_LC_gradient_norm": float(w_lc_norm[class_id]),
                "W_y_BCL_gradient_norm": float(w_bcl_norm[class_id]),
                "W_y_BCL_to_LC_norm_ratio": float(
                    w_bcl_norm[class_id] / max(w_lc_norm[class_id], 1e-12)
                ),
                "W_y_objective_weighted_LC_gradient_norm": float(w_lc_weighted_norm[class_id]),
                "W_y_objective_weighted_BCL_gradient_norm": float(w_bcl_weighted_norm[class_id]),
                "W_y_objective_weighted_BCL_to_LC_norm_ratio": float(
                    w_bcl_weighted_norm[class_id] / max(w_lc_weighted_norm[class_id], 1e-12)
                ),
                "same_f_feature_gradient_cosine_mean": float(np.mean(f_cos[class_mask])),
                "same_f_feature_gradient_cosine_median": float(np.median(f_cos[class_mask])),
                "same_f_feature_conflict_fraction": float(np.mean(f_cos[class_mask] < 0)),
                "same_f_LC_gradient_norm_mean": float(np.mean(f_lc_norm[class_mask])),
                "same_f_BCL_gradient_norm_mean": float(np.mean(f_bcl_norm[class_mask])),
                "same_f_BCL_to_LC_norm_ratio": float(
                    np.mean(f_bcl_norm[class_mask])
                    / max(float(np.mean(f_lc_norm[class_mask])), 1e-12)
                ),
                "same_f_objective_weighted_LC_gradient_norm_mean": float(
                    np.mean(f_lc_weighted_norm[class_mask])
                ),
                "same_f_objective_weighted_BCL_gradient_norm_mean": float(
                    np.mean(f_bcl_weighted_norm[class_mask])
                ),
                "same_f_objective_weighted_BCL_to_LC_norm_ratio": float(
                    np.mean(f_bcl_weighted_norm[class_mask])
                    / max(float(np.mean(f_lc_weighted_norm[class_mask])), 1e-12)
                ),
            }
        )
    return rows




def _snapshot_bn_running_buffers(model: BCLCIFARModel) -> dict[str, torch.Tensor]:
    suffixes = ("running_mean", "running_var", "num_batches_tracked")
    return {
        name: buffer.detach().clone()
        for name, buffer in model.named_buffers()
        if name.endswith(suffixes)
    }


def _restore_bn_running_buffers(model: BCLCIFARModel, snapshot: dict[str, torch.Tensor]):
    with torch.no_grad():
        for name, buffer in model.named_buffers():
            if name in snapshot:
                buffer.copy_(snapshot[name])


def _flatten_gradient_tensors(gradients) -> torch.Tensor:
    pieces = [
        gradient.detach().reshape(-1).to(device="cpu", dtype=torch.float64)
        for gradient in gradients
        if gradient is not None
    ]
    if not pieces:
        return torch.zeros(0, dtype=torch.float64)
    return torch.cat(pieces)


def scale_batch_mean_gradient(
    gradient: torch.Tensor,
    batch_size: int,
    total_count: int,
) -> torch.Tensor:
    """Rescale a batch-mean loss gradient to the full-probe mean-objective scale."""
    if batch_size <= 0 or total_count <= 0 or batch_size > total_count:
        raise ValueError(
            f"Invalid batch/probe sizes: batch_size={batch_size}, total_count={total_count}"
        )
    return gradient.detach() * (float(batch_size) / float(total_count))


def _scalar_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denominator = float(a.norm().item() * b.norm().item())
    if denominator <= 1e-30:
        return float("nan")
    return float(torch.dot(a.reshape(-1), b.reshape(-1)).item() / denominator)


def compute_native_three_view_gradient_diagnostics(
    model: BCLCIFARModel,
    epoch: int,
    probe: dict,
    class_meta: dict[int, dict[str, str]],
    criterion_lc: LogitCompensatedCE,
    criterion_bcl: BalancedContrastiveLoss,
    device: torch.device,
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Measure the native three-view train-mode graph using LT TRAIN probes only.

    Unlike the common-input eval probe, this uses one concatenated [v1,v2,v3]
    forward with train-mode BatchNorm. It compares gradients at classifier W,
    backbone parameters, the three-view feature tensor, and the image inputs.
    BN running buffers are restored after every diagnostic batch, so analysis
    cannot mutate the checkpoint being measured.
    """
    labels_all = probe["labels"].detach().cpu().long()
    count_all = int(labels_all.numel())
    if count_all == 0:
        raise RuntimeError("Native three-view gradient probe is empty")
    if probe.get("metadata", {}).get("test_split_loaded") is not False:
        raise RuntimeError("Gradient probes must be sourced only from LT TRAIN data")

    was_training = model.training
    bn_snapshot = _snapshot_bn_running_buffers(model)
    model.train()

    weight_lc_sum = torch.zeros_like(
        model.classifier.weight.detach(), device="cpu", dtype=torch.float64
    )
    weight_bcl_sum = torch.zeros_like(weight_lc_sum)
    backbone_lc_sum = None
    backbone_bcl_sum = None
    class_feature_records: dict[int, dict[int, dict[str, list[float]]]] = {}
    input_records = {
        view: {"cosine": [], "lc_norm": [], "bcl_norm": [], "conflict": []}
        for view in (1, 2, 3)
    }

    backbone_parameters = list(model.backbone.parameters())
    for start in range(0, count_all, batch_size):
        end = min(start + batch_size, count_all)
        labels = labels_all[start:end].to(device)
        current_batch = int(labels.shape[0])
        views = [
            probe[f"view{view}"][start:end].to(device, non_blocking=True)
            for view in (1, 2, 3)
        ]
        all_views = torch.cat(views, dim=0).detach().requires_grad_(True)

        try:
            features, all_logits, all_embeddings, prototypes = model(
                all_views, compute_contrastive=True
            )
            logits_v1 = all_logits[:current_batch]
            z2 = all_embeddings[current_batch : 2 * current_batch]
            z3 = all_embeddings[2 * current_batch : 3 * current_batch]
            two_view_embeddings = torch.stack([z2, z3], dim=1)
            loss_lc = criterion_lc(logits_v1, labels)
            loss_bcl = criterion_bcl(prototypes, two_view_embeddings, labels)

            gradient_targets = [
                model.classifier.weight,
                features,
                *backbone_parameters,
                all_views,
            ]
            gradients_lc = torch.autograd.grad(
                loss_lc,
                gradient_targets,
                retain_graph=True,
                allow_unused=True,
            )
            gradients_bcl = torch.autograd.grad(
                loss_bcl,
                gradient_targets,
                allow_unused=True,
            )

            weight_lc_sum += scale_batch_mean_gradient(
                gradients_lc[0], current_batch, count_all
            ).cpu().double()
            weight_bcl_sum += scale_batch_mean_gradient(
                gradients_bcl[0], current_batch, count_all
            ).cpu().double()

            backbone_lc = _flatten_gradient_tensors(
                gradients_lc[2 : 2 + len(backbone_parameters)]
            )
            backbone_bcl = _flatten_gradient_tensors(
                gradients_bcl[2 : 2 + len(backbone_parameters)]
            )
            if backbone_lc_sum is None:
                backbone_lc_sum = torch.zeros_like(backbone_lc)
                backbone_bcl_sum = torch.zeros_like(backbone_bcl)
            backbone_lc_sum += scale_batch_mean_gradient(
                backbone_lc, current_batch, count_all
            )
            backbone_bcl_sum += scale_batch_mean_gradient(
                backbone_bcl, current_batch, count_all
            )

            # Losses are batch means. Scale each row by B/N to undo its local
            # 1/B reduction and express per-sample rows on the full-probe mean
            # objective scale. This also corrects the smaller final batch.
            feature_lc = scale_batch_mean_gradient(
                gradients_lc[1], current_batch, count_all
            )
            feature_bcl = scale_batch_mean_gradient(
                gradients_bcl[1], current_batch, count_all
            )
            input_lc = scale_batch_mean_gradient(
                gradients_lc[-1], current_batch, count_all
            )
            input_bcl = scale_batch_mean_gradient(
                gradients_bcl[-1], current_batch, count_all
            )
            labels_np = labels.detach().cpu().numpy()

            for view_index in (1, 2, 3):
                view_slice = slice(
                    (view_index - 1) * current_batch,
                    view_index * current_batch,
                )
                f_lc = feature_lc[view_slice]
                f_bcl = feature_bcl[view_slice]
                feature_cos = safe_rowwise_cosine(f_lc, f_bcl).cpu().numpy()
                feature_lc_norm = f_lc.norm(dim=1).cpu().numpy()
                feature_bcl_norm = f_bcl.norm(dim=1).cpu().numpy()

                x_lc = input_lc[view_slice].flatten(1)
                x_bcl = input_bcl[view_slice].flatten(1)
                input_cos = safe_rowwise_cosine(x_lc, x_bcl).cpu().numpy()
                input_lc_norm = x_lc.norm(dim=1).cpu().numpy()
                input_bcl_norm = x_bcl.norm(dim=1).cpu().numpy()
                input_records[view_index]["cosine"].extend(input_cos.tolist())
                input_records[view_index]["lc_norm"].extend(input_lc_norm.tolist())
                input_records[view_index]["bcl_norm"].extend(input_bcl_norm.tolist())
                input_records[view_index]["conflict"].extend((input_cos < 0).tolist())

                for sample_index, class_id in enumerate(labels_np.tolist()):
                    record = class_feature_records.setdefault(
                        int(class_id),
                        {
                            view: {
                                "cosine": [],
                                "lc_norm": [],
                                "bcl_norm": [],
                                "conflict": [],
                            }
                            for view in (1, 2, 3)
                        },
                    )[view_index]
                    record["cosine"].append(float(feature_cos[sample_index]))
                    record["lc_norm"].append(float(feature_lc_norm[sample_index]))
                    record["bcl_norm"].append(float(feature_bcl_norm[sample_index]))
                    record["conflict"].append(bool(feature_cos[sample_index] < 0))
        finally:
            # Train-mode BN uses current-batch statistics in the forward pass;
            # restore running buffers so the checkpoint remains byte-identical.
            _restore_bn_running_buffers(model, bn_snapshot)
            batch_bn_snapshot = _snapshot_bn_running_buffers(model)
            if batch_bn_snapshot.keys() != bn_snapshot.keys() or any(
                not torch.equal(batch_bn_snapshot[name], bn_snapshot[name])
                for name in bn_snapshot
            ):
                raise RuntimeError(
                    "Native three-view gradient probe failed to restore a BatchNorm buffer "
                    "after the current diagnostic batch"
                )

    current_bn_snapshot = _snapshot_bn_running_buffers(model)
    if current_bn_snapshot.keys() != bn_snapshot.keys() or any(
        not torch.equal(current_bn_snapshot[name], bn_snapshot[name])
        for name in bn_snapshot
    ):
        model.train(was_training)
        raise RuntimeError("Native three-view gradient probe changed a BatchNorm running buffer")
    bn_buffers_restored = True
    model.train(was_training)

    if backbone_lc_sum is None or backbone_bcl_sum is None:
        raise RuntimeError("Native three-view gradient probe produced no backbone gradients")

    weight_cos = safe_rowwise_cosine(weight_lc_sum, weight_bcl_sum).numpy()
    weight_lc_norm = weight_lc_sum.norm(dim=1).numpy()
    weight_bcl_norm = weight_bcl_sum.norm(dim=1).numpy()
    weight_lc_weighted_norm = LAMBDA_LC * weight_lc_norm
    weight_bcl_weighted_norm = MU_BCL * weight_bcl_norm

    class_rows = []
    for class_id in sorted(class_feature_records):
        meta = class_meta[class_id]
        row = {
            "probe_mode": "native_three_view_train",
            "source_split": "LT_train_probe",
            "bn_mode": "train",
            "epoch": epoch,
            "class_id": class_id,
            "coarse_id": meta["coarse_id"],
            "shot_group": meta["shot_group"],
            "lc_objective_weight": float(LAMBDA_LC),
            "bcl_objective_weight": float(MU_BCL),
            "W_y_gradient_cosine": float(weight_cos[class_id]),
            "W_y_conflict_negative_cosine": bool(weight_cos[class_id] < 0),
            "W_y_LC_gradient_norm": float(weight_lc_norm[class_id]),
            "W_y_BCL_gradient_norm": float(weight_bcl_norm[class_id]),
            "W_y_BCL_to_LC_norm_ratio": float(
                weight_bcl_norm[class_id] / max(weight_lc_norm[class_id], 1e-12)
            ),
            "W_y_objective_weighted_LC_gradient_norm": float(weight_lc_weighted_norm[class_id]),
            "W_y_objective_weighted_BCL_gradient_norm": float(weight_bcl_weighted_norm[class_id]),
            "W_y_objective_weighted_BCL_to_LC_norm_ratio": float(
                weight_bcl_weighted_norm[class_id]
                / max(weight_lc_weighted_norm[class_id], 1e-12)
            ),
        }
        for view_index in (1, 2, 3):
            values = class_feature_records[class_id][view_index]
            suffix = f"v{view_index}"
            row[f"feature_gradient_cosine_{suffix}_mean"] = float(np.mean(values["cosine"]))
            row[f"feature_gradient_conflict_{suffix}_fraction"] = float(
                np.mean(values["conflict"])
            )
            row[f"feature_gradient_LC_{suffix}_norm_mean"] = float(np.mean(values["lc_norm"]))
            row[f"feature_gradient_BCL_{suffix}_norm_mean"] = float(np.mean(values["bcl_norm"]))
            row[f"feature_gradient_BCL_to_LC_{suffix}_norm_ratio"] = float(
                np.mean(values["bcl_norm"]) / max(np.mean(values["lc_norm"]), 1e-12)
            )
            row[f"feature_gradient_objective_weighted_LC_{suffix}_norm_mean"] = float(
                LAMBDA_LC * np.mean(values["lc_norm"])
            )
            row[f"feature_gradient_objective_weighted_BCL_{suffix}_norm_mean"] = float(
                MU_BCL * np.mean(values["bcl_norm"])
            )
            row[f"feature_gradient_objective_weighted_BCL_to_LC_{suffix}_norm_ratio"] = float(
                MU_BCL * np.mean(values["bcl_norm"])
                / max(LAMBDA_LC * np.mean(values["lc_norm"]), 1e-12)
            )
        class_rows.append(row)

    global_row = {
        "probe_mode": "native_three_view_train",
        "source_split": "LT_train_probe",
        "bn_mode": "train",
        "epoch": epoch,
        "lc_objective_weight": float(LAMBDA_LC),
        "bcl_objective_weight": float(MU_BCL),
        "backbone_gradient_cosine": _scalar_cosine(
            backbone_lc_sum, backbone_bcl_sum
        ),
        "backbone_LC_gradient_norm": float(backbone_lc_sum.norm().item()),
        "backbone_BCL_gradient_norm": float(backbone_bcl_sum.norm().item()),
        "backbone_BCL_to_LC_norm_ratio": float(
            backbone_bcl_sum.norm().item() / max(backbone_lc_sum.norm().item(), 1e-12)
        ),
        "backbone_objective_weighted_LC_gradient_norm": float(
            LAMBDA_LC * backbone_lc_sum.norm().item()
        ),
        "backbone_objective_weighted_BCL_gradient_norm": float(
            MU_BCL * backbone_bcl_sum.norm().item()
        ),
        "backbone_objective_weighted_BCL_to_LC_norm_ratio": float(
            MU_BCL * backbone_bcl_sum.norm().item()
            / max(LAMBDA_LC * backbone_lc_sum.norm().item(), 1e-12)
        ),
        "bn_running_buffers_restored": bn_buffers_restored,
        "gradient_source_split": "LT_train_probe_only",
    }
    for view_index in (1, 2, 3):
        values = input_records[view_index]
        suffix = f"v{view_index}"
        global_row[f"input_gradient_cosine_{suffix}_mean"] = float(np.mean(values["cosine"]))
        global_row[f"input_gradient_conflict_{suffix}_fraction"] = float(
            np.mean(values["conflict"])
        )
        global_row[f"input_gradient_LC_{suffix}_norm_mean"] = float(np.mean(values["lc_norm"]))
        global_row[f"input_gradient_BCL_{suffix}_norm_mean"] = float(np.mean(values["bcl_norm"]))
        global_row[f"input_gradient_objective_weighted_LC_{suffix}_norm_mean"] = float(
            LAMBDA_LC * np.mean(values["lc_norm"])
        )
        global_row[f"input_gradient_objective_weighted_BCL_{suffix}_norm_mean"] = float(
            MU_BCL * np.mean(values["bcl_norm"])
        )
        global_row[f"input_gradient_objective_weighted_BCL_to_LC_{suffix}_norm_ratio"] = float(
            MU_BCL * np.mean(values["bcl_norm"])
            / max(LAMBDA_LC * np.mean(values["lc_norm"]), 1e-12)
        )
    return class_rows, global_row


def compute_test_margin_rows(
    group: str,
    labels: np.ndarray,
    logits: np.ndarray,
    adjusted_logits: np.ndarray,
    class_meta: dict[int, dict[str, str]],
) -> list[dict[str, Any]]:
    """Read-only TEST margin calculation; intentionally contains no autograd."""
    labels = np.asarray(labels, dtype=np.int64)
    logits = np.asarray(logits, dtype=np.float64)
    adjusted_logits = np.asarray(adjusted_logits, dtype=np.float64)
    if logits.shape != (labels.size, 100) or adjusted_logits.shape != logits.shape:
        raise RuntimeError("TEST logits must have shape [num_test, 100]")
    if labels.size == 0 or np.any((labels < 0) | (labels >= 100)):
        raise RuntimeError("Invalid TEST labels for margin calculation")

    row_index = np.arange(labels.size)
    raw_true = logits[row_index, labels]
    adjusted_true = adjusted_logits[row_index, labels]
    raw_competitor = logits.copy()
    adjusted_competitor = adjusted_logits.copy()
    raw_competitor[row_index, labels] = -np.inf
    adjusted_competitor[row_index, labels] = -np.inf
    raw_margin = raw_true - raw_competitor.max(axis=1)
    adjusted_margin = adjusted_true - adjusted_competitor.max(axis=1)
    predictions = logits.argmax(axis=1)

    rows = []
    for class_id in range(100):
        mask = labels == class_id
        if not np.any(mask):
            raise RuntimeError(f"TEST split contains no examples for class {class_id}")
        meta = class_meta[class_id]
        rows.append({
            "group": group,
            "source_split": "balanced_TEST_read_only",
            "class_id": class_id,
            "coarse_id": meta["coarse_id"],
            "shot_group": meta["shot_group"],
            "test_count": int(mask.sum()),
            "raw_top1": 100.0 * float(np.mean(predictions[mask] == labels[mask])),
            "raw_margin_mean": float(np.mean(raw_margin[mask])),
            "raw_margin_median": float(np.median(raw_margin[mask])),
            "adjusted_margin_mean": float(np.mean(adjusted_margin[mask])),
            "adjusted_margin_median": float(np.median(adjusted_margin[mask])),
        })
    return rows


def compute_pair_effects(
    labels: np.ndarray,
    predictions_lc: np.ndarray,
    predictions_bcl: np.ndarray,
    logits_lc: np.ndarray,
    logits_bcl: np.ndarray,
    adjusted_lc: np.ndarray,
    adjusted_bcl: np.ndarray,
    pair_rows: list[dict[str, str]],
):
    effects = []
    for pair in pair_rows:
        head_id = int(pair["head_fine_id"])
        tail_id = int(pair["tail_fine_id"])
        same_superclass = parse_bool(pair["same_superclass"])
        dino_value = pair.get("dino_topk_union")
        dino_near = None
        if dino_value is not None and str(dino_value).strip():
            dino_near = parse_bool(dino_value)

        for direction, source_id, target_id in (
            ("Head_to_Tail", head_id, tail_id),
            ("Tail_to_Head", tail_id, head_id),
        ):
            mask = labels == source_id
            if not np.any(mask):
                raise RuntimeError(f"No evaluation examples for source class {source_id}")

            def rate(predictions):
                return float(np.mean(predictions[mask] == target_id))

            def pair_margin(logits):
                return float(np.mean(logits[mask, source_id] - logits[mask, target_id]))

            lc_confusion = rate(predictions_lc)
            bcl_confusion = rate(predictions_bcl)
            lc_margin = pair_margin(logits_lc)
            bcl_margin = pair_margin(logits_bcl)
            lc_adjusted_margin = pair_margin(adjusted_lc)
            bcl_adjusted_margin = pair_margin(adjusted_bcl)

            effects.append(
                {
                    "direction": direction,
                    "source_split": "balanced_TEST_read_only",
                    "head_class": head_id,
                    "tail_class": tail_id,
                    "source_class": source_id,
                    "target_class": target_id,
                    "same_superclass": same_superclass,
                    "dino_topk_union": "" if dino_near is None else dino_near,
                    "LC_confusion_rate": lc_confusion,
                    "BCL_confusion_rate": bcl_confusion,
                    "delta_confusion_BCL_minus_LC": bcl_confusion - lc_confusion,
                    "LC_raw_pair_margin": lc_margin,
                    "BCL_raw_pair_margin": bcl_margin,
                    "delta_raw_margin_BCL_minus_LC": bcl_margin - lc_margin,
                    "LC_adjusted_pair_margin": lc_adjusted_margin,
                    "BCL_adjusted_pair_margin": bcl_adjusted_margin,
                    "delta_adjusted_margin_BCL_minus_LC": (
                        bcl_adjusted_margin - lc_adjusted_margin
                    ),
                }
            )
    return effects


def summarize_did(effects: list[dict[str, Any]]):
    summaries = []
    relation_specs = [
        ("same_superclass", "SameSuperclass", "DifferentSuperclass"),
        ("dino_topk_union", "DINO_near", "DINO_not_near"),
    ]
    metrics = [
        "delta_confusion_BCL_minus_LC",
        "delta_raw_margin_BCL_minus_LC",
        "delta_adjusted_margin_BCL_minus_LC",
    ]
    for relation_key, positive_label, negative_label in relation_specs:
        relation_rows = effects
        if relation_key == "dino_topk_union":
            relation_rows = [row for row in effects if row[relation_key] != ""]
            if not relation_rows:
                continue
        for direction in ("Head_to_Tail", "Tail_to_Head"):
            directional = [row for row in relation_rows if row["direction"] == direction]
            positive = [
                row for row in directional if parse_bool(row[relation_key])
            ]
            negative = [
                row for row in directional if not parse_bool(row[relation_key])
            ]
            for metric in metrics:
                pos_values = [float(row[metric]) for row in positive]
                neg_values = [float(row[metric]) for row in negative]
                pos_mean = float(np.mean(pos_values)) if pos_values else float("nan")
                neg_mean = float(np.mean(neg_values)) if neg_values else float("nan")
                did = pos_mean - neg_mean if pos_values and neg_values else float("nan")
                summaries.append(
                    {
                        "relation": relation_key,
                        "direction": direction,
                        "metric": metric,
                        "slice": positive_label,
                        "num_pairs": len(pos_values),
                        "mean_delta": pos_mean,
                        "DID_positive_minus_negative": did,
                    }
                )
                summaries.append(
                    {
                        "relation": relation_key,
                        "direction": direction,
                        "metric": metric,
                        "slice": negative_label,
                        "num_pairs": len(neg_values),
                        "mean_delta": neg_mean,
                        "DID_positive_minus_negative": did,
                    }
                )
    return summaries


def validate_phase1_artifacts(output_root: Path, indices_file: Path) -> dict[str, Any]:
    """Refuse Phase-2 analysis unless the paired Phase-1 pilot is complete."""
    required = [
        output_root / "paired_data_stream_audit.json",
        output_root / "paired_pilot_summary.json",
        output_root / "init_seed42.pth",
    ]
    for group in ("LC_only", "BCL"):
        group_root = output_root / f"{group}_seed42"
        required.extend([
            group_root / "training_summary.json",
            group_root / "final_test_metrics.json",
            group_root / "final_test_predictions.npz",
        ])
        required.extend(
            group_root / "checkpoints" / filename for _, filename in CHECKPOINTS
        )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Phase-1 artifacts are incomplete; do not run Phase 2: " + ", ".join(missing)
        )

    with (output_root / "paired_data_stream_audit.json").open(encoding="utf-8") as stream:
        stream_audit = json.load(stream)
    if (
        stream_audit.get("paired_data_streams_identical") is not True
        or stream_audit.get("epochs_compared") != 200
        or stream_audit.get("mismatch_count") != 0
    ):
        raise RuntimeError("Phase-1 paired data-stream audit is not a complete 200/200 pass")

    with (output_root / "paired_pilot_summary.json").open(encoding="utf-8") as stream:
        paired_summary = json.load(stream)
    if paired_summary.get("seed") != 42 or paired_summary.get("epochs") != 200:
        raise RuntimeError("Phase-1 summary does not describe the required seed-42 200-epoch pilot")
    if not indices_file.is_file():
        raise FileNotFoundError(f"Missing fixed IF=100 indices: {indices_file}")
    actual_indices_hash = file_sha256(indices_file)
    if paired_summary.get("indices_file_sha256") != actual_indices_hash:
        raise RuntimeError("Phase-1 summary index SHA-256 does not match the supplied fixed indices")

    initial_hash = paired_summary.get("initial_model_state_sha256")
    if not isinstance(initial_hash, str) or len(initial_hash) != 64:
        raise RuntimeError("Phase-1 summary is missing the initial model-state SHA-256")
    initial_checkpoint = safe_torch_load(output_root / "init_seed42.pth", map_location="cpu")
    initial_state = initial_checkpoint.get("model_state")
    if (
        initial_checkpoint.get("seed") != 42
        or initial_checkpoint.get("model_state_sha256") != initial_hash
        or not isinstance(initial_state, dict)
        or state_dict_sha256(initial_state) != initial_hash
    ):
        raise RuntimeError("Saved init_seed42.pth does not match the paired initial-state SHA-256")
    for group in ("LC_only", "BCL"):
        group_root = output_root / f"{group}_seed42"
        with (group_root / "training_summary.json").open(encoding="utf-8") as stream:
            training_summary = json.load(stream)
        if (
            training_summary.get("group") != group
            or training_summary.get("seed") != 42
            or training_summary.get("epochs") != 200
            or training_summary.get("initial_model_state_sha256") != initial_hash
        ):
            raise RuntimeError(f"{group} training summary is inconsistent with the paired pilot")
        with np.load(group_root / "final_test_predictions.npz") as archive:
            required_arrays = {"labels", "predictions", "logits", "adjusted_logits"}
            if not required_arrays.issubset(set(archive.files)):
                raise RuntimeError(f"{group} final TEST archive is incomplete")
            labels = archive["labels"]
            predictions = archive["predictions"]
            logits = archive["logits"]
            adjusted_logits = archive["adjusted_logits"]
            if (
                labels.shape != (10_000,)
                or predictions.shape != (10_000,)
                or logits.shape != (10_000, 100)
                or adjusted_logits.shape != (10_000, 100)
                or not np.array_equal(np.bincount(labels.astype(np.int64), minlength=100),
                                      np.full(100, 100, dtype=np.int64))
            ):
                raise RuntimeError(f"{group} final TEST archive has invalid shapes or class counts")
        with (group_root / "final_test_metrics.json").open(encoding="utf-8") as stream:
            metrics = json.load(stream)
        if metrics.get("group") != group or metrics.get("seed") != 42:
            raise RuntimeError(f"{group} final TEST metrics are not paired seed-42 results")
    return {
        "paired_data_stream_audit": stream_audit,
        "initial_model_state_sha256": initial_hash,
        "indices_file_sha256": actual_indices_hash,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Offline paired BCL diagnostics: margins, dual-layer gradients, Head-Tail DID"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--indices-file", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("BCL_AUDIT/outputs"))
    parser.add_argument("--data-audit-dir", type=Path, default=None)
    parser.add_argument("--probe-per-class", type=int, default=5)
    parser.add_argument("--probe-seed", type=int, default=777)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    output_root = args.output_root.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    indices_file = args.indices_file.expanduser().resolve()
    data_audit_dir = (
        args.data_audit_dir.expanduser().resolve()
        if args.data_audit_dir
        else output_root / "data_audit"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    phase2_output_names = (
        "offline_probe_margin_metrics.csv",
        "offline_dual_gradient_metrics.csv",
        "offline_native_three_view_gradient_class_metrics.csv",
        "offline_native_three_view_gradient_global_metrics.csv",
        "offline_test_margin_metrics.csv",
        "offline_test_margin_summary.json",
        "head_tail_pair_effects.csv",
        "head_tail_did_summary.csv",
        "stage_gate.json",
        "phase2_h1_class_gain.csv",
        "phase2_h2_few_exposure.csv",
        "phase2_h2_adjusted_associations.csv",
        "phase2_h2_pair_matched_did.csv",
        "phase2_h3_overlap_gradient_summary.csv",
        "phase2_h3_overlap_margin_trajectory.csv",
        "PHASE2_DECISION_BRIEF.md",
        "phase2_decision_summary.json",
    )
    existing_phase2_outputs = [
        str(output_root / name) for name in phase2_output_names
        if (output_root / name).exists()
    ]
    if existing_phase2_outputs:
        raise FileExistsError(
            "Refusing to overwrite Phase-2 diagnostics; preserve the previous "
            "analysis and use a fresh, separately copied Phase-1 output root: "
            + ", ".join(existing_phase2_outputs)
        )
    static_audit = validate_static_data_audit(data_audit_dir, indices_file)
    phase1_artifacts = validate_phase1_artifacts(output_root, indices_file)
    if phase1_artifacts["indices_file_sha256"] != static_audit["indices_file_sha256"]:
        raise RuntimeError("Phase-1 pilot and static data-audit indices SHA-256 differ")

    if not (1 <= args.probe_per_class <= 5):
        parser.error("--probe-per-class must be between 1 and 5 for the fixed IF=100 split")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    class_rows = read_csv(data_audit_dir / "class_counts.csv")
    class_meta = {
        int(row["fine_id"]): {
            "coarse_id": row["coarse_id"],
            "shot_group": row["shot_group"],
        }
        for row in class_rows
    }
    if set(class_meta) != set(range(100)):
        raise RuntimeError("class_counts.csv must contain all official fine IDs 0..99")

    # Validate all saved DINO artifacts and the complete 35x30 pair sidecar before
    # creating probe caches or running any expensive gradient diagnostics.
    topology = load_and_validate_dino_topology(data_audit_dir, class_meta)
    pair_rows = load_pair_topology(data_audit_dir)

    probe_cache_path = output_root / "offline_probe" / (
        f"balanced_lt_probe_{args.probe_per_class}perclass_seed{args.probe_seed}.pt"
    )
    probe = load_or_create_probe_cache(
        data_root=data_root,
        indices_file=indices_file,
        cache_path=probe_cache_path,
        per_class=args.probe_per_class,
        seed=args.probe_seed,
    )
    probe["labels"] = probe["labels"].long()
    probe_metadata = probe.get("metadata", {})
    if probe_metadata.get("test_split_loaded") is not False:
        raise RuntimeError("Offline gradients/margins require a TRAIN-only probe cache")
    if "training split" not in str(probe_metadata.get("source_split", "")):
        raise RuntimeError("Probe cache does not certify the original TRAIN split")
    expected_probe_order = "per-class fixed-count selection followed by seeded global shuffle"
    if probe_metadata.get("probe_order") != expected_probe_order:
        raise RuntimeError(
            "Probe cache is stale or class-major; use a fresh Phase-2 output root "
            "to regenerate the globally shuffled LT TRAIN-only probe."
        )
    class_counts = cifar100lt_class_counts()
    criterion_lc = LogitCompensatedCE(class_counts).to(device)
    criterion_bcl = BalancedContrastiveLoss(class_counts, temperature=0.1).to(device)

    all_margin_rows = []
    all_gradient_rows = []
    all_native_gradient_class_rows = []
    all_native_gradient_global_rows = []
    for group in ("LC_only", "BCL"):
        run_dir = output_root / f"{group}_seed42" / "checkpoints"
        for epoch, filename in CHECKPOINTS:
            model = load_model_checkpoint(
                run_dir / filename,
                device,
                expected_group=group,
                expected_epoch=epoch,
                expected_initial_sha256=phase1_artifacts["initial_model_state_sha256"],
            )
            all_margin_rows.extend(
                compute_probe_margin_rows(
                    model=model,
                    group=group,
                    epoch=epoch,
                    probe=probe,
                    class_meta=class_meta,
                    criterion_lc=criterion_lc,
                    device=device,
                    batch_size=args.batch_size,
                )
            )
            if group == "BCL":
                all_gradient_rows.extend(
                    compute_bcl_gradient_rows(
                        model=model,
                        epoch=epoch,
                        probe=probe,
                        class_meta=class_meta,
                        criterion_lc=criterion_lc,
                        criterion_bcl=criterion_bcl,
                        device=device,
                        batch_size=args.batch_size,
                    )
                )
                native_class_rows, native_global_row = compute_native_three_view_gradient_diagnostics(
                    model=model,
                    epoch=epoch,
                    probe=probe,
                    class_meta=class_meta,
                    criterion_lc=criterion_lc,
                    criterion_bcl=criterion_bcl,
                    device=device,
                    batch_size=args.batch_size,
                )
                all_native_gradient_class_rows.extend(native_class_rows)
                all_native_gradient_global_rows.append(native_global_row)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    write_csv(
        output_root / "offline_probe_margin_metrics.csv",
        all_margin_rows,
        [
            "group", "source_split", "view_type", "epoch", "class_id", "coarse_id", "shot_group",
            "probe_count", "raw_top1", "raw_margin_mean", "raw_margin_median",
            "adjusted_margin_mean", "adjusted_margin_median",
        ],
    )
    write_csv(
        output_root / "offline_dual_gradient_metrics.csv",
        all_gradient_rows,
        [
            "probe_mode", "source_split", "bn_mode", "epoch", "class_id", "coarse_id", "shot_group",
            "W_y_gradient_cosine", "W_y_conflict_negative_cosine",
            "lc_objective_weight", "bcl_objective_weight",
            "W_y_LC_gradient_norm", "W_y_BCL_gradient_norm",
            "W_y_BCL_to_LC_norm_ratio",
            "W_y_objective_weighted_LC_gradient_norm",
            "W_y_objective_weighted_BCL_gradient_norm",
            "W_y_objective_weighted_BCL_to_LC_norm_ratio",
            "same_f_feature_gradient_cosine_mean",
            "same_f_feature_gradient_cosine_median",
            "same_f_feature_conflict_fraction",
            "same_f_LC_gradient_norm_mean",
            "same_f_BCL_gradient_norm_mean",
            "same_f_BCL_to_LC_norm_ratio",
            "same_f_objective_weighted_LC_gradient_norm_mean",
            "same_f_objective_weighted_BCL_gradient_norm_mean",
            "same_f_objective_weighted_BCL_to_LC_norm_ratio",
        ],
    )

    write_csv(
        output_root / "offline_native_three_view_gradient_class_metrics.csv",
        all_native_gradient_class_rows,
        [
            "probe_mode", "source_split", "bn_mode", "epoch", "class_id",
            "coarse_id", "shot_group",
            "W_y_gradient_cosine", "W_y_conflict_negative_cosine",
            "lc_objective_weight", "bcl_objective_weight",
            "W_y_LC_gradient_norm", "W_y_BCL_gradient_norm",
            "W_y_BCL_to_LC_norm_ratio",
            "W_y_objective_weighted_LC_gradient_norm",
            "W_y_objective_weighted_BCL_gradient_norm",
            "W_y_objective_weighted_BCL_to_LC_norm_ratio",
            "feature_gradient_cosine_v1_mean", "feature_gradient_conflict_v1_fraction",
            "feature_gradient_LC_v1_norm_mean", "feature_gradient_BCL_v1_norm_mean",
            "feature_gradient_BCL_to_LC_v1_norm_ratio",
            "feature_gradient_objective_weighted_LC_v1_norm_mean",
            "feature_gradient_objective_weighted_BCL_v1_norm_mean",
            "feature_gradient_objective_weighted_BCL_to_LC_v1_norm_ratio",
            "feature_gradient_cosine_v2_mean", "feature_gradient_conflict_v2_fraction",
            "feature_gradient_LC_v2_norm_mean", "feature_gradient_BCL_v2_norm_mean",
            "feature_gradient_BCL_to_LC_v2_norm_ratio",
            "feature_gradient_objective_weighted_LC_v2_norm_mean",
            "feature_gradient_objective_weighted_BCL_v2_norm_mean",
            "feature_gradient_objective_weighted_BCL_to_LC_v2_norm_ratio",
            "feature_gradient_cosine_v3_mean", "feature_gradient_conflict_v3_fraction",
            "feature_gradient_LC_v3_norm_mean", "feature_gradient_BCL_v3_norm_mean",
            "feature_gradient_BCL_to_LC_v3_norm_ratio",
            "feature_gradient_objective_weighted_LC_v3_norm_mean",
            "feature_gradient_objective_weighted_BCL_v3_norm_mean",
            "feature_gradient_objective_weighted_BCL_to_LC_v3_norm_ratio",
        ],
    )
    write_csv(
        output_root / "offline_native_three_view_gradient_global_metrics.csv",
        all_native_gradient_global_rows,
        [
            "probe_mode", "source_split", "bn_mode", "epoch",
            "lc_objective_weight", "bcl_objective_weight",
            "backbone_gradient_cosine", "backbone_LC_gradient_norm",
            "backbone_BCL_gradient_norm", "backbone_BCL_to_LC_norm_ratio",
            "backbone_objective_weighted_LC_gradient_norm",
            "backbone_objective_weighted_BCL_gradient_norm",
            "backbone_objective_weighted_BCL_to_LC_norm_ratio",
            "bn_running_buffers_restored", "gradient_source_split",
            "input_gradient_cosine_v1_mean", "input_gradient_conflict_v1_fraction",
            "input_gradient_LC_v1_norm_mean", "input_gradient_BCL_v1_norm_mean",
            "input_gradient_objective_weighted_LC_v1_norm_mean",
            "input_gradient_objective_weighted_BCL_v1_norm_mean",
            "input_gradient_objective_weighted_BCL_to_LC_v1_norm_ratio",
            "input_gradient_cosine_v2_mean", "input_gradient_conflict_v2_fraction",
            "input_gradient_LC_v2_norm_mean", "input_gradient_BCL_v2_norm_mean",
            "input_gradient_objective_weighted_LC_v2_norm_mean",
            "input_gradient_objective_weighted_BCL_v2_norm_mean",
            "input_gradient_objective_weighted_BCL_to_LC_v2_norm_ratio",
            "input_gradient_cosine_v3_mean", "input_gradient_conflict_v3_fraction",
            "input_gradient_LC_v3_norm_mean", "input_gradient_BCL_v3_norm_mean",
            "input_gradient_objective_weighted_LC_v3_norm_mean",
            "input_gradient_objective_weighted_BCL_v3_norm_mean",
            "input_gradient_objective_weighted_BCL_to_LC_v3_norm_ratio",
        ],
    )

    prediction_files = {
        group: output_root / f"{group}_seed42" / "final_test_predictions.npz"
        for group in ("LC_only", "BCL")
    }
    loaded = {}
    for group, path in prediction_files.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing final test prediction archive: {path}")
        archive = np.load(path)
        required = {"labels", "predictions", "logits", "adjusted_logits"}
        if not required.issubset(set(archive.files)):
            raise RuntimeError(
                f"{path} lacks {sorted(required - set(archive.files))}; "
                "rerun the updated final-evaluation stage"
            )
        loaded[group] = {key: archive[key] for key in required}

    labels_lc = loaded["LC_only"]["labels"].astype(np.int64)
    labels_bcl = loaded["BCL"]["labels"].astype(np.int64)
    if not np.array_equal(labels_lc, labels_bcl):
        raise RuntimeError("LC-only and BCL final evaluation labels are not paired")
    expected_test_counts = np.full(100, 100, dtype=np.int64)
    if not np.array_equal(np.bincount(labels_lc, minlength=100), expected_test_counts):
        raise RuntimeError("Expected the balanced CIFAR-100 TEST split: 100 examples per class")

    test_margin_rows = []
    for group in ("LC_only", "BCL"):
        test_margin_rows.extend(
            compute_test_margin_rows(
                group=group,
                labels=loaded[group]["labels"],
                logits=loaded[group]["logits"],
                adjusted_logits=loaded[group]["adjusted_logits"],
                class_meta=class_meta,
            )
        )
    write_csv(
        output_root / "offline_test_margin_metrics.csv",
        test_margin_rows,
        [
            "group", "source_split", "class_id", "coarse_id", "shot_group",
            "test_count", "raw_top1", "raw_margin_mean", "raw_margin_median",
            "adjusted_margin_mean", "adjusted_margin_median",
        ],
    )
    test_margin_summary = {}
    for group in ("LC_only", "BCL"):
        group_rows = [row for row in test_margin_rows if row["group"] == group]
        test_margin_summary[group] = {}
        for shot_group in ("many", "medium", "few"):
            subset = [row for row in group_rows if row["shot_group"] == shot_group]
            test_margin_summary[group][shot_group] = {
                "num_classes": len(subset),
                "raw_margin_mean": float(np.mean([row["raw_margin_mean"] for row in subset])),
                "adjusted_margin_mean": float(
                    np.mean([row["adjusted_margin_mean"] for row in subset])
                ),
                "raw_top1_macro_class": float(np.mean([row["raw_top1"] for row in subset])),
            }
    with (output_root / "offline_test_margin_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(test_margin_summary, stream, indent=2, ensure_ascii=False)

    effects = compute_pair_effects(
        labels=labels_lc,
        predictions_lc=loaded["LC_only"]["predictions"],
        predictions_bcl=loaded["BCL"]["predictions"],
        logits_lc=loaded["LC_only"]["logits"],
        logits_bcl=loaded["BCL"]["logits"],
        adjusted_lc=loaded["LC_only"]["adjusted_logits"],
        adjusted_bcl=loaded["BCL"]["adjusted_logits"],
        pair_rows=pair_rows,
    )
    did_rows = summarize_did(effects)
    write_csv(
        output_root / "head_tail_pair_effects.csv",
        effects,
        [
            "direction", "source_split", "head_class", "tail_class", "source_class", "target_class",
            "same_superclass", "dino_topk_union",
            "LC_confusion_rate", "BCL_confusion_rate",
            "delta_confusion_BCL_minus_LC",
            "LC_raw_pair_margin", "BCL_raw_pair_margin",
            "delta_raw_margin_BCL_minus_LC",
            "LC_adjusted_pair_margin", "BCL_adjusted_pair_margin",
            "delta_adjusted_margin_BCL_minus_LC",
        ],
    )
    write_csv(
        output_root / "head_tail_did_summary.csv",
        did_rows,
        [
            "relation", "direction", "metric", "slice",
            "num_pairs", "mean_delta", "DID_positive_minus_negative",
        ],
    )

    # H1-H3 synthesis is part of this same one-shot offline pass; it reuses
    # the saved predictions, existing margin/gradient CSVs and TRAIN-only DINO graph.
    decision_summary = run_phase2_decision_audit(
        output_root=output_root,
        data_audit_dir=data_audit_dir,
        class_meta=class_meta,
        topology=topology,
    )

    metric_names = (
        "overall_top1",
        "many_top1_macro_class",
        "medium_top1_macro_class",
        "few_top1_macro_class",
    )
    performance = {}
    for group in ("LC_only", "BCL"):
        metrics_path = output_root / f"{group}_seed42" / "final_test_metrics.json"
        with metrics_path.open("r", encoding="utf-8") as stream:
            performance[group] = json.load(stream)
    performance_delta = {
        metric: float(performance["BCL"][metric] - performance["LC_only"][metric])
        for metric in metric_names
    }

    gate = {
        "status": "HOLD_SINGLE_SEED_RECONNAISSANCE",
        "oracle_allowed": False,
        "seed": 42,
        "reason": (
            "A single paired seed can reveal candidate directional patterns but "
            "cannot establish repeatability. Do not run an Oracle from this result alone."
        ),
        "paired_performance_delta_BCL_minus_LC_only": performance_delta,
        "phase2_hypothesis_decision": decision_summary,
        "gradient_diagnostics": (
            "All gradient probes use the fixed LT TRAIN-only probe cache. The common-input "
            "probe uses model.eval() and the same v2 feature tensor for LC/BCL gradients. "
            "The native_three_view_train probe concatenates v1/v2/v3 in one train-mode "
            "forward and reports W_y, feature, backbone-parameter and input gradients; "
            "BatchNorm running buffers are restored after each diagnostic batch."
        ),
        "margin_data_sources": (
            "Training margins use the fixed five-per-class LT TRAIN probe. TEST margins "
            "are computed read-only from saved final logits; no TEST autograd or "
            "gradient calculation is performed."
        ),
        "head_tail_diagnostics": (
            "C[a->b] = P(prediction=b | true=a); Head-to-Tail and Tail-to-Head are "
            "separate. DID = mean(delta C | same CIFAR-100 coarse superclass) - "
            "mean(delta C | different superclass), with delta C = BCL - LC-only. "
            "DINO Top-K union is reported as a separate exogenous relation when available."
        ),
        "no_circular_overlap_definition": True,
        "oracle_menu_not_executed": {
            "prototype": "replace W-derived prototypes with balanced reference-feature prototypes",
            "relation": "suppress or down-weight same-superclass negative prototypes",
            "gradient": (
                "project BCL gradient against LC only when conflicting: "
                "g2* = g2 - min(0, g1^T g2)/(||g1||^2 + eps) * g1"
            ),
        },
        "stage_gate": [
            "Repeat the paired comparison across additional seeds before claiming a stable effect.",
            "Gradient direction/norm, margin change, directional confusion change, and Many/Medium/Few/Overall performance must tell a coherent mechanism-consistent story.",
            "Only then run the Oracle aligned with the observed mechanism; do not run all Oracles as a fishing expedition.",
            "Stop if the Oracle has no stable benefit or improves one group only by sacrificing other groups.",
        ],
        "numeric_thresholds": "No numeric go/no-go threshold was pre-registered; do not invent one after seeing results.",
    }
    with (output_root / "stage_gate.json").open("w", encoding="utf-8") as stream:
        json.dump(gate, stream, indent=2, ensure_ascii=False)

    print(f"Offline diagnostics completed on {device}.")
    print(f"Probe cache: {probe_cache_path}")
    print(f"Margin rows: {output_root / 'offline_probe_margin_metrics.csv'}")
    print(f"Common-input gradient rows: {output_root / 'offline_dual_gradient_metrics.csv'}")
    print(f"Native train-mode gradient rows: {output_root / 'offline_native_three_view_gradient_class_metrics.csv'}")
    print(f"TEST margin rows: {output_root / 'offline_test_margin_metrics.csv'}")
    print(f"Pair effects: {output_root / 'head_tail_pair_effects.csv'}")
    print(f"DID summary: {output_root / 'head_tail_did_summary.csv'}")
    print(f"H1 class gains: {output_root / 'phase2_h1_class_gain.csv'}")
    print(f"H2 adjusted associations: {output_root / 'phase2_h2_adjusted_associations.csv'}")
    print(f"H2 matched pair DID: {output_root / 'phase2_h2_pair_matched_did.csv'}")
    print(f"H3 overlap gradients: {output_root / 'phase2_h3_overlap_gradient_summary.csv'}")
    print(f"H3 margin trajectory: {output_root / 'phase2_h3_overlap_margin_trajectory.csv'}")
    print(f"Phase 2 decision brief: {output_root / 'PHASE2_DECISION_BRIEF.md'}")
    print(f"Stage gate: {output_root / 'stage_gate.json'}")
    print("Oracle execution: NOT RUN (single-seed gate is not sufficient).")


if __name__ == "__main__":
    main()
