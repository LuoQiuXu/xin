from __future__ import annotations

import argparse
import csv
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


def load_or_create_probe_cache(
    data_root: Path,
    indices_file: Path,
    cache_path: Path,
    per_class: int,
    seed: int,
):
    """Create fixed multi-view probes from the selected LT TRAIN subset only."""
    if cache_path.is_file():
        cache = safe_torch_load(cache_path, map_location="cpu")
        expected = 100 * per_class
        if cache["labels"].numel() != expected:
            raise RuntimeError(
                f"Probe cache has {cache['labels'].numel()} rows; expected {expected}"
            )
        return cache

    if not indices_file.is_file():
        raise FileNotFoundError(f"Missing fixed CIFAR-100-LT indices: {indices_file}")

    train_set = CIFAR100TrainOnly(
        root=str(data_root),
        train=True,
        download=False,
        transform=None,
    )
    source_labels = np.asarray(train_set.targets, dtype=np.int64)
    selected_lt_indices = np.load(indices_file).astype(np.int64)
    if selected_lt_indices.ndim != 1:
        raise RuntimeError("The LT index file must be one-dimensional")
    if np.any(selected_lt_indices < 0) or np.any(selected_lt_indices >= len(source_labels)):
        raise RuntimeError("The LT index file contains out-of-range source indices")
    if np.unique(selected_lt_indices).size != selected_lt_indices.size:
        raise RuntimeError("The LT index file contains duplicate source indices")

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


def load_model_checkpoint(path: Path, device: torch.device):
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    checkpoint = safe_torch_load(path, map_location="cpu")
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
                "epoch": epoch,
                "class_id": class_id,
                "coarse_id": meta["coarse_id"],
                "shot_group": meta["shot_group"],
                "probe_count": int(mask.sum().item()),
                "raw_top1": 100.0 * float((predictions[mask] == labels[mask]).float().mean().cpu()),
                "raw_margin_mean": float(raw_margin[mask].mean().cpu()),
                "adjusted_margin_mean": float(adjusted_margin[mask].mean().cpu()),
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

    f_cos = safe_rowwise_cosine(grad_lc_f, grad_bcl_f).detach().cpu().numpy()
    f_lc_norm = grad_lc_f.norm(dim=1).detach().cpu().numpy()
    f_bcl_norm = grad_bcl_f.norm(dim=1).detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()

    rows = []
    for class_id in range(100):
        class_mask = labels_np == class_id
        meta = class_meta[class_id]
        rows.append(
            {
                "epoch": epoch,
                "class_id": class_id,
                "coarse_id": meta["coarse_id"],
                "shot_group": meta["shot_group"],
                "W_y_gradient_cosine": float(w_cos[class_id]),
                "W_y_conflict_negative_cosine": bool(w_cos[class_id] < 0),
                "W_y_LC_gradient_norm": float(w_lc_norm[class_id]),
                "W_y_BCL_gradient_norm": float(w_bcl_norm[class_id]),
                "W_y_BCL_to_LC_norm_ratio": float(
                    w_bcl_norm[class_id] / max(w_lc_norm[class_id], 1e-12)
                ),
                "same_f_feature_gradient_cosine_mean": float(np.mean(f_cos[class_mask])),
                "same_f_feature_gradient_cosine_median": float(np.median(f_cos[class_mask])),
                "same_f_feature_conflict_fraction": float(np.mean(f_cos[class_mask] < 0)),
                "same_f_LC_gradient_norm_mean": float(np.mean(f_lc_norm[class_mask])),
                "same_f_BCL_gradient_norm_mean": float(np.mean(f_bcl_norm[class_mask])),
            }
        )
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
        dino_value = pair.get("dino_topk_union", "")
        dino_near = parse_bool(dino_value) if str(dino_value).strip() else None

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
        if relation_key == "dino_topk_union" and not any(
            row[relation_key] != "" for row in effects
        ):
            continue
        for direction in ("Head_to_Tail", "Tail_to_Head"):
            directional = [row for row in effects if row["direction"] == direction]
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

    pair_rows = read_csv(data_audit_dir / "head_tail_pair_topology.csv")
    if not pair_rows:
        raise RuntimeError("head_tail_pair_topology.csv is empty; run the static topology audit first")

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
    class_counts = cifar100lt_class_counts()
    criterion_lc = LogitCompensatedCE(class_counts).to(device)
    criterion_bcl = BalancedContrastiveLoss(class_counts, temperature=0.1).to(device)

    all_margin_rows = []
    all_gradient_rows = []
    for group in ("LC_only", "BCL"):
        run_dir = output_root / f"{group}_seed42" / "checkpoints"
        for epoch, filename in CHECKPOINTS:
            model = load_model_checkpoint(run_dir / filename, device)
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
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    write_csv(
        output_root / "offline_probe_margin_metrics.csv",
        all_margin_rows,
        [
            "group", "epoch", "class_id", "coarse_id", "shot_group",
            "probe_count", "raw_top1", "raw_margin_mean", "adjusted_margin_mean",
        ],
    )
    write_csv(
        output_root / "offline_dual_gradient_metrics.csv",
        all_gradient_rows,
        [
            "epoch", "class_id", "coarse_id", "shot_group",
            "W_y_gradient_cosine", "W_y_conflict_negative_cosine",
            "W_y_LC_gradient_norm", "W_y_BCL_gradient_norm",
            "W_y_BCL_to_LC_norm_ratio",
            "same_f_feature_gradient_cosine_mean",
            "same_f_feature_gradient_cosine_median",
            "same_f_feature_conflict_fraction",
            "same_f_LC_gradient_norm_mean",
            "same_f_BCL_gradient_norm_mean",
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
            "direction", "head_class", "tail_class", "source_class", "target_class",
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
        "gradient_diagnostics": (
            "BCL checkpoint only; compare LC and BCL gradients at classifier rows W_y "
            "and the same v2 feature tensor f(x). Offline model.eval() freezes BN "
            "running statistics; train-mode cross-view BN coupling is reported separately."
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
    print(f"Gradient rows: {output_root / 'offline_dual_gradient_metrics.csv'}")
    print(f"Pair effects: {output_root / 'head_tail_pair_effects.csv'}")
    print(f"DID summary: {output_root / 'head_tail_did_summary.csv'}")
    print(f"Stage gate: {output_root / 'stage_gate.json'}")
    print("Oracle execution: NOT RUN (single-seed gate is not sufficient).")


if __name__ == "__main__":
    main()
