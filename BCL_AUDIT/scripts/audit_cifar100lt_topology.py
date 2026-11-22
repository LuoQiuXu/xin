#!/usr/bin/env python
"""Leakage-safe CIFAR-100-LT topology audit.

Only the original CIFAR-100 *training* split is loaded. The test split is never
instantiated or read by this script.

Examples (PowerShell):
  python BCL_AUDIT/scripts/audit_cifar100lt_topology.py `
    --data-root D:\datasets\cifar100 --output-dir BCL_AUDIT\outputs\data_audit
  python BCL_AUDIT/scripts/audit_cifar100lt_topology.py `
    --data-root D:\datasets\cifar100 --output-dir BCL_AUDIT\outputs\data_audit `
    --compute-dino --dino-batch-size 16

The long-tail selection follows the common LDAM/PaCo CIFAR implementation:
class IDs remain in official fine-label order, counts use int() truncation,
and a fixed split seed selects the examples within each class.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _pickle_value(obj: dict, key: str):
    if key in obj:
        return obj[key]
    bkey = key.encode("utf-8")
    if bkey in obj:
        return obj[bkey]
    raise KeyError(f"Missing key {key!r} in CIFAR-100 pickle")


def _load_original_train(data_root: Path):
    """Download/load only torchvision CIFAR100(train=True), then read coarse labels."""
    try:
        import torchvision.datasets as tv_datasets
    except Exception as exc:
        raise RuntimeError("torchvision is required for the CIFAR-100 training split") from exc

    train_set = tv_datasets.CIFAR100(
        root=str(data_root),
        train=True,
        download=True,
        transform=None,
    )
    raw_train_path = data_root / "cifar-100-python" / "train"
    raw_meta_path = data_root / "cifar-100-python" / "meta"
    if not raw_train_path.is_file() or not raw_meta_path.is_file():
        raise FileNotFoundError(
            f"Expected extracted CIFAR-100 training files under {data_root / 'cifar-100-python'}"
        )

    with raw_train_path.open("rb") as f:
        raw_train = pickle.load(f, encoding="bytes")
    with raw_meta_path.open("rb") as f:
        raw_meta = pickle.load(f, encoding="bytes")

    fine = np.asarray(_pickle_value(raw_train, "fine_labels"), dtype=np.int64)
    coarse = np.asarray(_pickle_value(raw_train, "coarse_labels"), dtype=np.int64)
    images = np.asarray(train_set.data)

    if images.shape[0] != 50_000 or fine.shape[0] != 50_000 or coarse.shape[0] != 50_000:
        raise RuntimeError(
            f"Unexpected CIFAR-100 train sizes: images={images.shape}, "
            f"fine={fine.shape}, coarse={coarse.shape}"
        )
    if not np.array_equal(fine, np.asarray(train_set.targets, dtype=np.int64)):
        raise RuntimeError("Fine labels in torchvision and raw training pickle do not match")

    fine_names = list(_pickle_value(raw_meta, "fine_label_names"))
    coarse_names = list(_pickle_value(raw_meta, "coarse_label_names"))
    fine_names = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in fine_names]
    coarse_names = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in coarse_names]

    fine_to_coarse = np.full(100, -1, dtype=np.int64)
    for fine_id, coarse_id in zip(fine.tolist(), coarse.tolist()):
        previous = fine_to_coarse[fine_id]
        if previous == -1:
            fine_to_coarse[fine_id] = coarse_id
        elif previous != coarse_id:
            raise RuntimeError(f"Fine class {fine_id} maps to multiple coarse classes")

    if np.any(fine_to_coarse < 0):
        raise RuntimeError("At least one fine class has no coarse-superclass mapping")
    coarse_members = [np.where(fine_to_coarse == c)[0].tolist() for c in range(20)]
    if any(len(members) != 5 for members in coarse_members):
        raise RuntimeError(
            "CIFAR-100 taxonomy check failed: expected exactly five fine classes per superclass"
        )

    return images, fine, coarse, fine_names, coarse_names, fine_to_coarse


def _make_long_tail_indices(labels: np.ndarray, imb_factor: float, split_seed: int):
    class_count = 100
    img_max = len(labels) / class_count
    counts = np.asarray(
        [
            int(img_max * (imb_factor ** (class_id / (class_count - 1.0))))
            for class_id in range(class_count)
        ],
        dtype=np.int64,
    )
    rng = np.random.RandomState(split_seed)
    selected = []
    for class_id, requested_count in enumerate(counts.tolist()):
        class_indices = np.where(labels == class_id)[0].copy()
        rng.shuffle(class_indices)
        selected.extend(class_indices[:requested_count].tolist())
    selected = np.asarray(selected, dtype=np.int64)

    observed = np.bincount(labels[selected], minlength=100)
    if not np.array_equal(observed, counts):
        raise RuntimeError(f"Selected counts {observed.tolist()} != requested {counts.tolist()}")
    if counts[0] != 500 or counts[-1] != 5:
        raise RuntimeError(
            f"IF=100 endpoint check failed: first={counts[0]}, last={counts[-1]}"
        )
    return counts, selected


def _shot_group(n: int) -> str:
    if n > 100:
        return "many"
    if n >= 20:
        return "medium"
    return "few"


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_static_audit(
    out_dir: Path,
    counts: np.ndarray,
    selected_indices: np.ndarray,
    fine_names: list[str],
    coarse_names: list[str],
    fine_to_coarse: np.ndarray,
    split_seed: int,
    imb_factor: float,
):
    groups = [_shot_group(int(n)) for n in counts]
    class_rows = []
    for fine_id in range(100):
        coarse_id = int(fine_to_coarse[fine_id])
        class_rows.append(
            {
                "fine_id": fine_id,
                "fine_name": fine_names[fine_id],
                "coarse_id": coarse_id,
                "coarse_name": coarse_names[coarse_id],
                "train_count": int(counts[fine_id]),
                "shot_group": groups[fine_id],
            }
        )
    _write_csv(
        out_dir / "class_counts.csv",
        class_rows,
        ["fine_id", "fine_name", "coarse_id", "coarse_name", "train_count", "shot_group"],
    )

    superclass_rows = []
    for coarse_id in range(20):
        member_ids = np.where(fine_to_coarse == coarse_id)[0].tolist()
        many_ids = [i for i in member_ids if groups[i] == "many"]
        medium_ids = [i for i in member_ids if groups[i] == "medium"]
        few_ids = [i for i in member_ids if groups[i] == "few"]
        superclass_rows.append(
            {
                "coarse_id": coarse_id,
                "coarse_name": coarse_names[coarse_id],
                "fine_class_ids": ";".join(map(str, member_ids)),
                "num_many": len(many_ids),
                "num_medium": len(medium_ids),
                "num_few": len(few_ids),
                "total_train_samples": int(counts[member_ids].sum()),
                "within_superclass_many_few_pairs": len(many_ids) * len(few_ids),
            }
        )
    _write_csv(
        out_dir / "superclass_counts.csv",
        superclass_rows,
        [
            "coarse_id",
            "coarse_name",
            "fine_class_ids",
            "num_many",
            "num_medium",
            "num_few",
            "total_train_samples",
            "within_superclass_many_few_pairs",
        ],
    )

    pair_rows = []
    same_coarse_count = 0
    cross_coarse_count = 0
    for head_id in range(100):
        if groups[head_id] != "many":
            continue
        for tail_id in range(100):
            if groups[tail_id] != "few":
                continue
            same = int(fine_to_coarse[head_id]) == int(fine_to_coarse[tail_id])
            if same:
                same_coarse_count += 1
            else:
                cross_coarse_count += 1
            pair_rows.append(
                {
                    "head_fine_id": head_id,
                    "head_fine_name": fine_names[head_id],
                    "tail_fine_id": tail_id,
                    "tail_fine_name": fine_names[tail_id],
                    "head_coarse_id": int(fine_to_coarse[head_id]),
                    "tail_coarse_id": int(fine_to_coarse[tail_id]),
                    "same_superclass": same,
                    "dino_cosine_similarity": "",
                    "dino_topk_union": "",
                }
            )
    _write_csv(
        out_dir / "head_tail_pair_topology.csv",
        pair_rows,
        [
            "head_fine_id",
            "head_fine_name",
            "tail_fine_id",
            "tail_fine_name",
            "head_coarse_id",
            "tail_coarse_id",
            "same_superclass",
            "dino_cosine_similarity",
            "dino_topk_union",
        ],
    )

    np.save(out_dir / f"cifar100lt_if100_train_indices_seed{split_seed}.npy", selected_indices)
    summary = {
        "dataset": "CIFAR-100",
        "source_split": "original balanced CIFAR-100 training split only",
        "test_split_loaded": False,
        "num_original_train_images": 50_000,
        "num_fine_classes": 100,
        "num_coarse_superclasses": 20,
        "fine_classes_per_superclass": 5,
        "imbalance_type": "exponential",
        "imbalance_factor": imb_factor,
        "count_formula": "int(500 * imb_factor ** (fine_class_id / 99.0))",
        "class_order": "official fine-label IDs 0..99; not shuffled",
        "count_rounding": "int truncation",
        "split_seed": split_seed,
        "shot_group_definition": {"many": ">100", "medium": "20..100 inclusive", "few": "<20"},
        "class_counts": counts.tolist(),
        "long_tail_train_size": int(len(selected_indices)),
        "num_many_classes": int(sum(g == "many" for g in groups)),
        "num_medium_classes": int(sum(g == "medium" for g in groups)),
        "num_few_classes": int(sum(g == "few" for g in groups)),
        "within_superclass_many_few_pairs": same_coarse_count,
        "cross_superclass_many_few_pairs": cross_coarse_count,
        "total_many_few_pairs": same_coarse_count + cross_coarse_count,
        "dino_similarity_computed": False,
    }
    with (out_dir / "data_audit_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary, pair_rows


class _DinoCIFARTrainDataset:
    def __init__(self, images: np.ndarray, labels: np.ndarray):
        from torch.utils.data import Dataset
        from torchvision import transforms
        from torchvision.transforms import InterpolationMode

        self.images = images
        self.labels = labels
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = Image.fromarray(self.images[index])
        return self.transform(image), int(self.labels[index])


def _compute_dino_similarity(
    images: np.ndarray,
    labels: np.ndarray,
    out_dir: Path,
    model_name: str,
    batch_size: int,
    num_workers: int,
    topk: int,
):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_hub_dir = torch.hub.get_dir()
    start_time = time.time()

    try:
        model = torch.hub.load("facebookresearch/dinov2", model_name, trust_repo=True)
    except TypeError:
        model = torch.hub.load("facebookresearch/dinov2", model_name)
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    dataset = _DinoCIFARTrainDataset(images, labels)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    class_sums = None
    class_counts = torch.zeros(100, dtype=torch.float64, device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        for batch_images, batch_labels in loader:
            batch_images = batch_images.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            output = model.forward_features(batch_images)
            if isinstance(output, dict):
                if "x_norm_clstoken" not in output:
                    raise RuntimeError(
                        "DINOv2 forward_features did not return x_norm_clstoken; "
                        f"available keys: {list(output.keys())}"
                    )
                features = output["x_norm_clstoken"]
            else:
                features = output
            features = F.normalize(features.float(), dim=1)
            if class_sums is None:
                class_sums = torch.zeros(
                    100, features.shape[1], dtype=torch.float64, device=device
                )
            class_sums.index_add_(0, batch_labels, features.double())
            class_counts.index_add_(
                0, batch_labels, torch.ones_like(batch_labels, dtype=torch.float64)
            )

    if class_sums is None or not torch.all(class_counts == 500):
        raise RuntimeError(
            f"DINO class-count check failed: {class_counts.detach().cpu().tolist()}"
        )

    class_means = class_sums / class_counts[:, None]
    class_means = F.normalize(class_means.float(), dim=1)
    similarity = (class_means @ class_means.T).float()
    similarity.fill_diagonal_(1.0)

    class_means_np = class_means.detach().cpu().numpy()
    similarity_np = similarity.detach().cpu().numpy()
    np.save(out_dir / "DINO_CLASS_MEAN_FEATURES.npy", class_means_np)
    np.save(out_dir / "DINO_CLASS_SIMILARITY.npy", similarity_np)

    # Symmetric-union Top-K: pair (i,j) is near if j is in i's Top-K OR i is in j's Top-K.
    near_pairs = set()
    for class_id in range(100):
        row = similarity_np[class_id].copy()
        row[class_id] = -np.inf
        neighbors = np.argsort(-row)[:topk]
        for neighbor in neighbors.tolist():
            near_pairs.add(tuple(sorted((class_id, int(neighbor)))))

    dino_seconds = time.time() - start_time
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    checkpoint_candidates = sorted(
        (Path(torch_hub_dir) / "checkpoints").glob(f"{model_name}*.pth")
    )
    checkpoint_file = checkpoint_candidates[0] if checkpoint_candidates else None
    checkpoint_sha256 = None
    if checkpoint_file is not None:
        digest = hashlib.sha256()
        with checkpoint_file.open("rb") as checkpoint_stream:
            for chunk in iter(lambda: checkpoint_stream.read(1024 * 1024), b""):
                digest.update(chunk)
        checkpoint_sha256 = digest.hexdigest()

    metadata = {
        "model": model_name,
        "source": "facebookresearch/dinov2 via torch.hub",
        "torch_hub_cache": torch_hub_dir,
        "checkpoint_file": str(checkpoint_file) if checkpoint_file else None,
        "checkpoint_sha256": checkpoint_sha256,
        "source_split": "original balanced CIFAR-100 training split only",
        "test_split_loaded": False,
        "images_per_fine_class": 500,
        "input_resize": "32x32 to 224x224, Bicubic",
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "feature": "L2-normalized CLS feature; mean within class; L2-normalized class mean",
        "similarity": "cosine similarity between normalized class means",
        "topk": topk,
        "topk_pair_rule": "symmetric union: either class lists the other in its Top-K",
        "device": str(device),
        "batch_size": batch_size,
        "num_workers": num_workers,
        "elapsed_seconds": dino_seconds,
        "peak_cuda_memory_bytes": peak_memory,
    }
    with (out_dir / "dino_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    return similarity_np, near_pairs, metadata


def _update_pair_csv_with_dino(
    out_dir: Path,
    pair_rows: list[dict[str, Any]],
    similarity: np.ndarray,
    near_pairs: set[tuple[int, int]],
):
    updated = []
    for row in pair_rows:
        head_id = int(row["head_fine_id"])
        tail_id = int(row["tail_fine_id"])
        key = tuple(sorted((head_id, tail_id)))
        new_row = dict(row)
        new_row["dino_cosine_similarity"] = f"{float(similarity[head_id, tail_id]):.8f}"
        new_row["dino_topk_union"] = key in near_pairs
        updated.append(new_row)
    _write_csv(
        out_dir / "head_tail_pair_topology.csv",
        updated,
        [
            "head_fine_id",
            "head_fine_name",
            "tail_fine_id",
            "tail_fine_name",
            "head_coarse_id",
            "tail_coarse_id",
            "same_superclass",
            "dino_cosine_similarity",
            "dino_topk_union",
        ],
    )

    summary_path = out_dir / "data_audit_summary.json"
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    summary["dino_similarity_computed"] = True
    summary["dino_topk_union_many_few_pairs"] = int(
        sum(tuple(sorted((int(r["head_fine_id"]), int(r["tail_fine_id"])))) in near_pairs
            for r in updated)
    )
    summary["dino_topk_union_within_superclass_many_few_pairs"] = int(
        sum(
            tuple(sorted((int(r["head_fine_id"]), int(r["tail_fine_id"])))) in near_pairs
            for r in updated
            if bool(r["same_superclass"])
        )
    )
    summary["dino_topk_union_cross_superclass_many_few_pairs"] = int(
        sum(
            tuple(sorted((int(r["head_fine_id"]), int(r["tail_fine_id"])))) in near_pairs
            for r in updated
            if not bool(r["same_superclass"])
        )
    )
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--imb-factor", type=float, default=0.01)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--compute-dino", action="store_true")
    parser.add_argument("--dino-model", type=str, default="dinov2_vits14")
    parser.add_argument("--dino-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    if abs(args.imb_factor - 0.01) > 1e-12:
        parser.error("This Stage -1 audit is fixed to CIFAR-100-LT IF=100 (--imb-factor 0.01)")
    if not (1 <= args.topk < 100):
        parser.error("--topk must be in [1, 99]")

    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    data_root = args.data_root.expanduser().resolve()

    images, fine, coarse, fine_names, coarse_names, fine_to_coarse = _load_original_train(
        data_root
    )
    counts, selected_indices = _make_long_tail_indices(
        fine, args.imb_factor, args.split_seed
    )
    summary, pair_rows = _write_static_audit(
        out_dir,
        counts,
        selected_indices,
        fine_names,
        coarse_names,
        fine_to_coarse,
        args.split_seed,
        args.imb_factor,
    )

    print("CIFAR-100-LT static audit completed")
    print(f"  source split: original balanced training split only")
    print(f"  test split loaded: False")
    print(f"  long-tail train size: {summary['long_tail_train_size']}")
    print(f"  count endpoints: {counts[0]} -> {counts[-1]}")
    print(
        "  many/medium/few classes: "
        f"{summary['num_many_classes']}/"
        f"{summary['num_medium_classes']}/"
        f"{summary['num_few_classes']}"
    )
    print(
        "  many-few pairs, within/cross superclass: "
        f"{summary['within_superclass_many_few_pairs']}/"
        f"{summary['cross_superclass_many_few_pairs']}"
    )
    print(f"  outputs: {out_dir}")

    if args.compute_dino:
        similarity, near_pairs, metadata = _compute_dino_similarity(
            images=images,
            labels=fine,
            out_dir=out_dir,
            model_name=args.dino_model,
            batch_size=args.dino_batch_size,
            num_workers=args.num_workers,
            topk=args.topk,
        )
        _update_pair_csv_with_dino(out_dir, pair_rows, similarity, near_pairs)
        print(
            f"DINOv2 completed: {metadata['model']}, "
            f"elapsed={metadata['elapsed_seconds']:.1f}s, "
            f"peak_cuda={metadata['peak_cuda_memory_bytes']} bytes"
        )
        print(f"  similarity matrix: {out_dir / 'DINO_CLASS_SIMILARITY.npy'}")


if __name__ == "__main__":
    main()
