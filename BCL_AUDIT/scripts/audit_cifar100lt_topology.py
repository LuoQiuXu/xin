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


_DINO_OUTPUTS = (
    "DINO_CLASS_MEAN_FEATURES.npy",
    "DINO_CLASS_SIMILARITY.npy",
    "dino_metadata.json",
    "head_tail_pair_topology_dino.csv",
    "DINO_CLASS_TOPOLOGY.csv",
    "data_audit_dino_summary.json",
)


def _pickle_value(obj: dict, key: str):
    if key in obj:
        return obj[key]
    bkey = key.encode("utf-8")
    if bkey in obj:
        return obj[bkey]
    raise KeyError(f"Missing key {key!r} in CIFAR-100 pickle")


def _md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_original_train(data_root: Path):
    """Read and validate only the original CIFAR-100 TRAIN and metadata pickles."""
    raw_train_path = data_root / "cifar-100-python" / "train"
    raw_meta_path = data_root / "cifar-100-python" / "meta"
    if not raw_train_path.is_file() or not raw_meta_path.is_file():
        raise FileNotFoundError(
            "Expected locally extracted CIFAR-100 TRAIN and meta files under "
            f"{data_root / 'cifar-100-python'}. This strict audit does not "
            "download the combined archive because it also contains TEST data."
        )

    # Official CIFAR-100 checksums for TRAIN and metadata only. Deliberately
    # do not inspect the test file's existence, checksum, or contents.
    expected_train_md5 = "16019d7e3df5f24257cddd939b257f8d"
    expected_meta_md5 = "7973b15100ade9c7d40fb424638fde48"
    observed_train_md5 = _md5_file(raw_train_path)
    observed_meta_md5 = _md5_file(raw_meta_path)
    if observed_train_md5 != expected_train_md5:
        raise RuntimeError(
            f"CIFAR-100 TRAIN checksum mismatch: {observed_train_md5}"
        )
    if observed_meta_md5 != expected_meta_md5:
        raise RuntimeError(
            f"CIFAR-100 meta checksum mismatch: {observed_meta_md5}"
        )

    with raw_train_path.open("rb") as stream:
        raw_train = pickle.load(stream, encoding="bytes")
    with raw_meta_path.open("rb") as stream:
        raw_meta = pickle.load(stream, encoding="bytes")

    fine = np.asarray(_pickle_value(raw_train, "fine_labels"), dtype=np.int64)
    coarse = np.asarray(_pickle_value(raw_train, "coarse_labels"), dtype=np.int64)
    flat_images = np.asarray(_pickle_value(raw_train, "data"), dtype=np.uint8)
    if flat_images.shape != (50_000, 3 * 32 * 32):
        raise RuntimeError(f"Unexpected CIFAR-100 TRAIN image array: {flat_images.shape}")
    images = flat_images.reshape(50_000, 3, 32, 32).transpose(0, 2, 3, 1).copy()

    if images.shape != (50_000, 32, 32, 3) or fine.shape != (50_000,) or coarse.shape != (50_000,):
        raise RuntimeError(
            f"Unexpected CIFAR-100 TRAIN sizes: images={images.shape}, "
            f"fine={fine.shape}, coarse={coarse.shape}"
        )
    if not np.array_equal(np.bincount(fine, minlength=100), np.full(100, 500)):
        raise RuntimeError("CIFAR-100 TRAIN must contain exactly 500 images per fine class")
    if np.any((coarse < 0) | (coarse >= 20)):
        raise RuntimeError("CIFAR-100 coarse labels are outside [0, 19]")

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
    with path.open("x", newline="", encoding="utf-8-sig") as f:
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

    with (out_dir / f"cifar100lt_if100_train_indices_seed{split_seed}.npy").open("xb") as f:
        np.save(f, selected_indices)
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
    with (out_dir / "data_audit_summary.json").open("x", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary, pair_rows


def _load_existing_static_audit(
    out_dir: Path,
    counts: np.ndarray,
    selected_indices: np.ndarray,
    split_seed: int,
    imb_factor: float,
):
    """Validate and load existing static outputs without rewriting them."""
    indices_path = out_dir / f"cifar100lt_if100_train_indices_seed{split_seed}.npy"
    summary_path = out_dir / "data_audit_summary.json"
    pair_path = out_dir / "head_tail_pair_topology.csv"
    required = [
        out_dir / "class_counts.csv",
        out_dir / "superclass_counts.csv",
        pair_path,
        indices_path,
        summary_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Cannot reuse an incomplete static audit. Missing: " + ", ".join(missing)
        )

    with summary_path.open("r", encoding="utf-8") as stream:
        summary = json.load(stream)
    if summary.get("source_split") != "original balanced CIFAR-100 training split only":
        raise RuntimeError("Existing audit does not identify the original TRAIN split")
    if summary.get("test_split_loaded") is not False:
        raise RuntimeError("Existing audit does not certify test_split_loaded=false")
    if int(summary.get("split_seed", -1)) != int(split_seed):
        raise RuntimeError("Existing static audit split_seed does not match this run")
    if abs(float(summary.get("imbalance_factor", -1.0)) - float(imb_factor)) > 1e-12:
        raise RuntimeError("Existing static audit imbalance factor does not match")
    if summary.get("class_counts") != counts.tolist():
        raise RuntimeError("Existing static audit class-count vector does not match")
    if int(summary.get("long_tail_train_size", -1)) != int(selected_indices.size):
        raise RuntimeError("Existing static audit train size does not match")

    saved_indices = np.load(indices_path).astype(np.int64)
    if saved_indices.ndim != 1 or not np.array_equal(saved_indices, selected_indices):
        raise RuntimeError("Existing saved IF=100 indices differ from the current TRAIN split")
    if np.unique(saved_indices).size != saved_indices.size:
        raise RuntimeError("Existing saved IF=100 indices contain duplicates")

    with pair_path.open("r", newline="", encoding="utf-8-sig") as stream:
        pair_rows = list(csv.DictReader(stream))
    expected_pairs = int(summary["total_many_few_pairs"])
    if len(pair_rows) != expected_pairs:
        raise RuntimeError(
            f"Existing Head-Few pair table has {len(pair_rows)} rows; expected {expected_pairs}"
        )
    for row in pair_rows:
        row["head_fine_id"] = int(row["head_fine_id"])
        row["tail_fine_id"] = int(row["tail_fine_id"])
        row["head_coarse_id"] = int(row["head_coarse_id"])
        row["tail_coarse_id"] = int(row["tail_coarse_id"])
        row["same_superclass"] = str(row["same_superclass"]).strip().lower() == "true"
        row["dino_cosine_similarity"] = row.get("dino_cosine_similarity") or ""
        row["dino_topk_union"] = row.get("dino_topk_union") or ""
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
    return class_means_np, similarity_np, near_pairs, metadata


def _write_dino_outputs(
    out_dir: Path,
    summary: dict[str, Any],
    pair_rows: list[dict[str, Any]],
    class_means: np.ndarray,
    similarity: np.ndarray,
    near_pairs: set[tuple[int, int]],
    metadata: dict[str, Any],
    fine_names: list[str],
    coarse_names: list[str],
    fine_to_coarse: np.ndarray,
):
    if not pair_rows:
        raise RuntimeError("Cannot enrich an empty static Many–Few pair table")

    updated = []
    for row in pair_rows:
        head_id = int(row["head_fine_id"])
        tail_id = int(row["tail_fine_id"])
        key = tuple(sorted((head_id, tail_id)))
        new_row = dict(row)
        new_row["dino_cosine_similarity"] = f"{float(similarity[head_id, tail_id]):.8f}"
        new_row["dino_topk_union"] = key in near_pairs
        updated.append(new_row)
    near = sum(row["dino_topk_union"] for row in updated)
    near_within = sum(
        row["dino_topk_union"] for row in updated if row["same_superclass"]
    )

    # Audit the complete 100-node graph, not only the Many-Few slice.
    num_classes = int(similarity.shape[0])
    topk = int(metadata["topk"])
    if similarity.shape != (num_classes, num_classes):
        raise RuntimeError(f"Expected a square DINO similarity matrix, got {similarity.shape}")
    if len(fine_names) != num_classes or len(fine_to_coarse) != num_classes:
        raise RuntimeError("DINO class names / superclass mapping do not cover every fine class")
    if len(coarse_names) == 0:
        raise RuntimeError("DINO superclass names are missing")

    directed_neighbors: dict[int, list[int]] = {}
    directed_same_edges = 0
    union_degrees = np.zeros(num_classes, dtype=np.int64)
    union_same_degrees = np.zeros(num_classes, dtype=np.int64)
    union_cross_degrees = np.zeros(num_classes, dtype=np.int64)
    for class_id in range(num_classes):
        row = similarity[class_id].astype(np.float64, copy=True)
        row[class_id] = -np.inf
        neighbors = np.argsort(-row)[:topk].astype(np.int64).tolist()
        directed_neighbors[class_id] = neighbors
        directed_same_edges += sum(
            int(fine_to_coarse[class_id]) == int(fine_to_coarse[neighbor])
            for neighbor in neighbors
        )

    for left, right in near_pairs:
        left, right = int(left), int(right)
        if left == right or not (0 <= left < num_classes and 0 <= right < num_classes):
            raise RuntimeError(f"Invalid symmetric-union DINO edge: {(left, right)}")
        union_degrees[left] += 1
        union_degrees[right] += 1
        if int(fine_to_coarse[left]) == int(fine_to_coarse[right]):
            union_same_degrees[left] += 1
            union_same_degrees[right] += 1
        else:
            union_cross_degrees[left] += 1
            union_cross_degrees[right] += 1

    directed_edge_count = num_classes * topk
    directed_same_fraction = (
        float(directed_same_edges / directed_edge_count) if directed_edge_count else float("nan")
    )
    coarse_sizes = np.bincount(np.asarray(fine_to_coarse, dtype=np.int64))
    random_same_expectation = float(
        np.mean([(int(coarse_sizes[int(c)]) - 1) / max(num_classes - 1, 1) for c in fine_to_coarse])
    )
    union_edge_count = int(len(near_pairs))
    union_same_edge_count = int(sum(
        int(fine_to_coarse[left]) == int(fine_to_coarse[right])
        for left, right in near_pairs
    ))
    union_cross_edge_count = union_edge_count - union_same_edge_count
    union_same_fraction = (
        float(union_same_edge_count / union_edge_count) if union_edge_count else float("nan")
    )

    class_topology_rows = []
    for class_id in range(num_classes):
        neighbors = directed_neighbors[class_id]
        same_directed = sum(
            int(fine_to_coarse[class_id]) == int(fine_to_coarse[neighbor])
            for neighbor in neighbors
        )
        class_topology_rows.append({
            "fine_id": class_id,
            "fine_name": fine_names[class_id],
            "coarse_id": int(fine_to_coarse[class_id]),
            "coarse_name": coarse_names[int(fine_to_coarse[class_id])],
            "directed_topk_neighbor_ids": ";".join(map(str, neighbors)),
            "directed_topk_same_superclass_count": int(same_directed),
            "directed_topk_same_superclass_fraction": (
                float(same_directed / max(len(neighbors), 1))
            ),
            "symmetric_union_degree": int(union_degrees[class_id]),
            "symmetric_union_same_superclass_degree": int(union_same_degrees[class_id]),
            "symmetric_union_cross_superclass_degree": int(union_cross_degrees[class_id]),
        })

    dino_summary = {
        **summary,
        "dino_enrichment_completed": True,
        "dino_topk_union_many_few_pairs": near,
        "dino_topk_union_within_superclass_many_few_pairs": near_within,
        "dino_topk_union_cross_superclass_many_few_pairs": near - near_within,
        "dino_topk_union_edge_count_all_classes": union_edge_count,
        "dino_topk_union_same_superclass_edge_count_all_classes": union_same_edge_count,
        "dino_topk_union_cross_superclass_edge_count_all_classes": union_cross_edge_count,
        "dino_topk_union_same_superclass_edge_fraction_all_classes": union_same_fraction,
        "dino_topk_union_degree_min": int(union_degrees.min()),
        "dino_topk_union_degree_median": float(np.median(union_degrees)),
        "dino_topk_union_degree_max": int(union_degrees.max()),
        "dino_topk_union_degree_mean": float(union_degrees.mean()),
        "dino_topk_directed_edge_count": int(directed_edge_count),
        "dino_topk_directed_same_superclass_edge_count": int(directed_same_edges),
        "dino_topk_directed_same_superclass_fraction": directed_same_fraction,
        "dino_topk_directed_same_superclass_random_expectation": random_same_expectation,
        "dino_topk_directed_same_superclass_enrichment_ratio": (
            directed_same_fraction / random_same_expectation
            if random_same_expectation > 0 else float("nan")
        ),
    }

    with (out_dir / "DINO_CLASS_MEAN_FEATURES.npy").open("xb") as f:
        np.save(f, class_means)
    with (out_dir / "DINO_CLASS_SIMILARITY.npy").open("xb") as f:
        np.save(f, similarity)
    with (out_dir / "dino_metadata.json").open("x", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    _write_csv(out_dir / "head_tail_pair_topology_dino.csv", updated, list(updated[0]))
    _write_csv(
        out_dir / "DINO_CLASS_TOPOLOGY.csv",
        class_topology_rows,
        [
            "fine_id", "fine_name", "coarse_id", "coarse_name",
            "directed_topk_neighbor_ids",
            "directed_topk_same_superclass_count",
            "directed_topk_same_superclass_fraction",
            "symmetric_union_degree",
            "symmetric_union_same_superclass_degree",
            "symmetric_union_cross_superclass_degree",
        ],
    )
    # Write the completion summary after the other DINO outputs.
    with (out_dir / "data_audit_dino_summary.json").open("x", encoding="utf-8") as f:
        json.dump(dino_summary, f, indent=2, ensure_ascii=False)


def _dino_outputs_complete(out_dir: Path) -> bool:
    paths = [out_dir / name for name in _DINO_OUTPUTS]
    if not any(path.exists() or path.is_symlink() for path in paths):
        return False
    if all(path.is_file() and path.stat().st_size > 0 for path in paths):
        try:
            with (out_dir / "data_audit_dino_summary.json").open(encoding="utf-8") as f:
                summary = json.load(f)
            with (out_dir / "dino_metadata.json").open(encoding="utf-8") as f:
                metadata = json.load(f)
            class_means = np.load(out_dir / "DINO_CLASS_MEAN_FEATURES.npy")
            similarity = np.load(out_dir / "DINO_CLASS_SIMILARITY.npy")
            with (out_dir / "DINO_CLASS_TOPOLOGY.csv").open(
                "r", newline="", encoding="utf-8-sig"
            ) as f:
                class_rows = list(csv.DictReader(f))

            num_classes = int(class_means.shape[0]) if class_means.ndim == 2 else -1
            topk = int(metadata.get("topk", 0))
            class_ids = {int(row["fine_id"]) for row in class_rows}
            if (
                not isinstance(summary, dict)
                or summary.get("dino_enrichment_completed") is not True
                or metadata.get("test_split_loaded") is not False
                or "training split" not in str(metadata.get("source_split", ""))
                or num_classes < 2
                or similarity.shape != (num_classes, num_classes)
                or not np.isfinite(class_means).all()
                or not np.isfinite(similarity).all()
                or len(class_rows) != num_classes
                or class_ids != set(range(num_classes))
                or not (1 <= topk < num_classes)
            ):
                raise ValueError("DINO metadata, arrays or class-topology rows are inconsistent")

            degrees = []
            for row in class_rows:
                degree = int(row["symmetric_union_degree"])
                same_degree = int(row["symmetric_union_same_superclass_degree"])
                cross_degree = int(row["symmetric_union_cross_superclass_degree"])
                neighbors = [
                    value for value in row["directed_topk_neighbor_ids"].split(";") if value
                ]
                neighbor_ids = [int(value) for value in neighbors]
                if degree != same_degree + cross_degree or len(neighbor_ids) != topk:
                    raise ValueError("DINO class-topology degree / neighbor fields are inconsistent")
                if (
                    len(set(neighbor_ids)) != topk
                    or any(value < 0 or value >= num_classes for value in neighbor_ids)
                    or int(row["fine_id"]) in neighbor_ids
                ):
                    raise ValueError("DINO class-topology neighbor list is invalid")
                degrees.append(degree)

            if (
                int(summary.get("dino_topk_union_degree_min", -1)) != int(np.min(degrees))
                or not np.isclose(
                    float(summary.get("dino_topk_union_degree_median", float("nan"))),
                    float(np.median(degrees)),
                    rtol=0.0,
                    atol=1e-9,
                )
                or int(summary.get("dino_topk_union_degree_max", -1)) != int(np.max(degrees))
                or int(summary.get("dino_topk_union_edge_count_all_classes", -1)) * 2
                    != int(np.sum(degrees))
                or int(summary.get("dino_topk_union_same_superclass_edge_count_all_classes", -1))
                    + int(summary.get("dino_topk_union_cross_superclass_edge_count_all_classes", -1))
                    != int(summary.get("dino_topk_union_edge_count_all_classes", -2))
                or not np.isfinite(float(summary.get(
                    "dino_topk_directed_same_superclass_random_expectation", float("nan")
                )))
                or not np.isfinite(float(summary.get(
                    "dino_topk_directed_same_superclass_enrichment_ratio", float("nan")
                )))
            ):
                raise ValueError("DINO summary statistics do not match the class-topology table")
            return True
        except (KeyError, TypeError, ValueError, OSError, EOFError, json.JSONDecodeError):
            pass
    raise FileExistsError(
        "DINOv2 outputs are partial or inconsistent. Refusing to overwrite existing files in "
        f"{out_dir}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--imb-factor", type=float, default=0.01)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--compute-dino", action="store_true")
    parser.add_argument("--reuse-static-audit", action="store_true")
    parser.add_argument("--dino-model", type=str, default="dinov2_vits14")
    parser.add_argument("--dino-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    if abs(args.imb_factor - 0.01) > 1e-12:
        parser.error("This Stage -1 audit is fixed to CIFAR-100-LT IF=100 (--imb-factor 0.01)")
    if not (1 <= args.topk < 100):
        parser.error("--topk must be in [1, 99]")
    if args.reuse_static_audit and not args.compute_dino:
        parser.error("--reuse-static-audit requires --compute-dino")

    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    dino_complete = args.compute_dino and _dino_outputs_complete(out_dir)
    data_root = args.data_root.expanduser().resolve()

    images, fine, coarse, fine_names, coarse_names, fine_to_coarse = _load_original_train(
        data_root
    )
    counts, selected_indices = _make_long_tail_indices(
        fine, args.imb_factor, args.split_seed
    )

    if args.reuse_static_audit:
        summary, pair_rows = _load_existing_static_audit(
            out_dir,
            counts,
            selected_indices,
            args.split_seed,
            args.imb_factor,
        )
        print("Reusing validated static audit outputs without rewriting them")
    else:
        static_paths = [
            out_dir / "class_counts.csv",
            out_dir / "superclass_counts.csv",
            out_dir / "head_tail_pair_topology.csv",
            out_dir / f"cifar100lt_if100_train_indices_seed{args.split_seed}.npy",
            out_dir / "data_audit_summary.json",
        ]
        existing_static = [str(path) for path in static_paths if path.exists()]
        if existing_static:
            raise FileExistsError(
                "Refusing to overwrite existing static audit outputs. Choose a "
                "fresh --output-dir, or use --compute-dino --reuse-static-audit. "
                "Existing: " + ", ".join(existing_static)
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
        if dino_complete:
            print("DINOv2 outputs already complete; skipping without overwriting")
        else:
            if summary.get("dino_similarity_computed") is True:
                raise FileExistsError("The static audit already reports DINO results. Inspect its outputs.")
            class_means, similarity, near_pairs, metadata = _compute_dino_similarity(
                images=images,
                labels=fine,
                model_name=args.dino_model,
                batch_size=args.dino_batch_size,
                num_workers=args.num_workers,
                topk=args.topk,
            )
            _write_dino_outputs(
                out_dir,
                summary,
                pair_rows,
                class_means,
                similarity,
                near_pairs,
                metadata,
                fine_names,
                coarse_names,
                fine_to_coarse,
            )
            print(
                f"DINOv2 completed: {metadata['model']}, "
                f"elapsed={metadata['elapsed_seconds']:.1f}s, "
                f"peak_cuda={metadata['peak_cuda_memory_bytes']} bytes"
            )
            print(f"  similarity matrix: {out_dir / 'DINO_CLASS_SIMILARITY.npy'}")


if __name__ == "__main__":
    main()
