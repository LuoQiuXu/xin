from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from ccsam_common import (
    ROOT, STABILITY_RUN, DEFAULT_DATA_ROOT, EXPECTED_CLEAN_SHA256,
    EXPECTED_TRAIN_PICKLE_SHA256, NUM_CLASSES, TRAIN_SAMPLES, VALIDATION_SAMPLES,
    SEEDS, sha256_file, hash_arrays, write_json, read_json, install_test_guard,
    guard_pass, resolve_checkpoint, build_transforms, build_models, extract_gap,
    metrics_from_predictions, make_class_aware_plan, seed_everything,
)


def load_official_train_pickle(data_root: Path):
    train_pickle = data_root / "cifar-100-python" / "train"
    if not train_pickle.is_file():
        raise FileNotFoundError(
            f"Missing CIFAR-100 official TRAIN pickle: {train_pickle}. "
            "Place the existing CIFAR-100 data here; this script does not download data."
        )
    digest = sha256_file(train_pickle)
    if digest != EXPECTED_TRAIN_PICKLE_SHA256:
        raise RuntimeError(
            "CIFAR-100 TRAIN pickle SHA256 mismatch. "
            f"actual={digest}; expected={EXPECTED_TRAIN_PICKLE_SHA256}"
        )
    with train_pickle.open("rb") as f:
        raw = pickle.load(f, encoding="bytes")
    images = raw[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).copy()
    labels = np.asarray(raw[b"fine_labels"], dtype=np.int64)
    if images.shape != (50000, 32, 32, 3) or labels.shape != (50000,):
        raise RuntimeError(f"Unexpected CIFAR-100 TRAIN shapes: {images.shape}, {labels.shape}")
    return images, labels, train_pickle, digest


def save_array_without_overwrite(path: Path, array: np.ndarray):
    """Verify existing partial preparation exactly, or exclusively create a missing array."""
    if path.exists():
        before = sha256_file(path)
        existing = np.load(path, mmap_mode="r", allow_pickle=False)
        if existing.shape != array.shape or existing.dtype != array.dtype or not np.array_equal(existing, array):
            raise RuntimeError(f"Existing prepared array differs from recomputed sealed data: {path}")
        if sha256_file(path) != before:
            raise RuntimeError(f"Existing prepared array changed during verification: {path}")
        return {"action": "reused_unchanged", "sha256": before}
    with path.open("xb") as stream:
        np.save(stream, array, allow_pickle=False)
    return {"action": "created_missing", "sha256": sha256_file(path)}


def write_prepared_json_without_overwrite(path: Path, payload):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description="Prepare the sealed Clean Stage I basis for CC-SAM Stage II transfer.")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--complete-existing-preparation", action="store_true",
                        help="Validate partial arrays exactly and create only missing outputs; never overwrite.")
    args = parser.parse_args()

    if tuple(args.seeds) != tuple(SEEDS):
        raise RuntimeError(f"This frozen protocol requires Stage II seeds {SEEDS}; got {tuple(args.seeds)}")
    prepared = ROOT / "prepared"
    if prepared.exists() and not args.complete_existing_preparation:
        raise FileExistsError(
            f"Refusing to overwrite existing preparation directory: {prepared}. "
            "Inspect the existing manifest before deciding how to proceed."
        )
    if args.complete_existing_preparation:
        if not prepared.is_dir():
            raise FileNotFoundError("No partial prepared directory exists to complete")
        allowed = {"train_images.npy", "train_labels.npy", "validation_images.npy",
                   "validation_labels.npy", "validation_gap_features.npy"}
        allowed.update(f"sampling_plan_seed{seed}.npy" for seed in SEEDS)
        unexpected = [p.name for p in prepared.iterdir() if not p.is_file() or p.name not in allowed]
        if unexpected:
            raise RuntimeError(f"Prepared directory contains completed or unexpected artifacts; refusing modification: {unexpected}")

    data_root = Path(args.data_root).expanduser().resolve()
    guard = install_test_guard(data_root)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment; CPU fallback is intentionally disabled.")
    if torch.cuda.device_count() < 1:
        raise RuntimeError("No CUDA device is visible.")

    checkpoint_path = resolve_checkpoint(args.checkpoint)
    if sha256_file(checkpoint_path) != EXPECTED_CLEAN_SHA256:
        raise RuntimeError("Clean Stage I checkpoint integrity check failed.")

    clean_protocol_path = STABILITY_RUN / "clean_stage1_protocol.json"
    split_path = STABILITY_RUN / "split_indices.npz"
    if not clean_protocol_path.is_file() or not split_path.is_file():
        raise FileNotFoundError("The sealed Clean Stage I protocol or its frozen TRAIN split is missing.")
    clean_protocol = read_json(clean_protocol_path)
    if not clean_protocol.get("passed") or clean_protocol.get("TEST_loaded") or clean_protocol.get("TEST_selected"):
        raise RuntimeError("The referenced Clean Stage I protocol is not certified as TRAIN-only.")
    if clean_protocol.get("best_epoch") != 184:
        raise RuntimeError("Unexpected Clean Stage I best epoch.")
    if clean_protocol.get("best_checkpoint_sha256") != EXPECTED_CLEAN_SHA256:
        raise RuntimeError("The Clean Stage I protocol points to a different checkpoint.")

    split = np.load(split_path, allow_pickle=False)
    required_keys = {"train_raw_indices", "validation_raw_indices"}
    if not required_keys.issubset(set(split.files)):
        raise RuntimeError(f"Unexpected split file keys: {split.files}")
    train_ids = np.asarray(split["train_raw_indices"], dtype=np.int64)
    validation_ids = np.asarray(split["validation_raw_indices"], dtype=np.int64)
    if len(train_ids) != TRAIN_SAMPLES or len(validation_ids) != VALIDATION_SAMPLES:
        raise RuntimeError(f"Unexpected split sizes: train={len(train_ids)}, val={len(validation_ids)}")
    if np.intersect1d(train_ids, validation_ids).size != 0:
        raise RuntimeError("TRAIN and validation raw IDs overlap.")

    raw_images, raw_labels, train_pickle, train_pickle_sha = load_official_train_pickle(data_root)
    train_images = raw_images[train_ids]
    train_labels = raw_labels[train_ids]
    validation_images = raw_images[validation_ids]
    validation_labels = raw_labels[validation_ids]

    train_counts = np.bincount(train_labels, minlength=NUM_CLASSES)
    validation_counts = np.bincount(validation_labels, minlength=NUM_CLASSES)
    expected_counts = np.asarray(clean_protocol["LT_class_counts"], dtype=np.int64)
    if not np.array_equal(train_counts, expected_counts):
        raise RuntimeError("The reconstructed LT training class counts differ from the sealed Clean Stage I protocol.")
    lt_pixel_hash = hashlib.sha256(train_images.tobytes(order="C")).hexdigest()
    if lt_pixel_hash != clean_protocol["LT_pixel_hash"]:
        raise RuntimeError(
            "The reconstructed LT training pixels differ from the sealed Clean Stage I data. "
            f"actual={lt_pixel_hash}; expected={clean_protocol['LT_pixel_hash']}"
        )
    if validation_counts.tolist() != [0] + [20] * 99:
        raise RuntimeError("The frozen validation partition must contain 20 examples for classes 1..99 and none for class 0.")

    train_pixel_hashes = {hash_arrays(image) for image in train_images}
    overlap = sum(hash_arrays(image) in train_pixel_hashes for image in validation_images)
    if overlap != 0:
        raise RuntimeError(f"TRAIN/validation exact-pixel overlap detected: {overlap}")

    train_transform, eval_transform, transform_source_sha = build_transforms()
    seed_everything(42, deterministic=True)
    device = torch.device("cuda:0")
    backbone, classifier, checkpoint = build_models(checkpoint_path, device)

    # extract_gap delegates to the same Clean-compatible pooling used by Smoke/Stage II.
    validation_gap = extract_gap(
        backbone, classifier, validation_images, eval_transform, device, batch_size=128
    )
    validation_logits = classifier.module.fc(
        torch.from_numpy(validation_gap).to(device)
    ).detach().cpu().numpy()
    validation_predictions = validation_logits.argmax(axis=1)
    baseline_metrics = metrics_from_predictions(validation_predictions, validation_labels, train_counts)
    expected_overall = float(clean_protocol["validation"]["Overall"])
    if abs(baseline_metrics["Overall"] - expected_overall) > 1e-6:
        raise RuntimeError(
            "Clean checkpoint validation reproduction failed: "
            f"recomputed={baseline_metrics['Overall']:.10f}, sealed={expected_overall:.10f}"
        )

    print(
        f"Clean TRAIN-holdout validation reproduced: actual={baseline_metrics['Overall']:.10f}% "
        f"sealed={expected_overall:.10f}%; GAP shape={list(validation_gap.shape)}",
        flush=True,
    )
    if not prepared.exists():
        prepared.mkdir(parents=True, exist_ok=False)
    array_actions = {}
    for name, array in (
        ("train_images.npy", train_images), ("train_labels.npy", train_labels),
        ("validation_images.npy", validation_images), ("validation_labels.npy", validation_labels),
        ("validation_gap_features.npy", validation_gap.astype(np.float32, copy=False)),
    ):
        array_actions[name] = save_array_without_overwrite(prepared / name, array)

    sampling_plan_hashes = {}
    for seed in SEEDS:
        plan, statistics = make_class_aware_plan(train_labels, seed, return_statistics=True)
        plan_path = prepared / f"sampling_plan_seed{seed}.npy"
        plan_action = save_array_without_overwrite(plan_path, plan)
        sampling_plan_hashes[str(seed)] = {
            "path": plan_path.name,
            "sha256": sha256_file(plan_path),
            "shape": list(plan.shape),
            "dtype": str(plan.dtype),
            "file_action": plan_action["action"],
            "all_epochs_match_official_continuous_sampler": True,
            "official_sequence_validation": statistics,
        }

    manifest = {
        "prepared_at": __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds"),
        "checkpoint_path": str(checkpoint_path),
        "clean_stage1_sha256": sha256_file(checkpoint_path),
        "clean_stage1_epoch": int(checkpoint["epoch"]),
        "clean_stage1_seed": int(checkpoint["seed"]),
        "clean_stage1_selection_partition": checkpoint["selection_partition"],
        "split_path": str(split_path),
        "split_sha256": sha256_file(split_path),
        "official_train_pickle": str(train_pickle),
        "official_train_pickle_sha256": train_pickle_sha,
        "transform_source": str(STABILITY_RUN / "runtime" / "datasets" / "cifar100.py"),
        "transform_source_sha256": transform_source_sha,
        "train_samples": int(len(train_labels)),
        "validation_samples": int(len(validation_labels)),
        "train_class_counts": train_counts.tolist(),
        "LT_pixel_hash": lt_pixel_hash,
        "validation_class_counts": validation_counts.tolist(),
        "train_validation_raw_id_overlap": 0,
        "train_validation_exact_pixel_overlap": 0,
        "validation_baseline_metrics": baseline_metrics,
        "validation_baseline_overall_matches_sealed_clean_protocol": True,
        "validation_feature_shape": list(validation_gap.shape),
        "seeds": list(SEEDS),
        "sampling_plans": sampling_plan_hashes,
        "official_TEST_used": False,
        "official_TEST_images_or_labels_loaded": 0,
        "CIFAR100_train_false_calls": 0,
        "checkpoint_retrained": False,
        "class_statistics_recomputed_for_method": False,
        "preparation_mode": "complete_existing_without_overwrite" if args.complete_existing_preparation else "fresh",
        "prepared_array_actions": array_actions,
    }
    write_prepared_json_without_overwrite(prepared / "prepared_manifest.json", manifest)
    guard_result = guard_pass()
    write_prepared_json_without_overwrite(prepared / "test_guard_prepare.json", guard_result)
    print("CC-SAM PREPARATION PASS")
    print(f"Clean Stage I SHA256: {manifest['clean_stage1_sha256']}")
    print(f"TRAIN samples: {manifest['train_samples']}; validation samples: {manifest['validation_samples']}")
    print(f"TRAIN-only validation Overall reproduced: {baseline_metrics['Overall']:.6f}%")
    print("Official CIFAR-100 TEST was not opened, decoded, evaluated, or used for model selection.")


if __name__ == "__main__":
    main()
