"""Cache deterministic official TRAIN features from each fixed Stage-I checkpoint."""
from __future__ import annotations

import numpy as np
import torch

from cross_dataset_models import FEATURE_DIM, assert_encoder_frozen, freeze_encoder
from experiment_common import array_sha256, make_loader, record_run, require_complete, run_log, run_path
from prepare_experiment import train_dataset
from train_stage1 import CACHE_TRANSFORM, CHECKPOINT, eval_transform, load_stage1

FEATURES = "train_features.npy"
LABELS = "train_labels.npy"


def write_cache(directory, loader, labels, encoder, device):
    features = np.lib.format.open_memmap(directory / FEATURES, mode="w+", dtype=np.float32,
                                         shape=(len(labels), FEATURE_DIM))
    targets = np.lib.format.open_memmap(directory / LABELS, mode="w+", dtype=np.int64, shape=(len(labels),))
    start = 0
    with torch.inference_mode():
        for images, target in loader:
            stop = start + len(target)
            if not np.array_equal(target.numpy(), labels[start:stop]):
                raise RuntimeError(f"Feature extraction changed official row order at row {start}")
            vector = encoder(images.to(device, non_blocking=True))
            if vector.shape != (len(target), FEATURE_DIM) or vector.dtype != torch.float32 or not bool(torch.isfinite(vector).all()):
                raise RuntimeError(f"Malformed or nonfinite cached feature at row {start}")
            features[start:stop] = vector.cpu().numpy()
            targets[start:stop] = target.numpy()
            start = stop
            if start % 16000 == 0:
                run_log(directory, f"Extracted {start:,}/{len(labels):,} TRAIN vectors")
    if start != len(labels) or not np.array_equal(targets, labels):
        raise RuntimeError("Feature extraction did not preserve the full official label vector")
    features.flush()
    targets.flush()
    return {"feature_shape": list(features.shape), "feature_dtype": str(features.dtype),
            "label_shape": list(targets.shape), "label_dtype": str(targets.dtype),
            "finite": True, "labels_sha256": array_sha256(targets)}


def extract_all(exp, prepared: dict, device: torch.device) -> None:
    for name in exp.datasets:
        seal = prepared["datasets"][name]
        with train_dataset(exp, name, eval_transform(), seal) as (dataset, labels):
            for seed in exp.seeds:
                directory = run_path("features", name, seed)
                with record_run(directory, exp.config("extract", dataset=name, seed=seed,
                                                       transform=CACHE_TRANSFORM), device=device) as status:
                    encoder, _, stage1 = load_stage1(exp, name, seed, seal)
                    encoder.to(device)
                    snapshot = freeze_encoder(encoder)
                    loader = make_loader(dataset, exp.protocol["stage1"]["batch_size"], exp.workers, seed)
                    status.update(write_cache(directory, loader, labels, encoder, device))
                    del loader
                    assert_encoder_frozen(encoder, snapshot)
                    status.update(dataset=name, seed=seed, data_fingerprint=seal["data_fingerprint"],
                                  stage1_checkpoint_sha256=stage1["artifacts"][CHECKPOINT],
                                  encoder_state_sha256=stage1["encoder_state_sha256"],
                                  transform=CACHE_TRANSFORM, frozen_encoder_and_bn_unchanged=True)
                    del encoder, snapshot


def open_cache(exp, name, seed, seal, stage1):
    directory = run_path("features", name, seed)
    status = require_complete(directory, phase="extract", expected={
        "dataset": name, "seed": seed, "data_fingerprint": seal["data_fingerprint"],
        "stage1_checkpoint_sha256": stage1["artifacts"][CHECKPOINT], "transform": CACHE_TRANSFORM,
        "official_test_access": False, "frozen_encoder_and_bn_unchanged": True,
        "feature_shape": [seal["train_rows"], FEATURE_DIM], "label_shape": [seal["train_rows"]],
        "feature_dtype": "float32", "label_dtype": "int64", "finite": True, "labels_sha256": seal["labels_sha256"],
    })
    features = np.load(directory / FEATURES, mmap_mode="r", allow_pickle=False)
    labels = np.load(directory / LABELS, mmap_mode="r", allow_pickle=False)
    if (list(features.shape) != status["feature_shape"] or features.dtype != np.float32
        or list(labels.shape) != status["label_shape"] or labels.dtype != np.int64
        or array_sha256(labels) != seal["labels_sha256"]):
        raise RuntimeError(f"Feature cache shape, dtype or ordered labels differ in {directory}")
    return features, labels, status
