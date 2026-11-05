"""Two-stage CIFAR-100-LT feature-perturbation experiment.

Stage I is the locked Clean ResNet-32 checkpoint (epoch 184). Stage II trains
only a 64->100 linear classifier on frozen GAP features. The official TEST set
is deliberately not opened in this exploratory round.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image

ROOT = Path(__file__).resolve().parent
CLEAN_RUN = ROOT.parent / "clean_vs_runs" / "clean_s1_seed42_vs_seed123_20261013"
sys.path.insert(0, str(CLEAN_RUN))

import common as clean
from build_clean_statistics import frozen_models
from feature_geometry import (
    estimate_class_subspaces,
    make_boundary_targets,
    sample_isotropic_directions,
    sample_subspace_directions,
    scale_directions,
)

SEEDS = (42, 43, 44)
ARMS = ("crt", "h2t_pooled", "gaussian", "subspace_random", "subspace_boundary")
ARM_LABELS = {
    "crt": "cRT-style classifier-only baseline",
    "h2t_pooled": "H2T channel replacement after GAP (pooled-equivalent)",
    "gaussian": "Isotropic Gaussian feature perturbation",
    "subspace_random": "Class-conditional augmentation-subspace perturbation",
    "subspace_boundary": "Subspace-projected boundary-directed perturbation",
}
NUM_EPOCHS = 10
BATCH_SIZE = 128
NUM_CLASSES = 100
FEATURE_DIM = 64
NUM_CHANNELS_REPLACED = 16
SUBSPACE_RANK = 8
SHRINKAGE_TAU = 20.0
BOUNDARY_ALPHA = 0.5
MAX_NORM_RATIO = 0.25
MIN_ALIGNMENT = 0.05
FEATURE_SEED_OFFSET = 300009
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_arrays(*arrays):
    digest = hashlib.sha256()
    for value in arrays:
        array = np.asarray(value)
        digest.update(str(array.shape).encode("utf-8"))
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def hash_state_dict(state):
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def install_test_guard():
    """Block official CIFAR-100 TEST file opens and train=False dataset calls."""
    state = {
        "installed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "official_test_file_opens": 0,
        "CIFAR100_trainFalse_calls": 0,
        "blocked_attempts": [],
    }
    forbidden = os.path.normcase(
        os.path.abspath(clean.DATA / "cifar-100-python" / "test")
    )

    def deny(kind, path):
        state["blocked_attempts"].append({"kind": kind, "path": str(path)})
        write_json(ROOT / "test_guard.json", state)
        raise RuntimeError(f"OFFICIAL TEST ACCESS BLOCKED: {kind}: {path}")

    def audit(event, args):
        if event != "open" or not args:
            return
        candidate = args[0]
        if not isinstance(candidate, (str, bytes, os.PathLike)):
            return
        resolved = os.path.normcase(os.path.abspath(os.fsdecode(candidate)))
        if resolved == forbidden:
            state["official_test_file_opens"] += 1
            deny("official test file open", resolved)

    sys.addaudithook(audit)
    original_init = torchvision.datasets.CIFAR100.__init__

    def guarded_init(self, root, train=True, *args, **kwargs):
        if not train:
            state["CIFAR100_trainFalse_calls"] += 1
            deny("CIFAR100(train=False)", root)
        return original_init(self, root, train, *args, **kwargs)

    torchvision.datasets.CIFAR100.__init__ = guarded_init
    write_json(ROOT / "test_guard.json", state)
    return state


def guard_pass(state):
    assert state["official_test_file_opens"] == 0
    assert state["CIFAR100_trainFalse_calls"] == 0
    assert not state["blocked_attempts"]
    write_json(ROOT / "test_guard.json", state)


def verify_clean_inputs():
    assert CLEAN_RUN.is_dir(), f"Missing locked Clean Stage-I experiment: {CLEAN_RUN}"
    assert clean.ROOT.resolve() == CLEAN_RUN.resolve()
    clean.check_sources()

    protocol = read_json(CLEAN_RUN / "clean_stage1_protocol.json")
    preflight = read_json(CLEAN_RUN / "preflight.json")
    assert protocol["passed"] is True
    assert protocol["TEST_loaded"] is False and protocol["TEST_selected"] is False
    assert protocol["best_epoch"] == 184
    assert protocol["best_checkpoint_sha256"] == (
        "7ced9d07f71335357c08623573f038acd0dfdcdeadb55d95d57b332a14f83a88"
    )
    assert preflight["LT_samples"] == 10847
    assert preflight["LT_classes"] == 100
    assert preflight["validation_samples"] == 1980
    assert preflight["missing_classes"] == [0]

    labels = np.load(CLEAN_RUN / "train_labels.npy", allow_pickle=False)
    counts = np.bincount(labels, minlength=NUM_CLASSES)
    assert len(labels) == 10847
    assert counts.tolist() == preflight["LT_class_counts"]
    assert counts.min() == 5 and counts.max() == 500
    assert np.array_equal(
        np.load(CLEAN_RUN / "validation_labels.npy", allow_pickle=False),
        np.asarray(
            np.load(CLEAN_RUN / "validation_labels.npy", allow_pickle=False),
            dtype=np.int64,
        ),
    )
    return protocol, preflight, labels, counts


def verify_pool_equivalence():
    """H2T channel replacement commutes exactly with global average pooling."""
    generator = torch.Generator(device="cpu").manual_seed(20261105)
    receiver_map = torch.randn((5, 64, 8, 8), generator=generator)
    donor_map = torch.randn((5, 64, 8, 8), generator=generator)
    selected = torch.arange(16, dtype=torch.long)
    mask = torch.zeros(64, dtype=torch.bool)
    mask[selected] = True

    fused_map = torch.where(mask[None, :, None, None], donor_map, receiver_map)
    pooled_after_fusion = F.avg_pool2d(fused_map, 8).flatten(1)
    receiver_gap = F.avg_pool2d(receiver_map, 8).flatten(1)
    donor_gap = F.avg_pool2d(donor_map, 8).flatten(1)
    fused_gap = torch.where(mask[None, :], donor_gap, receiver_gap)
    if not torch.allclose(pooled_after_fusion, fused_gap, atol=1e-7, rtol=1e-7):
        raise AssertionError("H2T map-level fusion is not GAP-equivalent")
    return True


def extract_seed_feature_stream(
    seed,
    train_images,
    train_labels,
    train_transform,
    backbone,
    classifier_weight,
    classifier_bias,
    class_counts,
):
    """Create one paired, frozen feature stream shared by all five arms."""
    count = len(train_labels)
    batches = math.ceil(count / BATCH_SIZE)
    assert batches == 85
    class_indices = [np.flatnonzero(train_labels == c) for c in range(NUM_CLASSES)]
    rng = np.random.default_rng(seed)
    augmentation_generator = torch.Generator(device="cpu").manual_seed(
        seed + FEATURE_SEED_OFFSET
    )

    cache_dir = ROOT / "feature_cache" / f"seed{seed}"
    if cache_dir.exists():
        raise FileExistsError(f"Refusing to overwrite feature cache: {cache_dir}")
    cache_dir.mkdir(parents=True)

    receiver_features = np.lib.format.open_memmap(
        cache_dir / "receiver_features.npy",
        mode="w+",
        dtype=np.float32,
        shape=(NUM_EPOCHS, count, FEATURE_DIM),
    )
    donor_features = np.lib.format.open_memmap(
        cache_dir / "donor_features.npy",
        mode="w+",
        dtype=np.float32,
        shape=(NUM_EPOCHS, count, FEATURE_DIM),
    )
    receiver_indices = np.empty((NUM_EPOCHS, count), dtype=np.int32)
    donor_indices = np.empty((NUM_EPOCHS, count), dtype=np.int32)
    receiver_labels = np.empty((NUM_EPOCHS, count), dtype=np.int64)
    h2t_masks = np.empty(
        (NUM_EPOCHS, batches, NUM_CHANNELS_REPLACED), dtype=np.uint8
    )
    epoch_fingerprints = []

    backbone.eval()
    assert all(not parameter.requires_grad for parameter in backbone.parameters())
    with torch.no_grad():
        for epoch in range(NUM_EPOCHS):
            donor_idx = rng.permutation(count).astype(np.int32)
            recipient_classes = rng.integers(0, NUM_CLASSES, size=count)
            recipient_idx = np.fromiter(
                (rng.choice(class_indices[int(class_id)]) for class_id in recipient_classes),
                dtype=np.int32,
                count=count,
            )
            donor_indices[epoch] = donor_idx
            receiver_indices[epoch] = recipient_idx
            receiver_labels[epoch] = train_labels[recipient_idx]

            for batch, start in enumerate(range(0, count, BATCH_SIZE)):
                stop = min(start + BATCH_SIZE, count)
                donor_batch = donor_idx[start:stop]
                receiver_batch = recipient_idx[start:stop]
                selected = np.sort(
                    rng.choice(FEATURE_DIM, NUM_CHANNELS_REPLACED, replace=False)
                )
                h2t_masks[epoch, batch] = selected.astype(np.uint8)

                donor_images = []
                receiver_images = []
                # Preserve the original H2T Stage-II augmentation order:
                # instance-distribution donor first, balanced receiver second.
                with torch.random.fork_rng(devices=[]):
                    torch.set_rng_state(augmentation_generator.get_state())
                    for donor_id, receiver_id in zip(donor_batch, receiver_batch):
                        donor_images.append(
                            train_transform(Image.fromarray(train_images[int(donor_id)]))
                        )
                        receiver_images.append(
                            train_transform(Image.fromarray(train_images[int(receiver_id)]))
                        )
                    augmentation_generator.set_state(torch.get_rng_state())

                donor_tensor = torch.stack(donor_images).to(DEVICE, non_blocking=True)
                receiver_tensor = torch.stack(receiver_images).to(DEVICE, non_blocking=True)
                donor_map = backbone(donor_tensor)
                receiver_map = backbone(receiver_tensor)
                if donor_map.shape[1:] != (64, 8, 8):
                    raise AssertionError(f"Unexpected H2T feature map: {donor_map.shape}")
                donor_gap = F.avg_pool2d(donor_map, 8).flatten(1)
                receiver_gap = F.avg_pool2d(receiver_map, 8).flatten(1)
                if not torch.isfinite(donor_gap).all() or not torch.isfinite(receiver_gap).all():
                    raise FloatingPointError("Nonfinite frozen Stage-I feature")

                # Donor bank is indexed by original training-example ID. Every
                # donor ID appears exactly once per epoch because donor_idx is a permutation.
                donor_features[epoch, donor_batch] = donor_gap.cpu().numpy()
                receiver_features[epoch, start:stop] = receiver_gap.cpu().numpy()

            assert np.array_equal(np.sort(donor_idx), np.arange(count, dtype=np.int32))
            assert np.bincount(train_labels[donor_idx], minlength=NUM_CLASSES).tolist() == (
                class_counts.tolist()
            )
            recipient_hist = np.bincount(receiver_labels[epoch], minlength=NUM_CLASSES)
            epoch_fingerprints.append(
                {
                    "epoch": epoch + 1,
                    "receiver_indices_sha256": hash_arrays(receiver_idx := receiver_indices[epoch]),
                    "donor_indices_sha256": hash_arrays(donor_idx),
                    "receiver_labels_sha256": hash_arrays(receiver_labels[epoch]),
                    "receiver_class_histogram": recipient_hist.tolist(),
                    "donor_class_histogram": class_counts.tolist(),
                    "h2t_mask_sha256": hash_arrays(h2t_masks[epoch]),
                    "batches": batches,
                    "last_batch_size": count - (batches - 1) * BATCH_SIZE,
                }
            )
            print(
                f"feature-cache seed={seed} epoch={epoch + 1}/{NUM_EPOCHS} "
                f"batches={batches} last_batch={count - (batches - 1) * BATCH_SIZE}",
                flush=True,
            )

    receiver_features.flush()
    donor_features.flush()

    # Each training image has one donor-view feature in epoch 1 and another in
    # epoch 2. Their paired difference estimates augmentation-induced variation.
    basis, eigenvalues, subspace_diagnostics = estimate_class_subspaces(
        np.asarray(donor_features[0]),
        np.asarray(donor_features[1]),
        train_labels,
        num_classes=NUM_CLASSES,
        rank=SUBSPACE_RANK,
        shrinkage_tau=SHRINKAGE_TAU,
    )
    np.save(cache_dir / "subspace_basis.npy", basis)
    np.save(cache_dir / "subspace_eigenvalues.npy", eigenvalues)

    boundary_delta = np.lib.format.open_memmap(
        cache_dir / "boundary_delta.npy",
        mode="w+",
        dtype=np.float32,
        shape=(NUM_EPOCHS, count, FEATURE_DIM),
    )
    boundary_radius = np.lib.format.open_memmap(
        cache_dir / "boundary_radius.npy",
        mode="w+",
        dtype=np.float32,
        shape=(NUM_EPOCHS, count),
    )
    boundary_margin = np.lib.format.open_memmap(
        cache_dir / "boundary_margin.npy",
        mode="w+",
        dtype=np.float32,
        shape=(NUM_EPOCHS, count),
    )
    boundary_alignment = np.lib.format.open_memmap(
        cache_dir / "boundary_alignment.npy",
        mode="w+",
        dtype=np.float32,
        shape=(NUM_EPOCHS, count),
    )
    boundary_competitor = np.lib.format.open_memmap(
        cache_dir / "boundary_competitor.npy",
        mode="w+",
        dtype=np.int16,
        shape=(NUM_EPOCHS, count),
    )
    boundary_eligible = np.lib.format.open_memmap(
        cache_dir / "boundary_eligible.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(NUM_EPOCHS, count),
    )
    boundary_clipped = np.lib.format.open_memmap(
        cache_dir / "boundary_clipped.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(NUM_EPOCHS, count),
    )
    boundary_epoch_diagnostics = []

    for epoch in range(NUM_EPOCHS):
        result = make_boundary_targets(
            np.asarray(receiver_features[epoch]),
            receiver_labels[epoch],
            classifier_weight,
            classifier_bias,
            basis,
            class_counts,
            alpha=BOUNDARY_ALPHA,
            max_norm_ratio=MAX_NORM_RATIO,
            min_alignment=MIN_ALIGNMENT,
        )
        boundary_delta[epoch] = result["delta"]
        boundary_radius[epoch] = result["radius"]
        boundary_margin[epoch] = result["margin"]
        boundary_alignment[epoch] = result["alignment"]
        boundary_competitor[epoch] = result["competitor"].astype(np.int16)
        boundary_eligible[epoch] = result["eligible"]
        boundary_clipped[epoch] = result["clipped"]

        tail = result["tail"].astype(bool)
        eligible = result["eligible"].astype(bool)
        boundary_epoch_diagnostics.append(
            {
                "epoch": epoch + 1,
                "tail_draws": int(tail.sum()),
                "eligible_tail_draws": int(eligible.sum()),
                "eligible_fraction_of_tail": float(eligible[tail].mean()) if tail.any() else 0.0,
                "clipped_fraction_of_eligible": (
                    float(result["clipped"][eligible].mean()) if eligible.any() else 0.0
                ),
                "mean_tail_budget": float(result["radius"][tail].mean()) if tail.any() else 0.0,
                "mean_eligible_alignment": (
                    float(result["alignment"][eligible].mean()) if eligible.any() else 0.0
                ),
                "max_selected_boundary_crossing_violation": float(
                    np.maximum(-result["margin_after"][eligible], 0.0).max()
                ) if eligible.any() else 0.0,
            }
        )

    for value in (
        boundary_delta,
        boundary_radius,
        boundary_margin,
        boundary_alignment,
        boundary_competitor,
        boundary_eligible,
        boundary_clipped,
    ):
        value.flush()

    arrays = [
        "receiver_features.npy",
        "donor_features.npy",
        "receiver_indices.npy",
        "donor_indices.npy",
        "receiver_labels.npy",
        "h2t_masks.npy",
        "subspace_basis.npy",
        "subspace_eigenvalues.npy",
        "boundary_delta.npy",
        "boundary_radius.npy",
        "boundary_margin.npy",
        "boundary_alignment.npy",
        "boundary_competitor.npy",
        "boundary_eligible.npy",
        "boundary_clipped.npy",
    ]
    np.save(cache_dir / "receiver_indices.npy", receiver_indices)
    np.save(cache_dir / "donor_indices.npy", donor_indices)
    np.save(cache_dir / "receiver_labels.npy", receiver_labels)
    np.save(cache_dir / "h2t_masks.npy", h2t_masks)
    np.save(cache_dir / "class_counts.npy", class_counts.astype(np.int64))
    np.save(cache_dir / "initial_weight.npy", classifier_weight.astype(np.float32))
    np.save(cache_dir / "initial_bias.npy", classifier_bias.astype(np.float32))

    # The initial head is also saved once at experiment root for exact Stage-II initialization.
    file_hashes = {
        name: sha256_file(cache_dir / name)
        for name in arrays + ["class_counts.npy", "initial_weight.npy", "initial_bias.npy"]
    }
    manifest = {
        "seed": int(seed),
        "stage1_checkpoint_sha256": read_json(CLEAN_RUN / "clean_stage1_protocol.json")[
            "best_checkpoint_sha256"
        ],
        "feature_cache_files": file_hashes,
        "feature_cache_schedule_sha256": hash_arrays(
            receiver_indices, donor_indices, receiver_labels, h2t_masks
        ),
        "subspace_diagnostics": subspace_diagnostics,
        "boundary_diagnostics_by_epoch": boundary_epoch_diagnostics,
        "augmentation_rng_initial_seed": int(seed + FEATURE_SEED_OFFSET),
        "feature_shape_receiver": [NUM_EPOCHS, count, FEATURE_DIM],
        "feature_shape_donor": [NUM_EPOCHS, count, FEATURE_DIM],
        "donor_bank_alignment": "donor_features[epoch, original_train_index]",
        "receiver_bank_alignment": "receiver_features[epoch, balanced_draw_position]",
        "subspace_source": "paired instance-distribution donor views from epochs 1 and 2; TRAIN only",
        "boundary_source": "locked Stage-I classifier and current balanced TRAIN receiver feature",
    }
    write_json(cache_dir / "cache_manifest.json", manifest)
    return manifest


def prepare():
    if (ROOT / "prepare_manifest.json").exists():
        raise FileExistsError("prepare_manifest.json already exists; refusing to overwrite")
    if (ROOT / "feature_cache").exists():
        raise FileExistsError("feature_cache already exists; use a new experiment branch for reruns")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("This protocol requires the configured single CUDA GPU")

    protocol, preflight, train_labels, class_counts = verify_clean_inputs()
    cfg, transforms, backbone, classifier, criterion_class, methods, stage1_path, checkpoint = (
        frozen_models()
    )
    assert clean.sha(stage1_path) == protocol["best_checkpoint_sha256"]
    assert int(checkpoint["epoch"]) == 184
    assert not checkpoint.get("TEST_used", True)
    backbone.eval().requires_grad_(False)
    assert all(parameter.grad is None for parameter in backbone.parameters())

    classifier.eval()
    initial_fc = {
        key: value.detach().cpu().clone()
        for key, value in classifier.module.fc.state_dict().items()
    }
    initial_weight = initial_fc["weight"].numpy().astype(np.float32)
    initial_bias = initial_fc["bias"].numpy().astype(np.float32)
    initial_classifier_sha = hash_state_dict(initial_fc)
    torch.save(initial_fc, ROOT / "initial_classifier.pth")

    train_images = np.load(CLEAN_RUN / "train_images.npy", mmap_mode="r", allow_pickle=False)
    assert train_images.shape == (10847, 32, 32, 3)
    assert len(train_labels) == len(train_images)

    # Reuse only the already-extracted 1980-image TRAIN holdout feature maps.
    # This is a historical development split, not an independent test set.
    validation_maps_path = CLEAN_RUN / "validation_feature_maps.npy"
    validation_labels = np.load(CLEAN_RUN / "validation_labels.npy", allow_pickle=False)
    validation_maps = np.load(validation_maps_path, mmap_mode="r", allow_pickle=False)
    assert validation_maps.shape == (1980, 64, 8, 8)
    validation_features = np.empty((len(validation_labels), FEATURE_DIM), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(validation_labels), BATCH_SIZE):
            stop = min(start + BATCH_SIZE, len(validation_labels))
            batch = torch.from_numpy(np.array(validation_maps[start:stop], copy=True)).to(DEVICE)
            validation_features[start:stop] = F.avg_pool2d(batch, 8).flatten(1).cpu().numpy()
    np.save(ROOT / "validation_features.npy", validation_features)
    np.save(ROOT / "validation_labels.npy", validation_labels.astype(np.int64))

    pool_equivalence_passed = verify_pool_equivalence()
    seed_manifests = {}
    for seed in SEEDS:
        seed_manifests[str(seed)] = extract_seed_feature_stream(
            seed,
            train_images,
            train_labels,
            transforms["train_transform"],
            backbone,
            initial_weight,
            initial_bias,
            class_counts,
        )

    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset": "CIFAR-100-LT",
        "imbalance_factor": 100,
        "train_samples": int(len(train_labels)),
        "class_counts": class_counts.tolist(),
        "stage1_checkpoint": str(stage1_path),
        "stage1_checkpoint_sha256": clean.sha(stage1_path),
        "stage1_epoch": 184,
        "stage1_seed": 42,
        "initial_classifier_sha256": initial_classifier_sha,
        "initial_classifier_file_sha256": sha256_file(ROOT / "initial_classifier.pth"),
        "validation_features_sha256": sha256_file(ROOT / "validation_features.npy"),
        "validation_labels_sha256": sha256_file(ROOT / "validation_labels.npy"),
        "validation_samples": int(len(validation_labels)),
        "validation_missing_classes": [0],
        "validation_role": "historical TRAIN development split only; no model selection; not independent",
        "official_TEST_used": False,
        "pool_equivalence_test": pool_equivalence_passed,
        "stage2_seeds": list(SEEDS),
        "stage2_arms": list(ARMS),
        "stage2_epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "updates_per_epoch": 85,
        "feature_dim": FEATURE_DIM,
        "h2t_replaced_channels": NUM_CHANNELS_REPLACED,
        "subspace_rank": SUBSPACE_RANK,
        "shrinkage_tau": SHRINKAGE_TAU,
        "boundary_alpha": BOUNDARY_ALPHA,
        "max_norm_ratio": MAX_NORM_RATIO,
        "min_alignment": MIN_ALIGNMENT,
        "feature_cache": seed_manifests,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "source_clean_protocol": str(CLEAN_RUN / "CLEAN_STAGE1_PROTOCOL.md"),
        "source_clean_protocol_sha256": sha256_file(CLEAN_RUN / "CLEAN_STAGE1_PROTOCOL.md"),
    }
    write_json(ROOT / "prepare_manifest.json", manifest)
    guard_state = read_json(ROOT / "test_guard.json")
    guard_pass(guard_state)
    print("PREPARE PASS: frozen Clean Stage-I features, paired augmentation subspaces, "
          "boundary budgets, and five-arm matched schedules are ready.", flush=True)


def load_cache(seed):
    cache_dir = ROOT / "feature_cache" / f"seed{seed}"
    manifest = read_json(cache_dir / "cache_manifest.json")
    for name, expected in manifest["feature_cache_files"].items():
        actual = sha256_file(cache_dir / name)
        if actual != expected:
            raise RuntimeError(f"feature cache checksum mismatch: seed={seed}, file={name}")
    return {
        "receiver_features": np.load(cache_dir / "receiver_features.npy", mmap_mode="r", allow_pickle=False),
        "donor_features": np.load(cache_dir / "donor_features.npy", mmap_mode="r", allow_pickle=False),
        "receiver_indices": np.load(cache_dir / "receiver_indices.npy", mmap_mode="r", allow_pickle=False),
        "donor_indices": np.load(cache_dir / "donor_indices.npy", mmap_mode="r", allow_pickle=False),
        "receiver_labels": np.load(cache_dir / "receiver_labels.npy", mmap_mode="r", allow_pickle=False),
        "h2t_masks": np.load(cache_dir / "h2t_masks.npy", mmap_mode="r", allow_pickle=False),
        "subspace_basis": np.load(cache_dir / "subspace_basis.npy", allow_pickle=False),
        "subspace_eigenvalues": np.load(cache_dir / "subspace_eigenvalues.npy", allow_pickle=False),
        "boundary_delta": np.load(cache_dir / "boundary_delta.npy", mmap_mode="r", allow_pickle=False),
        "boundary_radius": np.load(cache_dir / "boundary_radius.npy", mmap_mode="r", allow_pickle=False),
        "boundary_margin": np.load(cache_dir / "boundary_margin.npy", mmap_mode="r", allow_pickle=False),
        "boundary_alignment": np.load(cache_dir / "boundary_alignment.npy", mmap_mode="r", allow_pickle=False),
        "boundary_competitor": np.load(cache_dir / "boundary_competitor.npy", mmap_mode="r", allow_pickle=False),
        "boundary_eligible": np.load(cache_dir / "boundary_eligible.npy", mmap_mode="r", allow_pickle=False),
        "boundary_clipped": np.load(cache_dir / "boundary_clipped.npy", mmap_mode="r", allow_pickle=False),
        "class_counts": np.load(cache_dir / "class_counts.npy", allow_pickle=False),
        "initial_weight": np.load(cache_dir / "initial_weight.npy", allow_pickle=False),
        "initial_bias": np.load(cache_dir / "initial_bias.npy", allow_pickle=False),
        "manifest": manifest,
    }


def set_epoch_lr(optimizer, epoch, cfg):
    # Exact cosine schedule from the locked original H2T Stage-II entrypoint.
    lr = 0.5 * cfg.lr * (1.0 + math.cos(epoch / cfg.num_epochs * math.pi))
    optimizer.param_groups[0]["lr"] = cfg.lr_scheduler.lr_factor * lr


def evaluate_head(head, validation_features, validation_labels, class_counts):
    head.eval()
    logits_parts = []
    with torch.no_grad():
        for start in range(0, len(validation_labels), BATCH_SIZE):
            stop = min(start + BATCH_SIZE, len(validation_labels))
            features = torch.from_numpy(
                np.array(validation_features[start:stop], copy=True)
            ).to(DEVICE)
            logits_parts.append(head(features).cpu().numpy())
    logits = np.concatenate(logits_parts, axis=0)
    predictions = logits.argmax(axis=1)
    metrics = clean.metrics(predictions, validation_labels)

    rows = np.arange(len(validation_labels))
    competitors_logits = logits.copy()
    competitors_logits[rows, validation_labels] = -np.inf
    competitors = competitors_logits.argmax(axis=1)
    margins = logits[rows, validation_labels] - logits[rows, competitors]
    true_few = class_counts[validation_labels] < 20
    true_many = class_counts[validation_labels] > 100
    predicted_few = class_counts[predictions] < 20
    predicted_many = class_counts[predictions] > 100

    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    np.add.at(confusion, (validation_labels, predictions), 1)
    cross_entropy = float(
        F.cross_entropy(
            torch.from_numpy(logits),
            torch.from_numpy(validation_labels.astype(np.int64)),
        ).item()
    )
    metrics.update(
        {
            "validation_CE": cross_entropy,
            "mean_true_vs_nearest_competitor_margin": float(margins.mean()),
            "mean_few_margin": float(margins[true_few].mean()) if true_few.any() else None,
            "few_negative_margin_fraction": float((margins[true_few] < 0).mean()) if true_few.any() else None,
            "mean_many_margin": float(margins[true_many].mean()) if true_many.any() else None,
            "tail_to_head_error_rate": (
                float(predicted_many[true_few].mean()) if true_few.any() else None
            ),
            "head_to_tail_error_rate": (
                float(predicted_few[true_many].mean()) if true_many.any() else None
            ),
            "validation_is_historical_train_holdout": True,
            "official_TEST_used": False,
        }
    )
    return metrics, logits, predictions, confusion


def run_one(seed, arm, epochs, output_dir, evaluate_final):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite run directory: {output_dir}")
    output_dir.mkdir(parents=True)

    cache = load_cache(seed)
    cfg, transforms, _, _, criterion_class, _ = clean.imports(2)
    assert cfg.num_epochs == NUM_EPOCHS
    assert cfg.batch_size == BATCH_SIZE
    assert cfg.loss.type == "CrossEntropyLoss" and not cfg.loss.CE.reweight_CE
    assert cfg.shift_bn is False

    initial_state = torch.load(ROOT / "initial_classifier.pth", map_location="cpu", weights_only=True)
    head = nn.Linear(FEATURE_DIM, NUM_CLASSES).to(DEVICE)
    head.load_state_dict(initial_state, strict=True)
    initial_hash = hash_state_dict(head.state_dict())
    expected_initial_hash = read_json(ROOT / "prepare_manifest.json")["initial_classifier_sha256"]
    if initial_hash != expected_initial_hash:
        raise AssertionError("Stage-II classifier initialization differs from Clean Stage-I fc")

    optimizer = torch.optim.SGD(
        [{"params": head.parameters()}],
        cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )
    criterion = criterion_class(cfg, cache["class_counts"]).to(DEVICE)
    rng_arm_code = ARMS.index(arm) + 1
    epoch_rows = []
    status = {
        "state": "running",
        "seed": int(seed),
        "arm": arm,
        "arm_description": ARM_LABELS[arm],
        "epochs_requested": int(epochs),
        "updates_per_epoch": 85,
        "initial_classifier_sha256": initial_hash,
        "stage1_checkpoint_sha256": read_json(ROOT / "prepare_manifest.json")[
            "stage1_checkpoint_sha256"
        ],
        "feature_cache_schedule_sha256": cache["manifest"]["feature_cache_schedule_sha256"],
        "official_TEST_used": False,
        "backbone_trainable": False,
        "backbone_BN_updated": False,
        "optimizer_state_fresh": len(optimizer.state) == 0,
    }
    write_json(output_dir / "status.json", status)

    receiver_features = cache["receiver_features"]
    donor_features = cache["donor_features"]
    donor_indices = cache["donor_indices"]
    receiver_labels = cache["receiver_labels"]
    masks = cache["h2t_masks"]
    boundary_delta = cache["boundary_delta"]
    boundary_radius = cache["boundary_radius"]
    boundary_eligible = cache["boundary_eligible"]
    basis = cache["subspace_basis"]
    eigenvalues = cache["subspace_eigenvalues"]

    with (output_dir / "train_log.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = [
            "epoch", "batch", "batch_size", "loss", "lr",
            "mean_perturbation_norm", "mean_boundary_budget",
            "boundary_eligible_fraction", "max_norm_match_error",
            "subspace_projection_residual",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()

        for epoch in range(epochs):
            head.train()
            set_epoch_lr(optimizer, epoch, cfg)
            total_loss = 0.0
            seen = 0
            epoch_norms = []
            epoch_budgets = []
            epoch_eligible = []
            epoch_norm_errors = []
            epoch_projection_residuals = []

            for batch, start in enumerate(range(0, 10847, BATCH_SIZE)):
                stop = min(start + BATCH_SIZE, 10847)
                z_np = np.asarray(receiver_features[epoch, start:stop], dtype=np.float32)
                y_np = np.asarray(receiver_labels[epoch, start:stop], dtype=np.int64)
                x_np = z_np.copy()
                delta_np = np.zeros_like(x_np)
                budget = np.asarray(boundary_radius[epoch, start:stop], dtype=np.float32)

                if arm == "h2t_pooled":
                    donor_idx = np.asarray(donor_indices[epoch, start:stop], dtype=np.int64)
                    donor_np = np.asarray(donor_features[epoch, donor_idx], dtype=np.float32)
                    selected = np.asarray(masks[epoch, batch], dtype=np.int64)
                    if len(np.unique(selected)) != NUM_CHANNELS_REPLACED:
                        raise AssertionError("H2T mask must contain exactly 16 unique channels")
                    x_np[:, selected] = donor_np[:, selected]

                elif arm in ("gaussian", "subspace_random", "subspace_boundary"):
                    rng = np.random.default_rng(
                        int(seed) * 1000003 + rng_arm_code * 10007 + epoch * 1009 + batch
                    )
                    if arm == "gaussian":
                        directions = sample_isotropic_directions(rng, len(z_np), FEATURE_DIM)
                        delta_np = scale_directions(directions, budget)
                    elif arm == "subspace_random":
                        directions = sample_subspace_directions(
                            rng, y_np, basis, eigenvalues
                        )
                        delta_np = scale_directions(directions, budget)
                        active = np.linalg.norm(delta_np, axis=1) > 1e-8
                        if active.any():
                            selected_basis = basis[y_np[active]].astype(np.float64)
                            selected_delta = delta_np[active].astype(np.float64)
                            coefficients = np.einsum(
                                "ndk,nd->nk", selected_basis, selected_delta
                            )
                            projected = np.einsum(
                                "ndk,nk->nd", selected_basis, coefficients
                            )
                            residual = np.linalg.norm(selected_delta - projected, axis=1)
                            denom = np.maximum(np.linalg.norm(selected_delta, axis=1), 1e-12)
                            epoch_projection_residuals.extend((residual / denom).tolist())
                    else:
                        delta_np = np.asarray(
                            boundary_delta[epoch, start:stop], dtype=np.float32
                        )
                    x_np = z_np + delta_np

                elif arm != "crt":
                    raise ValueError(f"unknown arm: {arm}")

                actual_norm = np.linalg.norm(delta_np, axis=1)
                if arm in ("gaussian", "subspace_random", "subspace_boundary"):
                    norm_error = np.abs(actual_norm - budget)
                    if norm_error.max(initial=0.0) > 2e-5:
                        raise AssertionError(
                            f"{arm} failed per-example perturbation-norm matching: "
                            f"{norm_error.max()}"
                        )
                    epoch_norm_errors.extend(norm_error.tolist())
                    epoch_budgets.extend(budget.tolist())
                    epoch_eligible.extend(
                        np.asarray(boundary_eligible[epoch, start:stop], dtype=np.uint8).tolist()
                    )
                epoch_norms.extend(actual_norm.tolist())

                features = torch.from_numpy(x_np).to(DEVICE, non_blocking=True)
                targets = torch.from_numpy(y_np).to(DEVICE, non_blocking=True)
                logits = head(features)
                loss = criterion(logits, targets, extra_info=None)
                if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                    raise FloatingPointError(f"nonfinite logits/loss at {arm} epoch={epoch} batch={batch}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if any(
                    parameter.grad is None or not torch.isfinite(parameter.grad).all()
                    for parameter in head.parameters()
                ):
                    raise FloatingPointError(f"nonfinite classifier gradient at {arm} epoch={epoch}")
                optimizer.step()

                batch_size = stop - start
                total_loss += float(loss.detach()) * batch_size
                seen += batch_size
                writer.writerow(
                    {
                        "epoch": epoch + 1,
                        "batch": batch,
                        "batch_size": batch_size,
                        "loss": float(loss.detach()),
                        "lr": optimizer.param_groups[0]["lr"],
                        "mean_perturbation_norm": float(actual_norm.mean()),
                        "mean_boundary_budget": float(budget.mean()),
                        "boundary_eligible_fraction": float(
                            np.asarray(boundary_eligible[epoch, start:stop]).mean()
                        ),
                        "max_norm_match_error": float(
                            np.abs(actual_norm - budget).max(initial=0.0)
                        ) if arm in ("gaussian", "subspace_random", "subspace_boundary") else 0.0,
                        "subspace_projection_residual": (
                            float(np.mean(epoch_projection_residuals[-batch_size:]))
                            if arm == "subspace_random" and epoch_projection_residuals
                            else 0.0
                        ),
                    }
                )

            if seen != 10847 or optimizer.state == {}:
                raise AssertionError("unexpected Stage-II sample/update count or empty optimizer")
            epoch_rows.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": total_loss / seen,
                    "lr": optimizer.param_groups[0]["lr"],
                    "steps": 85,
                    "samples": seen,
                    "mean_perturbation_norm": float(np.mean(epoch_norms)),
                    "mean_boundary_budget": float(np.mean(epoch_budgets)) if epoch_budgets else 0.0,
                    "boundary_eligible_fraction": float(np.mean(epoch_eligible)) if epoch_eligible else 0.0,
                    "max_norm_match_error": float(np.max(epoch_norm_errors)) if epoch_norm_errors else 0.0,
                    "mean_subspace_projection_residual": (
                        float(np.mean(epoch_projection_residuals))
                        if epoch_projection_residuals else 0.0
                    ),
                }
            )
            status["completed_epochs"] = epoch + 1
            status["epoch_metrics"] = epoch_rows.copy()
            write_json(output_dir / "status.json", status)
            print(
                f"seed={seed} arm={arm} epoch={epoch + 1}/{epochs} "
                f"loss={total_loss / seen:.6f}",
                flush=True,
            )

    final_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
    if hash_state_dict(final_state) == initial_hash:
        raise AssertionError("Stage-II classifier did not update")
    checkpoint = {
        "state_dict": final_state,
        "seed": int(seed),
        "arm": arm,
        "epochs": int(epochs),
        "initial_classifier_sha256": initial_hash,
        "stage1_checkpoint_sha256": status["stage1_checkpoint_sha256"],
        "feature_cache_schedule_sha256": status["feature_cache_schedule_sha256"],
        "official_TEST_used": False,
        "backbone_trainable": False,
        "backbone_BN_updated": False,
    }
    torch.save(checkpoint, output_dir / "classifier_final.pth")

    metrics = None
    if evaluate_final:
        validation_features = np.load(ROOT / "validation_features.npy", mmap_mode="r", allow_pickle=False)
        validation_labels = np.load(ROOT / "validation_labels.npy", allow_pickle=False)
        metrics, logits, predictions, confusion = evaluate_head(
            head, validation_features, validation_labels, cache["class_counts"]
        )
        np.save(output_dir / "confusion_matrix.npy", confusion)
        np.savez_compressed(
            output_dir / "validation_predictions.npz",
            logits=logits.astype(np.float32),
            predictions=predictions.astype(np.int64),
            labels=validation_labels.astype(np.int64),
        )
        write_json(output_dir / "metrics.json", metrics)

    status.update(
        {
            "state": "completed",
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "completed_epochs": int(epochs),
            "total_updates": int(epochs * 85),
            "initial_classifier_sha256": initial_hash,
            "final_classifier_sha256": hash_state_dict(final_state),
            "checkpoint_sha256": sha256_file(output_dir / "classifier_final.pth"),
            "metrics": metrics,
            "official_TEST_used": False,
        }
    )
    write_json(output_dir / "status.json", status)
    return status


def smoke():
    if (ROOT / "smoke_status.json").exists():
        raise FileExistsError("smoke_status.json already exists; refusing to overwrite")
    manifest = read_json(ROOT / "prepare_manifest.json")
    assert manifest["pool_equivalence_test"] is True
    smoke_root = ROOT / "runs" / "smoke"
    if smoke_root.exists():
        raise FileExistsError(f"Refusing to overwrite smoke output: {smoke_root}")

    results = {}
    for arm in ARMS:
        output_dir = smoke_root / "seed42" / arm
        status = run_one(42, arm, 1, output_dir, evaluate_final=False)
        results[arm] = {
            "state": status["state"],
            "completed_epochs": status["completed_epochs"],
            "total_updates": status["total_updates"],
            "initial_classifier_sha256": status["initial_classifier_sha256"],
            "final_classifier_sha256": status["final_classifier_sha256"],
            "finite_training": True,
        }

    initial_hashes = {value["initial_classifier_sha256"] for value in results.values()}
    assert len(initial_hashes) == 1, "All smoke arms must start from the same Stage-I classifier"
    assert all(value["state"] == "completed" and value["completed_epochs"] == 1 for value in results.values())
    output = {
        "passed": True,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed": 42,
        "arms": list(ARMS),
        "arm_results": results,
        "shared_initial_classifier": True,
        "official_TEST_used": False,
        "formal_training_authorized_by_smoke": True,
    }
    write_json(ROOT / "smoke_status.json", output)
    print("SMOKE PASS: all five arms completed one finite epoch with the same Stage-I classifier.", flush=True)


def train_formal():
    smoke_status_path = ROOT / "smoke_status.json"
    if not smoke_status_path.is_file() or not read_json(smoke_status_path).get("passed", False):
        raise RuntimeError("Formal training requires a successful five-arm smoke run")
    formal_root = ROOT / "runs" / "formal"
    if formal_root.exists():
        raise FileExistsError(f"Refusing to overwrite formal outputs: {formal_root}")

    results = []
    for seed in SEEDS:
        for arm in ARMS:
            output_dir = formal_root / f"seed{seed}" / arm
            status = run_one(seed, arm, NUM_EPOCHS, output_dir, evaluate_final=True)
            results.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "state": status["state"],
                    "metrics": status["metrics"],
                }
            )
    complete = {
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "stage1_seed": 42,
        "stage1_epoch": 184,
        "stage2_seeds": list(SEEDS),
        "arms": list(ARMS),
        "epochs_per_run": NUM_EPOCHS,
        "updates_per_run": NUM_EPOCHS * 85,
        "runs": results,
        "official_TEST_used": False,
        "validation_role": "historical TRAIN development split only; no checkpoint selection",
    }
    write_json(ROOT / "formal_complete.json", complete)
    print("FORMAL COMPLETE: 5 arms x 3 Stage-II seeds x 10 epochs; official TEST untouched.", flush=True)


def report():
    if not (ROOT / "formal_complete.json").is_file():
        raise RuntimeError("Cannot report before formal training is complete")
    complete = read_json(ROOT / "formal_complete.json")
    metric_names = [
        "Overall", "Head", "OfficialMedium", "Tail",
        "Many", "Medium", "Few", "Macro",
        "validation_CE", "tail_to_head_error_rate", "head_to_tail_error_rate",
        "mean_few_margin", "few_negative_margin_fraction",
    ]
    values = {}
    for seed in SEEDS:
        for arm in ARMS:
            metrics = read_json(ROOT / "runs" / "formal" / f"seed{seed}" / arm / "metrics.json")
            values[(seed, arm)] = metrics

    rows = []
    for arm in ARMS:
        for metric in metric_names:
            per_seed = [values[(seed, arm)].get(metric) for seed in SEEDS]
            finite = np.asarray([value for value in per_seed if value is not None], dtype=np.float64)
            deltas = [
                values[(seed, arm)].get(metric) - values[(seed, "crt")].get(metric)
                for seed in SEEDS
                if arm != "crt"
                and values[(seed, arm)].get(metric) is not None
                and values[(seed, "crt")].get(metric) is not None
            ]
            rows.append(
                {
                    "arm": arm,
                    "arm_description": ARM_LABELS[arm],
                    "metric": metric,
                    "seed42": per_seed[0],
                    "seed43": per_seed[1],
                    "seed44": per_seed[2],
                    "mean": float(finite.mean()) if finite.size else None,
                    "std": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
                    "paired_delta_vs_crt_mean": float(np.mean(deltas)) if deltas else 0.0,
                    "paired_delta_vs_crt_std": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
                }
            )

    with (ROOT / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "stage1_seed": 42,
        "stage1_epoch": 184,
        "stage1_checkpoint_sha256": read_json(ROOT / "prepare_manifest.json")[
            "stage1_checkpoint_sha256"
        ],
        "stage2_seeds": list(SEEDS),
        "arms": {arm: ARM_LABELS[arm] for arm in ARMS},
        "primary_endpoint": "fixed Stage-II epoch 10",
        "primary_metrics": ["Overall", "Many", "Medium", "Few", "Tail", "Macro"],
        "secondary_metrics": [
            "tail_to_head_error_rate", "head_to_tail_error_rate",
            "mean_few_margin", "few_negative_margin_fraction",
        ],
        "results": rows,
        "official_TEST_used": False,
        "validation_role": "1980-image historical TRAIN holdout; class 0 absent; development diagnostic only",
        "interpretation_limit": (
            "The three Stage-II seeds share one frozen Stage-I checkpoint. "
            "They measure Stage-II stochastic robustness, not independent end-to-end seeds. "
            "No official TEST result or statistical-significance claim is made."
        ),
    }
    write_json(ROOT / "summary.json", summary)

    markdown = [
        "# Structured Feature Perturbation: exploratory results",
        "",
        "Stage I is the locked Clean ResNet-32 checkpoint (seed 42, epoch 184).",
        "Stage II uses frozen 64-D GAP features and trains only the linear classifier.",
        "The official CIFAR-100 TEST set was not opened.",
        "",
        "The reported endpoint is fixed Stage-II epoch 10; no validation-best checkpoint is selected.",
        "The 1980-image validation set is a historical TRAIN holdout, lacks class 0, and is used only as a development diagnostic.",
        "",
        "| Arm | Overall | Many | Medium | Few | Tail | Macro |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        row_by_metric = {
            row["metric"]: row for row in rows if row["arm"] == arm
        }
        columns = []
        for metric in ("Overall", "Many", "Medium", "Few", "Tail", "Macro"):
            value = row_by_metric[metric]["mean"]
            columns.append("n/a" if value is None else f"{value:.3f}")
        markdown.append(f"| {arm} | " + " | ".join(columns) + " |")

    markdown.extend(
        [
            "",
            "## Mandatory interpretation",
            "",
            "- Compare each generated arm against cRT and H2T-pooled; do not infer success from Overall alone.",
            "- Gaussian and both structured arms share the exact per-example perturbation norm budget derived from the boundary construction.",
            "- Subspace-Random tests whether augmentation-induced directions help without boundary alignment.",
            "- Subspace-Boundary tests whether the projected nearest-boundary direction adds value under the same norm budget.",
            "- H2T-pooled is a controlled frozen-BN classifier-only comparator. It is GAP-equivalent to channel replacement on the 64x8x8 map, but is not claimed to be a full official H2T reproduction.",
            "- The three Stage-II seeds share one Stage-I checkpoint; they are not three independent Stage-I runs.",
            "- The historical TRAIN holdout has already been reused in earlier work. These results are exploratory and do not establish official TEST generalization.",
            "",
            "See summary.csv and summary.json for per-seed values and paired deltas.",
            "",
        ]
    )
    (ROOT / "RESULTS.md").write_text("\n".join(markdown), encoding="utf-8")
    print("REPORT COMPLETE: summary.csv, summary.json and RESULTS.md written.", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "smoke", "train", "report"))
    args = parser.parse_args()
    guard_state = install_test_guard()
    try:
        if args.action == "prepare":
            prepare()
        elif args.action == "smoke":
            smoke()
        elif args.action == "train":
            train_formal()
        else:
            report()
        guard_pass(guard_state)
    except BaseException:
        write_json(
            ROOT / "failure.json",
            {
                "action": args.action,
                "failed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "traceback": traceback.format_exc(),
                "test_guard": guard_state,
            },
        )
        raise


if __name__ == "__main__":
    main()
