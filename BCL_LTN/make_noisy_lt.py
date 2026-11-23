from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from BCL_AUDIT.cifar100lt.data import CIFAR100LT, cifar100lt_class_counts


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_symmetric_noise(clean_labels: np.ndarray, noise_rate: float = 0.4,
                         seed: int = 42, num_classes: int = 100):
    """Independent Bernoulli flips; replacement is uniform over the other K-1 labels."""
    clean = np.asarray(clean_labels, dtype=np.int64)
    if clean.ndim != 1 or clean.size != 10_847:
        raise ValueError(f"Expected exactly 10,847 selected LT labels, got {clean.shape}")
    if not 0.0 <= noise_rate <= 1.0:
        raise ValueError("noise_rate must be in [0,1]")
    if np.any((clean < 0) | (clean >= num_classes)):
        raise ValueError("Clean labels out of range")
    rng = np.random.RandomState(seed)
    is_noisy = rng.rand(clean.size) < float(noise_rate)
    noisy = clean.copy()
    n = int(is_noisy.sum())
    if n:
        offsets = rng.randint(1, num_classes, size=n)
        noisy[is_noisy] = (clean[is_noisy] + offsets) % num_classes
    if np.any(noisy[is_noisy] == clean[is_noisy]):
        raise RuntimeError("A flipped label unexpectedly retained its source class")
    if np.any(noisy[~is_noisy] != clean[~is_noisy]):
        raise RuntimeError("An unflipped label was modified")
    return noisy, clean.copy(), is_noisy


def group_class_ids(clean_counts: np.ndarray):
    return {
        "Many": np.flatnonzero(clean_counts > 100),
        "Medium": np.flatnonzero((clean_counts >= 20) & (clean_counts <= 100)),
        "Few": np.flatnonzero(clean_counts < 20),
    }


def build(args):
    data_root = args.data_root.expanduser().resolve()
    indices_path = args.indices_file.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve()
    if not indices_path.is_file():
        raise FileNotFoundError(f"Fixed indices missing: {indices_path}")
    if not data_root.is_dir():
        raise FileNotFoundError(f"CIFAR data root missing: {data_root}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # CIFAR100LT uses the strict TRAIN-only loader and validates the fixed indices.
    base = CIFAR100LT(root=data_root, mode="bcl", indices_file=indices_path,
                      split_seed=0, download=False)
    indices = base.indices.astype(np.int64, copy=True)
    clean = base.targets[indices].astype(np.int64, copy=True)
    expected = cifar100lt_class_counts()
    if len(indices) != 10_847 or len(np.unique(indices)) != 10_847:
        raise RuntimeError("Fixed IF=100 index integrity failed")
    if not np.array_equal(np.bincount(clean, minlength=100), expected):
        raise RuntimeError("Selected clean labels do not match the canonical IF=100 counts")

    noisy, clean_saved, is_noisy = make_symmetric_noise(
        clean, noise_rate=args.noise_rate, seed=args.seed, num_classes=100
    )
    observed_counts = np.bincount(noisy, minlength=100).astype(np.int64)
    if np.any(observed_counts <= 0) or int(observed_counts.sum()) != 10_847:
        raise RuntimeError("Noisy-label class-count vector is invalid")

    noisy_path = out_dir / "cifar100lt_if100_sym04_seed42.npy"
    clean_path = out_dir / "cifar100lt_if100_clean_seed42.npy"
    mask_path = out_dir / "cifar100lt_if100_is_noisy_seed42.npy"
    if any(p.exists() for p in (noisy_path, clean_path, mask_path)):
        raise FileExistsError("Refusing to overwrite existing noise arrays")
    np.save(noisy_path, noisy)
    np.save(clean_path, clean_saved)
    np.save(mask_path, is_noisy.astype(np.bool_))

    groups = group_class_ids(expected)
    class_group = np.empty(100, dtype=object)
    for name, ids in groups.items():
        class_group[ids] = name
    group_summary = {}
    for name, ids in groups.items():
        class_mask = np.isin(clean, ids)
        group_summary[name] = {
            "num_classes": int(len(ids)),
            "clean_samples": int(class_mask.sum()),
            "observed_noisy_label_samples": int(np.isin(noisy, ids).sum()),
            "clean_source_samples_flipped": int((is_noisy & class_mask).sum()),
            "clean_source_samples_unflipped": int((~is_noisy & class_mask).sum()),
        }

    transition = np.zeros((100, 100), dtype=np.int64)
    np.add.at(transition, (clean, noisy), 1)
    manifest = {
        "protocol": "instance-independent symmetric K-class label noise on fixed CIFAR-100-LT IF=100 train subset",
        "noise_rate_requested": float(args.noise_rate),
        "noise_rate_realized": float(is_noisy.mean()),
        "num_train_samples": int(len(clean)),
        "num_flipped": int(is_noisy.sum()),
        "num_unflipped": int((~is_noisy).sum()),
        "seed": int(args.seed),
        "num_classes": 100,
        "replacement": "uniform among the other 99 classes; original class excluded",
        "indices_file": str(indices_path),
        "indices_sha256": sha256_file(indices_path),
        "noisy_labels_file": noisy_path.name,
        "noisy_labels_sha256": sha256_file(noisy_path),
        "clean_labels_file": clean_path.name,
        "clean_labels_sha256": sha256_file(clean_path),
        "is_noisy_file": mask_path.name,
        "is_noisy_sha256": sha256_file(mask_path),
        "clean_class_counts": expected.tolist(),
        "observed_noisy_class_counts": observed_counts.tolist(),
        "many_medium_few": group_summary,
        "transition_matrix": transition.tolist(),
        "leakage_policy": "clean labels and is_noisy mask are offline diagnostics only; never passed to train dataset, loss, prior, or optimizer",
    }
    manifest_path = out_dir / "noise_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Noise requested={args.noise_rate:.4f}; realized={is_noisy.mean():.6f} "
          f"({int(is_noisy.sum())}/{len(is_noisy)})")
    print("Group | classes | clean-source samples | noisy labels landing in group | source flips")
    for name, row in group_summary.items():
        print(f"{name:6s} | {row['num_classes']:7d} | {row['clean_samples']:20d} | "
              f"{row['observed_noisy_label_samples']:29d} | {row['clean_source_samples_flipped']:12d}")
    print(f"Saved noisy labels: {noisy_path}")
    print(f"Saved clean labels and mask for offline diagnostics: {clean_path}, {mask_path}")
    print(f"Manifest: {manifest_path}")
    return manifest


def main():
    p = argparse.ArgumentParser(description="Create reproducible symmetric-noise CIFAR-100-LT labels")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--indices-file", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("BCL_LTN/data"))
    p.add_argument("--noise-rate", type=float, default=0.4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if abs(args.noise_rate - 0.4) > 1e-12 or args.seed != 42:
        raise ValueError("This locked experiment requires NR=0.4 and seed=42")
    build(args)


if __name__ == "__main__":
    main()
