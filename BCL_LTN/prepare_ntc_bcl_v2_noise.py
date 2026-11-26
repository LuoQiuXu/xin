"""Prepare isolated observed-label files for the NTC-BCL v2 Stage-1 audit.

Synthetic labels are generated from the already audited CIFAR-100-LT IF=100
TRAIN IDs. CIFAR-100N human fine labels are reused from the previous audited
local cache and re-aligned by raw Python-CIFAR TRAIN IDs. Clean labels and true
noise matrices are written only under each regime's truth/ subdirectory.
Training code must read observed_labels.npy only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "H2T" / "data" / "cifar100"
DEFAULT_INDICES = ROOT / "BCL_AUDIT" / "outputs" / "data_audit_recheck_20261122_01" / "cifar100lt_if100_train_indices_seed0.npy"
DEFAULT_HUMAN_LABELS = (
    ROOT / "H2T_repro_audit" / "noisy_lt_failure_screen" / "results"
    / "cifar100n_lt_seed42_20261029_01" / "primary_cache" / "inputs" / "human_labels.npz"
)
DEFAULT_LEGACY_SPLIT = (
    ROOT / "H2T_repro_audit" / "noisy_lt_failure_screen" / "results"
    / "cifar100n_lt_seed42_20261029_01" / "primary_cache" / "inputs" / "split.npz"
)
EXPECTED_TRAIN_SHA256 = "735e79b04f092ca3d2e6d07f368c0a7d70d48c48d28865950cc24454cf45129b"
EXPECTED_HUMAN_LABELS_NPZ_SHA256 = "e50697c3d0e956ea837ba689b298d6fc2ccd97841b075ae613fc2d2e90d94e87"
EXPECTED_SPLIT_NPZ_SHA256 = "2f2b478a349cf94f632f92b766aba5197cc18edc983b4ffea82582f7a000bc88"
EXPECTED_INDICES_SHA256 = "be73a9441289dbadec4190c8c30a1bfc551434f17b33f6dc0ceea7fb77340ac7"
EXPECTED_CIFAR100N_LT_NOISE_RATE = 0.35973080114317324
EXPECTED_V1_SYM04_NOISY_LABELS_SHA256 = "691b8c8bf5758811d54c33c213e3bf2db12f586aa3d857f37da309a55d03554f"
N, K = 10_847, 100


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def empirical_transition(clean: np.ndarray, observed: np.ndarray) -> np.ndarray:
    counts = np.zeros((K, K), dtype=np.float64)
    np.add.at(counts, (clean, observed), 1.0)
    row_sums = counts.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise RuntimeError("A clean class has no samples; refusing to create a benchmark")
    return counts / row_sums

def coarse_cycle_mapping(fine_full: np.ndarray, coarse_full: np.ndarray) -> np.ndarray:
    """Map each CIFAR-100 fine class to the next fine class in its superclass.

    This defines an explicit, reproducible asymmetric audit protocol. It is
    named coarse-superclass cyclic noise and is not presented as the only
    canonical CIFAR-100 asymmetric-noise convention.
    """
    fine = np.asarray(fine_full, dtype=np.int64)
    coarse = np.asarray(coarse_full, dtype=np.int64)
    if fine.shape != (50_000,) or coarse.shape != (50_000,):
        raise ValueError("fine/coarse TRAIN labels must both have shape [50000]")
    fine_to_coarse = np.full(K, -1, dtype=np.int64)
    for class_id in range(K):
        values = np.unique(coarse[fine == class_id])
        if values.size != 1:
            raise RuntimeError(f"Fine class {class_id} does not map to exactly one superclass")
        fine_to_coarse[class_id] = int(values[0])
    mapping = np.full(K, -1, dtype=np.int64)
    for superclass in np.unique(fine_to_coarse):
        members = np.flatnonzero(fine_to_coarse == superclass)
        if members.size != 5:
            raise RuntimeError(f"Expected five fine classes per CIFAR-100 superclass, got {members.size}")
        mapping[members] = np.roll(members, -1)
    if np.any(mapping < 0) or np.any(mapping == np.arange(K)):
        raise RuntimeError("Invalid coarse-superclass cyclic transition mapping")
    return mapping


def make_asymmetric_coarse_cycle(
    clean: np.ndarray,
    target_map: np.ndarray,
    *,
    noise_rate: float = 0.4,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    clean = np.asarray(clean, dtype=np.int64)
    target_map = np.asarray(target_map, dtype=np.int64)
    if clean.ndim != 1 or target_map.shape != (K,):
        raise ValueError("clean must be [N] and target_map must be [100]")
    if not 0.0 <= noise_rate <= 1.0 or np.any((target_map < 0) | (target_map >= K)):
        raise ValueError("Invalid asymmetric noise rate or target mapping")
    if np.any(target_map == np.arange(K)):
        raise ValueError("Asymmetric target mapping must not retain the source class")
    rng = np.random.RandomState(seed)
    mask = rng.rand(clean.size) < float(noise_rate)
    observed = clean.copy()
    observed[mask] = target_map[clean[mask]]
    transition = np.eye(K, dtype=np.float64) * (1.0 - float(noise_rate))
    transition[np.arange(K), target_map] = float(noise_rate)
    return observed, mask, transition


def write_regime(
    output_root: Path,
    name: str,
    clean: np.ndarray,
    observed: np.ndarray,
    indices: np.ndarray,
    *,
    source: str,
    indices_sha256: str,
    requested_noise_rate: float | None = None,
    source_sha256: dict | None = None,
    theoretical_transition: np.ndarray | None = None,
) -> dict:
    if clean.shape != (N,) or observed.shape != (N,) or clean.dtype != np.int64 or observed.dtype != np.int64:
        raise ValueError(f"{name}: clean and observed labels must be int64 [{N}]")
    if np.any((clean < 0) | (clean >= K)) or np.any((observed < 0) | (observed >= K)):
        raise ValueError(f"{name}: label outside [0, {K})")
    observed_counts = np.bincount(observed, minlength=K)
    clean_counts = np.bincount(clean, minlength=K)
    if clean_counts.min() != 5 or clean_counts.max() != 500 or np.any(observed_counts == 0):
        raise RuntimeError(f"{name}: invalid IF=100 clean support or empty observed class")

    directory = output_root / name
    directory.mkdir(parents=False, exist_ok=False)
    truth_dir = directory / "truth"
    truth_dir.mkdir()
    observed_path = directory / "observed_labels.npy"
    np.save(observed_path, observed.astype(np.int64, copy=False), allow_pickle=False)

    noise_mask = clean != observed
    true_transition = empirical_transition(clean, observed)
    clean_path = truth_dir / "clean_labels.npy"
    mask_path = truth_dir / "is_noisy.npy"
    transition_path = truth_dir / "true_transition.npy"
    theoretical_path = truth_dir / "theoretical_transition.npy"
    np.save(clean_path, clean.astype(np.int64, copy=False), allow_pickle=False)
    np.save(mask_path, noise_mask.astype(np.bool_, copy=False), allow_pickle=False)
    np.save(transition_path, true_transition.astype(np.float64, copy=False), allow_pickle=False)
    if theoretical_transition is not None:
        theoretical_transition = np.asarray(theoretical_transition, dtype=np.float64)
        if (theoretical_transition.shape != (K, K) or not np.isfinite(theoretical_transition).all()
                or np.any(theoretical_transition < 0)
                or not np.allclose(theoretical_transition.sum(axis=1), 1.0, atol=1e-10)):
            raise ValueError("theoretical_transition must be a valid row-stochastic [100,100] matrix")
        np.save(theoretical_path, theoretical_transition, allow_pickle=False)

    public_manifest = {
        "protocol": "CIFAR-100-LT IF=100 fixed TRAIN IDs; observed labels only for training",
        "regime": name,
        "label_source": source,
        "num_samples": N,
        "num_classes": K,
        "indices_sha256": indices_sha256,
        "observed_labels_file": observed_path.name,
        "observed_labels_sha256": sha256(observed_path),
        "observed_class_counts": observed_counts.tolist(),
        "training_access_policy": "Training may load observed_labels.npy only; truth/ is offline evaluation only.",
    }
    truth_manifest = {
        "regime": name,
        "requested_noise_rate": requested_noise_rate,
        "realized_noise_rate": float(noise_mask.mean()),
        "num_noisy": int(noise_mask.sum()),
        "num_clean": int((~noise_mask).sum()),
        "clean_class_counts": clean_counts.tolist(),
        "true_transition_orientation": "T[clean_class, observed_class]",
        "true_transition_sha256": sha256(transition_path),
        "theoretical_transition_sha256": sha256(theoretical_path) if theoretical_transition is not None else None,
        "transition_reference": "theoretical" if theoretical_transition is not None else "empirical_only",
        "clean_labels_sha256": sha256(clean_path),
        "is_noisy_sha256": sha256(mask_path),
        "source_sha256": source_sha256 or {},
        "offline_only": True,
    }
    write_json(directory / "manifest.json", public_manifest)
    write_json(truth_dir / "truth_manifest.json", truth_manifest)
    return {"manifest": public_manifest, "truth": truth_manifest}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--indices-file", type=Path, default=DEFAULT_INDICES)
    parser.add_argument("--human-labels", type=Path, default=DEFAULT_HUMAN_LABELS)
    parser.add_argument("--legacy-split", type=Path, default=DEFAULT_LEGACY_SPLIT)
    parser.add_argument(
        "--output-root", type=Path,
        default=ROOT / "BCL_LTN" / "outputs" / "ntc_bcl_v2_stage1_data_20261126",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.seed != 42:
        raise ValueError("Stage-1 protocol is locked to seed 42")
    train_file = args.data_root / "cifar-100-python" / "train"
    for path in (train_file, args.indices_file, args.human_labels, args.legacy_split):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output root: {args.output_root}")

    train_sha = sha256(train_file)
    indices_sha = sha256(args.indices_file)
    human_sha = sha256(args.human_labels)
    split_sha = sha256(args.legacy_split)
    if train_sha != EXPECTED_TRAIN_SHA256:
        raise RuntimeError("CIFAR-100 Python TRAIN source SHA256 mismatch")
    if indices_sha != EXPECTED_INDICES_SHA256:
        raise RuntimeError("Fixed CIFAR-100-LT IF=100 indices SHA256 mismatch")
    if human_sha != EXPECTED_HUMAN_LABELS_NPZ_SHA256:
        raise RuntimeError("Cached official CIFAR-100N human_labels.npz SHA256 mismatch")
    if split_sha != EXPECTED_SPLIT_NPZ_SHA256:
        raise RuntimeError("Previously audited CIFAR-100N LT split.npz SHA256 mismatch")

    with train_file.open("rb") as stream:
        raw = pickle.load(stream, encoding="latin1")
    clean_full = np.asarray(raw["fine_labels"], dtype=np.int64)
    coarse_full = np.asarray(raw.get("coarse_labels"), dtype=np.int64)
    if (clean_full.shape != (50_000,) or np.any((clean_full < 0) | (clean_full >= K))
            or coarse_full.shape != (50_000,) or np.any((coarse_full < 0) | (coarse_full >= 20))):
        raise RuntimeError("Invalid CIFAR-100 Python TRAIN fine/coarse labels")
    indices = np.load(args.indices_file, allow_pickle=False)
    if (indices.shape != (N,) or indices.dtype != np.int64 or np.unique(indices).size != N
            or np.any((indices < 0) | (indices >= 50_000))):
        raise RuntimeError("Invalid fixed IF=100 raw TRAIN IDs")

    with np.load(args.legacy_split, allow_pickle=False) as split:
        legacy_train_ids = np.asarray(split["train_ids"], dtype=np.int64)
        legacy_val_ids = np.asarray(split["val_ids"], dtype=np.int64)
    if (legacy_train_ids.shape != (N,) or legacy_val_ids.shape != (1_980,)
            or not np.array_equal(np.sort(legacy_train_ids), np.sort(indices))
            or np.intersect1d(indices, legacy_val_ids).size != 0):
        raise RuntimeError("Current IF=100 IDs do not match the previously audited CIFAR-100N split")
    clean_lt = clean_full[indices]
    clean_counts = np.bincount(clean_lt, minlength=K)
    target_map = coarse_cycle_mapping(clean_full, coarse_full)
    if clean_counts.min() != 5 or clean_counts.max() != 500 or np.any(clean_counts == 0):
        raise RuntimeError("The selected clean subset is not CIFAR-100-LT IF=100")

    args.output_root.mkdir(parents=True, exist_ok=False)
    results = {}
    for rate, name in ((0.2, "sym02"), (0.4, "sym04"), (0.6, "sym06")):
        # Match the v1 generator exactly: Bernoulli mask first, then draw
        # offsets only for flipped examples from {1,...,99}.
        rng = np.random.RandomState(args.seed)
        mask = rng.rand(N) < float(rate)
        observed = clean_lt.copy()
        if int(mask.sum()):
            offsets = rng.randint(1, K, size=int(mask.sum()))
            observed[mask] = (clean_lt[mask] + offsets) % K
        theoretical_transition = np.full((K, K), float(rate) / (K - 1), dtype=np.float64)
        np.fill_diagonal(theoretical_transition, 1.0 - float(rate))
        results[name] = write_regime(
            args.output_root, name, clean_lt, observed, indices,
            source="synthetic instance-independent symmetric noise; other 99 classes equally likely",
            indices_sha256=indices_sha,
            requested_noise_rate=rate,
            source_sha256={"train": train_sha, "indices_file": indices_sha},
            theoretical_transition=theoretical_transition,
        )

    if results["sym04"]["manifest"]["observed_labels_sha256"] != EXPECTED_V1_SYM04_NOISY_LABELS_SHA256:
        raise RuntimeError("Generated Sym-0.4 labels do not exactly match the audited v1 noise realization")

    asym_observed, asym_mask, asym_transition = make_asymmetric_coarse_cycle(
        clean_lt, target_map, noise_rate=0.4, seed=args.seed
    )
    results["asym04_coarse_cycle"] = write_regime(
        args.output_root, "asym04_coarse_cycle", clean_lt, asym_observed, indices,
        source="synthetic instance-dependent-free asymmetric noise: cyclic fine-class mapping within each CIFAR-100 superclass",
        indices_sha256=indices_sha,
        requested_noise_rate=0.4,
        source_sha256={"train": train_sha, "indices_file": indices_sha},
        theoretical_transition=asym_transition,
    )
    
    with np.load(args.human_labels, allow_pickle=False) as human:
        human_clean = np.asarray(human["clean_label"], dtype=np.int64)
        human_noisy = np.asarray(human["noisy_label"], dtype=np.int64)
    if (human_clean.shape != (50_000,) or human_noisy.shape != (50_000,)
            or not np.array_equal(human_clean, clean_full)
            or np.any((human_noisy < 0) | (human_noisy >= K))):
        raise RuntimeError("CIFAR-100N official fine labels do not align with Python TRAIN order")
    real_clean = human_clean[indices]
    real_observed = human_noisy[indices]
    real_noise_rate = float(np.mean(real_clean != real_observed))
    if abs(real_noise_rate - EXPECTED_CIFAR100N_LT_NOISE_RATE) > 1e-12:
        raise RuntimeError("CIFAR-100N LT noise rate differs from the previously audited 35.9730801%")
    results["cifar100n_real"] = write_regime(
        args.output_root, "cifar100n_real", real_clean, real_observed, indices,
        source="official CIFAR-100N human fine noisy_label",
        indices_sha256=indices_sha,
        requested_noise_rate=None,
        source_sha256={"human_labels_npz": human_sha, "train": train_sha, "indices_file": indices_sha},
    )

    summary = {
        "seed": args.seed,
        "num_samples": N,
        "num_classes": K,
        "indices_file_sha256": indices_sha,
        "legacy_train_id_set_matches": True,
        "legacy_validation_count": int(len(legacy_val_ids)),
        "official_human_clean_labels_match_all_50000": True,
        "regimes": {
            name: {
                "observed_label_sha256": value["manifest"]["observed_labels_sha256"],
                "observed_class_min": min(value["manifest"]["observed_class_counts"]),
                "observed_class_max": max(value["manifest"]["observed_class_counts"]),
                "realized_noise_rate_offline": value["truth"]["realized_noise_rate"],
            }
            for name, value in results.items()
        },
        "leakage_policy": "Only each regime's observed_labels.npy may be read by the trainer. truth/ is opened after predictions are sealed.",
    }
    write_json(args.output_root / "DATA_PREPARATION_AUDIT.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
