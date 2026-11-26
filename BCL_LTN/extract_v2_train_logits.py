"""Extract ordered, deterministic TRAIN logits from a saved BCL checkpoint.

Only CIFAR-100 TRAIN pixels and the fixed LT indices are read. This script
never opens observed labels, clean-label sidecars, corruption masks, or TEST.
The resulting logits are aligned to the exact order in the indices file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from BCL_AUDIT.cifar100lt.data import build_cifar_transforms
from BCL_AUDIT.cifar100lt.models import BCLCIFARModel
from BCL_AUDIT.cifar100lt.train import file_sha256, state_dict_sha256
from BCL_LTN.audit_epoch160_margin_reliability import (
    EvalOnlyTrainDataset,
    load_cifar100_train_pixels_only,
)
from BCL_LTN.train_noisy_bcl import safe_torch_load


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "H2T" / "data" / "cifar100"
DEFAULT_INDICES = (
    ROOT / "BCL_AUDIT" / "outputs" / "data_audit_recheck_20261122_01"
    / "cifar100lt_if100_train_indices_seed0.npy"
)
EXPECTED_INDICES_SHA256 = "be73a9441289dbadec4190c8c30a1bfc551434f17b33f6dc0ceea7fb77340ac7"
EXPECTED_TRAIN_SHA256 = "735e79b04f092ca3d2e6d07f368c0a7d70d48c48d28865950cc24454cf45129b"
N, K, BATCH_SIZE = 10_847, 100, 256


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description="Extract deterministic CIFAR-100-LT TRAIN logits")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--state-key", choices=("model_state", "ema_model_state"), default="model_state")
    parser.add_argument("--expected-checkpoint-sha256", type=str, default="")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--indices-file", type=Path, default=DEFAULT_INDICES)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    train_file = args.data_root / "cifar-100-python" / "train"
    for path in (args.checkpoint, args.indices_file, train_file):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.output.exists() or args.output.with_suffix(args.output.suffix + ".json").exists():
        raise FileExistsError("Refusing to overwrite logits or provenance output")

    indices_sha = sha256(args.indices_file)
    train_sha = sha256(train_file)
    if indices_sha != EXPECTED_INDICES_SHA256:
        raise RuntimeError("Fixed CIFAR-100-LT IF=100 indices SHA256 mismatch")
    if train_sha != EXPECTED_TRAIN_SHA256:
        raise RuntimeError("CIFAR-100 Python TRAIN source SHA256 mismatch")
    checkpoint_sha = sha256(args.checkpoint)
    if args.expected_checkpoint_sha256 and checkpoint_sha != args.expected_checkpoint_sha256.lower():
        raise RuntimeError("Checkpoint SHA256 differs from the explicitly expected v1 artifact")

    checkpoint = safe_torch_load(args.checkpoint, torch.device("cpu"))
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Checkpoint payload must be a dictionary")
    state = checkpoint.get(args.state_key)
    if not isinstance(state, dict):
        raise RuntimeError(f"Checkpoint does not contain a {args.state_key} state dictionary")
    actual_state_sha = state_dict_sha256(state)
    recorded_state_sha = checkpoint.get(
        "ema_state_sha256" if args.state_key == "ema_model_state" else "model_state_sha256"
    )
    if recorded_state_sha is not None and recorded_state_sha != actual_state_sha:
        raise RuntimeError("Checkpoint's recorded model-state SHA256 is inconsistent")

    indices = np.load(args.indices_file, allow_pickle=False)
    if (indices.shape != (N,) or indices.dtype != np.int64 or np.unique(indices).size != N
            or np.any((indices < 0) | (indices >= 50_000))):
        raise RuntimeError("Invalid fixed IF=100 TRAIN indices")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "cuda":
        torch.cuda.init()
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False

    model = BCLCIFARModel()
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    pixels = load_cifar100_train_pixels_only(args.data_root)
    eval_transform = build_cifar_transforms()[2]
    dataset = EvalOnlyTrainDataset(pixels, indices, eval_transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    logits = np.empty((N, K), dtype=np.float32)
    seen = np.zeros(N, dtype=np.bool_)
    for images, positions in loader:
        pos = positions.numpy()
        if seen[pos].any():
            raise RuntimeError("Duplicate TRAIN position during logit extraction")
        _, output, _, _ = model(images.to(device, non_blocking=True), compute_contrastive=False)
        if output.shape != (len(pos), K) or not bool(torch.isfinite(output).all()):
            raise RuntimeError("Model returned invalid TRAIN logits")
        logits[pos] = output.float().cpu().numpy()
        seen[pos] = True
    if not seen.all():
        raise RuntimeError("Logit extraction did not cover every fixed TRAIN position")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        np.save(stream, logits, allow_pickle=False)
    receipt = {
        "state": "complete",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_state_key": args.state_key,
        "checkpoint_state_sha256": actual_state_sha,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_seed": checkpoint.get("seed"),
        "checkpoint_group": checkpoint.get("group"),
        "expected_checkpoint_sha256": args.expected_checkpoint_sha256 or None,
        "indices_sha256": indices_sha,
        "train_source_sha256": train_sha,
        "logits_path": str(args.output.resolve()),
        "logits_sha256": sha256(args.output),
        "logits_shape": list(logits.shape),
        "logits_dtype": str(logits.dtype),
        "split": "fixed 10,847-image CIFAR-100-LT TRAIN subset; deterministic eval transform; indices-file order",
        "teacher_type": "frozen v1 checkpoint; not EMA unless state-key=ema_model_state",
        "clean_labels_loaded": False,
        "observed_labels_loaded": False,
        "corruption_mask_loaded": False,
        "official_test_opened": False,
        "precision": "fp32",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "batch_size": int(args.batch_size),
    }
    receipt_path = args.output.with_suffix(args.output.suffix + ".json")
    with receipt_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(receipt, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
