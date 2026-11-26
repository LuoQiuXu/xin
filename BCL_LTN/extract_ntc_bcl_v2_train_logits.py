"""Extract ordered raw TRAIN logits from a local BCL/NTC checkpoint.

Images only are loaded. No observed or clean labels and no official TEST data
are needed. Output logits align with the fixed IF=100 indices-file order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from BCL_AUDIT.cifar100lt.data import build_cifar_transforms
from BCL_AUDIT.cifar100lt.models import BCLCIFARModel
from BCL_LTN.audit_epoch160_margin_reliability import load_cifar100_train_pixels_only


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "H2T" / "data" / "cifar100"
DEFAULT_INDICES = ROOT / "BCL_AUDIT" / "outputs" / "data_audit_recheck_20261122_01" / "cifar100lt_if100_train_indices_seed0.npy"
BATCH_SIZE, K, N = 256, 100, 10_847


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class EvalDataset(Dataset):
    def __init__(self, pixels, indices, transform):
        self.pixels, self.indices, self.transform = pixels, indices, transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        raw_id = int(self.indices[position])
        return self.transform(Image.fromarray(self.pixels[raw_id])), position


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--state-key", choices=("model_state", "ema_model_state"), default="model_state")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--indices-file", type=Path, default=DEFAULT_INDICES)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    for path in (args.checkpoint, args.indices_file, args.data_root / "cifar-100-python" / "train"):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output.exists() or args.output.with_suffix(args.output.suffix + ".json").exists():
        raise FileExistsError("Refusing to overwrite logits or provenance output")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = checkpoint.get(args.state_key)
    if not isinstance(state, dict):
        raise RuntimeError(f"Checkpoint does not contain {args.state_key}")
    model = BCLCIFARModel()
    model.load_state_dict(state, strict=True)
    model.eval()

    indices = np.load(args.indices_file, allow_pickle=False)
    if (indices.shape != (N,) or indices.dtype != np.int64 or np.unique(indices).size != N
            or np.any((indices < 0) | (indices >= 50_000))):
        raise RuntimeError("Invalid fixed IF=100 indices")
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for checkpoint logit extraction")
    torch.cuda.init()
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    model.to(device).eval()

    pixels = load_cifar100_train_pixels_only(args.data_root)
    eval_transform = build_cifar_transforms()[2]
    dataset = EvalDataset(pixels, indices, eval_transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    logits = np.empty((N, K), dtype=np.float32)
    seen = np.zeros(N, dtype=np.bool_)
    for images, positions in loader:
        pos = positions.numpy()
        if seen[pos].any():
            raise RuntimeError("Duplicate TRAIN position during logit extraction")
        _, output, _, _ = model(images.to(device, non_blocking=True), compute_contrastive=False)
        if output.shape != (len(pos), K) or not bool(torch.isfinite(output).all()):
            raise RuntimeError("Invalid model logits")
        logits[pos] = output.cpu().numpy()
        seen[pos] = True
    if not seen.all():
        raise RuntimeError("Logit extraction did not cover all fixed TRAIN images")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, logits, allow_pickle=False)
    receipt = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_state_key": args.state_key,
        "checkpoint_state_sha256": checkpoint.get(
            "ema_state_sha256" if args.state_key == "ema_model_state" else "model_state_sha256"
        ),
        "indices_sha256": sha256(args.indices_file),
        "logits_sha256": sha256(args.output),
        "logits_shape": list(logits.shape),
        "logits_dtype": str(logits.dtype),
        "split": "fixed 10,847-image CIFAR-100-LT TRAIN subset; eval transform; ordered by indices-file",
        "clean_labels_loaded": False,
        "observed_labels_loaded": False,
        "official_test_opened": False,
        "precision": "fp32",
        "gpu": torch.cuda.get_device_name(device),
    }
    receipt_path = args.output.with_suffix(args.output.suffix + ".json")
    with receipt_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(receipt, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
