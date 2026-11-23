from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.datasets.utils import check_integrity
from torchvision.transforms import AutoAugmentPolicy


CIFAR100_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR100_STD = (0.2023, 0.1994, 0.2010)

class CIFAR100TrainOnly(datasets.CIFAR100):
    """CIFAR-100 loader that validates and reads only the training split.

    torchvision's stock CIFAR100(train=True) integrity check iterates over
    both train_list and test_list. This subclass intentionally checks only
    train_list; the superclass then loads the TRAIN pickle and validates the
    metadata file. It never opens or hashes the TEST pickle.
    """

    def _check_integrity(self) -> bool:
        for filename, md5 in self.train_list:
            file_path = Path(self.root) / self.base_folder / filename
            if not check_integrity(str(file_path), md5):
                return False
        return True



def cifar100lt_class_counts(imb_factor: float = 0.01) -> np.ndarray:
    """Standard exponential CIFAR-LT counts in official fine-label order."""
    if abs(imb_factor - 0.01) > 1e-12:
        raise ValueError("This first BCL audit is fixed to CIFAR-100-LT IF=100 (imb_factor=0.01)")
    return np.asarray(
        [int(500 * (imb_factor ** (i / 99.0))) for i in range(100)],
        dtype=np.int64,
    )


def make_cifar100lt_indices(
    targets: np.ndarray,
    split_seed: int = 0,
    imb_factor: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact IF=100 class counts and selected original train indices."""
    targets = np.asarray(targets, dtype=np.int64)
    if targets.shape[0] != 50_000:
        raise ValueError(f"Expected 50,000 CIFAR-100 training labels, got {targets.shape[0]}")
    counts = cifar100lt_class_counts(imb_factor)
    rng = np.random.RandomState(split_seed)
    selected = []
    for class_id, count in enumerate(counts.tolist()):
        indices = np.where(targets == class_id)[0].copy()
        if indices.size != 500:
            raise RuntimeError(f"Fine class {class_id} should have 500 source images, got {indices.size}")
        rng.shuffle(indices)
        selected.extend(indices[:count].tolist())
    selected = np.asarray(selected, dtype=np.int64)
    observed = np.bincount(targets[selected], minlength=100)
    if not np.array_equal(observed, counts):
        raise RuntimeError("Selected CIFAR-100-LT class counts do not match the protocol")
    if int(counts.sum()) != 10_847 or counts[0] != 500 or counts[-1] != 5:
        raise RuntimeError("IF=100 count-vector integrity check failed")
    return counts, selected


class Cutout:
    """Cutout on a tensor, matching the CIFAR recipe's length=16 mask."""

    def __init__(self, length: int = 16):
        self.length = int(length)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 3:
            raise ValueError(f"Cutout expects [C,H,W], got {tuple(image.shape)}")
        _, height, width = image.shape
        center_y = int(torch.randint(height, (1,)).item())
        center_x = int(torch.randint(width, (1,)).item())
        half = self.length // 2
        y1 = max(0, center_y - half)
        y2 = min(height, center_y + half)
        x1 = max(0, center_x - half)
        x2 = min(width, center_x + half)
        mask = torch.ones((height, width), dtype=image.dtype, device=image.device)
        mask[y1:y2, x1:x2] = 0
        return image * mask.unsqueeze(0)


class GaussianBlur:
    """SimCLR Gaussian blur helper used by the released BCL utility code."""

    def __init__(self, sigma=(0.1, 2.0)):
        self.sigma = sigma

    def __call__(self, image: Image.Image) -> Image.Image:
        radius = random.uniform(float(self.sigma[0]), float(self.sigma[1]))
        return image.filter(ImageFilter.GaussianBlur(radius=radius))


def build_cifar_transforms():
    normalize = transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)

    # BCL CIFAR classification view: crop/flip + CIFAR AutoAugment + Cutout.
    transform_v1 = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(policy=AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
            Cutout(length=16),
            normalize,
        ]
    )

    # Two independently sampled SimAug views for the contrastive branch.
    transform_sim = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur((0.1, 2.0))], p=0.5),
            transforms.ToTensor(),
            normalize,
        ]
    )

    transform_eval = transforms.Compose(
        [
            transforms.ToTensor(),
            normalize,
        ]
    )
    return transform_v1, transform_sim, transform_eval


class CIFAR100LT(Dataset):
    """CIFAR-100-LT train subset with the same three-view contract for both groups."""

    def __init__(
        self,
        root: str | Path,
        mode: str,
        indices_file: Optional[str | Path] = None,
        split_seed: int = 0,
        download: bool = False,
    ):
        if mode not in {"lc", "bcl"}:
            raise ValueError("mode must be 'lc' or 'bcl'")
        self.mode = mode
        if download:
            raise ValueError(
                "CIFAR100LT uses a strict TRAIN-only loader and will not download "
                "the combined CIFAR archive (which also contains the TEST split). "
                "Place the extracted CIFAR-100 TRAIN and meta files locally first."
            )
        if indices_file is None:
            raise ValueError(
                "CIFAR100LT requires the exact preflight-generated IF=100 indices_file; "
                "it will not regenerate a split."
            )
        indices_path = Path(indices_file)
        if not indices_path.is_file():
            raise FileNotFoundError(
                f"Required preflight-generated IF=100 indices are missing: {indices_path}"
            )
        self.base = CIFAR100TrainOnly(
            root=str(root),
            train=True,
            download=False,
            transform=None,
        )
        self.data = self.base.data
        self.targets = np.asarray(self.base.targets, dtype=np.int64)
        self.cls_num_list = cifar100lt_class_counts()

        self.indices = np.load(indices_path).astype(np.int64)
        if self.indices.ndim != 1:
            raise RuntimeError("Provided train indices must be a one-dimensional array")
        if self.indices.size != int(self.cls_num_list.sum()):
            raise RuntimeError(
                f"Provided train indices have {self.indices.size} rows; "
                f"expected {int(self.cls_num_list.sum())}"
            )
        if np.any(self.indices < 0) or np.any(self.indices >= len(self.targets)):
            raise RuntimeError("Provided train indices contain out-of-range values")
        if np.unique(self.indices).size != self.indices.size:
            raise RuntimeError("Provided train indices contain duplicates")
        observed = np.bincount(self.targets[self.indices], minlength=100)
        if not np.array_equal(observed, self.cls_num_list):
            raise RuntimeError("Provided train indices do not match the IF=100 class counts")

        self.transform_v1, self.transform_sim, self.transform_eval = build_cifar_transforms()

    def __len__(self):
        return int(self.indices.shape[0])

    def __getitem__(self, item: int):
        original_index = int(self.indices[item])
        image = Image.fromarray(self.data[original_index])
        label = int(self.targets[original_index])

        # Both LC-only and BCL receive the same three independently augmented views.
        # The LC-only objective consumes only v1 logits, but v1/v2/v3 are all
        # concatenated through the shared backbone so BatchNorm2d sees a matched
        # three-view input distribution in both paired groups.
        view1 = self.transform_v1(image)
        view2 = self.transform_sim(image)
        view3 = self.transform_sim(image)
        return (view1, view2, view3), label


class CIFAR100BalancedTest(Dataset):
    """Balanced CIFAR-100 test set; instantiate only after both training runs finish."""

    def __init__(self, root: str | Path, download: bool = False):
        self.base = datasets.CIFAR100(
            root=str(root),
            train=False,
            download=download,
            transform=None,
        )
        _, _, transform_eval = build_cifar_transforms()
        self.data = self.base.data
        self.targets = np.asarray(self.base.targets, dtype=np.int64)
        self.transform = transform_eval

    def __len__(self):
        return int(len(self.targets))

    def __getitem__(self, index: int):
        image = Image.fromarray(self.data[index])
        return self.transform(image), int(self.targets[index])
