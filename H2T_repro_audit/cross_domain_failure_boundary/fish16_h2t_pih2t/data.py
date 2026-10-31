"""Locked manifests and deterministic, matched real-image views."""
import csv
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from preflight import MANIFEST, IMAGES, EXPECTED_SHA, sha


def load_rows():
    assert sha(MANIFEST) == EXPECTED_SHA
    with MANIFEST.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["target"] = int(Path(row["relative_path"]).parts[0].removeprefix("fish_")) - 1
        assert 0 <= row["target"] < 16
    return rows


def transform(training):
    ops = [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
           transforms.ColorJitter(.4)] if training else [transforms.Resize(256), transforms.CenterCrop(224)]
    return transforms.Compose(ops + [transforms.ToTensor(), transforms.Normalize(
        (.485, .456, .406), (.229, .224, .225))])


def safe_image_path(row):
    path = (IMAGES / row["relative_path"]).resolve()
    assert path.is_relative_to(IMAGES.resolve())
    assert not any("mask" in part.lower() for part in path.parts)
    assert int(path.parent.name.removeprefix("fish_")) in range(1, 17)
    return path


class Views(Dataset):
    def __init__(self, rows, recipients, training, epoch=0, donors=None):
        self.rows, self.recipients, self.donors = rows, recipients, donors
        self.training, self.epoch = training, epoch
        self.tx = transform(training)

    def __len__(self):
        return len(self.recipients)

    def image(self, index, position, stream):
        with Image.open(safe_image_path(self.rows[int(index)])) as image:
            image = image.convert("RGB")
            if not self.training:
                return self.tx(image)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(42 + self.epoch * 1000000 + position * 2 + stream)
                return self.tx(image)

    def __getitem__(self, position):
        index = int(self.recipients[position])
        result = {"recipient": self.image(index, position, 0), "label": self.rows[index]["target"],
                  "index": index, "position": position}
        if self.donors is not None:
            donor = int(self.donors[position])
            result["donor"] = self.image(donor, position, 1)
        return result


def worker_init(_):
    torch.set_num_threads(1)


def loader(dataset, batch=16, workers=4):
    return DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=workers,
                      pin_memory=True, drop_last=False, worker_init_fn=worker_init,
                      generator=torch.Generator().manual_seed(42))


def schedules(rows):
    train = np.array([i for i, r in enumerate(rows) if r["split_primary"] == "train"], dtype=np.int64)
    val = np.array([i for i, r in enumerate(rows) if r["split_primary"] == "val"], dtype=np.int64)
    assert len(train) == 16269 and len(val) == 5396
    rng = np.random.default_rng(42)
    stage1 = rng.permutation(train)
    sampled_classes = rng.integers(0, 16, size=len(train))
    recipients = np.empty(len(train), dtype=np.int64)
    for label in range(16):
        pool = train[[rows[int(i)]["target"] == label for i in train]]
        positions = np.flatnonzero(sampled_classes == label)
        recipients[positions] = rng.choice(pool, size=len(positions), replace=True)
    donors = rng.choice(train, size=len(train), replace=True)
    return train, val, stage1, recipients, donors
