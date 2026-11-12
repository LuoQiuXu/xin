"""Dataset adapters for the accepted ImageNet-LT Parquet and Places-LT manifests.

The caller must choose an explicit split. Training code should construct only
split='train'. The official test split is intentionally not loaded by preflight,
training, or Smoke code.
"""
from __future__ import annotations

import bisect
from collections import OrderedDict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
EXCLUDED_DIRS = {".git", ".cache", "__pycache__", ".ipynb_checkpoints"}
IMAGENET_EXPECTED = {
    "train": {"rows": 115846, "classes": 1000, "shards": 25},
    "validation": {"rows": 20000, "classes": 1000, "shards": 5},
    "test": {"rows": 50000, "classes": 1000, "shards": 15},
}
PLACES_EXPECTED = {
    "train": {"rows": 62500, "classes": 365, "split_dir": "train"},
    "test": {"rows": 36500, "classes": 365, "split_dir": "val"},
}


def normalize_relpath(value: str) -> str:
    value = str(value).strip().replace("\\", "/")
    while value.startswith("./"):
        value = value[2:]
    return str(PurePosixPath(value)).lstrip("/")


def classify_parquet_split(path: Path) -> Optional[str]:
    match = re.match(r"^(train|validation|valid|val|test)(?:[-_.]|$)", path.name.lower())
    if match:
        raw = match.group(1)
    else:
        raw = path.parent.name.lower()
        if raw not in {"train", "validation", "valid", "val", "test"}:
            return None
    return "validation" if raw in {"valid", "val"} else raw


def resolve_label(value, wnid_to_id: Optional[Dict[str, int]] = None) -> Optional[int]:
    """Mirror the accepted audit's label resolver for HF Parquet rows."""
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if hasattr(value, "as_py"):
        value = value.as_py()
    if isinstance(value, dict):
        for key in ("id", "label", "class_id", "category_id"):
            if key in value:
                result = resolve_label(value[key], wnid_to_id)
                if result is not None:
                    return result
        return None
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    if wnid_to_id and text in wnid_to_id:
        return int(wnid_to_id[text])
    if wnid_to_id:
        match = re.search(r"(n\d{8})", text)
        if match and match.group(1) in wnid_to_id:
            return int(wnid_to_id[match.group(1)])
    match = re.match(r"^(\d+)\s+", text)
    if match:
        return int(match.group(1))
    return None


def image_payload_to_bytes_or_path(value) -> Tuple[Optional[bytes], Optional[str]]:
    if value is None:
        return None, None
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value), None
    if isinstance(value, dict):
        raw = value.get("bytes")
        path = value.get("path")
        if raw is not None:
            return bytes(raw), str(path) if path else None
        return None, str(path) if path else None
    return None, None


def parse_annotation_file(path: Path) -> List[Tuple[str, int]]:
    entries: List[Tuple[str, int]] = []
    with Path(path).open("r", encoding="utf-8-sig", errors="strict") as handle:
        for line_no, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.rsplit(maxsplit=1)
            if len(parts) != 2:
                raise ValueError("Malformed annotation at %s:%d: %r" % (path, line_no, line))
            try:
                label = int(parts[1])
            except ValueError as exc:
                raise ValueError("Non-integer label at %s:%d" % (path, line_no)) from exc
            entries.append((normalize_relpath(parts[0]), label))
    return entries


def places_class_folders(train_entries: Sequence[Tuple[str, int]]) -> Dict[int, str]:
    folders: Dict[int, str] = {}
    for rel, label in train_entries:
        parts = PurePosixPath(rel).parts
        if len(parts) < 4 or parts[0] != "data_256" or parts[1] != "train":
            raise ValueError("Unexpected official Places-LT train path: %r" % rel)
        folder = "-".join(parts[2:-1])
        if not folder:
            raise ValueError("Empty Places-LT class folder for label %d" % label)
        previous = folders.setdefault(int(label), folder)
        if previous != folder:
            raise ValueError("Places-LT label %d maps to multiple class folders" % label)
    if len(set(folders.values())) != len(folders):
        raise ValueError("Multiple Places-LT labels map to the same class folder")
    if set(folders) != set(range(365)):
        raise ValueError("Places-LT train manifest must cover original labels 0..364")
    return folders


class PlacesLTManifestDataset(Dataset):
    """Places-LT image dataset driven exclusively by the official TXT manifest."""

    def __init__(
        self,
        root: Path,
        manifest_path: Path,
        split: str,
        transform=None,
        class_folders: Optional[Dict[int, str]] = None,
        expected_rows: Optional[int] = None,
        verify_files: bool = True,
    ):
        if split not in ("train", "test"):
            raise ValueError("Places-LT split must be 'train' or 'test'; there is no validation split")
        self.root = Path(root).resolve()
        self.manifest_path = Path(manifest_path).resolve()
        self.split = split
        self.transform = transform
        self.entries = parse_annotation_file(self.manifest_path)

        if split == "train":
            self.class_folders = places_class_folders(self.entries)
        else:
            if class_folders is None:
                raise ValueError("Places-LT test loading requires class_folders derived from the official train manifest")
            self.class_folders = dict(class_folders)

        self.paths: List[Path] = []
        self.targets: List[int] = []
        seen = set()
        split_dir = PLACES_EXPECTED[split]["split_dir"]
        for rel, label in self.entries:
            if label < 0 or label >= 365:
                raise ValueError("Places-LT label outside [0,364]: %d" % label)
            parts = PurePosixPath(rel).parts
            folder = self.class_folders.get(int(label))
            if folder is None:
                raise ValueError("No official train-derived class folder for Places-LT label %d" % label)
            if split == "train":
                if len(parts) < 4 or parts[0] != "data_256" or parts[1] != "train":
                    raise ValueError("Unexpected Places-LT train manifest path: %r" % rel)
            else:
                if len(parts) != 2 or parts[0] != "val_256":
                    raise ValueError("Unexpected Places-LT official test manifest path: %r" % rel)
            candidate = (self.root / split_dir / folder / parts[-1]).resolve()
            if candidate in seen:
                raise ValueError("Duplicate Places-LT manifest rows resolve to the same image: %s" % candidate)
            seen.add(candidate)
            if verify_files and not candidate.is_file():
                raise FileNotFoundError("Official Places-LT %s image is missing: %s" % (split, candidate))
            self.paths.append(candidate)
            self.targets.append(int(label))

        if expected_rows is None:
            expected_rows = PLACES_EXPECTED[split]["rows"]
        if len(self.targets) != int(expected_rows):
            raise ValueError("Places-LT %s row count expected %d, observed %d" %
                             (split, int(expected_rows), len(self.targets)))
        self.targets_array = np.asarray(self.targets, dtype=np.int64)
        self.class_counts = np.bincount(self.targets_array, minlength=365).astype(np.int64)
        if self.class_counts.shape != (365,) or np.any(self.class_counts <= 0):
            raise ValueError("Places-LT %s manifest must contain all 365 classes" % split)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        path = self.paths[int(index)]
        with Image.open(path) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(self.targets[index])


@dataclass(frozen=True)
class _RowGroup:
    path: str
    row_group: int
    start: int
    rows: int


class ImageNetLTParquetDataset(Dataset):
    """Map-style ImageNet-LT reader with lazy row-group image payload caching.

    It scans labels only at construction. Image payloads are read lazily from the
    requested row group, so all Parquet image bytes are not materialized in RAM.
    """

    def __init__(
        self,
        root: Path,
        split: str = "train",
        transform=None,
        expected_rows: Optional[int] = None,
        expected_shards: Optional[int] = None,
        num_classes: int = 1000,
        row_group_cache_size: int = 4,
    ):
        if split not in ("train", "validation", "test"):
            raise ValueError("ImageNet-LT split must be train, validation, or test")
        if row_group_cache_size < 1:
            raise ValueError("row_group_cache_size must be positive")
        try:
            import pyarrow.parquet as pq
        except Exception as exc:
            raise RuntimeError("pyarrow is required for ImageNet-LT Parquet loading") from exc

        self.root = Path(root).resolve()
        self.split = split
        self.transform = transform
        self.num_classes = int(num_classes)
        self.row_group_cache_size = int(row_group_cache_size)
        all_files = sorted(
            path for path in self.root.rglob("*.parquet")
            if not any(part in EXCLUDED_DIRS for part in path.parts)
        )
        self.files = [path for path in all_files if classify_parquet_split(path) == split]
        if not self.files:
            raise FileNotFoundError("No ImageNet-LT Parquet shards found for split %s under %s" %
                                    (split, self.root))
        if expected_shards is None:
            expected_shards = IMAGENET_EXPECTED[split]["shards"]
        if len(self.files) != int(expected_shards):
            raise ValueError("ImageNet-LT %s shard count expected %d, observed %d" %
                             (split, int(expected_shards), len(self.files)))

        self.label_column: Optional[str] = None
        self.image_column: Optional[str] = None
        self.row_groups: List[_RowGroup] = []
        self._row_group_starts: List[int] = []
        labels: List[int] = []
        cursor = 0

        for shard in self.files:
            parquet_file = pq.ParquetFile(str(shard))
            names = list(parquet_file.schema_arrow.names)
            if self.label_column is None:
                self.label_column = next(
                    (name for name in ("label", "labels", "target", "fine_label", "class", "category_id")
                     if name in names), None
                )
            if self.image_column is None:
                self.image_column = next(
                    (name for name in ("image", "images", "img", "pixel_values") if name in names), None
                )
            if self.label_column is None or self.image_column is None:
                raise ValueError("Unrecognized ImageNet-LT Parquet schema in %s: %s" % (shard, names))

            for rg_index in range(parquet_file.metadata.num_row_groups):
                row_count = int(parquet_file.metadata.row_group(rg_index).num_rows)
                self._row_group_starts.append(cursor)
                self.row_groups.append(_RowGroup(str(shard), rg_index, cursor, row_count))
                table = parquet_file.read_row_group(rg_index, columns=[self.label_column], use_threads=False)
                values = table.column(0).to_pylist()
                if len(values) != row_count:
                    raise RuntimeError("ImageNet-LT row-group label count mismatch in %s" % shard)
                for value in values:
                    label = resolve_label(value)
                    if label is None:
                        raise ValueError("Null or unrecognized ImageNet-LT label in %s" % shard)
                    if label < 0 or label >= self.num_classes:
                        raise ValueError("ImageNet-LT label outside [0,%d): %d" % (self.num_classes, label))
                    labels.append(label)
                cursor += row_count

        self.labels = np.asarray(labels, dtype=np.int64)
        if len(self.labels) != cursor:
            raise RuntimeError("ImageNet-LT total label count does not match row-group metadata")
        if expected_rows is None:
            expected_rows = IMAGENET_EXPECTED[split]["rows"]
        if len(self.labels) != int(expected_rows):
            raise ValueError("ImageNet-LT %s rows expected %d, observed %d" %
                             (split, int(expected_rows), len(self.labels)))
        self.class_counts = np.bincount(self.labels, minlength=self.num_classes).astype(np.int64)
        if self.class_counts.shape != (self.num_classes,) or np.any(self.class_counts <= 0):
            raise ValueError("ImageNet-LT %s must contain all %d classes" % (split, self.num_classes))

        self._handles = {}
        self._row_group_cache: OrderedDict = OrderedDict()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"] = {}
        state["_row_group_cache"] = OrderedDict()
        return state

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def _get_row_group_payloads(self, record: _RowGroup):
        import pyarrow.parquet as pq

        key = (record.path, record.row_group)
        cached = self._row_group_cache.get(key)
        if cached is not None:
            self._row_group_cache.move_to_end(key)
            return cached

        handle = self._handles.get(record.path)
        if handle is None:
            handle = pq.ParquetFile(record.path)
            self._handles[record.path] = handle
        table = handle.read_row_group(
            record.row_group, columns=[self.image_column], use_threads=False
        )
        payloads = table.column(0).to_pylist()
        if len(payloads) != record.rows:
            raise RuntimeError("ImageNet-LT image row-group length mismatch: %s rg=%d" %
                               (record.path, record.row_group))
        self._row_group_cache[key] = payloads
        self._row_group_cache.move_to_end(key)
        while len(self._row_group_cache) > self.row_group_cache_size:
            self._row_group_cache.popitem(last=False)
        return payloads

    def __getitem__(self, index: int):
        index = int(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        record_index = bisect.bisect_right(self._row_group_starts, index) - 1
        record = self.row_groups[record_index]
        local_index = index - record.start
        payload = self._get_row_group_payloads(record)[local_index]
        image_bytes, local_path = image_payload_to_bytes_or_path(payload)
        if image_bytes is not None:
            with Image.open(BytesIO(image_bytes)) as image:
                image = image.convert("RGB")
        elif local_path:
            candidate = Path(local_path)
            if not candidate.is_absolute():
                candidate = self.root / candidate
            with Image.open(candidate) as image:
                image = image.convert("RGB")
        else:
            raise ValueError("ImageNet-LT Parquet row has neither image bytes nor a local path")

        if self.transform is not None:
            image = self.transform(image)
        return image, int(self.labels[index])
