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
import os
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


def discover_imagenet_shards(root: Path, split: str) -> List[Path]:
    """Collect only paths belonging to the requested split."""
    paths = []
    for folder, directories, names in os.walk(root):
        directories[:] = [
            name for name in directories
            if name.lower() not in EXCLUDED_DIRS
            and classify_parquet_split(Path(name)) in (None, split)
        ]
        for name in names:
            if name.lower().endswith(".parquet"):
                path = Path(folder) / name
                if classify_parquet_split(path) == split:
                    paths.append(path)
    return sorted(paths)


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
        if len(parts) < 4 or parts[0] != "data_256" or parts[1] != parts[2][0]:
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


def resolve_places_root(root: Path) -> Path:
    roots = [p for p in (root, root / "places-lt") if (p / "train").is_dir()]
    if len(roots) != 1:
        raise ValueError(f"Expected one extracted Places-LT train directory under {root}, found {len(roots)}")
    return roots[0]


def imagenet_wnid_map(train_entries: Sequence[Tuple[str, int]]) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for rel, label in train_entries:
        wnid = PurePosixPath(rel).parent.name
        if not re.fullmatch(r"n\d{8}", wnid) or mapping.setdefault(wnid, label) != label:
            raise ValueError(f"Invalid official ImageNet-LT WNID mapping at {rel}")
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("Multiple official ImageNet-LT WNIDs map to the same label")
    return mapping


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
            if split == "test" and (len(parts) != 2 or parts[0] != "val_256"):
                raise ValueError("Unexpected Places-LT official test manifest path: %r" % rel)
            candidate = (self.root / split_dir / folder / parts[-1]).resolve()
            if not candidate.is_relative_to(self.root / split_dir):
                raise ValueError(f"Places-LT {split} image escapes its split: {candidate}")
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
        shard_paths: Optional[Sequence[Path]] = None,
        wnid_to_id: Optional[Dict[str, int]] = None,
        official_entries: Optional[Sequence[Tuple[str, int]]] = None,
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
        self.files = (discover_imagenet_shards(self.root, split) if shard_paths is None
                      else sorted(Path(path).resolve() for path in shard_paths))
        if len(set(self.files)) != len(self.files) or any(
            not path.is_relative_to(self.root) or classify_parquet_split(path) != split for path in self.files
        ):
            raise ValueError(f"ImageNet-LT {split} received duplicate or out-of-split shard paths")
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
        physical_rows: Dict[Tuple[str, int], int] = {}
        cursor = 0

        for shard in self.files:
            with pq.ParquetFile(shard) as parquet_file:
                names = parquet_file.schema_arrow.names
                if self.label_column is None:
                    self.label_column = next(
                        (name for name in ("label", "labels", "target", "fine_label", "class", "category_id")
                         if name in names), None
                    )
                    if self.image_column is None:
                        self.image_column = next(
                            (name for name in ("image", "images", "img", "pixel_values") if name in names), None
                        )
                if self.label_column not in names or self.image_column not in names:
                    raise ValueError(f"Unrecognized ImageNet-LT Parquet schema in {shard}: {names}")
                columns = [self.label_column]
                if official_entries is not None:
                    # Project the filename only. The image byte column stays unopened.
                    columns.append(f"{self.image_column}.path")
                for rg_index in range(parquet_file.metadata.num_row_groups):
                    row_count = int(parquet_file.metadata.row_group(rg_index).num_rows)
                    if row_count == 0:
                        continue
                    self._row_group_starts.append(cursor)
                    self.row_groups.append(_RowGroup(str(shard), rg_index, cursor, row_count))
                    table = parquet_file.read_row_group(rg_index, columns=columns, use_threads=False)
                    values = table[self.label_column].to_pylist()
                    if len(values) != row_count:
                        raise RuntimeError(f"ImageNet-LT row-group label count mismatch in {shard}")
                    filenames = None
                    if official_entries is not None:
                        if self.image_column not in table.column_names:
                            raise ValueError(f"ImageNet-LT manifest alignment needs the image.path field in {shard}")
                        filenames = table[self.image_column].to_pylist()
                    for offset, value in enumerate(values):
                        label = resolve_label(value, wnid_to_id)
                        if label is None or not 0 <= label < self.num_classes:
                            raise ValueError(f"Invalid ImageNet-LT label in {shard}: {value!r}")
                        labels.append(label)
                        if filenames is not None:
                            item = filenames[offset]
                            if not item or not item.get("path"):
                                raise ValueError(f"Missing ImageNet-LT filename in {shard}, row group {rg_index}")
                            key = (PurePosixPath(normalize_relpath(item["path"])).name, label)
                            if key in physical_rows:
                                raise ValueError(f"Duplicate ImageNet-LT filename/label in Parquet: {key}")
                            physical_rows[key] = cursor + offset
                    cursor += row_count

        labels = np.asarray(labels, dtype=np.int64)
        self._physical_indices = None
        if official_entries is not None:
            ordered = [(PurePosixPath(path).name, label) for path, label in official_entries]
            if len(ordered) != len(set(ordered)) or set(ordered) != physical_rows.keys():
                missing = next((key for key in ordered if key not in physical_rows), None)
                raise ValueError(f"ImageNet-LT {split} Parquet does not match official manifest membership; missing example {missing}")
            self._physical_indices = np.fromiter((physical_rows[key] for key in ordered), dtype=np.int64)
            labels = labels[self._physical_indices]
        self.labels = labels
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

    def close(self) -> None:
        """Release Arrow file handles, including on Windows."""
        self._row_group_cache.clear()
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

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
        payloads = table.column(0)
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
        physical = index if self._physical_indices is None else int(self._physical_indices[index])
        record_index = bisect.bisect_right(self._row_group_starts, physical) - 1
        record = self.row_groups[record_index]
        payload = self._get_row_group_payloads(record)[physical - record.start].as_py()
        image_bytes, local_path = image_payload_to_bytes_or_path(payload)
        if image_bytes is not None:
            with Image.open(BytesIO(image_bytes)) as image:
                image = image.convert("RGB")
        elif local_path:
            candidate = Path(local_path)
            if not candidate.is_absolute():
                candidate = self.root / candidate
            candidate = candidate.resolve()
            if not candidate.is_relative_to(self.root):
                raise ValueError(f"ImageNet-LT image path escapes dataset root: {candidate}")
            if self.split == "train" and any(
                part.lower() in {"test", "val", "valid", "validation"}
                for part in candidate.relative_to(self.root).parts
            ):
                raise ValueError(f"Out-of-split ImageNet-LT train image path: {candidate}")
            with Image.open(candidate) as image:
                image = image.convert("RGB")
        else:
            raise ValueError("ImageNet-LT Parquet row has neither image bytes nor a local path")

        if self.transform is not None:
            image = self.transform(image)
        return image, int(self.labels[index])
