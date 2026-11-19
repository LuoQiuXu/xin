"""Byte-range-restricted holdout reader and tamper-evident mask sealing."""
from __future__ import annotations

from contextlib import contextmanager
import contextvars
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from causal_core import canonical_hash

_PERMITTED_OPEN = contextvars.ContextVar("oracle_holdout_permitted_open", default=False)
_PERMITTED_ROWS = contextvars.ContextVar("oracle_holdout_permitted_rows", default=None)


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def _reader_permission():
    token = _PERMITTED_OPEN.set(True)
    try:
        yield
    finally:
        _PERMITTED_OPEN.reset(token)


def read_npy_rows(path, positions):
    """Read header and ONLY specified C-order rows; never mmap the holdout."""
    positions = np.asarray(positions, dtype=np.int64)
    permission = _PERMITTED_ROWS.get()
    if permission != (str(Path(path).resolve()), tuple(positions.tolist())):
        raise RuntimeError("Row-reader request lacks partition authorization")
    if positions.ndim != 1 or len(np.unique(positions)) != len(positions):
        raise ValueError("Unique 1-D row indices required")
    with _reader_permission(), Path(path).open("rb", buffering=0) as handle:
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(handle)
        else:
            raise ValueError(f"Unsupported NPY version {version}")
        if fortran or dtype.hasobject or not shape:
            raise ValueError("Only non-object C-order row arrays are supported")
        if np.any(positions < 0) or np.any(positions >= shape[0]):
            raise IndexError("Row outside NPY array")
        offset = handle.tell()
        row_bytes = int(np.prod(shape[1:], dtype=np.int64)) * dtype.itemsize
        result = np.empty((len(positions), *shape[1:]), dtype=dtype)
        for out_row, position in enumerate(positions):
            handle.seek(offset + int(position) * row_bytes)
            block = handle.read(row_bytes)
            if len(block) != row_bytes:
                raise EOFError("Truncated prepared array")
            result[out_row] = np.frombuffer(block, dtype=dtype).reshape(shape[1:])
    return result


class HoldoutVault:
    """Evaluation is inaccessible until all masks and every dependency verify.

    Legacy full-array loaders are blocked by a process audit hook, including
    np.load and memory mapping.  Only this row reader receives temporary file
    access.  The public interface accepts partition names, not arbitrary rows.
    """
    def __init__(self, prepared, split, seal_path):
        self.prepared = Path(prepared).resolve()
        self.split = split
        self.seal_path = Path(seal_path)
        self.events = []
        self._sealed = None
        self._protected = {
            str((self.prepared / name).resolve()).casefold()
            for name in ("validation_gap_features.npy", "validation_labels.npy", "validation_images.npy")
        }
        self._active = True
        def audit(event, args):
            if not self._active or event != "open" or not args or not isinstance(args[0], (str, bytes, Path)):
                return
            try:
                path = str(Path(args[0]).resolve()).casefold()
            except (TypeError, ValueError, OSError):
                return
            if path in self._protected and not _PERMITTED_OPEN.get():
                self.events.append({"partition": "BLOCKED", "path": path})
                raise RuntimeError("Holdout must be accessed through the isolated row reader")
        sys.addaudithook(audit)

    def close(self):
        self._active = False

    def verify_seal(self):
        if not self.seal_path.is_file():
            raise RuntimeError("Evaluation remains locked: no mask seal")
        record = json.loads(self.seal_path.read_text(encoding="utf-8"))
        payload = record["payload"]
        if record.get("payload_sha256") != canonical_hash(payload):
            raise RuntimeError("Mask seal payload was changed")
        if payload.get("split_sha256") != canonical_hash(self.split):
            raise RuntimeError("Mask seal belongs to a different split")
        if payload.get("seeds") != [123, 124, 125]:
            raise RuntimeError("All three seeds must be sealed together")
        if not payload.get("evaluation_unopened_before_seal"):
            raise RuntimeError("Seal lacks isolation certification")
        for filename, expected in payload["dependencies"].items():
            if file_sha(filename) != expected:
                raise RuntimeError(f"Sealed dependency changed: {filename}")
        self._sealed = record["payload_sha256"]
        return payload

    def read(self, partition):
        if partition not in ("calibration", "evaluation"):
            raise ValueError("Unknown partition")
        if partition == "evaluation":
            self.verify_seal()
        positions = self.split[f"{partition}_positions"]
        loaded = []
        for name in ("validation_gap_features.npy", "validation_labels.npy"):
            path = self.prepared / name
            token = _PERMITTED_ROWS.set((str(path.resolve()), tuple(positions)))
            try:
                loaded.append(read_npy_rows(path, positions))
            finally:
                _PERMITTED_ROWS.reset(token)
        features, labels = loaded
        if features.dtype != np.float32 or labels.dtype != np.int64:
            raise RuntimeError("Prepared holdout dtypes changed")
        if not np.isfinite(features).all():
            raise RuntimeError("Nonfinite prepared features")
        expected = np.asarray(self.split["allocation_classes"], dtype=np.int64)[positions]
        if not np.array_equal(labels, expected):
            raise RuntimeError(f"{partition} labels differ from the pre-existing allocation order")
        count = self.split["per_class_per_half"]
        if not all(np.count_nonzero(labels == cls) == count for cls in np.unique(expected)):
            raise RuntimeError("Incorrect stratified partition sizes")
        self.events.append({"partition": partition, "positions": positions,
                            "feature_shape": list(features.shape),
                            "mask_seal_sha256": self._sealed})
        return features, labels

    def sealed_integrity_hash(self, name):
        self.verify_seal()
        path = self.prepared / name
        if str(path.resolve()).casefold() not in self._protected:
            raise ValueError("Not a protected holdout cache")
        with _reader_permission():
            result = file_sha(path)
        return result


def seal_masks(path, payload):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite mask seal: {path}")
    if payload.get("seeds") != [123, 124, 125] or not payload.get("evaluation_unopened_before_seal"):
        raise ValueError("Incomplete all-seed mask seal")
    for filename, expected in payload["dependencies"].items():
        if file_sha(filename) != expected:
            raise RuntimeError(f"Invalid mask dependency before sealing: {filename}")
    record = {"payload": payload, "payload_sha256": canonical_hash(payload)}
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return record
