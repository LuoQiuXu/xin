"""Pinned MDLT OfficeHome split, metadata-only audit and guarded image loading."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

MDLT_COMMIT = "2efbfefd34542e365293f798d79f70cee5e54303"
CSV_URL = f"https://raw.githubusercontent.com/YyzHarry/multi-domain-imbalance/{MDLT_COMMIT}/mdlt/dataset/split/OfficeHome.csv"
CSV_SHA256 = "0a6c4e7800f629b8c6b67ba71dd238a83d4f513ac77694ab58796e528985a5b5"
DOMAINS = ("Art", "Clipart", "Product", "Real World")
ARMS = ("DR-NoFusion", "H2T-Global", "H2T-DomainMatched", "H2T-CrossDomain")
SEEDS = (42, 43, 44)
N_CLASSES = 65


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows, fields=None):
    rows = list(rows)
    names = fields or (list(rows[0]) if rows else [])
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


@dataclass(frozen=True)
class Row:
    index: int
    domain: int
    label: int
    path: str
    split: str


def validate_rows(rows):
    seen, names = {}, {}
    for r in rows:
        p = PurePosixPath(r.path)
        if p.is_absolute() or ".." in p.parts or "\\" in r.path or len(p.parts) != 3:
            raise ValueError(f"Unsafe or unexpected relative image path: {r.path}")
        if not 0 <= r.domain < 4 or p.parts[0] != DOMAINS[r.domain]:
            raise ValueError(f"Domain/path mismatch: {r}")
        if not 0 <= r.label < N_CLASSES or r.split not in ("train", "val", "test"):
            raise ValueError(f"Invalid label or split: {r}")
        key = r.path.casefold()
        if key in seen:
            raise ValueError(f"Duplicate/overlapping path: {r.path}, {seen[key]} / {r.split}")
        seen[key] = r.split
        if r.label in names and names[r.label] != p.parts[1]:
            raise ValueError(f"Inconsistent class folder for label {r.label}")
        names[r.label] = p.parts[1]
    if len(set(names.values())) != len(names):
        raise ValueError("Different labels map to the same class directory")


def read_protocol(csv_path):
    if sha256(csv_path) != CSV_SHA256:
        raise ValueError("CSV is not the pinned official OfficeHome split")
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as handle:
        table = csv.DictReader(handle)
        if table.fieldnames != ["env", "label", "path", "split"]:
            raise ValueError(f"Unexpected CSV schema: {table.fieldnames}")
        rows = [Row(i, DOMAINS.index(r["env"]), int(r["label"]), r["path"], r["split"])
                for i, r in enumerate(table)]
    validate_rows(rows)
    return rows


def split_rows(rows, split):
    return [r for r in rows if r.split == split]


def cell_counts(rows, split="train"):
    counts = np.zeros((4, N_CLASSES), dtype=np.int64)
    for r in rows:
        if r.split == split:
            counts[r.domain, r.label] += 1
    return counts


def audit_dataset(rows, root, csv_path, out):
    """Only metadata and filesystem stat calls; never Image.open or test prediction."""
    root, out = Path(root).resolve(), Path(out)
    counts = {s: cell_counts(rows, s) for s in ("train", "val", "test")}
    missing, resolved_seen = [], set()
    for r in rows:
        path = (root / r.path).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Image escapes root: {r.path}")
        key = str(path).casefold()
        if key in resolved_seen:
            raise ValueError(f"Resolved path alias/overlap: {r.path}")
        resolved_seen.add(key)
        if not path.is_file():
            missing.append({"raw_index": r.index, "split": r.split, "path": r.path})
    cells = [{"domain": DOMAINS[d], "class_id": c,
              "class_name": next(PurePosixPath(r.path).parts[1] for r in rows if r.label == c),
              "train": int(counts["train"][d, c]), "val": int(counts["val"][d, c]),
              "test": int(counts["test"][d, c]),
              "shot": shot(int(counts["train"][d, c]))}
             for d in range(4) for c in range(N_CLASSES)]
    write_csv(out / "domain_class_counts.csv", cells)
    write_csv(out / "per_domain_counts.csv", [
        {"domain": DOMAINS[d], **{s: int(a[d].sum()) for s, a in counts.items()}} for d in range(4)])
    write_csv(out / "class_counts.csv", [
        {"class_id": c, **{s: int(a[:, c].sum()) for s, a in counts.items()}} for c in range(N_CLASSES)])
    write_csv(out / "missing_paths.csv", missing, ["raw_index", "split", "path"])
    totals = counts["train"].sum(0)
    audit = {
        "csv_path": str(Path(csv_path).resolve()), "csv_url": CSV_URL, "csv_sha256": sha256(csv_path),
        "mdlt_commit": MDLT_COMMIT, "image_root": str(root), "total_rows": len(rows),
        "split_counts": dict(Counter(r.split for r in rows)),
        "domain_counts_train": {DOMAINS[d]: int(counts["train"][d].sum()) for d in range(4)},
        "global_class_train_min": int(totals.min()), "global_class_train_max": int(totals.max()),
        "global_train_min_over_max": float(totals.min() / totals.max()),
        "zero_train_cells": int((counts["train"] == 0).sum()),
        "domain_class_group_counts": {s: sum(r["shot"] == s for r in cells) for s in ("Many", "Medium", "Few")},
        "val_count_per_cell": sorted(np.unique(counts["val"]).tolist()),
        "test_count_per_cell_metadata_only": sorted(np.unique(counts["test"]).tolist()),
        "duplicate_paths": 0, "split_path_intersections": 0, "resolved_path_aliases": 0,
        "missing_image_count": len(missing), "present_image_count": len(rows) - len(missing),
        "missing_domain_directories": [str(root / d) for d in DOMAINS if not (root / d).is_dir()],
        "ready": not missing and bool((totals > 0).all()) and bool((counts["val"] > 0).all()),
        "audit_decoded_images": 0, "test_pixels_read": False,
        "all_domains_in_all_splits": all((a.sum(1) > 0).all() for a in counts.values()),
    }
    write_json(out / "dataset_audit.json", audit)
    return audit


def shot(n):
    return "Many" if n >= 60 else "Medium" if n >= 20 else "Few"


def image_transform(augment):
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if augment:
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(), transforms.ToTensor(), norm])
    return transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), norm])


def validate_test_lock(lock_path):
    """All 12 epoch-10 classifiers must already exist with their sealed hashes."""
    if lock_path is None:
        raise RuntimeError("TEST LOCKED: all 12 final checkpoints must be sealed first")
    path = Path(lock_path).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    expected = {(s, a) for s in SEEDS for a in ARMS}
    entries = data["checkpoints"]
    if len(entries) != len(expected) or {(e["seed"], e["arm"]) for e in entries} != expected:
        raise RuntimeError("TEST LOCKED: incomplete/duplicated seed x arm manifest")
    for e in entries:
        checkpoint = (path.parent / e["path"]).resolve()
        if not checkpoint.is_relative_to(path.parent) or e["epoch"] != 10:
            raise RuntimeError("TEST LOCKED: invalid final checkpoint record")
        if not checkpoint.is_file() or sha256(checkpoint) != e["sha256"]:
            raise RuntimeError("TEST LOCKED: missing or changed final checkpoint")
    if data["csv_sha256"] != CSV_SHA256:
        raise RuntimeError("TEST LOCKED: wrong split provenance")
    return data


class OfficeHomeImages(Dataset):
    def __init__(self, root, rows, split, augment=False, test_lock=None):
        if split == "test":
            validate_test_lock(test_lock)
        if split not in ("train", "val", "test") or any(r.split != split for r in rows):
            raise ValueError("Mixed or unexpected image split")
        if augment and split != "train":
            raise ValueError("Only train images may use augmentation")
        self.root, self.rows = Path(root), list(rows)
        self.transform = image_transform(augment)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        r = self.rows[index]
        with Image.open(self.root / r.path) as image:
            x = self.transform(image.convert("RGB"))
        return x, r.label, r.domain, r.index


def prediction_metrics(rows, predictions, train_counts):
    predictions = np.asarray(predictions, dtype=np.int64)
    if predictions.shape != (len(rows),) or ((predictions < 0) | (predictions >= N_CLASSES)).any():
        raise ValueError("Prediction shape or class range invalid")
    labels = np.array([r.label for r in rows])
    domains = np.array([r.domain for r in rows])
    correct = predictions == labels
    cell_table = []
    for d in range(4):
        for c in range(N_CLASSES):
            mask = (domains == d) & (labels == c)
            n = int(mask.sum())
            cell_table.append({"domain": DOMAINS[d], "class_id": c, "n_train": int(train_counts[d, c]),
                               "shot": shot(train_counts[d, c]), "n_eval": n,
                               "correct": int(correct[mask].sum()),
                               "accuracy": 100 * float(correct[mask].mean()) if n else None})
    available = [r for r in cell_table if r["n_eval"]]
    metrics = {"Overall": 100 * float(correct.mean()),
               "DomainClassMacro": float(np.mean([r["accuracy"] for r in available])),
               "n_images": len(rows), "n_cells": len(available)}
    for group in ("Many", "Medium", "Few"):
        group_rows = [r for r in available if r["shot"] == group]
        n = sum(r["n_eval"] for r in group_rows)
        metrics[f"DomainClass{group}"] = 100 * sum(r["correct"] for r in group_rows) / n if n else None
        metrics[f"{group}_cells"] = len(group_rows)
        metrics[f"{group}_images"] = n
    worst = min(r["accuracy"] for r in available)
    metrics["WorstCell"] = worst
    metrics["WorstCells"] = [{k: r[k] for k in ("domain", "class_id", "n_train", "n_eval")}
                             for r in available if r["accuracy"] == worst]
    per_domain = [{"domain": DOMAINS[d], "n_eval": int((domains == d).sum()),
                   "accuracy": 100 * float(correct[domains == d].mean())}
                  for d in range(4) if (domains == d).any()]
    return metrics, cell_table, per_domain


def paired_changes(rows, candidate, baseline, train_counts):
    candidate, baseline = np.asarray(candidate), np.asarray(baseline)
    labels = np.array([r.label for r in rows])
    if candidate.shape != labels.shape or baseline.shape != labels.shape:
        raise ValueError("Unpaired prediction arrays")
    cand_ok, base_ok = candidate == labels, baseline == labels
    groups = ["Overall", "Many", "Medium", "Few"]
    result = []
    for group in groups:
        mask = np.ones(len(rows), dtype=bool) if group == "Overall" else np.array(
            [shot(train_counts[r.domain, r.label]) == group for r in rows])
        n = int(mask.sum())
        gains = int((cand_ok & ~base_ok & mask).sum())
        losses = int((~cand_ok & base_ok & mask).sum())
        net = gains - losses
        if int(cand_ok[mask].sum()) - int(base_ok[mask].sum()) != net:
            raise AssertionError("Paired net disagrees with accuracy counts")
        result.append({"group": group, "n": n, "gain": gains, "loss": losses, "net": net,
                       "delta_pp": 100 * net / n if n else None})
    return result
