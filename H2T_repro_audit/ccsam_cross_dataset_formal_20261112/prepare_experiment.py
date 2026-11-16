"""Seal the two official TRAIN splits and the pinned sampling plans."""
from __future__ import annotations

import csv
import hashlib
import random
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np

from cross_dataset_data import (
    ImageNetLTParquetDataset, PlacesLTManifestDataset, discover_imagenet_shards,
    imagenet_wnid_map, parse_annotation_file, resolve_places_root,
)
from experiment_common import (
    REPO, OUTPUT_ROOT, TRAIN_MANIFESTS, array_sha256, environment, json_sha256, read_json,
    record_run, require_complete, run_log, sha256_file, write_json,
)

PREFLIGHT = OUTPUT_ROOT / "preflight"
AUDIT = REPO / "H2T_repro_audit/cross_dataset_data_audit_20261111/results"


def official_plan(source: Path, expected_sha: str, labels: np.ndarray, seed: int, epochs: int) -> np.ndarray:
    source_bytes = source.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != expected_sha:
        raise RuntimeError(f"Pinned official CC-SAM sampler SHA256 mismatch: {source}")
    rng_state = random.getstate()
    try:
        module = ModuleType("_official_ccsam_sampler")
        module.__file__ = str(source)
        exec(compile(source_bytes, str(source), "exec"), module.__dict__)
        random.seed(seed)
        sampler = module.ClassAwareSampler(SimpleNamespace(labels=labels))
        if len(sampler) != len(labels):
            raise RuntimeError("Official sampler length differs from N_train")
        plan = np.empty((epochs, len(labels)), dtype=np.int32)
        for epoch in range(epochs):
            plan[epoch] = np.fromiter(iter(sampler), dtype=np.int32)
    finally:
        random.setstate(rng_state)
    if np.any(plan < 0) or np.any(plan >= len(labels)):
        raise RuntimeError("Official sampler produced an out-of-range index")
    return plan


def train_inventory(dataset, name: str) -> dict:
    if name == "imagenet_lt":
        files = []
        for shard in dataset.files:
            info = shard.stat()
            files.append({
                "path": shard.relative_to(dataset.root).as_posix(),
                "size_bytes": info.st_size, "mtime_ns": info.st_mtime_ns,
            })
        return {"shards": files, "row_order": "official train TXT via unique filename and original label"}
    paths = [path.relative_to(dataset.root).as_posix() for path in dataset.paths]
    return {"resolved_relative_paths_sha256": json_sha256(paths), "row_order": "official train TXT"}


@contextmanager
def train_dataset(exp, name: str, transform=None, seal: dict | None = None):
    spec = exp.datasets[name]
    root = Path(spec["root"]).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Missing local dataset root: {root}")
    manifest = TRAIN_MANIFESTS[name]
    if seal is not None and sha256_file(manifest) != seal["official_train_manifest_sha256"]:
        raise RuntimeError(f"Official TRAIN manifest changed after preflight: {manifest}")

    if name == "imagenet_lt":
        entries = parse_annotation_file(manifest)
        shards = discover_imagenet_shards(root, "train")
        wnid = imagenet_wnid_map(entries)
        if len(wnid) != spec["num_classes"]:
            raise RuntimeError("Official ImageNet-LT TRAIN WNID count mismatch")
        dataset = ImageNetLTParquetDataset(
            root, "train", transform, expected_rows=spec["train_images"],
            expected_shards=spec["train_shards"], num_classes=spec["num_classes"],
            shard_paths=shards, wnid_to_id=wnid, official_entries=entries,
        )
        labels = dataset.labels
    else:
        dataset = PlacesLTManifestDataset(
            resolve_places_root(root), manifest, "train", transform,
            expected_rows=spec["train_images"],
        )
        labels = dataset.targets_array
    try:
        if seal is not None and (
            array_sha256(labels) != seal["labels_sha256"]
            or train_inventory(dataset, name) != seal["inventory"]
        ):
            raise RuntimeError(f"{name} train split no longer matches preflight")
        yield dataset, labels
    finally:
        if isinstance(dataset, ImageNetLTParquetDataset):
            dataset.close()


def accepted_audit() -> dict:
    with (AUDIT / "ARTIFACT_SHA256.csv").open(encoding="utf-8-sig", newline="") as handle:
        indexed = {row["file"]: row["sha256"] for row in csv.DictReader(handle)}
    for name in ("audit_report.md", "audit_report.json"):
        if sha256_file(AUDIT / name) != indexed[name]:
            raise RuntimeError(f"Accepted audit SHA256 mismatch for {name}")
    report = read_json(AUDIT / "audit_report.json")
    if report["overall_status"] != "pass":
        raise RuntimeError("Accepted input audit did not pass")
    return report


def seal_train_split(exp, name: str, dataset, labels: np.ndarray, audit: dict) -> dict:
    spec = exp.datasets[name]
    manifest = TRAIN_MANIFESTS[name]
    manifest_sha = sha256_file(manifest)
    reference_shas = audit["run_verification"]["input_integrity"]["after"]["manifest_hashes"]
    if manifest_sha != reference_shas[manifest.name]:
        raise RuntimeError(f"Official TRAIN manifest SHA256 differs from the audit: {manifest}")

    entries = parse_annotation_file(manifest)
    official_labels = np.fromiter((label for _, label in entries), dtype=np.int64)
    reference = audit[name]["splits"]["train"]
    counts = np.bincount(official_labels, minlength=spec["num_classes"])
    audited_counts = [reference["class_counts"][str(i)] for i in range(spec["num_classes"])]
    if not np.array_equal(labels, official_labels) or counts.tolist() != audited_counts:
        raise RuntimeError(f"{name} original TRAIN IDs or class counts differ from the reference")
    bins = [int((counts > 100).sum()), int(((counts >= 20) & (counts <= 100)).sum()), int((counts < 20).sum())]
    if bins != [spec["many_classes"], spec["medium_classes"], spec["few_classes"]]:
        raise RuntimeError(f"{name} train frequency groups differ from the frozen protocol")

    inventory = train_inventory(dataset, name)
    fingerprint = {"manifest_sha256": manifest_sha}
    if name == "imagenet_lt":
        audited = {item["name"].replace("\\", "/"): item["size_bytes"] for item in reference["files"]}
        if {item["path"]: item["size_bytes"] for item in inventory["shards"]} != audited:
            raise RuntimeError("ImageNet-LT train shards differ from the audited inventory")
        fingerprint["train_shard_sha256"] = {
            path.relative_to(dataset.root).as_posix(): sha256_file(path) for path in dataset.files
        }
    else:
        fingerprint["resolved_relative_paths_sha256"] = inventory["resolved_relative_paths_sha256"]
        if len(dataset.class_folders) != 365 or len(dataset.paths) != reference["image_paths_resolved"]:
            raise RuntimeError("Places-LT train paths differ from the audited inventory")

    return {
        "dataset": name, "root": str(dataset.root), "train_rows": len(labels),
        "train_class_counts": counts.tolist(), "group_class_counts": dict(zip(("Many", "Medium", "Few"), bins)),
        "official_train_manifest_sha256": manifest_sha, "labels_sha256": array_sha256(labels),
        "inventory": inventory, "fingerprint_inputs": fingerprint, "data_fingerprint": json_sha256(fingerprint),
    }


def prepare(exp) -> dict:
    config = exp.config("preflight")
    with record_run(PREFLIGHT, config) as status:
        env = environment()
        status["environment"] = env
        audit = accepted_audit()
        sampler = exp.protocol["official_sampler"]
        sampler_path = REPO / sampler["path"]
        if sha256_file(sampler_path) != sampler["sha256"]:
            raise RuntimeError("Pinned CC-SAM sampler does not match the frozen SHA256")

        datasets = {}
        epochs = exp.protocol["stage2"]["epochs"]
        for name in exp.datasets:
            with train_dataset(exp, name) as (dataset, labels):
                seal = seal_train_split(exp, name, dataset, labels, audit)
                seal["plans"] = {}
                for seed in exp.seeds:
                    plan = official_plan(sampler_path, sampler["sha256"], labels, seed, epochs)
                    filename = f"{name}_plan_seed{seed}.npy"
                    with (PREFLIGHT / filename).open("xb") as handle:
                        np.save(handle, plan, allow_pickle=False)
                    per_epoch_counts = [
                        np.bincount(labels[indices], minlength=len(seal["train_class_counts"])).tolist()
                        for indices in plan
                    ]
                    seal["plans"][str(seed)] = {
                        "file": filename, "sha256": sha256_file(PREFLIGHT / filename),
                        "shape": list(plan.shape), "dtype": str(plan.dtype),
                        "per_epoch_indices_sha256": [array_sha256(row) for row in plan],
                        "per_epoch_class_counts": per_epoch_counts,
                    }
                    run_log(PREFLIGHT, f"{name} seed {seed}: sealed {epochs} x {len(labels):,} official plan indices")
                datasets[name] = seal
        write_json(PREFLIGHT / "prepared_manifest.json", {
            "datasets": datasets, "source_sha256": exp.source_hashes,
            "pinned_sampler_sha256": sampler["sha256"],
            "accepted_audit_sha256": sha256_file(AUDIT / "audit_report.json"),
            "official_test_access": False, "environment": env,
        })
        status["datasets"] = list(datasets)
    return load_prepared(exp)


def load_prepared(exp) -> dict:
    require_complete(PREFLIGHT, phase="preflight", expected={"official_test_access": False})
    manifest = read_json(PREFLIGHT / "prepared_manifest.json")
    if manifest["source_sha256"] != exp.source_hashes:
        raise RuntimeError("Implementation/protocol source hashes changed after preflight")
    if manifest["pinned_sampler_sha256"] != sha256_file(REPO / exp.protocol["official_sampler"]["path"]):
        raise RuntimeError("Pinned sampler SHA256 changed after preflight")
    return manifest


def open_plan(seal: dict, seed: int, epochs: int) -> tuple[np.ndarray, dict]:
    record = seal["plans"][str(seed)]
    path = PREFLIGHT / record["file"]
    if sha256_file(path) != record["sha256"]:
        raise RuntimeError(f"Sampling plan checksum mismatch: {path}")
    plan = np.load(path, mmap_mode="r", allow_pickle=False)
    if plan.dtype != np.int32 or plan.shape != (epochs, seal["train_rows"]):
        raise RuntimeError(f"Sampling plan shape/dtype mismatch: {path}")
    return plan, record
