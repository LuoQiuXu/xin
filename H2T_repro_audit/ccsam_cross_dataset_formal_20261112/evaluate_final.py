"""Evaluate fixed checkpoints only after all 24 D0-D3 Stage-II runs are verified."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from cross_dataset_core import D0, D2
from cross_dataset_data import (
    ImageNetLTParquetDataset, PlacesLTManifestDataset, discover_imagenet_shards,
    imagenet_wnid_map, parse_annotation_file, places_class_folders, resolve_places_root,
)
from experiment_common import (
    REPO, OUTPUT_ROOT, TRAIN_MANIFESTS, make_loader, now, read_json, record_run, run_log, run_path,
    save_status, sha256_file, write_csv,
)
from extract_features import open_cache
from prepare_experiment import accepted_audit, open_plan
from train_stage1 import eval_transform, load_stage1, stage1_status
from train_stage2 import HEAD, head_status, load_head, pairing_binding

FINAL = OUTPUT_ROOT / "final_evaluation"
DISPLAY = {"imagenet_lt": "ImageNet_LT", "places_lt": "Places_LT"}
STAGE1 = "Stage-I"


def verify_all_checkpoints(exp, prepared: dict):
    """Verify completed runs and pairing before calling any test loader."""
    epochs = exp.protocol["stage2"]["epochs"]
    for name, seed in exp.jobs():
        for arm in exp.arms:
            directory = run_path("stage2", name, seed, arm)
            path = directory / "status.json"
            if not path.is_file():
                raise RuntimeError(f"Official TEST remains sealed; missing Stage-II status: {path}")
            status = read_json(path)
            if status.get("state") != "completed" or status.get("final_epoch") != epochs:
                raise RuntimeError(f"Official TEST remains sealed; Stage-II did not finish epoch {epochs}: {path}")

    models = {name: {} for name in exp.datasets}
    provenance = {}
    for name, seed in exp.jobs():
        seal = prepared["datasets"][name]
        stage1 = stage1_status(exp, name, seed, seal)
        encoder, original, _ = load_stage1(exp, name, seed, seal, stage1)
        encoder.requires_grad_(False).eval()
        _, _, cache = open_cache(exp, name, seed, seal, stage1)
        _, plan_record = open_plan(seal, seed, epochs)
        binding = pairing_binding(exp, name, seed, seal, stage1, cache, plan_record)
        classifiers = {STAGE1: original.requires_grad_(False).eval()}
        statuses = {}
        for arm in exp.arms:
            directory = run_path("stage2", name, seed, arm)
            status = head_status(directory, binding, arm, epochs)
            if (not {"epoch_metrics.csv", "sampling_trace.csv"} <= status["artifacts"].keys()
                or status["epoch_indices_sha256"] != plan_record["per_epoch_indices_sha256"]
                or status.get("frozen_encoder_and_bn_unchanged") is not True):
                raise RuntimeError(f"Incomplete sampling or frozen-state evidence in {directory}")
            if arm == D2:
                if (status.get("teacher_checkpoint_sha256") != statuses[D0]["artifacts"][HEAD]
                    or status.get("teacher_frozen_and_unchanged") is not True):
                    raise RuntimeError(f"D2 checkpoint does not certify its matching D0 teacher in {directory}")
            classifiers[arm] = load_head(directory, status, binding, arm, epochs, torch.device("cpu"), frozen=True)
            statuses[arm] = status
        models[name][seed] = (encoder, classifiers)
        provenance[f"{name}/seed_{seed}"] = {"stage1": stage1, "stage2": statuses}
    return models, provenance


@contextmanager
def official_test_dataset(exp, name: str, reference_shas: dict, mark_access):
    spec = exp.datasets[name]
    root = Path(spec["root"]).resolve()
    train_entries = parse_annotation_file(TRAIN_MANIFESTS[name])
    manifest = REPO / f"H2T/datasets/data_txt/{DISPLAY[name]}_test.txt"
    mark_access()
    if sha256_file(manifest) != reference_shas[manifest.name]:
        raise RuntimeError(f"Official TEST manifest SHA256 mismatch: {manifest}")
    if name == "imagenet_lt":
        entries = parse_annotation_file(manifest)
        dataset = ImageNetLTParquetDataset(
            root, "test", eval_transform(), expected_rows=spec["test_images"], expected_shards=spec["test_shards"],
            num_classes=spec["num_classes"], shard_paths=discover_imagenet_shards(root, "test"),
            wnid_to_id=imagenet_wnid_map(train_entries), official_entries=entries,
        )
        targets = dataset.labels
    else:
        dataset = PlacesLTManifestDataset(
            resolve_places_root(root), manifest, "test", eval_transform(),
            class_folders=places_class_folders(train_entries), expected_rows=spec["test_images"],
        )
        targets = dataset.targets_array
    try:
        if len(dataset) != spec["test_images"] or not np.all(dataset.class_counts == spec["test_images"] // spec["num_classes"]):
            raise RuntimeError(f"{name} official test row count or class distribution mismatch")
        yield dataset, targets
    finally:
        if isinstance(dataset, ImageNetLTParquetDataset):
            dataset.close()


def classification_metrics(prediction, targets, counts):
    classes = len(counts)
    if (prediction.shape != targets.shape or prediction.ndim != 1 or len(prediction) == 0
        or np.any(prediction < 0) or np.any(prediction >= classes)):
        raise RuntimeError("Malformed official TEST predictions")
    support = np.bincount(targets, minlength=classes)
    correct = np.bincount(targets[prediction == targets], minlength=classes)
    if len(support) != classes or np.any(support == 0):
        raise RuntimeError("Official TEST does not cover the full original label space")
    accuracy = correct * 100.0 / support
    metrics = {
        "Overall": 100.0 * float(correct.sum()) / len(targets),
        "Many": float(accuracy[counts > 100].mean()),
        "Medium": float(accuracy[(counts >= 20) & (counts <= 100)].mean()),
        "Few": float(accuracy[counts < 20].mean()),
    }
    return metrics, support, correct, accuracy


def evaluate_dataset(exp, name, seal, models, reference_shas, device, mark_access):
    for encoder, classifiers in models.values():
        encoder.to(device)
        for classifier in classifiers.values():
            classifier.to(device)
    with official_test_dataset(exp, name, reference_shas, mark_access) as (dataset, targets):
        loader = make_loader(dataset, exp.protocol["stage1"]["batch_size"], exp.workers, exp.seeds[0])
        keys = [(seed, arm) for seed in exp.seeds for arm in (STAGE1, *exp.arms)]
        predictions = {key: np.empty(len(targets), dtype=np.int64) for key in keys}
        offset = 0
        with torch.inference_mode():
            for images, y in loader:
                stop = offset + len(y)
                if not np.array_equal(y.numpy(), targets[offset:stop]):
                    raise RuntimeError(f"{name} official TEST row order changed")
                images = images.to(device, non_blocking=True)
                for seed, (encoder, classifiers) in models.items():
                    features = encoder(images)
                    for arm, classifier in classifiers.items():
                        logits = classifier(features)
                        if not bool(torch.isfinite(logits).all()):
                            raise RuntimeError(f"Nonfinite official TEST logits in {name} seed {seed} {arm}")
                        predictions[seed, arm][offset:stop] = logits.argmax(dim=1).cpu().numpy()
                offset = stop
                if offset % 16000 == 0:
                    run_log(FINAL, f"{name} final evaluation: {offset:,}/{len(targets):,}")
        del loader
        if offset != len(targets):
            raise RuntimeError(f"{name} official TEST was not consumed exactly once")
    counts = np.asarray(seal["train_class_counts"], dtype=np.int64)
    rows = []
    for (seed, arm), prediction in predictions.items():
        metrics, _, _, _ = classification_metrics(prediction, targets, counts)
        epoch = exp.protocol["stage1" if arm == STAGE1 else "stage2"]["epochs"]
        rows.append({"dataset": name, "seed": seed, "arm": arm, "epoch": epoch, "test_rows": len(targets), **metrics})
        with (FINAL / f"{name}_seed{seed}_{arm}_predictions.npz").open("xb") as handle:
            np.savez_compressed(handle, predictions=prediction, labels=targets)
    write_csv(FINAL / f"{DISPLAY[name]}_metrics.csv", rows)
    for encoder, classifiers in models.values():
        encoder.cpu()
        for classifier in classifiers.values():
            classifier.cpu()
    return len(targets)


def evaluate(exp, prepared: dict, device):
    if FINAL.exists():
        raise FileExistsError(f"Refusing to repeat official TEST evaluation: {FINAL}")
    models, provenance = verify_all_checkpoints(exp, prepared)
    reference_shas = accepted_audit()["run_verification"]["input_integrity"]["after"]["manifest_hashes"]
    with record_run(FINAL, exp.config("evaluate-final"), device=device) as status:
        status["formal_gate_verified_at"] = now()
        status["stage1_verified"] = len(provenance)
        status["stage2_verified"] = sum(len(run["stage2"]) for run in provenance.values())
        status["test_splits"] = {}
        for name in exp.datasets:
            access = {}

            def mark_access():
                access["opened_at"] = now()
                status["official_test_access"] = True
                status["test_splits"][name] = access
                save_status(FINAL, status)

            access["rows"] = evaluate_dataset(
                exp, name, prepared["datasets"][name], models[name], reference_shas, device, mark_access
            )
            access["completed_at"] = now()
            save_status(FINAL, status)
