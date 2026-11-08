"""C3 selective feature perturbation: one intervention, three paired Stage-II seeds.

The original nine formal runs are immutable. Reuse train_ccsam.train_one_arm,
including paired batch SHA checks, frozen backbone, and its exact optimizer.
Only the C3 arm may be newly trained. Official CIFAR TEST is forbidden.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ccsam_common import (
    ROOT, ARMS, SEEDS, NUM_CLASSES, BATCH_SIZE, STAGE2_EPOCHS,
    EXPECTED_CLEAN_SHA256, sha256_file, state_hash, hash_arrays,
    read_json, install_test_guard, guard_pass, resolve_checkpoint,
    build_transforms, build_models, augment_batch, seed_everything,
    class_weights_from_counts, check_backbone_unchanged,
    clean_compatible_gap, STAGE2_LR, MOMENTUM, WEIGHT_DECAY,
)
from train_ccsam import (
    C3_ARM, load_prepared, compute_arm_loss, train_one_arm,
    analytic_classwise_feature_delta, autograd_reference_delta,
)

C3_DIR = ROOT / "c3_selective"
BASELINE_ARMS = tuple(ARMS)


def stamp():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def exclusive_json(path: Path, value):
    """No silent overwrites of previous checks or runs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as fp:
        json.dump(value, fp, ensure_ascii=False, indent=2)
        fp.write("\n")


def baseline_file_snapshot():
    """Read all nine original run artifacts and reject missing/incomplete runs."""
    completed_path = ROOT / "formal_completed.json"
    test_guard_path = ROOT / "formal_test_guard.json"
    if not completed_path.is_file() or not test_guard_path.is_file():
        raise FileNotFoundError("Sealed C0/C1/C2 formal completion markers are required")
    marker, test_guard = read_json(completed_path), read_json(test_guard_path)
    if (marker.get("formal_runs") != 9
            or marker.get("epochs_per_run") != STAGE2_EPOCHS
            or marker.get("seeds") != list(SEEDS)
            or set(marker.get("arms", [])) != set(BASELINE_ARMS)
            or marker.get("official_TEST_used") is not False):
        raise RuntimeError("The baseline formal completion manifest is not the sealed nine-arm protocol")
    if test_guard.get("official_TEST_used") is not False or test_guard.get("blocked_attempts"):
        raise RuntimeError("Original formal TEST guard is not clean")
    snapshots = {}
    for seed in SEEDS:
        for arm in BASELINE_ARMS:
            directory = ROOT / "results" / f"seed_{seed}" / arm
            stat_path = directory / "status.json"
            if not stat_path.is_file():
                raise FileNotFoundError(f"Missing baseline run status: {stat_path}")
            status = read_json(stat_path)
            if (status.get("seed") != seed or status.get("arm") != arm
                    or status.get("state") != "completed"
                    or status.get("final_epoch") != STAGE2_EPOCHS
                    or len(status.get("epochs", [])) != STAGE2_EPOCHS
                    or status.get("official_TEST_used") is not False):
                raise RuntimeError(f"Unsealed baseline run: {directory}")
            for filename in ("status.json", "metrics.csv", "batch_trace.csv", "final_classifier.pth"):
                file = directory / filename
                if not file.is_file():
                    raise FileNotFoundError(f"Original baseline local artifact missing: {file}")
                key = str(file.relative_to(ROOT)).replace("\\", "/")
                snapshots[key] = sha256_file(file)
            checkpoint_key = str((directory / "final_classifier.pth").relative_to(ROOT)).replace("\\", "/")
            if snapshots[checkpoint_key] != status.get("final_classifier_sha256"):
                raise RuntimeError(f"Baseline classifier checkpoint SHA mismatch: {directory}")
            with (directory / "batch_trace.csv").open("r", encoding="utf-8", newline="") as fp:
                if sum(1 for _ in fp) != STAGE2_EPOCHS * 170 + 1:
                    raise RuntimeError(f"Baseline batch trace must have 5100 rows: {directory}")
    for filename in ("formal_completed.json", "formal_test_guard.json", "summary.json",
                     "final_epoch_metrics.csv", "paired_deltas.csv", "CCSAM_STAGE2_RESULTS.md"):
        file = ROOT / filename
        if not file.is_file():
            raise FileNotFoundError(f"Missing sealed original analysis: {file}")
        snapshots[filename] = sha256_file(file)

    # Anchor the original nine local/ignored batch traces and checkpoints to
    # the previously published independent SHA256 index, not just to metadata
    # from the same local directory.
    sealed_index = ROOT / "formal_launch_20261106_01" / "ARTIFACT_SHA256.csv"
    if not sealed_index.is_file():
        raise FileNotFoundError("Published original artifact checksum index missing")
    with sealed_index.open("r", encoding="utf-8", newline="") as fp:
        expected = {row["path"].replace("\\", "/"): row["sha256"]
                    for row in csv.DictReader(fp)}
    for key, actual in snapshots.items():
        published = expected.get(key)
        if published is None:
            if key.startswith("results/"):
                raise RuntimeError(f"Original result missing from published checksum index: {key}")
            continue
        if actual != published:
            raise RuntimeError(f"Original artifact differs from published formal run: {key}")
    return snapshots


def preflight(manifest):
    """Validate local originals; never regenerate baseline traces or prepared arrays."""
    if manifest.get("clean_stage1_sha256") != EXPECTED_CLEAN_SHA256:
        raise RuntimeError("Wrong Clean Stage I checkpoint in prepared manifest")
    checkpoint = resolve_checkpoint(manifest["checkpoint_path"])
    if sha256_file(checkpoint) != EXPECTED_CLEAN_SHA256:
        raise RuntimeError("Clean Stage I checkpoint bytes changed")
    required = ["train_images.npy", "train_labels.npy", "validation_images.npy",
                "validation_gap_features.npy", "validation_labels.npy"]
    for name in required:
        if not (ROOT / "prepared" / name).is_file():
            raise FileNotFoundError(f"Missing sealed prepared array: {name}")
    for seed in SEEDS:
        plan_path = ROOT / "prepared" / f"sampling_plan_seed{seed}.npy"
        if not plan_path.is_file():
            raise FileNotFoundError(f"Missing paired sampling plan: {plan_path}")
        plan = np.load(plan_path, mmap_mode="r", allow_pickle=False)
        if plan.shape != (STAGE2_EPOCHS, 10847):
            raise RuntimeError(f"Invalid original sampling plan shape: {plan.shape}")
    counts = np.asarray(manifest["train_class_counts"], dtype=np.int64)
    if counts.shape != (NUM_CLASSES,) or np.any(counts <= 0):
        raise RuntimeError("Original class counts invalid")
    many_classes = np.flatnonzero(counts > 100).tolist()
    if not many_classes or not np.any(counts <= 100):
        raise RuntimeError("C3 needs both Many and non-Many classes")
    snapshot = baseline_file_snapshot()
    source_files = ("train_ccsam.py", "ccsam_common.py", "experiment_protocol.json",
                    "smoke_report.json", "prepared/prepared_manifest.json")
    source_snapshot = {}
    for name in source_files:
        file = ROOT / name
        if not file.is_file():
            raise FileNotFoundError(f"Missing control input: {file}")
        source_snapshot[name] = sha256_file(file)
    return {
        "checked_at": stamp(),
        "baseline_artifact_sha256": snapshot,
        "protocol_and_source_sha256": source_snapshot,
        "clean_checkpoint_sha256": EXPECTED_CLEAN_SHA256,
        "class_counts": counts.tolist(),
        "many_class_ids": many_classes,
        "nonmany_class_ids": np.flatnonzero(counts <= 100).tolist(),
        "many_count_condition": "TRAIN count strictly greater than 100",
        "sealed_stage2_seeds": list(SEEDS),
    }


def compare_baselines(previous):
    current = baseline_file_snapshot()
    if current != previous["baseline_artifact_sha256"]:
        difference = [k for k in sorted(set(current) | set(previous["baseline_artifact_sha256"]))
                      if current.get(k) != previous["baseline_artifact_sha256"].get(k)]
        raise RuntimeError(f"Original baseline artifacts changed: {difference}")
    for name, old in previous["protocol_and_source_sha256"].items():
        if sha256_file(ROOT / name) != old:
            raise RuntimeError(f"Protected source/prepared/Smoke changed: {name}")


def verify_masking_on_batch(gap, labels, classifier_fc, weights, counts):
    """Compare C1/C2/C3 under IDENTICAL features/head; includes all-Many/non-Many cases."""
    results = {}
    cases = {
        "real_mixed": labels,
        "all_many_class_0": torch.zeros_like(labels),
        "all_nonmany_class_99": torch.full_like(labels, NUM_CLASSES - 1),
    }
    for name, ys in cases.items():
        c1 = compute_arm_loss("C1_CCSAM_FEATURE_GRAD", gap, ys, classifier_fc, weights, 0)
        c2 = compute_arm_loss("C2_WEIGHTED_CE_NO_PERTURB", gap, ys, classifier_fc, weights, 0)
        c3 = compute_arm_loss(C3_ARM, gap, ys, classifier_fc, weights, 0, train_counts=counts)
        d1, d3 = c1[4], c3[4]
        mask = torch.as_tensor(counts, dtype=torch.long, device=ys.device)[ys] > 100
        if not torch.equal(d3[mask], torch.zeros_like(d3[mask])):
            raise RuntimeError(f"{name}: a Many example received a nonzero delta")
        if not torch.equal(d3[~mask], d1[~mask]):
            raise RuntimeError(f"{name}: C3 changed a non-Many delta")
        if name == "all_many_class_0":
            if not torch.equal(c3[0], c2[0]):
                raise RuntimeError("All-Many C3 loss must be exactly identical to C2 loss")
        if name == "all_nonmany_class_99":
            if not torch.equal(c3[0], c1[0]):
                raise RuntimeError("All-nonMany C3 loss must be exactly identical to C1 loss")
        results[name] = {
            "many_samples": int(mask.sum().item()),
            "nonmany_samples": int((~mask).sum().item()),
            "many_delta_zero_exact": True,
            "nonmany_delta_equals_C1_bitwise": True,
            "c3_loss": float(c3[0].detach().cpu()),
            "mean_delta_l2": c3[5]["mean_delta_l2_per_sample"],
        }
    real_counts = results["real_mixed"]
    if real_counts["many_samples"] == 0 or real_counts["nonmany_samples"] == 0:
        raise RuntimeError("The selected real Smoke batch must contain both groups")
    return results


def smoke_c3(manifest, train_images, train_labels, transform, device):
    """One disposable classifier update per seed; no training checkpoint or history touched."""
    if (C3_DIR / "smoke_report.json").exists():
        raise FileExistsError("C3 smoke already exists; never overwrite or rerun it")
    results = []
    counts = np.asarray(manifest["train_class_counts"], dtype=np.int64)
    for seed in SEEDS:
        seed_everything(seed, deterministic=True)
        checkpoint = resolve_checkpoint(manifest["checkpoint_path"])
        backbone, classifier, _ = build_models(checkpoint, device)
        frozen = {k: v.detach().cpu().clone() for k, v in backbone.state_dict().items()}
        initial_classifier_hash = state_hash(classifier)
        plan = np.load(ROOT / "prepared" / f"sampling_plan_seed{seed}.npy", mmap_mode="r")
        # The first two real batches cover the class-aware cycle boundary and duplicates.
        generator = torch.Generator(device="cpu").manual_seed(seed + 300009)
        seed_results = []
        for batch_id in (0, 1):
            start = batch_id * BATCH_SIZE
            indices = np.asarray(plan[0, start:start + BATCH_SIZE], dtype=np.int64)
            labels = torch.as_tensor(np.asarray(train_labels[indices], dtype=np.int64), device=device)
            x = augment_batch(train_images, indices, transform, generator)
            with torch.no_grad():
                gap = clean_compatible_gap(backbone(x.to(device)))
            weights = class_weights_from_counts(counts, device)
            delta1, _ = analytic_classwise_feature_delta(
                classifier.module.fc(gap), gap, labels, weights,
                classifier.module.fc.weight,
            )
            reference = autograd_reference_delta(gap, labels, classifier.module.fc, weights)
            if not torch.allclose(delta1, reference, rtol=1e-5, atol=1e-7):
                raise RuntimeError(f"Original analytic delta failed at seed={seed} batch={batch_id}")
            cases = verify_masking_on_batch(gap, labels, classifier.module.fc, weights, counts)
            seed_results.append({
                "batch": batch_id,
                "paired_input_sha256": hash_arrays(indices, labels.detach().cpu().numpy(), x.numpy()),
                "gap_sha256": hash_arrays(gap.detach().cpu().numpy()),
                "mask_checks": cases,
            })
            if batch_id == 1:
                # Disposable classifier update on the second real batch.
                loss = compute_arm_loss(
                    C3_ARM, gap, labels, classifier.module.fc, weights, 0, train_counts=counts
                )[0]
                optimizer = torch.optim.SGD(classifier.module.fc.parameters(),
                                            lr=STAGE2_LR, momentum=MOMENTUM,
                                            weight_decay=WEIGHT_DECAY, nesterov=False)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if any(p.grad is None or not torch.isfinite(p.grad).all()
                       for p in classifier.module.fc.parameters()):
                    raise RuntimeError("C3 Smoke classifier gradient missing/nonfinite")
                optimizer.step()
                check_backbone_unchanged(backbone, frozen)
                if state_hash(classifier) == initial_classifier_hash:
                    raise RuntimeError("C3 Smoke classifier did not update")
                del optimizer
        results.append({
            "seed": seed,
            "batches": seed_results,
            "disposable_classifier_updated": True,
            "backbone_and_bn_unchanged": True,
        })
        del backbone, classifier
        torch.cuda.empty_cache()
    report = {
        "passed": True, "completed_at": stamp(), "arm": C3_ARM,
        "three_seeds": list(SEEDS), "original_formal_training_repeated": False,
        "only_disposable_updates": True, "checks": results,
        "source_sha256": sha256_file(ROOT / "train_ccsam.py"),
        "runner_sha256": sha256_file(ROOT / "train_c3.py"),
        "prepared_manifest_sha256": sha256_file(ROOT / "prepared" / "prepared_manifest.json"),
        "test_guard": guard_pass(), "official_TEST_used": False,
    }
    if report["test_guard"].get("blocked_attempts") or report["test_guard"].get("official_TEST_used"):
        raise RuntimeError("TEST guard violation in C3 Smoke")
    exclusive_json(C3_DIR / "smoke_report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description="C3 only: no Many feature perturbation")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--smoke-only", action="store_true")
    group.add_argument("--formal", action="store_true")
    args = parser.parse_args()

    manifest_path = ROOT / "prepared" / "prepared_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Original prepared manifest unavailable")
    preliminary = read_json(manifest_path)
    data_root = Path(preliminary["official_train_pickle"]).resolve().parent.parent
    install_test_guard(data_root)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no silent CPU or AMP changes")
    device = torch.device("cuda:0")
    manifest, train_images, train_labels, validation_gap, validation_labels = load_prepared()
    transform, _, transform_sha = build_transforms()
    if transform_sha != manifest["transform_source_sha256"]:
        raise RuntimeError("TRAIN augmentation source changed")
    snapshot = preflight(manifest)

    if args.preflight:
        report = {**snapshot, "mode": "preflight", "passed": True,
                  "official_TEST_used": False, "test_guard": guard_pass()}
        exclusive_json(C3_DIR / "preflight_report.json", report)
        print("C3 PREFLIGHT PASS; no training performed", flush=True)
        return

    if args.smoke_only:
        result = smoke_c3(manifest, train_images, train_labels, transform, device)
        compare_baselines(snapshot)
        print(f"C3 SMOKE PASS, {len(result['checks'])} seeds; no formal training performed", flush=True)
        return

    # --formal: only C3; original nine runs must be present on this computer.
    smoke_path = C3_DIR / "smoke_report.json"
    if not smoke_path.is_file():
        raise FileNotFoundError("Run --smoke-only successfully first")
    smoke = read_json(smoke_path)
    if (smoke.get("passed") is not True
            or smoke.get("source_sha256") != sha256_file(ROOT / "train_ccsam.py")
            or smoke.get("runner_sha256") != sha256_file(ROOT / "train_c3.py")
            or smoke.get("prepared_manifest_sha256") != sha256_file(manifest_path)
            or smoke.get("official_TEST_used") is not False):
        raise RuntimeError("C3 Smoke report is invalid or source/prepared differs")
    if (C3_DIR / "formal_completed.json").exists():
        raise FileExistsError("C3 formal completion marker already exists")
    for seed in SEEDS:
        dst = ROOT / "results" / f"seed_{seed}" / C3_ARM
        if dst.exists():
            raise FileExistsError(f"C3 result directory already exists: {dst}")
    completed = []
    for seed in SEEDS:
        status = train_one_arm(seed, C3_ARM, manifest, train_images, train_labels,
                               validation_gap, validation_labels, transform, device)
        if (status.get("state") != "completed" or status.get("final_epoch") != STAGE2_EPOCHS):
            raise RuntimeError(f"C3 did not complete all epochs: seed={seed}")
        completed.append({"seed": seed, "epoch": status["final_epoch"],
                          "final_classifier_sha256": status["final_classifier_sha256"]})
        compare_baselines(snapshot)
    guard = guard_pass()
    if guard.get("official_TEST_used") or guard.get("blocked_attempts"):
        raise RuntimeError("TEST guard violation in C3 formal run")
    compare_baselines(snapshot)
    result = {
        "passed": True, "arm": C3_ARM, "completed_at": stamp(),
        "seeds": list(SEEDS), "epochs_per_seed": STAGE2_EPOCHS,
        "only_c3_new_training": True, "original_nine_runs_unchanged": True,
        "runs": completed, "official_TEST_used": False, "test_guard": guard,
        "baseline_artifact_sha256": snapshot["baseline_artifact_sha256"],
        "source_sha256": sha256_file(ROOT / "train_ccsam.py"),
        "prepared_manifest_sha256": sha256_file(manifest_path),
    }
    exclusive_json(C3_DIR / "formal_completed.json", result)
    print("C3 FORMAL THREE-SEED TRAINING COMPLETED. Original nine runs unchanged.", flush=True)


if __name__ == "__main__":
    main()
