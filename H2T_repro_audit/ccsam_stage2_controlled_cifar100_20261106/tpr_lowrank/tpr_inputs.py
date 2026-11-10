"""Read-only gates for the sealed Stage-II controls and TPR inputs."""
from __future__ import annotations

import csv
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

OUT = Path(__file__).resolve().parent
if str(OUT.parent) not in sys.path:
    sys.path.insert(0, str(OUT.parent))

import ccsam_common as c
from ab_head_preservation import train_ab as c4
from train_ccsam import C3_ARM, load_prepared

BRANCH = "experiment/ccsam-tpr-lowrank-20261110"
BASE = "4ab34ad00c86bcac60d765123f153b76ff097f47"
ARM = "TPR_LOW_RANK_TAIL"
C1 = c.ARMS[1]
C2 = c.ARMS[2]
STEPS = (c.TRAIN_SAMPLES + c.BATCH_SIZE - 1) // c.BATCH_SIZE
PREFIX = OUT.relative_to(c.REPO_ROOT).as_posix()
ARCHIVES = (
    (c.ROOT, "formal_launch_20261106_01/ARTIFACT_SHA256.csv",
     "f916957ea4a7b90cf82dc5839eb193e044d881dd", 52),
    (c.ROOT, "c3_selective/ARTIFACT_SHA256.csv",
     "10d8d52966b13db0b8992eb7e2d2c4dd9f4f258c", 26),
    (c4.AB_DIR, "ARTIFACT_SHA256.csv", BASE, 31),
)
SOURCE_NAMES = (
    "TPR_PROTOCOL.md", "CODEX_CCSAM_TPR_20261110.txt", "README.md",
    "tpr_model.py", "tpr_inputs.py", "train_tpr.py", "analyze_tpr.py", "test_tpr_model.py",
)


@dataclass(frozen=True)
class Inputs:
    manifest: dict
    train_images: np.ndarray
    train_labels: np.ndarray
    validation_gap: np.ndarray
    validation_labels: np.ndarray
    counts: np.ndarray
    checkpoint: Path
    train_transform: object
    clean_transform: object
    plans: dict


def require(condition, reason):
    if not condition:
        raise RuntimeError(reason)


def git(*args):
    return subprocess.check_output(["git", "--no-optional-locks", *args], cwd=c.REPO_ROOT)


def rel(path):
    return path.relative_to(c.REPO_ROOT).as_posix()


def csv_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def checked_file(path, expected):
    if not path.is_file():
        raise FileNotFoundError(f"Required sealed file is missing: {path}")
    digest = c.sha256_file(path)
    require(digest == expected, f"SHA256 mismatch: {path} ({digest}, expected {expected})")
    return digest


def protected_snapshot():
    branch = git("branch", "--show-current").decode().strip()
    require(branch == BRANCH, f"Unexpected working branch: {branch}")
    git("merge-base", "--is-ancestor", BASE, "HEAD")
    require(not git("ls-files", "-u", "-z"), "Git index contains unresolved entries")
    outside = [name for name in git("diff", BASE, "--name-only", "-z").decode().split("\0")
               if name and not name.startswith(PREFIX + "/")]
    require(not outside, f"Protected tracked files changed: {outside}")
    historical = c.read_json(c.ROOT / "validation_20261106_05/PRE_SWITCH_UNTRACKED.json")
    untracked = set(git("ls-files", "--others", "--exclude-standard", "-z").decode().split("\0")) - {""}
    outside_untracked = {name for name in untracked if not name.startswith(PREFIX + "/")}
    require(outside_untracked == set(historical), "The set of historical untracked files changed")
    for name, item in historical.items():
        checked_file(c.REPO_ROOT / name, item["sha256"])

    indexes, files = {}, {}
    for directory, name, anchor, expected_count in ARCHIVES:
        path = directory / name
        require(path.is_file() and path.read_bytes() == git("cat-file", "blob", f"{anchor}:{rel(path)}"),
                f"Independent archive index differs from its published commit: {path}")
        rows = csv_rows(path)
        require(len(rows) == len({row["path"] for row in rows}) == expected_count,
                f"Unexpected archive index coverage: {path}")
        for row in rows:
            target = (directory / row["path"]).resolve()
            require(target.is_relative_to(directory), f"Path leaves its sealed archive: {row['path']}")
            digest = checked_file(target, row["sha256"])
            require(target.stat().st_size == int(row["bytes"]), f"File size changed: {target}")
            files[rel(target)] = digest
        indexes[rel(path)] = {"sha256": c.sha256_file(path), "commit": anchor, "entries": expected_count}

    original = c.read_json(c.ROOT / "formal_completed.json")
    c3 = c.read_json(c.ROOT / "c3_selective/formal_completed.json")
    c4_done = c.read_json(c4.AB_DIR / "formal_completed.json")
    require(original.get("formal_runs") == 9 and original.get("seeds") == list(c.SEEDS)
            and original.get("epochs_per_run") == 30 and original.get("official_TEST_used") is False,
            "C0-C2 completion provenance failed")
    require(c3.get("passed") is True and c3.get("original_nine_runs_unchanged") is True
            and c3.get("seeds") == list(c.SEEDS) and c3.get("official_TEST_used") is False,
            "C3 completion provenance failed")
    require(c4_done.get("passed") is True and c4_done.get("only_c4_new_training") is True
            and c4_done.get("seeds") == list(c.SEEDS) and c4_done.get("epochs_per_seed") == 30
            and c4_done.get("optimizer_steps") == 15300 and c4_done.get("official_TEST_used") is False,
            "C4 completion provenance failed")
    for guard in (c.read_json(c.ROOT / "formal_test_guard.json"), c3["test_guard"], c4_done["test_guard"]):
        require(guard["official_TEST_used"] is False and not guard["blocked_attempts"]
                and guard["official_test_file_open_attempts"] == guard["CIFAR100_train_false_attempts"] == 0,
                "Protected TEST guard provenance failed")
    return {"branch": branch, "head": git("rev-parse", "HEAD").decode().strip(),
            "base_commit": BASE, "indexes": indexes, "files": files,
            "historical_untracked_files_verified": len(historical)}


def guarded_manifest():
    path = c.ROOT / "prepared/prepared_manifest.json"
    preliminary = c.read_json(path)
    train_pickle = Path(preliminary["official_train_pickle"]).resolve()
    require(train_pickle.name == "train" and train_pickle.parent.name == "cifar-100-python",
            "Prepared manifest does not identify the official TRAIN pickle")
    c.install_test_guard(train_pickle.parent.parent)
    return preliminary


def collect_inputs():
    manifest = guarded_manifest()
    protected = protected_snapshot()
    seal = c.read_json(c.ROOT / "c3_selective/C3_INPUT_SEAL.json")["inputs"]
    require(len(seal) == 16, "Expected 16 sealed prepared/source input entries")
    for item in seal.values():
        path = (c.REPO_ROOT / item["path"]).resolve()
        require(path.is_relative_to(c.REPO_ROOT), f"Invalid input path: {path}")
        checked_file(path, item["sha256"])
    checkpoint = c.resolve_checkpoint(manifest["checkpoint_path"])
    require(rel(checkpoint) == seal["clean_checkpoint"]["path"], "Clean checkpoint path changed")
    require(manifest["train_validation_raw_id_overlap"] == manifest["train_validation_exact_pixel_overlap"] == 0
            and manifest["train_samples"] == c.TRAIN_SAMPLES and manifest["validation_samples"] == c.VALIDATION_SAMPLES,
            "Prepared TRAIN/holdout split failed")
    manifest_loaded, train_images, train_labels, validation_gap, validation_labels = load_prepared()
    require(manifest_loaded == manifest, "Prepared manifest changed during verification")
    train_transform, clean_transform, transform_sha = c.build_transforms()
    require(transform_sha == manifest["transform_source_sha256"] == seal["TRAIN_transform"]["sha256"],
            "Sealed transform changed")
    counts = np.asarray(manifest["train_class_counts"], dtype=np.int64)
    require(train_images.dtype == np.uint8 and train_labels.dtype.kind in "iu"
            and validation_labels.dtype.kind in "iu" and validation_gap.dtype == np.float32
            and np.isfinite(validation_gap).all(), "Unexpected prepared array dtype or finiteness")
    require(train_labels.min() >= 0 and train_labels.max() < c.NUM_CLASSES
            and validation_labels.min() >= 0 and validation_labels.max() < c.NUM_CLASSES,
            "Label outside the sealed class range")
    require(counts.shape == (c.NUM_CLASSES,) and np.all(counts > 0)
            and np.array_equal(counts, np.bincount(train_labels, minlength=c.NUM_CLASSES)),
            "Prepared TRAIN counts differ from labels")
    require(np.bincount(validation_labels, minlength=c.NUM_CLASSES).tolist()
            == manifest["validation_class_counts"] == [0] + [20] * 99, "Sealed holdout counts changed")

    plans, teachers = {}, {}
    for seed in c.SEEDS:
        recorded = manifest["sampling_plans"][str(seed)]
        file = c.ROOT / "prepared" / recorded["path"]
        require(rel(file) == seal[f"sampling_seed_{seed}"]["path"]
                and recorded["sha256"] == seal[f"sampling_seed_{seed}"]["sha256"]
                and recorded["all_epochs_match_official_continuous_sampler"] is True,
                f"Unsealed sampling plan for seed {seed}")
        plan = np.load(file, mmap_mode="r", allow_pickle=False)
        require(plan.shape == (c.STAGE2_EPOCHS, c.TRAIN_SAMPLES) and plan.dtype == np.int32
                and plan.min() >= 0 and plan.max() < c.TRAIN_SAMPLES, f"Malformed sampling plan: {file}")
        plans[seed] = plan
        teacher = c.ROOT / "results" / f"seed_{seed}" / C2
        status = c.read_json(teacher / "status.json")
        checkpoint_file = teacher / "final_classifier.pth"
        sha = protected["files"][rel(checkpoint_file)]
        require(status["state"] == "completed" and status["seed"] == seed and status["arm"] == C2
                and status["final_epoch"] == c.STAGE2_EPOCHS and status["official_TEST_used"] is False
                and sha == status["final_classifier_sha256"], f"Unsealed C2 base for seed {seed}")
        trace = c.ROOT / "results" / f"seed_{seed}" / C1 / "batch_trace.csv"
        require(len(csv_rows(trace)) == c.STAGE2_EPOCHS * STEPS, f"C1 trace length changed: {trace}")
        teachers[str(seed)] = {"path": rel(checkpoint_file), "status": rel(teacher / "status.json"), "sha256": sha}

    source_paths = [OUT / name for name in SOURCE_NAMES] + [
        c.ROOT / name for name in ("ccsam_common.py", "train_ccsam.py", "train_c3.py", "analyze_c3.py",
                                   "prepare_ccsam.py", "experiment_protocol.json")
    ] + [c4.AB_DIR / "train_ab.py", c.RUNTIME / "models/resnet_cifar.py", c.OFFICIAL_SAMPLER_SOURCE]
    require(c.sha256_file(c.OFFICIAL_SAMPLER_SOURCE) == c.OFFICIAL_SAMPLER_SHA256,
            "Pinned class-aware sampler changed")
    sources = {rel(path): c.sha256_file(path) for path in source_paths}
    snapshot = {
        "protected": protected, "sealed_inputs": seal, "sources": sources,
        "teachers": teachers, "class_counts": counts.tolist(),
        "clean_stage1_sha256": c.EXPECTED_CLEAN_SHA256,
        "protocol_sha256": sources[rel(OUT / "TPR_PROTOCOL.md")],
        "validation_partition": "1,980-image holdout from official TRAIN; class 0 absent",
    }
    return Inputs(manifest, train_images, train_labels, validation_gap, validation_labels,
                  counts, checkpoint, train_transform, clean_transform, plans), snapshot


def load_base(seed, initial_classifier, device, snapshot):
    """Reuse the C4 loader for the independently sealed final C2 checkpoint."""
    record = snapshot["teachers"][str(seed)]
    checked_file(c.REPO_ROOT / record["path"], record["sha256"])
    base, sha, initial_sha = c4.load_teacher(seed, initial_classifier, device)
    require(sha == record["sha256"] and all(p.dtype == torch.float32 and torch.isfinite(p).all()
            and not p.requires_grad and p.grad is None for p in base.parameters()),
            f"Nonfinite or unfrozen C2 base for seed {seed}")
    return base, sha, initial_sha


def preflight_marker(snapshot):
    marker = c.read_json(OUT / "preflight_report.json")
    require(marker["passed"] is True and marker["official_TEST_used"] is False
            and marker["snapshot"] == snapshot, "TPR Preflight marker or input fingerprint changed")
    return marker


def run_dir(seed):
    return OUT / "results" / f"seed_{seed}" / ARM


def historical_dir(seed, arm):
    parent = c4.AB_DIR if arm == c4.C4_ARM else c.ROOT
    return parent / "results" / f"seed_{seed}" / arm


HISTORICAL_ARMS = c.ARMS + (C3_ARM, c4.C4_ARM)
