"""CC-SAM-TPR: classifier-stage tail parameter reallocation.

This is a MORE-inspired adaptation, not a reproduction of the full MORE paper:
the Clean Stage-I backbone is frozen, the matching seed's completed C2
classifier is the frozen general parameter set, and a masked low-rank residual
is learned only for non-Many output rows.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from ccsam_common import (  # noqa: E402
    ROOT, NUM_CLASSES, FEATURE_DIM, BATCH_SIZE, STAGE2_EPOCHS, STAGE2_LR,
    MOMENTUM, WEIGHT_DECAY, SEEDS, EXPECTED_CLEAN_SHA256,
    sha256_file, hash_arrays, state_hash, write_json, read_json,
    install_test_guard, guard_pass, resolve_checkpoint, build_transforms,
    build_models, seed_everything, augment_batch, class_weights_from_counts,
    alpha_for_epoch, check_backbone_unchanged, clean_compatible_gap,
    metrics_from_predictions, RUNTIME, OFFICIAL_SAMPLER_SOURCE,
    OFFICIAL_SAMPLER_SHA256, validate_class_aware_plan,
)
from train_ccsam import (  # noqa: E402
    load_prepared, compute_arm_loss, read_trace, verify_analytic_gradient,
)
from train_c3 import baseline_file_snapshot  # noqa: E402
from train_ab import c3_artifact_snapshot  # noqa: E402


TPR_DIR = Path(__file__).resolve().parent
AB_DIR = ROOT / "ab_head_preservation"
TPR_ARM = "CCSAM_TPR"
C2_ARM = "C2_WEIGHTED_CE_NO_PERTURB"
C4_ARM = "C4_CCSAM_HEAD_PRESERVE"
MANY_THRESHOLD = 100
TPR_RANK = 8
TAIL_NEIGHBORS = 2
MORE_AMPLITUDE_PER_CLASS = 2.0
MORE_PEAK = MORE_AMPLITUDE_PER_CLASS * NUM_CLASSES
ORTHOGONALITY_LAMBDA = 0.1
TAIL_DIVERSITY_LAMBDA = 0.1
TRACE_FIELDS = (
    "epoch", "batch", "batch_size", "indices_labels_sha256",
    "augmented_images_sha256", "gap_features_sha256",
)
SOURCE_FILES = (
    "train_ccsam.py",
    "ccsam_common.py",
    "train_c3.py",
    "experiment_protocol.json",
    "prepared/prepared_manifest.json",
    "formal_launch_20261106_01/ARTIFACT_SHA256.csv",
    "c3_selective/formal_completed.json",
    "c3_selective/c3_summary.json",
    "c3_selective/C3_RESULTS.md",
    "c3_selective/ARTIFACT_SHA256.csv",
    "official_reference/Samplers.py",
    "ab_head_preservation/train_ab.py",
    "ab_head_preservation/AB_PROTOCOL.md",
    "ab_head_preservation/formal_completed.json",
    "ab_head_preservation/ARTIFACT_SHA256.csv",
)


def stamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def exclusive_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as fp:
        json.dump(value, fp, ensure_ascii=False, indent=2, allow_nan=False)
        fp.write("\n")


def exclusive_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def c4_artifact_snapshot() -> dict[str, str]:
    marker_path = AB_DIR / "formal_completed.json"
    index_path = AB_DIR / "ARTIFACT_SHA256.csv"
    if not marker_path.is_file() or not index_path.is_file():
        raise FileNotFoundError("Completed C4 marker and checksum index are required")
    marker = read_json(marker_path)
    if (
        marker.get("passed") is not True
        or marker.get("arm") != C4_ARM
        or marker.get("seeds") != list(SEEDS)
        or marker.get("epochs_per_seed") != STAGE2_EPOCHS
        or marker.get("official_TEST_used") is not False
    ):
        raise RuntimeError("C4 completion marker is invalid or TEST provenance is missing")
    with index_path.open("r", encoding="utf-8", newline="") as fp:
        rows = list(csv.DictReader(fp))
    if not rows:
        raise RuntimeError("C4 artifact checksum index is empty")
    snapshot = {}
    for row in rows:
        relative = row["path"].replace("\\", "/")
        path = AB_DIR / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing C4 archived artifact: {relative}")
        actual = sha256_file(path)
        if actual != row["sha256"]:
            raise RuntimeError(f"C4 artifact checksum mismatch: {relative}")
        snapshot[relative] = actual
    return snapshot


def prepared_snapshot() -> dict[str, str]:
    names = (
        "train_images.npy", "train_labels.npy", "validation_images.npy",
        "validation_gap_features.npy", "validation_labels.npy",
        "sampling_plan_seed123.npy", "sampling_plan_seed124.npy",
        "sampling_plan_seed125.npy",
    )
    result = {}
    for name in names:
        path = ROOT / "prepared" / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing protected prepared array: {name}")
        result[name] = sha256_file(path)
    manifest = read_json(ROOT / "prepared" / "prepared_manifest.json")
    clean_checkpoint = resolve_checkpoint(manifest.get("checkpoint_path"))
    result["clean_stage1_checkpoint"] = sha256_file(clean_checkpoint)
    return result


def source_snapshot() -> dict[str, str]:
    snapshot = {}
    for relative in SOURCE_FILES:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing protected source/protocol input: {relative}")
        snapshot[relative] = sha256_file(path)
    for name, path in (
        ("runtime_cifar100_transform", RUNTIME / "datasets" / "cifar100.py"),
        ("pinned_official_sampler", OFFICIAL_SAMPLER_SOURCE),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing pinned runtime source: {path}")
        snapshot[name] = sha256_file(path)
    if snapshot["pinned_official_sampler"] != OFFICIAL_SAMPLER_SHA256:
        raise RuntimeError("Pinned official sampler source checksum mismatch")
    for filename in (
        "train_tpr.py", "test_tpr.py", "analyze_tpr.py", "finalize_tpr.py",
        "TPR_PROTOCOL.md", "README.md", ".gitignore",
        "CODEX_TPR_PARAMETER_REALLOCATION_20261110.txt",
    ):
        path = TPR_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing TPR implementation file: {path.name}")
        snapshot[str(path.relative_to(ROOT)).replace("\\", "/")] = sha256_file(path)
    return snapshot


def protected_snapshot() -> dict:
    return {
        "c0_c1_c2": baseline_file_snapshot(),
        "c3": c3_artifact_snapshot(),
        "c4": c4_artifact_snapshot(),
        "prepared": prepared_snapshot(),
        "sources": source_snapshot(),
    }


def compare_protected(previous: dict) -> None:
    current = protected_snapshot()
    if current != previous:
        changed = []
        for section in ("c0_c1_c2", "c3", "c4", "prepared", "sources"):
            before = previous.get(section, {})
            after = current.get(section, {})
            keys = set(before) | set(after)
            changed.extend(
                f"{section}:{key}" for key in keys
                if before.get(key) != after.get(key)
            )
        raise RuntimeError(f"Protected C0-C4/source inputs changed: {sorted(changed)}")


def load_c2_base(seed: int, expected_initial_classifier_sha256: str | None = None):
    directory = ROOT / "results" / f"seed_{seed}" / C2_ARM
    status_path = directory / "status.json"
    checkpoint_path = directory / "final_classifier.pth"
    if not status_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing local C2 base classifier for seed {seed}")
    status = read_json(status_path)
    if (
        status.get("state") != "completed"
        or status.get("seed") != seed
        or status.get("arm") != C2_ARM
        or status.get("final_epoch") != STAGE2_EPOCHS
        or status.get("official_TEST_used") is not False
    ):
        raise RuntimeError(f"Unsealed C2 base status for seed {seed}")
    checkpoint_sha = sha256_file(checkpoint_path)
    if checkpoint_sha != status.get("final_classifier_sha256"):
        raise RuntimeError(f"C2 base checkpoint/status SHA mismatch for seed {seed}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("epoch") != STAGE2_EPOCHS
        or checkpoint.get("seed") != seed
        or checkpoint.get("arm") != C2_ARM
        or checkpoint.get("clean_stage1_sha256") != EXPECTED_CLEAN_SHA256
        or checkpoint.get("official_TEST_used") is not False
        or checkpoint.get("backbone_retrained") is not False
    ):
        raise RuntimeError(f"Invalid C2 base checkpoint metadata for seed {seed}")
    state = checkpoint.get("state_dict_classifier")
    if not isinstance(state, dict) or "fc.weight" not in state or "fc.bias" not in state:
        raise RuntimeError(f"Unexpected C2 classifier state dictionary for seed {seed}")
    weight = state["fc.weight"].detach().float().clone()
    bias = state["fc.bias"].detach().float().clone()
    if tuple(weight.shape) != (NUM_CLASSES, FEATURE_DIM) or tuple(bias.shape) != (NUM_CLASSES,):
        raise RuntimeError("C2 classifier tensor shapes do not match [100,64] and [100]")
    if checkpoint.get("initial_classifier_sha256") != status.get("initial_classifier_sha256"):
        raise RuntimeError(f"C2 initial-classifier provenance mismatch for seed {seed}")
    if (
        expected_initial_classifier_sha256 is not None
        and checkpoint.get("initial_classifier_sha256") != expected_initial_classifier_sha256
    ):
        raise RuntimeError(
            f"C2 seed {seed} was not initialized from the currently sealed Clean Stage-I classifier"
        )
    return weight, bias, checkpoint_sha, checkpoint


class LowRankTailClassifier(nn.Module):
    """Frozen C2 general classifier plus a class-masked low-rank tail residual."""

    def __init__(
        self,
        base_weight: torch.Tensor,
        base_bias: torch.Tensor,
        tail_indices: torch.Tensor,
        rank: int = TPR_RANK,
    ):
        super().__init__()
        if base_weight.shape != (NUM_CLASSES, FEATURE_DIM):
            raise ValueError("base_weight must have shape [100,64]")
        if base_bias.shape != (NUM_CLASSES,):
            raise ValueError("base_bias must have shape [100]")
        if tail_indices.ndim != 1 or tail_indices.dtype != torch.long:
            raise ValueError("tail_indices must be a one-dimensional int64 tensor")
        if tail_indices.numel() == 0 or int(tail_indices.min()) < 0 or int(tail_indices.max()) >= NUM_CLASSES:
            raise ValueError("tail_indices must contain valid non-Many class IDs")
        if len(torch.unique(tail_indices)) != len(tail_indices):
            raise ValueError("tail_indices contains duplicates")
        self.register_buffer("base_weight", base_weight.detach().float().clone())
        self.register_buffer("base_bias", base_bias.detach().float().clone())
        self.register_buffer("tail_indices", tail_indices.detach().clone())
        self.rank = int(rank)
        if self.rank <= 0 or self.rank > min(FEATURE_DIM, int(tail_indices.numel())):
            raise ValueError("Invalid low-rank dimension")
        self.A = nn.Parameter(torch.empty(self.rank, FEATURE_DIM))
        self.B = nn.Parameter(torch.empty(int(tail_indices.numel()), self.rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.normal_(self.B, mean=0.0, std=1e-3)

    def tail_residual(self) -> torch.Tensor:
        return self.B @ self.A

    def full_residual(self) -> torch.Tensor:
        return torch.zeros_like(self.base_weight).index_copy(
            0, self.tail_indices, self.tail_residual()
        )

    @property
    def weight(self) -> torch.Tensor:
        return self.base_weight + self.full_residual()

    @property
    def bias(self) -> torch.Tensor:
        return self.base_bias

    def base_logits(self, features: torch.Tensor) -> torch.Tensor:
        return F.linear(features, self.base_weight, self.base_bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.linear(features, self.weight, self.base_bias)

    def orthogonality_loss(self) -> torch.Tensor:
        base_rows = F.normalize(self.base_weight[self.tail_indices], dim=1, eps=1e-8)
        residual_rows = F.normalize(self.tail_residual(), dim=1, eps=1e-8)
        cosine = (base_rows * residual_rows).sum(dim=1)
        return cosine.square().mean()

    def diversity_loss(self, edge_rows: torch.Tensor) -> torch.Tensor:
        """Decorrelate neighboring classes' *effective classifier residual rows*.

        The learned classifier update is Delta W = B @ A. Since A is trainable
        and is not constrained to be an isometry, cosine(B_i, B_j) need not
        equal cosine(Delta W_i, Delta W_j). The penalty therefore acts on the
        actual 64-D rows added to the frozen general classifier.
        """
        if edge_rows.ndim != 2 or edge_rows.shape[1] != 2:
            raise ValueError("edge_rows must have shape [E,2]")
        if edge_rows.numel() == 0:
            return self.tail_residual().sum() * 0.0
        if int(edge_rows.min()) < 0 or int(edge_rows.max()) >= self.B.shape[0]:
            raise ValueError("Tail-neighbor edge index is out of range")
        residual_rows = F.normalize(self.tail_residual(), dim=1, eps=1e-8)
        cosine = (
            residual_rows[edge_rows[:, 0]] * residual_rows[edge_rows[:, 1]]
        ).sum(dim=1)
        return cosine.square().mean()

    def base_hash(self) -> str:
        return hash_arrays(
            self.base_weight.detach().cpu().numpy(),
            self.base_bias.detach().cpu().numpy(),
            self.tail_indices.detach().cpu().numpy(),
        )

    def merged_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "fc.weight": self.weight.detach().cpu().clone(),
            "fc.bias": self.base_bias.detach().cpu().clone(),
        }


def more_discrepancy_loss(
    classifier: LowRankTailClassifier,
    features: torch.Tensor,
    labels: torch.Tensor,
    train_counts,
):
    """MORE Eq. (2)-(3): class-prior-weighted squared logit discrepancy.

    The discrepancy is the squared L2 norm over all 100 logits. The sample
    weight is pi_y = N_y / N from the sealed official TRAIN counts.
    """
    if labels.dtype != torch.long or labels.ndim != 1 or labels.shape[0] != features.shape[0]:
        raise ValueError("MORE labels must be int64 with one label per feature row")
    counts = torch.as_tensor(train_counts, dtype=torch.float32, device=features.device)
    if counts.shape != (NUM_CLASSES,) or bool((counts <= 0).any().item()):
        raise ValueError("MORE requires 100 positive sealed TRAIN class counts")
    if bool(((labels < 0) | (labels >= NUM_CLASSES)).any().item()):
        raise ValueError("MORE label outside [0,99]")
    full_logits = classifier(features)
    general_logits = classifier.base_logits(features)
    delta_logits = full_logits - general_logits
    per_sample_discrepancy = delta_logits.square().sum(dim=1)
    class_prior = counts / counts.sum()
    loss = (class_prior[labels] * per_sample_discrepancy).mean()
    if not torch.isfinite(loss) or not torch.isfinite(delta_logits).all():
        raise RuntimeError("Non-finite MORE discrepancy")
    return loss, delta_logits


def sinusoidal_more_weight(step: int, total_steps: int, peak: float = MORE_PEAK) -> float:
    if total_steps <= 1:
        raise ValueError("total_steps must be greater than one")
    if step < 0 or step >= total_steps:
        raise ValueError("step must be in [0,total_steps)")
    phase = math.pi * float(step) / float(total_steps - 1)
    return float(peak * math.sin(phase))


def build_tail_neighbor_graph(
    backbone,
    train_images: np.ndarray,
    train_labels: np.ndarray,
    train_counts: np.ndarray,
    eval_transform,
    device: torch.device,
    batch_size: int = 128,
) -> dict:
    """Build a train-only feature-proximity graph; no validation data is used."""
    observed_counts = np.bincount(
        np.asarray(train_labels, dtype=np.int64), minlength=NUM_CLASSES
    )
    if not np.array_equal(observed_counts, np.asarray(train_counts, dtype=np.int64)):
        raise RuntimeError("TRAIN labels do not match the sealed class-count manifest")
    tail_ids = np.flatnonzero(observed_counts <= MANY_THRESHOLD)
    if len(tail_ids) < 2:
        raise RuntimeError("At least two non-Many classes are required for tail-neighbor constraints")
    sums = torch.zeros(NUM_CLASSES, FEATURE_DIM, dtype=torch.float64, device=device)
    counts_tensor = torch.zeros(NUM_CLASSES, dtype=torch.float64, device=device)
    backbone.eval()
    with torch.no_grad():
        for start in range(0, len(train_images), batch_size):
            end = min(start + batch_size, len(train_images))
            images = torch.stack([
                eval_transform(Image.fromarray(image))
                for image in train_images[start:end]
            ]).to(device, non_blocking=True)
            labels = torch.as_tensor(
                np.asarray(train_labels[start:end], dtype=np.int64),
                dtype=torch.long, device=device,
            )
            gap = clean_compatible_gap(backbone(images)).double()
            sums.index_add_(0, labels, gap)
            counts_tensor.index_add_(0, labels, torch.ones_like(labels, dtype=torch.float64))
    if not torch.equal(counts_tensor.cpu().long(), torch.as_tensor(observed_counts)):
        raise RuntimeError("Feature-centroid extraction did not cover every TRAIN example")
    centroids = sums / counts_tensor.clamp_min(1.0).unsqueeze(1)
    normalized = F.normalize(centroids.float(), dim=1, eps=1e-8)
    tail_tensor = torch.as_tensor(tail_ids, dtype=torch.long, device=device)
    similarity = normalized[tail_tensor] @ normalized[tail_tensor].T
    similarity.fill_diagonal_(-float("inf"))
    edge_set = set()
    k = min(TAIL_NEIGHBORS, len(tail_ids) - 1)
    for row in range(len(tail_ids)):
        neighbors = torch.topk(similarity[row], k=k, largest=True).indices.cpu().tolist()
        for other in neighbors:
            if row != int(other):
                edge_set.add(tuple(sorted((row, int(other)))))
    edges = sorted(edge_set)
    if not edges:
        raise RuntimeError("Tail-neighbor graph is empty")
    centroid_array = centroids.detach().cpu().numpy().astype(np.float32)
    return {
        "method": "symmetric_topk_cosine_neighbors_of_clean_GAP_class_centroids",
        "source": "official TRAIN images and labels only; deterministic eval transform; frozen Clean Stage-I backbone",
        "k_neighbors_per_class": int(k),
        "tail_class_ids": tail_ids.tolist(),
        "edge_rows": [list(edge) for edge in edges],
        "edge_class_ids": [[int(tail_ids[a]), int(tail_ids[b])] for a, b in edges],
        "centroid_sha256": hash_arrays(centroid_array),
        "train_labels_sha256": hash_arrays(np.asarray(train_labels, dtype=np.int64)),
        "train_counts_sha256": hash_arrays(np.asarray(train_counts, dtype=np.int64)),
        "clean_stage1_sha256": EXPECTED_CLEAN_SHA256,
    }


def validate_neighbor_graph(graph: dict, train_counts: np.ndarray) -> torch.Tensor:
    tail_ids = np.flatnonzero(np.asarray(train_counts, dtype=np.int64) <= MANY_THRESHOLD)
    if graph.get("tail_class_ids") != tail_ids.tolist():
        raise RuntimeError("Tail-neighbor graph class IDs do not match the sealed TRAIN counts")
    if graph.get("clean_stage1_sha256") != EXPECTED_CLEAN_SHA256:
        raise RuntimeError("Tail-neighbor graph references a different Clean Stage-I model")
    if graph.get("train_counts_sha256") != hash_arrays(np.asarray(train_counts, dtype=np.int64)):
        raise RuntimeError("Tail-neighbor graph TRAIN-count provenance mismatch")
    edges = torch.as_tensor(graph.get("edge_rows", []), dtype=torch.long)
    if edges.ndim != 2 or edges.shape[1] != 2 or edges.numel() == 0:
        raise RuntimeError("Malformed or empty tail-neighbor graph")
    if bool((edges[:, 0] == edges[:, 1]).any()) or int(edges.min()) < 0 or int(edges.max()) >= len(tail_ids):
        raise RuntimeError("Tail-neighbor graph contains invalid edges")
    canonical = sorted({tuple(sorted((int(a), int(b)))) for a, b in edges.tolist()})
    if len(canonical) != len(edges):
        raise RuntimeError("Tail-neighbor graph contains duplicate edges")
    return edges


def evaluate_classifier(classifier, validation_gap, validation_labels, train_counts, device):
    classifier.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(validation_labels), 512):
            end = min(start + 512, len(validation_labels))
            features = torch.from_numpy(
                np.array(validation_gap[start:end], dtype=np.float32, copy=True)
            ).to(device)
            logits = classifier(features)
            if logits.shape != (end - start, NUM_CLASSES) or not torch.isfinite(logits).all():
                raise RuntimeError("Malformed or non-finite TPR validation logits")
            predictions.append(logits.argmax(dim=1).cpu().numpy())
    classifier.train()
    pred = np.concatenate(predictions)
    return metrics_from_predictions(pred, validation_labels, train_counts)


def evaluate_base(base_weight, base_bias, validation_gap, validation_labels, train_counts, device):
    predictions = []
    with torch.no_grad():
        for start in range(0, len(validation_labels), 512):
            end = min(start + 512, len(validation_labels))
            features = torch.from_numpy(
                np.array(validation_gap[start:end], dtype=np.float32, copy=True)
            ).to(device)
            logits = F.linear(features, base_weight, base_bias)
            predictions.append(logits.argmax(dim=1).cpu().numpy())
    return metrics_from_predictions(np.concatenate(predictions), validation_labels, train_counts)


def collect_preflight(manifest: dict) -> dict:
    if manifest.get("clean_stage1_sha256") != EXPECTED_CLEAN_SHA256:
        raise RuntimeError("Prepared arrays do not reference the sealed Clean Stage-I checkpoint")
    if manifest.get("official_TEST_used") is not False:
        raise RuntimeError("Prepared manifest does not certify TEST-free preparation")
    if manifest.get("validation_baseline_overall_matches_sealed_clean_protocol") is not True:
        raise RuntimeError("Prepared TRAIN-holdout baseline verification is not sealed")
    checkpoint = resolve_checkpoint(manifest.get("checkpoint_path"))
    if sha256_file(checkpoint) != EXPECTED_CLEAN_SHA256:
        raise RuntimeError("Clean Stage-I checkpoint checksum mismatch")
    _, clean_classifier, _ = build_models(checkpoint, torch.device("cpu"))
    clean_initial_classifier_sha256 = state_hash(clean_classifier)
    del clean_classifier
    counts = np.asarray(manifest.get("train_class_counts"), dtype=np.int64)
    if counts.shape != (NUM_CLASSES,) or np.any(counts <= 0):
        raise RuntimeError("Invalid sealed 100-class TRAIN counts")
    if not np.any(counts > MANY_THRESHOLD) or not np.any(counts <= MANY_THRESHOLD):
        raise RuntimeError("TPR requires both Many and non-Many classes")
    required = (
        "train_images.npy", "train_labels.npy", "validation_images.npy",
        "validation_gap_features.npy", "validation_labels.npy",
        "sampling_plan_seed123.npy", "sampling_plan_seed124.npy",
        "sampling_plan_seed125.npy",
    )
    for filename in required:
        if not (ROOT / "prepared" / filename).is_file():
            raise FileNotFoundError(f"Missing sealed prepared input: {filename}")
    c2_hashes = {}
    train_labels = np.load(ROOT / "prepared" / "train_labels.npy", allow_pickle=False)
    observed_counts = np.bincount(np.asarray(train_labels, dtype=np.int64), minlength=NUM_CLASSES)
    if not np.array_equal(observed_counts, counts):
        raise RuntimeError("Prepared TRAIN labels do not match the sealed class-count manifest")
    for seed in SEEDS:
        _, _, c2_sha, _ = load_c2_base(seed, clean_initial_classifier_sha256)
        c2_hashes[str(seed)] = c2_sha
        plan = np.load(
            ROOT / "prepared" / f"sampling_plan_seed{seed}.npy",
            mmap_mode="r", allow_pickle=False,
        )
        if plan.shape != (STAGE2_EPOCHS, len(train_labels)):
            raise RuntimeError(f"Invalid sealed sampling plan shape for seed {seed}")
        if plan.dtype != np.int32:
            raise RuntimeError(f"Invalid sealed sampling plan dtype for seed {seed}")
        validate_class_aware_plan(train_labels, seed, plan)
        c1_trace = ROOT / "results" / f"seed_{seed}" / "C1_CCSAM_FEATURE_GRAD" / "batch_trace.csv"
        c4_trace = AB_DIR / "results" / f"seed_{seed}" / C4_ARM / "batch_trace.csv"
        if len(read_trace(c1_trace)) != STAGE2_EPOCHS * 170:
            raise RuntimeError(f"Original C1 trace length mismatch for seed {seed}")
        if len(read_trace(c4_trace)) != STAGE2_EPOCHS * 170:
            raise RuntimeError(f"Archived C4 trace length mismatch for seed {seed}")
    protected = protected_snapshot()
    return {
        "checked_at": stamp(),
        "passed": True,
        "arm": TPR_ARM,
        "clean_stage1_sha256": EXPECTED_CLEAN_SHA256,
        "seeds": list(SEEDS),
        "epochs_per_seed": STAGE2_EPOCHS,
        "batch_size": BATCH_SIZE,
        "steps_per_epoch": 170,
        "train_samples": int(len(train_labels)),
        "validation_samples": 1980,
        "many_rule": "sealed TRAIN count strictly greater than 100",
        "many_class_ids": np.flatnonzero(counts > MANY_THRESHOLD).tolist(),
        "tail_parameter_class_ids": np.flatnonzero(counts <= MANY_THRESHOLD).tolist(),
        "base_arm": C2_ARM,
        "base_checkpoint_sha256": c2_hashes,
        "rank": TPR_RANK,
        "tail_neighbors": TAIL_NEIGHBORS,
        "more_amplitude_per_class": MORE_AMPLITUDE_PER_CLASS,
        "more_peak": MORE_PEAK,
        "orthogonality_lambda": ORTHOGONALITY_LAMBDA,
        "tail_diversity_lambda": TAIL_DIVERSITY_LAMBDA,
        "official_TEST_used": False,
        "protected_inputs": protected,
    }


def compare_preflight(current: dict, saved: dict) -> None:
    a = {key: value for key, value in current.items() if key != "checked_at"}
    b = {key: value for key, value in saved.items() if key != "checked_at"}
    if a != b:
        raise RuntimeError("TPR preflight snapshot changed; do not refresh or bypass it")


def trace_row(epoch, batch_index, indices, target_np, images_cpu, gap):
    return {
        "epoch": str(epoch),
        "batch": str(batch_index),
        "batch_size": str(len(indices)),
        "indices_labels_sha256": hash_arrays(indices, target_np),
        "augmented_images_sha256": hash_arrays(images_cpu.numpy()),
        "gap_features_sha256": hash_arrays(gap.detach().cpu().numpy()),
    }


def smoke_one_seed(
    seed, manifest, train_images, train_labels, train_transform,
    graph, device,
):
    checkpoint_path = resolve_checkpoint(manifest["checkpoint_path"])
    seed_everything(seed, deterministic=True)
    backbone, clean_classifier, _ = build_models(checkpoint_path, device)
    frozen_backbone_hash = state_hash(backbone)
    clean_initial_classifier_sha256 = state_hash(clean_classifier)
    base_weight, base_bias, base_sha, _ = load_c2_base(seed, clean_initial_classifier_sha256)
    tail_ids = torch.as_tensor(
        np.asarray(graph["tail_class_ids"], dtype=np.int64),
        dtype=torch.long, device=device,
    )
    model = LowRankTailClassifier(base_weight.to(device), base_bias.to(device), tail_ids).to(device)
    base_hash = model.base_hash()
    initial_model_hash = state_hash(model)
    counts = np.asarray(manifest["train_class_counts"], dtype=np.int64)
    weights = class_weights_from_counts(counts, device)
    edge_rows = validate_neighbor_graph(graph, counts).to(device)
    plan = np.load(ROOT / "prepared" / f"sampling_plan_seed{seed}.npy", mmap_mode="r", allow_pickle=False)
    c1_trace = read_trace(
        ROOT / "results" / f"seed_{seed}" / "C1_CCSAM_FEATURE_GRAD" / "batch_trace.csv"
    )
    c4_trace = read_trace(
        AB_DIR / "results" / f"seed_{seed}" / C4_ARM / "batch_trace.csv"
    )
    generator = torch.Generator(device="cpu").manual_seed(seed + 300009)
    reports = []
    optimizer = None
    for batch_id in (0, 1):
        start = batch_id * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(train_labels))
        indices = np.asarray(plan[0, start:end], dtype=np.int64)
        target_np = np.asarray(train_labels[indices], dtype=np.int64)
        target = torch.as_tensor(target_np, dtype=torch.long, device=device)
        images_cpu = augment_batch(train_images, indices, train_transform, generator)
        with torch.no_grad():
            gap = clean_compatible_gap(backbone(images_cpu.to(device)))
        current_trace = trace_row(1, batch_id, indices, target_np, images_cpu, gap)
        if any(current_trace[key] != c1_trace[batch_id][key] for key in TRACE_FIELDS):
            raise RuntimeError(f"TPR smoke differs from C1 input trace: seed={seed}, batch={batch_id}")
        if any(current_trace[key] != c4_trace[batch_id][key] for key in TRACE_FIELDS):
            raise RuntimeError(f"TPR smoke differs from C4 input trace: seed={seed}, batch={batch_id}")

        c1_loss, logits_clean, _, _, delta, pstats, _ = compute_arm_loss(
            "C1_CCSAM_FEATURE_GRAD", gap, target, model, weights, 0
        )
        gradient_check = verify_analytic_gradient(gap, target, model, weights) if batch_id == 0 else None
        more_loss, delta_logits = more_discrepancy_loss(model, gap.detach(), target, counts)
        orth_loss = model.orthogonality_loss()
        div_loss = model.diversity_loss(edge_rows)
        more_weight = sinusoidal_more_weight(batch_id, STAGE2_EPOCHS * 170)
        total_loss = (
            c1_loss
            + more_weight * more_loss
            + ORTHOGONALITY_LAMBDA * orth_loss
            + TAIL_DIVERSITY_LAMBDA * div_loss
        )
        if not torch.isfinite(total_loss):
            raise RuntimeError("TPR smoke produced a non-finite total loss")
        many_mask = torch.as_tensor(counts, dtype=torch.long, device=device)[target] > MANY_THRESHOLD
        if batch_id == 1 and (not bool(many_mask.any()) or bool(many_mask.all())):
            raise RuntimeError("Second smoke batch must contain both Many and non-Many samples")
        reports.append({
            "batch": batch_id,
            "paired_c1_and_c4_trace_exact": True,
            "c1_base_loss": float(c1_loss.detach().cpu()),
            "more_loss": float(more_loss.detach().cpu()),
            "more_weight": float(more_weight),
            "weighted_more_loss": float((more_weight * more_loss).detach().cpu()),
            "orthogonality_loss": float(orth_loss.detach().cpu()),
            "tail_diversity_loss": float(div_loss.detach().cpu()),
            "total_loss": float(total_loss.detach().cpu()),
            "many_samples": int(many_mask.sum().item()),
            "nonmany_samples": int((~many_mask).sum().item()),
            "mean_feature_delta_l2": float(pstats["mean_delta_l2_per_sample"]),
            "mean_logit_delta_l2": float(torch.linalg.vector_norm(delta_logits, dim=1).mean().detach().cpu()),
            "analytic_gradient_check": gradient_check,
        })
        if batch_id == 1:
            optimizer = torch.optim.SGD(
                model.parameters(), lr=STAGE2_LR, momentum=MOMENTUM,
                weight_decay=WEIGHT_DECAY, nesterov=False,
            )
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError("TPR adapter gradients are missing or non-finite")
            if any(p.grad is not None for p in backbone.parameters()):
                raise RuntimeError("Frozen backbone received a TPR smoke gradient")
            optimizer.step()
            if state_hash(model) == initial_model_hash:
                raise RuntimeError("TPR smoke did not update the low-rank adapter")
            if model.base_hash() != base_hash:
                raise RuntimeError("Frozen C2 base classifier changed during TPR smoke")
            many_output_rows = torch.ones(NUM_CLASSES, dtype=torch.bool, device=device)
            many_output_rows[model.tail_indices] = False
            if not torch.equal(model.weight[many_output_rows], model.base_weight[many_output_rows]):
                raise RuntimeError("TPR residual modified a Many output row")
            check_backbone_unchanged(backbone, {
                key: value.detach().cpu().clone()
                for key, value in backbone.state_dict().items()
            })
            if state_hash(backbone) != frozen_backbone_hash:
                raise RuntimeError("Frozen Clean Stage-I backbone changed during TPR smoke")

    result = {
        "seed": int(seed),
        "c2_base_checkpoint_sha256": base_sha,
        "paired_with_c1_and_c4": True,
        "batches": reports,
        "adapter_updated": True,
        "c2_base_bitwise_unchanged": True,
        "many_output_rows_unchanged": True,
        "backbone_and_bn_bitwise_unchanged": True,
    }
    del optimizer, backbone, clean_classifier, model
    torch.cuda.empty_cache()
    return result


def smoke_all(manifest, train_images, train_labels, train_transform, eval_transform, device, preflight):
    graph_path = TPR_DIR / "tail_neighbor_graph.json"
    smoke_path = TPR_DIR / "smoke_report.json"
    if graph_path.exists() or smoke_path.exists():
        raise FileExistsError("TPR graph or Smoke artifact already exists; do not overwrite or rerun")
    seed = SEEDS[0]
    seed_everything(seed, deterministic=True)
    checkpoint_path = resolve_checkpoint(manifest["checkpoint_path"])
    backbone, clean_classifier, _ = build_models(checkpoint_path, device)
    graph = build_tail_neighbor_graph(
        backbone, train_images, train_labels,
        np.asarray(manifest["train_class_counts"], dtype=np.int64),
        eval_transform, device,
    )
    graph["graph_sha256"] = hash_arrays(
        np.asarray(graph["tail_class_ids"], dtype=np.int64),
        np.asarray(graph["edge_rows"], dtype=np.int64),
    )
    exclusive_json(graph_path, graph)
    del backbone, clean_classifier
    torch.cuda.empty_cache()

    reports = [
        smoke_one_seed(seed, manifest, train_images, train_labels, train_transform, graph, device)
        for seed in SEEDS
    ]
    compare_protected(preflight["protected_inputs"])
    guard = guard_pass()
    if guard.get("official_TEST_used") or guard.get("blocked_attempts"):
        raise RuntimeError("Official TEST guard violation during TPR smoke")
    report = {
        "passed": True,
        "completed_at": stamp(),
        "arm": TPR_ARM,
        "seeds": list(SEEDS),
        "batches_per_seed": 2,
        "formal_training_started": False,
        "graph_sha256": graph["graph_sha256"],
        "graph_centroid_sha256": graph["centroid_sha256"],
        "rank": TPR_RANK,
        "more_peak": MORE_PEAK,
        "orthogonality_lambda": ORTHOGONALITY_LAMBDA,
        "tail_diversity_lambda": TAIL_DIVERSITY_LAMBDA,
        "checks": reports,
        "graph_file_sha256": sha256_file(graph_path),
        "preflight_report_sha256": sha256_file(TPR_DIR / "preflight_report.json"),
        "source_sha256": sha256_file(TPR_DIR / "train_tpr.py"),
        "protocol_sha256": sha256_file(TPR_DIR / "TPR_PROTOCOL.md"),
        "official_TEST_used": False,
        "test_guard": guard,
    }
    exclusive_json(smoke_path, report)
    return report


def train_one_seed(
    seed, manifest, train_images, train_labels, validation_gap,
    validation_labels, train_transform, graph, device, base_sha_expected,
):
    out = TPR_DIR / "results" / f"seed_{seed}" / TPR_ARM
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing TPR run: {out}")
    out.mkdir(parents=True, exist_ok=False)
    seed_everything(seed, deterministic=True)
    checkpoint_path = resolve_checkpoint(manifest["checkpoint_path"])
    backbone, clean_classifier, _ = build_models(checkpoint_path, device)
    clean_initial_classifier_sha256 = state_hash(clean_classifier)
    frozen_backbone = {
        key: value.detach().cpu().clone()
        for key, value in backbone.state_dict().items()
    }
    base_weight, base_bias, base_sha, base_checkpoint = load_c2_base(
        seed, clean_initial_classifier_sha256
    )
    if base_sha != base_sha_expected:
        raise RuntimeError(f"C2 base checkpoint changed after preflight for seed {seed}")
    tail_ids = torch.as_tensor(
        np.asarray(graph["tail_class_ids"], dtype=np.int64),
        dtype=torch.long, device=device,
    )
    model = LowRankTailClassifier(base_weight.to(device), base_bias.to(device), tail_ids).to(device)
    base_hash = model.base_hash()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=STAGE2_LR, momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY, nesterov=False,
    )
    if optimizer.state:
        raise RuntimeError("TPR optimizer must start with empty momentum state")
    counts = np.asarray(manifest["train_class_counts"], dtype=np.int64)
    class_weights = class_weights_from_counts(counts, device)
    edge_rows = validate_neighbor_graph(graph, counts).to(device)
    base_validation = evaluate_base(
        model.base_weight, model.base_bias, validation_gap,
        validation_labels, counts, device,
    )
    reference_validation = base_checkpoint.get("final_validation")
    if not isinstance(reference_validation, dict):
        raise RuntimeError("C2 checkpoint is missing its fixed-epoch validation metrics")
    for metric in ("Overall", "Macro", "Many", "MediumN", "Few", "Head", "OfficialMedium", "Tail"):
        if abs(float(base_validation[metric]) - float(reference_validation[metric])) > 1e-5:
            raise RuntimeError(
                f"Recomputed frozen C2 validation metric differs for {metric}: "
                f"{base_validation[metric]} vs {reference_validation[metric]}"
            )
    plan = np.load(ROOT / "prepared" / f"sampling_plan_seed{seed}.npy", mmap_mode="r", allow_pickle=False)
    c1_trace_path = ROOT / "results" / f"seed_{seed}" / "C1_CCSAM_FEATURE_GRAD" / "batch_trace.csv"
    reference_rows = read_trace(c1_trace_path)
    expected_batches = STAGE2_EPOCHS * 170
    if len(reference_rows) != expected_batches:
        raise RuntimeError(f"Original C1 trace length mismatch for seed {seed}")
    augmentation_generator = torch.Generator(device="cpu").manual_seed(seed + 300009)
    trace_path = out / "batch_trace.csv"
    metrics_path = out / "metrics.csv"
    metric_fields = [
        "epoch", "alpha_ccsam", "mean_alpha_more", "max_alpha_more",
        "learning_rate", "train_total_loss", "train_c1_base_loss",
        "train_more_loss", "train_weighted_more_loss",
        "train_orthogonality_loss", "train_tail_diversity_loss",
        "train_clean_accuracy", "validation_Overall", "validation_Macro",
        "validation_Head", "validation_OfficialMedium", "validation_Tail",
        "validation_Many", "validation_MediumN", "validation_Few",
        "base_validation_Overall", "base_validation_Many", "base_validation_Few",
        "mean_feature_delta_l2", "mean_logit_delta_l2",
        "mean_logit_delta_l2_many", "mean_logit_delta_l2_medium",
        "mean_logit_delta_l2_few", "mean_tail_weight_norm",
        "mean_orthogonality_cosine_squared", "mean_neighbor_residual_cosine_squared",
        "steps", "samples",
    ]
    status = {
        "state": "running",
        "seed": int(seed),
        "arm": TPR_ARM,
        "started_at": stamp(),
        "clean_stage1_sha256": manifest["clean_stage1_sha256"],
        "base_arm": C2_ARM,
        "base_checkpoint_sha256": base_sha,
        "base_classifier_frozen": True,
        "base_initial_classifier_sha256": base_checkpoint.get("initial_classifier_sha256"),
        "base_validation": base_validation,
        "rank": TPR_RANK,
        "tail_class_ids": graph["tail_class_ids"],
        "tail_neighbor_graph_sha256": graph["graph_sha256"],
        "tail_neighbor_centroid_sha256": graph["centroid_sha256"],
        "more_discrepancy": "class-prior-weighted squared L2 distance between full and frozen-base 100-class logits",
        "more_amplitude_per_class": MORE_AMPLITUDE_PER_CLASS,
        "more_peak": MORE_PEAK,
        "orthogonality_lambda": ORTHOGONALITY_LAMBDA,
        "tail_diversity_lambda": TAIL_DIVERSITY_LAMBDA,
        "optimizer": {
            "type": "SGD", "lr": STAGE2_LR, "momentum": MOMENTUM,
            "weight_decay": WEIGHT_DECAY, "nesterov": False,
        },
        "backbone_frozen": True,
        "bn_eval_and_buffers_frozen": True,
        "official_TEST_used": False,
        "epochs": [],
    }
    write_json(out / "status.json", status)
    global_batch = 0
    with trace_path.open("x", encoding="utf-8", newline="") as trace_file, \
         metrics_path.open("x", encoding="utf-8", newline="") as metric_file:
        trace_writer = csv.DictWriter(trace_file, fieldnames=list(TRACE_FIELDS))
        metric_writer = csv.DictWriter(metric_file, fieldnames=metric_fields)
        trace_writer.writeheader()
        metric_writer.writeheader()
        total_steps = STAGE2_EPOCHS * 170
        for epoch in range(STAGE2_EPOCHS):
            model.train()
            sums = {
                "total": 0.0, "c1": 0.0, "more": 0.0, "weighted_more": 0.0,
                "orth": 0.0, "div": 0.0, "feature_delta": 0.0,
                "logit_delta": 0.0, "logit_many": 0.0,
                "logit_medium": 0.0, "logit_few": 0.0,
                "tail_weight_norm": 0.0, "orth_cos2": 0.0, "neighbor_residual_cos2": 0.0,
            }
            group_counts = {"many": 0, "medium": 0, "few": 0}
            correct = 0
            seen = 0
            alpha_values = []
            epoch_steps = 0
            for batch_index, start in enumerate(range(0, len(train_labels), BATCH_SIZE)):
                end = min(start + BATCH_SIZE, len(train_labels))
                indices = np.asarray(plan[epoch, start:end], dtype=np.int64)
                target_np = np.asarray(train_labels[indices], dtype=np.int64)
                target = torch.as_tensor(target_np, dtype=torch.long, device=device)
                images_cpu = augment_batch(
                    train_images, indices, train_transform, augmentation_generator
                )
                with torch.no_grad():
                    gap = clean_compatible_gap(backbone(images_cpu.to(device, non_blocking=True)))
                if gap.shape != (end - start, FEATURE_DIM) or not torch.isfinite(gap).all():
                    raise RuntimeError("Malformed or non-finite TPR GAP features")
                current_trace = trace_row(epoch + 1, batch_index, indices, target_np, images_cpu, gap)
                expected = reference_rows[global_batch]
                if any(current_trace[key] != expected[key] for key in TRACE_FIELDS):
                    raise RuntimeError(
                        f"TPR/C1 paired input mismatch: seed={seed}, epoch={epoch + 1}, batch={batch_index}"
                    )
                trace_writer.writerow(current_trace)
                c1_loss, logits_clean, _, _, delta, pstats, alpha_ccsam = compute_arm_loss(
                    "C1_CCSAM_FEATURE_GRAD", gap, target, model,
                    class_weights, epoch,
                )
                more_loss, delta_logits = more_discrepancy_loss(model, gap.detach(), target, counts)
                orth_loss = model.orthogonality_loss()
                div_loss = model.diversity_loss(edge_rows)
                global_step = epoch * 170 + batch_index
                alpha_more = sinusoidal_more_weight(global_step, total_steps)
                total_loss = (
                    c1_loss
                    + alpha_more * more_loss
                    + ORTHOGONALITY_LAMBDA * orth_loss
                    + TAIL_DIVERSITY_LAMBDA * div_loss
                )
                if not torch.isfinite(total_loss):
                    raise RuntimeError("Non-finite TPR total loss")
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise RuntimeError(f"Missing/non-finite TPR adapter gradient at seed={seed}")
                if any(p.grad is not None for p in backbone.parameters()):
                    raise RuntimeError("Frozen Clean Stage-I backbone received a TPR gradient")
                optimizer.step()
                batch_size = end - start
                norms = torch.linalg.vector_norm(delta_logits.detach(), dim=1)
                sample_many = counts[target.detach().cpu().numpy()] > MANY_THRESHOLD
                sample_medium = (counts[target.detach().cpu().numpy()] >= 20) & (~sample_many)
                sample_few = counts[target.detach().cpu().numpy()] < 20
                for name, mask_np in (
                    ("many", sample_many), ("medium", sample_medium), ("few", sample_few)
                ):
                    n = int(mask_np.sum())
                    group_counts[name] += n
                    if n:
                        sums[f"logit_{name}"] += float(norms[torch.as_tensor(mask_np, device=device)].sum().cpu())
                sums["total"] += float(total_loss.detach().cpu()) * batch_size
                sums["c1"] += float(c1_loss.detach().cpu()) * batch_size
                sums["more"] += float(more_loss.detach().cpu()) * batch_size
                sums["weighted_more"] += float((alpha_more * more_loss).detach().cpu()) * batch_size
                sums["orth"] += float(orth_loss.detach().cpu()) * batch_size
                sums["div"] += float(div_loss.detach().cpu()) * batch_size
                sums["feature_delta"] += pstats["mean_delta_l2_per_sample"] * batch_size
                sums["logit_delta"] += float(norms.mean().cpu()) * batch_size
                sums["tail_weight_norm"] += float(torch.linalg.vector_norm(model.tail_residual()).detach().cpu())
                base_rows = F.normalize(model.base_weight[model.tail_indices], dim=1, eps=1e-8)
                residual_rows = F.normalize(model.tail_residual(), dim=1, eps=1e-8)
                sums["orth_cos2"] += float(((base_rows * residual_rows).sum(1).square().mean()).detach().cpu())
                sums["neighbor_residual_cos2"] += float((
                    (residual_rows[edge_rows[:, 0]] * residual_rows[edge_rows[:, 1]]).sum(1).square().mean()
                ).detach().cpu())
                correct += int((logits_clean.detach().argmax(dim=1) == target).sum().item())
                seen += batch_size
                alpha_values.append(alpha_more)
                epoch_steps += 1
                global_batch += 1
            if seen != len(train_labels) or epoch_steps != 170:
                raise RuntimeError(f"TPR epoch accounting mismatch: samples={seen}, steps={epoch_steps}")
            check_backbone_unchanged(backbone, frozen_backbone)
            if model.base_hash() != base_hash:
                raise RuntimeError("Frozen C2 base classifier changed during TPR formal training")
            many_rows = torch.ones(NUM_CLASSES, dtype=torch.bool, device=device)
            many_rows[model.tail_indices] = False
            if not torch.equal(model.weight[many_rows], model.base_weight[many_rows]):
                raise RuntimeError("TPR modified a Many output classifier row")
            validation = evaluate_classifier(model, validation_gap, validation_labels, counts, device)
            row = {
                "epoch": epoch + 1,
                "alpha_ccsam": float(alpha_for_epoch(epoch)),
                "mean_alpha_more": float(np.mean(alpha_values)),
                "max_alpha_more": float(np.max(alpha_values)),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "train_total_loss": sums["total"] / seen,
                "train_c1_base_loss": sums["c1"] / seen,
                "train_more_loss": sums["more"] / seen,
                "train_weighted_more_loss": sums["weighted_more"] / seen,
                "train_orthogonality_loss": sums["orth"] / seen,
                "train_tail_diversity_loss": sums["div"] / seen,
                "train_clean_accuracy": 100.0 * correct / seen,
                "validation_Overall": validation["Overall"],
                "validation_Macro": validation["Macro"],
                "validation_Head": validation["Head"],
                "validation_OfficialMedium": validation["OfficialMedium"],
                "validation_Tail": validation["Tail"],
                "validation_Many": validation["Many"],
                "validation_MediumN": validation["MediumN"],
                "validation_Few": validation["Few"],
                "base_validation_Overall": base_validation["Overall"],
                "base_validation_Many": base_validation["Many"],
                "base_validation_Few": base_validation["Few"],
                "mean_feature_delta_l2": sums["feature_delta"] / seen,
                "mean_logit_delta_l2": sums["logit_delta"] / seen,
                "mean_logit_delta_l2_many": sums["logit_many"] / max(group_counts["many"], 1),
                "mean_logit_delta_l2_medium": sums["logit_medium"] / max(group_counts["medium"], 1),
                "mean_logit_delta_l2_few": sums["logit_few"] / max(group_counts["few"], 1),
                "mean_tail_weight_norm": sums["tail_weight_norm"] / epoch_steps,
                "mean_orthogonality_cosine_squared": sums["orth_cos2"] / epoch_steps,
                "mean_neighbor_residual_cosine_squared": sums["neighbor_residual_cos2"] / epoch_steps,
                "steps": epoch_steps,
                "samples": seen,
            }
            metric_writer.writerow(row)
            metric_file.flush()
            trace_file.flush()
            status["epochs"].append({
                "epoch": epoch + 1,
                "metrics": row,
                "validation": validation,
                "base_validation": base_validation,
                "frozen_base_classifier_unchanged": True,
                "frozen_backbone_and_bn_bitwise_unchanged": True,
                "many_output_rows_unchanged": True,
            })
            write_json(out / "status.json", status)
            print(
                f"seed={seed} TPR epoch={epoch + 1}/{STAGE2_EPOCHS} "
                f"loss={row['train_total_loss']:.5f} more={row['train_weighted_more_loss']:.5f} "
                f"overall={row['validation_Overall']:.3f} many={row['validation_Many']:.3f} "
                f"few={row['validation_Few']:.3f}",
                flush=True,
            )
    check_backbone_unchanged(backbone, frozen_backbone)
    if model.base_hash() != base_hash:
        raise RuntimeError("Frozen C2 base classifier changed before checkpoint save")
    final_validation = status["epochs"][-1]["validation"]
    payload = {
        "epoch": STAGE2_EPOCHS,
        "seed": int(seed),
        "arm": TPR_ARM,
        "base_arm": C2_ARM,
        "base_checkpoint_sha256": base_sha,
        "state_dict_classifier": model.merged_state_dict(),
        "adapter_state": {
            "A": model.A.detach().cpu().clone(),
            "B": model.B.detach().cpu().clone(),
        },
        "tail_class_ids": graph["tail_class_ids"],
        "tail_neighbor_graph_sha256": graph["graph_sha256"],
        "rank": TPR_RANK,
        "clean_stage1_sha256": manifest["clean_stage1_sha256"],
        "final_validation": final_validation,
        "primary_epoch_fixed": True,
        "official_TEST_used": False,
        "backbone_retrained": False,
        "base_classifier_frozen": True,
    }
    torch.save(payload, out / "final_classifier.pth")
    status["state"] = "completed"
    status["completed_at"] = stamp()
    status["final_epoch"] = STAGE2_EPOCHS
    status["final_validation"] = final_validation
    status["final_classifier_sha256"] = sha256_file(out / "final_classifier.pth")
    status["official_TEST_used"] = False
    status["base_classifier_frozen"] = True
    write_json(out / "status.json", status)
    del backbone, clean_classifier, model, optimizer
    torch.cuda.empty_cache()
    return status


def main():
    parser = argparse.ArgumentParser(description="CC-SAM-TPR: MORE-inspired tail parameter reallocation")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--smoke-only", action="store_true")
    mode.add_argument("--formal", action="store_true")
    args = parser.parse_args()

    manifest_path = ROOT / "prepared" / "prepared_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Sealed prepared manifest is missing")
    preliminary = read_json(manifest_path)
    if preliminary.get("official_TEST_used") is not False:
        raise RuntimeError("Prepared manifest does not certify TEST-free preparation")
    data_root = Path(preliminary["official_train_pickle"]).resolve().parent.parent
    install_test_guard(data_root)
    manifest, train_images, train_labels, validation_gap, validation_labels = load_prepared()
    train_transform, eval_transform, transform_sha = build_transforms()
    if transform_sha != manifest["transform_source_sha256"]:
        raise RuntimeError("Audited CIFAR transform source changed after preparation")

    current_preflight = collect_preflight(manifest)
    preflight_path = TPR_DIR / "preflight_report.json"
    if args.preflight:
        if preflight_path.exists():
            raise FileExistsError("TPR preflight report already exists; do not overwrite")
        exclusive_json(preflight_path, current_preflight)
        print("TPR PREFLIGHT PASS; C0-C4, C2 bases, data, traces and source hashes are sealed.")
        return
    if not preflight_path.is_file():
        raise FileNotFoundError("Run --preflight first")
    saved_preflight = read_json(preflight_path)
    compare_preflight(current_preflight, saved_preflight)

    if args.smoke_only:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; no CPU fallback")
        smoke_all(
            manifest, train_images, train_labels, train_transform,
            eval_transform, torch.device("cuda:0"),
            saved_preflight["protected_inputs"],
        )
        print("TPR SMOKE PASS for all three seeds; formal training has not started.")
        return

    smoke_path = TPR_DIR / "smoke_report.json"
    graph_path = TPR_DIR / "tail_neighbor_graph.json"
    if not smoke_path.is_file() or not graph_path.is_file():
        raise FileNotFoundError("Successful TPR Smoke and train-only neighbor graph are required")
    smoke = read_json(smoke_path)
    graph = read_json(graph_path)
    graph_edges = validate_neighbor_graph(graph, np.asarray(manifest["train_class_counts"], dtype=np.int64))
    if (
        smoke.get("passed") is not True
        or smoke.get("arm") != TPR_ARM
        or smoke.get("seeds") != list(SEEDS)
        or smoke.get("graph_sha256") != graph.get("graph_sha256")
        or smoke.get("graph_file_sha256") != sha256_file(graph_path)
        or smoke.get("preflight_report_sha256") != sha256_file(preflight_path)
        or smoke.get("source_sha256") != sha256_file(TPR_DIR / "train_tpr.py")
        or smoke.get("protocol_sha256") != sha256_file(TPR_DIR / "TPR_PROTOCOL.md")
        or smoke.get("official_TEST_used") is not False
        or len(smoke.get("checks", [])) != len(SEEDS)
    ):
        raise RuntimeError("TPR Smoke report, graph, source or provenance is invalid")
    if (TPR_DIR / "formal_completed.json").exists():
        raise FileExistsError("TPR formal completion marker already exists")
    for seed in SEEDS:
        if (TPR_DIR / "results" / f"seed_{seed}" / TPR_ARM).exists():
            raise FileExistsError(f"TPR output already exists for seed {seed}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no CPU fallback")
    compare_protected(saved_preflight["protected_inputs"])

    base_hashes = saved_preflight["base_checkpoint_sha256"]
    completed = []
    for seed in SEEDS:
        status = train_one_seed(
            seed, manifest, train_images, train_labels, validation_gap,
            validation_labels, train_transform, graph, torch.device("cuda:0"),
            base_hashes[str(seed)],
        )
        if status.get("state") != "completed" or status.get("final_epoch") != STAGE2_EPOCHS:
            raise RuntimeError(f"TPR did not complete 30 epochs for seed {seed}")
        completed.append({
            "seed": seed,
            "epoch": status["final_epoch"],
            "final_classifier_sha256": status["final_classifier_sha256"],
            "base_checkpoint_sha256": status["base_checkpoint_sha256"],
        })
        compare_protected(saved_preflight["protected_inputs"])

    guard = guard_pass()
    if guard.get("official_TEST_used") or guard.get("blocked_attempts"):
        raise RuntimeError("Official TEST guard violation during TPR formal training")
    compare_protected(saved_preflight["protected_inputs"])
    result = {
        "passed": True,
        "arm": TPR_ARM,
        "completed_at": stamp(),
        "seeds": list(SEEDS),
        "epochs_per_seed": STAGE2_EPOCHS,
        "new_epochs": len(SEEDS) * STAGE2_EPOCHS,
        "optimizer_steps": len(SEEDS) * STAGE2_EPOCHS * 170,
        "only_tpr_new_training": True,
        "original_c0_c1_c2_c3_c4_unchanged": True,
        "runs": completed,
        "rank": TPR_RANK,
        "more_peak": MORE_PEAK,
        "orthogonality_lambda": ORTHOGONALITY_LAMBDA,
        "tail_diversity_lambda": TAIL_DIVERSITY_LAMBDA,
        "official_TEST_used": False,
        "test_guard": guard,
        "preflight_report_sha256": sha256_file(preflight_path),
        "smoke_report_sha256": sha256_file(smoke_path),
        "graph_file_sha256": sha256_file(graph_path),
        "source_sha256": sha256_file(TPR_DIR / "train_tpr.py"),
        "protocol_sha256": sha256_file(TPR_DIR / "TPR_PROTOCOL.md"),
    }
    exclusive_json(TPR_DIR / "formal_completed.json", result)
    print("TPR FORMAL PASS: 3 seeds, 90 epochs, 15,300 optimizer steps.")
    print("Run analyze_tpr.py only after formal_completed.json is present.")


if __name__ == "__main__":
    main()
