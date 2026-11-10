"""Gated CC-SAM-TPR preparation, Smoke, and three-seed Stage-II training."""
from __future__ import annotations

import argparse
import csv
import json

import numpy as np
import torch
import torch.nn.functional as F

from tpr_inputs import (
    ARM, C1, OUT, STEPS, c, collect_inputs, csv_rows, historical_dir, load_base,
    preflight_marker, protected_snapshot, rel, require, run_dir,
)
from tpr_model import (
    TailLowRankResidual, confusion_aware_diversity_loss, dominant_right_subspace,
    dominant_subspace_overlap, many_head_preservation_kl, select_confusion_pairs,
)
from train_ccsam import compute_arm_loss, verify_analytic_gradient

RANK = 8
TEMPERATURE = 2.0
WEIGHTS = {"hp": 1.0, "sub": 0.01, "div": 0.01}
OPTIMIZER = {"lr": c.STAGE2_LR, "momentum": c.MOMENTUM, "weight_decay": c.WEIGHT_DECAY,
             "nesterov": False}
TRACE_FIELDS = (
    "epoch", "batch", "batch_size", "indices_labels_sha256",
    "augmented_images_sha256", "gap_features_sha256",
)
LOSS_FIELDS = ("total", "c1", "hp", "sub", "div", "weighted_hp", "weighted_sub", "weighted_div")
BATCH_SCALARS = LOSS_FIELDS + ("grad_norm_a", "grad_norm_b", "delta_weight_norm",
                              "mean_feature_delta_l2", "mean_relative_feature_delta_l2")
PREPARATION = OUT / "train_confusion_graphs.json"


class EffectiveLinear:
    """One merged weight reused by both C1 forwards and its analytic delta."""

    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def __call__(self, features):
        return F.linear(features, self.weight, self.bias)


def exclusive_json(path, payload):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def bitwise_equal(left, right):
    return left.dtype == right.dtype and left.shape == right.shape and torch.equal(
        left.detach().contiguous().view(torch.uint8), right.detach().contiguous().view(torch.uint8)
    )


def guard_record():
    guard = c.guard_pass()
    require(guard["official_TEST_used"] is False, "Official TEST was accessed")
    return guard


def environment():
    return {
        "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(), "cuda_available": torch.cuda.is_available(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
    }


def preflight(inputs, snapshot):
    marker = OUT / "preflight_report.json"
    require(not marker.exists(), f"Refusing to overwrite {marker.name}")
    require(not (OUT / "results").exists(), "An existing TPR run occupies the result directory")
    _, initial, _ = c.build_models(inputs.checkpoint, torch.device("cpu"))
    verified = {}
    for seed in c.SEEDS:
        _, sha, init = load_base(seed, initial, torch.device("cpu"), snapshot)
        verified[str(seed)] = {"checkpoint_sha256": sha, "clean_initial_classifier_sha256": init,
                               "final_epoch": c.STAGE2_EPOCHS, "metadata_verified": True}
    exclusive_json(marker, {
        "passed": True, "checked_at": c.now(), "seeds": list(c.SEEDS),
        "snapshot": snapshot, "teachers": verified, "environment": environment(),
        "test_guard": guard_record(), "official_TEST_used": False,
    })
    print("TPR PREFLIGHT PASS: C0-C4 and all three C2 bases verified; TEST=0.", flush=True)


def preparation_provenance(snapshot):
    return {
        "dataset": "prepared Stage-II TRAIN only, unaugmented",
        "train_samples": c.TRAIN_SAMPLES,
        "train_images_sha256": snapshot["sealed_inputs"]["train_images.npy"]["sha256"],
        "train_labels_sha256": snapshot["sealed_inputs"]["train_labels.npy"]["sha256"],
        "clean_stage1_sha256": snapshot["clean_stage1_sha256"],
        "transform_sha256": snapshot["sealed_inputs"]["TRAIN_transform"]["sha256"],
        "protocol_sha256": snapshot["protocol_sha256"],
        "tpr_source_sha256": snapshot["sources"][rel(OUT / "train_tpr.py")],
        "model_source_sha256": snapshot["sources"][rel(OUT / "tpr_model.py")],
    }


def validate_basis(basis, singular):
    require(basis.shape == (c.FEATURE_DIM, RANK) and singular.shape == (c.FEATURE_DIM,)
            and basis.dtype == singular.dtype == torch.float32
            and torch.isfinite(basis).all() and torch.isfinite(singular).all(),
            "Invalid FP32 SVD shape or values")
    ortho = float((basis.T @ basis - torch.eye(RANK)).abs().max())
    gap = float(singular[RANK - 1] - singular[RANK])
    require(ortho <= 1e-5 and singular[0] > 0 and singular[-1] >= 0
            and bool((singular[:-1] >= singular[1:]).all())
            and gap > float(singular[0]) * c.FEATURE_DIM * torch.finfo(torch.float32).eps,
            "Unstable or unresolved dominant rank-8 SVD subspace")
    energy = float(singular[:RANK].square().sum() / singular.square().sum())
    require(0 < energy <= 1, "Invalid top-8 singular energy fraction")
    return {"orthonormality_max_abs_error": ortho, "boundary_singular_value_gap": gap,
            "top8_singular_energy_fraction": energy}


@torch.no_grad()
def make_preparation(inputs, snapshot, device):
    """One clean TRAIN feature pass, followed by three fixed-base predictions."""
    c.seed_everything(c.SEEDS[0], deterministic=True)
    backbone, initial, _ = c.build_models(inputs.checkpoint, device)
    original = c.state_hash(backbone)
    features = c.extract_gap(backbone, initial, inputs.train_images, inputs.clean_transform, device)
    require(features.shape == (c.TRAIN_SAMPLES, c.FEATURE_DIM)
            and features.dtype == np.float32 and np.isfinite(features).all(),
            "Clean TRAIN GAP features failed shape/finiteness")
    runs = {}
    for seed in c.SEEDS:
        base, sha, _ = load_base(seed, initial, device, snapshot)
        frozen_base = c.state_hash(base)
        base_weight = base.weight.detach().cpu()
        # CPU FP32 SVD is prepared once for each seed and reused in Smoke and Formal.
        basis, singular = dominant_right_subspace(base_weight, k=RANK)
        svd_stats = validate_basis(basis, singular)
        predicted = []
        for start in range(0, len(features), 512):
            logits = base(torch.from_numpy(features[start:start + 512]).to(device))
            require(logits.shape[1] == c.NUM_CLASSES and torch.isfinite(logits).all(),
                    f"Nonfinite clean TRAIN base logits for seed {seed}")
            predicted.append(logits.argmax(1).cpu().numpy())
        pred = np.concatenate(predicted)
        confusion = np.bincount(c.NUM_CLASSES * inputs.train_labels.astype(np.int64) + pred,
                                minlength=c.NUM_CLASSES ** 2).reshape(c.NUM_CLASSES, c.NUM_CLASSES)
        require(np.array_equal(confusion.sum(axis=1), inputs.counts), "TRAIN confusion coverage mismatch")
        pairs = select_confusion_pairs(torch.from_numpy(confusion), torch.from_numpy(inputs.counts <= 100))
        runs[str(seed)] = {
            "base_checkpoint_sha256": sha, "base_state_sha256": frozen_base,
            "train_prediction_sha256": c.hash_arrays(pred),
            "confusion": confusion.tolist(), "pairs": pairs, "empty_graph": not pairs,
            "basis": basis.tolist(), "singular_values": singular.tolist(), **svd_stats,
        }
        require(c.state_hash(base) == frozen_base and all(p.grad is None for p in base.parameters()),
                f"Base changed during TRAIN graph construction for seed {seed}")
        print(f"TPR TRAIN graph seed={seed} pairs={len(pairs)} top8_energy={svd_stats['top8_singular_energy_fraction']:.6f}",
              flush=True)
    require(c.state_hash(backbone) == original and all(p.grad is None for p in backbone.parameters()),
            "Backbone changed during TRAIN graph construction")
    exclusive_json(PREPARATION, {
        "passed": True, "created_at": c.now(), "provenance": preparation_provenance(snapshot),
        "train_gap_sha256": c.hash_arrays(features), "runs": runs,
        "test_guard": guard_record(), "official_TEST_used": False,
    })


def load_preparation(inputs, snapshot):
    prepared = c.read_json(PREPARATION)
    require(prepared["passed"] is True and prepared["official_TEST_used"] is False
            and prepared["provenance"] == preparation_provenance(snapshot)
            and list(prepared["runs"]) == [str(seed) for seed in c.SEEDS], "Stale TRAIN graph/SVD preparation")
    for seed in c.SEEDS:
        item = prepared["runs"][str(seed)]
        confusion = np.asarray(item["confusion"])
        require(confusion.shape == (c.NUM_CLASSES, c.NUM_CLASSES) and confusion.dtype.kind in "iu"
                and (confusion >= 0).all() and np.array_equal(confusion.sum(axis=1), inputs.counts),
                f"Invalid sealed TRAIN confusion for seed {seed}")
        pairs = select_confusion_pairs(torch.from_numpy(confusion), torch.from_numpy(inputs.counts <= 100))
        require(pairs == [tuple(pair) for pair in item["pairs"]] and item["empty_graph"] == (not pairs),
                f"Sealed graph pairs disagree with normalized TRAIN confusion for seed {seed}")
        require(item["base_checkpoint_sha256"] == snapshot["teachers"][str(seed)]["sha256"],
                f"TRAIN graph base SHA mismatch for seed {seed}")
        basis = torch.tensor(item["basis"], dtype=torch.float32)
        singular = torch.tensor(item["singular_values"], dtype=torch.float32)
        validate_basis(basis, singular)
    return prepared


def build_run(seed, inputs, snapshot, preparation, device):
    c.seed_everything(seed, deterministic=True)
    backbone, initial, _ = c.build_models(inputs.checkpoint, device)
    base, sha, initial_sha = load_base(seed, initial, device, snapshot)
    graph = preparation["runs"][str(seed)]
    require(c.state_hash(base) == graph["base_state_sha256"], f"Frozen base tensor SHA mismatch for seed {seed}")
    residual = TailLowRankResidual(inputs.counts, feature_dim=c.FEATURE_DIM, rank=RANK, seed=seed).to(device)
    require(tuple(dict(residual.named_parameters())) == ("A", "B")
            and all(p.requires_grad for p in residual.parameters())
            and residual.A.shape == (RANK, c.FEATURE_DIM) and residual.B.shape == (c.NUM_CLASSES, RANK)
            and int(torch.count_nonzero(residual.A)) > 0 and int(torch.count_nonzero(residual.B)) == 0,
            "Invalid A/B-only initialization")
    basis = torch.tensor(graph["basis"], dtype=torch.float32, device=device)
    counts = torch.as_tensor(inputs.counts, device=device)
    optimizer = torch.optim.SGD(residual.parameters(), **OPTIMIZER)
    return backbone, base, residual, basis, counts, optimizer, sha, initial_sha


def paired_batch(backbone, inputs, plan, reference, generator, epoch, batch, device):
    start = batch * c.BATCH_SIZE
    index = np.asarray(plan[epoch, start:start + c.BATCH_SIZE], dtype=np.int64)
    label_array = np.asarray(inputs.train_labels[index], dtype=np.int64)
    labels = torch.from_numpy(label_array).to(device)
    images = c.augment_batch(inputs.train_images, index, inputs.train_transform, generator)
    with torch.no_grad():
        gap = c.clean_compatible_gap(backbone(images.to(device, non_blocking=True)))
    require(torch.isfinite(gap).all(), "Nonfinite paired GAP features")
    row = {
        "epoch": str(epoch + 1), "batch": str(batch), "batch_size": str(len(index)),
        "indices_labels_sha256": c.hash_arrays(index, label_array),
        "augmented_images_sha256": c.hash_arrays(images.numpy()),
        "gap_features_sha256": c.hash_arrays(gap.cpu().numpy()),
    }
    if row != reference[epoch * STEPS + batch]:
        differing = [key for key, value in row.items() if reference[epoch * STEPS + batch].get(key) != value]
        raise RuntimeError(f"C1 paired trace mismatch epoch={epoch + 1} batch={batch}: {differing}")
    return gap, labels, row


def objective(residual, base, basis, pairs, counts, weights, gap, labels, epoch):
    delta = residual.delta_weight()
    head = EffectiveLinear(base.weight + delta, base.bias)
    loss_c1, logits, _, _, feature_delta, perturb, alpha = compute_arm_loss(
        C1, gap, labels, head, weights, epoch
    )
    with torch.no_grad():
        teacher_logits = base(gap)
    hp, mask, many_count = many_head_preservation_kl(logits, teacher_logits, labels, counts, TEMPERATURE)
    sub = dominant_subspace_overlap(delta, basis)
    div = confusion_aware_diversity_loss(delta, pairs)
    weighted = (WEIGHTS["hp"] * hp, WEIGHTS["sub"] * sub, WEIGHTS["div"] * div)
    total = loss_c1 + weighted[0] + weighted[1] + weighted[2]
    values = torch.stack((total, loss_c1, hp, sub, div, *weighted)).detach()
    require(torch.isfinite(values).all() and not feature_delta.requires_grad, "Invalid TPR objective or detached C1 delta")
    stats = dict(zip(LOSS_FIELDS, values.cpu().tolist()))
    with torch.no_grad():
        predictions = logits.argmax(1)
        stats.update(
            many_count=many_count,
            many_agreement_count=int((predictions[mask] == teacher_logits.argmax(1)[mask]).sum()),
            correct=int((predictions == labels).sum()),
            delta_weight_norm=float(torch.linalg.vector_norm(delta)),
            mean_feature_delta_l2=perturb["mean_delta_l2_per_sample"],
            mean_relative_feature_delta_l2=perturb["mean_relative_delta_l2"],
            alpha=alpha,
        )
    return total, stats, logits.detach(), teacher_logits


def check_scale(stats, step):
    if step < 10:
        for name in ("sub", "div"):
            weighted = stats[f"weighted_{name}"]
            require(weighted <= 0.10 * stats["c1"],
                    f"Regularizer scale gate at step={step}: weighted_{name}={weighted}, L_C1={stats['c1']}")


def backward(total, residual, base, backbone, optimizer):
    optimizer.zero_grad(set_to_none=True)
    total.backward()
    norms = {}
    for name, parameter in residual.named_parameters():
        require(parameter.grad is not None and torch.isfinite(parameter.grad).all(), f"Invalid gradient for {name}")
        norms[f"grad_norm_{name.lower()}"] = float(torch.linalg.vector_norm(parameter.grad))
    require(all(p.grad is None and not p.requires_grad for p in base.parameters())
            and all(p.grad is None and not p.requires_grad for p in backbone.parameters()),
            "Frozen base or backbone received gradients")
    return norms


@torch.no_grad()
def check_frozen(backbone, base, residual, frozen):
    require(c.state_hash(backbone) == frozen["backbone"] and c.state_hash(base) == frozen["base"],
            "Frozen backbone/BN or C2 base changed")
    require(all(not module.training for module in backbone.modules()) and not base.training
            and all(p.grad is None and not p.requires_grad for p in list(backbone.parameters()) + list(base.parameters())),
            "Frozen model eval or gradient invariant failed")
    residual.assert_many_rows_zero()
    many = residual.tail_mask == 0
    require(bitwise_equal(residual.effective_weight(base.weight)[many], base.weight[many])
            and torch.isfinite(residual.A).all() and torch.isfinite(residual.B).all(),
            "Many effective rows changed or A/B became nonfinite")


@torch.no_grad()
def evaluate(residual, base, inputs, device):
    head = EffectiveLinear(residual.effective_weight(base.weight), base.bias)
    student, teacher = [], []
    for start in range(0, c.VALIDATION_SAMPLES, 512):
        gap = torch.from_numpy(np.array(inputs.validation_gap[start:start + 512], copy=True)).to(device)
        student.append(head(gap))
        teacher.append(base(gap))
    student_logits, teacher_logits = torch.cat(student), torch.cat(teacher)
    labels = torch.as_tensor(inputs.validation_labels, dtype=torch.long, device=device)
    kd, many, n_many = many_head_preservation_kl(student_logits, teacher_logits, labels, inputs.counts, TEMPERATURE)
    require(n_many > 0 and torch.isfinite(kd), "Invalid Many holdout KL")
    student_pred, teacher_pred = student_logits.argmax(1), teacher_logits.argmax(1)
    student_metrics = c.metrics_from_predictions(student_pred.cpu().numpy(), inputs.validation_labels, inputs.counts)
    teacher_metrics = c.metrics_from_predictions(teacher_pred.cpu().numpy(), inputs.validation_labels, inputs.counts)
    return student_metrics, {
        "many_kl": float(kd), "many_top1_agreement": float((student_pred[many] == teacher_pred[many]).float().mean()),
        "many_samples": n_many, "teacher_many_accuracy": teacher_metrics["Many"],
    }


@torch.no_grad()
def residual_diagnostics(residual, base, basis, pairs):
    delta = residual.delta_weight()
    singular = torch.linalg.svdvals(delta.cpu())
    effective = torch.linalg.svdvals(residual.effective_weight(base.weight).cpu())
    rank = int((singular > singular[0] * max(delta.shape) * torch.finfo(torch.float32).eps).sum())
    require(torch.isfinite(singular).all() and torch.isfinite(effective).all() and 0 < rank <= RANK,
            "Trained residual has invalid numerical rank or spectrum")
    normalized = F.normalize(delta, p=2, dim=1, eps=1e-12)
    return {
        "delta_weight_norm": float(torch.linalg.vector_norm(delta)),
        "dominant_projection_ratio": float(dominant_subspace_overlap(delta, basis)),
        "row_norms": torch.linalg.vector_norm(delta, dim=1).cpu().tolist(),
        "numerical_rank": rank, "residual_singular_values": singular.tolist(),
        "effective_singular_values": effective.tolist(),
        "pair_cosines": [{"class_a": a, "class_b": b, "confusion_score": score,
                          "cosine": float(torch.dot(normalized[a], normalized[b]))} for a, b, score in pairs],
    }



def frozen_hashes(backbone, base):
    return {"backbone": c.state_hash(backbone), "base": c.state_hash(base)}


def check_step_zero(residual, base, logits, teacher):
    delta = residual.delta_weight()
    require(bitwise_equal(delta, torch.zeros_like(delta))
            and bitwise_equal(residual.effective_weight(base.weight), base.weight)
            and bitwise_equal(logits, teacher), "Step-0 effective classifier differs bitwise from C2")


def smoke(inputs, snapshot, preparation, device):
    out = OUT / "smoke_report.json"
    require(not out.exists() and not (OUT / "results").exists(),
            "Existing Smoke or formal outputs; refusing to overwrite")
    checks = []
    for seed in c.SEEDS:
        print(f"TPR SMOKE seed={seed} begin", flush=True)
        backbone, base, residual, basis, counts, optimizer, sha, init = build_run(
            seed, inputs, snapshot, preparation, device
        )
        frozen = frozen_hashes(backbone, base)
        graph = preparation["runs"][str(seed)]
        pairs = graph["pairs"]
        weights = c.class_weights_from_counts(inputs.counts, device)
        reference = csv_rows(historical_dir(seed, C1) / "batch_trace.csv")
        generator = torch.Generator(device="cpu").manual_seed(seed + 300009)
        reports = []
        for step in (0, 1):
            print(f"TPR SMOKE seed={seed} step={step} verify", flush=True)
            gap, labels, trace = paired_batch(
                backbone, inputs, inputs.plans[seed], reference, generator, 0, step, device
            )
            check_frozen(backbone, base, residual, frozen)
            head = EffectiveLinear(residual.effective_weight(base.weight).detach(), base.bias)
            analytic = verify_analytic_gradient(gap, labels, head, weights)
            total, stats, logits, teacher = objective(
                residual, base, basis, pairs, counts, weights, gap, labels, 0
            )
            if step == 0:
                check_step_zero(residual, base, logits, teacher)
                require(stats["sub"] == stats["div"] == 0.0, "Initial residual regularizers must be zero")
            check_scale(stats, step)
            norms = backward(total, residual, base, backbone, optimizer)
            if step == 0:
                require(norms["grad_norm_a"] == 0.0 and norms["grad_norm_b"] > 0.0,
                        f"Unexpected zero-residual A/B gradients for seed {seed}: {norms}")
            else:
                require(norms["grad_norm_a"] > 0.0 and norms["grad_norm_b"] > 0.0,
                        f"A/B gradients did not activate on the second step for seed {seed}: {norms}")
            old_a = residual.A.detach().clone()
            old_b = residual.B.detach().clone()
            optimizer.step()
            a_changed = not bitwise_equal(residual.A, old_a)
            b_changed = not bitwise_equal(residual.B, old_b)
            require(b_changed and (a_changed or step == 0), f"Missing A/B parameter update for seed {seed}, step {step}")
            check_frozen(backbone, base, residual, frozen)
            reports.append({"trace": trace, "analytic_gradient_check": analytic,
                            "a_updated": a_changed, "b_updated": b_changed, **stats, **norms})
        checks.append({
            "seed": seed, "base_checkpoint_sha256": sha, "clean_initial_classifier_sha256": init,
            "step0_bitwise_equal_to_C2": True, "paired_with_C1": True, "frozen_and_many_rows_verified": True,
            "disposable_steps": 2, "batches": reports, "empty_graph": graph["empty_graph"],
        })
        print(f"TPR SMOKE seed={seed} PASS: paired hashes, grad_B0={reports[0]['grad_norm_b']:.7g}, "
              f"grad_A1={reports[1]['grad_norm_a']:.7g}", flush=True)
    require(protected_snapshot() == snapshot["protected"], "Protected artifacts changed during Smoke")
    exclusive_json(out, {
        "passed": True, "completed_at": c.now(), "seeds": list(c.SEEDS), "checks": checks,
        "formal_training_started": False, "preflight_sha256": c.sha256_file(OUT / "preflight_report.json"),
        "preparation_sha256": c.sha256_file(PREPARATION), "protocol_sha256": snapshot["protocol_sha256"],
        "environment": {**environment(), "device": torch.cuda.get_device_name(device)},
        "test_guard": guard_record(), "official_TEST_used": False,
    })
    print("TPR SMOKE PASS: all three seeds, two real batches and disposable updates each.", flush=True)


def check_smoke_marker(snapshot):
    marker = c.read_json(OUT / "smoke_report.json")
    require(marker["passed"] is True and marker["official_TEST_used"] is False
            and marker["formal_training_started"] is False and marker["seeds"] == list(c.SEEDS)
            and marker["preflight_sha256"] == c.sha256_file(OUT / "preflight_report.json")
            and marker["preparation_sha256"] == c.sha256_file(PREPARATION)
            and marker["protocol_sha256"] == snapshot["protocol_sha256"]
            and [item["seed"] for item in marker["checks"]] == list(c.SEEDS),
            "Missing, stale, or incompatible three-seed Smoke gate")
    for check in marker["checks"]:
        seed = check["seed"]
        require(check["step0_bitwise_equal_to_C2"] and check["paired_with_C1"]
                and check["frozen_and_many_rows_verified"] and check["disposable_steps"] == 2
                and check["base_checkpoint_sha256"] == snapshot["teachers"][str(seed)]["sha256"]
                and len(check["batches"]) == 2, f"Smoke invariant failed for seed {seed}")
    return marker


METRICS = ("Overall", "Macro", "Many", "MediumN", "Few", "Head", "OfficialMedium", "Tail")
EPOCH_FIELDS = (
    "epoch", "steps", "samples", "alpha", "learning_rate", "train_clean_accuracy",
    "many_count", "train_many_top1_agreement", *BATCH_SCALARS,
    *(f"validation_{key}" for key in METRICS),
    "validation_many_kl", "validation_many_top1_agreement", "teacher_many_accuracy",
)
BATCH_FIELDS = (
    "epoch", "batch", "optimizer_step", "batch_size", "alpha", "many_count", "many_agreement_count",
    *BATCH_SCALARS,
)


def train_seed(seed, inputs, snapshot, preparation, device):
    out = run_dir(seed)
    out.mkdir(parents=True, exist_ok=False)
    state = {"state": "running", "arm": ARM, "seed": seed, "started_at": c.now(),
             "epochs": [], "official_TEST_used": False}
    status_path = out / "status.json"
    c.write_json(status_path, state)
    steps, samples_seen = 0, 0
    location, trace = {}, None
    try:
        backbone, base, residual, basis, counts, optimizer, sha, init = build_run(
            seed, inputs, snapshot, preparation, device
        )
        frozen = frozen_hashes(backbone, base)
        pairs = preparation["runs"][str(seed)]["pairs"]
        weights = c.class_weights_from_counts(inputs.counts, device)
        reference_path = historical_dir(seed, C1) / "batch_trace.csv"
        reference = csv_rows(reference_path)
        generator = torch.Generator(device="cpu").manual_seed(seed + 300009)
        initial_a = residual.A.detach().clone()
        state.update(
            base_checkpoint_sha256=sha, base_state_sha256=frozen["base"], backbone_sha256=frozen["backbone"],
            clean_initial_classifier_sha256=init, clean_stage1_sha256=c.EXPECTED_CLEAN_SHA256,
            optimizer={"type": "SGD", **OPTIMIZER}, temperature=TEMPERATURE, loss_weights=WEIGHTS,
            rank=RANK, trainable_parameters=["A", "B"], protocol_sha256=snapshot["protocol_sha256"],
            preparation_sha256=c.sha256_file(PREPARATION),
            initial_A_sha256=c.hash_tensor(initial_a),
        )
        with (out / "batch_trace.csv").open("x", encoding="utf-8", newline="") as trace_file, \
             (out / "batch_diagnostics.csv").open("x", encoding="utf-8", newline="") as batch_file, \
             (out / "metrics.csv").open("x", encoding="utf-8", newline="") as metrics_file:
            trace_writer = csv.DictWriter(trace_file, fieldnames=TRACE_FIELDS)
            batch_writer = csv.DictWriter(batch_file, fieldnames=BATCH_FIELDS, extrasaction="ignore")
            metrics_writer = csv.DictWriter(metrics_file, fieldnames=EPOCH_FIELDS)
            for writer in (trace_writer, batch_writer, metrics_writer):
                writer.writeheader()
            for epoch in range(c.STAGE2_EPOCHS):
                records = []
                for batch in range(STEPS):
                    location, trace = {"epoch": epoch + 1, "batch": batch}, None
                    gap, labels, trace = paired_batch(
                        backbone, inputs, inputs.plans[seed], reference, generator, epoch, batch, device
                    )
                    total, stats, logits, teacher = objective(
                        residual, base, basis, pairs, counts, weights, gap, labels, epoch
                    )
                    if steps == 0:
                        check_step_zero(residual, base, logits, teacher)
                        state["step0_bitwise_equal_to_C2"] = True
                    check_scale(stats, steps)
                    norms = backward(total, residual, base, backbone, optimizer)
                    if steps == 0:
                        require(norms["grad_norm_a"] == 0.0 and norms["grad_norm_b"] > 0.0,
                                "Formal step-0 gradients disagree with zero-initialized factorization")
                    optimizer.step()
                    steps += 1
                    samples_seen += len(labels)
                    trace_writer.writerow(trace)
                    row = {"epoch": epoch + 1, "batch": batch, "optimizer_step": steps,
                           "batch_size": len(labels), **stats, **norms}
                    batch_writer.writerow(row)
                    records.append(row)
                require(sum(item["batch_size"] for item in records) == c.TRAIN_SAMPLES,
                        f"Unexpected TRAIN sample count at seed={seed}, epoch={epoch + 1}")
                check_frozen(backbone, base, residual, frozen)
                validation, teacher_metrics = evaluate(residual, base, inputs, device)
                many_count = sum(item["many_count"] for item in records)
                require(many_count > 0, "No Many samples seen during formal epoch")
                metrics = {
                    "epoch": epoch + 1, "steps": len(records), "samples": c.TRAIN_SAMPLES,
                    "alpha": records[-1]["alpha"], "learning_rate": OPTIMIZER["lr"],
                    "train_clean_accuracy": 100.0 * sum(item["correct"] for item in records) / c.TRAIN_SAMPLES,
                    "many_count": many_count,
                    "train_many_top1_agreement": sum(item["many_agreement_count"] for item in records) / many_count,
                    **{key: float(np.mean([item[key] for item in records])) for key in BATCH_SCALARS},
                    **{f"validation_{key}": validation[key] for key in METRICS},
                    "validation_many_kl": teacher_metrics["many_kl"],
                    "validation_many_top1_agreement": teacher_metrics["many_top1_agreement"],
                    "teacher_many_accuracy": teacher_metrics["teacher_many_accuracy"],
                }
                metrics_writer.writerow(metrics)
                for stream in (trace_file, batch_file, metrics_file):
                    stream.flush()
                state["epochs"].append({"epoch": epoch + 1, "metrics": metrics, "validation": validation,
                                        "teacher": teacher_metrics, "frozen_bitwise": True})
                state.update(optimizer_steps=steps, samples_seen=samples_seen)
                c.write_json(status_path, state)
                print(f"seed={seed} TPR epoch={epoch + 1}/{c.STAGE2_EPOCHS} "
                      f"loss={metrics['total']:.5f} overall={validation['Overall']:.3f} "
                      f"many={validation['Many']:.3f} few={validation['Few']:.3f}", flush=True)
        require(steps == c.STAGE2_EPOCHS * STEPS and samples_seen == c.STAGE2_EPOCHS * c.TRAIN_SAMPLES,
                "Run totals differ from sealed plan")
        require(not bitwise_equal(residual.A, initial_a) and int(torch.count_nonzero(residual.B)) > 0,
                "Trained factors did not update")
        trace_sha = c.sha256_file(out / "batch_trace.csv")
        require(trace_sha == c.sha256_file(reference_path), "Complete C1 trace differs byte-for-byte")
        final = state["epochs"][-1]
        diagnostics = residual_diagnostics(residual, base, basis, pairs)
        final_file = out / "final_classifier.pth"
        with final_file.open("xb") as stream:
            torch.save({
                "arm": ARM, "seed": seed, "epoch": c.STAGE2_EPOCHS,
                "A": residual.A.detach().cpu(), "B": residual.B.detach().cpu(),
                "tail_mask": residual.tail_mask.detach().cpu(),
                "base_checkpoint_sha256": sha, "base_state_sha256": frozen["base"],
                "clean_stage1_sha256": c.EXPECTED_CLEAN_SHA256, "optimizer": state["optimizer"],
                "protocol_sha256": snapshot["protocol_sha256"], "source_sha256": snapshot["sources"],
                "preparation_sha256": state["preparation_sha256"], "official_TEST_used": False,
            }, stream)
        state.update(
            state="completed", completed_at=c.now(), final_epoch=c.STAGE2_EPOCHS,
            final_validation=final["validation"], final_teacher=final["teacher"],
            residual_diagnostics=diagnostics, batch_trace_sha256=trace_sha,
            final_classifier_sha256=c.sha256_file(final_file), frozen_bitwise_all_epochs=True,
        )
        c.write_json(status_path, state)
        return state
    except Exception as error:
        state.update(state="failed", failed_at=c.now(), optimizer_steps=steps, samples_seen=samples_seen,
                     failure={"type": type(error).__name__, "message": str(error), **location,
                              "paired_input": trace})
        c.write_json(status_path, state)
        raise


def formal(inputs, snapshot, preparation, device):
    check_smoke_marker(snapshot)
    marker = OUT / "formal_completed.json"
    guard_path = OUT / "formal_test_guard.json"
    require(not marker.exists() and not guard_path.exists() and not (OUT / "results").exists(),
            "Existing formal files or partial results; no resume or overwrite")
    runs = []
    for seed in c.SEEDS:
        status = train_seed(seed, inputs, snapshot, preparation, device)
        require(protected_snapshot() == snapshot["protected"], "Protected C0-C4 results changed")
        runs.append({key: status[key] for key in ("seed", "final_epoch", "optimizer_steps", "samples_seen",
                                                "base_checkpoint_sha256", "batch_trace_sha256",
                                                "final_classifier_sha256", "frozen_bitwise_all_epochs")})
    guard = guard_record()
    exclusive_json(guard_path, guard)
    exclusive_json(marker, {
        "passed": True, "completed_at": c.now(), "arm": ARM, "seeds": list(c.SEEDS), "runs": runs,
        "epochs_completed": sum(run["final_epoch"] for run in runs),
        "optimizer_steps": sum(run["optimizer_steps"] for run in runs),
        "train_samples_seen": sum(run["samples_seen"] for run in runs),
        "protected_C0_C4_unchanged": True, "official_TEST_used": False, "test_guard": guard,
        "preflight_sha256": c.sha256_file(OUT / "preflight_report.json"),
        "smoke_sha256": c.sha256_file(OUT / "smoke_report.json"),
        "preparation_sha256": c.sha256_file(PREPARATION), "protocol_sha256": snapshot["protocol_sha256"],
    })
    print("TPR FORMAL PASS: 3 seeds, 90 epochs, 15300 updates; C0-C4 sealed.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "smoke", "formal", "analyze"), required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(c.SEEDS))
    parser.add_argument("--prepare-missing", action="store_true",
                        help="Allow Smoke to create its own missing TRAIN confusion/SVD preparation once")
    args = parser.parse_args()
    if tuple(args.seeds) != c.SEEDS:
        parser.error("Frozen protocol requires exactly --seeds 123 124 125 in order")
    if args.prepare_missing and args.phase != "smoke":
        parser.error("--prepare-missing is only valid in Smoke")
    failure_path = OUT / f"failure_{args.phase}.json"
    require(not failure_path.exists(), f"Preserved failed phase {args.phase}; no automatic restart")
    try:
        inputs, snapshot = collect_inputs()
        if args.phase == "preflight":
            preflight(inputs, snapshot)
            return
        preflight_marker(snapshot)
        if args.phase == "analyze":
            from analyze_tpr import analyze
            analyze(inputs, snapshot, load_preparation(inputs, snapshot))
            return
        require(torch.cuda.is_available(), "CUDA is required under the paired protocol")
        device = torch.device("cuda:0")
        if args.phase == "smoke":
            require(not (OUT / "smoke_report.json").exists() and not (OUT / "results").exists(),
                    "Existing Smoke or formal output")
            if not PREPARATION.exists():
                require(args.prepare_missing, "Missing TRAIN confusion/SVD preparation; Smoke requires --prepare-missing")
                make_preparation(inputs, snapshot, device)
        preparation = load_preparation(inputs, snapshot)
        if args.phase == "smoke":
            smoke(inputs, snapshot, preparation, device)
        else:
            formal(inputs, snapshot, preparation, device)
    except Exception as error:
        exclusive_json(failure_path, {"phase": args.phase, "state": "failed", "failed_at": c.now(),
                                      "type": type(error).__name__, "message": str(error)})
        raise


if __name__ == "__main__":
    main()
