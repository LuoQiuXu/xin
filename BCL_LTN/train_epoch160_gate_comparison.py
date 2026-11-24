"""Resume the three locked BCL-LTN arms from the same epoch-160 state."""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from BCL_AUDIT.cifar100lt.data import CIFAR100BalancedTest, CIFAR100LT, cifar100lt_class_counts
from BCL_AUDIT.cifar100lt.losses import BalancedContrastiveLoss, LogitCompensatedCE
from BCL_AUDIT.cifar100lt.models import BCLCIFARModel
from BCL_AUDIT.cifar100lt.train import (
    BASE_LR, BATCH_SIZE, MOMENTUM, TEMPERATURE, WEIGHT_DECAY, classification_logits_v1,
    compose_group_loss, epoch_seed_for, file_sha256, first_batch_views_sha256,
    learning_rate_for_epoch, seed_worker, set_seed, state_dict_sha256,
)
from BCL_LTN.make_noisy_lt import group_class_ids
from BCL_LTN.train_noisy_bcl import (
    FIXED_INDICES_SHA256, FIXED_TRAIN_SOURCE_SHA256, NOISE_ARRAYS, NoisyBCLDataset,
    OfficialTestAudit, TrainDiagnosticDataset, bcl_sgd_step, json_dump, load_verified_noise,
    safe_torch_load, write_train_diagnostics_csv,
)

ARMS = ("BCL_ORIGINAL", "BCL_GLOBAL_RAW_MARGIN", "BCL_FEW_CLASS_ADAPTIVE_Q90")
GROUPS = ("Many", "Medium", "Few")
ALL_GROUPS = ("Overall", *GROUPS)
START, END, SEED = 160, 200, 42
MILESTONES = (180, 200)
SAMPLES, STEPS = 10_847, 43
MANIFEST_SHA = "0903d755db1e490cdf7a507f38a2c05ccdb32c7ecbd01d029487ea95205bd5e8"
NOISY_SHA = "691b8c8bf5758811d54c33c213e3bf2db12f586aa3d857f37da309a55d03554f"
EPOCH160_FILE_SHA = "10f0cb41db390f04a9d401433cc121396ab67fad7eb11dee0095c40350d75428"
EPOCH160_MODEL_SHA = "ef1a96e6d018a0dc2f6888b8816643935ab100856b6ad41655371a4ec5498102"
RELATION_FIELDS = (
    "self_view_positive_links", "cross_sample_positive_links", "prototype_positive_links",
    "masked_sample_candidates", "masked_target_prototype_candidates", "active_denominator_candidates",
)
ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class GateReference:
    groups: torch.Tensor
    means: torch.Tensor
    stds: torch.Tensor
    few_q90: float

    def to(self, device):
        return GateReference(self.groups.to(device), self.means.to(device), self.stds.to(device), self.few_q90)


@torch.no_grad()
def observed_margin(logits, targets):
    selected = logits.gather(1, targets[:, None]).squeeze(1)
    other = logits.clone()
    other.scatter_(1, targets[:, None], -torch.inf)
    return selected - other.amax(1)


@torch.no_grad()
def reliable_samples(arm, logits, observed, reference):
    if arm == ARMS[0]:
        return torch.ones_like(observed, dtype=torch.bool)
    margin = observed_margin(logits, observed)
    if arm == ARMS[1]:
        return margin > 0
    if arm != ARMS[2]:
        raise ValueError(f"Unknown arm {arm}")
    normalized = (margin - reference.means[observed]) / (reference.stds[observed] + 1e-8)
    return (reference.groups[observed] != GROUPS.index("Few")) | (normalized > reference.few_q90)


@dataclass(frozen=True)
class MaskedRelations:
    allowed: torch.Tensor
    positive: torch.Tensor
    class_counts: torch.Tensor
    weights: torch.Tensor
    partner: torch.Tensor


def masked_relations(targets, reliable, num_classes):
    """Build directed relations in the original [v2, v3, prototypes] order."""
    batch = targets.numel()
    anchors = 2 * batch
    i = torch.arange(anchors, device=targets.device)
    partner = (i + batch) % anchors
    y = targets.repeat(2)
    trusted = reliable.repeat(2)
    sample_allowed = trusted[None, :].expand(anchors, anchors).clone()
    sample_allowed[i, i] = False
    sample_allowed[i, partner] = True
    sample_positive = sample_allowed & trusted[:, None] & trusted[None, :] & (y[:, None] == y[None, :])
    sample_positive[i, partner] = True

    proto_ids = torch.arange(num_classes, device=targets.device)
    own_proto = y[:, None] == proto_ids[None, :]
    prototype_positive = own_proto & trusted[:, None]
    allowed = torch.cat((sample_allowed, ~own_proto | prototype_positive), dim=1)
    positive = torch.cat((sample_positive, prototype_positive), dim=1)

    candidates = torch.cat((y, proto_ids))[None, :].expand_as(allowed)
    class_counts = torch.zeros((anchors, num_classes), device=targets.device, dtype=torch.int64)
    class_counts.scatter_add_(1, candidates, allowed.to(torch.int64))
    # Count the excluded anchor itself, exactly as BCL does before subtracting positives.
    class_counts.scatter_add_(1, y[:, None], torch.ones_like(y[:, None]))
    weights = class_counts.gather(1, candidates) - positive.to(torch.int64)
    weights = torch.where(allowed, weights, 1)
    return MaskedRelations(allowed, positive, class_counts, weights, partner)


def masked_bcl_loss(class_prototypes, two_views, targets, reliable, temperature=TEMPERATURE):
    relations = masked_relations(targets, reliable, class_prototypes.shape[0])
    sample_features = torch.cat((two_views[:, 0], two_views[:, 1]), dim=0)
    features = torch.cat((sample_features, class_prototypes), dim=0)
    logits = torch.matmul(sample_features, features.T) / temperature
    i = torch.arange(sample_features.shape[0], device=targets.device)
    shift_candidates = relations.allowed.clone()
    # Include self in the row-max shift so the all-reliable case uses the original BCL shift.
    shift_candidates[i, i] = True
    shifted = logits - logits.masked_fill(~shift_candidates, -torch.inf).amax(1, keepdim=True).detach()
    if bool((relations.weights <= 0).any()) or not bool(relations.positive[i, relations.partner].all()):
        raise RuntimeError("Invalid masked BCL denominator or missing own-view positive")
    exp_values = shifted.masked_fill(~relations.allowed, -torch.inf).exp()
    denominator = (exp_values / relations.weights.to(exp_values.dtype)).sum(1, keepdim=True)
    log_prob = shifted - denominator.clamp_min(torch.finfo(shifted.dtype).tiny).log()
    loss = -(torch.where(relations.positive, log_prob, 0).sum(1) / relations.positive.sum(1)).view(2, -1).mean()
    return loss, relations


def relationship_totals(relations, labels, num_classes):
    anchors = 2 * labels.numel()
    if relations is None:
        label_counts = torch.bincount(labels, minlength=num_classes)
        cross = int(4 * (label_counts * (label_counts - 1)).sum())
        return dict(zip(RELATION_FIELDS, (anchors, cross, anchors, 0, 0,
                                         anchors * (anchors + num_classes - 1)), strict=True))
    parts = (
        relations.positive[:, :anchors].sum() - anchors,
        relations.positive[:, anchors:].sum(),
        anchors * (anchors - 1) - relations.allowed[:, :anchors].sum(),
        anchors * num_classes - relations.allowed[:, anchors:].sum(),
        relations.allowed.sum(),
    )
    return dict(zip(RELATION_FIELDS, (anchors, *torch.stack(parts).tolist()), strict=True))


def optimizer_fingerprint(state):
    metadata = {"param_groups": state["param_groups"], "parameter_states": []}
    tensors = {}
    for parameter_id, values in sorted(state["state"].items()):
        metadata["parameter_states"].append((parameter_id, sorted(values)))
        for name, tensor in values.items():
            if not isinstance(tensor, torch.Tensor):
                raise RuntimeError(f"Unexpected non-tensor SGD state for {parameter_id}/{name}")
            tensors[f"{parameter_id}/{name}"] = tensor
    metadata["tensors_sha256"] = state_dict_sha256(tensors)
    return hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()


def restore_arm(checkpoint, device):
    model = BCLCIFARModel()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=BASE_LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    optimizer.load_state_dict(deepcopy(checkpoint["optimizer_state"]))
    if len(optimizer.param_groups) != 1:
        raise RuntimeError("Expected one SGD parameter group in the epoch-160 checkpoint")
    group = optimizer.param_groups[0]
    expected = {"lr": BASE_LR, "momentum": MOMENTUM, "weight_decay": WEIGHT_DECAY,
                "dampening": 0, "nesterov": False, "maximize": False, "foreach": None,
                "fused": None, "differentiable": False}
    if any(group.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Restored SGD options differ from the original optimizer")
    if len(optimizer.state) != len(group["params"]):
        raise RuntimeError("Checkpoint SGD momentum is missing for some model parameters")
    for parameter in group["params"]:
        state = optimizer.state[parameter]
        buffer = state.get("momentum_buffer")
        if (set(state) != {"momentum_buffer"} or not isinstance(buffer, torch.Tensor)
                or buffer.shape != parameter.shape or buffer.dtype != parameter.dtype
                or buffer.device != parameter.device):
            raise RuntimeError("Checkpoint contains invalid SGD momentum for a model parameter")
    actual = {"model_state_sha256": state_dict_sha256(model.state_dict()),
              "optimizer_state_sha256": optimizer_fingerprint(optimizer.state_dict()),
              "momentum_tensors": len(optimizer.state)}
    expected_model = checkpoint["model_state_sha256"]
    if (actual["model_state_sha256"] != expected_model
            or actual["optimizer_state_sha256"] != optimizer_fingerprint(checkpoint["optimizer_state"])):
        raise RuntimeError("Restored model or optimizer differs from the saved state")
    generator = torch.Generator()
    generator.set_state(checkpoint["data_loader_generator_state"])
    rng = checkpoint["rng_state"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if device.type == "cuda":
        if len(rng["cuda"]) != torch.cuda.device_count():
            raise RuntimeError("Checkpoint CUDA RNG state does not match the visible devices")
        torch.cuda.set_rng_state_all(rng["cuda"])
    actual["loader_rng_sha256"] = hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest()
    return model, optimizer, generator, actual


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def verify_file(path, expected):
    actual = file_sha256(path)
    if actual != expected:
        raise RuntimeError(f"SHA256 mismatch for {path}: {actual} != {expected}")
    return actual


@dataclass(frozen=True)
class RunInputs:
    base: CIFAR100LT
    dataset: NoisyBCLDataset
    observed: np.ndarray
    observed_counts: np.ndarray
    reference: GateReference
    checkpoint: dict
    original_final_state: str
    original_history: dict
    hashes: dict
    reference_count: int


def locked_inputs(args):
    paths = {
        "indices": (args.indices_file, FIXED_INDICES_SHA256),
        "manifest": (args.noise_dir / "noise_manifest.json", MANIFEST_SHA),
        "epoch160": (args.baseline_dir / "checkpoints/epoch_160_pre_decay.pth", EPOCH160_FILE_SHA),
        "noisy_labels": (args.noise_dir / NOISE_ARRAYS["noisy_labels"], NOISY_SHA),
        **{name: (args.data_root / "cifar-100-python" / name, digest)
           for name, digest in FIXED_TRAIN_SOURCE_SHA256.items()},
    }
    hashes = {key: verify_file(path, digest) for key, (path, digest) in paths.items()}
    manifest = read_json(paths["manifest"][0])
    for key in ("clean_labels", "is_noisy"):
        if manifest[f"{key}_file"] != NOISE_ARRAYS[key]:
            raise RuntimeError(f"Unexpected truth file name in manifest for {key}")
        hashes[key] = verify_file(args.noise_dir / NOISE_ARRAYS[key], manifest[f"{key}_sha256"])
    # The hashed truth files are not decoded here. The training dataset receives only observed labels.
    observed = np.load(paths["noisy_labels"][0], allow_pickle=False)
    if observed.shape != (SAMPLES,) or observed.dtype != np.int64 or np.any((observed < 0) | (observed >= 100)):
        raise RuntimeError("The observed labels are not a valid fixed 10,847-vector")
    observed_counts = np.bincount(observed, minlength=100)
    clean_counts = cifar100lt_class_counts()
    if (manifest["indices_sha256"] != FIXED_INDICES_SHA256 or manifest["noisy_labels_sha256"] != NOISY_SHA
            or manifest["observed_noisy_class_counts"] != observed_counts.tolist()
            or manifest["clean_class_counts"] != clean_counts.tolist() or np.any(observed_counts <= 0)):
        raise RuntimeError("The manifest does not match the observed labels and frozen IF=100 groups")

    hashes["gate_reference"] = verify_file(args.gate_reference, args.gate_reference_sha256)
    ref = read_json(args.gate_reference)
    stats = ref["class_stats"]
    if (ref["noise_manifest_sha256"] != MANIFEST_SHA or ref["n_observed_few"] != 1536
            or ref["normalization_epsilon"] != 1e-8 or not math.isfinite(ref["few_q90_threshold"])
            or [row["class_id"] for row in stats] != list(range(100))
            or [row["observed_count"] for row in stats] != observed_counts.tolist()):
        raise RuntimeError("The score-only gate reference does not match the fixed observed-label data")
    audit = read_json(args.audit_receipt)
    score_hash = next(row["sha256"] for row in audit["outputs"] if row["path"] == "sample_scores.csv")
    if ref["sample_scores_sha256"] != score_hash:
        raise RuntimeError("Gate reference does not come from the frozen epoch-160 scores")
    means = torch.tensor([row["raw_margin_mean"] for row in stats], dtype=torch.float32)
    stds = torch.tensor([row["raw_margin_std"] for row in stats], dtype=torch.float32)
    if not bool(torch.isfinite(means).all() & torch.isfinite(stds).all() & (stds >= 0).all()):
        raise RuntimeError("Nonfinite or invalid gate-reference class statistics")
    group_lookup = torch.empty(100, dtype=torch.long)
    for group, ids in group_class_ids(clean_counts).items():
        group_lookup[ids] = GROUPS.index(group)
    reference = GateReference(group_lookup, means, stds, float(ref["few_q90_threshold"]))

    summary_path = args.baseline_dir / "training_summary.json"
    summary = read_json(summary_path)
    hashes["baseline_history"] = file_sha256(args.baseline_dir / "history.jsonl")
    history = [json.loads(line) for line in (args.baseline_dir / "history.jsonl").read_text(encoding="utf-8").splitlines()]
    if [row["epoch"] for row in history] != list(range(1, END + 1)):
        raise RuntimeError("Archived original training history is incomplete")
    checkpoint = safe_torch_load(paths["epoch160"][0], torch.device("cpu"))
    requirements = {"model_state", "optimizer_state", "rng_state", "data_loader_generator_state",
                    "class_counts_for_lc_and_bcl", "protocol"}
    diagnostic = read_json(args.baseline_dir / "train_diagnostics/epoch_160.json")
    protocol = checkpoint.get("protocol", {})
    if (not requirements.issubset(checkpoint) or checkpoint.get("epoch") != START or checkpoint.get("seed") != SEED
            or checkpoint.get("optimizer_updates") != START * STEPS
            or checkpoint.get("class_counts_for_lc_and_bcl") != observed_counts.tolist()
            or checkpoint.get("indices_sha256") != FIXED_INDICES_SHA256 or checkpoint.get("noise_manifest_sha256") != MANIFEST_SHA
            or checkpoint.get("model_state_sha256") != EPOCH160_MODEL_SHA
            or diagnostic["model_state_sha256"] != EPOCH160_MODEL_SHA
            or summary["checkpoint_sha256"][paths["epoch160"][0].name] != EPOCH160_FILE_SHA
            or protocol.get("batch_size") != BATCH_SIZE or protocol.get("precision") != "fp32"
            or protocol.get("amp") is not False or protocol.get("tf32") is not False):
        raise RuntimeError("Epoch-160 resume, class counts, objective protocol or provenance mismatch")
    if state_dict_sha256(checkpoint["model_state"]) != EPOCH160_MODEL_SHA:
        raise RuntimeError("The epoch-160 checkpoint model tensors differ from the saved state hash")
    final_path = args.baseline_dir / "checkpoints/epoch_200.pth"
    hashes["archived_final"] = verify_file(final_path, summary["checkpoint_sha256"][final_path.name])
    final = safe_torch_load(final_path, torch.device("cpu"))
    final_state = state_dict_sha256(final["model_state"])
    if (final["epoch"] != END or final["seed"] != SEED or final["model_state_sha256"] != final_state
            or read_json(args.baseline_dir / "train_diagnostics/epoch_200.json")["model_state_sha256"] != final_state):
        raise RuntimeError("Archived epoch-200 model does not agree with its checkpoint receipt")

    base = CIFAR100LT(root=args.data_root, mode="bcl", indices_file=args.indices_file, split_seed=0, download=False)
    targets = base.targets.copy()
    targets[base.indices] = observed
    base.targets = targets
    return RunInputs(base, NoisyBCLDataset(base), observed, observed_counts, reference, checkpoint,
                     final_state, {row["epoch"]: row for row in history}, hashes,
                     int(ref["expected_few_acceptance_at_reference"]))


def seed_epoch(epoch, optimizer, generator):
    epoch_seed = epoch_seed_for(SEED, epoch - 1)
    set_seed(epoch_seed)
    generator.manual_seed(epoch_seed)
    lr = learning_rate_for_epoch(epoch - 1)
    for group in optimizer.param_groups:
        group["lr"] = lr
    return epoch_seed, lr


def train_loader(inputs, generator):
    return DataLoader(inputs.dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                      pin_memory=True, drop_last=False, generator=generator, worker_init_fn=seed_worker)


def observed_acceptance(observed, reliable, reference):
    ids = reference.groups.numpy()[observed]
    total = np.bincount(ids, minlength=3)
    accepted = np.bincount(ids[reliable], minlength=3)
    return {f"{group}_{key}": int(value)
            for i, group in enumerate(GROUPS)
            for key, value in (("reliable", accepted[i]), ("unreliable", total[i] - accepted[i]))}


def train_step(arm, model, optimizer, lc, original_bcl, batch, cpu_ref, gpu_ref, device, *, smoke=False):
    views, observed, _ = batch
    labels_np = observed.numpy()
    if smoke:
        before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        if len(views) != 3 or any(view.dtype != torch.float32 or view.shape != (256, 3, 32, 32) for view in views):
            raise RuntimeError("Expected three FP32 views of a real full batch")
    if arm == ARMS[0]:
        metrics = bcl_sgd_step(model, optimizer, lc, original_bcl, views, observed, device, check_smoke=smoke)
        reliable_np = np.ones(len(labels_np), dtype=bool)
        relations = None
    else:
        labels = observed.to(device, non_blocking=True)
        size = len(labels)
        all_views = torch.cat([view.to(device, non_blocking=True) for view in views], dim=0)
        _, logits, embeddings, prototypes = model(all_views, compute_contrastive=True)
        logits_v1 = classification_logits_v1(logits, size)
        if any(value.dtype != torch.float32 for value in (logits_v1, embeddings, prototypes)):
            raise RuntimeError("The comparison requires FP32 outputs")
        reliable = reliable_samples(arm, logits_v1.detach(), labels, gpu_ref)
        loss_lc = lc(logits_v1, labels)
        two_views = torch.stack((embeddings[size:2 * size], embeddings[2 * size:]), dim=1)
        loss_bcl, relations = masked_bcl_loss(prototypes, two_views, labels, reliable)
        loss = compose_group_loss("BCL", loss_lc, loss_bcl)
        metrics = {"loss_lc": float(loss_lc.detach()), "loss_bcl": float(loss_bcl.detach()),
                   "loss_total": float(loss.detach())}
        if not all(math.isfinite(value) for value in metrics.values()):
            raise RuntimeError(f"{arm} produces nonfinite losses: {metrics}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if smoke and any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
            raise RuntimeError(f"{arm} Smoke found a missing or nonfinite gradient")
        optimizer.step()
        reliable_np = reliable.cpu().numpy()
    if smoke:
        params = dict(model.named_parameters())
        if any(not bool(torch.isfinite(p).all()) for p in params.values()):
            raise RuntimeError(f"{arm} Smoke found nonfinite parameters after its update")
        changed = [name for name, p in params.items() if not torch.equal(p.detach(), before[name])]
        expected = {"backbone", "classifier", "sample_mlp", "prototype_mlp"}
        if {name.split(".", 1)[0] for name in changed} != expected:
            raise RuntimeError(f"{arm} Smoke did not update all four model components")
        metrics.update({"optimizer_updates": 1, "finite_gradients": len(params),
                        "changed_parameter_tensors": len(changed), "updated_components": sorted(expected)})
    counts = relationship_totals(relations, observed, len(cpu_ref.groups))
    return metrics, {**observed_acceptance(labels_np, reliable_np, cpu_ref), **counts}


def batch_fingerprint(batch):
    views, observed, indices = batch
    return {
        "first_batch_views_sha256": first_batch_views_sha256(views),
        "first_batch_indices_sha256": hashlib.sha256(indices.numpy().tobytes()).hexdigest(),
        "first_batch_observed_labels_sha256": hashlib.sha256(observed.numpy().tobytes()).hexdigest(),
    }


def make_criteria(inputs, device):
    lc = LogitCompensatedCE(inputs.observed_counts).to(device)
    bcl = BalancedContrastiveLoss(inputs.observed_counts, temperature=TEMPERATURE).to(device)
    return lc, bcl


def run_smoke(output, inputs, provenance, guard, device):
    records = {}
    gpu_ref = inputs.reference.to(device)
    for arm in ARMS:
        model, optimizer, generator, restored = restore_arm(inputs.checkpoint, device)
        seed, lr = seed_epoch(START + 1, optimizer, generator)
        model.train()
        batch = next(iter(train_loader(inputs, generator)))
        first = batch_fingerprint(batch)
        lc, original_bcl = make_criteria(inputs, device)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        metrics, relationships = train_step(arm, model, optimizer, lc, original_bcl, batch,
                                            inputs.reference, gpu_ref, device, smoke=True)
        torch.cuda.synchronize(device)
        records[arm] = {"restored": restored, "batch": first, "epoch_seed": seed, "lr": lr,
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                        **metrics, **relationships}
        print(f"[SMOKE] {arm} total={metrics['loss_total']:.7f}, updated={metrics['changed_parameter_tensors']}", flush=True)
        del model, optimizer, batch
    if (any(records[a]["restored"] != records[ARMS[0]]["restored"]
            or records[a]["batch"] != records[ARMS[0]]["batch"] for a in ARMS)
            or len({records[a]["loss_lc"] for a in ARMS}) != 1 or guard.opens or guard.denied):
        raise RuntimeError("Smoke restore, input pairing, LC parity or TEST isolation failed")
    if any(records[ARMS[2]][f"{group}_unreliable"] for group in ("Many", "Medium")):
        raise RuntimeError("Adaptive Smoke gated Many or Medium")
    report = {"passed": True, "provenance": provenance, "arms": records,
              "official_test_opens": guard.opens, "official_test_denied": guard.denied,
              "formal_training_started": False}
    json_dump(output / "smoke_report.json", report)
    return report


def evaluation_logits(model, base, device):
    model.eval()
    state = state_dict_sha256(model.state_dict())
    data = TrainDiagnosticDataset(base.data, base.indices, base.transform_eval)
    loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    logits = np.empty((SAMPLES, 100), dtype=np.float32)
    seen = np.zeros(SAMPLES, dtype=bool)
    with torch.inference_mode():
        for images, positions in loader:
            _, values, _, _ = model(images.to(device, non_blocking=True), compute_contrastive=False)
            pos = positions.numpy()
            if values.dtype != torch.float32 or not bool(torch.isfinite(values).all()) or seen[pos].any():
                raise RuntimeError("Invalid or duplicated TRAIN diagnostic logits")
            logits[pos] = values.cpu().numpy()
            seen[pos] = True
    if not seen.all() or state_dict_sha256(model.state_dict()) != state:
        raise RuntimeError("TRAIN diagnostic coverage or model-state check failed")
    return logits, state


def eval_signals(logits, group_ids):
    many = group_ids == 0
    predicted = logits.argmax(1)
    stable = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)
    logp = stable - np.log(np.exp(stable).sum(axis=1, keepdims=True))
    prob = np.exp(logp)
    return predicted, {
        "max_head_logit": logits[:, many].max(1),
        "many_probability_mass": prob[:, many].sum(1),
        "maximum_probability": prob.max(1),
        "entropy": -(prob * logp).sum(1),
    }


def summarize_signals(signals, clean_few, many_to_few):
    from BCL_LTN.analyze_epoch160_source_attribution import binary_ranking_metrics

    def fractions(v):
        q25, median, q75 = np.quantile(v, [0.25, 0.50, 0.75])
        return {"n": len(v), "mean": float(v.mean()), "median": float(median),
                "q25": float(q25), "q75": float(q75), "iqr": float(q75 - q25)}

    n_clean, n_noisy = int(clean_few.sum()), int(many_to_few.sum())
    if (n_clean, n_noisy) != (194, 1097):
        raise RuntimeError("Fixed Few source subsets differ from Part 1")
    y = np.concatenate((np.zeros(n_clean, dtype=np.int64), np.ones(n_noisy, dtype=np.int64)))
    rows = {}
    for name, signal in signals.items():
        good, bad = signal[clean_few], signal[many_to_few]
        values = np.concatenate((good, bad))
        result = {"Clean_Few": fractions(good), "Many_to_Few": fractions(bad),
                  "auc_many_to_few_positive": binary_ranking_metrics(y, values)["roc_auc"]}
        if name == "entropy":
            result["minus_entropy_auc_many_to_few_positive"] = binary_ranking_metrics(y, -values)["roc_auc"]
        rows[name] = result
    return rows


def acceptance_result(mask, reliable):
    size = int(mask.sum())
    yes = int((mask & reliable).sum())
    return {"total": size, "reliable": yes, "unreliable": size - yes,
            "acceptance_rate": yes / size, "rejection_rate": (size - yes) / size}


def train_diagnostics(args, arm, model, inputs, epoch, checkpoint, counts, arm_dir, device):
    logits, state = evaluation_logits(model, inputs.base, device)
    if state != checkpoint["model_state_sha256"]:
        raise RuntimeError("TRAIN diagnostic state differs from the saved checkpoint")
    ids = inputs.reference.groups.numpy()
    prediction, signals = eval_signals(logits, ids)
    logit_tensor = torch.from_numpy(logits)
    observed_tensor = torch.from_numpy(inputs.observed)
    margin = observed_margin(logit_tensor, observed_tensor).numpy()
    reliable = reliable_samples(arm, logit_tensor, observed_tensor, inputs.reference).numpy()
    # All predictions, summaries, and gate decisions precede the offline evaluation join.
    base, _, observed, clean, noisy = load_verified_noise(args.data_root, args.indices_file, args.noise_dir)
    if not np.array_equal(observed, inputs.observed) or not np.array_equal(base.indices, inputs.base.indices):
        raise RuntimeError("Offline truth alignment differs from the training data")
    del base
    source_ids = ids[clean]
    observed_ids = ids[observed]
    clean_acc, noisy_acc, gate = {}, {}, {}
    for i, group in enumerate(ALL_GROUPS, -1):
        source = np.ones(SAMPLES, dtype=bool) if i == -1 else source_ids == i
        destination = np.ones(SAMPLES, dtype=bool) if i == -1 else observed_ids == i
        clean_accept = acceptance_result(source & ~noisy, prediction == clean)
        noisy_accept = acceptance_result(source & noisy, prediction == observed)
        clean_acc[group] = {"correct": clean_accept["reliable"], "total": clean_accept["total"],
                            "accuracy_percent": 100.0 * clean_accept["reliable"] / clean_accept["total"]}
        noisy_acc[group] = {"predicted_as_corrupted_observed_label": noisy_accept["reliable"],
                            "total_noisy_examples": noisy_accept["total"],
                            "memorization_rate_percent": 100.0 * noisy_accept["reliable"] / noisy_accept["total"]}
        gate[group] = {"all": acceptance_result(destination, reliable),
                       "clean": acceptance_result(destination & ~noisy, reliable),
                       "noisy": acceptance_result(destination & noisy, reliable)}
    clean_few = (observed_ids == 2) & ~noisy
    many_to_few = (observed_ids == 2) & (source_ids == 0) & noisy
    signals_summary = summarize_signals(signals, clean_few, many_to_few)
    sample_fields = ("position", "original_train_index", "observed_label", "observed_group", "clean_label",
                     "source_group", "is_noisy", "predicted_class", "predicted_group", "raw_margin",
                     "gate_reliable", *signals)
    sample_path = arm_dir / f"sample_logit_diagnostics_epoch{epoch}.csv"
    with sample_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(sample_fields)
        for i in range(SAMPLES):
            writer.writerow((i, inputs.base.indices[i], observed[i], GROUPS[observed_ids[i]], clean[i],
                             GROUPS[source_ids[i]], int(noisy[i]), prediction[i], GROUPS[ids[prediction[i]]],
                             float(margin[i]), int(reliable[i]), *(float(values[i]) for values in signals.values())))
    result = {"epoch": epoch, "samples": SAMPLES, "checkpoint_sha256": checkpoint["sha256"],
              "model_state_sha256": state, "model_state_unchanged": True, "eval_passes": 1,
              "clean_subset_accuracy": clean_acc, "noisy_subset_wrong_label_memorization_rate": noisy_acc,
              "gate_acceptance_by_observed_group_after_truth_join": gate,
              "training_epoch_gate_and_relationship_counts": counts, "few_logit_signals": signals_summary,
              "sample_diagnostics_local_only": str(sample_path)}
    json_dump(arm_dir / "train_diagnostics" / f"epoch_{epoch}.json", result)
    return result


def save_checkpoint(arm_dir, arm, model, optimizer, generator, epoch, epoch_seed, inputs, provenance):
    path = arm_dir / "checkpoints" / f"epoch_{epoch}.pth"
    path.parent.mkdir(parents=True, exist_ok=True)
    model_sha = state_dict_sha256(model.state_dict())
    checkpoint = {
        "arm": arm, "epoch": epoch, "seed": SEED, "epoch_seed": epoch_seed,
        "model_state": model.state_dict(), "model_state_sha256": model_sha,
        "optimizer_state": optimizer.state_dict(),
        "data_loader_generator_state": generator.get_state(),
        "rng_state": {"python": random.getstate(), "numpy": np.random.get_state(),
                      "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()},
        "class_counts_for_lc_and_bcl": inputs.observed_counts.tolist(),
        "updates_since_epoch160": STEPS * (epoch - START),
        "indices_sha256": inputs.hashes["indices"], "noise_manifest_sha256": inputs.hashes["manifest"],
        "gate_reference_sha256": inputs.hashes["gate_reference"], "source_sha256": provenance["source_sha256"],
        "protocol": {"batch_size": BATCH_SIZE, "precision": "fp32", "amp": False, "tf32": False,
                     "start_epoch": START, "end_epoch": END, "current_lr": learning_rate_for_epoch(epoch - 1)},
    }
    with path.open("xb") as stream:
        torch.save(checkpoint, stream)
    return {"epoch": epoch, "path": str(path), "sha256": file_sha256(path), "model_state_sha256": model_sha}


def train_arm(args, arm, inputs, provenance, smoke_record, paired, device):
    arm_dir = args.output_dir / arm
    arm_dir.mkdir(exist_ok=False)
    model, optimizer, generator, restored = restore_arm(inputs.checkpoint, device)
    if restored != smoke_record["restored"] or restored["model_state_sha256"] != EPOCH160_MODEL_SHA:
        raise RuntimeError(f"{arm} formal restore differs from Smoke or the locked model state")
    json_dump(arm_dir / "start_state.json", restored)
    lc, original_bcl = make_criteria(inputs, device)
    loader = train_loader(inputs, generator)
    if len(loader) != STEPS:
        raise RuntimeError(f"{arm} did not get the expected 43 batches")
    gpu_ref = inputs.reference.to(device)
    expected_indices = np.sort(inputs.base.indices)
    checkpoints, diagnostics, history = [], [], []
    stat_fields = tuple(f"{group}_{kind}" for group in GROUPS for kind in ("reliable", "unreliable")) + RELATION_FIELDS
    started = time.perf_counter()
    with (arm_dir / "history.jsonl").open("x", encoding="utf-8") as history_file, \
            (arm_dir / "gate_diagnostics.csv").open("x", encoding="utf-8", newline="") as gate_file:
        writer = csv.DictWriter(gate_file, ("epoch", "batch", "samples", *stat_fields))
        writer.writeheader()
        for epoch in range(START + 1, END + 1):
            epoch_seed, lr = seed_epoch(epoch, optimizer, generator)
            model.train()
            losses = dict.fromkeys(("loss_lc", "loss_bcl", "loss_total"), 0.0)
            sums = dict.fromkeys(stat_fields, 0)
            order = hashlib.sha256()
            seen = []
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            epoch_start = time.perf_counter()
            for batch_index, batch in enumerate(loader, 1):
                if batch_index == 1:
                    first = batch_fingerprint(batch)
                    wanted = smoke_record["batch"] if epoch == START + 1 else None
                    if wanted is not None and first != wanted:
                        raise RuntimeError(f"{arm} epoch {epoch} first batch differs from Smoke")
                    if paired is not None and any(first[k] != paired[epoch][k] for k in first):
                        raise RuntimeError(f"{arm} epoch {epoch} first-batch views or labels differ from A")
                indices = batch[2].numpy().astype(np.int64, copy=False)
                order.update(indices.tobytes())
                seen.extend(indices.tolist())
                size = len(indices)
                step, counts = train_step(arm, model, optimizer, lc, original_bcl, batch,
                                          inputs.reference, gpu_ref, device)
                writer.writerow({"epoch": epoch, "batch": batch_index, "samples": size, **counts})
                for key, value in counts.items():
                    sums[key] += value
                for key in losses:
                    losses[key] += step[key] * size
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - epoch_start
            original = inputs.original_history[epoch]
            if (batch_index != STEPS or not np.array_equal(np.sort(seen), expected_indices)
                    or order.hexdigest() != original["sample_order_sha256"]
                    or epoch_seed != original["epoch_seed"] or lr != original["lr"]):
                raise RuntimeError(f"{arm} epoch {epoch} coverage, order, seed or LR differs from the original")
            if any(
                sums[f"{group}_reliable"] + sums[f"{group}_unreliable"]
                != int(inputs.observed_counts[inputs.reference.groups.numpy() == i].sum())
                for i, group in enumerate(GROUPS)
            ) or sums["self_view_positive_links"] != 2 * SAMPLES:
                raise RuntimeError(f"{arm} epoch {epoch} gates or own-view counts are inconsistent")
            if arm == ARMS[2] and (sums["Many_unreliable"] or sums["Medium_unreliable"]):
                raise RuntimeError(f"{arm} epoch {epoch} gated Many or Medium")
            row = {
                "epoch": epoch, "epoch_seed": epoch_seed, "lr": lr, "samples": len(seen), "optimizer_updates": batch_index,
                "sample_order_sha256": order.hexdigest(), **first,
                **{key: value / SAMPLES for key, value in losses.items()},
                "training_gate_and_relationship_counts": sums, "epoch_seconds": elapsed,
                "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
                "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
                "lc_uses_all_observed_samples": True, "truth_used_in_optimizer_loop": False,
            }
            if arm == ARMS[0]:
                row["loss_deltas_vs_original"] = {key: row[key] - original[key] for key in losses}
            if epoch in MILESTONES:
                checkpoint = save_checkpoint(arm_dir, arm, model, optimizer, generator, epoch, epoch_seed, inputs, provenance)
                checkpoints.append(checkpoint)
                diagnostics.append(train_diagnostics(args, arm, model, inputs, epoch, checkpoint, sums, arm_dir, device))
                row["checkpoint"] = checkpoint
            history.append(row)
            history_file.write(json.dumps(row, allow_nan=False) + "\n")
            history_file.flush()
            gate_file.flush()
            few_rate = sums["Few_reliable"] / (sums["Few_reliable"] + sums["Few_unreliable"])
            print(f"[{arm}] epoch {epoch}/{END} lr={lr:.5g} total={row['loss_total']:.6f} "
                  f"few_accept={few_rate:.2%} seconds={elapsed:.1f} "
                  f"allocated={row['peak_cuda_memory_bytes']/1024**3:.2f}GiB", flush=True)
    if [d["epoch"] for d in diagnostics] != list(MILESTONES):
        raise RuntimeError(f"{arm} is missing a required TRAIN checkpoint diagnostic")
    write_train_diagnostics_csv(arm_dir / "train_diagnostics.csv", diagnostics)
    result = {"arm": arm, "epochs_completed": len(history), "first_epoch": START + 1, "last_epoch": END,
              "optimizer_updates": sum(row["optimizer_updates"] for row in history), "restored": restored,
              "checkpoints": checkpoints, "train_diagnostics": diagnostics,
              "wall_seconds": time.perf_counter() - started,
              "peak_allocated_bytes": max(row["peak_cuda_memory_bytes"] for row in history),
              "peak_reserved_bytes": max(row["peak_cuda_reserved_bytes"] for row in history)}
    json_dump(arm_dir / "arm_completed.json", result)
    return result, {row["epoch"]: row for row in history}


def validate_final_checkpoints(results, inputs):
    if list(results) != list(ARMS):
        raise RuntimeError("All three arm completions are required before TEST")
    saved = {}
    for arm, result in results.items():
        epochs = tuple(row["epoch"] for row in result["train_diagnostics"])
        if result["epochs_completed"] != END - START or result["optimizer_updates"] != (END - START) * STEPS or epochs != MILESTONES:
            raise RuntimeError(f"{arm} is incomplete")
        for entry in result["checkpoints"]:
            verify_file(Path(entry["path"]), entry["sha256"])
        path = Path(result["checkpoints"][-1]["path"])
        checkpoint = safe_torch_load(path, torch.device("cpu"))
        if (checkpoint["epoch"] != END or checkpoint["arm"] != arm
                or state_dict_sha256(checkpoint["model_state"]) != result["checkpoints"][-1]["model_state_sha256"]):
            raise RuntimeError(f"{arm} final checkpoint failed its model-state validation")
        saved[arm] = checkpoint["model_state"]
    if results[ARMS[0]]["checkpoints"][-1]["model_state_sha256"] != inputs.original_final_state:
        raise RuntimeError("Original A continuation is not bitwise-identical")
    return saved


def evaluate_test(args, guard, results, saved, reference, device):
    results_by_arm = {}
    with guard.final_evaluation(), torch.inference_mode():
        test_set = CIFAR100BalancedTest(root=args.data_root, download=False)
        loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
        reference_labels = None
        for arm in ARMS:
            model = BCLCIFARModel()
            model.load_state_dict(saved[arm], strict=True)
            model = model.to(device).eval()
            state = state_dict_sha256(model.state_dict())
            expected = results[arm]["checkpoints"][-1]
            if state != expected["model_state_sha256"]:
                raise RuntimeError(f"{arm} TEST model does not match the approved epoch-200 checkpoint")
            labels, predicted = [], []
            for images, target in loader:
                _, logits, _, _ = model(images.to(device, non_blocking=True), compute_contrastive=False)
                if not bool(torch.isfinite(logits).all()):
                    raise RuntimeError(f"{arm} TEST logits are nonfinite")
                labels.append(target.numpy())
                predicted.append(logits.argmax(1).cpu().numpy())
            labels = np.concatenate(labels)
            predicted = np.concatenate(predicted)
            if (labels.shape != (10_000,) or not np.array_equal(np.bincount(labels, minlength=100), np.full(100, 100))
                    or state_dict_sha256(model.state_dict()) != state):
                raise RuntimeError(f"{arm} TEST coverage or frozen model state failed validation")
            if reference_labels is not None and not np.array_equal(reference_labels, labels):
                raise RuntimeError("TEST label orders differ across arms")
            reference_labels = labels
            correct = np.bincount(labels[predicted == labels], minlength=100)
            per_class = correct.astype(np.float64)
            values = {"Overall": 100.0 * float(np.mean(predicted == labels))}
            values.update({group: float(per_class[reference.groups.numpy() == i].mean()) for i, group in enumerate(GROUPS)})
            record = {"metrics_pct": values, "per_class_accuracy_pct": per_class.tolist(),
                      "samples": len(labels), "test_eval_passes": 1,
                      "model_state_sha256": state, "checkpoint_sha256": expected["sha256"]}
            json_dump(args.output_dir / arm / "final_test_metrics.json", record)
            results_by_arm[arm] = record
            del model
    return results_by_arm


def build_report(summary):
    r = [
        "# BCL-LTN epoch-160 continuation comparison", "",
        "The three FP32 arms restored the same epoch-160 model, SGD momentum, and RNG. "
        "They used batch 256, all 10,847 TRAIN samples per epoch, and identical sampler and first-batch augmentation hashes.",
        "", f"Arm A matches the archived epoch-200 model state SHA256 {summary['epoch200_original_state_sha256']}.",
        f"The fixed observed-Few q90 was {summary['few_q90_threshold']:.12f}.",
        "Clean labels and the corruption mask were decoded for evaluation after each frozen TRAIN diagnostic pass.",
        "", "## Balanced TEST after all TRAIN diagnostics", "",
        "| Arm | Overall | Many | Medium | Few | Overall delta vs A (pp) | Few delta vs A (pp) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    control = summary["test"][ARMS[0]]["metrics_pct"]
    for arm in ARMS:
        values = summary["test"][arm]["metrics_pct"]
        r.append(f"| {arm} | " + " | ".join(f"{values[g]:.2f}%" for g in ALL_GROUPS)
                 + f" | {values['Overall'] - control['Overall']:+.2f} | {values['Few'] - control['Few']:+.2f} |")
    r += ["", "## TRAIN checkpoints", "",
          "Accuracy and memorization use original clean source-group membership. Gate rows use observed-label groups.",
          "", "| Arm | Epoch | Clean accuracy Overall / Many / Medium / Few | Wrong-label memorization Overall / Many / Medium / Few | Eval Few acceptance (all / clean / noisy) |",
          "| --- | ---: | --- | --- | --- |"]
    for arm, result in summary["training"].items():
        for diag in result["train_diagnostics"]:
            clean = diag["clean_subset_accuracy"]
            noisy = diag["noisy_subset_wrong_label_memorization_rate"]
            gate = diag["gate_acceptance_by_observed_group_after_truth_join"]["Few"]
            clean_display = " / ".join(f"{clean[g]['accuracy_percent']:.2f}%" for g in ALL_GROUPS)
            noisy_display = " / ".join(f"{noisy[g]['memorization_rate_percent']:.2f}%" for g in ALL_GROUPS)
            gate_display = " / ".join(f"{gate[g]['acceptance_rate']:.2%}" for g in ("all", "clean", "noisy"))
            r.append(f"| {arm} | {diag['epoch']} | {clean_display} | {noisy_display} | {gate_display} |")
    r += ["", "## Observed-Few logit signals", "",
          "Many-to-Few is the positive class for AUC. All signals came from the same required TRAIN passes at epochs 180 and 200.",
          "The epoch-160 source CSV supplied no logits or entropy.",
          "", "| Arm | Epoch | Signal | Clean Few n / mean / median / IQR | Many-to-Few n / mean / median / IQR | AUC |",
          "| --- | ---: | --- | --- | --- | ---: |"]
    for arm, result in summary["training"].items():
        for diag in result["train_diagnostics"]:
            for signal, row in diag["few_logit_signals"].items():
                values = [" / ".join((str(stat["n"]), *(f"{stat[name]:.4f}" for name in ("mean", "median", "iqr"))))
                          for stat in (row["Clean_Few"], row["Many_to_Few"])]
                r.append(f"| {arm} | {diag['epoch']} | {signal} | {values[0]} | {values[1]} | "
                         f"{row['auc_many_to_few_positive']:.4f} |")
                if "minus_entropy_auc_many_to_few_positive" in row:
                    r.append(f"| {arm} | {diag['epoch']} | negative entropy | | | "
                             f"{row['minus_entropy_auc_many_to_few_positive']:.4f} |")
    r += ["", "## Runtime and protocol evidence", "",
          "| Arm | Epochs | Actual SGD updates | Wall seconds | Peak allocated GiB | Peak reserved GiB |",
          "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for arm, result in summary["training"].items():
        r.append(f"| {arm} | {result['epochs_completed']} | {result['optimizer_updates']} | {result['wall_seconds']:.1f} | "
                 f"{result['peak_allocated_bytes']/1024**3:.3f} | {result['peak_reserved_bytes']/1024**3:.3f} |")
    r += ["", f"Official TEST opens until all TRAIN work completed {summary['official_test_before_gate']}. "
          f"Official TEST file opens during final evaluation {summary['official_test_file_opens_during_final']}.",
          "The environment, input and checkpoint hashes are in the JSON receipts. Per-batch relation counts are in each arm's gate_diagnostics.csv.",
          "The reported differences describe this fixed Seed 42 synthetic symmetric-noise continuation.", ""]
    return "\n".join(r)


def run(args):
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    guard = OfficialTestAudit(args.data_root)
    inputs = locked_inputs(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this fixed FP32 batch-256 comparison")
    device = torch.device("cuda:0")
    torch.cuda.init()
    torch.cuda.set_device(device)
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision("highest")
    set_seed(SEED)
    if (torch.is_autocast_enabled("cuda") or torch.backends.cuda.matmul.allow_tf32
            or torch.backends.cudnn.allow_tf32 or not torch.backends.cudnn.deterministic or torch.backends.cudnn.benchmark):
        raise RuntimeError("Unexpected CUDA precision or cuDNN flags")
    import torchvision

    source_paths = (
        Path(__file__).relative_to(ROOT), Path("BCL_LTN/train_noisy_bcl.py"),
        Path("BCL_LTN/make_noisy_lt.py"), Path("BCL_LTN/analyze_epoch160_source_attribution.py"),
        *(Path(f"BCL_AUDIT/cifar100lt/{name}.py") for name in ("train", "data", "models", "losses")),
    )
    provenance = {
        "seed": SEED, "first_epoch": START + 1, "last_epoch": END, "batch_size": BATCH_SIZE, "num_workers": 0,
        "precision": "fp32", "amp": False, "tf32": False, "cuda": torch.version.cuda, "device": str(device),
        "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__, "torchvision": torchvision.__version__,
        "cudnn": torch.backends.cudnn.version(), "python": sys.executable,
        "input_sha256": inputs.hashes, "few_q90_threshold": inputs.reference.few_q90,
        "reference_few_accepted": inputs.reference_count,
        "source_sha256": {p.as_posix(): file_sha256(ROOT / p) for p in source_paths},
    }
    smoke = None if args.smoke_only else read_json(args.smoke_report)
    if smoke is not None and (smoke["passed"] is not True or smoke["provenance"] != provenance
                              or set(smoke["arms"]) != set(ARMS)
                              or any(r["optimizer_updates"] != 1 for r in smoke["arms"].values())
                              or smoke["official_test_opens"] or smoke["official_test_denied"]
                              or smoke["formal_training_started"] is not False):
        raise RuntimeError("A passed, matching three-arm Smoke is required before formal training")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    json_dump(args.output_dir / "environment.json", provenance)
    if args.smoke_only:
        return run_smoke(args.output_dir, inputs, provenance, guard, device)

    run_start = time.perf_counter()
    resume = {
        "epoch160": START, "seed": SEED, "model_state_sha256": EPOCH160_MODEL_SHA,
        "optimizer_state_sha256": optimizer_fingerprint(inputs.checkpoint["optimizer_state"]),
        "epoch200_original_state_sha256": inputs.original_final_state,
        "smoke_report_sha256": file_sha256(args.smoke_report),
        "verified_start_states_in_smoke": {arm: smoke["arms"][arm]["restored"] for arm in ARMS},
        "input_sha256": inputs.hashes,
    }
    json_dump(args.output_dir / "resume_verification.json", resume)
    results, paired = {}, None
    for arm in ARMS:
        result, history = train_arm(args, arm, inputs, provenance, smoke["arms"][arm], paired, device)
        results[arm] = result
        if arm == ARMS[0]:
            observed = result["checkpoints"][-1]["model_state_sha256"]
            parity = {"passed": observed == inputs.original_final_state, "observed": observed,
                      "expected": inputs.original_final_state,
                      "epochs_with_loss_differences": [epoch for epoch, row in history.items()
                                                       if any(row["loss_deltas_vs_original"].values())]}
            json_dump(args.output_dir / "original_continuation_parity.json", parity)
            if not parity["passed"]:
                raise RuntimeError(f"Original A epoch-200 model state mismatch: {observed} != {inputs.original_final_state}")
            print("[PARITY] Original A epoch 200 is bitwise identical to the archived baseline.", flush=True)
            paired = history
    if guard.opens or guard.denied:
        raise RuntimeError("Official TEST must remain unopened until all three TRAIN arms are verified")
    saved = validate_final_checkpoints(results, inputs)
    json_dump(args.output_dir / "all_training_completed.json", {
        "state": "completed", "arms": list(results), "epochs_per_arm": END - START,
        "optimizer_updates": sum(result["optimizer_updates"] for result in results.values()),
        "paired_sample_and_first_view_hashes": True, "official_test_opens": guard.opens,
        "checkpoint_hashes": {arm: result["checkpoints"] for arm, result in results.items()},
    })
    print("[TEST GATE] All three arms and their TRAIN diagnostics are complete.", flush=True)
    test = evaluate_test(args, guard, results, saved, inputs.reference, device)
    summary = {
        "state": "completed", "training": results, "test": test, "few_q90_threshold": inputs.reference.few_q90,
        "epoch200_original_state_sha256": inputs.original_final_state, "original_continuation_bitwise_parity": True,
        "paired_sample_and_first_view_hashes": True, "official_test_before_gate": 0,
        "official_test_file_opens_during_final": guard.opens, "official_test_denied": guard.denied,
        "total_wall_seconds": time.perf_counter() - run_start, "environment": provenance,
    }
    json_dump(args.output_dir / "comparison_summary.json", summary)
    with (args.output_dir / "COMPARISON_REPORT.md").open("x", encoding="utf-8") as stream:
        stream.write(build_report(summary))
    print(json.dumps({"completed": True, "test_metrics": {arm: result["metrics_pct"] for arm, result in test.items()},
                      "output": str(args.output_dir)}), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-root", "indices-file", "noise-dir", "baseline-dir", "gate-reference", "audit-receipt", "output-dir"):
        parser.add_argument(f"--{name}", type=lambda value: Path(value).resolve(), required=True)
    parser.add_argument("--gate-reference-sha256", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke-only", action="store_true")
    mode.add_argument("--smoke-report", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
