"""Train D0, D1 and D2 on paired fixed feature caches and sampling plans."""
from __future__ import annotations

import csv
import hashlib

import numpy as np
import torch
from torch import nn

from cross_dataset_core import D0, D2, class_weights_from_counts, compute_method_loss
from cross_dataset_models import FEATURE_DIM, assert_encoder_frozen, freeze_encoder, state_dict_sha256
from experiment_common import make_sgd, record_run, require_complete, run_log, run_path, save_status, seed_everything
from extract_features import FEATURES, open_cache
from prepare_experiment import open_plan
from train_stage1 import CHECKPOINT, finite_gradients, load_stage1

HEAD = "final_classifier.pt"


def new_head(num_classes: int, state, device: torch.device) -> nn.Linear:
    classifier = nn.Linear(FEATURE_DIM, num_classes).to(device)
    classifier.load_state_dict(state, strict=True)
    return classifier


def head_step(arm, classifier, teacher, weights, counts, features, labels, epoch, optimizer):
    if features.dtype != torch.float32:
        raise RuntimeError("Stage-II feature input must be FP32")
    optimizer.zero_grad(set_to_none=True)
    loss, diagnostics = compute_method_loss(
        arm, features, labels, classifier, weights, epoch, counts, teacher=teacher,
    )
    loss.backward()
    if not finite_gradients(classifier):
        raise RuntimeError(f"{arm} received missing or nonfinite classifier gradients")
    optimizer.step()
    return float(loss.detach()), diagnostics


def save_head(path, classifier, binding: dict, arm: str, epoch: int, teacher_sha: str | None):
    state_hash = state_dict_sha256(classifier)
    with path.open("xb") as handle:
        torch.save({
            **binding, "arm": arm, "epoch": epoch, "teacher_checkpoint_sha256": teacher_sha,
            "classifier_state_sha256": state_hash,
            "classifier_state": {k: v.detach().cpu() for k, v in classifier.state_dict().items()},
        }, handle)
    return state_hash


def head_status(directory, binding: dict, arm: str, epoch: int, *, phase: str = "stage2") -> dict:
    status = require_complete(directory, phase=phase, expected={
        **binding, "arm": arm, "final_epoch": epoch, "epochs_completed": epoch,
        "official_test_access": False,
    })
    if HEAD not in status["artifacts"] or "classifier_state_sha256" not in status:
        raise RuntimeError(f"Missing classifier checksum in {directory}")
    return status


def load_head(directory, status, binding, arm, epoch, device, *, frozen=False):
    checkpoint = torch.load(directory / HEAD, map_location="cpu", weights_only=True)
    expected = {
        **binding, "arm": arm, "epoch": epoch,
        "classifier_state_sha256": status["classifier_state_sha256"],
        "teacher_checkpoint_sha256": status.get("teacher_checkpoint_sha256"),
    }
    if any(checkpoint.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"Classifier checkpoint provenance mismatch in {directory}")
    classifier = new_head(binding["num_classes"], checkpoint["classifier_state"], device)
    if state_dict_sha256(classifier) != status["classifier_state_sha256"]:
        raise RuntimeError(f"Classifier state digest mismatch in {directory}")
    if frozen:
        classifier.requires_grad_(False).eval()
    return classifier


def assert_teacher_frozen(teacher, expected_sha):
    if (teacher.training or any(p.requires_grad or p.grad is not None for p in teacher.parameters())
        or state_dict_sha256(teacher) != expected_sha):
        raise RuntimeError("D2 teacher or its gradients changed")


def pairing_binding(exp, name, seed, seal, stage1, cache, plan_record):
    return {
        "dataset": name, "seed": seed, "num_classes": exp.datasets[name]["num_classes"],
        "data_fingerprint": seal["data_fingerprint"], "stage1_checkpoint_sha256": stage1["artifacts"][CHECKPOINT],
        "initial_classifier_sha256": stage1["classifier_state_sha256"],
        "feature_cache_sha256": cache["artifacts"][FEATURES], "sampling_plan_sha256": plan_record["sha256"],
        "labels_sha256": seal["labels_sha256"],
    }


def train_arm(exp, arm, binding, inputs, plan_record, initial_state, encoder, snapshot, device):
    name, seed = binding["dataset"], binding["seed"]
    features, labels, plan, counts = inputs
    settings = exp.protocol["stage2"]
    epochs, batch_size = settings["epochs"], settings["batch_size"]
    directory = run_path("stage2", name, seed, arm)
    teacher = teacher_sha = teacher_state_sha = None
    paired_trace = None
    if arm != D0:
        d0_path = run_path("stage2", name, seed, D0)
        reference = head_status(d0_path, binding, D0, epochs)
        paired_trace = reference["epoch_indices_sha256"]
        if arm == D2:
            teacher = load_head(d0_path, reference, binding, D0, epochs, device, frozen=True)
            teacher_sha = reference["artifacts"][HEAD]
            teacher_state_sha = reference["classifier_state_sha256"]
            assert_teacher_frozen(teacher, teacher_state_sha)

    config = exp.config("stage2", **binding, arm=arm, teacher_checkpoint_sha256=teacher_sha,
                        train_class_counts=counts.tolist())
    with record_run(directory, config, device=device) as status:
        status.update(binding, arm=arm, epochs_completed=0, optimizer_steps=0, epoch_indices_sha256=[],
                      teacher_checkpoint_sha256=teacher_sha)
        seed_everything(seed)
        classifier = new_head(binding["num_classes"], initial_state, device)
        if state_dict_sha256(classifier) != binding["initial_classifier_sha256"]:
            raise RuntimeError(f"{arm} did not start from the sealed Stage-I classifier")
        optimizer = make_sgd(classifier.parameters(), settings)
        weights = class_weights_from_counts(counts, device)
        train_counts = torch.as_tensor(counts, dtype=torch.long, device=device)
        n_rows = len(labels)
        with (directory / "epoch_metrics.csv").open("x", encoding="utf-8", newline="") as metrics, (
            directory / "sampling_trace.csv"
        ).open("x", encoding="utf-8", newline="") as trace:
            trace_writer = csv.DictWriter(trace, fieldnames=["epoch", "steps", "samples", "indices_sha256"])
            metric_writer = csv.DictWriter(metrics, fieldnames=[
                "epoch", "alpha", "train_loss", "delta_mean_l2", "mean_head_preservation_kl",
                "many_inputs_distilled", "steps", "samples",
            ])
            trace_writer.writeheader()
            metric_writer.writeheader()
            for epoch, planned in enumerate(plan):
                digest = hashlib.sha256()
                seen = steps = many = 0
                loss_sum = delta_sum = kl_sum = 0.0
                status["current_epoch"] = epoch + 1
                for start in range(0, n_rows, batch_size):
                    status["current_batch"] = steps + 1
                    indices = planned[start:start + batch_size]
                    digest.update(indices.tobytes())
                    x = torch.from_numpy(features[indices]).to(device)
                    y = torch.from_numpy(labels[indices]).to(device)
                    loss, diagnostics = head_step(
                        arm, classifier, teacher, weights, train_counts, x, y, epoch, optimizer,
                    )
                    if arm == D0 and diagnostics["delta_mean_l2"] != 0:
                        raise RuntimeError("D0 used a nonzero feature perturbation")
                    size = len(indices)
                    seen += size
                    steps += 1
                    status["optimizer_steps"] += 1
                    loss_sum += loss * size
                    delta_sum += diagnostics["delta_mean_l2"] * size
                    batch_many = diagnostics["many_inputs_in_batch"]
                    kl_sum += diagnostics["head_preservation_kl"] * batch_many
                    many += batch_many
                trace_hash = digest.hexdigest()
                if (seen != n_rows or trace_hash != plan_record["per_epoch_indices_sha256"][epoch]
                    or (paired_trace is not None and trace_hash != paired_trace[epoch])):
                    raise RuntimeError(f"{arm} consumed an unpaired sampling sequence at epoch {epoch + 1}")
                trace_writer.writerow({"epoch": epoch + 1, "steps": steps, "samples": seen, "indices_sha256": trace_hash})
                row = {
                    "epoch": epoch + 1, "alpha": diagnostics["alpha"], "train_loss": loss_sum / seen,
                    "delta_mean_l2": delta_sum / seen, "mean_head_preservation_kl": kl_sum / max(many, 1),
                    "many_inputs_distilled": many, "steps": steps, "samples": seen,
                }
                metric_writer.writerow(row)
                trace.flush()
                metrics.flush()
                status["epoch_indices_sha256"].append(trace_hash)
                status.update(epochs_completed=epoch + 1, last_epoch_metrics=row)
                save_status(directory, status)
                run_log(directory, f"{name} seed {seed} {arm} epoch {epoch + 1}/{epochs} loss={row['train_loss']:.6f}")
        assert_encoder_frozen(encoder, snapshot)
        if teacher is not None:
            assert_teacher_frozen(teacher, teacher_state_sha)
        if any(not bool(torch.isfinite(parameter).all()) for parameter in classifier.parameters()):
            raise RuntimeError(f"{arm} ended with nonfinite classifier parameters")
        state_sha = save_head(directory / HEAD, classifier, binding, arm, epochs, teacher_sha)
        status.update(final_epoch=epochs, classifier_state_sha256=state_sha,
                      frozen_encoder_and_bn_unchanged=True, teacher_frozen_and_unchanged=teacher is not None)


def train_all(exp, prepared, device):
    for name, seed in exp.jobs():
        seal = prepared["datasets"][name]
        encoder, initial, stage1 = load_stage1(exp, name, seed, seal)
        snapshot = freeze_encoder(encoder)
        features, labels, cache = open_cache(exp, name, seed, seal, stage1)
        plan, plan_record = open_plan(seal, seed, exp.protocol["stage2"]["epochs"])
        binding = pairing_binding(exp, name, seed, seal, stage1, cache, plan_record)
        counts = np.asarray(seal["train_class_counts"], dtype=np.int64)
        for arm in exp.arms:
            train_arm(exp, arm, binding, (features, labels, plan, counts), plan_record,
                      initial.state_dict(), encoder, snapshot, device)
