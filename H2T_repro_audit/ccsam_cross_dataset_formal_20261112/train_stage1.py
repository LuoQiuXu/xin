"""Train the six fixed-epoch, randomly initialized Stage-I backbones."""
from __future__ import annotations

import csv
import math
from itertools import chain

import numpy as np
import torch
from torch.nn import functional as F
from torchvision import transforms

from cross_dataset_models import ARCHITECTURE, FEATURE_DIM, build_stage1_models, state_dict_sha256
from diagnostics.amp_gradient_probe.gradient_evidence import (
    buffer_snapshot, gradients, parameter_snapshot, parameter_updates, restore_buffers,
    scalar_json, scaled_optimizer_step, tensor_summary, trainable_parameters,
)
from experiment_common import (
    array_sha256, make_loader, make_sgd, record_run, require_complete, run_log, run_path, save_status, seed_everything,
)
from prepare_experiment import train_dataset

CHECKPOINT = "epoch_90.pt"
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
CACHE_TRANSFORM = "Resize256_CenterCrop224_ImageNetNormalize"
MAX_SMOKE_AMP_RETRIES = 8


def train_transform(size: int):
    return transforms.Compose([
        transforms.RandomResizedCrop(size), transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])


def eval_transform():
    return transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])


def epoch_lr(epoch: int, total: int, base: float) -> float:
    if epoch <= 5:
        return base * epoch / 5.0
    progress = (epoch - 5) / (total - 5)
    return base * (1.0 + math.cos(math.pi * progress)) / 2.0


def finite_gradients(model) -> bool:
    grads = [parameter.grad for parameter in model.parameters()]
    return bool(grads) and all(grad is not None for grad in grads) and bool(
        torch.stack([torch.isfinite(grad).all() for grad in grads]).all()
    )


def stage1_forward(encoder, classifier, mixed, labels, partner, weight):
    with torch.autocast("cuda", dtype=torch.float16):
        features = encoder(mixed)
        logits = classifier(features)
        loss = weight * F.cross_entropy(logits, labels) + (1.0 - weight) * F.cross_entropy(logits, labels[partner])
    if features.shape != (len(labels), FEATURE_DIM) or logits.shape != (len(labels), classifier.out_features):
        raise RuntimeError(f"Malformed Stage-I features/logits: {features.shape}, {logits.shape}")
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("Nonfinite Stage-I MixUp loss")
    return loss, features, logits


def smoke_amp_update(encoder, classifier, optimizer, scaler, forward, record_attempt):
    """Retry a fixed disposable batch only after a verified GradScaler skip."""
    parameters = trainable_parameters(encoder, classifier)
    before = parameter_snapshot(parameters)
    buffers = buffer_snapshot(encoder, classifier)
    for attempt in range(MAX_SMOKE_AMP_RETRIES + 1):
        if attempt:
            restore_buffers(buffers)
        optimizer.zero_grad(set_to_none=True)
        loss, features, logits = forward()
        forward_info = {
            "features": tensor_summary(features), "logits": tensor_summary(logits),
            "loss": scalar_json(loss), "loss_finite": bool(torch.isfinite(loss)),
        }
        if not (forward_info["features"]["finite"] and forward_info["logits"]["finite"] and forward_info["loss_finite"]):
            raise RuntimeError("Nonfinite Stage-I Smoke forward")

        scale_before = scaler.get_scale()
        scaled_loss = scaler.scale(loss)
        scaled_loss_value = scalar_json(scaled_loss)
        scaled_loss.backward()
        scaled_state, scaled_rows = gradients(parameters, optimizer)
        scaler.unscale_(optimizer)
        state, rows = gradients(parameters, optimizer)
        missing, nonfinite = state["missing_names"], state["nonfinite_names"]
        outside_optimizer = state["not_in_optimizer_names"]
        observed_scaler = not (missing or outside_optimizer)
        stepped = scaled_optimizer_step(scaler, optimizer) if observed_scaler else False
        scale_after = scaler.get_scale()
        update, delta_rows = parameter_updates(parameters, before)
        changed = update["changed_parameter_tensors"]
        if missing:
            outcome = "missing_gradient"
            error = f"Missing Stage-I gradient for: {', '.join(missing)}"
        elif outside_optimizer:
            outcome = "optimizer_membership_error"
            error = f"Trainable Stage-I parameters outside the optimizer: {', '.join(outside_optimizer)}"
        elif nonfinite:
            if stepped or changed or not scale_after < scale_before:
                outcome = "nonfinite_gradient_without_verified_skip"
                error = f"Invalid GradScaler transition for nonfinite Stage-I gradients: {', '.join(nonfinite)}"
            else:
                outcome = "gradscaler_skipped_nonfinite"
                error = None
        elif stepped and changed and update["parameters_finite_after"]:
            outcome, error = "finite_optimizer_update", None
        else:
            outcome = "no_effective_finite_update"
            error = "Stage-I Smoke did not perform an effective optimizer update with finite parameters"

        summary = {
            "attempt": attempt, "outcome": outcome, "scale_before": scale_before, "scale_after": scale_after,
            "forward": forward_info, "scaled_loss": scaled_loss_value, "optimizer_step": stepped,
            "gradscaler_update_called": observed_scaler, "scaled_gradient_state": scaled_state,
            "unscaled_gradient_state": state, "update": update,
        }
        record_attempt(summary, scaled_rows, rows, delta_rows)
        if error:
            raise RuntimeError(error)
        if outcome == "finite_optimizer_update":
            return loss, features, logits, summary
    raise RuntimeError(f"Stage-I Smoke exhausted {MAX_SMOKE_AMP_RETRIES} GradScaler retries")


def stage1_batch(encoder, classifier, optimizer, scaler, images, labels, mix_rng, permutation_rng,
                 alpha: float, *, strict_gradients: bool = False, record_attempt=None):
    optimizer.zero_grad(set_to_none=True)
    weight = float(mix_rng.beta(alpha, alpha))
    partner = torch.randperm(len(labels), generator=permutation_rng, device=labels.device)
    mixed = weight * images + (1.0 - weight) * images[partner]

    def forward():
        return stage1_forward(encoder, classifier, mixed, labels, partner, weight)

    evidence = {}
    if strict_gradients:
        if not bool(torch.isfinite(images).all() & torch.isfinite(mixed).all()):
            raise RuntimeError("Nonfinite Stage-I Smoke image input")
        identity = {
            "input_images_finite": True, "mixed_images_finite": True,
            "images_sha256": array_sha256(images.cpu().numpy()),
            "labels_sha256": array_sha256(labels.cpu().numpy()),
            "mixed_images_sha256": array_sha256(mixed.cpu().numpy()),
            "mixup_lambda": weight, "mixup_permutation": partner.cpu().tolist(),
        }

        def trace(summary, scaled_rows, rows, delta_rows):
            if record_attempt is not None:
                record_attempt(identity, summary, scaled_rows, rows, delta_rows)

        loss, features, logits, final = smoke_amp_update(encoder, classifier, optimizer, scaler, forward, trace)
        scale_before, scale_after = final["scale_before"], final["scale_after"]
        stepped = final["optimizer_step"]
        evidence = {"batch_identity": identity, "amp_attempt_count": final["attempt"] + 1,
                    "final_unscaled_gradient_state": final["unscaled_gradient_state"], "update": final["update"]}
    else:
        loss, features, logits = forward()
        scale_before = scaler.get_scale()
        scaler.scale(loss).backward()
        stepped = scaled_optimizer_step(scaler, optimizer)
        scale_after = scaler.get_scale()
    prediction = logits.detach().argmax(dim=1)
    correct = weight * (prediction == labels).sum() + (1.0 - weight) * (prediction == labels[partner]).sum()
    return {
        "loss": float(loss.detach()), "mixup_correct": float(correct), "mixup_lambda": weight,
        "scale_before": scale_before, "scale_after": scale_after, "optimizer_step": stepped,
        "feature_shape": list(features.shape), "logit_shape": list(logits.shape),
        **evidence,
    }


def train_seed(exp, dataset, name: str, seed: int, seal: dict, device: torch.device) -> None:
    settings = exp.protocol["stage1"]
    directory = run_path("stage1", name, seed)
    binding = {"dataset": name, "seed": seed, "data_fingerprint": seal["data_fingerprint"]}
    config = exp.config("stage1", **binding, train_labels_sha256=seal["labels_sha256"])
    with record_run(directory, config, device=device) as status:
        status.update(binding, epochs_completed=0, optimizer_steps=0, amp_skips=0)
        seed_everything(seed)
        encoder, classifier = build_stage1_models(exp.datasets[name]["num_classes"])
        encoder.to(device).train()
        classifier.to(device).train()
        optimizer = make_sgd(chain(encoder.parameters(), classifier.parameters()), settings)
        scaler = torch.amp.GradScaler("cuda")
        mix_rng = np.random.default_rng(seed)
        permutation_rng = torch.Generator(device=device).manual_seed(seed)
        loader = make_loader(dataset, settings["batch_size"], exp.workers, seed, shuffle=True)
        with (directory / "epoch_metrics.csv").open("x", encoding="utf-8", newline="") as handle:
            columns = ["epoch", "lr", "train_mixup_loss", "train_mixup_accuracy", "samples", "batches",
                       "optimizer_steps", "amp_skips", "amp_scale"]
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for epoch in range(1, settings["epochs"] + 1):
                lr = epoch_lr(epoch, settings["epochs"], settings["learning_rate"])
                for group in optimizer.param_groups:
                    group["lr"] = lr
                loss_sum = correct = 0.0
                seen = steps = batches = 0
                status["current_epoch"] = epoch
                for batch, (images, labels) in enumerate(loader, 1):
                    status["current_batch"] = batch
                    images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                    result = stage1_batch(encoder, classifier, optimizer, scaler, images, labels, mix_rng,
                                          permutation_rng, settings["mixup_alpha"])
                    seen += len(labels)
                    loss_sum += result["loss"] * len(labels)
                    correct += result["mixup_correct"]
                    steps += int(result["optimizer_step"])
                    batches = batch
                    status["optimizer_steps"] += int(result["optimizer_step"])
                    status["amp_skips"] += int(not result["optimizer_step"])
                    if batch % 250 == 0:
                        run_log(directory, f"{name} seed {seed} epoch {epoch} batch {batch}/{len(loader)}")
                        save_status(directory, status)
                if seen != seal["train_rows"]:
                    raise RuntimeError(f"Stage-I did not consume all official train rows: {seen}")
                row = {
                    "epoch": epoch, "lr": lr, "train_mixup_loss": loss_sum / seen,
                    "train_mixup_accuracy": 100.0 * correct / seen, "samples": seen, "batches": batches,
                    "optimizer_steps": steps, "amp_skips": batches - steps, "amp_scale": scaler.get_scale(),
                }
                writer.writerow(row)
                handle.flush()
                status.update(epochs_completed=epoch, last_epoch_metrics=row)
                save_status(directory, status)
                run_log(directory, f"{name} seed {seed} epoch {epoch}/{settings['epochs']} loss={row['train_mixup_loss']:.6f} optimizer_steps={steps}")
        del loader
        for module in (encoder, classifier):
            if any(value.is_floating_point() and not bool(torch.isfinite(value).all())
                   for value in module.state_dict().values()):
                raise RuntimeError("Nonfinite Stage-I final parameter or buffer")
        hashes = {"encoder_state_sha256": state_dict_sha256(encoder),
                  "classifier_state_sha256": state_dict_sha256(classifier)}
        with (directory / CHECKPOINT).open("xb") as handle:
            torch.save({
                **binding, **hashes, "epoch": settings["epochs"], "architecture": ARCHITECTURE,
                "num_classes": classifier.out_features, "config": config,
                "encoder_state": {k: v.detach().cpu() for k, v in encoder.state_dict().items()},
                "classifier_state": {k: v.detach().cpu() for k, v in classifier.state_dict().items()},
            }, handle)
        status.update(hashes, final_epoch=settings["epochs"], checkpoint=CHECKPOINT)


def stage1_status(exp, name: str, seed: int, seal: dict) -> dict:
    directory = run_path("stage1", name, seed)
    epoch = exp.protocol["stage1"]["epochs"]
    status = require_complete(directory, phase="stage1", expected={
        "dataset": name, "seed": seed, "data_fingerprint": seal["data_fingerprint"],
        "epochs_completed": epoch, "final_epoch": epoch, "checkpoint": CHECKPOINT, "official_test_access": False,
    })
    if CHECKPOINT not in status["artifacts"] or "epoch_metrics.csv" not in status["artifacts"]:
        raise RuntimeError(f"Missing Stage-I artifact hashes in {directory}")
    return status


def load_stage1(exp, name: str, seed: int, seal: dict, status: dict | None = None):
    status = stage1_status(exp, name, seed, seal) if status is None else status
    checkpoint = torch.load(run_path("stage1", name, seed) / CHECKPOINT, map_location="cpu", weights_only=True)
    expected = {
        "dataset": name, "seed": seed, "epoch": exp.protocol["stage1"]["epochs"],
        "architecture": ARCHITECTURE, "num_classes": exp.datasets[name]["num_classes"],
        "data_fingerprint": seal["data_fingerprint"],
        "encoder_state_sha256": status["encoder_state_sha256"],
        "classifier_state_sha256": status["classifier_state_sha256"],
    }
    if any(checkpoint.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"Stage-I checkpoint provenance mismatch for {name} seed {seed}")
    encoder, classifier = build_stage1_models(expected["num_classes"])
    encoder.load_state_dict(checkpoint["encoder_state"], strict=True)
    classifier.load_state_dict(checkpoint["classifier_state"], strict=True)
    if state_dict_sha256(encoder) != expected["encoder_state_sha256"] or state_dict_sha256(classifier) != expected["classifier_state_sha256"]:
        raise RuntimeError(f"Stage-I state digest mismatch for {name} seed {seed}")
    return encoder, classifier, status


def train_all(exp, prepared: dict, device: torch.device) -> None:
    for name in exp.datasets:
        seal = prepared["datasets"][name]
        with train_dataset(exp, name, train_transform(exp.protocol["stage1"]["input_size"]), seal) as (dataset, _):
            for seed in exp.seeds:
                train_seed(exp, dataset, name, seed, seal, device)
