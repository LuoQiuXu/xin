"""Seed-42 real-data smoke plus isolated B0 Stage-I validation.

The default path runs the two Stage-I and four Stage-II one-epoch smoke
conditions. --b0-stage1-only runs only one full B0 Stage-I epoch and validation.
There is deliberately no formal-training or TEST-evaluation entry point here.
"""
import argparse
from collections import Counter
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
from torch import nn

import models
from data import load_rows, schedules, Views, loader, transform
from preflight import EXPECTED_SHA, MANIFEST
from protocol_config import (BRANCH, NUMERIC_MODE, CPU_VALIDATION_SOURCE_FILES, FORMAL_RUNNER_SOURCE_FILES, LOCKED_UPSTREAM_SOURCE_HASHES, EXPECTED_CPU_TEST_METHODS)
from amp_recovery import (
    GradientContractError, OptimizerStepCounter, guarded_scaler_step, make_grad_scaler
)


def save_json(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def write_json(path, value):
    """Atomically update the small execution-status record in a fresh run dir."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)


def digest_tensor(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def seed():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)


def cleanup(model):
    model.cpu()
    gc.collect()
    torch.cuda.empty_cache()


def verify_gate_records(preflight_record, cpu_validation_record):
    """Bind Smoke to successful CPU/preflight gates on this exact source tree."""
    preflight_path = Path(preflight_record).resolve()
    cpu_path = Path(cpu_validation_record).resolve()
    assert preflight_path.is_file(), f"Missing preflight record: {preflight_path}"
    assert cpu_path.is_file(), f"Missing CPU validation record: {cpu_path}"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    cpu = json.loads(cpu_path.read_text(encoding="utf-8"))

    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=models.ROOT, text=True
    ).strip()
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=models.ROOT, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=models.ROOT, text=True
    ).strip()
    assert not dirty, "Git worktree must be clean before Smoke"
    assert branch == BRANCH, f"Wrong Smoke branch: {branch}"
    assert preflight.get("status") == "PASS_PREFLIGHT_ONLY", "Preflight did not pass"
    assert cpu.get("status") == "PASS", "CPU validation did not pass"
    assert preflight.get("branch") == branch and cpu.get("branch") == branch
    assert preflight.get("head") == head and cpu.get("head") == head, (
        "CPU/preflight records do not match the current HEAD"
    )
    assert preflight.get("worktree_clean") is True
    assert cpu.get("worktree_clean_before") is True and cpu.get("worktree_clean_after") is True
    assert preflight["data_gate"]["manifest_sha256"] == EXPECTED_SHA
    assert preflight["data_gate"]["test_image_bytes_read"] == 0
    assert preflight.get("test_image_bytes_read") == 0
    assert preflight["model_gate"]["status"] == "PASS"
    assert preflight["model_gate"]["official_weight_sha256"] == models.ORIGINAL_SHA
    assert preflight["model_gate"]["upstream_source_hashes"] == LOCKED_UPSTREAM_SOURCE_HASHES
    assert models.source_hashes() == LOCKED_UPSTREAM_SOURCE_HASHES

    current_source_hashes = {
        name: models.sha(Path(__file__).parent / name)
        for name in CPU_VALIDATION_SOURCE_FILES
    }
    assert cpu.get("source_sha256") == current_source_hashes, (
        "Source tree changed after CPU validation"
    )
    assert preflight.get("source_sha256") == current_source_hashes, (
        "Source tree changed after preflight"
    )
    runner_hashes = {
        name: current_source_hashes[name]
        for name in FORMAL_RUNNER_SOURCE_FILES
    }
    assert cpu.get("source_hashes") == runner_hashes, (
        "CPU validation record is incompatible with the formal runner"
    )
    assert cpu.get("test_methods_run") == EXPECTED_CPU_TEST_METHODS
    checks = {item["name"]: item for item in cpu.get("checks", [])}
    assert checks.get("SYNTAX", {}).get("exit_code") == 0
    assert checks.get("CPU_ALL_SUITES", {}).get("exit_code") == 0
    assert checks.get("CPU_ALL_SUITES", {}).get("test_methods_run") == EXPECTED_CPU_TEST_METHODS

    return {
        "cpu_validation_record": str(cpu_path),
        "cpu_validation_sha256": models.sha(cpu_path),
        "cpu_test_methods": cpu["test_methods_run"],
        "preflight_record": str(preflight_path),
        "preflight_record_sha256": models.sha(preflight_path),
        "preflight_status": preflight["status"],
        "test_image_bytes_read": 0,
        "source_sha256": current_source_hashes,
    }


def finite_gradients(parameters):
    gradients = [p.grad for p in parameters if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients), "Nonfinite/missing gradients"
    assert any(g.abs().sum().item() > 0 for g in gradients), "All gradients zero"


def cpu_tree(value):
    """Copy nested optimizer/model state to CPU for a portable failure snapshot."""
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def gradient_failure_diagnostics(named_parameters):
    """Expensive per-parameter detail, called only after a rejected gradient window."""
    summaries = []
    missing = []
    for name, parameter in named_parameters:
        gradient = parameter.grad
        if gradient is None:
            missing.append(name)
            continue
        gradient = gradient.detach()
        if gradient.is_sparse:
            gradient = gradient.coalesce().values()
        finite_mask = torch.isfinite(gradient)
        nan_mask = torch.isnan(gradient)
        positive_inf_mask = torch.isposinf(gradient)
        negative_inf_mask = torch.isneginf(gradient)
        nan_count = int(nan_mask.sum().item())
        positive_inf_count = int(positive_inf_mask.sum().item())
        negative_inf_count = int(negative_inf_mask.sum().item())
        if nan_count or positive_inf_count or negative_inf_count:
            finite_values = gradient[finite_mask]
            bad_positions = torch.nonzero(~finite_mask, as_tuple=False)[:64].cpu().tolist()
            summaries.append({
                "name": name,
                "parameter_shape": list(parameter.shape),
                "gradient_shape": list(gradient.shape),
                "gradient_dtype": str(gradient.dtype),
                "nan_count": nan_count,
                "positive_inf_count": positive_inf_count,
                "negative_inf_count": negative_inf_count,
                "nonfinite_count": nan_count + positive_inf_count + negative_inf_count,
                "nonfinite_positions_first_64": bad_positions[:64],
                "positions_truncated": nan_count + positive_inf_count + negative_inf_count > 64,
                "finite_count": int(finite_mask.sum().item()),
                "finite_min": float(finite_values.min().item()) if finite_values.numel() else None,
                "finite_max": float(finite_values.max().item()) if finite_values.numel() else None,
                "finite_max_abs": float(finite_values.abs().max().item()) if finite_values.numel() else None,
                "finite_l2": float(torch.linalg.vector_norm(finite_values.float()).item()) if finite_values.numel() else None,
            })
    return {
        "optimizer_parameter_count": len(named_parameters),
        "missing_parameters": missing,
        "nonfinite_parameters": summaries,
        "all_present": not missing and len(named_parameters) > 0,
        "all_finite": not missing and not summaries and len(named_parameters) > 0,
    }


def save_b0_gradient_failure(
    folder, *, model, optimizer, scaler, named_parameters, step_counter,
    failure_context, exception, mixed_input, target, logits, loss, mixup_context,
    item, optimizer_step_events,
):
    """Persist the exact rejected B0 window locally before the process exits."""
    torch.cuda.synchronize()
    grad_summary = gradient_failure_diagnostics(named_parameters)
    failure = {
        **failure_context,
        "optimizer_updates_at_exception": int(step_counter.count),
        "exception_type": type(exception).__name__,
        "exception_message": str(exception),
        "guard_diagnostics": getattr(exception, "diagnostics", {}),
        "gradient_diagnostics": grad_summary,
        "optimizer_updates_before_failure": int(
            failure_context.get("optimizer_updates_before_failed_window", step_counter.count)
        ),
        "optimizer_step_events_before_failure": optimizer_step_events,
        "pif_parameters_at_failure": [
            float(value)
            for parameter in model.extractor.pif.parameters()
            for value in parameter.detach().cpu().reshape(-1).tolist()
        ],
        "batch_indices_external_only": item["index"].detach().cpu().tolist(),
        "batch_positions_external_only": item["position"].detach().cpu().tolist(),
    }
    save_json(folder / "GRADIENT_FAILURE.json", failure)

    nonfinite_gradients = {}
    for name, parameter in named_parameters:
        gradient = parameter.grad
        if gradient is None:
            continue
        values = gradient.detach()
        if values.is_sparse:
            values = values.coalesce().values()
        if not torch.isfinite(values).all() or name == "extractor.pif.fuse.weight":
            nonfinite_gradients[name] = values.cpu().clone()

    snapshot = {
        "scope": "B0_STAGE1_GRADIENT_FAILURE_FORENSICS_ONLY",
        "batch_index": failure_context["batch_index"],
        "model_state_after_forward_before_optimizer_step": cpu_tree(model.state_dict()),
        "optimizer_state_before_failed_step": cpu_tree(optimizer.state_dict()),
        "scaler_state_after_unscale_before_update": cpu_tree(scaler.state_dict()),
        "mixed_input": mixed_input.detach().cpu().clone(),
        "original_target": target.detach().cpu().clone(),
        "mixup_target_a": cpu_tree(mixup_context["target_a"]) if mixup_context else None,
        "mixup_target_b": cpu_tree(mixup_context["target_b"]) if mixup_context else None,
        "mixup_lambda": mixup_context["lambda"] if mixup_context else None,
        "logits": logits.detach().float().cpu().clone(),
        "loss": loss.detach().float().cpu().clone(),
        "nonfinite_or_pif_gradients": nonfinite_gradients,
        "cpu_rng_state": torch.get_rng_state().clone(),
        "cuda_rng_states": [state.cpu().clone() for state in torch.cuda.get_rng_state_all()],
    }
    with (folder / "GRADIENT_FAILURE_SNAPSHOT.pt").open("xb") as handle:
        torch.save(snapshot, handle)
    print("B0_GRADIENT_FAILURE_SNAPSHOT_SAVED", flush=True)


def metrics(logits, labels, groups):
    predictions = logits.argmax(1)
    correct = predictions == labels
    per_class = [float(correct[labels == c].float().mean() * 100) for c in range(16)]
    return {"overall": float(correct.float().mean() * 100), "macro": sum(per_class) / 16,
            "worst_class": min(per_class), "per_class": per_class,
            "group_image": {g: float(correct[torch.isin(labels, torch.tensor(ids))].float().mean() * 100)
                            for g, ids in groups.items()},
            "group_macro": {g: sum(per_class[i] for i in ids) / len(ids) for g, ids in groups.items()}}


@torch.no_grad()
def evaluate_validation(model, rows, val, groups, batch):
    model.eval()
    outputs, labels = [], []
    for item in loader(Views(rows, val, False), batch):
        with torch.amp.autocast("cuda", enabled=False):
            logits = model(item["recipient"].cuda(non_blocking=True))
        assert torch.isfinite(logits).all()
        outputs.append(logits.float().cpu())
        labels.append(item["label"])
    outputs, labels = torch.cat(outputs), torch.cat(labels)
    assert len(labels) == 5396
    return metrics(outputs, labels, groups)


def batch_smoke(rows, rec, donors, batch, output):
    # This controlled batch is discarded; it never initializes a later smoke phase.
    one = next(iter(loader(Views(rows, rec[:batch], True, donors=donors[:batch]), batch)))
    proofs = []
    for name, pif, mode in (("A1", False, "crt"), ("A2", False, "h2t"), ("B1", True, "crt"), ("B2", True, "pi")):
        seed()
        model = models.FishModel(pif).cuda()
        model.freeze()
        before = models.state_sha(model.extractor)
        weight_before = model.classifier.weight.detach().clone()
        opt = torch.optim.SGD(model.classifier.parameters(), lr=.01, momentum=.9, weight_decay=5e-4)
        opt.zero_grad(set_to_none=True)
        scaler = make_grad_scaler()
        assert not scaler.is_enabled(), "FP32 batch smoke must not enable GradScaler"
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
            r = model.extractor(one["recipient"].cuda())
            d = model.extractor(one["donor"].cuda()) if mode != "crt" else None
            assert r.shape == (batch, 2048, 7, 7)
            fused, extra = models.stage2_features(r, d, one["label"].cuda(), model.classifier, mode)
        with torch.amp.autocast("cuda", enabled=False):
            loss = nn.functional.cross_entropy(model.classifier(fused), one["label"].cuda())
        assert torch.isfinite(loss)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        finite_gradients(model.classifier.parameters())
        assert all(p.grad is None for p in model.extractor.parameters())
        scaler.step(opt)
        scaler.update()
        assert not torch.equal(model.classifier.weight, weight_before)
        assert before == models.state_sha(model.extractor)
        proofs.append({"condition": name, "mode": mode, "feature_shape": list(r.shape),
                       "dtype": str(r.dtype), "loss": float(loss.detach()), "classifier_updated": True,
                       "extractor_and_bn_unchanged": True, "h2t_channels": len(extra) if mode == "h2t" else None,
                       "ratio_detached": not extra.requires_grad if mode == "pi" else None})
        cleanup(model)
        del model, opt, scaler, r, d, fused, loss, weight_before
        torch.cuda.empty_cache()
    save_json(output / "BATCH_SMOKE.json", proofs)
    print("REAL_BATCH_SMOKE_PASS_ALL_FOUR_CONDITIONS", flush=True)


def one_epoch(name, model, rows, indices, donors, mode, stage, batch, groups, val, out, max_batches=None):
    seed()
    folder = out / name
    folder.mkdir(exist_ok=False)
    model.cuda()
    if stage == 1:
        model.train()
        parameters = list(model.parameters())
        lr, wd = .01 / 5, 1e-4  # Pilot's first warmup epoch, not a 1-epoch-rescaled schedule.
    else:
        model.freeze()
        parameters = list(model.classifier.parameters())
        lr, wd = .01, 5e-4  # First epoch of fixed 10-epoch cosine schedule.
    before_extractor = models.state_sha(model.extractor)
    before_classifier = models.state_sha(model.classifier)
    before_pif = [float(p.detach().flatten()[i]) for p in model.extractor.pif.parameters() for i in range(p.numel())]
    opt = torch.optim.SGD(parameters, lr=lr, momentum=.9, weight_decay=wd)
    scaler = make_grad_scaler()
    assert not scaler.is_enabled(), "FP32 Smoke must not enable GradScaler"
    step_counter = OptimizerStepCounter(opt)
    optimizer_ids = {id(p) for group in opt.param_groups for p in group["params"]}
    named_optimizer_parameters = [
        (name, p) for name, p in model.named_parameters() if id(p) in optimizer_ids
    ]
    assert len(named_optimizer_parameters) == len(parameters)
    accumulation = 16 // batch
    dataset = Views(rows, indices, True, donors=donors)
    batches = loader(dataset, batch)
    controls = []
    channel_indices, ratios, ratio_labels = [], [], []
    opt.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    loss_sum, updates, samples, gradient_audit_windows = 0., 0, 0, 0
    attempted_steps = 0
    overflow_skips = 0
    consecutive_overflow_skips = 0
    max_consecutive_overflow_skips = 0
    optimizer_step_events = []
    for i, item in enumerate(batches):
        if max_batches is not None and i >= max_batches:
            break
        target = item["label"].cuda(non_blocking=True)
        mixup_context = None
        row = {"batch": i, "samples": len(target), "recipient_sha256": digest_tensor(item["recipient"]),
               "donor_sha256": digest_tensor(item["donor"]) if donors is not None else None}
        x = item["recipient"].cuda(non_blocking=True)
        if stage == 1:
            row["mixup_rng_sha256"] = digest_tensor(torch.get_rng_state())
            x, a, b, lam = models.methods.mixup_data(x, target, alpha=1.0)
            mixup_context = {"target_a": a.detach().cpu().clone(),
                             "target_b": b.detach().cpu().clone(), "lambda": float(lam)}
            row["mixup_lambda"] = float(lam)
            with torch.amp.autocast("cuda", enabled=False):
                logits = model(x)
                loss = lam * nn.functional.cross_entropy(logits, a) + (1-lam) * nn.functional.cross_entropy(logits, b)
        else:
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
                r = model.extractor(x)
                d = model.extractor(item["donor"].cuda(non_blocking=True)) if donors is not None else None
                features, extra = models.stage2_features(r, d, target, model.classifier, mode)
                if mode == "h2t":
                    assert len(extra) == 512
                    channel_indices.append(extra.numpy().astype(np.int16))
                elif mode == "pi":
                    assert torch.isfinite(extra).all() and not extra.requires_grad
                    assert (extra >= 0).all() and (extra <= 1).all()
                    ratios.append(extra.float().cpu().numpy())
                    ratio_labels.append(item["label"].numpy())
            with torch.amp.autocast("cuda", enabled=False):
                logits = model.classifier(features)
                loss = nn.functional.cross_entropy(logits, target)
        assert torch.isfinite(loss) and torch.isfinite(logits).all(), "Nonfinite forward"
        group_start = (i // accumulation) * 16
        group_size = min(16, len(dataset)-group_start)
        scaler.scale(loss * len(target) / group_size).backward()
        if (i+1) % accumulation == 0 or i+1 == len(batches):
            scale_before_unscale = float(scaler.get_scale())
            optimizer_updates_before_window = int(step_counter.count)
            try:
                scaler.unscale_(opt)
                assert model.classifier.weight.grad is not None
                if stage == 2:
                    assert all(p.grad is None for p in model.extractor.parameters())
                outcome = guarded_scaler_step(
                    optimizer=opt,
                    scaler=scaler,
                    named_parameters=named_optimizer_parameters,
                    step_counter=step_counter,
                    allow_pif_b_overflow=False,
                    consecutive_overflow_skips=consecutive_overflow_skips,
                    max_consecutive_overflow_skips=2,
                )
            except GradientContractError as exc:
                if name == "B0" and stage == 1:
                    failure_context = {
                        "condition": name,
                        "stage": stage,
                        "batch_index": i,
                        "batch_start": group_start,
                        "batch_end": i,
                        "batch_samples": len(target),
                        "samples_completed_before_failed_batch": samples,
                        "attempted_optimizer_windows_before_failure": attempted_steps,
                        "failed_optimizer_window_number": attempted_steps + 1,
                        "loss_scale_before_unscale": scale_before_unscale,
                        "loss_scale_after_unscale": float(scaler.get_scale()),
                        "loss": float(loss.detach().float().item()),
                        "loss_finite": bool(torch.isfinite(loss).item()),
                        "logits_finite": bool(torch.isfinite(logits).all().item()),
                        "batch_control": row,
                        "mixup_lambda": mixup_context["lambda"] if mixup_context else None,
                        "pif_parameter_names": [n for n, _ in model.extractor.pif.named_parameters()],
                        "optimizer_updates_before_failed_window": optimizer_updates_before_window,
                        "failed_optimizer_step_executed": int(step_counter.count) > optimizer_updates_before_window,
                        "scaler_step_called_for_rejected_window": bool(getattr(exc, "diagnostics", {}).get("scaler_step_called", False)),
                        "scaler_update_called_for_rejected_window": bool(getattr(exc, "diagnostics", {}).get("scaler_update_called", False)),
                    }
                    save_b0_gradient_failure(
                        folder, model=model, optimizer=opt, scaler=scaler,
                        named_parameters=named_optimizer_parameters, step_counter=step_counter,
                        failure_context=failure_context, exception=exc,
                        mixed_input=x, target=target, logits=logits, loss=loss,
                        mixup_context=mixup_context, item=item,
                        optimizer_step_events=optimizer_step_events,
                    )
                step_counter.close()
                raise
            attempted_steps += 1
            gradient_audit_windows += 1
            if outcome.skipped:
                overflow_skips += 1
                consecutive_overflow_skips += 1
                max_consecutive_overflow_skips = max(
                    max_consecutive_overflow_skips, consecutive_overflow_skips
                )
            else:
                updates += 1
                consecutive_overflow_skips = 0
            event = outcome.as_dict()
            event.update({
                "condition": name,
                "batch_start": group_start,
                "batch_end": i,
                "optimizer_updates_total": updates,
                "optimizer_skips_total": overflow_skips,
                "consecutive_optimizer_skips": consecutive_overflow_skips,
            })
            optimizer_step_events.append(event)
            opt.zero_grad(set_to_none=True)
        samples += len(target)
        loss_sum += float(loss.detach()) * len(target)
        controls.append(row)
        if (i+1) % 100 == 0:
            print(f"{name} smoke epoch1 batch {i+1}/{len(batches)}, loss={float(loss.detach()):.5f}", flush=True)
    step_counter.close()
    torch.cuda.synchronize()
    duration = time.perf_counter()-start
    if max_batches is not None:
        partial = {
            "condition": name,
            "status": "BOUNDED_FORENSIC_WINDOW_COMPLETED_NO_FULL_EPOCH",
            "stage": stage,
            "batches_processed": len(controls),
            "samples": samples,
            "attempted_optimizer_steps": attempted_steps,
            "optimizer_updates": updates,
            "optimizer_skips": overflow_skips,
            "grad_scaler_enabled": scaler.is_enabled(), "numerical_mode": NUMERIC_MODE,
            "train_loss_so_far": loss_sum / samples if samples else None,
            "train_seconds": duration,
            "peak_allocated_mib": torch.cuda.max_memory_allocated()/1024**2,
            "pif_parameters_initial": before_pif,
            "pif_parameters_after_window": [
                float(value)
                for parameter in model.extractor.pif.parameters()
                for value in parameter.detach().cpu().reshape(-1).tolist()
            ],
            "note": "Partial B0 forensic window only; no validation, checkpoint, or six-condition completion claim.",
        }
        save_json(folder / "PARTIAL_FORENSIC_RESULT.json", partial)
        save_json(folder / "optimizer_step_events.json", optimizer_step_events)
        save_json(folder / "input_controls.json", controls)
        cleanup(model)
        del opt, scaler, model, batches
        gc.collect()
        torch.cuda.empty_cache()
        return partial, controls

    expected_attempts = math.ceil(16269 / 16)
    assert samples == 16269 and attempted_steps == expected_attempts
    assert updates + overflow_skips == attempted_steps
    assert updates == step_counter.count
    assert overflow_skips == 0, "FP32 protocol permits no optimizer skip or AMP recovery"
    assert max_consecutive_overflow_skips == 0
    save_json(folder / "optimizer_step_events.json", optimizer_step_events)
    if stage == 2:
        assert before_extractor == models.state_sha(model.extractor), "Extractor or BN mutated"
    assert before_classifier != models.state_sha(model.classifier)
    validation = evaluate_validation(model, rows, val, groups, batch)
    after_pif = [float(p.detach().flatten()[i]) for p in model.extractor.pif.parameters() for i in range(p.numel())]
    checkpoint = folder / "smoke_epoch001.pth"
    with checkpoint.open("xb") as f:
        torch.save({"smoke_only": True, "condition": name, "epoch": 1, "seed": 42,
                    "model": model.state_dict(), "optimizer": opt.state_dict(),
                    "scaler": scaler.state_dict(), "validation": validation}, f)
    save_json(folder / "input_controls.json", controls)
    result = {"condition": name, "scope": "ONE_EPOCH_SMOKE_ONLY", "stage": stage,
              "samples": samples, "optimizer_updates": updates,
              "attempted_optimizer_steps": attempted_steps, "optimizer_skips": overflow_skips,
              "max_consecutive_optimizer_skips": max_consecutive_overflow_skips,
              "grad_scaler_enabled": scaler.is_enabled(), "numerical_mode": NUMERIC_MODE,
              "gradient_audit_windows": gradient_audit_windows,
              "lr": lr, "momentum": .9, "weight_decay": wd, "micro_batch": batch,
              "accumulation": accumulation, "effective_batch": 16, "amp": False,
              "autocast_enabled": False, "workers": 4,
              "train_loss": loss_sum / samples, "train_seconds": duration,
              "train_images_per_second": samples / duration,
              "peak_allocated_mib": torch.cuda.max_memory_allocated()/1024**2,
              "validation": validation, "pif_before": before_pif, "pif_after": after_pif,
              "extractor_initial_sha256": before_extractor,
              "extractor_final_sha256": models.state_sha(model.extractor),
              "classifier_initial_sha256": before_classifier,
              "checkpoint": str(checkpoint), "checkpoint_sha256": models.sha(checkpoint),
              "selection": "Only epoch1; smoke best/current/final refer to this same file; validation only"}
    if channel_indices:
        channels = np.stack(channel_indices)
        np.savez_compressed(folder / "h2t_channel_indices.npz", indices=channels)
        save_json(folder / "h2t_channel_audit.json", {"batches": len(channels), "count_per_batch": 512,
                  "batch_shared": True, "first_indices": channels[0].tolist(),
                  "selection_counts": np.bincount(channels.ravel(), minlength=2048).tolist()})
    if ratios:
        values, labels = np.concatenate(ratios), np.concatenate(ratio_labels)
        np.savez_compressed(folder / "pi_ratios.npz", recipient_mass=values, target=labels)
        def stats(v):
            return {"count": len(v), "mean": float(v.mean()), "std": float(v.std()),
                    "min": float(v.min()), "max": float(v.max()), "histogram_20": np.histogram(v, bins=20, range=(0, 1))[0].tolist()}
        save_json(folder / "pi_ratio_audit.json", {"ratio_detach": True, "recipient_mass": stats(values),
                  "donor_mass": stats(1-values), "per_class": {str(c): {"recipient_mass": stats(values[labels==c]),
                  "donor_mass": stats(1-values[labels==c])} for c in range(16)}})
    save_json(folder / "result.json", result)
    cleanup(model)
    del opt, scaler, model, batches
    gc.collect()
    torch.cuda.empty_cache()
    print(f"{name} SMOKE_DONE train_seconds={duration:.2f}, val={validation['overall']:.4f}", flush=True)
    return result, controls


def run(output, batch, preflight_record, cpu_validation_record, forensic_b0_only=False, max_batches=100, b0_stage1_only=False):
    assert not (forensic_b0_only and b0_stage1_only)
    gate_evidence = verify_gate_records(preflight_record, cpu_validation_record)
    execution_status = {
        "status": "RUNNING",
        "scope": (
            "B0_GRADIENT_FORENSIC_ONLY" if forensic_b0_only else
            "B0_STAGE1_ONE_EPOCH_ONLY" if b0_stage1_only else
            "SIX_CONDITION_ONE_EPOCH_SMOKE_ONLY"
        ),
        "batch_smoke": "NOT_RUN",
        "completed_conditions": [],
        "stage1_completed": [],
        "stage2_completed": [],
        "current_condition": None,
        "current_stage": None,
        "formal_training_started": False,
        "test_evaluations": 0,
        "additional_seeds_started": [],
    }
    write_json(output / "ACTUAL_EXECUTION_STATUS.json", execution_status)
    assert torch.cuda.is_available()
    assert models.sha(models.ORIGINAL_WEIGHT) == models.ORIGINAL_SHA
    original_sources = LOCKED_UPSTREAM_SOURCE_HASHES
    assert models.source_hashes() == original_sources
    assert models.sha(MANIFEST) == EXPECTED_SHA
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed()
    rows = load_rows()
    train, val, stage1, rec, donors = schedules(rows)
    counts = Counter(rows[int(i)]["target"] for i in train)
    names = {r["target"]: r["label"] for r in rows}
    ranked = sorted(range(16), key=lambda c: (-counts[c], names[c]))
    groups = {"Head": ranked[:6], "Medium": ranked[6:11], "Tail": ranked[11:]}
    group_by_class = {c: g for g, ids in groups.items() for c in ids}
    np.savez_compressed(output / "smoke_sampling_schedule.npz", stage1=stage1, recipients=rec, donors=donors)
    with (output / "donor_pair_audit.csv").open("x", newline="", encoding="utf-8") as f:
        fields = ["position", "recipient_index", "donor_index", "recipient_class", "donor_class", "donor_group", "same_image", "same_track", "same_class"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, (r, d) in enumerate(zip(rec, donors)):
            a, b = rows[int(r)], rows[int(d)]
            w.writerow(dict(zip(fields, [i, int(r), int(d), a["target"], b["target"], group_by_class[b["target"]],
                                        bool(r==d), a["base_group"]==b["base_group"], a["target"]==b["target"]])))
    save_json(output / "run_manifest.json", {"scope": "SMOKE_ONLY_NO_FORMAL_TRAINING_AUTHORIZATION", "seed": 42,
              "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=models.ROOT, text=True).strip(),
              "branch": BRANCH, "manifest_sha256": EXPECTED_SHA, "weight_sha256": models.ORIGINAL_SHA,
              "environment": {
                  "python_executable": sys.executable,
                  "python_version": platform.python_version(),
                  "torch": torch.__version__,
                  "cuda_build": torch.version.cuda,
                  "cudnn": torch.backends.cudnn.version(),
                  "gpu": torch.cuda.get_device_name(0),
              },
              "original_source_hashes": original_sources, "new_source_hashes": gate_evidence["source_sha256"],
              "gate_evidence": gate_evidence,
              "micro_batch": batch, "effective_batch": 16, "amp": False, "workers": 4,
              "numerical_mode": NUMERIC_MODE, "autocast_enabled": False, "grad_scaler_enabled": False,
              "stage1_epoch_budget_per_branch": 0 if forensic_b0_only else 1,
              "stage2_epoch_budget_per_condition": 0 if (forensic_b0_only or b0_stage1_only) else 1,
              "forensic_b0_only": forensic_b0_only,
              "forensic_max_batches": max_batches if forensic_b0_only else None,
              "b0_stage1_only": b0_stage1_only,
              "selected_conditions": ["B0"] if b0_stage1_only else None,
              "actual_execution_status": "See ACTUAL_EXECUTION_STATUS.json; budgets are not completion evidence",
              "schedule_denominators": [90, 10],
              "train_transform": str(transform(True)), "validation_transform": str(transform(False)),
              "numeric_config": {"precision": NUMERIC_MODE, "autocast_enabled": False, "grad_scaler_enabled": False,
                                 "cudnn_deterministic": True, "benchmark": False, "cudnn_tf32": False, "matmul_tf32": False},
              "test_model_evaluations": 0, "class_groups": groups, "note": "BN frozen for Stage-II; differs from original shift_bn=True"})
    if forensic_b0_only:
        execution_status["batch_smoke"] = "SKIPPED_FORENSIC_B0_ONLY"
        execution_status["current_condition"] = "B0"
        execution_status["current_stage"] = "stage1"
        write_json(output / "ACTUAL_EXECUTION_STATUS.json", execution_status)
        forensic_result, _ = one_epoch(
            "B0", models.FishModel(True), rows, stage1, None, "stage1", 1,
            batch, groups, val, output, max_batches=max_batches,
        )
        execution_status["status"] = "B0_FORENSIC_WINDOW_COMPLETED_NO_FAILURE"
        execution_status["current_condition"] = None
        execution_status["current_stage"] = None
        execution_status["forensic_result"] = forensic_result
        write_json(output / "ACTUAL_EXECUTION_STATUS.json", execution_status)
        return

    if b0_stage1_only:
        execution_status["batch_smoke"] = "SKIPPED_B0_STAGE1_ONLY"
        execution_status["current_condition"] = "B0"
        execution_status["current_stage"] = "stage1"
        write_json(output / "ACTUAL_EXECUTION_STATUS.json", execution_status)
        b0_result, _ = one_epoch(
            "B0", models.FishModel(True), rows, stage1, None, "stage1", 1,
            batch, groups, val, output,
        )
        execution_status["completed_conditions"].append("B0")
        execution_status["stage1_completed"].append("B0")
        execution_status["current_condition"] = None
        execution_status["current_stage"] = None
        execution_status["status"] = "COMPLETE_B0_STAGE1_ONE_EPOCH_ONLY"
        execution_status["b0_stage1_result"] = b0_result
        save_json(output / "B0_STAGE1_ONLY_SUMMARY.json", {
            "status": execution_status["status"],
            "completed_conditions": ["B0"],
            "stage1_completed": ["B0"],
            "stage2_completed": [],
            "formal_training_started": False,
            "test_evaluations": 0,
            "result": b0_result,
        })
        write_json(output / "ACTUAL_EXECUTION_STATUS.json", execution_status)
        return

    execution_status["batch_smoke"] = "RUNNING"
    write_json(output / "ACTUAL_EXECUTION_STATUS.json", execution_status)
    batch_smoke(rows, rec, donors, batch, output)

    results, controls = {}, {}
    for name, pif in (("A0", False), ("B0", True)):
        execution_status["current_condition"] = name
        execution_status["current_stage"] = "stage1"
        write_json(output / "ACTUAL_EXECUTION_STATUS.json", execution_status)
        results[name], controls[name] = one_epoch(name, models.FishModel(pif), rows, stage1, None, "stage1", 1, batch, groups, val, output)
        execution_status["completed_conditions"].append(name)
        execution_status["stage1_completed"].append(name)
        execution_status["current_condition"] = None
        execution_status["current_stage"] = None
        write_json(output / "ACTUAL_EXECUTION_STATUS.json", execution_status)
    for key in ("recipient_sha256", "mixup_rng_sha256", "mixup_lambda"):
        assert [r[key] for r in controls["A0"]] == [r[key] for r in controls["B0"]], key
    for name, source, mode in (("A1", "A0", "crt"), ("A2", "A0", "h2t"), ("B1", "B0", "crt"), ("B2", "B0", "pi")):
        execution_status["current_condition"] = name
        execution_status["current_stage"] = "stage2"
        write_json(output / "ACTUAL_EXECUTION_STATUS.json", execution_status)
        model = models.FishModel(source == "B0")
        state = torch.load(results[source]["checkpoint"], map_location="cpu", weights_only=True)
        assert state["smoke_only"] and state["seed"] == 42
        model.load_state_dict(state["model"], strict=True)
        results[name], controls[name] = one_epoch(name, model, rows, rec, donors if mode != "crt" else None, mode, 2, batch, groups, val, output)
        execution_status["completed_conditions"].append(name)
        execution_status["stage2_completed"].append(name)
        execution_status["current_condition"] = None
        execution_status["current_stage"] = None
        write_json(output / "ACTUAL_EXECUTION_STATUS.json", execution_status)
    assert all([r["recipient_sha256"] for r in controls[n]] == [r["recipient_sha256"] for r in controls["A1"]] for n in ("A2", "B1", "B2"))
    assert [r["donor_sha256"] for r in controls["A2"]] == [r["donor_sha256"] for r in controls["B2"]]
    assert results["A1"]["extractor_initial_sha256"] == results["A2"]["extractor_initial_sha256"]
    assert results["B1"]["extractor_initial_sha256"] == results["B2"]["extractor_initial_sha256"]
    assert results["A1"]["classifier_initial_sha256"] == results["A2"]["classifier_initial_sha256"]
    assert results["B1"]["classifier_initial_sha256"] == results["B2"]["classifier_initial_sha256"]
    assert models.source_hashes() == original_sources
    assert models.sha(MANIFEST) == EXPECTED_SHA
    save_json(output / "SMOKE_SUMMARY.json", {"status": "PASS_SMOKE_ONLY", "results": results,
              "all_recipient_views_exactly_paired": True, "fusion_donor_views_exactly_paired": True,
              "stage1_mixup_and_views_exactly_paired": True, "source_initializations_matched": True,
              "completed_conditions": execution_status["completed_conditions"],
              "formal_training_started": False, "test_evaluations": 0, "additional_seeds": []})
    execution_status["status"] = "COMPLETE_SMOKE_ONLY"
    execution_status["completed_conditions"] = list(execution_status["completed_conditions"])
    write_json(output / "ACTUAL_EXECUTION_STATUS.json", execution_status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--preflight-record", required=True)
    parser.add_argument("--cpu-validation-record", required=True)
    parser.add_argument("--micro-batch", type=int, choices=(16, 8), default=16)
    run_mode = parser.add_mutually_exclusive_group()
    run_mode.add_argument("--forensic-b0-only", action="store_true",
                          help="Run only a bounded B0 Stage-I gradient-forensics window; never the six-condition smoke.")
    run_mode.add_argument("--b0-stage1-only", action="store_true",
                          help="Run exactly one full B0 Stage-I epoch plus validation; skip A0, all Stage-II conditions, and TEST.")
    parser.add_argument("--max-batches", type=int, default=100,
                        help="Maximum B0 batches for --forensic-b0-only (1..100).")
    args = parser.parse_args()
    if args.forensic_b0_only or args.b0_stage1_only:
        assert args.micro_batch == 16, "B0-only modes are locked to micro-batch 16."
    if args.forensic_b0_only:
        assert 1 <= args.max_batches <= 100, "--max-batches must be between 1 and 100."
    output = Path(args.output_dir).resolve()
    assert not output.is_relative_to(models.ROOT)
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    try:
        run(output, args.micro_batch, args.preflight_record, args.cpu_validation_record,
            forensic_b0_only=args.forensic_b0_only,
            max_batches=args.max_batches, b0_stage1_only=args.b0_stage1_only)
    except Exception as exc:
        status_path = output / "ACTUAL_EXECUTION_STATUS.json"
        if status_path.is_file():
            execution_status = json.loads(status_path.read_text(encoding="utf-8"))
        else:
            execution_status = {
                "scope": "SIX_CONDITION_ONE_EPOCH_SMOKE_ONLY",
                "completed_conditions": [],
                "stage1_completed": [],
                "stage2_completed": [],
            }
        execution_status["status"] = "FAILED_STOPPED"
        execution_status["failure_type"] = type(exc).__name__
        execution_status["failure_message"] = str(exc)
        write_json(status_path, execution_status)
        save_json(output / "FAILURE.json", {"error_type": type(exc).__name__, "error": str(exc),
                  "traceback": traceback.format_exc(), "scope": "SMOKE_ONLY",
                  "elapsed_seconds": time.time()-started,
                  "completed_conditions": execution_status.get("completed_conditions", []),
                  "formal_training_started": False, "test_evaluations": 0})
        raise
    if args.forensic_b0_only:
        print(f"B0_FORENSIC_WINDOW_COMPLETE elapsed_seconds={time.time()-started:.2f}; NO_FULL_SMOKE_OR_FORMAL_TRAINING", flush=True)
    elif args.b0_stage1_only:
        print(f"B0_STAGE1_ONE_EPOCH_ONLY_COMPLETE elapsed_seconds={time.time()-started:.2f}; NO_STAGE2_OR_TEST", flush=True)
    else:
        print(f"SMOKE_COMPLETE_ONLY elapsed_seconds={time.time()-started:.2f}; FORMAL_TRAINING_NOT_STARTED", flush=True)
