"""Bounded B0-only replay and disposable same-batch numerical counterfactuals.

Imports the frozen smoke operations, preserving its finite-gradient gate.
All data, tensors and diagnostic outputs remain outside Git.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
import numpy as np
import torch
from torch import nn
import data
import models
import smoke
from preflight import EXPECTED_SHA, MANIFEST, BRANCH

OLD = Path(r"D:\h2t_cross_domain_results\fish16_smoke_seed42_20261031_01")


def save_json(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def save_tensor(path, value):
    with Path(path).open("xb") as f:
        torch.save(value, f)


def tensor_stats(value):
    if value is None:
        return {"is_none": True, "dtype": None, "shape": None, "numel": 0, "finite": None,
                "nan_count": 0, "inf_count": 0, "finite_count": 0, "min": None, "max": None,
                "max_abs": None, "l2_norm": None, "finite_max_abs": None, "finite_l2_norm": None}
    x = value.detach().cpu()
    y = x.to(torch.float64).reshape(-1)
    finite = torch.isfinite(y)
    good = y[finite]
    nan, inf = int(torch.isnan(y).sum()), int(torch.isinf(y).sum())
    finite_max = float(good.abs().max()) if good.numel() else None
    finite_l2 = float(torch.linalg.vector_norm(good)) if good.numel() else 0.
    return {"is_none": False, "dtype": str(x.dtype), "shape": list(x.shape), "numel": x.numel(),
            "finite": bool(finite.all()), "nan_count": nan, "inf_count": inf,
            "finite_count": int(finite.sum()), "min": float(good.min()) if good.numel() else None,
            "max": float(good.max()) if good.numel() else None,
            "max_abs": "NaN" if nan else "Inf" if inf else finite_max,
            "l2_norm": "NaN" if nan else "Inf" if inf else finite_l2,
            "finite_max_abs": finite_max, "finite_l2_norm": finite_l2}


def parameter_group(name):
    if name.startswith("extractor.pif."):
        return "PIF"
    if name.startswith("classifier."):
        return "classifier"
    if name.startswith("extractor.backbone.layer"):
        return "backbone." + name.split(".")[2]
    return "backbone.stem"


def gradient_records(model, scenario, batch, phase):
    rows = []
    for name, parameter in model.named_parameters():
        rows.append({"scenario": scenario, "batch": batch, "phase": phase, "parameter": name,
                     "group": parameter_group(name), "parameter_dtype": str(parameter.dtype),
                     **tensor_stats(parameter.grad)})
    return rows


def gradient_groups(rows):
    result = {}
    for group in sorted({r["group"] for r in rows}):
        chosen = [r for r in rows if r["group"] == group]
        bad = [r["parameter"] for r in chosen if r["finite"] is False]
        missing = [r["parameter"] for r in chosen if r["is_none"]]
        result[group] = {"parameter_tensors": len(chosen), "missing_parameters": missing,
                         "nonfinite_parameters": bad, "nan_count": sum(r["nan_count"] for r in chosen),
                         "inf_count": sum(r["inf_count"] for r in chosen),
                         "all_present_and_finite": not bad and not missing,
                         "finite_max_abs": max((r["finite_max_abs"] or 0 for r in chosen), default=0),
                         "finite_l2_norm": math.sqrt(sum((r["finite_l2_norm"] or 0)**2 for r in chosen))}
    return result


def write_gradients(folder, rows):
    with (folder / "parameter_gradients.csv").open("x", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return copy.deepcopy(value)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state_all()}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])


class Probes:
    """Read-only hooks return None; they never replace activations or gradients."""
    def __init__(self, model):
        self.outputs, self.activation_grads, self.handles, self.convolution = {}, {}, [], {}
        modules = {"backbone.layer" + str(i): getattr(model.extractor.backbone, "layer" + str(i)) for i in range(1, 5)}
        modules.update({"PIF": model.extractor.pif, "PIF.Conv1d": model.extractor.pif.fuse,
                        "classifier": model.classifier})
        for name, module in modules.items():
            self.handles.append(module.register_forward_hook(self.make_hook(name)))
        self.handles.append(model.classifier.register_forward_pre_hook(self.pool_hook))

    def pool_hook(self, module, inputs):
        self.outputs["pooled_features"] = inputs[0].detach()

    def make_hook(self, name):
        def hook(module, inputs, output):
            self.outputs[name] = output.detach()
            if name == "PIF":
                self.outputs["PIF_input"] = inputs[0].detach()
            if name in {"PIF", "PIF.Conv1d", "classifier"} and output.requires_grad:
                self.handles.append(output.register_hook(self.grad_hook(name + ".output_scaled_grad")))
            if name == "PIF.Conv1d":
                self.convolution["module_weight_dtype"] = str(module.weight.dtype)
                self.convolution["input_dtype"] = str(inputs[0].dtype)
                queue, visited = [(output.grad_fn, 0)], set()
                while queue:
                    node, depth = queue.pop(0)
                    if node is None or node in visited or depth > 4:
                        continue
                    visited.add(node)
                    weight = getattr(node, "_saved_weight", None)
                    if isinstance(weight, torch.Tensor) and weight.numel() == 2:
                        self.convolution.update(saved_weight_dtype=str(weight.dtype), saved_weight_shape=list(weight.shape),
                                                backward_node=type(node).__name__)
                        if weight.requires_grad:
                            self.handles.append(weight.register_hook(self.grad_hook("PIF.Conv1d.cast_weight_scaled_grad")))
                        break
                    queue.extend((n, depth+1) for n, _ in getattr(node, "next_functions", []))
        return hook

    def grad_hook(self, name):
        def hook(gradient):
            self.activation_grads[name] = gradient.detach().cpu().clone()
        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()


def forward_backward(model, optimizer, scaler, x, targets_a, targets_b, lam, folder, tag, batch):
    probes = Probes(model)
    scale_before = scaler.get_scale() if scaler else None
    pif_before = model.extractor.pif.fuse.weight.detach().cpu().reshape(-1).tolist()
    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=scaler is not None):
        logits = model(x)
        loss = lam * nn.functional.cross_entropy(logits, targets_a) + (1-lam) * nn.functional.cross_entropy(logits, targets_b)
    forward = {name: tensor_stats(tensor) for name, tensor in probes.outputs.items()}
    forward.update(input=tensor_stats(x), loss=tensor_stats(loss))
    save_json(folder / "forward.json", {"ranges": forward, "pif_parameters": pif_before, "convolution": probes.convolution,
              "input_sha256": smoke.digest_tensor(x), "feature_sha256": smoke.digest_tensor(probes.outputs["PIF"]),
              "logit_sha256": smoke.digest_tensor(logits), "loss_sha256": smoke.digest_tensor(loss)})
    assert torch.isfinite(loss) and torch.isfinite(logits).all(), "Nonfinite forward before gradient gate"
    # Same objective and order as original smoke: micro/effective=16, final partial batch handled identically.
    objective = loss * len(targets_a) / len(targets_a)
    (scaler.scale(objective) if scaler else objective).backward()
    before_rows = gradient_records(model, tag, batch, "before_unscale" if scaler else "fp32_unscaled")
    if scaler:
        scaler.unscale_(optimizer)
    scale_after = scaler.get_scale() if scaler else None
    after_rows = gradient_records(model, tag, batch, "after_unscale" if scaler else "fp32_unscaled_final")
    write_gradients(folder, before_rows + after_rows)
    groups = gradient_groups(after_rows)
    save_json(folder / "gradient_groups.json", {"before_unscale": gradient_groups(before_rows), "after_unscale": groups})
    save_json(folder / "activation_gradients.json", {k: tensor_stats(v) for k, v in probes.activation_grads.items()})
    pif_gradient = model.extractor.pif.fuse.weight.grad
    save_tensor(folder / "pif_gradient_tensors.pt", {"parameter_gradient_after_unscale": cpu_tree(pif_gradient),
                "cast_weight_scaled_gradient": probes.activation_grads.get("PIF.Conv1d.cast_weight_scaled_grad")})
    passed, gate_error = True, None
    try:
        smoke.finite_gradients(list(model.parameters()))  # The immutable actual smoke gate.
        assert model.classifier.weight.grad is not None
    except AssertionError as exc:
        passed, gate_error = False, str(exc)
    summary = {"scenario": tag, "batch": batch, "amp": scaler is not None, "scale_before_unscale": scale_before,
               "scale_after_unscale": scale_after, "strict_gate_passed": passed, "gate_error": gate_error,
               "loss": float(loss.detach()), "pif_parameters_before": pif_before,
               "pif_gradient_after_unscale": pif_gradient.detach().cpu().flatten().tolist() if pif_gradient is not None and torch.isfinite(pif_gradient).all() else None,
               "nonfinite_parameters": [r["parameter"] for r in after_rows if r["finite"] is False],
               "missing_gradient_parameters": [r["parameter"] for r in after_rows if r["is_none"]],
               "gradient_groups": groups, "convolution": probes.convolution,
               "pif_cast_gradient_stats": tensor_stats(probes.activation_grads.get("PIF.Conv1d.cast_weight_scaled_grad"))}
    save_json(folder / "backward.json", summary)
    probes.close()
    return summary


def fresh_optimizer(model, state=None):
    optimizer = torch.optim.SGD(model.parameters(), lr=.002, momentum=.9, weight_decay=1e-4)
    if state is not None:
        optimizer.load_state_dict(copy.deepcopy(state))
    return optimizer


def replay_until_failure(rows, order, output):
    old_controls = json.loads((OLD / "A0/input_controls.json").read_text(encoding="utf-8"))
    model = models.FishModel(True)
    smoke.seed()  # Same reset position as one_epoch: model is constructed first.
    model.cuda().train()
    optimizer = fresh_optimizer(model)
    scaler = torch.amp.GradScaler("cuda")
    optimizer.zero_grad(set_to_none=True)
    batches = data.loader(data.Views(rows, order, True), 16)
    replay = output / "reproduction"
    replay.mkdir()
    with (output / "batch_log.csv").open("x", newline="", encoding="utf-8") as log:
        fields = ["batch", "samples", "loss", "scale_before_unscale", "scale_after_unscale", "strict_gate_passed", "nonfinite_parameters", "missing_gradient_parameters", "optimizer_step_executed"]
        writer = csv.DictWriter(log, fieldnames=fields)
        writer.writeheader()
        for i, item in enumerate(batches):
            if i >= 100:
                break  # Original failure preceded its first 100-batch progress message.
            folder = replay / f"batch_{i:05d}"
            folder.mkdir()
            snapshot = {"model": cpu_tree(model.state_dict()), "optimizer": cpu_tree(optimizer.state_dict()),
                        "scaler": scaler.state_dict(), "batch": i}
            indices = item["index"].tolist()
            metadata = [{"row_index": j, "image_id": rows[j]["image_id"], "track_id": rows[j]["base_group"],
                         "group_id": rows[j]["group_id"], "label": rows[j]["target"], "class_name": rows[j]["label"],
                         "image_sha256": rows[j]["image_sha256"]} for j in indices]
            for j in indices:
                assert rows[j]["split_primary"] == "train"
                assert models.sha(data.safe_image_path(rows[j])) == rows[j]["image_sha256"]
            target = item["label"].cuda(non_blocking=True)
            before_rng = rng_state()
            expected_lam = float(np.random.beta(1., 1.))
            permutation = torch.randperm(len(target))
            restore_rng(before_rng)
            raw = item["recipient"].cuda(non_blocking=True)
            mixed, a, b, lam = models.methods.mixup_data(raw, target, alpha=1.)
            assert lam == expected_lam and torch.equal(b.cpu(), target.cpu()[permutation])
            snapshot.update(mixed=cpu_tree(mixed), raw_input=cpu_tree(item["recipient"]), targets_a=cpu_tree(a), targets_b=cpu_tree(b),
                            lam=lam, permutation=permutation, samples=metadata, rng_before_mixup=before_rng, rng_before_forward=rng_state())
            trace = {"batch": i, "samples": metadata, "mixup_lambda": lam, "mixup_permutation": permutation.tolist(),
                     "labels_a": a.cpu().tolist(), "labels_b": b.cpu().tolist(),
                     "raw_input_sha256": smoke.digest_tensor(item["recipient"]),
                     "rng_before_mixup_sha256": smoke.digest_tensor(before_rng["torch_cpu"]),
                     "raw_input_range": tensor_stats(item["recipient"]), "mixed_input_range": tensor_stats(mixed)}
            assert trace["raw_input_sha256"] == old_controls[i]["recipient_sha256"], "Historical image-view mismatch"
            assert trace["rng_before_mixup_sha256"] == old_controls[i]["mixup_rng_sha256"], "Historical MixUp RNG mismatch"
            assert trace["mixup_lambda"] == old_controls[i]["mixup_lambda"], "Historical MixUp lambda mismatch"
            trace["historical_A0_same_stream_pixel_rng_lambda_matches"] = True
            save_json(folder / "batch_trace.json", trace)
            summary = forward_backward(model, optimizer, scaler, mixed, a, b, lam, folder, "original_amp_reproduction", i)
            stepped = False
            if summary["strict_gate_passed"]:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                stepped = True
            writer.writerow({**{k: summary[k] for k in fields if k in summary}, "samples": len(target),
                             "optimizer_step_executed": stepped})
            log.flush()
            print(f"B0 diagnostic batch={i} scale={summary['scale_before_unscale']} gate={summary['strict_gate_passed']} nonfinite={summary['nonfinite_parameters']} missing={summary['missing_gradient_parameters']}", flush=True)
            if not stepped:
                save_tensor(output / "exact_failed_batch_snapshot.pt", snapshot)
                save_json(output / "FAILING_BATCH.json", {**trace, **summary, "actual_reproduction_optimizer_step_skipped_by_strict_gate": True,
                          "successful_optimizer_steps_before_failure": i})
                model.cpu()
                del model, optimizer, scaler
                torch.cuda.empty_cache()
                return snapshot, summary
    model.cpu()
    return None, None


def diagnostic_case(snapshot, output, tag, scale, normal_step=False):
    folder = output / tag
    folder.mkdir()
    model = models.FishModel(True)
    model.load_state_dict(snapshot["model"], strict=True)
    model.cuda().train()
    optimizer = fresh_optimizer(model, snapshot["optimizer"])
    optimizer.zero_grad(set_to_none=True)
    scaler = torch.amp.GradScaler("cuda", init_scale=float(scale)) if scale is not None else None
    if scaler:
        scaler.load_state_dict({**snapshot["scaler"], "scale": float(scale)})
    restore_rng(snapshot["rng_before_forward"])
    summary = forward_backward(model, optimizer, scaler, snapshot["mixed"].cuda(), snapshot["targets_a"].cuda(),
                               snapshot["targets_b"].cuda(), snapshot["lam"], folder, tag, snapshot["batch"])
    summary["scope"] = "DISPOSABLE_IDENTICAL_PRE_FAILURE_CLONE_ONLY"
    if normal_step:
        assert scaler is not None
        before_params = {k: p.detach().cpu().clone() for k, p in model.named_parameters()}
        before_buffers = {k: b.detach().cpu().clone() for k, b in model.named_buffers()}
        step_calls = []
        handle = optimizer.register_step_post_hook(lambda *args, **kwargs: step_calls.append(True))
        scaler.step(optimizer)  # Allowed only here, on a disposable diagnostic clone.
        scaler.update()
        handle.remove()
        summary["normal_scaler_behavior"] = {"optimizer_step_calls": len(step_calls), "scale_after_update": scaler.get_scale(),
            "parameters_unchanged": all(torch.equal(p.detach().cpu(), before_params[k]) for k, p in model.named_parameters()),
            "post_forward_bn_buffers_unchanged_by_step": all(torch.equal(b.detach().cpu(), before_buffers[k]) for k, b in model.named_buffers()),
            "bn_note": "Training forward updates BN naturally; comparison is before vs after scaler.step/update, not against pre-forward BN"}
    finite = {k: p.grad.detach().cpu().clone() for k, p in model.named_parameters() if p.grad is not None and torch.isfinite(p.grad).all()}
    save_tensor(folder / "finite_unscaled_gradients.pt", finite)
    save_json(folder / "case_summary.json", summary)
    model.cpu()
    del model, optimizer, scaler
    torch.cuda.empty_cache()
    print(f"CASE {tag} gate={summary['strict_gate_passed']} nonfinite={summary['nonfinite_parameters']}", flush=True)
    return summary


def run(output):
    old_manifest = json.loads((OLD / "run_manifest.json").read_text(encoding="utf-8"))
    for name, digest in old_manifest["new_source_hashes"].items():
        assert models.sha(BASE / name) == digest, name
    assert models.source_hashes() == old_manifest["original_source_hashes"]
    assert models.sha(MANIFEST) == EXPECTED_SHA
    assert models.sha(models.ORIGINAL_WEIGHT) == models.ORIGINAL_SHA
    assert torch.cuda.is_available()
    assert subprocess.check_output(["git", "branch", "--show-current"], cwd=models.ROOT, text=True).strip() == BRANCH
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    rows = data.load_rows()
    with np.load(OLD / "smoke_sampling_schedule.npz") as values:
        order = values["stage1"].copy()
    assert np.array_equal(order, data.schedules(rows)[2])
    save_json(output / "RUN_MANIFEST.json", {"branch": BRANCH,
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=models.ROOT, text=True).strip(),
        "original_smoke_source_hashes": old_manifest["new_source_hashes"], "upstream_source_hashes": models.source_hashes(),
        "diagnostic_source_sha256": models.sha(Path(__file__)), "manifest_sha256": EXPECTED_SHA,
        "original_weight_sha256": models.ORIGINAL_SHA, "schedule_sha256": models.sha(OLD / "smoke_sampling_schedule.npz"),
        "seed": 42, "batch_size": 16, "effective_batch": 16, "workers": 4, "lr": .002,
        "momentum": .9, "weight_decay": 1e-4, "mixup_alpha": 1., "max_reproduction_batches": 100,
        "bound_reason": "Original failure occurred before the first 100-batch progress print; do not expand training if not reproduced there",
        "numeric_config": old_manifest["numeric_config"], "environment": {"python": sys.executable,
            "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0)}, "formal_training": False, "test_evaluation": False,
        "counterfactuals_are_diagnostic_only": True, "original_smoke_gate_modified": False})
    snapshot, failure = replay_until_failure(rows, order, output)
    if snapshot is None:
        save_json(output / "DIAGNOSIS_RESULTS.json", {"status": "NOT_REPRODUCED_WITHIN_ORIGINAL_FAILURE_BOUND", "root_cause": "UNRESOLVED"})
        return
    cases = []
    for tag, scale in [("amp_original_scale", failure["scale_before_unscale"]), ("amp_32768", 32768),
                       ("amp_16384", 16384), ("amp_8192", 8192), ("amp_4096", 4096), ("fp32", None)]:
        cases.append(diagnostic_case(snapshot, output, tag, scale))
    normal = diagnostic_case(snapshot, output, "normal_scaler_disposable", failure["scale_before_unscale"], normal_step=True)
    save_json(output / "DIAGNOSIS_RESULTS.json", {"status": "NUMERICAL_DIAGNOSTICS_COMPLETED",
              "failed_batch": snapshot["batch"], "reproduction": failure, "same_batch_cases": cases,
              "normal_scaler_disposable": normal, "formal_training_started": False,
              "test_evaluations": 0, "extra_seeds": [], "stage2_epochs": 0})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    assert not output.is_relative_to(models.ROOT)
    output.mkdir(parents=True, exist_ok=True)
    assert not (output / "RUN_MANIFEST.json").exists(), "Refuse to restart or overwrite a diagnostic run"
    start = time.perf_counter()
    try:
        run(output)
    except Exception:
        save_json(output / "DIAGNOSTIC_FAILURE.json", {"traceback": traceback.format_exc(), "formal_training": False})
        raise
    print(f"B0_DIAGNOSTICS_STOPPED elapsed={time.perf_counter()-start:.3f}s; NO_FORMAL_TRAINING", flush=True)
