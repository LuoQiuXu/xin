"""One saved A0 batch, three AMP scales, no optimizer/scaler updates or images."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
from unittest.mock import patch

import torch
from torch import nn

import formal_train as formal
import models
from amp_recovery import OptimizerStepCounter, parameter_sha

SCALES = (65536, 131072, 262144)
SNAPSHOT = Path(r"D:\h2t_cross_domain_results\fish16_formal_seed42_90x10_20261102_01\FAILURE_SNAPSHOT.pt")
SNAPSHOT_SHA = "e07a48ec6a6cd95cc47b5c7875d6736e6c6f656fa65c05fb62c8505ac192400c"


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def number(value):
    value = float(value)
    return value if math.isfinite(value) else "NaN" if math.isnan(value) else "+Inf" if value > 0 else "-Inf"


def compare(actual, saved):
    actual, saved = actual.detach().float().cpu(), saved.detach().float().cpu()
    difference = (actual - saved).abs()
    return {"shape": list(actual.shape), "dtype": str(actual.dtype),
            "actual_finite": bool(torch.isfinite(actual).all()),
            "saved_finite": bool(torch.isfinite(saved).all()),
            "equal": bool(torch.equal(actual, saved)),
            "max_abs_error": number(difference.max()),
            "max_relative_error": number((difference / saved.abs().clamp_min(1e-12)).max()),
            "atol": 1e-6, "rtol": 1e-5,
            "allclose": bool(torch.allclose(actual, saved, atol=1e-6, rtol=1e-5))}


def group_of(name):
    if name.startswith("classifier."):
        return "classifier"
    if ".pif." in name:
        return "PIF"
    for stage in range(1, 5):
        if f".layer{stage}." in name:
            return f"backbone_layer{stage}"
    return "backbone_stem"


def worker(scale, directory):
    directory.mkdir(exist_ok=False)
    if models.sha(SNAPSHOT) != SNAPSHOT_SHA:
        raise ValueError("Saved failure snapshot SHA256 mismatch")
    snapshot = torch.load(SNAPSHOT, map_location="cpu", weights_only=True)
    context = snapshot["context"]
    if (context["condition"], context["epoch"], context["batch"]) != ("A0", 4, 976):
        raise ValueError("Not the authorized A0 failed batch")
    if len(snapshot["targets"]) != 16 or snapshot["mixed_input"].shape != (16, 3, 224, 224):
        raise ValueError("Saved full batch shape changed")
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    model = models.FishModel(use_pif=False)
    model.load_state_dict(snapshot["model"], strict=True)
    model.cuda().train()
    optimizer = torch.optim.SGD(model.parameters(), lr=.008, momentum=.9, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", init_scale=float(scale), growth_factor=2.,
                                 backoff_factor=.5, growth_interval=100000, enabled=True)
    if not scaler.is_enabled():
        raise RuntimeError("Native CUDA GradScaler is not enabled")
    counter = OptimizerStepCounter(optimizer)
    calls = {"optimizer_step_calls": 0, "scaler_step_calls": 0, "scaler_update_calls": 0}

    def forbidden(key):
        def fail(*args, **kwargs):
            calls[key] += 1
            raise RuntimeError("Replay forbids " + key)
        return fail

    named = list(model.named_parameters())
    before = parameter_sha(named)
    x = snapshot["mixed_input"].cuda()
    targets = snapshot["targets"].cuda()
    target_a, target_b = snapshot["target_a"].cuda(), snapshot["target_b"].cuda()
    lam = snapshot["mixup_lambda"]
    formal.restore_rng(snapshot["rng"])
    with patch.object(optimizer, "step", forbidden("optimizer_step_calls")), \
         patch.object(scaler, "step", forbidden("scaler_step_calls")), \
         patch.object(scaler, "update", forbidden("scaler_update_calls")):
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(x)
            loss = lam * nn.functional.cross_entropy(logits, target_a) + (1-lam) * nn.functional.cross_entropy(logits, target_b)
        # Preserve the exact original multiply/divide expression, even for a full batch.
        scaler.scale(loss * len(targets) / 16).backward()
        scaler.unscale_(optimizer)
    rows, missing, groups = [], [], {}
    for name, parameter in named:
        row = {"name": name, "group": group_of(name), "shape": list(parameter.shape),
               "missing": parameter.grad is None}
        if parameter.grad is None:
            missing.append(name)
        else:
            grad = parameter.grad.detach().cpu()
            finite = torch.isfinite(grad)
            good = grad[finite].double()
            n_nan, n_pos, n_neg = (int(check(grad).sum()) for check in (torch.isnan, torch.isposinf, torch.isneginf))
            row.update(dtype=str(grad.dtype), finite=bool(finite.all()), nan_count=n_nan,
                       positive_inf_count=n_pos, negative_inf_count=n_neg, inf_count=n_pos+n_neg,
                       nonfinite_positions=(~finite).nonzero().tolist(),
                       max_finite_abs=float(good.abs().max()) if good.numel() else None,
                       finite_l2=float(good.norm()) if good.numel() else None)
            group = groups.setdefault(row["group"], {"max_finite_abs": 0., "nan_count": 0, "inf_count": 0})
            group["max_finite_abs"] = max(group["max_finite_abs"], row["max_finite_abs"] or 0.)
            group["nan_count"] += n_nan
            group["inf_count"] += n_pos+n_neg
        rows.append(row)
    after = parameter_sha(named)
    nonfinite = [r for r in rows if not r.get("finite", False) and not r["missing"]]
    result = {"scale": scale, "snapshot_sha256": SNAPSHOT_SHA, "context": context,
              "input_shape": list(x.shape), "input_dtype": str(x.dtype), "mixup_lambda": lam,
              "saved_loss": float(snapshot["loss"]), "actual_loss": float(loss.detach()),
              "logits": compare(logits, snapshot["logits"]), "loss": compare(loss, snapshot["loss"]),
              "missing_gradients": missing, "nonfinite_gradients": nonfinite,
              "nan_count": sum(r.get("nan_count", 0) for r in rows),
              "inf_count": sum(r.get("inf_count", 0) for r in rows),
              "all_gradients_finite": not missing and not nonfinite,
              "classifier_bias_0_gradient": number(model.classifier.bias.grad[0]),
              "maximum_finite_gradient_by_group": groups,
              **calls, "actual_optimizer_post_hook_count": counter.count,
              "parameters_unchanged": before == after, "parameter_sha_before": before, "parameter_sha_after": after,
              "scale_after_unscale": float(scaler.get_scale()),
              "native_scaler_state": scaler.state_dict(),
              "numeric_flags": {"cudnn_deterministic": torch.backends.cudnn.deterministic,
                                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                                "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32},
              "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                              "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                              "gpu": torch.cuda.get_device_name(0)},
              "data_loaders": 0, "image_reads": 0, "test_accesses": 0}
    if any(calls.values()) or counter.count or before != after or models.sha(SNAPSHOT) != SNAPSHOT_SHA:
        raise RuntimeError("Replay preservation/no-update contract failed")
    write_json(directory/"RESULT.json", result)
    write_json(directory/"PARAMETER_GRADIENTS.json", rows)
    counter.close()
    print(json.dumps({k: result[k] for k in ("scale", "all_gradients_finite", "nan_count", "inf_count", "classifier_bias_0_gradient", "optimizer_step_calls")}, allow_nan=False), flush=True)


def run(directory):
    directory = directory.resolve()
    if directory.is_relative_to(models.ROOT):
        raise ValueError("Raw replay output must remain outside Git")
    directory.mkdir(parents=True, exist_ok=False)
    if models.sha(SNAPSHOT) != SNAPSHOT_SHA:
        raise ValueError("Snapshot missing, unreadable or SHA256 mismatch")
    results = []
    for scale in SCALES:
        command = [sys.executable, "-B", "-X", "utf8", "-u", str(Path(__file__).resolve()),
                   "--worker-scale", str(scale), "--output-dir", str(directory/f"scale_{scale}")]
        print("REPLAY_COMMAND " + subprocess.list2cmdline(command), flush=True)
        with (directory/f"scale_{scale}.log").open("x", encoding="utf-8") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        if completed.returncode:
            write_json(directory/"REPLAY_FAILURE.json", {"scale": scale, "exit_code": completed.returncode,
                       "formal_training_authorized": False, "automatic_retry": False})
            raise RuntimeError("Targeted replay failed; stop without retry")
        result = json.loads((directory/f"scale_{scale}"/"RESULT.json").read_text(encoding="utf-8"))
        results.append(result)
        print(json.dumps({k:result[k] for k in ("scale", "all_gradients_finite", "nan_count", "inf_count", "classifier_bias_0_gradient")}), flush=True)
    high = results[2]
    boundary = (results[0]["all_gradients_finite"] and not high["missing_gradients"] and
                high["nan_count"] == 0 and high["inf_count"] == 1 and
                len(high["nonfinite_gradients"]) == 1 and
                high["nonfinite_gradients"][0]["name"] == "classifier.bias" and
                high["nonfinite_gradients"][0]["nonfinite_positions"] == [[0]] and
                high["nonfinite_gradients"][0]["negative_inf_count"] == 1)
    forward_matches = all(r[k]["allclose"] and r[k]["actual_finite"] and r[k]["saved_finite"]
                          for r in results for k in ("logits", "loss"))
    gate = boundary and forward_matches
    write_json(directory/"REPLAY_SUMMARY.json", {"status": "PASS" if gate else "STOP_BOUNDARY_NOT_REPRODUCED",
               "snapshot": str(SNAPSHOT), "snapshot_sha256": SNAPSHOT_SHA,
               "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=models.ROOT, text=True).strip(),
               "replay_script_sha256": models.sha(__file__), "expected_boundary_reproduced": boundary,
               "saved_forward_matches": forward_matches, "formal_training_authorized": gate,
               "separate_process_per_scale": True, "scales": results,
               "optimizer_step_calls": 0, "scaler_step_update_calls": 0,
               "test_accesses": 0, "additional_seeds": []})
    with (directory/"SCALE_COMPARISON.csv").open("x", encoding="utf-8", newline="") as handle:
        fields = ["scale", "all_gradients_finite", "nan_count", "inf_count", "classifier_bias_0_gradient", "optimizer_step_calls"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k:r[k] for k in fields} for r in results)
    if not gate:
        raise RuntimeError("Expected numerical boundary or saved forward did not reproduce; no formal run")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--worker-scale", type=int, choices=SCALES)
    args = parser.parse_args()
    if args.worker_scale is None:
        run(args.output_dir)
    else:
        worker(args.worker_scale, args.output_dir)
