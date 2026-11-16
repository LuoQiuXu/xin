"""Rebuild the original ImageNet-LT Smoke batch for paired FP32 and AMP backward."""
from __future__ import annotations

import argparse
import copy
import re
import sys
from contextlib import nullcontext
from itertools import chain
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

EXPERIMENT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXPERIMENT))

from cross_dataset_models import build_stage1_models, state_dict_sha256
from experiment_common import (
    array_sha256, cuda_device, load_experiment, make_loader, make_sgd, record_run, run_log,
    seed_everything, sha256_file, write_json,
)
from prepare_experiment import PREFLIGHT, load_prepared, train_dataset
from run_cross_dataset import require_cuda_check
from train_stage1 import epoch_lr, train_transform
from gradient_evidence import (
    buffer_snapshot, gradients, parameter_snapshot, parameter_updates, restore_buffers,
    scalar_json, scaled_optimizer_step, tensor_summary, trainable_parameters,
)

MAX_RETRIES = 8


class WithRowIndex(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, target = self.dataset[index]
        return image, target, index


def make_model(initial, device):
    return tuple(copy.deepcopy(model).to(device).train() for model in initial)


def forward(encoder, classifier, mixed, labels, partner, weight, *, amp: bool):
    context = torch.autocast("cuda", dtype=torch.float16) if amp else nullcontext()
    with context:
        features = encoder(mixed)
        logits = classifier(features)
        loss = weight * F.cross_entropy(logits, labels) + (1.0 - weight) * F.cross_entropy(logits, labels[partner])
    return loss, {
        "features": tensor_summary(features), "logits": tensor_summary(logits),
        "loss": scalar_json(loss), "loss_finite": bool(torch.isfinite(loss)),
    }


def run_fp32(output, initial, device, settings, inputs):
    encoder, classifier = make_model(initial, device)
    starting = [state_dict_sha256(model) for model in (encoder, classifier)]
    optimizer = make_sgd(chain(encoder.parameters(), classifier.parameters()), settings)
    parameters = trainable_parameters(encoder, classifier)
    snapshot = parameter_snapshot(parameters)
    optimizer.zero_grad(set_to_none=True)
    loss, forward_info = forward(encoder, classifier, *inputs, amp=False)
    loss.backward()
    summary, rows = gradients(parameters, optimizer)
    update, update_rows = parameter_updates(parameters, snapshot)
    result = {
        "initial_state_sha256": starting, "forward": forward_info,
        "scaled_loss": None, "scale_before": None, "scale_after": None,
        "gradient_state": summary, "gradients": rows, "optimizer_step": False,
        "update": update, "parameter_deltas": update_rows,
    }
    write_json(output / "fp32.json", result)
    return result


def run_amp(output, initial, device, settings, inputs, fp32, *, allow_retry, max_retries):
    encoder, classifier = make_model(initial, device)
    starting = [state_dict_sha256(model) for model in (encoder, classifier)]
    if starting != fp32["initial_state_sha256"]:
        raise RuntimeError("FP32 and AMP models did not start from the same state_dict")
    optimizer = make_sgd(chain(encoder.parameters(), classifier.parameters()), settings)
    scaler = torch.amp.GradScaler("cuda")
    parameters = trainable_parameters(encoder, classifier)
    snapshot = parameter_snapshot(parameters)
    initial_buffers = buffer_snapshot(encoder, classifier)
    attempts = []
    reproduced_inf = False
    for number in range(max_retries + 1 if allow_retry else 1):
        if number:
            restore_buffers(initial_buffers)
        optimizer.zero_grad(set_to_none=True)
        loss, forward_info = forward(encoder, classifier, *inputs, amp=True)
        scale_before = scaler.get_scale()
        scaled = scaler.scale(loss)
        scaled_loss = scalar_json(scaled)
        scaled.backward()
        scaled_state, scaled_rows = gradients(parameters, optimizer)
        scaler.unscale_(optimizer)
        state, rows = gradients(parameters, optimizer)
        if number == 0:
            reproduced_inf = scaled_state["inf_values"] > 0 or state["inf_values"] > 0
        forward_finite = (forward_info["features"]["finite"] and forward_info["logits"]["finite"]
                          and forward_info["loss_finite"])
        missing = bool(state["missing_names"] or state["not_in_optimizer_names"])
        can_observe_step = (allow_retry and forward_finite and not missing
                            and (not state["nonfinite_names"] or reproduced_inf))
        steps_before = len(optimizer.state)
        stepped = scaled_optimizer_step(scaler, optimizer) if can_observe_step else False
        scale_after = scaler.get_scale()
        update, update_rows = parameter_updates(parameters, snapshot)
        entry = {
            "attempt": number, "initial_state_sha256": starting, "forward": forward_info,
            "scaled_loss": scaled_loss, "scale_before": scale_before, "scale_after": scale_after,
            "gradscaler_update_called": can_observe_step, "optimizer_step": stepped,
            "optimizer_state_entries_before": steps_before, "optimizer_state_entries_after": len(optimizer.state),
            "scaled_gradient_state": scaled_state, "unscaled_gradient_state": state,
            "scaled_gradients": scaled_rows, "unscaled_gradients": rows,
            "update": update, "parameter_deltas": update_rows,
        }
        write_json(output / f"amp_attempt_{number:02}.json", entry)
        attempts.append({key: value for key, value in entry.items()
                         if key not in ("scaled_gradients", "unscaled_gradients", "parameter_deltas")})
        run_log(output, f"AMP attempt {number}, scale {scale_before:g} -> {scale_after:g}, "
                        f"missing={len(state['missing_names'])}, NaN={state['nan_values']}, "
                        f"Inf={state['inf_values']}, optimizer_step={stepped}")
        if not can_observe_step:
            break
        if not state["nonfinite_names"]:
            if stepped and update["changed_parameter_tensors"] > 0 and update["parameters_finite_after"]:
                return {"reproduced_inf": reproduced_inf, "recovered": number > 0, "attempts": attempts,
                        "successful_update": True}
            break
        if stepped or update["changed_parameter_tensors"] or not scale_after < scale_before:
            break
    return {"reproduced_inf": reproduced_inf, "recovered": False, "attempts": attempts, "successful_update": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--max-retries", type=int, choices=range(MAX_RETRIES + 1), default=MAX_RETRIES)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", args.output_name):
        parser.error("--output-name must be a simple lowercase directory name")
    output = Path(__file__).resolve().parent / args.output_name
    exp = load_experiment(None)
    prepared = load_prepared(exp)
    device = cuda_device()
    require_cuda_check(exp, device)
    dataset_name = "imagenet_lt"
    seed = exp.seeds[0]
    stage1 = exp.protocol["stage1"]
    settings = {**stage1, "learning_rate": epoch_lr(1, stage1["epochs"], stage1["learning_rate"])}
    seal = prepared["datasets"][dataset_name]
    config = exp.config("amp-gradient-probe", dataset=dataset_name, seed=seed,
                        probe_source_sha256=sha256_file(Path(__file__)), max_retries=args.max_retries)
    with record_run(output, config, device=device) as status:
        seed_everything(seed)
        with train_dataset(exp, dataset_name, train_transform(stage1["input_size"]), seal) as (dataset, targets):
            initial = build_stage1_models(exp.datasets[dataset_name]["num_classes"])
            rng = np.random.default_rng(seed)
            generator = torch.Generator(device=device).manual_seed(seed)
            loader = make_loader(WithRowIndex(dataset), stage1["batch_size"], exp.workers, seed, shuffle=True)
            iterator = iter(loader)
            images, labels, indices = next(iterator)
            del iterator, loader
            if not np.array_equal(targets[indices.numpy()], labels.numpy()):
                raise RuntimeError("Rebuilt batch labels differ from the locked TRAIN dataset")
        if len(labels) != stage1["batch_size"] or not bool(torch.isfinite(images).all()):
            raise RuntimeError("The original Stage-I batch is malformed or has nonfinite inputs")
        x, y = images.to(device), labels.to(device)
        weight = float(rng.beta(stage1["mixup_alpha"], stage1["mixup_alpha"]))
        partner = torch.randperm(len(y), generator=generator, device=device)
        mixed = weight * x + (1.0 - weight) * x[partner]
        provenance = {
            "dataset": dataset_name, "seed": seed, "batch_one_based": 1, "workers": exp.workers,
            "prepared_manifest_sha256": sha256_file(PREFLIGHT / "prepared_manifest.json"),
            "initial_state_sha256": [state_dict_sha256(model) for model in initial],
            "indices": indices.tolist(), "indices_sha256": array_sha256(indices.numpy()),
            "labels": labels.tolist(), "labels_sha256": array_sha256(labels.numpy()),
            "images_sha256": array_sha256(images.numpy()), "images": tensor_summary(images),
            "mixed_images_sha256": array_sha256(mixed.cpu().numpy()), "mixed_images": tensor_summary(mixed),
            "mixup_lambda": weight, "mixup_permutation": partner.cpu().tolist(),
            "learning_rate": settings["learning_rate"], "official_test_access": False,
        }
        write_json(output / "batch_identity.json", provenance)
        inputs = mixed, y, partner, weight
        fp32 = run_fp32(output, initial, device, settings, inputs)
        if fp32["initial_state_sha256"] != provenance["initial_state_sha256"]:
            raise RuntimeError("The diagnostic FP32 copy differs from the initial model")
        summary = fp32["gradient_state"]
        forward_info = fp32["forward"]
        fp32_finite = (
            not (summary["missing_names"] or summary["nonfinite_names"] or summary["not_in_optimizer_names"])
            and forward_info["loss_finite"] and forward_info["features"]["finite"] and forward_info["logits"]["finite"]
        )
        # Always record the first AMP backward. A failed FP32 comparison cannot authorize AMP retry.
        amp = run_amp(output, initial, device, settings, inputs, fp32, allow_retry=fp32_finite,
                      max_retries=args.max_retries)
        write_json(output / "summary.json", {
            "fp32_finite": fp32_finite, "fp32_gradient_state": summary, **amp,
            "paired_initial_state_sha256": provenance["initial_state_sha256"],
            "max_retries": args.max_retries, "official_test_access": False,
        })
        status.update(fp32_finite=fp32_finite, amp_recovered=amp["recovered"], retries=len(amp["attempts"]) - 1)
        if not (fp32_finite and amp["reproduced_inf"] and amp["recovered"] and amp["successful_update"]):
            raise RuntimeError("The paired probe did not establish recoverable AMP overflow. Review the saved gradient records.")


if __name__ == "__main__":
    main()
