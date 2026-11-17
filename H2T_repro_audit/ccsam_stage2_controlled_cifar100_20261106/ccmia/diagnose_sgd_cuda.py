"""Replay the six real Smoke batches and attribute paired SGD roundoff."""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F

import train_ccmia as audit


PARAMETERS = ("weight", "bias")
TERMS = ("direct", "state", "decay")
SGD_OPTIONS = {
    "lr": audit.STAGE2_LR,
    "momentum": audit.MOMENTUM,
    "weight_decay": audit.WEIGHT_DECAY,
    "nesterov": False,
}


def compare(actual, predicted, *, save_deltas=False):
    predicted = predicted.detach().to(torch.float64)
    actual = actual.detach().to(device=predicted.device, dtype=torch.float64)
    if not (torch.isfinite(actual).all() and torch.isfinite(predicted).all()):
        raise RuntimeError("Non-finite SGD diagnostic tensor")
    error = (actual - predicted).abs()
    index = tuple(int(i) for i in torch.unravel_index(error.argmax(), error.shape))
    max_abs = error[index].item()
    actual_scale = actual.abs().max().item()
    predicted_scale = predicted.abs().max().item()
    denominator = max(actual_scale, predicted_scale, 1e-8)
    result = {
        "max_absolute_error": max_abs,
        "actual_scale": actual_scale,
        "predicted_scale": predicted_scale,
        "relative_error": max_abs / denominator,
        "max_error_index": index,
        "actual_at_max_error": actual[index].item(),
        "predicted_at_max_error": predicted[index].item(),
    }
    if save_deltas:
        result["actual_delta"] = actual.tolist()
        result["predicted_delta"] = predicted.tolist()
    return result


def snapshot(parameters, optimizer):
    result = {}
    for name, parameter in parameters.items():
        buffer = optimizer.state.get(parameter, {}).get("momentum_buffer")
        if parameter.dtype != torch.float32 or parameter.grad is None or parameter.grad.dtype != torch.float32:
            raise RuntimeError(f"Smoke optimizer input must have FP32 parameters and gradients: {name}")
        if buffer is not None and buffer.dtype != torch.float32:
            raise RuntimeError(f"Smoke momentum buffer must be FP32: {name}")
        result[name] = {
            "parameter": parameter.detach().clone(),
            "grad": parameter.grad.detach().clone(),
            "buffer": None if buffer is None else buffer.detach().clone(),
        }
    return result


def snapshot_hash(snapshot_by_name):
    digests = {}
    for field in ("parameter", "grad", "buffer"):
        values = [snapshot_by_name[name][field] for name in PARAMETERS]
        digests[field] = (
            audit.hash_arrays(*(value.cpu().numpy() for value in values))
            if all(value is not None for value in values) else None
        )
    return digests


def cpu_replay_inputs(before, defaults):
    parameters = {
        name: torch.nn.Parameter(before[name]["parameter"].cpu().clone())
        for name in PARAMETERS
    }
    optimizer = torch.optim.SGD(parameters.values(), **defaults)
    for name, parameter in parameters.items():
        parameter.grad = before[name]["grad"].cpu().clone()
        if before[name]["buffer"] is not None:
            optimizer.state[parameter]["momentum_buffer"] = before[name]["buffer"].cpu().clone()
    if snapshot_hash(snapshot(parameters, optimizer)) != snapshot_hash(before):
        raise RuntimeError("The CPU optimizer replay received different input tensors")
    return parameters, optimizer


def profiled_steps(optimizers):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        for optimizer in optimizers:
            optimizer.step()
    operations = {
        event.key: event.count
        for event in profile.key_averages() if event.key.startswith("aten::")
    }
    fused = any("_fused_sgd" in name for name in operations)
    foreach = any(name.startswith("aten::_foreach_") for name in operations)
    if fused or (not foreach and "aten::add_" not in operations):
        raise RuntimeError(f"Unexpected SGD dispatch in the unchanged protocol: {operations}")
    return {"foreach": foreach, "fused": fused, "aten_operations": operations}


def native_sgd_reference(before, foreach):
    parameters = [before[name]["parameter"] for name in PARAMETERS]
    grads = [before[name]["grad"] for name in PARAMETERS]
    previous = [before[name]["buffer"] for name in PARAMETERS]
    if foreach:
        coupled = torch._foreach_add(grads, parameters, alpha=audit.WEIGHT_DECAY)
        if previous[0] is None and previous[1] is None:
            buffers = coupled
        elif all(value is not None for value in previous):
            buffers = [value.clone() for value in previous]
            torch._foreach_mul_(buffers, audit.MOMENTUM)
            torch._foreach_add_(buffers, coupled, alpha=1.0)
        else:
            raise RuntimeError("Inconsistent momentum initialization between weight and bias")
        updated = [value.clone() for value in parameters]
        torch._foreach_add_(updated, buffers, alpha=-audit.STAGE2_LR)
    else:
        coupled = [
            grad.add(parameter, alpha=audit.WEIGHT_DECAY)
            for grad, parameter in zip(grads, parameters)
        ]
        buffers = [
            current if prior is None else prior.mul(audit.MOMENTUM).add(current)
            for prior, current in zip(previous, coupled)
        ]
        updated = [
            parameter.add(buffer, alpha=-audit.STAGE2_LR)
            for parameter, buffer in zip(parameters, buffers)
        ]
    return {
        name: {"coupled_grad": g, "buffer": b, "parameter": p}
        for name, g, b, p in zip(PARAMETERS, coupled, buffers, updated)
    }


def delta(arms, name, field, *, promote=False):
    first = arms[0][name][field]
    second = arms[1][name][field]
    if promote:
        return second.double() - first.double()
    return second - first


def fp64_gradients(heads, features, labels, weights, pack):
    # The realized C3 perturbation is detached in production. Promote its exact values.
    x = features.double()
    x_adv = x + pack["delta3"].double()
    w = weights.double()
    alpha = pack["alpha"]
    params = [
        tuple(parameter.detach().double().requires_grad_() for parameter in head.parameters())
        for head in heads
    ]
    clean2, clean3 = [F.linear(x, *pair) for pair in params]
    adv3 = F.linear(x_adv, *params[1])
    ce2, ce3, adv_ce3 = [F.cross_entropy(z, labels, reduction="none") for z in (clean2, clean3, adv3)]
    loss2 = alpha * (ce2 * w[labels]).mean() + (1.0 - alpha) * ce2.mean()
    loss3 = alpha * (adv_ce3 * w[labels]).mean() + (1.0 - alpha) * ce3.mean()
    auto2 = torch.autograd.grad(loss2, params[0])
    auto3 = torch.autograd.grad(loss3, params[1])
    source = audit.decompose_perturbation_gradient_by_source(
        x, labels, pack["delta3"].double(), clean3.detach(), adv3.detach(), w, alpha
    )
    clean_grad2 = audit.clean_weighted_ce_gradient(x, labels, clean2.detach(), w, alpha)
    clean_grad3 = audit.clean_weighted_ce_gradient(x, labels, clean3.detach(), w, alpha)
    return {
        name: {
            "source": source[i],
            "direct": source[i].sum(0),
            "state": clean_grad3[i] - clean_grad2[i],
            "autograd_total": auto3[i] - auto2[i],
        }
        for i, name in enumerate(PARAMETERS)
    }


def summed(components, name):
    return components[f"direct_{name}"] + components[f"state_{name}"] + components[f"decay_{name}"]


def rounding_bound(parameter, momentum_buffer):
    # Bound a binary32 scalar conversion, multiply and add. An FMA uses one rounding.
    u = 2.0 ** -24
    half_min_subnormal = 2.0 ** -150
    lr32 = float(np.float32(audit.STAGE2_LR))
    scaled = lr32 * momentum_buffer.abs()
    product_bound = u * scaled + half_min_subnormal
    scalar_bound = abs(lr32 - audit.STAGE2_LR) * momentum_buffer.abs()
    add_bound = u * (parameter.abs() + scaled + product_bound) + half_min_subnormal
    return scalar_bound + product_bound + add_bound


def residual_budget(name, before, after, native, predicted_buffer, predicted_parameter, pack, history):
    actual_parameter = delta(after, name, "parameter")
    actual_buffer = delta(after, name, "buffer")
    exact_parameter_difference = delta(after, name, "parameter", promote=True)
    exact_buffer_difference = delta(after, name, "buffer", promote=True)
    buffer_residual = exact_buffer_difference - predicted_buffer
    parameter_subtraction = actual_parameter.double() - exact_parameter_difference
    buffer_subtraction = actual_buffer.double() - exact_buffer_difference
    current_difference = delta(before, name, "parameter", promote=True)
    current_difference_fp32 = delta(before, name, "parameter")
    grad_residual = (
        delta(before, name, "grad", promote=True)
        - pack[f"source_grad_{name}"].double().sum(0) - pack[f"state_grad_{name}"].double()
    )

    update_rounding = []
    native_buffer_rounding = []
    coupled_rounding = []
    bounds = []
    local_double_reference = []
    for start, end, replay in zip(before, after, native):
        p = start[name]["parameter"].double()
        grad = start[name]["grad"].double()
        previous = start[name]["buffer"]
        previous_term = torch.zeros_like(p) if previous is None else audit.MOMENTUM * previous.double()
        stored_buffer = end[name]["buffer"].double()
        stored_parameter = end[name]["parameter"].double()
        coupled32 = replay[name]["coupled_grad"].double()
        coupled64 = grad + audit.WEIGHT_DECAY * p
        ideal_buffer = previous_term + coupled64
        ideal_parameter = p - audit.STAGE2_LR * ideal_buffer
        coupled_rounding.append(coupled32 - coupled64)
        native_buffer_rounding.append(stored_buffer - (previous_term + coupled32))
        update_rounding.append(stored_parameter - (p - audit.STAGE2_LR * stored_buffer))
        bounds.append(rounding_bound(p, stored_buffer))
        local_double_reference.append({
            "coupled_gradient": compare(coupled32, coupled64),
            "momentum_buffer": compare(stored_buffer, ideal_buffer),
            "parameter_update": compare(stored_parameter - p, ideal_parameter - p),
        })

    previous_residual = history.get("buffer_residual", torch.zeros_like(buffer_residual))
    buffer_terms = {
        "previous_buffer_residual_times_momentum": audit.MOMENTUM * previous_residual,
        "autograd_minus_analytic_gradient": grad_residual,
        "paired_weight_decay_difference_rounding": (
            audit.WEIGHT_DECAY * (current_difference - current_difference_fp32.double())
        ),
        "coupled_gradient_rounding": coupled_rounding[1] - coupled_rounding[0],
        "momentum_multiply_add_rounding": native_buffer_rounding[1] - native_buffer_rounding[0],
        "buffer_difference_subtraction_rounding": buffer_subtraction,
    }
    history["buffer_residual"] = buffer_residual
    history["parameter_rounding"] = history.get("parameter_rounding", 0) + update_rounding[1] - update_rounding[0]
    history["buffer_effect"] = history.get("buffer_effect", 0) - audit.STAGE2_LR * buffer_residual
    history["rounding_bound"] = history.get("rounding_bound", 0) + bounds[0] + bounds[1]
    parameter_terms = {
        "accumulated_parameter_update_rounding": history["parameter_rounding"],
        "accumulated_buffer_residual_effect": history["buffer_effect"],
        "parameter_difference_subtraction_rounding": parameter_subtraction,
    }
    parameter_residual = actual_parameter.double() - predicted_parameter
    effective_buffer_residual = actual_buffer.double() - predicted_buffer

    def attribution(residual, terms):
        index = tuple(int(i) for i in torch.unravel_index(residual.abs().argmax(), residual.shape))
        return {
            "max_error_index": index,
            "residual_at_max_error": residual[index].item(),
            "terms_at_max_error": {term: value[index].item() for term, value in terms.items()},
            "term_maxima": {term: value.abs().max().item() for term, value in terms.items()},
            "closure_max_absolute_error": (residual - sum(terms.values())).abs().max().item(),
        }

    scale = max(actual_parameter.abs().max().item(), predicted_parameter.abs().max().item(), 1e-8)
    return {
        "parameter_residual": attribution(parameter_residual, parameter_terms),
        "buffer_residual": attribution(effective_buffer_residual, buffer_terms),
        "fp32_sgd_vs_fp64_step_on_fixed_gradients": dict(zip(("C2", "C3"), local_double_reference)),
        "roundoff_bound_check": {
            "max_per_arm_observed_over_step_bound": max(
                (observed.abs() / bound).max().item() for observed, bound in zip(update_rounding, bounds)
            ),
            "accumulated_bound_at_worst_index": history["rounding_bound"].flatten()[
                parameter_residual.abs().flatten().argmax()
            ].item(),
            "max_error_over_relative_plus_roundoff_bound": (
                parameter_residual.abs() / (
                    audit.MOMENTUM_RECONSTRUCTION_REL_TOL * scale
                    + history["rounding_bound"] + parameter_subtraction.abs()
                )
            ).max().item(),
            "relative_after_removing_measured_parameter_roundoff": (
                parameter_residual - history["parameter_rounding"] - parameter_subtraction
            ).abs().max().item() / scale,
        },
    }


def step_diagnostics(classifiers, optimizers, tracker, reference_tracker, history, gap, labels, weights, pack):
    heads = [classifier.module.fc for classifier in classifiers]
    parameters = [dict(head.named_parameters()) for head in heads]
    before = [snapshot(params, opt) for params, opt in zip(parameters, optimizers)]
    gradients64 = fp64_gradients(heads, gap, labels, weights, pack)
    tracker.advance(
        pack["source_grad_weight"], pack["source_grad_bias"], pack["state_grad_weight"], pack["state_grad_bias"],
        delta(before, "weight", "parameter"), delta(before, "bias", "parameter"),
    )
    reference_tracker.advance(
        gradients64["weight"]["source"], gradients64["bias"]["source"],
        gradients64["weight"]["state"], gradients64["bias"]["state"],
        delta(before, "weight", "parameter", promote=True), delta(before, "bias", "parameter", promote=True),
    )

    cpu_parameters, cpu_optimizers = zip(*(
        cpu_replay_inputs(state, opt.defaults) for state, opt in zip(before, optimizers)
    ))
    cpu_before = [snapshot(params, opt) for params, opt in zip(cpu_parameters, cpu_optimizers)]
    dispatch_cpu = profiled_steps(cpu_optimizers)
    dispatch_cuda = profiled_steps(optimizers)
    after_cpu = [snapshot(params, opt) for params, opt in zip(cpu_parameters, cpu_optimizers)]
    after = [snapshot(params, opt) for params, opt in zip(parameters, optimizers)]
    native_cuda = [native_sgd_reference(state, dispatch_cuda["foreach"]) for state in before]
    native_cpu = [native_sgd_reference(state, dispatch_cpu["foreach"]) for state in cpu_before]

    native_order = {}
    for device_name, replay, observed in (("cuda", native_cuda, after), ("cpu", native_cpu, after_cpu)):
        native_order[device_name] = {
            arm: {
                field: {
                    name: torch.equal(expected[name][field], actual[name][field])
                    for name in PARAMETERS
                }
                for field in ("buffer", "parameter")
            }
            for arm, expected, actual in zip(("C2", "C3"), replay, observed)
        }

    native_components = {"parameter": tracker.parameter_components(), "buffer": tracker.buffer_components()}
    reference_components = {
        "parameter": reference_tracker.parameter_components(), "buffer": reference_tracker.buffer_components(),
    }
    record = {
        "same_input_optimizer_sha256": dict(zip(("C2", "C3"), map(snapshot_hash, before))),
        "first_momentum_buffer_initialized": all(state[name]["buffer"] is None for state in before for name in PARAMETERS),
        "observed_optimizer_dispatch": {"cuda": dispatch_cuda, "cpu": dispatch_cpu},
        "native_sgd_update_order_bitwise": native_order,
        "tensors": {},
    }
    for name in PARAMETERS:
        actual_delta = {field: delta(after, name, field) for field in native_components}
        predicted_delta = {field: summed(components, name) for field, components in native_components.items()}
        gradients32 = {
            "source": pack[f"source_grad_{name}"],
            "direct": pack[f"direct_grad_{name}"],
            "state": pack[f"state_grad_{name}"],
            "autograd_total": pack[f"total_grad_{name}"],
        }
        record["tensors"][name] = {
            "reconstruction": {
                field: compare(actual_delta[field], predicted_delta[field], save_deltas=name == "bias")
                for field in native_components
            },
            "fp32_vs_fp64_gradient": {
                **{term: compare(value, gradients64[name][term]) for term, value in gradients32.items()},
                "decay": compare(
                    audit.WEIGHT_DECAY * delta(before, name, "parameter").double(),
                    audit.WEIGHT_DECAY * delta(before, name, "parameter", promote=True),
                ),
            },
            "fp64_direct_plus_state_vs_autograd": compare(
                gradients64[name]["direct"] + gradients64[name]["state"], gradients64[name]["autograd_total"]
            ),
            "fp32_vs_fp64_propagated_components": {
                field: {
                    term: compare(components[f"{term}_{name}"], reference_components[field][f"{term}_{name}"])
                    for term in TERMS
                }
                for field, components in native_components.items()
            },
            "cuda_vs_cpu_sgd": {
                field: {
                    **{arm: compare(cuda[name][field], cpu[name][field]) for arm, cuda, cpu in zip(
                        ("C2", "C3"), after, after_cpu
                    )},
                    "paired_delta": compare(actual_delta[field], delta(after_cpu, name, field)),
                }
                for field in native_components
            },
            "roundoff_budget": residual_budget(
                name, before, after, native_cuda, predicted_delta["buffer"], predicted_delta["parameter"],
                pack, history[name],
            ),
        }

    try:
        audit.reconstruction_diagnostics(tracker, *classifiers, *optimizers)
    except RuntimeError as exc:
        if not str(exc).startswith("SGD momentum/weight-decay attribution failed reconstruction:"):
            raise
        record["frozen_reconstruction_gate"] = {"passed": False, "message": str(exc)}
    else:
        record["frozen_reconstruction_gate"] = {"passed": True}
    return record


def run_seed(seed, context, device):
    audit.seed_everything(seed, deterministic=True)
    checkpoint = audit.resolve_checkpoint(context["manifest"]["checkpoint_path"])
    backbone, classifier3, _ = audit.build_models(checkpoint, device)
    classifier2 = copy.deepcopy(classifier3).to(device)
    initial_sha = audit.state_hash(classifier3)
    if audit.state_hash(classifier2) != initial_sha:
        raise RuntimeError("C2/C3 initial heads differ")
    frozen = {key: value.detach().cpu().clone() for key, value in backbone.state_dict().items()}
    classifiers = (classifier2, classifier3)
    optimizers = [torch.optim.SGD(model.module.fc.parameters(), **SGD_OPTIONS) for model in classifiers]
    weights = audit.class_weights_from_counts(context["counts"], device)
    plan = np.load(audit.BASE_ROOT / "prepared" / f"sampling_plan_seed{seed}.npy", mmap_mode="r", allow_pickle=False)
    reference = audit.read_csv_rows(
        audit.BASE_ROOT / "results" / f"seed_{seed}" / "C0_BALANCED_CE" / "batch_trace.csv"
    )
    generator = torch.Generator(device="cpu").manual_seed(seed + 300009)
    val_features = torch.from_numpy(context["validation_gap"].copy()).to(device)
    val_labels = torch.as_tensor(context["validation_labels"], dtype=torch.long, device=device)
    with torch.no_grad():
        initial_logits = classifier3.module.fc(val_features)
    fixed, mask, _ = audit.choose_fixed_competitors(
        initial_logits, context["validation_labels"], context["counts"], device
    )
    tracker, reference_tracker = [
        audit.SGDInfluenceTracker(
            audit.NUM_CLASSES, audit.FEATURE_DIM, device, audit.MOMENTUM, audit.STAGE2_LR, audit.WEIGHT_DECAY,
        )
        for _ in range(2)
    ]
    history = {name: {} for name in PARAMETERS}
    for batch in range(2):
        start = batch * audit.BATCH_SIZE
        indices = np.asarray(plan[0, start:start + audit.BATCH_SIZE], dtype=np.int64)
        target_np = np.asarray(context["train_labels"][indices], dtype=np.int64)
        labels = torch.from_numpy(target_np).to(device)
        images_cpu = audit.augment_batch(context["train_images"], indices, context["train_transform"], generator)
        with torch.no_grad():
            gap = audit.clean_compatible_gap(backbone(images_cpu.to(device, non_blocking=True)))
        trace = audit.make_trace_row(0, batch, indices, target_np, images_cpu, gap)
        audit.check_trace_row(trace, reference[batch], seed, 0, batch)
        pack = audit.compute_paired_gradients(*classifiers, gap, labels, weights, context["counts"], 0)
        source = audit.project_source_gradients_to_fixed_margin(
            pack["source_grad_weight"], pack["source_grad_bias"], val_features, val_labels,
            fixed, mask, audit.STAGE2_LR,
        )
        direct = audit.project_gradient_to_fixed_margin(
            pack["direct_grad_weight"], pack["direct_grad_bias"], val_features, val_labels,
            fixed, mask, audit.STAGE2_LR,
        )
        additivity_error = (source["margin_step"].sum(dim=0) - direct["margin_step"]).abs().max().item()
        if additivity_error > 5e-5:
            raise RuntimeError(f"Smoke source-margin projection is not additive: {additivity_error}")
        yield {
            "seed": seed,
            "batch": batch,
            "c0_trace_matches": True,
            "batch_trace": trace,
            "initial_classifier_sha256": initial_sha,
            "optimizer_defaults": optimizers[0].defaults,
            "gradient_checks": pack["checks"],
            "source_margin_additivity_max_abs": additivity_error,
            **step_diagnostics(
                classifiers, optimizers, tracker, reference_tracker, history, gap, labels, weights, pack
            ),
        }
    audit.check_backbone_unchanged(backbone, frozen)
    audit.require_test_guard()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.run_name):
        parser.error("--run-name must be a single directory name")
    output = audit.CCMIA_ROOT / "diagnostics" / args.run_name
    output.mkdir(parents=True, exist_ok=False)

    context = audit.prepare_context()
    inputs = audit.validate_inputs(context)
    preflight_path = audit.CCMIA_ROOT / "preflight_report.json"
    passed = audit.read_json(preflight_path)
    required = ("passed", "prepared_manifest_sha256", "code_sha256", "clean_stage1_sha256", "seeds")
    if any(passed.get(key) != inputs[key] for key in required):
        raise RuntimeError("Existing Preflight provenance differs from the diagnostic inputs")
    if not torch.cuda.is_available():
        raise RuntimeError("This diagnostic requires CUDA")
    device = torch.device("cuda:0")
    failed = []
    with (output / "batches.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
        for seed in audit.SEEDS:
            for record in run_seed(seed, context, device):
                handle.write(json.dumps(record, allow_nan=False) + "\n")
                handle.flush()
                gate = record["frozen_reconstruction_gate"]
                bias_rel = record["tensors"]["bias"]["reconstruction"]["parameter"]["relative_error"]
                print(f"seed={seed} batch={record['batch']} bias_rel={bias_rel:.9g} gate_passed={gate['passed']}")
                if not gate["passed"]:
                    failed.append({"seed": seed, "batch": record["batch"], "message": gate["message"]})
    audit.write_json(output / "provenance.json", {
        "diagnostic_only": True,
        "formal_training_started": False,
        "original_smoke_invoked": False,
        "python_version": sys.version.split()[0],
        "torch_version": str(torch.__version__),
        "cuda_build_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "cuda_capability": torch.cuda.get_device_capability(device),
        "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "preflight_sha256": audit.sha256_file(preflight_path),
        "diagnostic_source_sha256": audit.sha256_file(__file__),
        "audited_inputs": inputs,
        "reconstruction_relative_tolerance": audit.MOMENTUM_RECONSTRUCTION_REL_TOL,
        "batches_per_seed": 2,
        "frozen_backbone_and_bn_bitwise_unchanged": True,
        "frozen_gate_failures": failed,
        "test_guard": audit.require_test_guard(),
    })


if __name__ == "__main__":
    main()
