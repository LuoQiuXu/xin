"""Evidence helpers for the Stage-I gradient probe and Smoke."""
from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import torch


def trainable_parameters(encoder, classifier) -> dict[str, torch.nn.Parameter]:
    return {
        f"{part}.{name}": parameter
        for part, module in (("encoder", encoder), ("classifier", classifier))
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }


def _values(array: np.ndarray) -> dict:
    finite = np.isfinite(array)
    nan_count = int(np.count_nonzero(np.isnan(array)))
    inf_count = int(np.count_nonzero(np.isinf(array)))
    all_finite = nan_count == inf_count == 0
    finite_values = array if all_finite else array[finite]
    l2 = math.sqrt(float(np.square(finite_values, dtype=np.float64).sum()))
    max_abs = float(np.max(np.abs(finite_values), initial=0.0))
    invalid = "NaN" if nan_count else "Inf" if inf_count else None
    return {
        "all_finite": all_finite, "nan_count": nan_count, "inf_count": inf_count,
        "l2_norm": invalid if invalid else l2, "max_abs": invalid if invalid else max_abs,
        "finite_l2_norm": l2, "finite_max_abs": max_abs,
    }


def gradients(parameters: Mapping[str, torch.nn.Parameter], optimizer) -> tuple[dict, list[dict]]:
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    rows = []
    for name, parameter in parameters.items():
        grad = parameter.grad
        measures = (
            _values(grad.detach().cpu().numpy()) if grad is not None else
            dict(all_finite=None, nan_count=0, inf_count=0, l2_norm=None, max_abs=None,
                 finite_l2_norm=None, finite_max_abs=None)
        )
        rows.append({
            "name": name, "shape": list(parameter.shape), "parameter_dtype": str(parameter.dtype),
            "requires_grad": parameter.requires_grad, "in_optimizer": id(parameter) in optimizer_ids,
            "grad_is_none": grad is None, "gradient_dtype": str(grad.dtype) if grad is not None else None,
            **measures,
        })
    return {
        "trainable_tensors": len(rows),
        "finite_tensors": sum(row["all_finite"] is True for row in rows),
        "missing_names": [row["name"] for row in rows if row["grad_is_none"]],
        "nonfinite_names": [row["name"] for row in rows if row["all_finite"] is False],
        "not_in_optimizer_names": [row["name"] for row in rows if not row["in_optimizer"]],
        "nan_values": sum(row["nan_count"] for row in rows),
        "inf_values": sum(row["inf_count"] for row in rows),
    }, rows


def parameter_snapshot(parameters) -> dict[str, np.ndarray]:
    return {name: value.detach().cpu().numpy().copy() for name, value in parameters.items()}


def parameter_updates(parameters, previous: dict[str, np.ndarray]) -> tuple[dict, list[dict]]:
    rows = []
    for name, parameter in parameters.items():
        after = parameter.detach().cpu().numpy()
        before = previous[name]
        changed = int(np.count_nonzero(after != before))
        measure = _values(np.subtract(after, before, dtype=np.float64)) if changed else None
        rows.append({
            "name": name, "changed_elements": changed,
            "delta_l2_norm": measure["l2_norm"] if measure else 0.0,
            "delta_max_abs": measure["max_abs"] if measure else 0.0,
            "after_all_finite": bool(np.isfinite(after).all()) if changed else True,
        })
    finite = all(row["after_all_finite"] for row in rows)
    return {
        "changed_parameter_tensors": sum(row["changed_elements"] > 0 for row in rows),
        "changed_parameter_elements": sum(row["changed_elements"] for row in rows),
        "max_abs_delta": max((row["delta_max_abs"] for row in rows), default=0.0) if finite else None,
        "parameters_finite_after": finite,
    }, rows


def scaled_optimizer_step(scaler, optimizer) -> bool:
    """Observe the actual SGD step, then let GradScaler update its own scale."""
    stepped = False

    def after_step(*_):
        nonlocal stepped
        stepped = True

    handle = optimizer.register_step_post_hook(after_step)
    try:
        scaler.step(optimizer)
        scaler.update()
    finally:
        handle.remove()
    return stepped


def buffer_snapshot(*modules):
    return [(value, value.detach().clone()) for module in modules for value in module.buffers()]


@torch.no_grad()
def restore_buffers(snapshot) -> None:
    for value, original in snapshot:
        value.copy_(original)


def tensor_summary(tensor: torch.Tensor) -> dict:
    values = tensor.detach().cpu().numpy()
    finite = values[np.isfinite(values)]
    return {
        "shape": list(values.shape), "dtype": str(tensor.dtype), "finite": len(finite) == values.size,
        "min": float(finite.min()) if finite.size else None,
        "max": float(finite.max()) if finite.size else None,
    }


def scalar_json(tensor: torch.Tensor) -> float | str:
    value = float(tensor.detach())
    return value if math.isfinite(value) else str(value)
