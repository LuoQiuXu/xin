from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def _check_linear_batch(features, labels, logits, num_classes):
    if features.ndim != 2 or labels.ndim != 1 or logits.ndim != 2:
        raise ValueError("Expected features [B,D], labels [B], logits [B,C].")
    batch, dim = features.shape
    if batch == 0 or labels.shape != (batch,) or logits.shape != (batch, num_classes):
        raise ValueError("Inconsistent or empty batch shapes.")
    if labels.dtype != torch.long:
        raise TypeError("labels must be torch.long.")
    if labels.device != features.device or logits.device != features.device:
        raise ValueError("All tensors must share one device.")
    if not torch.isfinite(features).all() or not torch.isfinite(logits).all():
        raise ValueError("Non-finite features or logits.")
    if bool(((labels < 0) | (labels >= num_classes)).any().item()):
        raise ValueError("Label outside classifier class range.")
    return batch, dim


def clean_weighted_ce_gradient(
    features: torch.Tensor,
    labels: torch.Tensor,
    logits_clean: torch.Tensor,
    class_weights: torch.Tensor,
    alpha: float,
):
    """Exact FP32 gradient of alpha*mean(w_y*CE_clean)+(1-alpha)*mean(CE_clean).

    This is the C2 gradient evaluated at an arbitrary current linear-head state.
    It is returned as (dW, db), with W shaped [C,D] and b shaped [C].
    """
    num_classes = int(logits_clean.shape[1])
    batch, _ = _check_linear_batch(features, labels, logits_clean, num_classes)
    if class_weights.shape != (num_classes,) or class_weights.device != features.device:
        raise ValueError("class_weights must have shape [C] on the batch device.")
    if class_weights.dtype != features.dtype:
        raise TypeError("class_weights and features must have the same dtype.")
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must be in [0,1].")

    one_hot = F.one_hot(labels, num_classes=num_classes).to(features.dtype)
    residual = logits_clean.detach().softmax(dim=1) - one_hot
    scale = (float(alpha) * class_weights[labels] + (1.0 - float(alpha))) / float(batch)
    grad_logits = residual * scale.unsqueeze(1)
    grad_weight = grad_logits.transpose(0, 1) @ features.detach()
    grad_bias = grad_logits.sum(dim=0)
    if not torch.isfinite(grad_weight).all() or not torch.isfinite(grad_bias).all():
        raise RuntimeError("Non-finite clean weighted-CE gradient.")
    return grad_weight.detach(), grad_bias.detach()


def decompose_perturbation_gradient_by_source(
    features: torch.Tensor,
    labels: torch.Tensor,
    delta: torch.Tensor,
    logits_clean: torch.Tensor,
    logits_adv: torch.Tensor,
    class_weights: torch.Tensor,
    alpha: float,
):
    """Decompose grad(L_C3-L_C2) at one identical classifier state by source class.

    C3 uses alpha*mean(w_y*CE(f+delta))+(1-alpha)*mean(CE(f)).
    C2 uses alpha*mean(w_y*CE(f))+(1-alpha)*mean(CE(f)).
    Therefore the direct perturbation gradient is exactly
      alpha/B * sum_i w[y_i] * (grad CE(f_i+delta_i)-grad CE(f_i)).
    The returned tensors have shapes [source_class,target_classifier_row,D]
    and [source_class,target_classifier_row]. The source axis is the ground-
    truth class of the perturbed input; the target axis is the classifier row.
    Delta is treated as detached, matching train_ccsam.compute_arm_loss.
    """
    num_classes = int(logits_clean.shape[1])
    batch, dim = _check_linear_batch(features, labels, logits_clean, num_classes)
    _check_linear_batch(features, labels, logits_adv, num_classes)
    if delta.shape != features.shape or delta.device != features.device:
        raise ValueError("delta must match features shape and device.")
    if class_weights.shape != (num_classes,) or class_weights.device != features.device:
        raise ValueError("class_weights must have shape [C] on the batch device.")
    if class_weights.dtype != features.dtype or delta.dtype != features.dtype:
        raise TypeError("features, delta and class_weights must share dtype.")
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must be in [0,1].")

    f = features.detach()
    d = delta.detach()
    p_clean = logits_clean.detach().softmax(dim=1)
    p_adv = logits_adv.detach().softmax(dim=1)
    one_hot = F.one_hot(labels, num_classes=num_classes).to(f.dtype)
    f_adv = f + d

    # Per-sample difference of the linear-classifier weight gradients.
    per_sample_weight = (
        (p_adv - one_hot).unsqueeze(2) * f_adv.unsqueeze(1)
        - (p_clean - one_hot).unsqueeze(2) * f.unsqueeze(1)
    )
    scale = (float(alpha) / float(batch)) * class_weights[labels]
    per_sample_weight = per_sample_weight * scale.view(batch, 1, 1)

    # The one-hot terms cancel for bias gradients because labels are unchanged.
    per_sample_bias = (p_adv - p_clean) * scale.unsqueeze(1)

    source_weight = torch.zeros(
        num_classes, num_classes, dim, dtype=f.dtype, device=f.device
    )
    source_bias = torch.zeros(
        num_classes, num_classes, dtype=f.dtype, device=f.device
    )
    source_weight.index_add_(0, labels, per_sample_weight)
    source_bias.index_add_(0, labels, per_sample_bias)

    if not torch.isfinite(source_weight).all() or not torch.isfinite(source_bias).all():
        raise RuntimeError("Non-finite source-conditioned gradient decomposition.")
    return source_weight.detach(), source_bias.detach()


def project_gradient_to_fixed_margin(
    grad_weight: torch.Tensor,
    grad_bias: torch.Tensor,
    validation_features: torch.Tensor,
    validation_labels: torch.Tensor,
    fixed_competitors: torch.Tensor,
    many_mask: torch.Tensor,
    learning_rate: float,
):
    """Project a gradient difference to one hypothetical SGD step on fixed margins.

    The fixed margin for validation item i is z[y_i]-z[q_i], where q_i is a
    non-Many competitor selected once from the initial classifier. Returned
    values are logit changes for the target row, competitor row and their
    difference. Negative margin_step means the gradient difference would
    shrink that fixed Many-vs-non-Many margin under an SGD step of learning_rate.
    """
    if grad_weight.ndim != 2 or grad_bias.shape != (grad_weight.shape[0],):
        raise ValueError("Expected grad_weight [C,D] and grad_bias [C].")
    if validation_features.ndim != 2:
        raise ValueError("validation_features must be [N,D].")
    n, dim = validation_features.shape
    if grad_weight.shape[1] != dim:
        raise ValueError("Feature dimension mismatch.")
    if validation_labels.shape != (n,) or fixed_competitors.shape != (n,) or many_mask.shape != (n,):
        raise ValueError("Validation labels, competitors and mask must have shape [N].")
    if validation_labels.dtype != torch.long or fixed_competitors.dtype != torch.long or many_mask.dtype != torch.bool:
        raise TypeError("Validation labels/competitors must be long and many_mask bool.")
    if not (validation_features.device == validation_labels.device == fixed_competitors.device == many_mask.device == grad_weight.device):
        raise ValueError("All margin projection tensors must share a device.")

    mask = many_mask
    y = validation_labels[mask]
    q = fixed_competitors[mask]
    x = validation_features[mask]
    if x.shape[0] == 0:
        raise ValueError("No Many validation examples are available for margin projection.")

    target_grad = (grad_weight[y] * x).sum(dim=1) + grad_bias[y]
    competitor_grad = (grad_weight[q] * x).sum(dim=1) + grad_bias[q]
    target_step = -float(learning_rate) * target_grad
    competitor_step = -float(learning_rate) * competitor_grad
    return {
        "target_logit_step": target_step.detach(),
        "competitor_logit_step": competitor_step.detach(),
        "margin_step": (target_step - competitor_step).detach(),
    }


def project_source_gradients_to_fixed_margin(
    source_grad_weight: torch.Tensor,
    source_grad_bias: torch.Tensor,
    validation_features: torch.Tensor,
    validation_labels: torch.Tensor,
    fixed_competitors: torch.Tensor,
    many_mask: torch.Tensor,
    learning_rate: float,
):
    """Source-class-resolved version of project_gradient_to_fixed_margin."""
    if source_grad_weight.ndim != 3:
        raise ValueError("source_grad_weight must be [S,C,D].")
    if source_grad_bias.shape != source_grad_weight.shape[:2]:
        raise ValueError("source_grad_bias must be [S,C].")
    mask = many_mask
    y = validation_labels[mask]
    q = fixed_competitors[mask]
    x = validation_features[mask]
    if x.shape[0] == 0:
        raise ValueError("No Many validation examples are available for source projection.")

    target_weight = source_grad_weight.index_select(1, y)
    competitor_weight = source_grad_weight.index_select(1, q)
    target_bias = source_grad_bias.index_select(1, y)
    competitor_bias = source_grad_bias.index_select(1, q)

    target_grad = torch.einsum("snd,nd->sn", target_weight, x) + target_bias
    competitor_grad = torch.einsum("snd,nd->sn", competitor_weight, x) + competitor_bias
    target_step = -float(learning_rate) * target_grad
    competitor_step = -float(learning_rate) * competitor_grad
    return {
        "target_logit_step": target_step.detach(),
        "competitor_logit_step": competitor_step.detach(),
        "margin_step": (target_step - competitor_step).detach(),
        "target_labels": y.detach(),
    }


def project_parameter_delta_to_fixed_margin(
    delta_weight: torch.Tensor,
    delta_bias: torch.Tensor,
    validation_features: torch.Tensor,
    validation_labels: torch.Tensor,
    fixed_competitors: torch.Tensor,
    many_mask: torch.Tensor,
):
    """Project an actual parameter delta (not a gradient) to fixed Many margins."""
    if delta_weight.ndim != 2 or delta_bias.shape != (delta_weight.shape[0],):
        raise ValueError("Expected parameter delta [C,D] and [C].")
    mask = many_mask
    y = validation_labels[mask]
    q = fixed_competitors[mask]
    x = validation_features[mask].to(dtype=delta_weight.dtype)
    delta_bias = delta_bias.to(dtype=delta_weight.dtype)
    target = (delta_weight[y] * x).sum(1) + delta_bias[y]
    competitor = (delta_weight[q] * x).sum(1) + delta_bias[q]
    return {
        "target_logit_delta": target.detach(),
        "competitor_logit_delta": competitor.detach(),
        "margin_delta": (target - competitor).detach(),
    }


def project_source_parameter_delta_to_fixed_margin(
    source_delta_weight: torch.Tensor,
    source_delta_bias: torch.Tensor,
    validation_features: torch.Tensor,
    validation_labels: torch.Tensor,
    fixed_competitors: torch.Tensor,
    many_mask: torch.Tensor,
):
    """Project accumulated source-specific parameter deltas to fixed Many margins."""
    mask = many_mask
    y = validation_labels[mask]
    q = fixed_competitors[mask]
    x = validation_features[mask].to(dtype=source_delta_weight.dtype)
    source_delta_bias = source_delta_bias.to(dtype=source_delta_weight.dtype)
    target_weight = source_delta_weight.index_select(1, y)
    competitor_weight = source_delta_weight.index_select(1, q)
    target_bias = source_delta_bias.index_select(1, y)
    competitor_bias = source_delta_bias.index_select(1, q)
    target = torch.einsum("snd,nd->sn", target_weight, x) + target_bias
    competitor = torch.einsum("snd,nd->sn", competitor_weight, x) + competitor_bias
    return {
        "target_logit_delta": target.detach(),
        "competitor_logit_delta": competitor.detach(),
        "margin_delta": (target - competitor).detach(),
        "target_labels": y.detach(),
    }


@dataclass
class SGDInfluenceTracker:
    """Exact additive decomposition of paired SGD momentum/weight-decay updates.

    At each step, with C3 state W3 and C2 state W2:
      g3(W3)-g2(W2) = [g3(W3)-g2(W3)] + [g2(W3)-g2(W2)].
    The first term is split by source class. The second is state/trajectory
    feedback. Coupled weight decay is tracked separately. All components are
    propagated through the same SGD momentum recurrence and learning rate.
    Float64 accumulators reduce long-run attribution drift; the actual model
    parameters and optimizer remain FP32.
    """

    num_classes: int
    feature_dim: int
    device: torch.device
    momentum: float
    learning_rate: float
    weight_decay: float

    def __post_init__(self):
        shape_source = (self.num_classes, self.num_classes, self.feature_dim)
        shape_row = (self.num_classes, self.feature_dim)
        shape_source_bias = (self.num_classes, self.num_classes)
        shape_bias = (self.num_classes,)
        kw = {"dtype": torch.float64, "device": self.device}
        self.direct_buffer_weight = torch.zeros(shape_source, **kw)
        self.direct_buffer_bias = torch.zeros(shape_source_bias, **kw)
        self.state_buffer_weight = torch.zeros(shape_row, **kw)
        self.state_buffer_bias = torch.zeros(shape_bias, **kw)
        self.decay_buffer_weight = torch.zeros(shape_row, **kw)
        self.decay_buffer_bias = torch.zeros(shape_bias, **kw)

        self.direct_parameter_weight = torch.zeros(shape_source, **kw)
        self.direct_parameter_bias = torch.zeros(shape_source_bias, **kw)
        self.state_parameter_weight = torch.zeros(shape_row, **kw)
        self.state_parameter_bias = torch.zeros(shape_bias, **kw)
        self.decay_parameter_weight = torch.zeros(shape_row, **kw)
        self.decay_parameter_bias = torch.zeros(shape_bias, **kw)

    @torch.no_grad()
    def advance(
        self,
        source_grad_weight: torch.Tensor,
        source_grad_bias: torch.Tensor,
        state_grad_weight: torch.Tensor,
        state_grad_bias: torch.Tensor,
        current_weight_difference: torch.Tensor,
        current_bias_difference: torch.Tensor,
    ):
        """Advance one paired optimizer step using pre-step parameter differences."""
        self.direct_buffer_weight.mul_(self.momentum).add_(source_grad_weight.to(torch.float64))
        self.direct_buffer_bias.mul_(self.momentum).add_(source_grad_bias.to(torch.float64))
        self.state_buffer_weight.mul_(self.momentum).add_(state_grad_weight.to(torch.float64))
        self.state_buffer_bias.mul_(self.momentum).add_(state_grad_bias.to(torch.float64))

        decay_grad_weight = float(self.weight_decay) * current_weight_difference.to(torch.float64)
        decay_grad_bias = float(self.weight_decay) * current_bias_difference.to(torch.float64)
        self.decay_buffer_weight.mul_(self.momentum).add_(decay_grad_weight)
        self.decay_buffer_bias.mul_(self.momentum).add_(decay_grad_bias)

        self.direct_parameter_weight.add_(self.direct_buffer_weight, alpha=-float(self.learning_rate))
        self.direct_parameter_bias.add_(self.direct_buffer_bias, alpha=-float(self.learning_rate))
        self.state_parameter_weight.add_(self.state_buffer_weight, alpha=-float(self.learning_rate))
        self.state_parameter_bias.add_(self.state_buffer_bias, alpha=-float(self.learning_rate))
        self.decay_parameter_weight.add_(self.decay_buffer_weight, alpha=-float(self.learning_rate))
        self.decay_parameter_bias.add_(self.decay_buffer_bias, alpha=-float(self.learning_rate))

    @torch.no_grad()
    def parameter_components(self):
        direct_weight = self.direct_parameter_weight.sum(dim=0)
        direct_bias = self.direct_parameter_bias.sum(dim=0)
        return {
            "direct_weight": direct_weight,
            "direct_bias": direct_bias,
            "state_weight": self.state_parameter_weight,
            "state_bias": self.state_parameter_bias,
            "decay_weight": self.decay_parameter_weight,
            "decay_bias": self.decay_parameter_bias,
        }

    @torch.no_grad()
    def buffer_components(self):
        direct_weight = self.direct_buffer_weight.sum(dim=0)
        direct_bias = self.direct_buffer_bias.sum(dim=0)
        return {
            "direct_weight": direct_weight,
            "direct_bias": direct_bias,
            "state_weight": self.state_buffer_weight,
            "state_bias": self.state_buffer_bias,
            "decay_weight": self.decay_buffer_weight,
            "decay_bias": self.decay_buffer_bias,
        }


def max_abs_and_relative_error(actual: torch.Tensor, predicted: torch.Tensor):
    diff = (actual.detach().to(torch.float64) - predicted.detach().to(torch.float64)).abs()
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    actual_scale = float(actual.detach().to(torch.float64).abs().max().item())
    predicted_scale = float(predicted.detach().to(torch.float64).abs().max().item())
    denom = max(actual_scale, predicted_scale, 1e-8)
    return max_abs, max_abs / denom


class ParameterWritebackRoundingLedger:
    """Track one optimizer's parameter writeback using its own step operands."""

    def __init__(self):
        self.cumulative = {}
        self.last_step = {}
        self.steps = 0
        self._pending = None

    @torch.no_grad()
    def capture(self, named_parameters, optimizer):
        from torch.optim.optimizer import _default_to_fused_or_foreach

        if self._pending is not None:
            raise RuntimeError("The previous writeback capture has not been recorded")
        if not isinstance(optimizer, torch.optim.SGD):
            raise TypeError("Writeback replay requires torch.optim.SGD")
        names = {id(parameter): name for name, parameter in named_parameters}
        groups = []
        for group in optimizer.param_groups:
            params = group["params"]
            if group["differentiable"] or group["fused"]:
                raise ValueError("Writeback replay supports nondifferentiable, unfused SGD")
            if group["foreach"] is None and group["fused"] is None:
                _, foreach = _default_to_fused_or_foreach(params, differentiable=False, use_fused=False)
            else:
                foreach = bool(group["foreach"])
            settings = {key: group[key] for key in (
                "lr", "momentum", "weight_decay", "dampening", "nesterov", "maximize"
            )}
            if not isinstance(settings["lr"], (float, int)):
                raise TypeError("Writeback replay requires a scalar learning rate")
            entries = []
            for parameter in params:
                if id(parameter) not in names:
                    raise ValueError("Every SGD parameter must have a ledger name")
                grad = parameter.grad
                if parameter.dtype not in (torch.float32, torch.float64):
                    raise TypeError("Writeback replay requires FP32 or FP64 parameters")
                if grad is None or grad.is_sparse or grad.dtype != parameter.dtype:
                    raise ValueError("Writeback replay requires dense gradients for every parameter")
                buffer = optimizer.state.get(parameter, {}).get("momentum_buffer")
                values = [parameter, grad] + ([] if buffer is None else [buffer])
                if not all(torch.isfinite(value).all() for value in values):
                    raise RuntimeError("Nonfinite input to parameter writeback replay")
                entries.append({
                    "name": names[id(parameter)],
                    "parameter": parameter,
                    "before": parameter.detach().clone(),
                    "grad": grad.detach().clone(),
                    "buffer": None if buffer is None else buffer.detach().clone(),
                })
            groups.append({"settings": settings, "foreach": foreach, "entries": entries, "group": group})
        if not groups or not any(group["entries"] for group in groups):
            raise ValueError("Writeback replay requires at least one parameter")
        if len(names) != sum(len(group["entries"]) for group in groups):
            raise ValueError("Ledger names and optimizer parameters must match exactly")
        self._pending = (optimizer, groups)

    @staticmethod
    def _directions(group, optimizer):
        settings = group["settings"]
        entries = group["entries"]
        before = [entry["before"] for entry in entries]
        grads = [entry["grad"].clone() for entry in entries]
        if settings["maximize"]:
            grads = [grad.neg() for grad in grads]
        if settings["weight_decay"]:
            if group["foreach"]:
                grads = torch._foreach_add(grads, before, alpha=settings["weight_decay"])
            else:
                grads = [grad.add(value, alpha=settings["weight_decay"]) for grad, value in zip(grads, before)]
        if not settings["momentum"]:
            return grads

        old_buffers = [entry["buffer"] for entry in entries]
        if group["foreach"] and all(buffer is not None for buffer in old_buffers):
            expected_buffers = [buffer.clone() for buffer in old_buffers]
            torch._foreach_mul_(expected_buffers, settings["momentum"])
            torch._foreach_add_(expected_buffers, grads, alpha=1.0 - settings["dampening"])
        else:
            expected_buffers = [
                grad.clone() if previous is None else previous.mul(settings["momentum"]).add(
                    grad, alpha=1.0 - settings["dampening"]
                )
                for grad, previous in zip(grads, old_buffers)
            ]
        actual_buffers = [optimizer.state[entry["parameter"]]["momentum_buffer"].detach() for entry in entries]
        for expected, actual in zip(expected_buffers, actual_buffers):
            if not torch.equal(expected.view(torch.uint8), actual.contiguous().view(torch.uint8)):
                raise RuntimeError("Independent SGD momentum-buffer replay differs from the real optimizer")
        if settings["nesterov"]:
            if group["foreach"]:
                torch._foreach_add_(grads, actual_buffers, alpha=settings["momentum"])
                return grads
            return [grad.add(buffer, alpha=settings["momentum"]) for grad, buffer in zip(grads, actual_buffers)]
        return actual_buffers

    @staticmethod
    def _rounding_bound(before, direction, learning_rate):
        """Bound scalar conversion, multiply and add, including an FMA path."""
        info = torch.finfo(before.dtype)
        unit_roundoff = info.eps / 2.0
        min_subnormal = info.tiny * info.eps
        native_lr = torch.tensor(learning_rate, dtype=before.dtype).item()
        p, u = before.double().abs(), direction.double().abs()
        product = abs(native_lr) * u
        product_error = unit_roundoff * product + min_subnormal
        coefficient_error = abs(float(learning_rate) - native_lr) * u
        add_error = unit_roundoff * (p + product + product_error) + min_subnormal
        return coefficient_error + product_error + add_error

    @torch.no_grad()
    def record(self):
        if self._pending is None:
            raise RuntimeError("Capture parameters before the real optimizer step")
        optimizer, groups = self._pending
        measured = {}
        for group in groups:
            settings = group["settings"]
            if any(group["group"][key] != value for key, value in settings.items()):
                raise RuntimeError("SGD settings changed during the captured step")
            direction = self._directions(group, optimizer)
            replay = [entry["before"].clone() for entry in group["entries"]]
            if group["foreach"]:
                torch._foreach_add_(replay, direction, alpha=-settings["lr"])
            else:
                for value, update in zip(replay, direction):
                    value.add_(update, alpha=-settings["lr"])
            for entry, replayed, update in zip(group["entries"], replay, direction):
                actual = entry["parameter"].detach()
                if not torch.isfinite(actual).all() or not torch.isfinite(update).all():
                    raise RuntimeError("Nonfinite SGD parameter or update direction")
                if not torch.equal(replayed.view(torch.uint8), actual.contiguous().view(torch.uint8)):
                    raise RuntimeError("Independent SGD parameter-writeback replay differs from the real optimizer")
                ideal = entry["before"].double() - float(settings["lr"]) * update.double()
                residual = actual.double() - ideal
                bound = self._rounding_bound(entry["before"], update, settings["lr"])
                if torch.any(residual.abs() > bound):
                    raise RuntimeError("Parameter writeback residual exceeds its operator-based rounding bound")
                measured[entry["name"]] = {
                    "residual": residual,
                    "bound": bound,
                    "before": entry["before"],
                    "after": actual.clone(),
                    "update_direction": update.clone(),
                    "learning_rate": float(settings["lr"]),
                    "foreach": group["foreach"],
                    "nesterov": settings["nesterov"],
                    "momentum_buffer_initialized": bool(settings["momentum"] and entry["buffer"] is None),
                    "replay_bitwise_equal": True,
                }
        for name, values in measured.items():
            previous = self.cumulative.get(name, torch.zeros_like(values["residual"]))
            self.cumulative[name] = previous + values["residual"]
        self.last_step = measured
        self.steps += 1
        self._pending = None
        return measured
