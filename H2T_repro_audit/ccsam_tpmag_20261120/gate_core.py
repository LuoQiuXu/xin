from __future__ import annotations

import torch


GATE_EPS = 1e-12


def build_margin_reference(
    calibration_features: torch.Tensor,
    calibration_labels: torch.Tensor,
    initial_logits: torch.Tensor,
    train_counts,
    many_threshold: int = 100,
):
    """Build frozen class-conditional margin sensitivities from Calibration only.

    Tail margin:
        z_s(x) - logsumexp_{k in represented Many} z_k(x)

    Head margin:
        z_y(x) - z_q_y(x), where q_y is the strongest non-Many class by
        mean initial Calibration logit over examples of class y.

    The returned reference tensors are float64 on the input device.
    """
    if calibration_features.ndim != 2:
        raise ValueError("calibration_features must have shape [N,D]")
    n, dim = calibration_features.shape
    if n == 0 or dim == 0:
        raise ValueError("Calibration features must be non-empty")
    if calibration_labels.shape != (n,) or calibration_labels.dtype != torch.long:
        raise ValueError("calibration_labels must be torch.long with shape [N]")
    if initial_logits.ndim != 2 or initial_logits.shape[0] != n:
        raise ValueError("initial_logits must have shape [N,C]")
    num_classes = int(initial_logits.shape[1])
    if num_classes < 2:
        raise ValueError("At least two classifier classes are required")
    if train_counts.shape != (num_classes,):
        raise ValueError("train_counts must have shape [C]")
    if not (
        calibration_features.device
        == calibration_labels.device
        == initial_logits.device
        == train_counts.device
    ):
        raise ValueError("All reference inputs must share one device")
    if not torch.isfinite(calibration_features).all():
        raise ValueError("Non-finite Calibration features")
    if not torch.isfinite(initial_logits).all():
        raise ValueError("Non-finite initial Calibration logits")
    if bool(((calibration_labels < 0) | (calibration_labels >= num_classes)).any()):
        raise ValueError("Calibration label outside classifier range")
    if bool((train_counts <= 0).any()):
        raise ValueError("Original TRAIN class counts must be positive")

    x = calibration_features.detach().to(torch.float64)
    y = calibration_labels.detach()
    z = initial_logits.detach().to(torch.float64)
    counts = train_counts.detach().to(torch.long)
    represented = torch.unique(y, sorted=True)
    many_ids = represented[counts[represented] > int(many_threshold)]
    source_ids = represented[counts[represented] <= int(many_threshold)]
    if many_ids.numel() == 0 or source_ids.numel() == 0:
        raise ValueError("Calibration must represent both Many and non-Many classes")

    head_mean = torch.zeros((num_classes, dim), dtype=torch.float64, device=x.device)
    tail_weighted = torch.zeros(
        (num_classes, num_classes, dim), dtype=torch.float64, device=x.device
    )
    tail_probability = torch.zeros(
        (num_classes, num_classes), dtype=torch.float64, device=x.device
    )
    eligible = torch.zeros(
        (num_classes, num_classes), dtype=torch.bool, device=x.device
    )
    competitor_ids = torch.full(
        (num_classes,), -1, dtype=torch.long, device=x.device
    )

    for target in many_ids.tolist():
        target_mask = y == int(target)
        target_features = x[target_mask]
        target_logits = z[target_mask]
        if target_features.shape[0] == 0:
            raise RuntimeError("A represented Many target has no Calibration examples")
        head_mean[int(target)] = target_features.mean(dim=0)
        nonmany_logits = target_logits.index_select(1, source_ids)
        mean_nonmany_logits = nonmany_logits.mean(dim=0)
        competitor_ids[int(target)] = source_ids[mean_nonmany_logits.argmax()]

    for source in source_ids.tolist():
        source_mask = y == int(source)
        source_features = x[source_mask]
        source_logits = z[source_mask]
        if source_features.shape[0] == 0:
            raise RuntimeError("A represented non-Many source has no Calibration examples")
        many_logits = source_logits.index_select(1, many_ids)
        many_probabilities = torch.softmax(many_logits, dim=1)
        for local_index, target in enumerate(many_ids.tolist()):
            p = many_probabilities[:, local_index]
            tail_probability[int(source), int(target)] = p.mean()
            tail_weighted[int(source), int(target)] = (
                p.unsqueeze(1) * source_features
            ).mean(dim=0)
            eligible[int(source), int(target)] = True

    if int(eligible.sum().item()) != int(source_ids.numel() * many_ids.numel()):
        raise RuntimeError("Eligible source-target family is incomplete")
    if bool((competitor_ids[many_ids] < 0).any()):
        raise RuntimeError("A Many target lacks a frozen non-Many competitor")
    if bool((counts[competitor_ids[many_ids]] > int(many_threshold)).any()):
        raise RuntimeError("Frozen head competitor unexpectedly belongs to Many")

    return {
        "head_mean_features": head_mean,
        "tail_weighted_features": tail_weighted,
        "tail_probability_mean": tail_probability,
        "eligible": eligible,
        "source_ids": source_ids.detach().clone(),
        "target_ids": many_ids.detach().clone(),
        "competitor_ids": competitor_ids,
        "many_threshold": int(many_threshold),
        "tail_margin_definition": "z_source - logsumexp(z_represented_many)",
        "head_margin_definition": "z_target_many - z_frozen_nonmany_competitor",
        "reference_precision": "float64",
    }


def compute_tail_preserving_gate(
    source_grad_weight: torch.Tensor,
    source_grad_bias: torch.Tensor,
    reference: dict,
    eps: float = GATE_EPS,
):
    """Return a continuous [source,target] gate and first-order diagnostics.

    The direct pair gradient G is applied by SGD as -lr*G.
    Head benefit per unit lr is -(G_W dot mean_feature_y + G_b).
    Tail-margin shift per unit lr is G_W dot E[p_y^M(x)x|s] +
    G_b * E[p_y^M(x)|s].
    """
    if source_grad_weight.ndim != 3:
        raise ValueError("source_grad_weight must have shape [S,C,D]")
    num_sources, num_targets, dim = source_grad_weight.shape
    if source_grad_bias.shape != (num_sources, num_targets):
        raise ValueError("source_grad_bias must have shape [S,C]")
    if source_grad_weight.device != source_grad_bias.device:
        raise ValueError("Source gradient tensors must share a device")
    if not torch.isfinite(source_grad_weight).all() or not torch.isfinite(source_grad_bias).all():
        raise RuntimeError("Non-finite source-conditioned gradient")

    head_mean = reference["head_mean_features"]
    tail_weighted = reference["tail_weighted_features"]
    tail_probability = reference["tail_probability_mean"]
    eligible = reference["eligible"]
    expected_device = source_grad_weight.device
    if head_mean.shape != (num_targets, dim):
        raise ValueError("Head reference shape mismatch")
    if tail_weighted.shape != (num_sources, num_targets, dim):
        raise ValueError("Tail feature reference shape mismatch")
    if tail_probability.shape != (num_sources, num_targets):
        raise ValueError("Tail probability reference shape mismatch")
    if eligible.shape != (num_sources, num_targets) or eligible.dtype != torch.bool:
        raise ValueError("Eligible-pair mask shape/dtype mismatch")
    if any(t.device != expected_device for t in (head_mean, tail_weighted, tail_probability, eligible)):
        raise ValueError("Reference and gradient devices differ")
    if not float(eps) > 0:
        raise ValueError("eps must be positive")

    sw = source_grad_weight.detach().to(torch.float64)
    sb = source_grad_bias.detach().to(torch.float64)
    head = head_mean.to(torch.float64)
    tail_x = tail_weighted.to(torch.float64)
    tail_p = tail_probability.to(torch.float64)

    head_raw = -(
        torch.einsum("std,td->st", sw, head) + sb
    )
    tail_shift = (
        torch.einsum("std,std->st", sw, tail_x) + sb * tail_p
    )
    benefit = torch.clamp_min(head_raw, 0.0)
    risk = torch.clamp_min(-tail_shift, 0.0)

    gate64 = torch.ones(
        (num_sources, num_targets), dtype=torch.float64, device=expected_device
    )
    gate64[eligible] = benefit[eligible] / (
        benefit[eligible] + risk[eligible] + float(eps)
    )
    pair_activity = sw.abs().sum(dim=2) + sb.abs()
    active_pairs = eligible & (pair_activity > 0.0)
    # An exactly zero pair contributes nothing; use identity rather than
    # counting an absent contribution as a closed gate.
    gate64[eligible & ~active_pairs] = 1.0
    if not torch.isfinite(gate64).all():
        raise RuntimeError("Non-finite TP-MAG gate")
    if bool(((gate64 < 0) | (gate64 > 1)).any()):
        raise RuntimeError("TP-MAG gate escaped [0,1]")
    if not torch.equal(gate64[~eligible], torch.ones_like(gate64[~eligible])):
        raise RuntimeError("An ineligible source-target component was modified")

    diagnostics = {
        "head_margin_gain_raw": head_raw,
        "head_benefit": benefit,
        "tail_margin_shift": tail_shift,
        "tail_risk": risk,
        "active_pairs": active_pairs,
    }
    return gate64.to(dtype=source_grad_weight.dtype), diagnostics


def apply_tail_preserving_gate(
    clean_grad_weight: torch.Tensor,
    clean_grad_bias: torch.Tensor,
    source_grad_weight: torch.Tensor,
    source_grad_bias: torch.Tensor,
    gate: torch.Tensor,
):
    """Keep clean gradients and sum gated source-conditioned direct gradients."""
    if source_grad_weight.ndim != 3:
        raise ValueError("source_grad_weight must have shape [S,C,D]")
    sources, classes, dim = source_grad_weight.shape
    if source_grad_bias.shape != (sources, classes):
        raise ValueError("source_grad_bias shape mismatch")
    if gate.shape != (sources, classes):
        raise ValueError("gate shape mismatch")
    if gate.device != source_grad_weight.device:
        raise ValueError("Gate and source gradients must share a device")
    if clean_grad_weight.shape != (classes, dim) or clean_grad_bias.shape != (classes,):
        raise ValueError("Clean gradient shape mismatch")
    if not torch.isfinite(gate).all() or bool(((gate < 0) | (gate > 1)).any()):
        raise ValueError("Gate must be finite and lie in [0,1]")
    if not all(torch.isfinite(t).all() for t in (
        clean_grad_weight, clean_grad_bias, source_grad_weight, source_grad_bias
    )):
        raise RuntimeError("Non-finite gradient input")

    gated_weight = source_grad_weight * gate.to(source_grad_weight.dtype).unsqueeze(-1)
    gated_bias = source_grad_bias * gate.to(source_grad_bias.dtype)
    result_weight = clean_grad_weight + gated_weight.sum(dim=0)
    result_bias = clean_grad_bias + gated_bias.sum(dim=0)
    if not torch.isfinite(result_weight).all() or not torch.isfinite(result_bias).all():
        raise RuntimeError("Non-finite TP-MAG optimizer gradient")
    return result_weight, result_bias, gated_weight, gated_bias
