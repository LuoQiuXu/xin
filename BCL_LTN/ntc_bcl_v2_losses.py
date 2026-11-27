"""Loss modules, EMA teacher, and online anchored EM estimator for NTC-BCL v2.

Transition convention:
    T[c, j] = P(observed label = j | clean class = c).

Key v2 design properties:
1. No known noise rate, corruption mask, or symmetry assumption.
2. Visual-confidence anchored M-step (depends only on predicted visual
   evidence, never conditioning sample selection on observed labels) prevents
   the multi-step circular T -> I collapse on in-sample noisy datasets.
3. Regularized EM over the clean prior pi given anchored T (with Laplace
   pseudocount >= 1.0 by default) prevents multiplicative tail-prior starvation.
4. Soft-BCL v2 decouples classifier direction from magnitude when feeding
   prototype_mlp, bounds tail denominator weights via support_floor_ratio and
   denominator_epsilon, and supports smooth semantic_ramp annealing from 0 to 1.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from BCL_LTN.noise_transition_estimator import (
    _global_template,
    _normalize_rows,
    _validate_inputs,
    prototype_health_diagnostics,
)


@dataclass
class AnchoredTransitionEstimate:
    """Outputs of one online anchored-T + regularized-prior EM update."""

    transition: torch.Tensor
    clean_prior: torch.Tensor
    posterior_clean: torch.Tensor
    anchor_support: torch.Tensor
    effective_support: torch.Tensor
    global_diagonal_retention: torch.Tensor
    global_support_class_ids: torch.Tensor
    estimated_noise_rate: float
    iterations: int
    converged: bool
    final_delta: float


@torch.no_grad()
def estimate_anchored_transition_and_prior(
    teacher_probabilities: torch.Tensor,
    observed_labels: torch.Tensor,
    *,
    confidence_power: float = 2.0,
    shrinkage_mass: float = 10.0,
    high_support_fraction: float = 0.35,
    min_high_support_classes: int = 5,
    prior_pseudocount: float = 1.0,
    max_iter: int = 200,
    tolerance: float = 1e-6,
    initial_prior: Optional[torch.Tensor] = None,
    reference_prior: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> AnchoredTransitionEstimate:
    """Estimate T and clean prior pi from balanced EMA teacher probabilities.

    Args:
        teacher_probabilities: [N, K] balanced visual probabilities
            p_bal(c | x_i) = softmax(ema_logits).
        observed_labels: [N] noisy/observed label IDs.
        confidence_power: Exponent applied to visual-only max-probability
            reliability when forming anchor weights for T. Setting > 0 focuses T
            estimation on visually confident predictions without conditioning on
            observed labels.
        shrinkage_mass: Empirical-Bayes prior mass pulling low-support rows of T
            toward the high-support global transition template.
        prior_pseudocount: Per-class Dirichlet Laplace pseudocount added during
            the pi M-step so weak tail priors cannot collapse multiplicatively.
        reference_prior: Previous blended clean prior used only to form the
            PiW visual-side posterior q_i ∝ p_bal(c|x_i) * pi_ref,c. If omitted,
            a uniform prior is used for backward-compatible unweighted anchoring.
    """
    probs, observed = _validate_inputs(
        teacher_probabilities.detach().to(torch.float64), observed_labels
    )
    scalar_values = (
        confidence_power,
        shrinkage_mass,
        high_support_fraction,
        prior_pseudocount,
        tolerance,
        eps,
    )
    if not all(math.isfinite(float(v)) for v in scalar_values):
        raise ValueError("Estimator hyperparameters must be finite")
    if confidence_power < 0 or shrinkage_mass < 0 or prior_pseudocount < 0:
        raise ValueError(
            "confidence_power, shrinkage_mass and prior_pseudocount must be non-negative"
        )
    if not 0.0 < high_support_fraction <= 1.0:
        raise ValueError("high_support_fraction must be in (0, 1]")
    if min_high_support_classes < 1 or max_iter < 1 or tolerance <= 0 or eps <= 0:
        raise ValueError(
            "min_high_support_classes, max_iter, tolerance and eps must be positive"
        )

    n, k = probs.shape
    dtype, device = probs.dtype, probs.device

    # PiW-EstT: calibrate balanced visual evidence by the previous blended prior.
    # The first online update receives observed_counts / N from the trainer;
    # subsequent updates receive the previous blended clean prior.
    if reference_prior is None:
        ref_pi = torch.full((k,), 1.0 / k, dtype=dtype, device=device)
    else:
        ref_pi = torch.as_tensor(reference_prior, dtype=dtype, device=device).detach().clone()
        if (
            ref_pi.shape != (k,)
            or not bool(torch.isfinite(ref_pi).all())
            or bool((ref_pi <= 0).any())
            or bool(ref_pi.sum() <= 0)
        ):
            raise ValueError("reference_prior must be a finite positive [K] vector")
        ref_pi = ref_pi / ref_pi.sum()
    q_joint = probs * ref_pi.unsqueeze(0)
    q_mass = q_joint.sum(dim=1, keepdim=True)
    if bool((q_mass <= eps).any()):
        raise RuntimeError("PiW reference-prior posterior has zero row mass")
    q_anchor = q_joint / q_mass

    # Step 1: Visual-confidence anchored M-step for row-stochastic transition T.
    max_prob = probs.max(dim=1).values
    visual_reliability = ((max_prob - 1.0 / k) / (1.0 - 1.0 / k)).clamp_min(0.0)
    if float(confidence_power) > 0.0:
        sample_weights = visual_reliability.pow(float(confidence_power))
        if float(sample_weights.sum().item()) <= eps:
            sample_weights = torch.ones(n, dtype=dtype, device=device)
    else:
        sample_weights = torch.ones(n, dtype=dtype, device=device)

    # Raw soft support S_c = sum_i w_i q_ic. Do not replace this with Kish ESS:
    # diffuse low-probability leakage would spuriously inflate tail support.
    anchor_weights = q_anchor * sample_weights.unsqueeze(1)
    observed_one_hot = F.one_hot(observed, k).to(dtype=dtype)
    anchor_counts = anchor_weights.T @ observed_one_hot
    anchor_support = anchor_weights.sum(0)

    template, global_diagonal, high_ids = _global_template(
        anchor_counts,
        anchor_support,
        high_support_fraction=high_support_fraction,
        min_high_support_classes=min_high_support_classes,
        eps=eps,
    )
    empirical = _normalize_rows(anchor_counts, eps=eps)
    shrink = anchor_support / (anchor_support + float(shrinkage_mass)).clamp_min(eps)
    transition = _normalize_rows(
        shrink.unsqueeze(1) * empirical + (1.0 - shrink).unsqueeze(1) * template,
        eps=eps,
    )

    # Step 2: Concave EM over clean prior pi with T anchored.
    if initial_prior is None:
        prior = torch.full((k,), 1.0 / k, dtype=dtype, device=device)
    else:
        prior = torch.as_tensor(initial_prior, dtype=dtype, device=device).detach().clone()
        if (
            prior.shape != (k,)
            or not bool(torch.isfinite(prior).all())
            or bool((prior <= 0).any())
            or bool(prior.sum() <= 0)
        ):
            raise ValueError("initial_prior must be a finite positive [K] vector")
        prior = prior / prior.sum()

    likelihood = transition[:, observed].T  # [N, K]
    converged = False
    final_delta = float("inf")
    actual_iterations = 0
    for iteration in range(1, int(max_iter) + 1):
        previous_prior = prior.clone()
        joint = probs * prior.unsqueeze(0) * likelihood
        mass = joint.sum(dim=1, keepdim=True)
        if bool((mass <= eps).any()):
            raise RuntimeError("Anchored EM E-step posterior has zero mass")
        posterior = joint / mass

        prior = posterior.sum(dim=0) + float(prior_pseudocount)
        prior = prior.clamp_min(eps)
        prior = prior / prior.sum()

        final_delta = float((prior - previous_prior).abs().max().item())
        actual_iterations = iteration
        if final_delta <= float(tolerance):
            converged = True
            break

    joint = probs * prior.unsqueeze(0) * likelihood
    mass = joint.sum(dim=1, keepdim=True)
    if bool((mass <= eps).any()):
        raise RuntimeError("Anchored EM final posterior has zero mass")
    posterior = joint / mass
    effective_support = posterior.sum(dim=0)
    est_noise = float(1.0 - (prior * transition.diag()).sum().item())

    return AnchoredTransitionEstimate(
        transition=transition,
        clean_prior=prior,
        posterior_clean=posterior,
        anchor_support=anchor_support,
        effective_support=effective_support,
        global_diagonal_retention=global_diagonal,
        global_support_class_ids=high_ids,
        estimated_noise_rate=est_noise,
        iterations=actual_iterations,
        converged=converged,
        final_delta=final_delta,
    )


@torch.no_grad()
def blend_transition_and_prior(
    current_prior: torch.Tensor,
    current_transition: torch.Tensor,
    target_prior: torch.Tensor,
    target_transition: torch.Tensor,
    *,
    blend_alpha: float = 0.35,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Smoothly update online (pi, T) toward the newest EM estimate."""
    if not math.isfinite(blend_alpha) or not 0.0 < blend_alpha <= 1.0:
        raise ValueError("blend_alpha must be in (0, 1]")
    alpha = float(blend_alpha)
    cur_pi = torch.as_tensor(current_prior, dtype=torch.float64)
    tgt_pi = torch.as_tensor(target_prior, dtype=torch.float64, device=cur_pi.device)
    cur_t = torch.as_tensor(current_transition, dtype=torch.float64, device=cur_pi.device)
    tgt_t = torch.as_tensor(target_transition, dtype=torch.float64, device=cur_pi.device)

    if cur_pi.ndim != 1 or tgt_pi.shape != cur_pi.shape:
        raise ValueError("Prior vectors must be matching 1-D tensors")
    k = cur_pi.numel()
    if cur_t.shape != (k, k) or tgt_t.shape != (k, k):
        raise ValueError("Transition matrices must have matching [K, K] shapes")

    cur_pi = cur_pi / cur_pi.sum().clamp_min(eps)
    tgt_pi = tgt_pi / tgt_pi.sum().clamp_min(eps)
    cur_t = _normalize_rows(cur_t, eps=eps)
    tgt_t = _normalize_rows(tgt_t, eps=eps)

    new_pi = ((1.0 - alpha) * cur_pi + alpha * tgt_pi).clamp_min(eps)
    new_pi = new_pi / new_pi.sum()
    new_t = _normalize_rows((1.0 - alpha) * cur_t + alpha * tgt_t, eps=eps)
    return new_pi, new_t


class EMATeacher:
    """Exponential moving average teacher with exact student BN buffer copy."""

    def __init__(
        self,
        student: nn.Module,
        decay: float = 0.99,
        *,
        warmup_ramp: bool = True,
    ):
        if not math.isfinite(decay) or not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must be finite and in [0, 1)")
        self.model = copy.deepcopy(student).eval()
        self.decay = float(decay)
        self.warmup_ramp = bool(warmup_ramp)
        self.num_updates = 0
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @property
    def effective_decay(self) -> float:
        if not self.warmup_ramp:
            return self.decay
        return min(self.decay, float(self.num_updates + 1) / float(self.num_updates + 10))

    @torch.no_grad()
    def update(self, student: nn.Module) -> float:
        alpha = self.effective_decay
        student_params = dict(student.named_parameters())
        teacher_params = dict(self.model.named_parameters())
        if student_params.keys() != teacher_params.keys():
            raise ValueError("Student and EMA teacher parameter names differ")
        for name, t_param in teacher_params.items():
            src = student_params[name].detach().to(t_param)
            t_param.mul_(alpha).add_(src, alpha=1.0 - alpha)

        student_buffers = dict(student.named_buffers())
        teacher_buffers = dict(self.model.named_buffers())
        if student_buffers.keys() != teacher_buffers.keys():
            raise ValueError("Student and EMA teacher buffer names differ")
        for name, t_buf in teacher_buffers.items():
            t_buf.copy_(student_buffers[name].detach().to(t_buf))

        self.num_updates += 1
        self.model.eval()
        return alpha

    def state_dict(self) -> Dict[str, object]:
        return {
            "model_state": self.model.state_dict(),
            "decay": self.decay,
            "warmup_ramp": self.warmup_ramp,
            "num_updates": self.num_updates,
            "buffer_policy": "copy_student_buffers_each_update",
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self.model.load_state_dict(state["model_state"], strict=True)
        self.decay = float(state["decay"])
        self.warmup_ramp = bool(state["warmup_ramp"])
        self.num_updates = int(state["num_updates"])
        self.model.eval()


class EstimatedTransitionForwardCorrection(nn.Module):
    """Forward noise-corrected balanced softmax with updatable (pi, T) buffers."""

    def __init__(
        self,
        clean_prior: torch.Tensor,
        transition: torch.Tensor,
        *,
        epsilon: float = 1e-8,
    ):
        super().__init__()
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        self.epsilon = float(epsilon)
        pi_t, t_t = self._validated_tensors(clean_prior, transition)
        self.register_buffer("prior", pi_t.to(torch.float32))
        self.register_buffer("transition", t_t.to(torch.float32))
        self.register_buffer("log_prior", pi_t.log().to(torch.float32))
        self.register_buffer(
            "log_transition", t_t.clamp_min(self.epsilon).log().to(torch.float32)
        )

    @staticmethod
    def _validated_tensors(
        clean_prior: torch.Tensor, transition: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pi = torch.as_tensor(clean_prior, dtype=torch.float64).detach()
        t = torch.as_tensor(transition, dtype=torch.float64).detach()
        if (
            pi.ndim != 1
            or pi.numel() < 2
            or not bool(torch.isfinite(pi).all())
            or bool(torch.any(pi <= 0))
        ):
            raise ValueError("clean_prior must be a positive finite 1-D vector")
        k = int(pi.numel())
        if (
            t.shape != (k, k)
            or not bool(torch.isfinite(t).all())
            or bool(torch.any(t < 0))
        ):
            raise ValueError("transition must be a finite non-negative [K, K] matrix")
        if not torch.allclose(t.sum(dim=1), torch.ones(k, dtype=t.dtype), atol=1e-6, rtol=0):
            raise ValueError("transition rows must sum to one")
        return pi / pi.sum(), t

    @torch.no_grad()
    def update_transition_and_prior(
        self, clean_prior: torch.Tensor, transition: torch.Tensor
    ) -> None:
        pi_t, t_t = self._validated_tensors(clean_prior, transition)
        device = self.log_prior.device
        self.prior.copy_(pi_t.to(device=device, dtype=self.prior.dtype))
        self.transition.copy_(t_t.to(device=device, dtype=self.transition.dtype))
        self.log_prior.copy_(pi_t.log().to(device=device, dtype=self.log_prior.dtype))
        self.log_transition.copy_(
            t_t.clamp_min(self.epsilon).log().to(device=device, dtype=self.log_transition.dtype)
        )

    def _adjusted_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] != self.log_prior.numel():
            raise ValueError("logits must have shape [nonempty batch, K]")
        if not logits.is_floating_point() or not bool(torch.isfinite(logits).all()):
            raise ValueError("logits must be finite floating-point values")
        return logits + self.log_prior.to(logits)

    def clean_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        return F.softmax(self._adjusted_logits(logits), dim=1)

    def observed_log_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        log_clean = F.log_softmax(self._adjusted_logits(logits), dim=1)
        log_t = self.log_transition.to(logits)
        return torch.logsumexp(log_clean.unsqueeze(2) + log_t.unsqueeze(0), dim=1)

    def forward(self, logits: torch.Tensor, observed_targets: torch.Tensor) -> torch.Tensor:
        log_clean = F.log_softmax(self._adjusted_logits(logits), dim=1)
        if observed_targets.ndim != 1 or observed_targets.shape[0] != logits.shape[0]:
            raise ValueError("observed_targets must have shape [batch]")
        if (
            observed_targets.is_floating_point()
            or observed_targets.is_complex()
            or observed_targets.dtype == torch.bool
        ):
            raise ValueError("observed_targets must contain integer class IDs")
        targets = observed_targets.to(device=logits.device, dtype=torch.long)
        if bool(torch.any((targets < 0) | (targets >= logits.shape[1]))):
            raise ValueError("observed target outside class range")
        selected_log_t = self.log_transition.to(logits)[:, targets].T
        return -torch.logsumexp(log_clean + selected_log_t, dim=1).mean()


class NoiseAwareSoftBalancedContrastiveLossV2(nn.Module):
    """Soft-BCL v2 with denominator floor, support floor, and continuous semantic ramp.

    When support_floor_ratio == 0 and semantic_ramp == 1 and soft_targets is
    one-hot, forward reduces numerically to BCL's BalancedContrastiveLoss.
    When semantic_ramp == 0, forward reduces to two-view instance NT-Xent
    without touching class_prototypes or soft_targets.
    """

    def __init__(
        self,
        num_classes: int = 100,
        temperature: float = 0.1,
        *,
        support_floor_ratio: float = 0.5,
        denominator_epsilon: float = 1e-6,
    ):
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least two")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if not math.isfinite(support_floor_ratio) or support_floor_ratio < 0:
            raise ValueError("support_floor_ratio must be finite and non-negative")
        if not math.isfinite(denominator_epsilon) or denominator_epsilon <= 0:
            raise ValueError("denominator_epsilon must be finite and positive")
        self.num_classes = int(num_classes)
        self.temperature = float(temperature)
        self.support_floor_ratio = float(support_floor_ratio)
        self.denominator_epsilon = float(denominator_epsilon)

    @staticmethod
    def confidence_reliability(q: torch.Tensor) -> torch.Tensor:
        k = q.shape[1]
        minimum, maximum = torch.aminmax(q, dim=1)
        reliability = ((maximum - 1.0 / k) / (1.0 - 1.0 / k)).clamp(0.0, 1.0)
        return reliability.masked_fill(maximum == minimum, 0.0)

    def _instance_only_loss(self, two_view_features: torch.Tensor) -> torch.Tensor:
        batch_size = int(two_view_features.shape[0])
        features = F.normalize(
            torch.cat((two_view_features[:, 0], two_view_features[:, 1]), dim=0),
            dim=1,
        )
        logits = (features @ features.T) / self.temperature
        num_views = 2 * batch_size
        diagonal = torch.eye(num_views, device=logits.device, dtype=torch.bool)
        logits = logits.masked_fill(diagonal, torch.finfo(logits.dtype).min)
        indices = torch.arange(num_views, device=logits.device)
        paired = (indices + batch_size) % num_views
        return F.cross_entropy(logits, paired)

    def forward(
        self,
        class_prototypes: Optional[torch.Tensor],
        two_view_features: torch.Tensor,
        soft_targets: Optional[torch.Tensor] = None,
        *,
        semantic_ramp: float = 1.0,
        return_diagnostics: bool = False,
    ):
        if not math.isfinite(semantic_ramp) or not 0.0 <= semantic_ramp <= 1.0:
            raise ValueError("semantic_ramp must be in [0, 1]")
        if (
            two_view_features.ndim != 3
            or two_view_features.shape[1] != 2
            or two_view_features.shape[0] == 0
            or two_view_features.shape[2] == 0
        ):
            raise ValueError("two_view_features must have shape [nonempty batch, 2, D]")
        if not two_view_features.is_floating_point() or not bool(
            torch.isfinite(two_view_features).all()
        ):
            raise ValueError("two_view_features must contain finite floating-point values")

        if semantic_ramp == 0.0:
            loss = self._instance_only_loss(two_view_features)
            if not return_diagnostics:
                return loss
            return loss, {
                "semantic_ramp": 0.0,
                "anchor_positive_mass_mean": 1.0,
                "anchor_positive_mass_min": 1.0,
                "denominator_weight_min": 1.0,
                "denominator_weight_median": 1.0,
                "q_entropy_mean": math.log(self.num_classes),
                "q_reliability_mean": 0.0,
            }

        batch_size, _, feat_dim = two_view_features.shape
        if class_prototypes is None or class_prototypes.shape != (self.num_classes, feat_dim):
            raise ValueError("class_prototypes must have shape [num_classes, D]")
        if soft_targets is None or soft_targets.shape != (batch_size, self.num_classes):
            raise ValueError("soft_targets must have shape [batch, num_classes]")
        prototypes = class_prototypes.to(two_view_features)
        if not bool(torch.isfinite(prototypes).all()):
            raise ValueError("class_prototypes must be finite")

        q = soft_targets.detach().to(two_view_features)
        if not bool(torch.isfinite(q).all() & (q >= 0).all()):
            raise ValueError("soft_targets must be finite and non-negative")
        q_mass = q.sum(dim=1, keepdim=True)
        if not bool(torch.isfinite(q_mass).all() & (q_mass > 0).all()):
            raise ValueError("every soft-target row must have finite positive mass")
        q = q / q_mass
        reliability = self.confidence_reliability(q)

        prototype_q = torch.eye(self.num_classes, device=q.device, dtype=q.dtype)
        candidate_q = torch.cat((q, q, prototype_q), dim=0)
        anchor_q = torch.cat((q, q), dim=0)
        candidate_reliability = torch.cat(
            (
                reliability,
                reliability,
                torch.ones(self.num_classes, device=q.device, dtype=q.dtype),
            ),
            dim=0,
        )
        anchor_reliability = torch.cat((reliability, reliability), dim=0)

        num_anchors = 2 * batch_size
        logits_mask = torch.ones(
            (num_anchors, num_anchors + self.num_classes),
            device=q.device,
            dtype=q.dtype,
        )
        anchor_indices = torch.arange(num_anchors, device=q.device)
        logits_mask[anchor_indices, anchor_indices] = 0.0
        logits_mask[:, num_anchors:] = float(semantic_ramp)

        positive_weights = anchor_q @ candidate_q.T
        positive_weights = (
            positive_weights
            * anchor_reliability.unsqueeze(1)
            * candidate_reliability.unsqueeze(0)
        )
        positive_weights[:, :num_anchors] = (
            positive_weights[:, :num_anchors] * float(semantic_ramp)
        )
        paired_indices = (anchor_indices + batch_size) % num_anchors
        positive_weights[anchor_indices, paired_indices] = 1.0
        positive_weights = positive_weights * logits_mask

        class_occurrences = candidate_q.sum(dim=0)
        effective_occurrences = class_occurrences
        if self.support_floor_ratio > 0.0:
            floor_value = torch.maximum(
                class_occurrences.new_tensor(1.0),
                class_occurrences.mean() * self.support_floor_ratio,
            )
            effective_occurrences = torch.maximum(class_occurrences, floor_value)
        candidate_class_mass = candidate_q @ effective_occurrences
        denominator_weight = (
            candidate_class_mass.unsqueeze(0) - positive_weights
        ).clamp_min(self.denominator_epsilon)

        sample_features = torch.cat(
            (two_view_features[:, 0], two_view_features[:, 1]), dim=0
        )
        all_features = torch.cat((sample_features, prototypes), dim=0)
        logits = (sample_features @ all_features.T) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        exp_logits = torch.exp(logits) * logits_mask
        denominator = (exp_logits / denominator_weight).sum(dim=1, keepdim=True)
        if not bool(torch.isfinite(denominator).all()) or bool(torch.any(denominator <= 0)):
            raise RuntimeError("Soft-BCL v2 denominator is non-finite or non-positive")
        log_prob = logits - torch.log(
            denominator.clamp_min(torch.finfo(logits.dtype).tiny)
        )

        positive_mass = positive_weights.sum(dim=1)
        if bool(torch.any(positive_mass <= 0)):
            raise RuntimeError("Every Soft-BCL v2 anchor must retain its paired-view positive")
        mean_log_prob_positive = (positive_weights * log_prob).sum(dim=1) / positive_mass
        loss = -mean_log_prob_positive.mean()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Soft-BCL v2 loss is non-finite")
        if not return_diagnostics:
            return loss

        q_safe = q.clamp_min(1e-12)
        q_entropy = float(-(q_safe * q_safe.log()).sum(dim=1).mean().item())
        return loss, {
            "semantic_ramp": float(semantic_ramp),
            "anchor_positive_mass_mean": float(positive_mass.mean().item()),
            "anchor_positive_mass_min": float(positive_mass.min().item()),
            "denominator_weight_min": float(denominator_weight.min().item()),
            "denominator_weight_median": float(denominator_weight.median().item()),
            "q_entropy_mean": q_entropy,
            "q_reliability_mean": float(reliability.mean().item()),
        }


@torch.no_grad()
def measure_prototype_and_classifier_health(
    classifier_weight: torch.Tensor,
    prototype_mlp: nn.Module,
    *,
    small_weight_threshold: float = 0.15,
) -> Dict[str, object]:
    """Audit classifier norms and both raw & direction-normalized prototype geometry."""
    was_training = prototype_mlp.training
    prototype_mlp.eval()
    try:
        w = classifier_weight.detach()
        raw_protos = prototype_mlp(w)
        dir_protos = prototype_mlp(F.normalize(w, dim=1))
        raw_health = prototype_health_diagnostics(
            w, raw_protos, small_weight_threshold=small_weight_threshold
        )
        dir_health = prototype_health_diagnostics(
            w, dir_protos, small_weight_threshold=small_weight_threshold
        )
    finally:
        prototype_mlp.train(was_training)

    w_norm = raw_health["classifier_weight_norm"]
    return {
        "classifier_weight_norm": w_norm.cpu().tolist(),
        "classifier_weight_norm_min": float(w_norm.min().item()),
        "classifier_weight_norm_median": float(w_norm.median().item()),
        "classifier_weight_norm_max": float(w_norm.max().item()),
        "small_weight_class_ids": torch.nonzero(
            raw_health["small_weight_mask"], as_tuple=False
        ).flatten().cpu().tolist(),
        "raw_mean_offdiag_cosine": float(raw_health["mean_offdiag_cosine"].item()),
        "raw_max_offdiag_cosine": float(
            raw_health["prototype_max_offdiag_cosine"].max().item()
        ),
        "raw_per_class_max_offdiag_cosine": raw_health[
            "prototype_max_offdiag_cosine"
        ].cpu().tolist(),
        "dir_mean_offdiag_cosine": float(dir_health["mean_offdiag_cosine"].item()),
        "dir_max_offdiag_cosine": float(
            dir_health["prototype_max_offdiag_cosine"].max().item()
        ),
        "dir_per_class_max_offdiag_cosine": dir_health[
            "prototype_max_offdiag_cosine"
        ].cpu().tolist(),
    }
