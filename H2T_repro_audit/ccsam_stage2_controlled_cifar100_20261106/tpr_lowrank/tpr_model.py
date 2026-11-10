"""Core CC-SAM-TPR low-rank tail residual and regularizers.

This module is intentionally independent of the repository's training runner so
that the parameterization and loss semantics can be unit-tested on CPU.
"""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class TailLowRankResidual(nn.Module):
    """Class-specific coefficients over one shared low-rank tail basis.

    The base classifier is supplied by the caller and remains frozen. Only A
    and B are trainable. Many rows are exactly masked to zero in DeltaW.
    """

    def __init__(
        self,
        class_counts: Iterable[int] | torch.Tensor,
        feature_dim: int = 64,
        rank: int = 8,
        many_threshold: int = 100,
        seed: int = 20261110,
    ) -> None:
        super().__init__()
        counts = torch.as_tensor(class_counts, dtype=torch.long)
        if counts.ndim != 1 or counts.numel() != 100:
            raise ValueError("class_counts must contain exactly 100 entries")
        if bool((counts <= 0).any().item()):
            raise ValueError("all class counts must be positive")
        if feature_dim <= 0 or rank <= 0 or rank > min(100, feature_dim):
            raise ValueError("invalid feature_dim/rank")
        if many_threshold < 0:
            raise ValueError("many_threshold must be non-negative")

        tail_mask = (counts <= int(many_threshold)).to(torch.float32)
        if not bool(tail_mask.any().item()) or bool((tail_mask == 1).all().item()):
            raise ValueError("protocol requires both Many and residual-eligible classes")

        self.num_classes = 100
        self.feature_dim = int(feature_dim)
        self.rank = int(rank)
        self.many_threshold = int(many_threshold)
        self.register_buffer("tail_mask", tail_mask)

        # Local generator makes initialization independent of global RNG state.
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        a = torch.empty(rank, feature_dim, dtype=torch.float32)
        nn.init.orthogonal_(a, gain=1.0, generator=generator)
        self.A = nn.Parameter(a)
        self.B = nn.Parameter(torch.zeros(100, rank, dtype=torch.float32))

    def delta_weight(self) -> torch.Tensor:
        return self.tail_mask[:, None] * (self.B @ self.A)

    def effective_weight(self, base_weight: torch.Tensor) -> torch.Tensor:
        if base_weight.shape != (self.num_classes, self.feature_dim):
            raise ValueError("base_weight must have shape [100, feature_dim]")
        if base_weight.device != self.A.device or base_weight.dtype != self.A.dtype:
            raise ValueError("base_weight and low-rank parameters must share dtype/device")
        return base_weight + self.delta_weight()

    def logits(
        self,
        features: torch.Tensor,
        base_weight: torch.Tensor,
        base_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError("features must have shape [B, feature_dim]")
        if features.device != self.A.device or features.dtype != self.A.dtype:
            raise ValueError("features and low-rank parameters must share dtype/device")
        return F.linear(features, self.effective_weight(base_weight), base_bias)

    @torch.no_grad()
    def assert_many_rows_zero(self) -> None:
        many = self.tail_mask == 0
        if not torch.equal(self.delta_weight()[many], torch.zeros_like(self.delta_weight()[many])):
            raise RuntimeError("Many residual rows must be exactly zero")


def dominant_right_subspace(weight: torch.Tensor, k: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    """Return top-k right singular vectors V_k and singular values of base W."""
    if weight.ndim != 2 or weight.shape != (100, 64):
        raise ValueError("base classifier weight must have shape [100,64]")
    if not torch.isfinite(weight).all():
        raise ValueError("base classifier contains non-finite values")
    if not 1 <= k <= min(weight.shape):
        raise ValueError("k is outside the valid SVD rank range")
    # Explicit FP32 deterministic computation; caller sets deterministic backend.
    _, singular_values, vh = torch.linalg.svd(weight.detach().to(torch.float32), full_matrices=False)
    basis = vh[:k].transpose(0, 1).contiguous()
    if not torch.isfinite(basis).all() or not torch.isfinite(singular_values).all():
        raise RuntimeError("non-finite SVD output")
    return basis, singular_values


def dominant_subspace_overlap(
    delta_weight: torch.Tensor,
    basis: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    if delta_weight.ndim != 2 or delta_weight.shape[1] != 64:
        raise ValueError("delta_weight must have shape [100,64]")
    if basis.ndim != 2 or basis.shape[0] != 64:
        raise ValueError("basis must have shape [64,k]")
    if delta_weight.device != basis.device:
        raise ValueError("delta_weight and basis must share a device")
    numerator = torch.linalg.matrix_norm(delta_weight @ basis, ord="fro").square()
    denominator = torch.linalg.matrix_norm(delta_weight, ord="fro").square() + eps
    return numerator / denominator


def select_confusion_pairs(
    confusion: torch.Tensor,
    tail_mask: torch.Tensor,
    top_k: int = 5,
) -> list[tuple[int, int, float]]:
    """Select top symmetric off-diagonal confusion pairs among tail classes.

    confusion[i,j] is the number of validation examples with true class i and
    predicted class j. Scores average row-normalized i->j and j->i confusion.
    """
    if confusion.shape != (100, 100) or tail_mask.shape != (100,):
        raise ValueError("expected confusion [100,100] and tail_mask [100]")
    if not torch.isfinite(confusion).all() or bool((confusion < 0).any().item()):
        raise ValueError("confusion matrix must be finite and non-negative")
    row_totals = confusion.sum(dim=1)
    pairs: list[tuple[int, int, float]] = []
    eligible = torch.nonzero(tail_mask.to(torch.bool), as_tuple=False).flatten().tolist()
    for pos, c in enumerate(eligible):
        for cp in eligible[pos + 1:]:
            if row_totals[c] <= 0 or row_totals[cp] <= 0:
                continue
            score = 0.5 * (
                confusion[c, cp] / row_totals[c]
                + confusion[cp, c] / row_totals[cp]
            )
            value = float(score.detach().cpu())
            if value > 0.0:
                pairs.append((int(c), int(cp), value))
    # Stable deterministic tie break: descending score, then ascending IDs.
    pairs.sort(key=lambda item: (-item[2], item[0], item[1]))
    return pairs[: max(0, int(top_k))]


def confusion_aware_diversity_loss(
    delta_weight: torch.Tensor,
    pairs: Iterable[tuple[int, int, float]],
    eps: float = 1e-12,
) -> torch.Tensor:
    """Penalize squared cosine similarity of residual rows for selected pairs."""
    if delta_weight.shape != (100, 64):
        raise ValueError("delta_weight must have shape [100,64]")
    pair_list = list(pairs)
    if not pair_list:
        return delta_weight.sum() * 0.0
    terms = []
    for c, cp, score in pair_list:
        if not (0 <= c < 100 and 0 <= cp < 100 and c != cp):
            raise ValueError(f"invalid confusion pair {(c, cp)}")
        if not (0.0 <= float(score) <= 1.0):
            raise ValueError("confusion score must lie in [0,1]")
        wc, wcp = delta_weight[c], delta_weight[cp]
        denom = wc.norm().clamp_min(eps) * wcp.norm().clamp_min(eps)
        cosine = torch.dot(wc, wcp) / denom
        terms.append(float(score) * cosine.square())
    return torch.stack(terms).mean()


def many_head_preservation_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    class_counts: torch.Tensor,
    temperature: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Full 100-way KL(teacher || student), masked by Many input labels only."""
    if student_logits.ndim != 2 or student_logits.shape[1] != 100:
        raise ValueError("student_logits must have shape [B,100]")
    if teacher_logits.shape != student_logits.shape:
        raise ValueError("teacher/student logits must have identical shapes")
    if labels.shape != (student_logits.shape[0],) or labels.dtype != torch.long:
        raise ValueError("labels must be int64 [B]")
    counts = torch.as_tensor(class_counts, dtype=torch.long, device=labels.device)
    if counts.shape != (100,) or bool((counts <= 0).any().item()):
        raise ValueError("class_counts must contain 100 positive values")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not (student_logits.device == teacher_logits.device == labels.device):
        raise ValueError("student, teacher and labels must share device")
    if not torch.isfinite(student_logits).all() or not torch.isfinite(teacher_logits).all():
        raise ValueError("teacher/student logits must be finite")
    mask = counts[labels] > 100
    n_many = int(mask.sum().item())
    if n_many == 0:
        return student_logits.sum() * 0.0, mask, 0
    log_student = F.log_softmax(student_logits[mask] / temperature, dim=1)
    with torch.no_grad():
        prob_teacher = F.softmax(teacher_logits[mask].detach() / temperature, dim=1)
    loss = F.kl_div(log_student, prob_teacher, reduction="batchmean") * (temperature ** 2)
    return loss, mask, n_many
