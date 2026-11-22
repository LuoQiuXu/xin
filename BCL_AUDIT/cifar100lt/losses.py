from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LogitCompensatedCE(nn.Module):
    """Logit adjustment with delta_y = log(class_prior_y), alpha_y = 1."""

    def __init__(self, class_counts):
        super().__init__()
        counts = torch.as_tensor(class_counts, dtype=torch.float32)
        if counts.ndim != 1 or torch.any(counts <= 0):
            raise ValueError("class_counts must be a positive 1-D vector")
        log_prior = torch.log(counts / counts.sum())
        self.register_buffer("log_prior", log_prior)

    def adjusted_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return logits + self.log_prior.to(device=logits.device, dtype=logits.dtype)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(self.adjusted_logits(logits), targets)


class BalancedContrastiveLoss(nn.Module):
    """BCL class-complement + class-averaging loss.

    This is a device-safe implementation of the released BalSCL formulation:
    https://github.com/FlamieZhu/Balanced-Contrastive-Learning/blob/main/loss/contrastive.py

    Args:
        class_counts: long-tail training count for every class, used to define K.
        temperature: contrastive temperature (0.1 for CIFAR-100-LT in BCL).
    """

    def __init__(self, class_counts, temperature: float = 0.1):
        super().__init__()
        counts = torch.as_tensor(class_counts, dtype=torch.float32)
        if counts.ndim != 1 or torch.any(counts <= 0):
            raise ValueError("class_counts must be a positive 1-D vector")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.num_classes = int(counts.numel())
        self.temperature = float(temperature)

    def forward(
        self,
        class_prototypes: torch.Tensor,
        two_view_features: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        if two_view_features.ndim != 3 or two_view_features.shape[1] != 2:
            raise ValueError(
                "two_view_features must have shape [batch, 2, feature_dim]"
            )
        batch_size = int(two_view_features.shape[0])
        if class_prototypes.shape[0] != self.num_classes:
            raise ValueError("class_prototypes must contain one prototype per class")
        if targets.ndim != 1 or targets.shape[0] != batch_size:
            raise ValueError("targets must have shape [batch]")
        if targets.min().item() < 0 or targets.max().item() >= self.num_classes:
            raise ValueError("target label outside class range")

        device = two_view_features.device
        dtype = two_view_features.dtype
        targets = targets.to(device=device, dtype=torch.long)

        # Official BCL ordering: all view-2 samples, then all view-3 samples.
        sample_features = torch.cat(
            [two_view_features[:, 0, :], two_view_features[:, 1, :]], dim=0
        )
        all_features = torch.cat(
            [sample_features, class_prototypes.to(device=device, dtype=dtype)], dim=0
        )

        sample_targets = torch.cat([targets, targets], dim=0)
        prototype_targets = torch.arange(self.num_classes, device=device, dtype=torch.long)
        all_targets = torch.cat([sample_targets, prototype_targets], dim=0)

        num_anchors = 2 * batch_size
        logits_mask = torch.ones(
            (num_anchors, num_anchors + self.num_classes),
            device=device,
            dtype=dtype,
        )
        anchor_indices = torch.arange(num_anchors, device=device)
        logits_mask[anchor_indices, anchor_indices] = 0

        positive_mask = (
            all_targets[:num_anchors].view(-1, 1)
            == all_targets.view(1, -1)
        ).to(dtype=dtype)
        positive_mask = positive_mask * logits_mask

        # Number of sample views plus one class prototype for each class.
        class_occurrences = torch.bincount(
            all_targets, minlength=self.num_classes
        ).to(device=device, dtype=dtype)

        logits = torch.matmul(all_features[:num_anchors], all_features.T)
        logits = logits / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        exp_logits = torch.exp(logits) * logits_mask
        # For positive entries subtract one because the anchor itself is excluded.
        per_entry_class_count = class_occurrences[all_targets].view(1, -1)
        per_entry_weight = per_entry_class_count.expand_as(exp_logits) - positive_mask
        if torch.any(per_entry_weight <= 0):
            raise RuntimeError("BCL class-averaging produced a non-positive denominator weight")

        denominator = (exp_logits / per_entry_weight).sum(dim=1, keepdim=True)
        log_prob = logits - torch.log(denominator.clamp_min(torch.finfo(dtype).tiny))

        positive_count = positive_mask.sum(dim=1)
        if torch.any(positive_count <= 0):
            raise RuntimeError("Every BCL anchor must have at least one positive")
        mean_log_prob_positive = (positive_mask * log_prob).sum(dim=1) / positive_count

        # Average over both contrastive views and the batch.
        loss = -mean_log_prob_positive.view(2, batch_size).mean()
        return loss
