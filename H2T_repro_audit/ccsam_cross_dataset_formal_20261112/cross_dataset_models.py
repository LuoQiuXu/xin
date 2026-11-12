"""Common ResNet-50 Stage-I model for both cross-dataset benchmarks."""
from __future__ import annotations

import hashlib
from typing import Dict

import torch
from torch import nn

FEATURE_DIM = 2048
ARCHITECTURE = "torchvision_resnet50_from_scratch"


def build_stage1_models(num_classes: int):
    """Return a ResNet-50 encoder and a separate linear classifier.

    No pretrained weights are loaded. Replacing the original fc with Identity
    makes the encoder return the 2,048-D global-average-pooled representation.
    """
    if int(num_classes) not in (365, 1000):
        raise ValueError("cross-dataset class count must be 365 or 1000")
    from torchvision.models import resnet50

    network = resnet50(weights=None)
    feature_dim = int(network.fc.in_features)
    if feature_dim != FEATURE_DIM:
        raise RuntimeError("Unexpected torchvision ResNet-50 feature dimension: %d" % feature_dim)
    network.fc = nn.Identity()
    classifier = nn.Linear(FEATURE_DIM, int(num_classes))
    return network, classifier


def state_dict_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def freeze_encoder(encoder: nn.Module) -> Dict[str, torch.Tensor]:
    """Freeze parameters and BN statistics; return a bitwise state snapshot."""
    encoder.eval()
    encoder.requires_grad_(False)
    for module in encoder.modules():
        module.eval()
    return {
        name: value.detach().cpu().clone()
        for name, value in encoder.state_dict().items()
    }


def assert_encoder_frozen(encoder: nn.Module, snapshot: Dict[str, torch.Tensor]) -> None:
    encoder.eval()
    if any(parameter.requires_grad for parameter in encoder.parameters()):
        raise RuntimeError("A frozen encoder parameter became trainable")
    if any(parameter.grad is not None for parameter in encoder.parameters()):
        raise RuntimeError("A frozen encoder parameter unexpectedly received a gradient")
    current = encoder.state_dict()
    if set(current) != set(snapshot):
        raise RuntimeError("Frozen encoder state keys changed")
    for name, value in current.items():
        if not torch.equal(value.detach().cpu(), snapshot[name]):
            raise RuntimeError("Frozen encoder or BN buffer changed: " + name)
