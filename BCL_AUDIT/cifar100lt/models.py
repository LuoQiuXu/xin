from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LambdaLayer(nn.Module):
    def __init__(self, function):
        super().__init__()
        self.function = function

    def forward(self, x):
        return self.function(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1, option: str = "A"):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == "A":
                # CIFAR ResNet Option A: spatial subsampling + zero-padding channels.
                channel_pad = (planes - in_planes) // 2
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(
                        x[:, :, ::2, ::2],
                        (0, 0, 0, 0, channel_pad, channel_pad),
                        mode="constant",
                        value=0,
                    )
                )
            elif option == "B":
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes, planes, kernel_size=1, stride=stride, bias=False
                    ),
                    nn.BatchNorm2d(planes),
                )
            else:
                raise ValueError("CIFAR ResNet shortcut option must be A or B")

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


class CIFARResNet32(nn.Module):
    """CIFAR-style ResNet-32: 3x3 stem and [5,5,5] residual blocks."""

    def __init__(self, num_classes: int = 100, shortcut_option: str = "A"):
        super().__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(
            3, 16, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, blocks=5, stride=1, option=shortcut_option)
        self.layer2 = self._make_layer(32, blocks=5, stride=2, option=shortcut_option)
        self.layer3 = self._make_layer(64, blocks=5, stride=2, option=shortcut_option)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = 64

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_layer(self, planes: int, blocks: int, stride: int, option: str):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for block_stride in strides:
            layers.append(
                BasicBlock(self.in_planes, planes, stride=block_stride, option=option)
            )
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        return torch.flatten(out, 1)


class ProjectionMLP(nn.Module):
    def __init__(self, input_dim: int = 64, hidden_dim: int = 512, output_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)


class BCLCIFARModel(nn.Module):
    """Shared CIFAR backbone, linear classifier, sample projector and prototype MLP."""

    def __init__(
        self,
        num_classes: int = 100,
        feature_dim: int = 64,
        projection_hidden_dim: int = 512,
        projection_dim: int = 128,
    ):
        super().__init__()
        if feature_dim != 64:
            raise ValueError("The CIFAR ResNet-32 feature dimension is fixed to 64")
        self.backbone = CIFARResNet32(num_classes=num_classes)
        # CIFAR port uses a standard linear classifier; do not import ImageNet s=30.
        self.classifier = nn.Linear(feature_dim, num_classes)
        # Match the Kaiming-normal linear-head initialization in the referenced
        # CIFAR ResNet-32 implementation; do not use the ImageNet cosine scale.
        nn.init.kaiming_normal_(self.classifier.weight)
        # Separate MLPs: sample projection and classifier-weight prototype mapping.
        self.sample_mlp = ProjectionMLP(
            feature_dim, projection_hidden_dim, projection_dim
        )
        self.prototype_mlp = ProjectionMLP(
            feature_dim, projection_hidden_dim, projection_dim
        )

    def forward(self, images: torch.Tensor, compute_contrastive: bool = True):
        features = self.backbone(images)
        logits = self.classifier(features)
        if not compute_contrastive:
            return features, logits, None, None

        sample_embeddings = self.sample_mlp(features)
        # nn.Linear.weight has shape [num_classes, feature_dim]; each row is W_y.
        class_prototypes = self.prototype_mlp(self.classifier.weight)
        return features, logits, sample_embeddings, class_prototypes
