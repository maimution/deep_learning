from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineCNN(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0.3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.pool(x)
        scale = F.relu(self.fc1(scale), inplace=True)
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        use_se: bool = False,
        se_reduction: int = 16,
    ) -> None:
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SEBlock(planes, reduction=se_reduction) if use_se else nn.Identity()
        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        return F.relu(out, inplace=True)


class SmallResNet(nn.Module):
    def __init__(
        self,
        layers: list[int],
        num_classes: int = 10,
        base_width: int = 64,
        use_se: bool = False,
        se_reduction: int = 16,
    ) -> None:
        super().__init__()
        self.in_planes = base_width
        self.use_se = use_se
        self.se_reduction = se_reduction

        self.stem = nn.Sequential(
            nn.Conv2d(3, base_width, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(base_width, layers[0], stride=1)
        self.layer2 = self._make_layer(base_width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(base_width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(base_width * 8, layers[3], stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(base_width * 8, num_classes)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [
            BasicBlock(
                self.in_planes,
                planes,
                stride=stride,
                use_se=self.use_se,
                se_reduction=self.se_reduction,
            )
        ]
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(
                BasicBlock(
                    self.in_planes,
                    planes,
                    stride=1,
                    use_se=self.use_se,
                    se_reduction=self.se_reduction,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class WideBasicBlock(nn.Module):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        stride: int,
        dropout: float,
        use_se: bool = False,
        se_reduction: int = 16,
    ) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(
            out_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=1,
                stride=stride,
                bias=False,
            )
        self.se = SEBlock(out_planes, reduction=se_reduction) if use_se else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(F.relu(self.bn1(x), inplace=True))
        out = self.dropout(out)
        out = self.conv2(F.relu(self.bn2(out), inplace=True))
        out = self.se(out)
        return out + self.shortcut(x)


class WideLayer(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        in_planes: int,
        out_planes: int,
        stride: int,
        dropout: float,
        use_se: bool,
        se_reduction: int,
    ) -> None:
        super().__init__()
        blocks = []
        strides = [stride] + [1] * (num_blocks - 1)
        current_in = in_planes
        for block_stride in strides:
            blocks.append(
                WideBasicBlock(
                    current_in,
                    out_planes,
                    block_stride,
                    dropout,
                    use_se=use_se,
                    se_reduction=se_reduction,
                )
            )
            current_in = out_planes
        self.block = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class WideResNet(nn.Module):
    def __init__(
        self,
        depth: int = 28,
        widen_factor: int = 2,
        dropout: float = 0.3,
        num_classes: int = 10,
        use_se: bool = False,
        se_reduction: int = 16,
    ) -> None:
        super().__init__()
        if (depth - 4) % 6 != 0:
            raise ValueError("WideResNet depth must satisfy 6n + 4.")

        num_blocks = (depth - 4) // 6
        widths = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        self.conv1 = nn.Conv2d(3, widths[0], kernel_size=3, padding=1, bias=False)
        self.layer1 = WideLayer(
            num_blocks,
            widths[0],
            widths[1],
            stride=1,
            dropout=dropout,
            use_se=use_se,
            se_reduction=se_reduction,
        )
        self.layer2 = WideLayer(
            num_blocks,
            widths[1],
            widths[2],
            stride=2,
            dropout=dropout,
            use_se=use_se,
            se_reduction=se_reduction,
        )
        self.layer3 = WideLayer(
            num_blocks,
            widths[2],
            widths[3],
            stride=2,
            dropout=dropout,
            use_se=use_se,
            se_reduction=se_reduction,
        )
        self.bn = nn.BatchNorm2d(widths[3])
        self.fc = nn.Linear(widths[3], num_classes)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.relu(self.bn(x), inplace=True)
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        return self.fc(x)


@dataclass(frozen=True)
class ModelSpec:
    model: nn.Module
    display_name: str


def create_model(
    name: str,
    num_classes: int = 10,
    dropout: float = 0.3,
    se_reduction: int = 16,
    wide_depth: int = 28,
    wide_factor: int = 2,
) -> ModelSpec:
    name = name.lower()
    if name == "baseline_cnn":
        return ModelSpec(BaselineCNN(num_classes=num_classes, dropout=dropout), "Baseline CNN")
    if name == "resnet18":
        return ModelSpec(
            SmallResNet([2, 2, 2, 2], num_classes=num_classes, use_se=False),
            "ResNet-18",
        )
    if name == "se_resnet18":
        return ModelSpec(
            SmallResNet(
                [2, 2, 2, 2],
                num_classes=num_classes,
                use_se=True,
                se_reduction=se_reduction,
            ),
            "SE-ResNet-18",
        )
    if name == "wideresnet":
        return ModelSpec(
            WideResNet(
                depth=wide_depth,
                widen_factor=wide_factor,
                dropout=dropout,
                num_classes=num_classes,
                use_se=False,
                se_reduction=se_reduction,
            ),
            f"WideResNet-{wide_depth}-{wide_factor}",
        )
    if name == "se_wideresnet":
        return ModelSpec(
            WideResNet(
                depth=wide_depth,
                widen_factor=wide_factor,
                dropout=dropout,
                num_classes=num_classes,
                use_se=True,
                se_reduction=se_reduction,
            ),
            f"SE-WideResNet-{wide_depth}-{wide_factor}",
        )
    raise ValueError(f"Unsupported model: {name}")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
