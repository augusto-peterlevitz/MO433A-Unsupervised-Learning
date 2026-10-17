
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Two convolution layers with normalization and ReLU."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class StudentCNN(nn.Module):
    """
    CNN encoder for self-supervised learning and distillation.

    Returns a global-average-pooled feature vector.
    No classification head is included.
    """

    def __init__(self, width_mult=1.0):
        super().__init__()

        if width_mult <= 0:
            raise ValueError("width_mult must be positive")

        # The architecture changes only through this width multiplier.
        base = int(32 * width_mult)

        if base < 1:
            raise ValueError("width_mult is too small")

        channels = [
            base,
            base * 2,
            base * 4,
            base * 8,
        ]

        self.stem = nn.Sequential(
            nn.Conv2d(
                3, channels[0],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )

        self.stage1 = ConvBlock(
            channels[0], channels[0], stride=1
        )
        self.stage2 = ConvBlock(
            channels[0], channels[1], stride=2
        )
        self.stage3 = ConvBlock(
            channels[1], channels[2], stride=2
        )
        self.stage4 = ConvBlock(
            channels[2], channels[3], stride=2
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.feature_dim = channels[3]

    def forward_features(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        # [batch, channels, height, width]
        x = self.pool(x)

        # [batch, channels, 1, 1] -> [batch, channels]
        return torch.flatten(x, start_dim=1)

    def forward(self, x):
        return self.forward_features(x)