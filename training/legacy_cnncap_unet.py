"""Repo-local copy of the legacy CNNCap U-Net."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LegacyCNNCapUNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, base_channels: int = 16, depth: int = 4) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"Legacy CNNCap U-Net depth must be >= 1, got {depth}")

        self.depth = depth
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()

        current_channels = in_channels
        for stage in range(depth):
            next_channels = base_channels * (2 ** stage)
            self.downs.append(_DoubleConv(current_channels, next_channels))
            self.pools.append(nn.MaxPool2d(kernel_size=2))
            current_channels = next_channels

        self.bottleneck = _DoubleConv(current_channels, current_channels * 2)
        current_channels *= 2

        self.ups = nn.ModuleList()
        for stage in range(depth - 1, -1, -1):
            skip_channels = base_channels * (2 ** stage)
            self.ups.append(_DoubleConv(current_channels + skip_channels, skip_channels))
            current_channels = skip_channels

        self.out_conv = nn.Conv2d(current_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        out = x

        for down, pool in zip(self.downs, self.pools):
            out = down(out)
            skips.append(out)
            out = pool(out)

        out = self.bottleneck(out)

        for up, skip in zip(self.ups, reversed(skips)):
            out = F.interpolate(out, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            out = torch.cat([skip, out], dim=1)
            out = up(out)

        return self.out_conv(out)
