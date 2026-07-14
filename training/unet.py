import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_group_norm(channels: int, max_groups: int = 8) -> nn.GroupNorm:
    groups = min(max_groups, channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, use_se: bool = False, use_dwsep: bool = False):
        super().__init__()
        conv_factory = _DepthwiseSeparableConv if use_dwsep else _StandardConv
        self.block = nn.Sequential(
            conv_factory(in_ch, out_ch),
            _make_group_norm(out_ch),
            nn.ReLU(inplace=True),
            conv_factory(out_ch, out_ch),
            _make_group_norm(out_ch),
            nn.ReLU(inplace=True),
        )
        self.se = SqueezeExcite(out_ch) if use_se else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.se(self.block(x))


class _StandardConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch,
            in_ch,
            kernel_size=3,
            padding=1,
            groups=in_ch,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.pool(x)
        scale = self.act(self.fc1(scale))
        scale = self.gate(self.fc2(scale))
        return x * scale


class UNet(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        base_ch: int = 16,
        depth: int = 3,
        *,
        use_se: bool = False,
        use_dwsep: bool = False,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError(f"U-Net depth must be at least 1, got {depth}")

        self.depth = depth
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()

        encoder_channels = []
        current_ch = in_ch
        for stage in range(depth):
            next_ch = base_ch * (2 ** stage)
            self.downs.append(DoubleConv(current_ch, next_ch, use_se=use_se, use_dwsep=use_dwsep))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            encoder_channels.append(next_ch)
            current_ch = next_ch

        bottleneck_ch = base_ch * (2 ** depth)
        self.bottleneck = DoubleConv(current_ch, bottleneck_ch, use_se=use_se, use_dwsep=use_dwsep)

        self.ups = nn.ModuleList()
        decoder_in = bottleneck_ch
        for skip_ch in reversed(encoder_channels):
            self.ups.append(DoubleConv(decoder_in + skip_ch, skip_ch, use_se=use_se, use_dwsep=use_dwsep))
            decoder_in = skip_ch

        self.out_conv = nn.Conv2d(decoder_in, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        out = x

        for down, pool in zip(self.downs, self.pools):
            out = down(out)
            skips.append(out)
            out = pool(out)

        out = self.bottleneck(out)

        for up, skip in zip(self.ups, reversed(skips)):
            out = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=False)
            if out.shape[-2:] != skip.shape[-2:]:
                out = F.interpolate(out, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            out = torch.cat([skip, out], dim=1)
            out = up(out)

        return self.out_conv(out)
