from __future__ import annotations

import re
from collections import OrderedDict
from pathlib import Path

import torch

try:
    from monai.networks.nets import UNet as MonaiUNet
except ImportError:
    MonaiUNet = None


DEFAULT_FULL_PIPELINE_MONAI_CONFIG = "D4_D_k5"
_MONAI_DEFAULT_ACT = ("RELU", {"inplace": True})
_MONAI_DEFAULT_DROPOUT = 0.0
_MONAI_DEFAULT_KERNEL_SIZE = 3
_MONAI_DEFAULT_ADN_ORDERING = "NDA"


FULL_PIPELINE_MONAI_CONFIGS: dict[str, dict[str, object]] = {
    "D4_B": {
        "channels": (32, 64, 128, 256, 512),
        "strides": (2, 2, 2, 2),
        "num_res_units": 0,
        "dropout": 0.0,
        "kernel_size": 3,
        "adn_ordering": "NDA",
        "act": _MONAI_DEFAULT_ACT,
    },
    "D4_C": {
        "channels": (32, 64, 128, 256, 512),
        "strides": (2, 2, 2, 2),
        "num_res_units": 1,
        "dropout": 0.0,
        "kernel_size": 3,
        "adn_ordering": "NDA",
        "act": _MONAI_DEFAULT_ACT,
    },
    "D4_D_k5": {
        "channels": (32, 64, 128, 256, 512),
        "strides": (2, 2, 2, 2),
        "num_res_units": 2,
        "dropout": 0.0,
        "kernel_size": 5,
        "adn_ordering": "NDA",
        "act": _MONAI_DEFAULT_ACT,
    },
    "D4_F_k5": {
        "channels": (48, 96, 192, 384, 768),
        "strides": (2, 2, 2, 2),
        "num_res_units": 2,
        "dropout": 0.0,
        "kernel_size": 5,
        "adn_ordering": "NDA",
        "act": _MONAI_DEFAULT_ACT,
    },
}


def _common_group_count(channels: tuple[int, ...], max_groups: int = 8) -> int:
    groups = min(max_groups, min(channels))
    while groups > 1 and any(channel % groups != 0 for channel in channels):
        groups -= 1
    return groups


def build_full_pipeline_unet(
    *,
    num_input_channels: int,
    monai_config: str = DEFAULT_FULL_PIPELINE_MONAI_CONFIG,
) -> torch.nn.Module:
    if MonaiUNet is None:
        raise RuntimeError("MONAI is required for full-pipeline model loading. Install it with `pip install monai`.")

    try:
        config = FULL_PIPELINE_MONAI_CONFIGS[str(monai_config)]
    except KeyError as exc:
        raise ValueError(
            f"Unknown full-pipeline MONAI config '{monai_config}'. Expected one of {sorted(FULL_PIPELINE_MONAI_CONFIGS)}"
        ) from exc

    channels = tuple(int(value) for value in config["channels"])
    strides = tuple(int(value) for value in config["strides"])
    if len(channels) != len(strides) + 1:
        raise ValueError(
            f"MONAI U-Net expects len(channels)=len(strides)+1, got channels={channels}, strides={strides}."
        )

    norm_groups = _common_group_count(tuple(int(value) for value in (*channels, num_input_channels, num_input_channels)))
    norm_cfg = ("GROUP", {"num_groups": norm_groups, "affine": True})

    return MonaiUNet(
        spatial_dims=2,
        in_channels=int(num_input_channels),
        out_channels=int(num_input_channels),
        channels=channels,
        strides=strides,
        kernel_size=int(config.get("kernel_size", _MONAI_DEFAULT_KERNEL_SIZE)),
        up_kernel_size=int(config.get("kernel_size", _MONAI_DEFAULT_KERNEL_SIZE)),
        num_res_units=int(config.get("num_res_units", 0)),
        act=config.get("act", _MONAI_DEFAULT_ACT),
        norm=norm_cfg,
        dropout=float(config.get("dropout", _MONAI_DEFAULT_DROPOUT)),
        bias=False,
        adn_ordering=str(config.get("adn_ordering", _MONAI_DEFAULT_ADN_ORDERING)),
    )


def load_model_checkpoint(
    path: Path,
    model: torch.nn.Module,
    device: torch.device,
    *,
    strict: bool = True,
) -> int:
    try:
        info = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        info = torch.load(path, map_location=device)

    state_dict = info.get("state_dict", info)
    cleaned_state: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, value in state_dict.items():
        cleaned_state[str(key).replace("module.", "")] = value

    try:
        model.load_state_dict(cleaned_state, strict=strict)
    except RuntimeError as exc:
        # Older MONAI checkpoints used sub0/sub1/subconv/subresidual naming.
        remapped_state: OrderedDict[str, torch.Tensor] = OrderedDict()
        for key, value in cleaned_state.items():
            new_key = re.sub(r"\.sub(\d+)", lambda match: f".submodule.{match.group(1)}", key)
            new_key = new_key.replace(".subconv.", ".submodule.conv.")
            new_key = new_key.replace(".subresidual.", ".submodule.residual.")
            remapped_state[new_key] = value
        if remapped_state.keys() == cleaned_state.keys():
            raise
        try:
            model.load_state_dict(remapped_state, strict=strict)
        except RuntimeError as remap_exc:
            raise RuntimeError(
                "Failed to load checkpoint after applying legacy MONAI key remap.\n"
                f"Original load error:\n{exc}\n\n"
                f"Remapped load error:\n{remap_exc}"
            ) from remap_exc

    return int(info.get("epoch", -1)) + 1
