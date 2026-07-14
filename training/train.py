#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import math
import random
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from monai.networks.nets import UNet as MonaiUNet
except ImportError:
    MonaiUNet = None

try:
    from monai.networks.nets import AttentionUnet as MonaiAttentionUnet
except ImportError:
    MonaiAttentionUnet = None

try:
    from ptflops import get_model_complexity_info
except ImportError:
    get_model_complexity_info = None

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:

    class SummaryWriter:  # type: ignore
        def __init__(self, *args, **kwargs):
            print("WARNING: tensorboard not available; proceeding without logging.")

        def add_scalar(self, *args, **kwargs):
            return None

        def add_figure(self, *args, **kwargs):
            return None

        def add_text(self, *args, **kwargs):
            return None

        def close(self):
            return None


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import resnet_custom
import legacy_cnncap_dataset
from binary_hdf5_cache import (
    BinaryHdf5CacheConfig,
    BinaryHdf5CapBenchDataset,
    build_binary_cache_spec,
    ensure_binary_hdf5_cache,
)
from losses import LOSS_CHOICES, compute_log_mse, compute_loss, compute_mlse, compute_msre
from legacy_cnncap_unet import LegacyCNNCapUNet
from train_utils import AverageMeter

if TYPE_CHECKING:
    from capbench.window_density_dataset import WindowCapDataset


VIZ_EVERY = 0
VIZ_DPI = 150
WINDOW_SPLIT_SEED = 42
RESNET_MODEL_TYPES = (
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet50_no_avgpool",
)
SCALAR_MODEL_TYPES = RESNET_MODEL_TYPES
DATASET_FORMAT_CHOICES = ("capbench", "cnncap_legacy")
_TENSORRT_LOGGER = None
FIXED_ERROR_RATIO_THRESHOLDS = (0.05, 0.10)


@dataclass(frozen=True)
class CapBenchModules:
    resolve_dataset_path: Any
    load_density_id_window_dataset: Any
    load_density_window_dataset: Any
    make_window_grouped_batch_sampler: Any
    window_dataset_cls: Any
    id_map_dataset_cls: Any
    create_window_level_splits: Any
    verify_no_data_leakage: Any


def _load_capbench_modules() -> CapBenchModules:
    try:
        from capbench.datasets import resolve_dataset_path
        from capbench.dataloaders import (
            load_density_id_window_dataset,
            load_density_window_dataset,
            make_window_grouped_batch_sampler,
        )
        from capbench.window_density_dataset import WindowCapDataset
        from capbench.window_id_map_dataset import IdMapWindowDataset
        from capbench._internal.common.window_splitting import (
            create_window_level_splits,
            verify_no_data_leakage,
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "CapBench is not installed. Install it first from its checkout with "
            "`pip install -e \".[all]\"`."
        ) from exc

    return CapBenchModules(
        resolve_dataset_path=resolve_dataset_path,
        load_density_id_window_dataset=load_density_id_window_dataset,
        load_density_window_dataset=load_density_window_dataset,
        make_window_grouped_batch_sampler=make_window_grouped_batch_sampler,
        window_dataset_cls=WindowCapDataset,
        id_map_dataset_cls=IdMapWindowDataset,
        create_window_level_splits=create_window_level_splits,
        verify_no_data_leakage=verify_no_data_leakage,
    )


def _resolve_dataset_root(dataset_spec: str, *, required_artifacts: Sequence[str]) -> Path:
    candidate = Path(dataset_spec).expanduser()
    if candidate.exists():
        return candidate.resolve()
    return _load_capbench_modules().resolve_dataset_path(dataset_spec, artifacts=required_artifacts).resolve()


def _normalize_goal(goal: str) -> str:
    normalized = goal.strip().lower()
    if normalized == "coupling":
        return "env"
    if normalized in {"total", "env"}:
        return normalized
    raise ValueError(f"Unsupported goal: {goal}")

_MONAI_DEFAULT_ACT = ("RELU", {"inplace": True})
_MONAI_DEFAULT_DROPOUT = 0.0
_MONAI_DEFAULT_KERNEL_SIZE = 3
_MONAI_DEFAULT_ADN_ORDERING = "NDA"
_MONAI_DEFAULT_ARCH = "unet"

MONAI_ABLATION_CONFIGS = {
    # Exact 13 configurations reported in the accepted-paper ablation.
    "D3_B": {"depth": 3, "base_ch": 16, "channels": (16, 32, 64, 128), "strides": (2, 2, 2), "num_res_units": 0},
    "D3_C": {"depth": 3, "base_ch": 16, "channels": (16, 32, 64, 128), "strides": (2, 2, 2), "num_res_units": 1},
    "D4_A": {"depth": 4, "base_ch": 24, "channels": (24, 48, 96, 192, 384), "strides": (2, 2, 2, 2), "num_res_units": 0},
    "D4_B": {"depth": 4, "base_ch": 32, "channels": (32, 64, 128, 256, 512), "strides": (2, 2, 2, 2), "num_res_units": 0},
    "D4_C": {"depth": 4, "base_ch": 32, "channels": (32, 64, 128, 256, 512), "strides": (2, 2, 2, 2), "num_res_units": 1},
    "D4_D": {"depth": 4, "base_ch": 32, "channels": (32, 64, 128, 256, 512), "strides": (2, 2, 2, 2), "num_res_units": 2},
    "D4_C_k5": {"depth": 4, "base_ch": 32, "channels": (32, 64, 128, 256, 512), "strides": (2, 2, 2, 2), "num_res_units": 1, "kernel_size": 5},
    "D4_D_k5": {"depth": 4, "base_ch": 32, "channels": (32, 64, 128, 256, 512), "strides": (2, 2, 2, 2), "num_res_units": 2, "kernel_size": 5},
    "D4_E": {"depth": 4, "base_ch": 48, "channels": (48, 96, 192, 384, 768), "strides": (2, 2, 2, 2), "num_res_units": 1},
    "D4_F": {"depth": 4, "base_ch": 48, "channels": (48, 96, 192, 384, 768), "strides": (2, 2, 2, 2), "num_res_units": 2},
    "D4_F_k5": {"depth": 4, "base_ch": 48, "channels": (48, 96, 192, 384, 768), "strides": (2, 2, 2, 2), "num_res_units": 2, "kernel_size": 5},
    "D5_B": {"depth": 5, "base_ch": 24, "channels": (24, 48, 96, 192, 384, 768), "strides": (2, 2, 2, 2, 2), "num_res_units": 1},
    "A4_A": {"depth": 4, "base_ch": 32, "channels": (32, 64, 128, 256, 512), "strides": (2, 2, 2, 2), "num_res_units": 0, "arch": "attention_unet"},
}


def _parse_int_tuple(raw_value: str, *, field_name: str) -> Tuple[int, ...]:
    try:
        values = tuple(int(token.strip()) for token in raw_value.split(",") if token.strip())
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a comma-separated integer list, got: {raw_value}") from exc
    if not values:
        raise ValueError(f"{field_name} must contain at least one integer.")
    return values


def _common_group_count(channels: Tuple[int, ...], max_groups: int = 8) -> int:
    groups = min(max_groups, min(channels))
    while groups > 1 and any(channel % groups != 0 for channel in channels):
        groups -= 1
    return groups


def _parse_act_arg(raw: str):
    """Parse a CLI activation string like ``"PRELU"`` or ``"LEAKYRELU:0.1"``."""
    if ":" in raw:
        name, slope = raw.split(":", 1)
        return (name.strip().upper(), {"negative_slope": float(slope)})
    return raw.strip().upper()


def _parse_norm_arg(raw: str):
    """Parse a CLI norm string like ``"INSTANCE"`` or ``"GROUP"``."""
    name = raw.strip().upper()
    if name == "INSTANCE":
        return ("INSTANCE", {"affine": True})
    return name


def resolve_monai_unet_config(
    args: argparse.Namespace,
    *,
    fallback_base_ch: int,
    fallback_depth: int,
) -> dict:
    if args.monai_config:
        config = MONAI_ABLATION_CONFIGS[args.monai_config]
        channels = tuple(int(v) for v in config["channels"])
        strides = tuple(int(v) for v in config["strides"])
        return {
            "channels": channels,
            "strides": strides,
            "num_res_units": int(config["num_res_units"]),
            "dropout": float(config.get("dropout", _MONAI_DEFAULT_DROPOUT)),
            "act": config.get("act", _MONAI_DEFAULT_ACT),
            "norm": config.get("norm", None),
            "kernel_size": int(config.get("kernel_size", _MONAI_DEFAULT_KERNEL_SIZE)),
            "adn_ordering": str(config.get("adn_ordering", _MONAI_DEFAULT_ADN_ORDERING)),
            "arch": str(config.get("arch", _MONAI_DEFAULT_ARCH)),
        }

    monai_depth = int(args.monai_depth) if args.monai_depth is not None else int(fallback_depth)
    if args.monai_channels:
        channels = _parse_int_tuple(args.monai_channels, field_name="--monai-channels")
    else:
        channels = tuple(int(fallback_base_ch) * (2 ** stage) for stage in range(monai_depth + 1))

    if args.monai_strides:
        strides = _parse_int_tuple(args.monai_strides, field_name="--monai-strides")
    else:
        strides = tuple(2 for _ in range(monai_depth))

    if len(channels) != monai_depth + 1:
        raise ValueError(
            f"MONAI channels length must equal monai_depth + 1. "
            f"Got depth={monai_depth}, len(channels)={len(channels)}."
        )
    if len(strides) != monai_depth:
        raise ValueError(
            f"MONAI strides length must equal monai_depth. "
            f"Got depth={monai_depth}, len(strides)={len(strides)}."
        )

    act = _parse_act_arg(args.monai_act) if args.monai_act else _MONAI_DEFAULT_ACT
    norm = _parse_norm_arg(args.monai_norm) if args.monai_norm else None

    return {
        "channels": channels,
        "strides": strides,
        "num_res_units": int(args.monai_num_res_units),
        "dropout": float(args.monai_dropout),
        "act": act,
        "norm": norm,
        "kernel_size": int(args.monai_kernel_size),
        "adn_ordering": str(args.monai_adn_ordering),
        "arch": _MONAI_DEFAULT_ARCH,
    }


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic


def get_device(device: str) -> torch.device:
    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("WARNING: CUDA requested but not available, falling back to CPU.")
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def is_scalar_model(model_type: str) -> bool:
    return model_type in SCALAR_MODEL_TYPES


def get_model(
    model_type: str,
    num_input_channels: int,
    base_ch: int,
    depth: int,
    *,
    monai_config: Optional[dict] = None,
) -> torch.nn.Module:
    if model_type == "unet":
        config = monai_config or {}
        arch = config.get("arch", _MONAI_DEFAULT_ARCH)
        channels = config.get("channels") or tuple(base_ch * (2 ** stage) for stage in range(depth + 1))
        strides = config.get("strides") or tuple(2 for _ in range(depth))
        if len(channels) != len(strides) + 1:
            raise ValueError(
                f"MONAI U-Net expects len(channels)=len(strides)+1, got "
                f"channels={channels}, strides={strides}."
            )
        kernel_size = int(config.get("kernel_size", _MONAI_DEFAULT_KERNEL_SIZE))
        dropout = float(config.get("dropout", _MONAI_DEFAULT_DROPOUT))

        if arch == "attention_unet":
            if MonaiAttentionUnet is None:
                raise RuntimeError(
                    "MONAI AttentionUnet is required. Install it with `pip install monai`."
                )
            return MonaiAttentionUnet(
                spatial_dims=2,
                in_channels=num_input_channels,
                out_channels=num_input_channels,
                channels=channels,
                strides=strides,
                kernel_size=kernel_size,
                up_kernel_size=kernel_size,
                dropout=dropout,
            )

        # Default: standard MONAI UNet
        if MonaiUNet is None:
            raise RuntimeError("MONAI is required for U-Net training. Install it with `pip install monai`.")

        num_res_units = int(config.get("num_res_units", 0))
        act = config.get("act", _MONAI_DEFAULT_ACT)
        adn_ordering = str(config.get("adn_ordering", _MONAI_DEFAULT_ADN_ORDERING))
        norm_cfg = config.get("norm", None)
        if norm_cfg is None:
            # Include in/out channels so GroupNorm stays valid on MONAI residual/top paths too.
            norm_groups = _common_group_count(tuple(int(v) for v in (*channels, num_input_channels, num_input_channels)))
            norm_cfg = ("GROUP", {"num_groups": norm_groups, "affine": True})

        return MonaiUNet(
            spatial_dims=2,
            in_channels=num_input_channels,
            out_channels=num_input_channels,
            channels=channels,
            strides=strides,
            kernel_size=kernel_size,
            up_kernel_size=kernel_size,
            num_res_units=num_res_units,
            act=act,
            norm=norm_cfg,
            dropout=dropout,
            bias=False,
            adn_ordering=adn_ordering,
        )
    if model_type == "cnncap_unet":
        return LegacyCNNCapUNet(
            in_channels=num_input_channels,
            out_channels=num_input_channels,
            base_channels=base_ch,
            depth=depth,
        )
    if model_type == "resnet18":
        return resnet_custom.resnet18(num_classes=1, num_input_channels=num_input_channels)
    if model_type == "resnet34":
        return resnet_custom.resnet34(num_classes=1, num_input_channels=num_input_channels)
    if model_type == "resnet50":
        return resnet_custom.resnet50(num_classes=1, num_input_channels=num_input_channels)
    if model_type == "resnet101":
        return resnet_custom.resnet101(num_classes=1, num_input_channels=num_input_channels)
    if model_type == "resnet50_no_avgpool":
        return resnet_custom.resnet50_no_avgpool(num_classes=1, num_input_channels=num_input_channels)
    raise ValueError(f"Unsupported model type: {model_type}")


def get_latest_git_commit_message(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "log", "-1", "--pretty=%s%n%b"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return "Unavailable"

    message = result.stdout.strip()
    return message if message else "Unavailable"


def build_tensorboard_description(args: argparse.Namespace, model_stats: Optional[str] = None) -> str:
    command_line = " ".join(shlex.quote(arg) for arg in sys.argv)
    commit_message = get_latest_git_commit_message(REPO_ROOT)
    description = (
        "Latest git commit message:\n"
        "```\n"
        f"{commit_message}\n"
        "```\n\n"
        "Command line:\n"
        "```bash\n"
        f"{command_line}\n"
        "```\n\n"
        "Parsed arguments:\n"
        "```\n"
        f"{args}\n"
        "```"
    )
    if model_stats:
        description += (
            "\n\n"
            "Pre-run model stats:\n"
            "```\n"
            f"{model_stats}\n"
            "```"
        )
    return description


def parameter_count(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


@dataclass(frozen=True)
class UnetRunStats:
    input_shape: Tuple[int, int, int]
    params: int
    flops: int
    avg_batch_ms: Optional[float]
    samples_per_second: Optional[float]
    throughput_error: Optional[str]


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    maxerr: float
    avgerr: float
    ratio_above_report: float
    ratio_above_5pct: float
    ratio_above_10pct: float


def estimate_flops_with_ptflops(model: torch.nn.Module, input_shape: Tuple[int, int, int]) -> int:
    if get_model_complexity_info is None:
        raise RuntimeError(
            "ptflops is required for FLOP estimation. Install it with `pip install ptflops` "
            "or update the environment from environment.yml."
        )

    was_training = model.training
    model.eval()
    macs, _params = get_model_complexity_info(
        model,
        input_shape,
        as_strings=False,
        print_per_layer_stat=False,
        verbose=False,
    )
    if was_training:
        model.train()
    return int(round(2.0 * float(macs)))


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def infer_input_shape(dataset: Dataset) -> Tuple[int, int, int]:
    sample = dataset[0]
    if not isinstance(sample, (tuple, list)) or not sample:
        raise RuntimeError("Failed to infer model input shape from the first training sample.")
    features = sample[0]
    if not isinstance(features, torch.Tensor):
        features = torch.as_tensor(features)
    if features.ndim != 3:
        raise RuntimeError(f"Expected a [C, H, W] input tensor, got shape {tuple(features.shape)}")
    return int(features.shape[0]), int(features.shape[1]), int(features.shape[2])




def _load_onnx_module():
    try:
        return importlib.import_module("onnx")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "TensorRT throughput measurement requires the Python 'onnx' package."
        ) from exc


def _load_tensorrt_module():
    try:
        return importlib.import_module("tensorrt")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "TensorRT throughput measurement requires the Python 'tensorrt' package."
        ) from exc


def _get_tensorrt_logger(trt_module):
    global _TENSORRT_LOGGER
    if _TENSORRT_LOGGER is None:
        _TENSORRT_LOGGER = trt_module.Logger(trt_module.Logger.WARNING)
    return _TENSORRT_LOGGER


def _trt_dtype_to_torch_dtype(trt_dtype, trt_module) -> torch.dtype:
    mapping = {
        trt_module.float32: torch.float32,
        trt_module.float16: torch.float16,
        trt_module.int32: torch.int32,
        trt_module.int8: torch.int8,
        trt_module.bool: torch.bool,
    }
    try:
        return mapping[trt_dtype]
    except KeyError as exc:
        raise RuntimeError(f"Unsupported TensorRT dtype: {trt_dtype}") from exc


def _collect_tensorrt_parser_errors(parser) -> str:
    count = int(parser.num_errors)
    messages = []
    for idx in range(count):
        messages.append(str(parser.get_error(idx)))
    return "\n".join(messages) if messages else "<no parser errors reported>"


class _TensorRTForwardWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        raw = self.model(features)
        if isinstance(raw, (tuple, list)):
            raw = raw[0]
        return raw


class _TensorRTBenchmarkModule:
    def __init__(self, serialized_engine: bytes, *, device: torch.device) -> None:
        if device.type != "cuda":
            raise RuntimeError("TensorRT throughput measurement requires a CUDA device.")
        self.device = torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)
        self.execution_stream = torch.cuda.Stream(device=self.device)
        self.trt = _load_tensorrt_module()
        self.logger = _get_tensorrt_logger(self.trt)
        self.runtime = self.trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(serialized_engine)
        if self.engine is None:
            raise RuntimeError("Could not deserialize the TensorRT benchmark engine.")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Could not create the TensorRT benchmark execution context.")
        self._use_tensor_api = hasattr(self.engine, "num_io_tensors")
        (
            self.input_name,
            self.output_name,
            self.input_index,
            self.output_index,
        ) = self._discover_bindings()
        self.input_dtype = _trt_dtype_to_torch_dtype(self._binding_dtype(self.input_name, self.input_index), self.trt)
        self.output_dtype = _trt_dtype_to_torch_dtype(self._binding_dtype(self.output_name, self.output_index), self.trt)
        self.input_min_shape, _input_opt_shape, self.input_max_shape = self._binding_profile_shapes(
            self.input_name,
            self.input_index,
        )
        self.max_batch_size = int(self.input_max_shape[0])
        self.num_input_channels = int(self.input_max_shape[1])
        self.target_size = int(self.input_max_shape[2])

    def _discover_bindings(self) -> tuple[str, str, int, int]:
        if self._use_tensor_api:
            input_names = []
            output_names = []
            index_by_name: dict[str, int] = {}
            for idx in range(int(self.engine.num_io_tensors)):
                name = str(self.engine.get_tensor_name(idx))
                index_by_name[name] = idx
                mode = self.engine.get_tensor_mode(name)
                if mode == self.trt.TensorIOMode.INPUT:
                    input_names.append(name)
                elif mode == self.trt.TensorIOMode.OUTPUT:
                    output_names.append(name)
            if len(input_names) != 1 or len(output_names) != 1:
                raise RuntimeError(
                    "TensorRT throughput measurement expects exactly one input and one output tensor, "
                    f"got inputs={input_names} outputs={output_names}"
                )
            return input_names[0], output_names[0], index_by_name[input_names[0]], index_by_name[output_names[0]]

        input_indices = []
        output_indices = []
        for idx in range(int(self.engine.num_bindings)):
            if bool(self.engine.binding_is_input(idx)):
                input_indices.append(idx)
            else:
                output_indices.append(idx)
        if len(input_indices) != 1 or len(output_indices) != 1:
            raise RuntimeError(
                "TensorRT throughput measurement expects exactly one input and one output binding, "
                f"got inputs={input_indices} outputs={output_indices}"
            )
        input_index = input_indices[0]
        output_index = output_indices[0]
        return (
            str(self.engine.get_binding_name(input_index)),
            str(self.engine.get_binding_name(output_index)),
            input_index,
            output_index,
        )

    def _binding_dtype(self, name: str, index: int):
        if self._use_tensor_api:
            return self.engine.get_tensor_dtype(name)
        return self.engine.get_binding_dtype(index)

    def _binding_profile_shapes(self, name: str, index: int):
        if self._use_tensor_api and hasattr(self.engine, "get_tensor_profile_shape"):
            min_shape, opt_shape, max_shape = self.engine.get_tensor_profile_shape(name, 0)
            return tuple(int(v) for v in min_shape), tuple(int(v) for v in opt_shape), tuple(int(v) for v in max_shape)
        if hasattr(self.engine, "get_profile_shape"):
            min_shape, opt_shape, max_shape = self.engine.get_profile_shape(0, index)
            return tuple(int(v) for v in min_shape), tuple(int(v) for v in opt_shape), tuple(int(v) for v in max_shape)
        static_shape = self._binding_static_shape(name, index)
        return static_shape, static_shape, static_shape

    def _binding_static_shape(self, name: str, index: int) -> tuple[int, ...]:
        if self._use_tensor_api:
            return tuple(int(v) for v in self.engine.get_tensor_shape(name))
        return tuple(int(v) for v in self.engine.get_binding_shape(index))

    def _set_input_shape(self, shape: tuple[int, ...]) -> None:
        if self._use_tensor_api:
            ok = self.context.set_input_shape(self.input_name, shape)
            if not ok:
                raise RuntimeError(f"TensorRT could not set input shape {shape} for the benchmark engine")
            return
        self.context.set_binding_shape(self.input_index, shape)

    def _get_output_shape(self) -> tuple[int, ...]:
        if self._use_tensor_api:
            return tuple(int(v) for v in self.context.get_tensor_shape(self.output_name))
        return tuple(int(v) for v in self.context.get_binding_shape(self.output_index))

    def _execute(self, prepared_input: torch.Tensor, output: torch.Tensor) -> None:
        caller_stream = torch.cuda.current_stream(device=self.device)
        self.execution_stream.wait_stream(caller_stream)
        prepared_input.record_stream(self.execution_stream)
        output.record_stream(self.execution_stream)
        stream_handle = self.execution_stream.cuda_stream
        if self._use_tensor_api:
            self.context.set_tensor_address(self.input_name, int(prepared_input.data_ptr()))
            self.context.set_tensor_address(self.output_name, int(output.data_ptr()))
            ok = self.context.execute_async_v3(stream_handle=int(stream_handle))
        else:
            bindings = [0] * int(self.engine.num_bindings)
            bindings[self.input_index] = int(prepared_input.data_ptr())
            bindings[self.output_index] = int(output.data_ptr())
            ok = self.context.execute_async_v2(bindings=bindings, stream_handle=int(stream_handle))
        if not ok:
            raise RuntimeError("TensorRT benchmark execution failed.")
        caller_stream.wait_stream(self.execution_stream)

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        if features.device.type != "cuda":
            raise RuntimeError("TensorRT throughput measurement requires CUDA tensors.")
        if features.device.index != self.device.index:
            raise RuntimeError(
                f"TensorRT benchmark engine was loaded on cuda:{self.device.index} but received features on {features.device}"
            )
        if features.ndim != 4:
            raise ValueError(f"TensorRT benchmark features must have shape [B, C, H, W], got {tuple(features.shape)}")

        batch_size = int(features.shape[0])
        if batch_size <= 0:
            raise ValueError(f"TensorRT benchmark batch size must be positive, got {batch_size}")
        if batch_size > int(self.max_batch_size):
            raise ValueError(
                f"TensorRT benchmark engine supports max batch {self.max_batch_size}, got {batch_size}"
            )
        if int(features.shape[1]) != int(self.num_input_channels):
            raise ValueError(
                f"TensorRT benchmark engine expects {self.num_input_channels} channels, got {int(features.shape[1])}"
            )
        if int(features.shape[2]) != int(self.target_size) or int(features.shape[3]) != int(self.target_size):
            raise ValueError(
                f"TensorRT benchmark engine expects spatial {self.target_size}x{self.target_size}, "
                f"got {int(features.shape[2])}x{int(features.shape[3])}"
            )

        prepared_input = features
        if prepared_input.dtype != self.input_dtype:
            prepared_input = prepared_input.to(dtype=self.input_dtype)
        if not prepared_input.is_contiguous():
            prepared_input = prepared_input.contiguous()

        input_shape = tuple(int(v) for v in prepared_input.shape)
        self._set_input_shape(input_shape)
        output_shape = self._get_output_shape()
        output = torch.empty(output_shape, device=prepared_input.device, dtype=self.output_dtype)
        self._execute(prepared_input, output)
        return output


def _compile_tensorrt_benchmark_model(
    model: torch.nn.Module,
    *,
    device: torch.device,
    input_shape: Tuple[int, int, int],
    opt_batch_size: int,
    max_batch_size: int,
) -> _TensorRTBenchmarkModule:
    if device.type != "cuda":
        raise RuntimeError("TensorRT throughput measurement requires CUDA.")
    if int(opt_batch_size) <= 0 or int(max_batch_size) <= 0:
        raise ValueError(
            f"TensorRT throughput measurement requires positive batch sizes, got opt={opt_batch_size} max={max_batch_size}"
        )
    if int(opt_batch_size) > int(max_batch_size):
        raise ValueError(
            f"TensorRT throughput measurement requires opt_batch_size <= max_batch_size, got opt={opt_batch_size} max={max_batch_size}"
        )

    _load_onnx_module()
    trt = _load_tensorrt_module()
    wrapper = _TensorRTForwardWrapper(model).eval()
    input_name = "features"
    output_name = "prediction"
    dynamic_axes = None
    if int(max_batch_size) > 1:
        dynamic_axes = {
            input_name: {0: "batch"},
            output_name: {0: "batch"},
        }

    example_input = torch.randn((1, *input_shape), device=device, dtype=torch.float32)

    with tempfile.TemporaryDirectory(prefix="cnncap_flash_trt_") as tmp_dir_str:
        onnx_path = Path(tmp_dir_str) / "throughput.onnx"
        with torch.inference_mode():
            torch.onnx.export(
                wrapper,
                example_input,
                str(onnx_path),
                input_names=[input_name],
                output_names=[output_name],
                # Keep the legacy exporter here because these TensorRT builds
                # rely on dynamic_axes and do not benefit from dynamo export.
                dynamo=False,
                dynamic_axes=dynamic_axes,
                opset_version=18,
                do_constant_folding=True,
            )

        logger = _get_tensorrt_logger(trt)
        builder = trt.Builder(logger)
        explicit_batch_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(explicit_batch_flag)
        parser = trt.OnnxParser(network, logger)
        # Recent PyTorch ONNX exports may store weights in a sibling *.data file.
        # Pass the model path so TensorRT can resolve those external initializers.
        onnx_bytes = onnx_path.read_bytes()
        if not parser.parse(onnx_bytes, str(onnx_path)):
            raise RuntimeError(
                "TensorRT could not parse the exported benchmark ONNX graph:\n"
                f"{_collect_tensorrt_parser_errors(parser)}"
            )

        config = builder.create_builder_config()
        if bool(builder.platform_has_fast_fp16):
            config.set_flag(trt.BuilderFlag.FP16)
        if hasattr(config, "builder_optimization_level"):
            config.builder_optimization_level = 5
        if hasattr(config, "max_num_tactics"):
            config.max_num_tactics = -1
        if hasattr(config, "set_memory_pool_limit") and hasattr(trt, "MemoryPoolType"):
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(1 << 30))
        elif hasattr(config, "max_workspace_size"):
            config.max_workspace_size = int(1 << 30)

        if int(max_batch_size) > 1:
            channels, height, width = (int(v) for v in input_shape)
            profile = builder.create_optimization_profile()
            profile.set_shape(
                input_name,
                (1, channels, height, width),
                (int(opt_batch_size), channels, height, width),
                (int(max_batch_size), channels, height, width),
            )
            config.add_optimization_profile(profile)

        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError("TensorRT failed to build the benchmark engine.")
        return _TensorRTBenchmarkModule(bytes(serialized_engine), device=device)


def measure_forward_throughput(
    model: torch.nn.Module,
    device: torch.device,
    input_shape: Tuple[int, int, int],
    *,
    batch_size: int = 64,
    warmup_iters: int = 5,
    timed_iters: int = 10,
) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    if device.type != "cuda":
        return None, None, "TensorRT throughput measurement requires CUDA"

    first_param = next(iter(model.parameters()), None)
    input_dtype = first_param.dtype if first_param is not None else torch.float32
    dummy = torch.zeros((batch_size, *input_shape), dtype=input_dtype, device=device)
    benchmark_model: Optional[_TensorRTBenchmarkModule] = None
    start: Optional[float] = None
    was_training = model.training
    model.eval()

    try:
        with torch.inference_mode():
            benchmark_model = _compile_tensorrt_benchmark_model(
                model,
                device=device,
                input_shape=input_shape,
                opt_batch_size=batch_size,
                max_batch_size=batch_size,
            )
            for _ in range(warmup_iters):
                _ = benchmark_model(dummy)
            synchronize_device(device)
            start = perf_counter()
            for _ in range(timed_iters):
                _ = benchmark_model(dummy)
            synchronize_device(device)
    except Exception as exc:  # stats-only path should not abort training
        if device.type == "cuda" and "out of memory" in str(exc).lower():
            if was_training:
                model.train()
            return None, None, f"TensorRT throughput measurement at batch_size={batch_size} failed: CUDA out of memory"
        if was_training:
            model.train()
        return None, None, f"TensorRT throughput measurement failed: {exc}"
    finally:
        del dummy
        if benchmark_model is not None:
            del benchmark_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if start is None:
        if was_training:
            model.train()
        return None, None, "TensorRT throughput measurement did not complete"

    elapsed = perf_counter() - start
    if was_training:
        model.train()
    avg_batch_ms = (elapsed / timed_iters) * 1000.0
    samples_per_second = (batch_size * timed_iters) / max(elapsed, 1e-12)
    return avg_batch_ms, samples_per_second, None


def collect_unet_run_stats(
    model: torch.nn.Module,
    device: torch.device,
    train_dataset: Dataset,
) -> UnetRunStats:
    input_shape = infer_input_shape(train_dataset)
    channels, height, width = input_shape

    params = parameter_count(model)
    flops = estimate_flops_with_ptflops(model, input_shape)
    avg_batch_ms, samples_per_second, throughput_error = measure_forward_throughput(
        model,
        device,
        input_shape,
        batch_size=64,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return UnetRunStats(
        input_shape=input_shape,
        params=params,
        flops=flops,
        avg_batch_ms=avg_batch_ms,
        samples_per_second=samples_per_second,
        throughput_error=throughput_error,
    )


def format_unet_run_stats(
    stats: UnetRunStats,
    *,
    device: torch.device,
) -> str:
    channels, height, width = stats.input_shape
    lines = [
        "U-Net pre-run stats:",
        f"  input: batch=64 channels={channels} size={height}x{width} device={device.type}",
        f"  parameters: {stats.params:,}",
        f"  approx FLOPs/sample: {stats.flops:,} (ptflops; 1 MAC = 2 FLOPs)",
    ]
    if stats.throughput_error:
        lines.append(f"  throughput@64: unavailable ({stats.throughput_error})")
    else:
        lines.append(
            f"  throughput@64 (TensorRT): {stats.samples_per_second:.3f} samples/s "
            f"({stats.avg_batch_ms:.3f} ms/batch)"
        )
    return "\n".join(lines)


def _format_summary_value(
    value: Optional[float],
    *,
    decimals: int,
    integer_like: bool = False,
) -> str:
    if value is None:
        return "na"
    if integer_like:
        return str(int(round(value)))
    return f"{value:.{decimals}f}"


def _compute_ratio_above_threshold(err_array: np.ndarray, threshold: float) -> float:
    if err_array.size == 0:
        return 0.0
    return float(np.mean(err_array > threshold))


def _finalize_epoch_metrics(
    losses: AverageMeter,
    maxerr: float,
    errs: Sequence[float],
    report_ratio: float,
) -> EpochMetrics:
    err_array = np.asarray(errs, dtype=np.float64)
    return EpochMetrics(
        loss=losses.avg,
        maxerr=maxerr,
        avgerr=float(np.mean(err_array)) if err_array.size else 0.0,
        ratio_above_report=_compute_ratio_above_threshold(err_array, report_ratio),
        ratio_above_5pct=_compute_ratio_above_threshold(err_array, FIXED_ERROR_RATIO_THRESHOLDS[0]),
        ratio_above_10pct=_compute_ratio_above_threshold(err_array, FIXED_ERROR_RATIO_THRESHOLDS[1]),
    )


def _format_epoch_ratio_fields(prefix: str, metrics: EpochMetrics, report_ratio: float) -> str:
    threshold_values = (
        (report_ratio, metrics.ratio_above_report),
        (FIXED_ERROR_RATIO_THRESHOLDS[0], metrics.ratio_above_5pct),
        (FIXED_ERROR_RATIO_THRESHOLDS[1], metrics.ratio_above_10pct),
    )
    seen: set[float] = set()
    parts: List[str] = []
    for threshold, value in threshold_values:
        normalized_threshold = round(float(threshold), 8)
        if normalized_threshold in seen:
            continue
        seen.add(normalized_threshold)
        parts.append(f"{prefix}_ratio>{threshold*100:.1f}%={value:.4f}")
    return " ".join(parts)


def build_run_summary_line(
    *,
    best_val_avgerr: float,
    best_val_ratio_gt_5pct: float,
    best_val_ratio_gt_10pct: float,
    run_stats: Optional[UnetRunStats],
) -> str:
    return (
        "TRAIN_SUMMARY "
        f"mare={best_val_avgerr:.6f} "
        f"ratio_gt_5pct={best_val_ratio_gt_5pct:.6f} "
        f"ratio_gt_10pct={best_val_ratio_gt_10pct:.6f} "
        f"flops={_format_summary_value(None if run_stats is None else float(run_stats.flops), decimals=0, integer_like=True)} "
        f"samples_per_s={_format_summary_value(None if run_stats is None else run_stats.samples_per_second, decimals=3)} "
        f"params={_format_summary_value(None if run_stats is None else float(run_stats.params), decimals=0, integer_like=True)}"
    )


def summarize_loaded_targets(dataset: WindowCapDataset) -> str:
    if not hasattr(dataset, "_window_samples"):
        return "Loaded target magnitudes: unavailable"

    value_chunks: List[np.ndarray] = []
    for window_samples in dataset._window_samples:  # pylint: disable=protected-access
        targets = getattr(window_samples, "targets", None)
        if isinstance(targets, np.ndarray):
            if targets.size > 0:
                value_chunks.append(np.abs(targets.astype(np.float64, copy=False)))
            continue

        values = [abs(float(sample.target_value)) for sample in window_samples]
        if values:
            value_chunks.append(np.asarray(values, dtype=np.float64))

    if not value_chunks:
        return "Loaded target magnitudes: unavailable (no samples)"

    value_array = value_chunks[0] if len(value_chunks) == 1 else np.concatenate(value_chunks)
    return (
        "Loaded target magnitudes: "
        f"min={value_array.min():.6g} "
        f"median={np.median(value_array):.6g} "
        f"max={value_array.max():.6g}"
    )


def _iter_unique_windows(datasets: Iterable[object]):
    seen: set[int] = set()
    for dataset in datasets:
        windows = getattr(dataset, "_windows", None)
        if windows is None:
            continue
        for window in windows:
            window_id = id(window)
            if window_id in seen:
                continue
            seen.add(window_id)
            yield window


def optimize_capbench_window_storage(
    *datasets: object,
    drop_id_maps: bool,
) -> None:
    for window in _iter_unique_windows(datasets):
        if drop_id_maps:
            if getattr(window, "id_maps", None) is not None:
                setattr(window, "id_maps", None)


def build_optimizer(model: torch.nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )


def compute_scheduled_lr(
    base_lr: float,
    epoch_idx: int,
    total_epochs: int,
    warmup_epochs: int,
    min_lr: float | None = None,
) -> float:
    if total_epochs <= 0:
        return base_lr

    resolved_min_lr = 0.1 * float(base_lr) if min_lr is None else float(min_lr)
    if resolved_min_lr < 0.0:
        raise ValueError(f"min_lr must be non-negative, got {resolved_min_lr}")
    if resolved_min_lr > float(base_lr):
        raise ValueError(f"min_lr must not exceed base_lr, got min_lr={resolved_min_lr} base_lr={base_lr}")

    warmup_epochs = max(0, min(warmup_epochs, total_epochs))
    if warmup_epochs > 0 and epoch_idx < warmup_epochs:
        scale = float(epoch_idx + 1) / float(warmup_epochs)
        return base_lr * scale

    if total_epochs <= warmup_epochs:
        return base_lr

    decay_steps = max(1, total_epochs - warmup_epochs - 1)
    progress = float(epoch_idx - warmup_epochs) / float(decay_steps)
    progress = min(max(progress, 0.0), 1.0)
    cosine_scale = 0.5 * (1.0 + math.cos(math.pi * progress))
    return resolved_min_lr + ((float(base_lr) - resolved_min_lr) * cosine_scale)


def apply_epoch_learning_rate(
    optimizer: torch.optim.Optimizer,
    *,
    base_lr: float,
    epoch_idx: int,
    total_epochs: int,
    warmup_epochs: int,
    min_lr: float | None = None,
) -> float:
    lr = compute_scheduled_lr(base_lr, epoch_idx, total_epochs, warmup_epochs, min_lr=min_lr)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def collate_grouped(batch):
    xs, local_maps, local_counts, query_local_ids, targets, valids = zip(*batch)
    max_k = max(q.shape[0] for q in query_local_ids)
    max_local = max(c.shape[0] for c in local_counts)
    batch_size = len(batch)

    local_counts_padded = torch.ones((batch_size, max_local), dtype=torch.float32)
    query_ids_padded = torch.zeros((batch_size, max_k), dtype=torch.long)
    targets_padded = torch.zeros((batch_size, max_k), dtype=torch.float32)
    valid_padded = torch.zeros((batch_size, max_k), dtype=torch.float32)

    for i, (counts, query_ids, vals, valid) in enumerate(zip(local_counts, query_local_ids, targets, valids)):
        k = query_ids.shape[0]
        c_len = counts.shape[0]
        local_counts_padded[i, :c_len] = counts
        query_ids_padded[i, :k] = query_ids
        targets_padded[i, :k] = vals
        valid_padded[i, :k] = valid

    return (
        torch.stack(xs),
        torch.stack(local_maps),
        local_counts_padded,
        query_ids_padded,
        targets_padded,
        valid_padded,
    )


def collate_total(batch):
    # Total mode is now window-grouped (variable conductors per window), same collation as env mode.
    return collate_grouped(batch)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    is_best: bool,
    save_dir: Path,
    savename: str,
    *,
    best_copy_path: Optional[Path] = None,
    metadata: Optional[Dict[str, object]] = None,
) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "loss": loss,
        "is_best": is_best,
    }
    if metadata:
        state["metadata"] = metadata
    ckpt_path = save_dir / savename
    torch.save(state, ckpt_path)
    if is_best and best_copy_path is not None:
        best_copy_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ckpt_path, best_copy_path)
    return ckpt_path


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    *,
    strict: bool = True,
) -> int:
    try:
        info = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        info = torch.load(path, map_location=device)
    state_dict = info.get("state_dict", info)
    cleaned_state = OrderedDict()
    for key, value in state_dict.items():
        cleaned_state[key.replace("module.", "")] = value

    try:
        model.load_state_dict(cleaned_state, strict=strict)
    except RuntimeError as exc:
        # Older MONAI UNet checkpoints named nested children as sub0/sub1/subconv/subresidual.
        remapped_state = OrderedDict()
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
    if optimizer is not None and "optimizer" in info:
        optimizer.load_state_dict(info["optimizer"])
    return int(info.get("epoch", -1)) + 1


class _WindowMaskDatasetBase(Dataset):
    def __init__(self, base_dataset: WindowCapDataset):
        self.base = base_dataset
        self.layer_name_map = {name: idx for idx, name in enumerate(self.base.active_layers)}

    def __len__(self) -> int:
        return len(self.base)

    def get_window_ids(self) -> List[str]:
        return self.base.get_window_ids()

    def get_window_sample_ranges(self) -> List[Tuple[int, int]]:
        return self.base.get_window_sample_ranges()

    def _build_masks(
        self,
        window_idx: int,
        sample_idx: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        sample = self.base._window_samples[window_idx][sample_idx]  # pylint: disable=protected-access
        local_map, _local_counts, actual_to_local = self.base._build_window_local_state(window_idx)  # pylint: disable=protected-access

        num_layers, height, width = self.base.tensor_shape
        master_masks = np.zeros((num_layers, height, width), dtype=np.float32)
        other_masks = np.zeros((num_layers, height, width), dtype=np.float32)

        positive_local_ids = [
            local_id
            for local_id in (actual_to_local.get(int(conductor_id), 0) for conductor_id in sample.positive_ids)
            if local_id > 0
        ]
        negative_local_ids = [
            local_id
            for local_id in (actual_to_local.get(int(conductor_id), 0) for conductor_id in sample.negative_ids)
            if local_id > 0
        ]

        if positive_local_ids:
            master_masks = np.isin(local_map, positive_local_ids).astype(np.float32, copy=False)

        occupied = local_map > 0
        other_masks = occupied.astype(np.float32, copy=False)
        if positive_local_ids:
            other_masks = np.where(master_masks > 0, 0.0, other_masks).astype(np.float32, copy=False)

        env_masks = np.zeros((len(negative_local_ids), num_layers, height, width), dtype=np.float32)
        for env_idx, local_id in enumerate(negative_local_ids):
            env_masks[env_idx] = (local_map == local_id).astype(np.float32, copy=False)

        return master_masks, other_masks, env_masks

    def _build_raw_features(self, window_idx: int) -> np.ndarray:
        return self.base._build_window_features(window_idx)  # pylint: disable=protected-access

    def _build_conductor_masks(self, window_idx: int, conductor_ids: List[int]) -> np.ndarray:
        return self.base._build_window_conductor_masks(window_idx, conductor_ids)  # pylint: disable=protected-access

    def _resolve_sample(self, index: int):
        window_idx, sample_idx = self.base._get_sample_indices(index)  # pylint: disable=protected-access
        sample = self.base._window_samples[window_idx][sample_idx]  # pylint: disable=protected-access
        return window_idx, sample_idx, sample


class CapBenchCouplingDataset(_WindowMaskDatasetBase):
    def __init__(self, base_dataset: WindowCapDataset):
        super().__init__(base_dataset)
        if self.base.get_grouped_coupling_case_count() <= 0:
            raise RuntimeError("No grouped coupling cases were generated from the dataset.")
        self._window_sample_ranges = self.base.get_grouped_coupling_case_ranges()

    def get_window_sample_ranges(self) -> List[Tuple[int, int]]:
        return list(self._window_sample_ranges)

    def __len__(self) -> int:
        return self.base.get_grouped_coupling_case_count()

    def __getitem__(self, index: int):
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range for dataset with {len(self)} grouped masters")
        window_idx, master_id, slave_ids, targets, valid = self.base.get_grouped_coupling_case(index)
        features = self._build_raw_features(window_idx)
        _window, cached = self.base._get_window_and_cache(window_idx)  # pylint: disable=protected-access
        self.base._apply_highlight(features, cached, master_id, positive=False)  # pylint: disable=protected-access
        local_map_np, local_counts_np, actual_to_local = self.base._build_window_local_state(window_idx)  # pylint: disable=protected-access
        query_local_ids = np.array([actual_to_local.get(int(cid), 0) for cid in slave_ids], dtype=np.int64)

        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(local_map_np, dtype=torch.long),
            torch.tensor(local_counts_np, dtype=torch.float32),
            torch.tensor(query_local_ids, dtype=torch.long),
            torch.tensor(targets, dtype=torch.float32),
            torch.tensor(valid, dtype=torch.float32),
        )

    def get_visualization_case(self, index: int):
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range for dataset with {len(self)} grouped masters")
        window_idx, master_id, slave_ids, targets, valid = self.base.get_grouped_coupling_case(index)
        features = self._build_raw_features(window_idx)
        _window, cached = self.base._get_window_and_cache(window_idx)  # pylint: disable=protected-access
        self.base._apply_highlight(features, cached, master_id, positive=False)  # pylint: disable=protected-access
        master_masks = self._build_conductor_masks(window_idx, [master_id])[0]
        env_masks = self._build_conductor_masks(window_idx, slave_ids)

        local_map_np, _local_counts, _actual_to_local = self.base._build_window_local_state(window_idx)  # pylint: disable=protected-access
        other_masks = (local_map_np > 0).astype(np.float32, copy=False)
        other_masks = other_masks.copy()
        other_masks[master_masks > 0] = 0.0

        env_vals = targets.copy()
        env_vals[valid <= 0] = 0.0

        return (
            features,
            master_masks,
            other_masks,
            env_masks,
            env_vals,
        )


class CapBenchTotalDataset(_WindowMaskDatasetBase):
    def __len__(self) -> int:
        return len(self.base.get_window_ids())

    def get_window_sample_ranges(self) -> List[Tuple[int, int]]:
        return [(idx, idx + 1) for idx in range(len(self))]

    def _collect_window_totals(self, window_idx: int) -> Tuple[List[int], np.ndarray]:
        window_samples = self.base._window_samples[window_idx]  # pylint: disable=protected-access
        positive_ids = getattr(window_samples, "positive_ids", None)
        negative_ids = getattr(window_samples, "negative_ids", None)
        targets = getattr(window_samples, "targets", None)

        if isinstance(positive_ids, np.ndarray) and isinstance(negative_ids, np.ndarray) and isinstance(targets, np.ndarray):
            mask = (positive_ids > 0) & (negative_ids <= 0)
            if not np.any(mask):
                return [], np.zeros((0,), dtype=np.float32)

            filtered_ids = positive_ids[mask].astype(np.int32, copy=False)
            filtered_targets = targets[mask].astype(np.float32, copy=False)
            conductor_ids = sorted(int(cid) for cid in np.unique(filtered_ids))
            total_targets = np.array(
                [float(filtered_targets[filtered_ids == cid].mean()) for cid in conductor_ids],
                dtype=np.float32,
            )
            return conductor_ids, total_targets

        totals_by_cid: OrderedDict[int, List[float]] = OrderedDict()
        for sample in window_samples:
            if len(sample.positive_ids) != 1 or len(sample.negative_ids) != 0:
                continue
            conductor_id = int(sample.positive_ids[0])
            totals_by_cid.setdefault(conductor_id, []).append(float(sample.target_value))

        conductor_ids = sorted(totals_by_cid.keys())
        total_targets = np.array(
            [float(np.mean(totals_by_cid[cid])) for cid in conductor_ids],
            dtype=np.float32,
        )
        return conductor_ids, total_targets

    def __getitem__(self, index: int):
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range for dataset with {len(self)} windows")

        window_idx = index
        features = self._build_raw_features(window_idx)
        local_map_np, local_counts_np, actual_to_local = self.base._build_window_local_state(window_idx)  # pylint: disable=protected-access
        conductor_ids, targets = self._collect_window_totals(window_idx)
        if len(conductor_ids) == 0:
            raise RuntimeError(
                f"Window {self.base.get_window_ids()[window_idx]} has no self-cap samples after filtering."
            )
        query_local_ids = np.array([actual_to_local.get(int(cid), 0) for cid in conductor_ids], dtype=np.int64)
        valid = np.ones((len(query_local_ids),), dtype=np.float32)
        valid[query_local_ids == 0] = 0.0
        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(local_map_np, dtype=torch.long),
            torch.tensor(local_counts_np, dtype=torch.float32),
            torch.tensor(query_local_ids, dtype=torch.long),
            torch.tensor(targets, dtype=torch.float32),
            torch.tensor(valid, dtype=torch.float32),
        )

    def get_visualization_case(self, index: int):
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range for dataset with {len(self)} windows")

        window_idx = index
        features = self._build_raw_features(window_idx)
        conductor_ids, targets = self._collect_window_totals(window_idx)
        conductor_masks = self._build_conductor_masks(window_idx, conductor_ids)
        if conductor_masks.shape[0] == 0:
            raise RuntimeError(
                f"Window {self.base.get_window_ids()[window_idx]} has no self-cap samples after filtering."
            )

        master_masks = conductor_masks[0]
        local_map_np, _local_counts, _actual_to_local = self.base._build_window_local_state(window_idx)  # pylint: disable=protected-access
        other_masks = (local_map_np > 0).astype(np.float32, copy=False)
        other_masks = other_masks.copy()
        other_masks[master_masks > 0] = 0.0

        return (
            features,
            conductor_masks,
            master_masks,
            other_masks,
            targets,
        )



def save_val_sample_visualization(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    goal: str,
    epoch: int,
    tb_writer: SummaryWriter,
    tb_logdir: Path,
    dpi: int,
) -> Optional[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return "matplotlib is not installed; skipping visualizations"

    if len(dataset) == 0:
        return "validation dataset is empty; skipping visualizations"
    if not hasattr(dataset, "get_visualization_case"):
        return "dataset does not expose get_visualization_case; skipping visualizations"

    model_was_training = model.training
    model.eval()
    with torch.no_grad():
        if goal == "env":
            dens_vec, master_masks, other_masks, env_masks, env_vals = dataset.get_visualization_case(0)
            xs = torch.tensor(dens_vec, dtype=torch.float32, device=device).unsqueeze(0)
            env_masks_t = torch.tensor(env_masks, dtype=torch.float32, device=device)
            env_vals_t = torch.tensor(env_vals, dtype=torch.float32, device=device)
            q_map = F.softplus(model(xs)[0])
            areas = env_masks_t.sum(dim=(1, 2, 3))
            preds_sum = (q_map.unsqueeze(0) * env_masks_t).sum(dim=(1, 2, 3))
            preds = preds_sum
            pred_tile_maps = q_map.detach().cpu().numpy()
            valid = env_vals_t > 0
            if valid.any():
                mean_rel_err = torch.mean(
                    torch.abs((preds[valid] - env_vals_t[valid]) / torch.clamp(env_vals_t[valid].abs(), min=1e-6))
                ).item()
                title = f"env | sample=0 | mean rel err={mean_rel_err:.4f} | targets={int(valid.sum().item())}"
            else:
                title = "env | sample=0 | no valid targets"
            scalar_text = (
                f"pred mean={preds.mean().item():.4g}, target mean={env_vals_t.mean().item():.4g}, "
                f"mask area min/mean/max={areas.min().item():.4g}/{areas.mean().item():.4g}/{areas.max().item():.4g}, "
                f"q_pos min/mean/max={q_map.min().item():.4g}/{q_map.mean().item():.4g}/{q_map.max().item():.4g}"
            )
        else:
            total_case = dataset.get_visualization_case(0)
            if len(total_case) == 5:
                dens_vec, conductor_masks, master_masks, other_masks, targets = total_case
                xs = torch.tensor(dens_vec, dtype=torch.float32, device=device).unsqueeze(0)
                conductor_masks_t = torch.tensor(conductor_masks, dtype=torch.float32, device=device)
                q_map = F.softplus(model(xs)[0])
                targets_t = torch.tensor(targets, dtype=torch.float32, device=device)
                areas = conductor_masks_t.sum(dim=(1, 2, 3))
                preds_sum = (q_map.unsqueeze(0) * conductor_masks_t).sum(dim=(1, 2, 3))
                preds = preds_sum
                pred_tile_maps = q_map.detach().cpu().numpy()
                valid = targets_t.abs() > 0
                if valid.any():
                    mean_rel_err = torch.mean(
                        torch.abs((preds[valid] - targets_t[valid]) / torch.clamp(targets_t[valid].abs(), min=1e-6))
                    ).item()
                    title = (
                        f"total | window=0 | mean rel err={mean_rel_err:.4f} | "
                        f"conductors={int(valid.sum().item())}"
                    )
                else:
                    title = "total | window=0 | no valid targets"
                scalar_text = (
                    f"pred mean={preds.mean().item():.4g}, target mean={targets_t.mean().item():.4g}, "
                    f"mask area min/mean/max={areas.min().item():.4g}/{areas.mean().item():.4g}/{areas.max().item():.4g}, "
                    f"q_pos min/mean/max={q_map.min().item():.4g}/{q_map.mean().item():.4g}/{q_map.max().item():.4g}"
                )
            else:
                dens_vec, master_masks, other_masks, target = total_case
                xs = torch.tensor(dens_vec, dtype=torch.float32, device=device).unsqueeze(0)
                master_masks_t = torch.tensor(master_masks, dtype=torch.float32, device=device)
                q_map = F.softplus(model(xs)[0])
                area = master_masks_t.sum()
                pred_sum = (q_map * master_masks_t).sum()
                pred_scalar = pred_sum / area.clamp(min=1.0)
                pred_tile_maps = q_map.detach().cpu().numpy()
                target_t = torch.tensor(float(target), dtype=torch.float32, device=device)
                rel_err = torch.abs((pred_scalar - target_t) / torch.clamp(target_t.abs(), min=1e-6)).item()
                title = f"total | sample=0 | rel err={rel_err:.4f}"
                scalar_text = (
                    f"pred mean={pred_scalar.item():.4g}, target={target_t.item():.4g}, "
                    f"mask area={area.item():.4g}"
                )

    if model_was_training:
        model.train()

    num_layers = int(pred_tile_maps.shape[0])
    layer_labels = [f"L{i}" for i in range(num_layers)]
    if hasattr(dataset, "layer_name_map"):
        inv_map = {idx: name for name, idx in dataset.layer_name_map.items()}
        layer_labels = [inv_map.get(i, f"L{i}") for i in range(num_layers)]

    master_masks = (master_masks > 0).astype(np.float32)
    other_masks = (other_masks > 0).astype(np.float32)
    vmin = float(np.min(pred_tile_maps))
    vmax = float(np.max(pred_tile_maps))
    if abs(vmax - vmin) < 1e-12:
        vmax = vmin + 1e-12

    fig, axes = plt.subplots(num_layers, 2, figsize=(14, max(4, 3.2 * num_layers)), squeeze=False)
    for layer_idx in range(num_layers):
        overlay = np.zeros((master_masks.shape[1], master_masks.shape[2], 3), dtype=np.float32)
        overlay[..., 0] = master_masks[layer_idx]
        overlay[..., 2] = other_masks[layer_idx]
        axes[layer_idx, 0].imshow(overlay, interpolation="nearest")
        axes[layer_idx, 0].set_title(f"{layer_labels[layer_idx]} masks: master (red), others (blue)")
        axes[layer_idx, 0].set_axis_off()

        img = axes[layer_idx, 1].imshow(pred_tile_maps[layer_idx], cmap="viridis", vmin=vmin, vmax=vmax)
        axes[layer_idx, 1].set_title(f"{layer_labels[layer_idx]} raw q_map per-tile values")
        axes[layer_idx, 1].set_axis_off()
        fig.colorbar(img, ax=axes[layer_idx, 1], fraction=0.046, pad=0.04)
    fig.suptitle(title, fontsize=12)
    fig.text(0.5, 0.01, scalar_text, ha="center", va="bottom", fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))

    out_dir = tb_logdir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / f"val_sample0_epoch_{epoch:04d}.png"
    fig.savefig(fig_path, dpi=dpi)
    tb_writer.add_figure("viz/val_sample0", fig, epoch)
    plt.close(fig)
    return None


def _reduce_qmap_to_queries(
    q_map: torch.Tensor,
    local_map: torch.Tensor,
    local_counts: torch.Tensor,
    query_local_ids: torch.Tensor,
    reduction: str = "mean",
) -> Tuple[torch.Tensor, torch.Tensor]:
    flat_q = q_map.reshape(q_map.shape[0], -1)
    flat_ids = local_map.reshape(local_map.shape[0], -1).long()
    sums = torch.zeros(
        (q_map.shape[0], local_counts.shape[1]),
        dtype=q_map.dtype,
        device=q_map.device,
    )
    sums.scatter_add_(1, flat_ids, flat_q)
    if reduction == "mean":
        reduced = sums / local_counts.clamp(min=1.0)
    elif reduction == "sum":
        reduced = sums
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")
    qids = query_local_ids.long()
    preds = torch.gather(reduced, 1, qids)
    areas = torch.gather(local_counts, 1, qids)
    return preds, areas


def run_epoch_env(
    loader: DataLoader,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    args: argparse.Namespace,
    report_ratio: float,
    *,
    train: bool,
) -> EpochMetrics:
    losses = AverageMeter()
    maxerr = 0.0
    errs: List[float] = []

    if train:
        model.train()
    else:
        model.eval()

    with torch.set_grad_enabled(train):
        for batch_idx, (xs, local_map, local_counts, query_local_ids, targets, valid) in enumerate(
            tqdm(loader, desc="Train" if train else "Val")
        ):
            xs = xs.to(device, non_blocking=True)
            local_map = local_map.to(device, non_blocking=True)
            local_counts = local_counts.to(device, non_blocking=True)
            query_local_ids = query_local_ids.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            valid = valid.to(device, non_blocking=True)

            q_map = F.softplus(model(xs))
            preds, area = _reduce_qmap_to_queries(
                q_map,
                local_map,
                local_counts,
                query_local_ids,
                reduction="sum",
            )
            targets = targets.abs()

            if args.debug and batch_idx == 0:
                with torch.no_grad():
                    print(
                        f"[debug] targets min/mean/max: "
                        f"{targets.min().item():.4g} {targets.mean().item():.4g} {targets.max().item():.4g}"
                    )
                    print(
                        f"[debug] mask area min/mean/max: "
                        f"{area.min().item():.4g} {area.mean().item():.4g} {area.max().item():.4g}"
                    )
                    print(
                        f"[debug] q_map pos min/mean/max: "
                        f"{q_map.min().item():.4g} {q_map.mean().item():.4g} {q_map.max().item():.4g}"
                    )

            loss = compute_loss(preds, targets, valid, args.loss)
            losses.update(loss.item(), xs.size(0))

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            denom = torch.clamp(targets.abs(), min=1e-6)
            rel_err = torch.abs((preds - targets) / denom)
            err = rel_err[valid.bool()].detach().cpu().numpy().reshape(-1)
            if err.size:
                maxerr = max(maxerr, float(err.max()))
                errs += err.tolist()

    return _finalize_epoch_metrics(losses, maxerr, errs, report_ratio)


def run_epoch_total(
    loader: DataLoader,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    args: argparse.Namespace,
    report_ratio: float,
    *,
    train: bool,
) -> EpochMetrics:
    losses = AverageMeter()
    maxerr = 0.0
    errs: List[float] = []

    if train:
        model.train()
    else:
        model.eval()

    with torch.set_grad_enabled(train):
        for xs, local_map, local_counts, query_local_ids, targets, valid in tqdm(loader, desc="Train" if train else "Val"):
            xs = xs.to(device, non_blocking=True)
            local_map = local_map.to(device, non_blocking=True)
            local_counts = local_counts.to(device, non_blocking=True)
            query_local_ids = query_local_ids.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            valid = valid.to(device, non_blocking=True)

            q_map = F.softplus(model(xs))
            preds, _area = _reduce_qmap_to_queries(
                q_map,
                local_map,
                local_counts,
                query_local_ids,
                reduction="sum",
            )

            loss = compute_loss(preds, targets, valid, args.loss)
            losses.update(loss.item(), xs.size(0))

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            denom = torch.clamp(targets.abs(), min=1e-6)
            rel_err = torch.abs((preds - targets) / denom)
            err = rel_err[valid.bool()].detach().cpu().numpy().reshape(-1)
            if err.size:
                maxerr = max(maxerr, float(err.max()))
                errs += err.tolist()

    return _finalize_epoch_metrics(losses, maxerr, errs, report_ratio)


def run_epoch_legacy_env(
    loader: DataLoader,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    args: argparse.Namespace,
    report_ratio: float,
    *,
    train: bool,
) -> EpochMetrics:
    losses = AverageMeter()
    maxerr = 0.0
    errs: List[float] = []

    if train:
        model.train()
    else:
        model.eval()

    with torch.set_grad_enabled(train):
        for batch_idx, (xs, env_masks, targets, valid) in enumerate(tqdm(loader, desc="Train" if train else "Val")):
            xs = xs.to(device, non_blocking=True)
            env_masks = env_masks.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            valid = valid.to(device, non_blocking=True)

            q_map = F.softplus(model(xs))
            areas = env_masks.sum(dim=(2, 3, 4))
            # Match the historical grouped-U-Net coupling semantics: integrate q over the queried conductor mask.
            preds = (q_map.unsqueeze(1) * env_masks).sum(dim=(2, 3, 4))
            targets = targets.abs()

            if args.debug and batch_idx == 0:
                with torch.no_grad():
                    print(
                        f"[debug] legacy env targets min/mean/max: "
                        f"{targets.min().item():.4g} {targets.mean().item():.4g} {targets.max().item():.4g}"
                    )
                    print(
                        f"[debug] legacy env areas min/mean/max: "
                        f"{areas.min().item():.4g} {areas.mean().item():.4g} {areas.max().item():.4g}"
                    )
                    print(
                        f"[debug] legacy env q_map min/mean/max: "
                        f"{q_map.min().item():.4g} {q_map.mean().item():.4g} {q_map.max().item():.4g}"
                    )

            loss = compute_loss(preds, targets, valid, args.loss)
            losses.update(loss.item(), xs.size(0))

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            denom = torch.clamp(targets.abs(), min=1e-6)
            rel_err = torch.abs((preds - targets) / denom)
            err = rel_err[valid.bool()].detach().cpu().numpy().reshape(-1)
            if err.size:
                maxerr = max(maxerr, float(err.max()))
                errs += err.tolist()

    return _finalize_epoch_metrics(losses, maxerr, errs, report_ratio)


def run_epoch_legacy_total(
    loader: DataLoader,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    args: argparse.Namespace,
    report_ratio: float,
    *,
    train: bool,
) -> EpochMetrics:
    losses = AverageMeter()
    maxerr = 0.0
    errs: List[float] = []

    if train:
        model.train()
    else:
        model.eval()

    with torch.set_grad_enabled(train):
        for xs, master_masks, targets in tqdm(loader, desc="Train" if train else "Val"):
            xs = xs.to(device, non_blocking=True)
            master_masks = master_masks.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            q_map = F.softplus(model(xs))
            areas = master_masks.sum(dim=(1, 2, 3))
            preds_sum = (q_map * master_masks).sum(dim=(1, 2, 3))
            preds = preds_sum / areas.clamp(min=1.0)

            valid = torch.ones_like(targets)
            loss = compute_loss(preds, targets, valid, args.loss)
            losses.update(loss.item(), xs.size(0))

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            denom = torch.clamp(targets.abs(), min=1e-6)
            rel_err = torch.abs((preds - targets) / denom)
            err = rel_err.detach().cpu().numpy().reshape(-1)
            if err.size:
                maxerr = max(maxerr, float(err.max()))
                errs += err.tolist()

    return _finalize_epoch_metrics(losses, maxerr, errs, report_ratio)


def run_epoch_scalar(
    loader: DataLoader,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    args: argparse.Namespace,
    report_ratio: float,
    *,
    train: bool,
) -> EpochMetrics:
    losses = AverageMeter()
    maxerr = 0.0
    errs: List[float] = []

    if train:
        model.train()
    else:
        model.eval()

    with torch.set_grad_enabled(train):
        for batch_idx, batch in enumerate(tqdm(loader, desc="Train" if train else "Val")):
            xs, targets = batch[0], batch[1]
            xs = xs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).reshape(xs.shape[0], -1)
            if args.goal == "env":
                targets = targets.abs()

            preds = model(xs).reshape_as(targets)
            valid = torch.ones_like(targets)

            if args.debug and batch_idx == 0:
                with torch.no_grad():
                    print(
                        f"[debug] scalar preds min/mean/max: "
                        f"{preds.min().item():.4g} {preds.mean().item():.4g} {preds.max().item():.4g}"
                    )
                    print(
                        f"[debug] scalar targets min/mean/max: "
                        f"{targets.min().item():.4g} {targets.mean().item():.4g} {targets.max().item():.4g}"
                    )

            loss = compute_loss(preds, targets, valid, args.loss)
            losses.update(loss.item(), xs.size(0))

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            denom = torch.clamp(targets.abs(), min=1e-6)
            rel_err = torch.abs((preds - targets) / denom)
            err = rel_err.detach().cpu().numpy().reshape(-1)
            if err.size:
                maxerr = max(maxerr, float(err.max()))
                errs += err.tolist()

    return _finalize_epoch_metrics(losses, maxerr, errs, report_ratio)


def resolve_window_ids(
    args: argparse.Namespace,
    window_dir: Path,
    spef_dir: Optional[Path],
) -> List[str]:
    capbench_modules = _load_capbench_modules()
    effective_max_windows = int(args.max_windows) if args.max_windows > 0 else None

    if args.window_ids:
        window_ids = list(args.window_ids)
        if effective_max_windows is None:
            return window_ids
        return window_ids[:effective_max_windows]

    return capbench_modules.window_dataset_cls.discover_limited_windows(
        window_dir=window_dir,
        max_windows=effective_max_windows,
        spef_dir=spef_dir,
    )


def main() -> float:
    parser = argparse.ArgumentParser(description="3D training on CapBench or legacy CNNCap datasets")
    parser.add_argument("--lr", "--learning-rate", default=3e-4, type=float, metavar="LR", dest="lr")
    parser.add_argument("--epoch", default=100, type=int, help="number of epochs")
    parser.add_argument("--seed", default=11037, type=int, help="random seed")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=False,
        help="Enable deterministic cuDNN kernels (slower, but reproducible)",
    )
    parser.add_argument("--batch_size", "--bs", default=16, type=int, help="batch size")
    parser.add_argument(
        "--num-workers",
        default=1,
        type=int,
        help="DataLoader worker count",
    )
    parser.add_argument(
        "--prefetch-factor",
        default=None,
        type=int,
        help="DataLoader prefetch factor (default: 2 for CapBench, 4 for legacy; ignored when num_workers=0)",
    )
    parser.add_argument(
        "--window-cache-size",
        default=2,
        type=int,
        help="CapBench only: number of window bundles cached per worker process",
    )
    parser.add_argument(
        "--binary-hdf5-cache-dir",
        type=str,
        default=None,
        help="CapBench U-Net only: optional directory for a binary-occupancy HDF5 cache",
    )
    parser.add_argument(
        "--binary-hdf5-compression",
        choices=["none", "lzf", "gzip"],
        default="lzf",
        help="Compression for the optional binary HDF5 cache",
    )
    parser.add_argument(
        "--binary-hdf5-chunk-windows",
        type=int,
        default=8,
        help="Number of windows per chunk in the optional binary HDF5 cache",
    )
    parser.add_argument(
        "--rebuild-binary-hdf5-cache",
        action="store_true",
        help="Rebuild the optional binary HDF5 cache even when a valid cache exists",
    )
    parser.add_argument("--logfile", default="log/log_train.txt", type=str, help="log file path")
    parser.add_argument("--savename", type=str, default="model_3d.pth", help="checkpoint name")
    parser.add_argument("--save-dir", type=str, default="saved_models", help="checkpoint output directory")
    parser.add_argument("--log", action="store_true", default=False, help="train on log(target) instead of raw target")
    parser.add_argument(
        "--loss",
        default="msre",
        choices=LOSS_CHOICES,
        help="Training loss: baseline MSRE, optional MLSE (also accepts the 'msle' alias), or legacy raw log-space MSE.",
    )
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["unet", "cnncap_unet", *RESNET_MODEL_TYPES],
        default="unet",
    )
    parser.add_argument("--base-ch", type=int, default=16, help="base channel width for U-Net")
    parser.add_argument("--depth", type=int, default=3, help="U-Net depth (number of downsampling stages)")
    parser.add_argument(
        "--monai-config",
        "--unet-model-id",
        dest="monai_config",
        type=str,
        choices=sorted(MONAI_ABLATION_CONFIGS.keys()),
        default=None,
        help=(
            "Predefined U-Net / AttentionUnet ablation config. "
            "Overrides manual MONAI shape args."
        ),
    )
    parser.add_argument(
        "--monai-depth",
        type=int,
        default=None,
        help="MONAI U-Net depth (used when --monai-config is not set).",
    )
    parser.add_argument(
        "--monai-channels",
        type=str,
        default=None,
        help="Comma-separated MONAI channels tuple, for example 32,64,128,256,512",
    )
    parser.add_argument(
        "--monai-strides",
        type=str,
        default=None,
        help="Comma-separated MONAI strides tuple, for example 2,2,2,2",
    )
    parser.add_argument(
        "--monai-num-res-units",
        type=int,
        default=0,
        help="MONAI num_res_units value.",
    )
    parser.add_argument(
        "--monai-dropout",
        type=float,
        default=0.0,
        help="MONAI dropout rate (used when --monai-config is not set).",
    )
    parser.add_argument(
        "--monai-act",
        type=str,
        default=None,
        help="MONAI activation, e.g. PRELU or LEAKYRELU:0.1 (used when --monai-config is not set).",
    )
    parser.add_argument(
        "--monai-norm",
        type=str,
        default=None,
        help="MONAI normalization, e.g. INSTANCE or GROUP (used when --monai-config is not set).",
    )
    parser.add_argument(
        "--monai-kernel-size",
        type=int,
        default=3,
        help="MONAI convolution kernel size (used when --monai-config is not set).",
    )
    parser.add_argument(
        "--monai-adn-ordering",
        type=str,
        default="NDA",
        help="MONAI ADN ordering string, e.g. NDA or AND (used when --monai-config is not set).",
    )
    parser.add_argument(
        "--goal",
        type=str,
        default="total",
        choices=["total", "env", "coupling"],
        help="Training target family. 'coupling' is accepted as an alias for 'env'.",
    )
    parser.add_argument(
        "--dataset-format",
        type=str,
        default="capbench",
        choices=list(DATASET_FORMAT_CHOICES),
        help="Dataset layout to load: current CapBench windows or legacy CNNCap layer grids.",
    )
    parser.add_argument("--wd", "--weight-decay", default=1e-4, type=float, dest="weight_decay")
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=5,
        help="Linear warmup epochs before cosine decay",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=None,
        help="Minimum learning-rate floor after warmup. Defaults to 10%% of --lr when omitted.",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--report-ratio",
        type=float,
        default=None,
        help="override ratio threshold for reporting error share",
    )
    parser.add_argument("--tb-logdir", type=str, default="runs", help="TensorBoard log directory")
    parser.add_argument("--debug", action="store_true", default=False, help="print one-batch diagnostics")
    parser.add_argument("--tech", type=str, default="tech/nangate45.yaml", help="Optional technology stack YAML file")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="datasets/nangate45/small",
        help="Dataset selector or root path. For cnncap_legacy, this is the legacy dataset root.",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default=legacy_cnncap_dataset.DEFAULT_LAYERS,
        help="Legacy CNNCap only: underscore-separated layer names to load.",
    )
    parser.add_argument(
        "--train-label",
        type=str,
        default=None,
        help="Legacy CNNCap only: optional override for the train label file.",
    )
    parser.add_argument(
        "--val-label",
        type=str,
        default=None,
        help="Legacy CNNCap only: optional override for the validation label file.",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=legacy_cnncap_dataset.DEFAULT_PADDING,
        help="Legacy CNNCap only: extra padding added around each cropped density window.",
    )
    parser.add_argument(
        "--window-dir",
        type=str,
        default=None,
        help=(
            "Directory containing window bundle directories "
            "(default: <dataset-path>/density_maps for CapBench models)"
        ),
    )
    parser.add_argument(
        "--spef-dir",
        type=str,
        default=None,
        help="Directory containing window SPEF files (default: <dataset-path>/labels_rwcap)",
    )
    parser.add_argument("--window-ids", type=str, nargs="*", help="Specific window IDs to use")
    parser.add_argument(
        "--max-windows",
        type=int,
        default=0,
        help="Maximum number of windows to use (default: 0 = use all available windows).",
    )
    parser.add_argument("--val-split", type=float, default=0.2, help="Fraction of windows reserved for validation")
    parser.add_argument(
        "--build-workers",
        type=int,
        default=2,
        help="Workers for dataset preprocessing (default: 2)",
    )
    parser.add_argument(
        "--labels-solver",
        type=str,
        default="rwcap",
        choices=["auto", "rwcap", "raphael"],
        help="Preferred SPEF solver when multiple label sources exist",
    )
    parser.add_argument(
        "--highlight-scale",
        type=float,
        default=1.0,
        help="Density boost applied to highlighted conductors (kept for compatibility)",
    )
    parser.add_argument(
        "--debug-layer-dir",
        type=str,
        default="",
        help="Directory to dump first-sample density plots (empty string to disable)",
    )
    parser.add_argument(
        "--debug-layer-count",
        type=int,
        default=8,
        help="Number of layers to visualize per conductor in debug plots",
    )
    parser.add_argument(
        "--debug-conductor-count",
        type=int,
        default=5,
        help="Number of conductors to visualize in debug plots",
    )

    args = parser.parse_args()
    if args.window_cache_size < 0:
        raise ValueError("--window-cache-size must be >= 0")
    if args.prefetch_factor is not None and args.prefetch_factor <= 0:
        raise ValueError("--prefetch-factor must be > 0 when provided")
    if args.binary_hdf5_chunk_windows <= 0:
        raise ValueError("--binary-hdf5-chunk-windows must be > 0")
    args.goal = _normalize_goal(args.goal)
    print(args)

    if args.log:
        raise ValueError("Log-domain training is not supported in grouped coupling mode.")

    set_seed(args.seed, deterministic=args.deterministic)
    device = get_device(args.device)
    print(f"Using device: {device}")

    log_path = Path(args.logfile)
    if log_path.parent:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    logfile = open(log_path, "w", encoding="utf-8")

    run_name = f"{args.goal}_seed{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    tb_logdir = Path(args.tb_logdir) / run_name
    tb_writer = SummaryWriter(log_dir=str(tb_logdir))

    scalar_model = is_scalar_model(args.model_type)
    enable_visualization = not scalar_model
    dataset_format = str(args.dataset_format)
    dataset_path: Path | None = None
    window_dir: Path | None = None
    spef_dir: Path | None = None
    extra_checkpoint_metadata: Dict[str, object] = {}

    if dataset_format == "capbench":
        capbench_modules = _load_capbench_modules()
        default_window_subdir = "density_maps"
        required_artifacts = []
        if args.window_dir is None:
            required_artifacts.append(default_window_subdir)
        if args.spef_dir is None:
            required_artifacts.append("labels_rwcap")
        dataset_path = _resolve_dataset_root(args.dataset_path, required_artifacts=required_artifacts)
        window_dir = Path(args.window_dir).resolve() if args.window_dir else dataset_path / default_window_subdir
        spef_dir = Path(args.spef_dir).resolve() if args.spef_dir else None
        if not window_dir.exists():
            raise FileNotFoundError(f"Window directory not found: {window_dir}")
        if spef_dir is not None and not spef_dir.exists():
            raise FileNotFoundError(f"SPEF directory not found: {spef_dir}")

        window_ids = resolve_window_ids(args, window_dir, spef_dir)
        if not window_ids:
            raise RuntimeError("No window IDs were found for training.")
        window_limit_note = "" if args.max_windows <= 0 else f" (limited by --max-windows={args.max_windows})"
        print(
            f"Resolved {len(window_ids)} windows from {window_dir}{window_limit_note}"
        )

        dataset_goal = "self" if args.goal == "total" else "coupling"
        if args.window_dir is None and spef_dir is None:
            dataset_loader = (
                capbench_modules.load_density_id_window_dataset
                if not scalar_model
                else capbench_modules.load_density_window_dataset
            )
            full_dataset = dataset_loader(
                args.dataset_path,
                goal=dataset_goal,
                solver_preference=args.labels_solver,
                window_ids=window_ids,
                build_workers=args.build_workers,
                highlight_scale=args.highlight_scale,
                window_cache_size=args.window_cache_size,
            )
        else:
            dataset_cls = capbench_modules.id_map_dataset_cls if not scalar_model else capbench_modules.window_dataset_cls
            dataset_kwargs = dict(
                window_dir=window_dir,
                spef_dir=spef_dir,
                window_ids=window_ids,
                goal=dataset_goal,
                highlight_scale=args.highlight_scale,
                solver_preference=args.labels_solver,
                build_workers=args.build_workers,
                window_cache_size=args.window_cache_size,
            )
            if not scalar_model and window_dir.name == "density_maps":
                dataset_kwargs["trim_margin"] = True
            full_dataset = dataset_cls(**dataset_kwargs)

        if args.binary_hdf5_cache_dir is not None:
            if scalar_model:
                raise ValueError("--binary-hdf5-cache-dir is only valid for binary-occupancy U-Net models")
            binary_cache_spec = build_binary_cache_spec(
                full_dataset,
                selector=args.dataset_path,
                source_root=dataset_path,
                trim_margin=bool(getattr(full_dataset, "_trim_margin", False)),
            )
            binary_cache_path = ensure_binary_hdf5_cache(
                full_dataset,
                binary_cache_spec,
                BinaryHdf5CacheConfig(
                    cache_dir=Path(args.binary_hdf5_cache_dir),
                    compression=args.binary_hdf5_compression,
                    chunk_windows=args.binary_hdf5_chunk_windows,
                    rebuild=args.rebuild_binary_hdf5_cache,
                ),
            )
            full_dataset = BinaryHdf5CapBenchDataset(full_dataset, binary_cache_path)
            print(f"Using binary-occupancy HDF5 cache: {binary_cache_path}")

        target_summary = summarize_loaded_targets(full_dataset)
        print(target_summary)
        logfile.write(f"{target_summary}\n")
        logfile.flush()

        if args.debug_layer_dir:
            debug_dir = Path(args.debug_layer_dir).resolve()
            print(f"Writing debug density plots to {debug_dir}")
            try:
                full_dataset.dump_layer_debug_visuals(
                    debug_dir,
                    num_conductors=max(1, args.debug_conductor_count),
                    num_layers=max(1, args.debug_layer_count),
                )
            except Exception as exc:
                print(f"WARNING: Failed to generate debug density plots: {exc}")

        if len(full_dataset) < 2:
            raise RuntimeError("Not enough samples to split into train/validation sets.")

        val_split = max(0.0, min(0.9, float(args.val_split)))
        train_ratio = 1.0 - val_split
        train_base, val_base, test_base = capbench_modules.create_window_level_splits(
            full_dataset,
            train_ratio=train_ratio,
            val_ratio=val_split,
            test_ratio=0.0,
            random_seed=WINDOW_SPLIT_SEED,
        )
        capbench_modules.verify_no_data_leakage(train_base, val_base, test_base)
        print(f"Window split seed: {WINDOW_SPLIT_SEED} (fixed)")

        optimize_capbench_window_storage(
            full_dataset,
            train_base,
            val_base,
            test_base,
            drop_id_maps=(args.goal == "total" and VIZ_EVERY <= 0 and not args.debug_layer_dir),
        )

        if args.goal == "env":
            report_ratio = 0.1 if args.report_ratio is None else args.report_ratio
            if scalar_model:
                train_dataset = train_base
                val_dataset = val_base
                collate_fn = None
                run_epoch_fn = run_epoch_scalar
            else:
                train_dataset = CapBenchCouplingDataset(train_base)
                val_dataset = CapBenchCouplingDataset(val_base)
                collate_fn = collate_grouped
                run_epoch_fn = run_epoch_env
        else:
            report_ratio = 0.05 if args.report_ratio is None else args.report_ratio
            if scalar_model:
                train_dataset = train_base
                val_dataset = val_base
                collate_fn = None
                run_epoch_fn = run_epoch_scalar
            else:
                train_dataset = CapBenchTotalDataset(train_base)
                val_dataset = CapBenchTotalDataset(val_base)
                collate_fn = collate_total
                run_epoch_fn = run_epoch_total

        num_input_channels = full_dataset.num_layers
        active_layers = full_dataset.active_layers
        extra_checkpoint_metadata = {
            "window_dir": str(window_dir),
            "spef_dir": None if spef_dir is None else str(spef_dir),
            "labels_solver": str(args.labels_solver),
        }
    else:
        dataset_path = Path(args.dataset_path).expanduser().resolve()
        if not dataset_path.exists():
            raise FileNotFoundError(f"Legacy CNNCap dataset root not found: {dataset_path}")

        train_label = (
            Path(args.train_label).expanduser().resolve()
            if args.train_label
            else legacy_cnncap_dataset.resolve_label_path(dataset_path, args.layers, args.goal, "train")
        )
        val_label = (
            Path(args.val_label).expanduser().resolve()
            if args.val_label
            else legacy_cnncap_dataset.resolve_label_path(dataset_path, args.layers, args.goal, "val")
        )
        if not train_label.exists():
            raise FileNotFoundError(f"Legacy CNNCap train label file not found: {train_label}")
        if not val_label.exists():
            raise FileNotFoundError(f"Legacy CNNCap validation label file not found: {val_label}")

        train_lines = train_label.read_text(encoding="utf-8").strip().splitlines()
        val_lines = val_label.read_text(encoding="utf-8").strip().splitlines()
        target_summary = legacy_cnncap_dataset.summarize_targets([*train_lines, *val_lines], args.goal)
        print(target_summary)
        logfile.write(f"{target_summary}\n")
        logfile.flush()

        if args.goal == "env":
            report_ratio = 0.1 if args.report_ratio is None else args.report_ratio
            if scalar_model:
                train_dataset = legacy_cnncap_dataset.ScalarCouplingDataset(
                    args.layers,
                    train_lines,
                    dataset_path,
                    padding=args.padding,
                )
                val_dataset = legacy_cnncap_dataset.ScalarCouplingDataset(
                    args.layers,
                    val_lines,
                    dataset_path,
                    padding=args.padding,
                )
                collate_fn = None
                run_epoch_fn = run_epoch_scalar
            else:
                train_dataset = legacy_cnncap_dataset.CouplingDataset(
                    args.layers,
                    train_lines,
                    dataset_path,
                    padding=args.padding,
                )
                val_dataset = legacy_cnncap_dataset.CouplingDataset(
                    args.layers,
                    val_lines,
                    dataset_path,
                    padding=args.padding,
                )
                collate_fn = legacy_cnncap_dataset.collate_grouped
                run_epoch_fn = run_epoch_legacy_env
        else:
            report_ratio = 0.05 if args.report_ratio is None else args.report_ratio
            if scalar_model:
                train_dataset = legacy_cnncap_dataset.ScalarTotalDataset(
                    args.layers,
                    train_lines,
                    dataset_path,
                    padding=args.padding,
                )
                val_dataset = legacy_cnncap_dataset.ScalarTotalDataset(
                    args.layers,
                    val_lines,
                    dataset_path,
                    padding=args.padding,
                )
                collate_fn = None
                run_epoch_fn = run_epoch_scalar
            else:
                train_dataset = legacy_cnncap_dataset.TotalDataset(
                    args.layers,
                    train_lines,
                    dataset_path,
                    padding=args.padding,
                )
                val_dataset = legacy_cnncap_dataset.TotalDataset(
                    args.layers,
                    val_lines,
                    dataset_path,
                    padding=args.padding,
                )
                collate_fn = legacy_cnncap_dataset.collate_total
                run_epoch_fn = run_epoch_legacy_total

        active_layers = [layer for layer in args.layers.split("_") if layer]
        num_input_channels = len(active_layers)
        extra_checkpoint_metadata = {
            "window_dir": str(dataset_path),
            "spef_dir": None,
            "labels_solver": None,
            "legacy_layers": str(args.layers),
            "train_label": str(train_label),
            "val_label": str(val_label),
            "padding": int(args.padding),
        }

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise RuntimeError(
            f"Empty split produced: train={len(train_dataset)} samples, val={len(val_dataset)} samples. "
            "Adjust the dataset or split settings."
        )

    print(f"Dataset format: {dataset_format}")
    print(f"Model type: {args.model_type}")
    print(f"Active dataset layers ({num_input_channels}): {active_layers}")
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    monai_cfg: Optional[dict] = None
    if args.model_type == "unet":
        monai_cfg = resolve_monai_unet_config(
            args,
            fallback_base_ch=args.base_ch,
            fallback_depth=args.depth,
        )
        print(
            "MONAI config:",
            f"arch={monai_cfg['arch']}",
            f"channels={monai_cfg['channels']}",
            f"strides={monai_cfg['strides']}",
            f"num_res_units={monai_cfg['num_res_units']}",
            f"dropout={monai_cfg['dropout']}",
            f"act={monai_cfg['act']}",
            f"norm={monai_cfg['norm']}",
            f"kernel_size={monai_cfg['kernel_size']}",
            f"adn_ordering={monai_cfg['adn_ordering']}",
            f"config_id={args.monai_config or 'custom'}",
        )

    model = get_model(
        args.model_type,
        num_input_channels=num_input_channels,
        base_ch=args.base_ch,
        depth=args.depth,
        monai_config=monai_cfg,
    ).to(device)

    loader_kwargs = {}
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        if args.prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = args.prefetch_factor
        elif dataset_format == "capbench":
            loader_kwargs["prefetch_factor"] = 2
        else:
            loader_kwargs["prefetch_factor"] = 4

    loader_common_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": collate_fn,
        **loader_kwargs,
    }

    if dataset_format == "capbench":
        train_batch_sampler = capbench_modules.make_window_grouped_batch_sampler(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            seed=args.seed,
        )
        val_batch_sampler = capbench_modules.make_window_grouped_batch_sampler(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            seed=args.seed,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            **loader_common_kwargs,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_sampler=val_batch_sampler,
            **loader_common_kwargs,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            **loader_common_kwargs,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            **loader_common_kwargs,
        )

    optimizer = build_optimizer(model, args)

    logfile.write(f"{args}\n")
    logfile.flush()

    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(Path(args.resume), model, optimizer, device)
    if args.pretrained and not args.resume:
        load_checkpoint(Path(args.pretrained), model, None, device)

    run_stats: Optional[UnetRunStats] = None
    run_description_stats: Optional[str] = None
    if not scalar_model:
        run_stats = collect_unet_run_stats(model, device, train_dataset)
        run_description_stats = format_unet_run_stats(run_stats, device=device)
        print(run_description_stats)
        logfile.write(f"{run_description_stats}\n")
        logfile.flush()

    tb_writer.add_text("run/description", build_tensorboard_description(args, run_description_stats), 0)

    best_val_loss = float("inf")
    best_val_avgerr = float("inf")
    best_val_avgerr_epoch = -1
    best_val_ratio_gt_5pct = float("inf")
    best_val_ratio_gt_5pct_epoch = -1
    best_val_ratio_gt_10pct = float("inf")
    best_val_ratio_gt_10pct_epoch = -1
    best_checkpoint_path: Optional[Path] = None
    save_dir = Path(args.save_dir)
    viz_disabled_reason: Optional[str] = None
    checkpoint_metadata = {
        "goal": str(args.goal),
        "model_type": str(args.model_type),
        "monai_config": None if args.model_type != "unet" else str(args.monai_config or "custom"),
        "monai_arch": None if monai_cfg is None else monai_cfg.get("arch", "unet"),
        "num_input_channels": int(num_input_channels),
        "active_layers": list(active_layers),
        "dataset_format": dataset_format,
        "savename": str(args.savename),
    }
    checkpoint_metadata.update(extra_checkpoint_metadata)

    for epoch in range(start_epoch, args.epoch):
        current_lr = apply_epoch_learning_rate(
            optimizer,
            base_lr=args.lr,
            epoch_idx=epoch,
            total_epochs=args.epoch,
            warmup_epochs=args.warmup_epochs,
            min_lr=args.min_lr,
        )
        train_metrics = run_epoch_fn(
            train_loader,
            model,
            optimizer,
            device,
            args,
            report_ratio,
            train=True,
        )
        val_metrics = run_epoch_fn(
            val_loader,
            model,
            None,
            device,
            args,
            report_ratio,
            train=False,
        )

        is_best = val_metrics.avgerr < best_val_avgerr
        best_val_loss = min(best_val_loss, val_metrics.loss)
        best_copy_path: Optional[Path] = None
        if val_metrics.avgerr < best_val_avgerr:
            best_val_avgerr = val_metrics.avgerr
            best_val_avgerr_epoch = epoch
            save_stem = Path(args.savename).stem
            best_filename = f"{save_stem}_best_mare_{best_val_avgerr:.6f}.pth"
            best_copy_path = save_dir / best_filename
            if best_checkpoint_path is not None and best_checkpoint_path.exists() and best_checkpoint_path != best_copy_path:
                best_checkpoint_path.unlink()
            best_checkpoint_path = best_copy_path
        if val_metrics.ratio_above_5pct < best_val_ratio_gt_5pct:
            best_val_ratio_gt_5pct = val_metrics.ratio_above_5pct
            best_val_ratio_gt_5pct_epoch = epoch
        if val_metrics.ratio_above_10pct < best_val_ratio_gt_10pct:
            best_val_ratio_gt_10pct = val_metrics.ratio_above_10pct
            best_val_ratio_gt_10pct_epoch = epoch
        save_checkpoint(
            model,
            optimizer,
            epoch,
            val_metrics.avgerr,
            is_best,
            save_dir,
            args.savename,
            best_copy_path=best_copy_path,
            metadata=checkpoint_metadata,
        )

        tb_writer.add_scalar("loss/train", train_metrics.loss, epoch)
        tb_writer.add_scalar("loss/val", val_metrics.loss, epoch)
        tb_writer.add_scalar("error/train_max", train_metrics.maxerr, epoch)
        tb_writer.add_scalar("error/val_max", val_metrics.maxerr, epoch)
        tb_writer.add_scalar("error/train_avg", train_metrics.avgerr, epoch)
        tb_writer.add_scalar("error/val_avg", val_metrics.avgerr, epoch)
        tb_writer.add_scalar("error/train_ratio", train_metrics.ratio_above_report, epoch)
        tb_writer.add_scalar("error/val_ratio", val_metrics.ratio_above_report, epoch)
        tb_writer.add_scalar("error/train_ratio_gt_5pct", train_metrics.ratio_above_5pct, epoch)
        tb_writer.add_scalar("error/val_ratio_gt_5pct", val_metrics.ratio_above_5pct, epoch)
        tb_writer.add_scalar("error/train_ratio_gt_10pct", train_metrics.ratio_above_10pct, epoch)
        tb_writer.add_scalar("error/val_ratio_gt_10pct", val_metrics.ratio_above_10pct, epoch)
        for i, param_group in enumerate(optimizer.param_groups):
            tb_writer.add_scalar(f"lr/group_{i}", param_group["lr"], epoch)

        if enable_visualization and VIZ_EVERY > 0 and epoch % VIZ_EVERY == 0 and viz_disabled_reason is None:
            viz_disabled_reason = save_val_sample_visualization(
                model=model,
                dataset=val_dataset,
                device=device,
                goal=args.goal,
                epoch=epoch,
                tb_writer=tb_writer,
                tb_logdir=tb_logdir,
                dpi=VIZ_DPI,
            )
            if viz_disabled_reason is not None:
                print(f"[viz] {viz_disabled_reason}")

        msg = (
            f"Epoch {epoch} lr={current_lr:.6g} "
            f"train_loss={train_metrics.loss:.6f} train_maxerr={train_metrics.maxerr:.4f} train_avgerr={train_metrics.avgerr:.4f} "
            f"{_format_epoch_ratio_fields('train', train_metrics, report_ratio)} "
            f"val_loss={val_metrics.loss:.6f} val_maxerr={val_metrics.maxerr:.4f} val_avgerr={val_metrics.avgerr:.4f} "
            f"{_format_epoch_ratio_fields('val', val_metrics, report_ratio)}\n"
        )
        print(msg, end="")
        logfile.write(msg)
        logfile.flush()

    final_msg = (
        f"Best validation loss: {best_val_loss:.6f}\n"
        f"Best validation MARE (val_avgerr): {best_val_avgerr:.6f} "
        f"(epoch={best_val_avgerr_epoch}) | "
        f"best val ratio>5.0%={best_val_ratio_gt_5pct:.4f} (epoch={best_val_ratio_gt_5pct_epoch}) | "
        f"best val ratio>10.0%={best_val_ratio_gt_10pct:.4f} (epoch={best_val_ratio_gt_10pct_epoch})\n"
        f"Best checkpoint path: {best_checkpoint_path if best_checkpoint_path is not None else 'n/a'}\n"
        f"TensorBoard run id: {run_name}\n"
        f"TensorBoard log dir: {tb_logdir}\n"
    )
    run_summary_line = build_run_summary_line(
        best_val_avgerr=best_val_avgerr,
        best_val_ratio_gt_5pct=best_val_ratio_gt_5pct,
        best_val_ratio_gt_10pct=best_val_ratio_gt_10pct,
        run_stats=run_stats,
    )
    print(final_msg, end="")
    print(run_summary_line)
    logfile.write(final_msg)
    logfile.write(f"{run_summary_line}\n")
    logfile.flush()

    tb_writer.close()
    logfile.close()
    return best_val_loss


if __name__ == "__main__":
    main()
