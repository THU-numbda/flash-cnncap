#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import OrderedDict
import importlib
import sys
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
TRAINING_DIR = REPO_ROOT / "training"
FULL_PIPELINE_DIR = REPO_ROOT / "full-pipeline"
for extra_path in (REPO_ROOT, TRAINING_DIR, FULL_PIPELINE_DIR):
    extra_path_str = str(extra_path)
    if extra_path_str not in sys.path:
        sys.path.insert(0, extra_path_str)

from capbench._internal.common.density_window_bundle import (  # pylint: disable=wrong-import-position
    discover_density_window_ids,
    load_density_window_conductor_map,
    load_density_window_density,
    load_density_window_ids,
    load_density_window_meta,
)
from capbench.datasets import resolve_dataset_path  # pylint: disable=wrong-import-position
from capbench.formats.spef.openrcx_to_simple_spef import parse_spef_components  # pylint: disable=wrong-import-position
from capbench.formats.spef.python_parser import load_dnet_totals  # pylint: disable=wrong-import-position
from flash_common.grouped_sparse_reduce import (  # pylint: disable=wrong-import-position
    build_sparse_index_tensors,
    load_sparse_reduce_cuda_extension,
    reduce_qmap_to_all_conductors_sparse,
)
from pipeline_model import (  # pylint: disable=wrong-import-position
    DEFAULT_FULL_PIPELINE_MONAI_CONFIG,
    FULL_PIPELINE_MONAI_CONFIGS,
    build_full_pipeline_unet,
)
import resnet_custom  # pylint: disable=wrong-import-position


DEFAULT_DATASET_SELECTORS = ("nangate45/small", "nangate45/medium", "nangate45/large")
DEFAULT_MAX_WINDOWS = 256
DEFAULT_WARMUP_WINDOWS = DEFAULT_MAX_WINDOWS
DEFAULT_BATCH_SIZE = 48
DEFAULT_WARMUP_BATCHES = 16
DEFAULT_PCT_NPOINTS = 1024
DEFAULT_TRT_OPSET = 18
DEFAULT_SAMPLE_SEED = 11037
DEFAULT_RASTER_WINDOW_CACHE_SIZE = 2
_TENSORRT_LOGGER = None

PCT_VARIANTS = (4, 6, 8, 10)
DEFAULT_BENCHMARK_UNET_CONFIGS = ("D4_B", "D4_D_k5")


@dataclass(frozen=True)
class CanonicalWindow:
    window_id: str
    conductor_ids: np.ndarray


@dataclass(frozen=True)
class RasterWindowMeta:
    window_id: str
    conductor_ids: np.ndarray


@dataclass
class RasterDatasetMeta:
    layer_catalog: Tuple[str, ...]
    max_h: int
    max_w: int
    windows_by_id: Dict[str, RasterWindowMeta]


@dataclass
class ImageWindow:
    window_id: str
    base_features: np.ndarray
    conductor_coords: Dict[int, List[Tuple[int, np.ndarray, np.ndarray]]]
    local_map: torch.Tensor
    local_counts: torch.Tensor


@dataclass
class UnetBatch:
    features: torch.Tensor
    local_counts: torch.Tensor
    sparse_indices: torch.Tensor
    sparse_counts: torch.Tensor


@dataclass(frozen=True)
class ResnetBatchPlan:
    window_indices: Tuple[int, ...]
    conductor_a_ids: Tuple[Optional[int], ...]
    conductor_b_ids: Tuple[Optional[int], ...]


@dataclass(frozen=True)
class UnetBatchPlan:
    window_indices: Tuple[int, ...]
    conductor_a_ids: Tuple[Optional[int], ...]


@dataclass
class RasterWindowData:
    window_id: str
    conductor_ids: np.ndarray
    density_base: np.ndarray
    occupancy_base: np.ndarray
    conductor_coords: Dict[int, List[Tuple[int, np.ndarray, np.ndarray]]]
    local_map: torch.Tensor
    local_counts: torch.Tensor


class LazyRasterWindowStore:
    def __init__(
        self,
        *,
        density_dir: Path,
        metadata: RasterDatasetMeta,
        cache_size: int = DEFAULT_RASTER_WINDOW_CACHE_SIZE,
    ) -> None:
        if cache_size <= 0:
            raise ValueError(f"cache_size must be positive, got {cache_size}")
        self.density_dir = density_dir
        self.metadata = metadata
        self.layer_index = {name: idx for idx, name in enumerate(metadata.layer_catalog)}
        self.num_layers = len(metadata.layer_catalog)
        self.cache_size = int(cache_size)
        self._cache: "OrderedDict[str, RasterWindowData]" = OrderedDict()

    @property
    def feature_shape(self) -> Tuple[int, int, int]:
        return (self.num_layers, int(self.metadata.max_h), int(self.metadata.max_w))

    def get_density_window(self, window_id: str) -> ImageWindow:
        return self._build_image_window(window_id, use_occupancy=False)

    def get_unet_window(self, window_id: str) -> ImageWindow:
        return self._build_image_window(window_id, use_occupancy=True)

    def _build_image_window(self, window_id: str, *, use_occupancy: bool) -> ImageWindow:
        item = self._get_window_data(window_id)
        return ImageWindow(
            window_id=window_id,
            base_features=item.occupancy_base if use_occupancy else item.density_base,
            conductor_coords=item.conductor_coords,
            local_map=item.local_map,
            local_counts=item.local_counts,
        )

    def _get_window_data(self, window_id: str) -> RasterWindowData:
        cached = self._cache.get(window_id)
        if cached is not None:
            self._cache.move_to_end(window_id)
            return cached

        meta = self.metadata.windows_by_id[window_id]
        item = _load_raster_window(
            window_id,
            meta,
            density_dir=self.density_dir,
            layer_index=self.layer_index,
            num_layers=self.num_layers,
            max_h=int(self.metadata.max_h),
            max_w=int(self.metadata.max_w),
        )
        self._cache[window_id] = item
        self._cache.move_to_end(window_id)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return item


@dataclass
class PCTWindow:
    window_id: str
    base_features: np.ndarray
    net_ids: np.ndarray
    id_to_name: Dict[int, str]
    name_to_id: Dict[str, int]
    sanitized_to_ids: Dict[str, List[int]]
    total_specs: Tuple[int, ...]
    env_specs: Tuple[Tuple[int, int], ...]


@dataclass(frozen=True)
class PCTSampleSpec:
    window_index: int
    master_id: int
    target_id: Optional[int]


@dataclass(frozen=True)
class PCTBatchPlan:
    sample_indices: Tuple[int, ...]


class PCTSelfAttention(nn.Module):
    def __init__(self, de: int, da: int) -> None:
        super().__init__()
        self.q_conv = nn.Conv1d(de, da, kernel_size=1, bias=False)
        self.k_conv = nn.Conv1d(de, da, kernel_size=1, bias=False)
        self.v_conv = nn.Conv1d(de, de, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q = self.q_conv(x)
        x_k = self.k_conv(x)
        x_v = self.v_conv(x)
        attention = self.softmax(torch.bmm(x_q.permute(0, 2, 1), x_k))
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
        z = torch.bmm(x_v, attention)
        return x + z


class PCTCap(nn.Module):
    def __init__(self, channels: int, num_sa: int, npoints: int) -> None:
        super().__init__()
        if npoints <= 0:
            raise ValueError(f"npoints must be positive, got {npoints}")
        self.npoints = int(npoints)
        self.conv1 = nn.Conv1d(channels, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 256, 1)
        self.bn2 = nn.BatchNorm1d(256)
        self.sa_list = nn.ModuleList(PCTSelfAttention(256, 256) for _ in range(int(num_sa)))
        self.fc1 = nn.Linear(self.npoints, self.npoints)
        self.fc2 = nn.Linear(self.npoints, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        for sa in self.sa_list:
            x = sa(x)
        x = x.permute(0, 2, 1)
        # Equivalent to adaptive_avg_pool1d(..., 1).view(-1, npoints) here, but
        # exports to a simpler ONNX ReduceMean that TensorRT can parse.
        x = x.mean(dim=2)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class _TensorRTForwardWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
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
            raise RuntimeError("TensorRT benchmarking requires a CUDA device.")
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
        self.input_min_shape, _opt_shape, self.input_max_shape = self._binding_profile_shapes(
            self.input_name,
            self.input_index,
        )
        self.max_batch_size = int(self.input_max_shape[0])
        self.input_tail_shape = tuple(int(v) for v in self.input_max_shape[1:])

    def _discover_bindings(self) -> Tuple[str, str, int, int]:
        if self._use_tensor_api:
            input_names: List[str] = []
            output_names: List[str] = []
            index_by_name: Dict[str, int] = {}
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
                    "TensorRT benchmark expects exactly one input and one output tensor, "
                    f"got inputs={input_names} outputs={output_names}"
                )
            return input_names[0], output_names[0], index_by_name[input_names[0]], index_by_name[output_names[0]]

        input_indices: List[int] = []
        output_indices: List[int] = []
        for idx in range(int(self.engine.num_bindings)):
            if bool(self.engine.binding_is_input(idx)):
                input_indices.append(idx)
            else:
                output_indices.append(idx)
        if len(input_indices) != 1 or len(output_indices) != 1:
            raise RuntimeError(
                "TensorRT benchmark expects exactly one input and one output binding, "
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

    def _binding_static_shape(self, name: str, index: int) -> Tuple[int, ...]:
        if self._use_tensor_api:
            return tuple(int(v) for v in self.engine.get_tensor_shape(name))
        return tuple(int(v) for v in self.engine.get_binding_shape(index))

    def _set_input_shape(self, shape: Tuple[int, ...]) -> None:
        if self._use_tensor_api:
            ok = self.context.set_input_shape(self.input_name, shape)
            if not ok:
                raise RuntimeError(f"TensorRT could not set input shape {shape} for the benchmark engine")
            return
        self.context.set_binding_shape(self.input_index, shape)

    def _get_output_shape(self) -> Tuple[int, ...]:
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
            raise RuntimeError("TensorRT benchmarking requires CUDA tensors.")
        if features.device.index != self.device.index:
            raise RuntimeError(
                f"TensorRT benchmark engine was loaded on cuda:{self.device.index} but received features on {features.device}"
            )
        if features.ndim != len(self.input_tail_shape) + 1:
            raise ValueError(
                f"TensorRT benchmark features must have rank {len(self.input_tail_shape) + 1}, got {tuple(features.shape)}"
            )
        batch_size = int(features.shape[0])
        if batch_size <= 0:
            raise ValueError(f"TensorRT benchmark batch size must be positive, got {batch_size}")
        if batch_size > self.max_batch_size:
            raise ValueError(
                f"TensorRT benchmark engine supports max batch {self.max_batch_size}, got {batch_size}"
            )
        tail_shape = tuple(int(v) for v in features.shape[1:])
        if tail_shape != self.input_tail_shape:
            raise ValueError(
                f"TensorRT benchmark engine expects input tail shape {self.input_tail_shape}, got {tail_shape}"
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


def _load_onnx_module():
    try:
        return importlib.import_module("onnx")
    except ModuleNotFoundError as exc:
        raise RuntimeError("TensorRT benchmarking requires the Python 'onnx' package.") from exc


def _load_tensorrt_module():
    try:
        return importlib.import_module("tensorrt")
    except ModuleNotFoundError as exc:
        raise RuntimeError("TensorRT benchmarking requires the Python 'tensorrt' package.") from exc


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
    messages = [str(parser.get_error(idx)) for idx in range(count)]
    return "\n".join(messages) if messages else "<no parser errors reported>"


def _compile_tensorrt_benchmark_model(
    model: nn.Module,
    *,
    device: torch.device,
    input_shape: Tuple[int, ...],
    opt_batch_size: int,
    max_batch_size: int,
) -> _TensorRTBenchmarkModule:
    if device.type != "cuda":
        raise RuntimeError("TensorRT benchmarking requires CUDA.")
    if opt_batch_size <= 0 or max_batch_size <= 0:
        raise ValueError(f"Batch sizes must be positive, got opt={opt_batch_size} max={max_batch_size}")
    if opt_batch_size > max_batch_size:
        raise ValueError(
            f"TensorRT benchmarking requires opt_batch_size <= max_batch_size, got opt={opt_batch_size} max={max_batch_size}"
        )

    _load_onnx_module()
    trt = _load_tensorrt_module()
    wrapper = _TensorRTForwardWrapper(model).eval()
    input_name = "features"
    output_name = "prediction"
    dynamic_axes = {
        input_name: {0: "batch"},
        output_name: {0: "batch"},
    }
    example_input = torch.randn((1, *input_shape), device=device, dtype=torch.float32)

    with tempfile.TemporaryDirectory(prefix="cnncap_flash_bench_trt_") as tmp_dir_str:
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
                opset_version=DEFAULT_TRT_OPSET,
                do_constant_folding=True,
            )

        logger = _get_tensorrt_logger(trt)
        builder = trt.Builder(logger)
        explicit_batch_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(explicit_batch_flag)
        parser = trt.OnnxParser(network, logger)
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
        profile = builder.create_optimization_profile()
        min_shape = (1, *input_shape)
        opt_shape = (int(opt_batch_size), *input_shape)
        max_shape = (int(max_batch_size), *input_shape)
        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)

        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError("TensorRT failed to build the benchmark engine.")
        return _TensorRTBenchmarkModule(bytes(serialized_engine), device=device)


def _is_via_layer(layer_name: str) -> bool:
    return "VIA" in str(layer_name).upper()


def _sorted_nonzero_ids(values: np.ndarray) -> np.ndarray:
    unique = np.unique(values.astype(np.int64, copy=False))
    unique = unique[unique > 0]
    return np.sort(unique).astype(np.int64, copy=False)


def _chunked(items: Iterable[Tuple[int, Optional[int], Optional[int]]], chunk_size: int) -> Iterator[List[Tuple[int, Optional[int], Optional[int]]]]:
    batch: List[Tuple[int, Optional[int], Optional[int]]] = []
    for item in items:
        batch.append(item)
        if len(batch) == chunk_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _resolve_gpu_device(gpu_id: int) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark, but no CUDA device is available.")
    device_count = torch.cuda.device_count()
    if gpu_id < 0 or gpu_id >= device_count:
        raise ValueError(f"--gpu-id must be in [0, {device_count - 1}], got {gpu_id}")
    torch.cuda.set_device(gpu_id)
    return torch.device(f"cuda:{gpu_id}")


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _build_monai_unet_model(*, in_channels: int, out_channels: int, monai_config: str) -> nn.Module:
    if int(in_channels) != int(out_channels):
        raise ValueError(
            "The benchmark expects the MONAI U-Net to preserve channel count, "
            f"got in_channels={in_channels}, out_channels={out_channels}."
        )
    return build_full_pipeline_unet(
        num_input_channels=int(in_channels),
        monai_config=str(monai_config),
    )


def _selector_supports_pct(selector: str) -> bool:
    return str(selector).strip().lower().endswith("/small")


def _window_stem_set(directory: Path, pattern: str) -> set[str]:
    return {path.stem for path in directory.glob(pattern)}


def _spef_window_id_set(directory: Path) -> set[str]:
    window_ids = {path.stem for path in directory.glob("*.spef")}
    window_ids.update(path.name[:-len(".spef.gz")] for path in directory.glob("*.spef.gz"))
    return window_ids


def _discover_common_window_ids(
    density_dir: Path,
    point_cloud_dir: Optional[Path],
    spef_dir: Path,
    *,
    max_windows: int,
    warmup_windows: int,
) -> Tuple[List[str], List[str]]:
    density_ids = set(discover_density_window_ids(density_dir))
    spef_ids = _spef_window_id_set(spef_dir)
    common_ids = density_ids & spef_ids
    if point_cloud_dir is not None:
        point_cloud_ids = _window_stem_set(point_cloud_dir, "*.npz")
        common_ids &= point_cloud_ids
    common = sorted(common_ids)
    if not common:
        required = "density_maps, labels_rwcap"
        if point_cloud_dir is not None:
            required += ", and point_clouds"
        raise RuntimeError(f"No windows exist across {required} for the selected dataset.")

    timed_count = min(int(max_windows), len(common))
    warmup_pool = max(len(common) - timed_count, 0)
    warmup_count = min(int(warmup_windows), warmup_pool)
    if warmup_count > 0:
        warmup_ids = common[:warmup_count]
        timed_ids = common[warmup_count:warmup_count + timed_count]
    else:
        warmup_ids = common[:min(int(warmup_windows), timed_count)]
        timed_ids = common[:timed_count]
    return warmup_ids, timed_ids


def _build_conductor_coords(
    layer_names: Sequence[str],
    id_maps: Sequence[np.ndarray],
    layer_index: Dict[str, int],
) -> Dict[int, List[Tuple[int, np.ndarray, np.ndarray]]]:
    coords: Dict[int, List[Tuple[int, np.ndarray, np.ndarray]]] = {}
    for local_idx, layer_name in enumerate(layer_names):
        global_idx = layer_index[layer_name]
        id_map = id_maps[local_idx]
        for conductor_id in _sorted_nonzero_ids(id_map):
            ys, xs = np.nonzero(id_map == conductor_id)
            if ys.size == 0:
                continue
            coords.setdefault(int(conductor_id), []).append(
                (
                    global_idx,
                    ys.astype(np.int64, copy=False),
                    xs.astype(np.int64, copy=False),
                )
            )
    return coords


def _scan_raster_window_metadata(window_ids: Sequence[str], density_dir: Path) -> RasterDatasetMeta:
    windows_by_id: Dict[str, RasterWindowMeta] = {}
    layer_catalog: List[str] = []
    max_h = 0
    max_w = 0

    for window_id in tqdm(window_ids, desc="Scanning density_maps", unit="win"):
        bundle_dir = density_dir / window_id
        meta = load_density_window_meta(bundle_dir)
        conductor_map = load_density_window_conductor_map(bundle_dir)
        if conductor_map:
            conductor_ids = _sorted_nonzero_ids(np.asarray(list(conductor_map.values()), dtype=np.int64))
        else:
            conductor_ids = _sorted_nonzero_ids(load_density_window_ids(bundle_dir, mmap_mode="r"))

        max_h = max(max_h, int(meta.shape[1]))
        max_w = max(max_w, int(meta.shape[2]))
        for layer_name, has_density in zip(meta.layer_names, meta.layer_has_density):
            if not has_density:
                continue
            if layer_name not in layer_catalog:
                layer_catalog.append(layer_name)

        windows_by_id[window_id] = RasterWindowMeta(window_id=window_id, conductor_ids=conductor_ids)

    if not layer_catalog:
        raise RuntimeError(f"No density-backed layers were found under {density_dir}")

    return RasterDatasetMeta(
        layer_catalog=tuple(layer_catalog),
        max_h=max_h,
        max_w=max_w,
        windows_by_id=windows_by_id,
    )


def _load_raster_window(
    window_id: str,
    meta: RasterWindowMeta,
    *,
    density_dir: Path,
    layer_index: Dict[str, int],
    num_layers: int,
    max_h: int,
    max_w: int,
) -> RasterWindowData:
    bundle_dir = density_dir / window_id
    bundle_meta = load_density_window_meta(bundle_dir)
    density_tensor = load_density_window_density(bundle_dir, mmap_mode="r")
    id_tensor = load_density_window_ids(bundle_dir, mmap_mode="r")

    density_base = np.zeros((num_layers, max_h, max_w), dtype=np.float32)
    occupancy_base = np.zeros((num_layers, max_h, max_w), dtype=np.float32)
    local_map_np = np.zeros((num_layers, max_h, max_w), dtype=np.int64)
    layer_names: List[str] = []
    id_maps: List[np.ndarray] = []

    for local_idx, layer_name in enumerate(bundle_meta.layer_names):
        global_idx = layer_index.get(layer_name)
        if global_idx is None:
            continue
        density = density_tensor[local_idx].astype(np.float32, copy=False)
        id_map = id_tensor[local_idx].astype(np.int32, copy=False)
        h, w = density.shape
        density_base[global_idx, :h, :w] = density
        occupancy_base[global_idx, :h, :w] = (id_map > 0).astype(np.float32, copy=False)
        local_map_np[global_idx, :h, :w] = id_map.astype(np.int64, copy=False)
        layer_names.append(layer_name)
        id_maps.append(id_map)

    conductor_coords = _build_conductor_coords(layer_names, id_maps, layer_index)
    local_map = torch.from_numpy(local_map_np).to(dtype=torch.long).contiguous()
    max_local_id = int(local_map.max().item())
    local_counts = torch.bincount(local_map.reshape(-1), minlength=max_local_id + 1).to(dtype=torch.long).contiguous()
    return RasterWindowData(
        window_id=window_id,
        conductor_ids=meta.conductor_ids,
        density_base=density_base,
        occupancy_base=occupancy_base,
        conductor_coords=conductor_coords,
        local_map=local_map,
        local_counts=local_counts,
    )


def _materialize_canonical_windows(
    window_ids: Sequence[str],
    by_id: Dict[str, RasterWindowMeta],
) -> List[CanonicalWindow]:
    return [
        CanonicalWindow(window_id=window_id, conductor_ids=by_id[window_id].conductor_ids)
        for window_id in window_ids
    ]


def _sanitize_net_name(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum() or ch in {"_", "."})


def _register_pct_conductor(
    cid: int,
    name: str,
    id_to_name: Dict[int, str],
    name_to_id: Dict[str, int],
    sanitized_to_ids: Dict[str, List[int]],
) -> None:
    actual = str(name).strip() or f"conductor_{cid}"
    id_to_name[cid] = actual
    name_to_id[actual.lower()] = cid
    sanitized = _sanitize_net_name(actual)
    sanitized_to_ids.setdefault(sanitized, []).append(cid)


def _lookup_pct_conductor(window: PCTWindow, net_name: str) -> Optional[int]:
    candidate = str(net_name).strip()
    if not candidate:
        return None
    lower = candidate.lower()
    if lower in window.name_to_id:
        return window.name_to_id[lower]
    sanitized = _sanitize_net_name(candidate)
    candidates = window.sanitized_to_ids.get(sanitized)
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    for cid in candidates:
        actual = window.id_to_name.get(cid, "")
        if actual.lower() == lower:
            return cid
    return candidates[0]


def _compute_pct_centroids(window: PCTWindow) -> Dict[int, Tuple[float, float, float]]:
    if window.base_features.size == 0:
        return {}
    coords = window.base_features[:, :3].astype(np.float64, copy=False)
    net_ids = window.net_ids.astype(np.int64, copy=False)
    unique_ids, inverse = np.unique(net_ids, return_inverse=True)
    sums = np.zeros((len(unique_ids), 3), dtype=np.float64)
    counts = np.zeros(len(unique_ids), dtype=np.int64)
    np.add.at(sums, inverse, coords)
    np.add.at(counts, inverse, 1)
    centroids: Dict[int, Tuple[float, float, float]] = {}
    for idx, cid in enumerate(unique_ids):
        if counts[idx] <= 0:
            continue
        centroids[int(cid)] = tuple((sums[idx] / counts[idx]).tolist())
    return centroids


def _coordinate_distance_sq(coord: Optional[Tuple[float, float, float]]) -> float:
    if coord is None:
        return float("inf")
    x, y, z = coord
    return float(x * x + y * y + z * z)


def _find_spef_for_window(window_id: str, point_cloud_dir: Path, spef_dir: Path) -> Path:
    exact = spef_dir / f"{window_id}.spef"
    if exact.exists():
        return exact
    primary = point_cloud_dir / f"{window_id}.spef"
    if primary.exists():
        return primary
    raise FileNotFoundError(f"No SPEF file matching window '{window_id}' in {spef_dir} or {point_cloud_dir}")


def _load_pct_window_map(window_ids: Sequence[str], point_cloud_dir: Path, spef_dir: Path) -> Dict[str, PCTWindow]:
    by_id: Dict[str, PCTWindow] = {}
    for window_id in tqdm(window_ids, desc="Loading point_clouds", unit="win"):
        npz_path = point_cloud_dir / f"{window_id}.npz"
        with np.load(npz_path, allow_pickle=True) as data:
            points = data["points"].astype(np.float32, copy=False)
            base_features = points[:, :7].astype(np.float32, copy=True)
            net_ids = points[:, 8].astype(np.int32, copy=True)
            conductor_ids = data.get("conductor_ids", np.unique(net_ids)).astype(np.int32)
            metadata_str = str(data.get("conductor_metadata_str", ""))
            point_names = data.get("point_net_names")

        id_to_name: Dict[int, str] = {}
        name_to_id: Dict[str, int] = {}
        sanitized_to_ids: Dict[str, List[int]] = {}
        entries = [entry for entry in metadata_str.split(";") if entry]
        for cid, entry in zip(conductor_ids, entries):
            name, *_layer = entry.split("|", maxsplit=1)
            _register_pct_conductor(int(cid), name, id_to_name, name_to_id, sanitized_to_ids)
        if len(id_to_name) < len(conductor_ids) and point_names is not None:
            point_names_arr = np.asarray(point_names)
            for cid in conductor_ids:
                if int(cid) in id_to_name:
                    continue
                matches = np.where(net_ids == int(cid))[0]
                if matches.size == 0:
                    continue
                _register_pct_conductor(int(cid), str(point_names_arr[matches[0]]), id_to_name, name_to_id, sanitized_to_ids)

        provisional = PCTWindow(
            window_id=window_id,
            base_features=base_features,
            net_ids=net_ids,
            id_to_name=id_to_name,
            name_to_id=name_to_id,
            sanitized_to_ids=sanitized_to_ids,
            total_specs=(),
            env_specs=(),
        )
        spef_path = _find_spef_for_window(window_id, point_cloud_dir, spef_dir)
        total_map = load_dnet_totals(str(spef_path))
        _ground_map, adjacency, _ground_cap = parse_spef_components(str(spef_path))
        centroids = _compute_pct_centroids(provisional)

        total_specs: List[int] = []
        for net_name in total_map:
            master_id = _lookup_pct_conductor(provisional, net_name)
            if master_id is not None:
                total_specs.append(int(master_id))

        pair_caps: Dict[Tuple[int, int], float] = {}
        for master_name, neighbors in adjacency.items():
            master_id = _lookup_pct_conductor(provisional, master_name)
            if master_id is None:
                continue
            neighbor_iter = neighbors.items() if isinstance(neighbors, dict) else neighbors
            for neighbor_name, value in neighbor_iter:
                if float(value) <= 1e-20:
                    continue
                target_id = _lookup_pct_conductor(provisional, neighbor_name)
                if target_id is None:
                    continue
                pair = tuple(sorted((int(master_id), int(target_id))))
                if pair in pair_caps:
                    pair_caps[pair] = 0.5 * (pair_caps[pair] + float(value))
                else:
                    pair_caps[pair] = float(value)

        env_specs: List[Tuple[int, int]] = []
        for (id_a, id_b), _value in pair_caps.items():
            centroid_a = centroids.get(id_a)
            centroid_b = centroids.get(id_b)
            if _coordinate_distance_sq(centroid_a) <= _coordinate_distance_sq(centroid_b):
                env_specs.append((id_a, id_b))
            else:
                env_specs.append((id_b, id_a))

        by_id[window_id] = PCTWindow(
            window_id=window_id,
            base_features=base_features,
            net_ids=net_ids,
            id_to_name=id_to_name,
            name_to_id=name_to_id,
            sanitized_to_ids=sanitized_to_ids,
            total_specs=tuple(total_specs),
            env_specs=tuple(env_specs),
        )
    return by_id


def _materialize_pct_windows(window_ids: Sequence[str], by_id: Dict[str, PCTWindow]) -> List[PCTWindow]:
    return [by_id[window_id] for window_id in window_ids]


def _iter_scalar_specs(windows: Sequence[CanonicalWindow], goal: str) -> Iterator[Tuple[int, Optional[int], Optional[int]]]:
    for window_idx, window in enumerate(windows):
        conductor_ids = [int(cid) for cid in window.conductor_ids]
        if goal == "total":
            for conductor_id in conductor_ids:
                yield (window_idx, conductor_id, None)
            continue
        for left in range(len(conductor_ids)):
            for right in range(left + 1, len(conductor_ids)):
                yield (window_idx, conductor_ids[left], conductor_ids[right])


def _iter_unet_specs(windows: Sequence[CanonicalWindow], goal: str) -> Iterator[Tuple[int, Optional[int], Optional[int]]]:
    for window_idx, window in enumerate(windows):
        conductor_ids = [int(cid) for cid in window.conductor_ids]
        if goal == "total":
            yield (window_idx, None, None)
            continue
        for conductor_id in conductor_ids:
            yield (window_idx, conductor_id, None)


def _build_resnet_batch(
    window_ids: Sequence[str],
    raster_store: LazyRasterWindowStore,
    plan: ResnetBatchPlan,
) -> torch.Tensor:
    loaded_windows = {
        window_idx: raster_store.get_density_window(window_ids[window_idx])
        for window_idx in dict.fromkeys(plan.window_indices)
    }
    batch = np.stack([loaded_windows[idx].base_features for idx in plan.window_indices], axis=0)
    for row, window_idx in enumerate(plan.window_indices):
        conductor_a = plan.conductor_a_ids[row]
        conductor_b = plan.conductor_b_ids[row]
        window = loaded_windows[window_idx]
        if conductor_a is not None:
            for channel_idx, ys, xs in window.conductor_coords.get(int(conductor_a), ()):
                batch[row, channel_idx, ys, xs] = batch[row, channel_idx, ys, xs] + 1.0
        if conductor_b is not None:
            for channel_idx, ys, xs in window.conductor_coords.get(int(conductor_b), ()):
                batch[row, channel_idx, ys, xs] = -batch[row, channel_idx, ys, xs]
    return torch.from_numpy(batch)


def _build_unet_batch(
    window_ids: Sequence[str],
    raster_store: LazyRasterWindowStore,
    plan: UnetBatchPlan,
) -> UnetBatch:
    loaded_windows = {
        window_idx: raster_store.get_unet_window(window_ids[window_idx])
        for window_idx in dict.fromkeys(plan.window_indices)
    }
    batch = np.stack([loaded_windows[idx].base_features for idx in plan.window_indices], axis=0)
    local_maps = [loaded_windows[idx].local_map for idx in plan.window_indices]
    local_counts_list = [loaded_windows[idx].local_counts for idx in plan.window_indices]
    sparse_indices, sparse_counts = build_sparse_index_tensors(local_maps, local_counts_list)
    max_local = int(sparse_counts.shape[1])
    local_counts_batch = torch.zeros((len(plan.window_indices), max_local), dtype=torch.long)
    for row_idx, counts in enumerate(local_counts_list):
        local_counts_batch[row_idx, : int(counts.shape[0])] = counts
    for row, window_idx in enumerate(plan.window_indices):
        conductor_a = plan.conductor_a_ids[row]
        if conductor_a is None:
            continue
        window = loaded_windows[window_idx]
        for channel_idx, ys, xs in window.conductor_coords.get(int(conductor_a), ()):
            batch[row, channel_idx, ys, xs] = -1.0
    return UnetBatch(
        features=torch.from_numpy(batch),
        local_counts=local_counts_batch.contiguous(),
        sparse_indices=sparse_indices.contiguous(),
        sparse_counts=sparse_counts.contiguous(),
    )


def _build_resnet_batch_plans(canonical: Sequence[CanonicalWindow], goal: str, batch_size: int) -> List[ResnetBatchPlan]:
    plans: List[ResnetBatchPlan] = []
    for specs in _chunked(_iter_scalar_specs(canonical, goal), batch_size):
        plans.append(
            ResnetBatchPlan(
                window_indices=tuple(int(window_idx) for window_idx, _a, _b in specs),
                conductor_a_ids=tuple(conductor_a for _window_idx, conductor_a, _b in specs),
                conductor_b_ids=tuple(conductor_b for _window_idx, _a, conductor_b in specs),
            )
        )
    return plans


def _build_unet_batch_plans(
    canonical: Sequence[CanonicalWindow],
    goal: str,
    batch_size: int,
) -> List[UnetBatchPlan]:
    plans: List[UnetBatchPlan] = []
    for specs in _chunked(_iter_unet_specs(canonical, goal), batch_size):
        plans.append(
            UnetBatchPlan(
                window_indices=tuple(int(window_idx) for window_idx, _a, _b in specs),
                conductor_a_ids=tuple(conductor_a for _window_idx, conductor_a, _unused in specs),
            )
        )
    return plans


def _resnet_batch_factory(
    window_ids: Sequence[str],
    raster_store: LazyRasterWindowStore,
    plans: Sequence[ResnetBatchPlan],
) -> Iterator[torch.Tensor]:
    for plan in plans:
        yield _build_resnet_batch(window_ids, raster_store, plan)


def _unet_batch_factory(
    window_ids: Sequence[str],
    raster_store: LazyRasterWindowStore,
    plans: Sequence[UnetBatchPlan],
) -> Iterator[UnetBatch]:
    for plan in plans:
        yield _build_unet_batch(window_ids, raster_store, plan)


def _pct_sample_seed(window_id: str, master_id: int, target_id: Optional[int]) -> int:
    token = f"{window_id}:{master_id}:{-1 if target_id is None else int(target_id)}"
    return DEFAULT_SAMPLE_SEED ^ zlib.adler32(token.encode("utf-8"))


def _pc_normalize(pc: np.ndarray) -> np.ndarray:
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    radius = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    if radius > 0:
        pc = pc / radius
    return pc


def _sample_pct_features(features: np.ndarray, *, npoints: int, seed: int) -> np.ndarray:
    if features.shape[0] == 0:
        raise RuntimeError("PCT point-cloud benchmark encountered an empty point cloud.")
    rng = np.random.default_rng(seed)
    sampled = features
    if sampled.shape[0] < npoints:
        extra_count = npoints - sampled.shape[0]
        extra_indices = rng.choice(sampled.shape[0], size=extra_count, replace=True)
        sampled = np.concatenate([sampled, sampled[extra_indices]], axis=0)
    if sampled.shape[0] > npoints:
        keep = rng.choice(sampled.shape[0], size=npoints, replace=False)
        sampled = sampled[keep]
    else:
        sampled = sampled.copy()
    sampled[:, 0:3] = _pc_normalize(sampled[:, 0:3].astype(np.float32, copy=False))
    return sampled.astype(np.float32, copy=False)


def _materialize_pct_sample(window: PCTWindow, sample: PCTSampleSpec, *, npoints: int) -> np.ndarray:
    flux = np.zeros_like(window.net_ids, dtype=np.float32)
    flux[window.net_ids == int(sample.master_id)] = 1.0
    if sample.target_id is not None:
        flux[window.net_ids == int(sample.target_id)] = -1.0
    features = np.concatenate([window.base_features, flux[:, np.newaxis]], axis=1)
    return _sample_pct_features(
        features,
        npoints=npoints,
        seed=_pct_sample_seed(window.window_id, sample.master_id, sample.target_id),
    )


def _build_pct_sample_specs(windows: Sequence[PCTWindow], goal: str) -> List[PCTSampleSpec]:
    specs: List[PCTSampleSpec] = []
    for window_index, window in enumerate(windows):
        if goal == "total":
            specs.extend(
                PCTSampleSpec(window_index=window_index, master_id=int(master_id), target_id=None)
                for master_id in window.total_specs
            )
            continue
        specs.extend(
            PCTSampleSpec(window_index=window_index, master_id=int(master_id), target_id=int(target_id))
            for master_id, target_id in window.env_specs
        )
    return specs


def _build_pct_batch_plans(samples: Sequence[PCTSampleSpec], batch_size: int) -> List[PCTBatchPlan]:
    plans: List[PCTBatchPlan] = []
    for start in range(0, len(samples), batch_size):
        chunk = tuple(range(start, min(start + batch_size, len(samples))))
        plans.append(PCTBatchPlan(sample_indices=chunk))
    return plans


def _build_pct_batch(
    windows: Sequence[PCTWindow],
    samples: Sequence[PCTSampleSpec],
    plan: PCTBatchPlan,
    *,
    npoints: int,
) -> torch.Tensor:
    batch = np.stack(
        [
            _materialize_pct_sample(windows[samples[sample_idx].window_index], samples[sample_idx], npoints=npoints)
            for sample_idx in plan.sample_indices
        ],
        axis=0,
    )
    return torch.from_numpy(batch)


def _pct_batch_factory(
    windows: Sequence[PCTWindow],
    samples: Sequence[PCTSampleSpec],
    plans: Sequence[PCTBatchPlan],
    *,
    npoints: int,
) -> Iterator[torch.Tensor]:
    for plan in plans:
        yield _build_pct_batch(windows, samples, plan, npoints=npoints)


def _run_scalar_engine_timed(
    benchmark_model: _TensorRTBenchmarkModule,
    warmup_batch_iter_factory,
    timed_batch_iter_factory,
    device: torch.device,
    *,
    warmup_batches: int,
) -> float:
    timed_count = 0
    with torch.inference_mode():
        if warmup_batches > 0:
            for warmup_count, warmup_batch_cpu in enumerate(warmup_batch_iter_factory(), start=1):
                batch = warmup_batch_cpu.to(device=device, dtype=torch.float32, non_blocking=True)
                _ = benchmark_model(batch)
                if warmup_count >= warmup_batches:
                    break
        _synchronize(device)

        event_pairs: List[Tuple[torch.cuda.Event, torch.cuda.Event]] = []
        last_output: Optional[torch.Tensor] = None
        for timed_count, batch_cpu in enumerate(timed_batch_iter_factory(), start=1):
            batch = batch_cpu.to(device=device, dtype=torch.float32, non_blocking=True)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            last_output = benchmark_model(batch)
            end_event.record()
            event_pairs.append((start_event, end_event))
        if timed_count == 0:
            return 0.0
        _synchronize(device)
        total_s = sum(float(start.elapsed_time(end)) for start, end in event_pairs) / 1000.0
    if last_output is None:
        return 0.0
    return total_s


def _run_unet_engine_timed(
    benchmark_model: _TensorRTBenchmarkModule,
    warmup_batch_iter_factory,
    timed_batch_iter_factory,
    device: torch.device,
    *,
    warmup_batches: int,
) -> float:
    timed_count = 0
    with torch.inference_mode():
        if warmup_batches > 0:
            for warmup_count, warmup_batch_cpu in enumerate(warmup_batch_iter_factory(), start=1):
                features = warmup_batch_cpu.features.to(device=device, dtype=torch.float32, non_blocking=True)
                local_counts = warmup_batch_cpu.local_counts.to(device=device, non_blocking=True)
                sparse_indices = warmup_batch_cpu.sparse_indices.to(device=device, non_blocking=True)
                sparse_counts = warmup_batch_cpu.sparse_counts.to(device=device, non_blocking=True)
                q_map = F.softplus(benchmark_model(features))
                _reduced, _areas = reduce_qmap_to_all_conductors_sparse(
                    q_map,
                    local_counts,
                    sparse_indices,
                    sparse_counts,
                    reduction="sum",
                    force_fp32=False,
                )
                if warmup_count >= warmup_batches:
                    break
        _synchronize(device)

        event_pairs: List[Tuple[torch.cuda.Event, torch.cuda.Event]] = []
        last_output: Optional[torch.Tensor] = None
        for timed_count, batch_cpu in enumerate(timed_batch_iter_factory(), start=1):
            features = batch_cpu.features.to(device=device, dtype=torch.float32, non_blocking=True)
            local_counts = batch_cpu.local_counts.to(device=device, non_blocking=True)
            sparse_indices = batch_cpu.sparse_indices.to(device=device, non_blocking=True)
            sparse_counts = batch_cpu.sparse_counts.to(device=device, non_blocking=True)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            q_map = F.softplus(benchmark_model(features))
            reduced, _areas = reduce_qmap_to_all_conductors_sparse(
                q_map,
                local_counts,
                sparse_indices,
                sparse_counts,
                reduction="sum",
                force_fp32=False,
            )
            end_event.record()
            event_pairs.append((start_event, end_event))
            last_output = reduced
        if timed_count == 0:
            return 0.0
        _synchronize(device)
        total_s = sum(float(start.elapsed_time(end)) for start, end in event_pairs) / 1000.0
    if last_output is None:
        return 0.0
    return total_s


def _average_conductor_count(windows: Sequence[CanonicalWindow]) -> float:
    counts = [int(window.conductor_ids.size) for window in windows]
    if not counts:
        raise RuntimeError("No canonical windows were loaded.")
    return float(sum(counts)) / float(len(counts))


def _total_scalar_runs(windows: Sequence[CanonicalWindow], goal: str) -> int:
    if goal == "total":
        return sum(int(window.conductor_ids.size) for window in windows)
    return sum(int(window.conductor_ids.size * (window.conductor_ids.size - 1) // 2) for window in windows)


def _total_unet_runs(windows: Sequence[CanonicalWindow], goal: str) -> int:
    if goal == "total":
        return len(windows)
    return sum(int(window.conductor_ids.size) for window in windows)


def _selector_label(selector: str) -> str:
    process_node, split = str(selector).split("/", maxsplit=1)
    return f"{process_node[:1].upper()}{process_node[1:]} {split.title()}"


def _format_markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, separator_line, *body])


def _parse_dataset_selectors(raw: str) -> List[str]:
    selectors = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not selectors:
        raise ValueError("At least one dataset selector must be provided.")
    return selectors


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark TensorRT full-matrix extraction runtime across ResNet, PCT, and MONAI U-Net.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=",".join(DEFAULT_DATASET_SELECTORS),
        help="Comma-separated CapBench dataset selectors to benchmark.",
    )
    parser.add_argument("--gpu-id", type=int, required=True, help="CUDA device index to use for the benchmark.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Maximum TensorRT batch size.")
    parser.add_argument("--max-windows", type=int, default=DEFAULT_MAX_WINDOWS, help="Timed window count per dataset split.")
    parser.add_argument("--warmup-windows", type=int, default=DEFAULT_WARMUP_WINDOWS, help="Warmup window count per dataset split.")
    parser.add_argument("--warmup-batches", type=int, default=DEFAULT_WARMUP_BATCHES, help="Warmup batches per phase.")
    parser.add_argument("--pct-npoints", type=int, default=DEFAULT_PCT_NPOINTS, help="Point count for each PCT sample.")
    parser.add_argument(
        "--unet-monai-config",
        type=str,
        choices=sorted(FULL_PIPELINE_MONAI_CONFIGS),
        default=None,
        help="Optional override to benchmark just one MONAI U-Net config. Defaults to the built-in pair D4_B and D4_D_k5.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.batch_size <= 0:
        raise ValueError(f"--batch-size must be positive, got {args.batch_size}")
    if args.max_windows <= 0:
        raise ValueError(f"--max-windows must be positive, got {args.max_windows}")
    if args.warmup_windows < 0:
        raise ValueError(f"--warmup-windows must be non-negative, got {args.warmup_windows}")
    if args.warmup_batches < 0:
        raise ValueError(f"--warmup-batches must be non-negative, got {args.warmup_batches}")
    if args.pct_npoints <= 0:
        raise ValueError(f"--pct-npoints must be positive, got {args.pct_npoints}")

    selectors = _parse_dataset_selectors(args.datasets)
    device = _resolve_gpu_device(int(args.gpu_id))
    torch.backends.cudnn.benchmark = True
    load_sparse_reduce_cuda_extension()
    selected_unet_configs = (
        (str(args.unet_monai_config),)
        if args.unet_monai_config is not None
        else tuple(str(name) for name in DEFAULT_BENCHMARK_UNET_CONFIGS)
    )
    unet_model_names = tuple(f"monai_{config_name}" for config_name in selected_unet_configs)

    results: Dict[str, Dict[str, float]] = {
        "resnet18": {},
        "resnet34": {},
        "resnet50": {},
        "resnet101": {},
        "pct_sa4": {},
        "pct_sa6": {},
        "pct_sa8": {},
        "pct_sa10": {},
    }
    for unet_model_name in unet_model_names:
        results[unet_model_name] = {}
    split_windows: Dict[str, int] = {}
    comparison_rows: List[Dict[str, object]] = []

    print(f"Selected GPU: cuda:{device.index}")
    print(f"TensorRT max batch size: {args.batch_size}")
    print(f"Warmup batches per phase: {args.warmup_batches}")
    print(f"U-Net MONAI configs: {', '.join(selected_unet_configs)}")
    print()

    for selector in selectors:
        benchmark_pct = _selector_supports_pct(selector)
        artifact_names = ("density_maps", "point_clouds", "labels_rwcap") if benchmark_pct else ("density_maps", "labels_rwcap")
        dataset_root = resolve_dataset_path(selector, artifacts=artifact_names).resolve()
        density_dir = (dataset_root / "density_maps").resolve()
        point_cloud_dir = (dataset_root / "point_clouds").resolve() if benchmark_pct else None
        spef_dir = (dataset_root / "labels_rwcap").resolve()
        required_dirs = (density_dir, spef_dir) if point_cloud_dir is None else (density_dir, point_cloud_dir, spef_dir)
        for required_dir in required_dirs:
            if not required_dir.exists():
                raise FileNotFoundError(f"Required benchmark directory not found: {required_dir}")

        warmup_window_ids, timed_window_ids = _discover_common_window_ids(
            density_dir,
            point_cloud_dir,
            spef_dir,
            max_windows=args.max_windows,
            warmup_windows=args.warmup_windows,
        )
        label = _selector_label(selector)
        split_windows[label] = len(timed_window_ids)
        print(
            f"[{label}] dataset={dataset_root} warmup_windows={len(warmup_window_ids)} timed_windows={len(timed_window_ids)}"
        )

        unique_window_ids = list(dict.fromkeys([*warmup_window_ids, *timed_window_ids]))
        raster_metadata = _scan_raster_window_metadata(unique_window_ids, density_dir)
        raster_store = LazyRasterWindowStore(density_dir=density_dir, metadata=raster_metadata)
        pct_window_map = _load_pct_window_map(unique_window_ids, point_cloud_dir, spef_dir) if point_cloud_dir is not None else None

        warmup_canonical = _materialize_canonical_windows(
            warmup_window_ids,
            raster_metadata.windows_by_id,
        )
        canonical_windows = _materialize_canonical_windows(
            timed_window_ids,
            raster_metadata.windows_by_id,
        )
        warmup_pct_windows = _materialize_pct_windows(warmup_window_ids, pct_window_map) if pct_window_map is not None else []
        pct_windows = _materialize_pct_windows(timed_window_ids, pct_window_map) if pct_window_map is not None else []

        avg_conductors = _average_conductor_count(canonical_windows)
        scalar_total_runs = _total_scalar_runs(canonical_windows, "total")
        scalar_env_runs = _total_scalar_runs(canonical_windows, "env")
        unet_total_runs = _total_unet_runs(canonical_windows, "total")
        unet_env_runs = _total_unet_runs(canonical_windows, "env")
        pct_total_specs: List[PCTSampleSpec] = _build_pct_sample_specs(pct_windows, "total") if benchmark_pct else []
        pct_env_specs: List[PCTSampleSpec] = _build_pct_sample_specs(pct_windows, "env") if benchmark_pct else []
        run_summary = (
            f"[{label}] avg_conductors={avg_conductors:.3f} "
            f"scalar_total_runs={scalar_total_runs} scalar_env_runs={scalar_env_runs} "
            f"unet_total_runs={unet_total_runs} unet_env_runs={unet_env_runs}"
        )
        if benchmark_pct:
            run_summary += f" pct_total_runs={len(pct_total_specs)} pct_env_runs={len(pct_env_specs)}"
        else:
            run_summary += " pct_runs=skipped"
        print(run_summary)

        warmup_resnet_total_plans = _build_resnet_batch_plans(warmup_canonical, "total", args.batch_size)
        warmup_resnet_env_plans = _build_resnet_batch_plans(warmup_canonical, "env", args.batch_size)
        timed_resnet_total_plans = _build_resnet_batch_plans(canonical_windows, "total", args.batch_size)
        timed_resnet_env_plans = _build_resnet_batch_plans(canonical_windows, "env", args.batch_size)
        raster_feature_shape = raster_store.feature_shape
        resnet_total_timed_factory = lambda: _resnet_batch_factory(timed_window_ids, raster_store, timed_resnet_total_plans)
        resnet_env_timed_factory = lambda: _resnet_batch_factory(timed_window_ids, raster_store, timed_resnet_env_plans)
        resnet_total_warmup_factory = (
            (lambda: _resnet_batch_factory(warmup_window_ids, raster_store, warmup_resnet_total_plans))
            if warmup_resnet_total_plans
            else resnet_total_timed_factory
        )
        resnet_env_warmup_factory = (
            (lambda: _resnet_batch_factory(warmup_window_ids, raster_store, warmup_resnet_env_plans))
            if warmup_resnet_env_plans
            else resnet_env_timed_factory
        )

        warmup_unet_total_plans = _build_unet_batch_plans(warmup_canonical, "total", args.batch_size)
        warmup_unet_env_plans = _build_unet_batch_plans(warmup_canonical, "env", args.batch_size)
        timed_unet_total_plans = _build_unet_batch_plans(canonical_windows, "total", args.batch_size)
        timed_unet_env_plans = _build_unet_batch_plans(canonical_windows, "env", args.batch_size)
        unet_total_timed_factory = lambda: _unet_batch_factory(timed_window_ids, raster_store, timed_unet_total_plans)
        unet_env_timed_factory = lambda: _unet_batch_factory(timed_window_ids, raster_store, timed_unet_env_plans)
        unet_total_warmup_factory = (
            (lambda: _unet_batch_factory(warmup_window_ids, raster_store, warmup_unet_total_plans))
            if warmup_unet_total_plans
            else unet_total_timed_factory
        )
        unet_env_warmup_factory = (
            (lambda: _unet_batch_factory(warmup_window_ids, raster_store, warmup_unet_env_plans))
            if warmup_unet_env_plans
            else unet_env_timed_factory
        )

        warmup_pct_total_specs: List[PCTSampleSpec] = []
        warmup_pct_env_specs: List[PCTSampleSpec] = []
        warmup_pct_total_plans: List[PCTBatchPlan] = []
        warmup_pct_env_plans: List[PCTBatchPlan] = []
        timed_pct_total_plans: List[PCTBatchPlan] = []
        timed_pct_env_plans: List[PCTBatchPlan] = []
        pct_total_timed_factory = None
        pct_env_timed_factory = None
        pct_total_warmup_factory = None
        pct_env_warmup_factory = None
        if benchmark_pct:
            warmup_pct_total_specs = _build_pct_sample_specs(warmup_pct_windows, "total")
            warmup_pct_env_specs = _build_pct_sample_specs(warmup_pct_windows, "env")
            warmup_pct_total_plans = _build_pct_batch_plans(warmup_pct_total_specs, args.batch_size)
            warmup_pct_env_plans = _build_pct_batch_plans(warmup_pct_env_specs, args.batch_size)
            timed_pct_total_plans = _build_pct_batch_plans(pct_total_specs, args.batch_size)
            timed_pct_env_plans = _build_pct_batch_plans(pct_env_specs, args.batch_size)
            pct_total_timed_factory = lambda: _pct_batch_factory(
                pct_windows,
                pct_total_specs,
                timed_pct_total_plans,
                npoints=args.pct_npoints,
            )
            pct_env_timed_factory = lambda: _pct_batch_factory(
                pct_windows,
                pct_env_specs,
                timed_pct_env_plans,
                npoints=args.pct_npoints,
            )
            pct_total_warmup_factory = (
                (lambda: _pct_batch_factory(
                    warmup_pct_windows,
                    warmup_pct_total_specs,
                    warmup_pct_total_plans,
                    npoints=args.pct_npoints,
                ))
                if warmup_pct_total_plans
                else pct_total_timed_factory
            )
            pct_env_warmup_factory = (
                (lambda: _pct_batch_factory(
                    warmup_pct_windows,
                    warmup_pct_env_specs,
                    warmup_pct_env_plans,
                    npoints=args.pct_npoints,
                ))
                if warmup_pct_env_plans
                else pct_env_timed_factory
            )

        density_channels = int(raster_feature_shape[0])
        resnet_builders = {
            "resnet18": lambda: resnet_custom.resnet18(num_classes=1, num_input_channels=density_channels),
            "resnet34": lambda: resnet_custom.resnet34(num_classes=1, num_input_channels=density_channels),
            "resnet50": lambda: resnet_custom.resnet50(num_classes=1, num_input_channels=density_channels),
            "resnet101": lambda: resnet_custom.resnet101(num_classes=1, num_input_channels=density_channels),
        }
        for model_name, builder in resnet_builders.items():
            print(f"[{label}] Benchmarking {model_name}...")
            model = builder().to(device=device, dtype=torch.float32)
            engine = _compile_tensorrt_benchmark_model(
                model,
                device=device,
                input_shape=tuple(int(v) for v in raster_feature_shape),
                opt_batch_size=args.batch_size,
                max_batch_size=args.batch_size,
            )
            total_time = _run_scalar_engine_timed(
                engine,
                resnet_total_warmup_factory,
                resnet_total_timed_factory,
                device,
                warmup_batches=args.warmup_batches,
            )
            env_time = _run_scalar_engine_timed(
                engine,
                resnet_env_warmup_factory,
                resnet_env_timed_factory,
                device,
                warmup_batches=args.warmup_batches,
            )
            results[model_name][label] = total_time + env_time
            del engine
            del model
            torch.cuda.empty_cache()

        unet_channels = int(raster_feature_shape[0])
        for unet_config_name, unet_model_name in zip(selected_unet_configs, unet_model_names):
            print(f"[{label}] Benchmarking {unet_model_name}...")
            unet_model = _build_monai_unet_model(
                in_channels=unet_channels,
                out_channels=unet_channels,
                monai_config=str(unet_config_name),
            ).to(
                device=device,
                dtype=torch.float32,
            )
            unet_engine = _compile_tensorrt_benchmark_model(
                unet_model,
                device=device,
                input_shape=tuple(int(v) for v in raster_feature_shape),
                opt_batch_size=args.batch_size,
                max_batch_size=args.batch_size,
            )
            unet_total_time = _run_unet_engine_timed(
                unet_engine,
                unet_total_warmup_factory,
                unet_total_timed_factory,
                device,
                warmup_batches=args.warmup_batches,
            )
            unet_env_time = _run_unet_engine_timed(
                unet_engine,
                unet_env_warmup_factory,
                unet_env_timed_factory,
                device,
                warmup_batches=args.warmup_batches,
            )
            results[unet_model_name][label] = unet_total_time + unet_env_time
            del unet_engine
            del unet_model
            torch.cuda.empty_cache()

        if benchmark_pct:
            for sa_layers in PCT_VARIANTS:
                model_name = f"pct_sa{sa_layers}"
                print(f"[{label}] Benchmarking {model_name}...")
                pct_model = PCTCap(channels=8, num_sa=sa_layers, npoints=args.pct_npoints).to(device=device, dtype=torch.float32)
                pct_engine = _compile_tensorrt_benchmark_model(
                    pct_model,
                    device=device,
                    input_shape=(args.pct_npoints, 8),
                    opt_batch_size=args.batch_size,
                    max_batch_size=args.batch_size,
                )
                pct_total_time = _run_scalar_engine_timed(
                    pct_engine,
                    pct_total_warmup_factory,
                    pct_total_timed_factory,
                    device,
                    warmup_batches=args.warmup_batches,
                )
                pct_env_time = _run_scalar_engine_timed(
                    pct_engine,
                    pct_env_warmup_factory,
                    pct_env_timed_factory,
                    device,
                    warmup_batches=args.warmup_batches,
                )
                results[model_name][label] = pct_total_time + pct_env_time
                del pct_engine
                del pct_model
                torch.cuda.empty_cache()

        comparison_rows.append(
            {
                "label": label,
                "resnet_runs": scalar_total_runs + scalar_env_runs,
                "unet_runs": unet_total_runs + unet_env_runs,
                "avg_conductors": avg_conductors,
                "speedups": {
                    unet_model_name: results["resnet34"][label] / max(results[unet_model_name][label], 1e-12)
                    for unet_model_name in unet_model_names
                },
            }
        )
        print()

    throughput_headers = [
        "Model",
        "Family",
        *[f"{label} Full Matrix (s)" for label in split_windows],
    ]
    throughput_rows: List[List[str]] = []
    model_order = (
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "pct_sa4",
        "pct_sa6",
        "pct_sa8",
        "pct_sa10",
        *unet_model_names,
    )
    family_by_model = {
        "resnet18": "resnet",
        "resnet34": "resnet",
        "resnet50": "resnet",
        "resnet101": "resnet",
        "pct_sa4": "pct",
        "pct_sa6": "pct",
        "pct_sa8": "pct",
        "pct_sa10": "pct",
    }
    for unet_model_name in unet_model_names:
        family_by_model[unet_model_name] = "unet"
    for model_name in model_order:
        throughput_rows.append(
            [
                model_name,
                family_by_model[model_name],
                *[
                    f"{results[model_name][label]:.6f}" if label in results[model_name] else "-"
                    for label in split_windows
                ],
            ]
        )

    comparison_headers = [
        "Dataset",
        "ResNet34 Runs",
        "U-Net Runs",
        "Avg. Conductors",
        *[f"ResNet34 vs {unet_model_name} Full-Matrix Speedup" for unet_model_name in unet_model_names],
    ]
    comparison_markdown_rows = [
        [
            str(row["label"]),
            f"{int(row['resnet_runs']):,}",
            f"{int(row['unet_runs']):,}",
            f"{float(row['avg_conductors']):.3f}",
            *[
                f"**{float(row['speedups'][unet_model_name]):.2f}x**"
                for unet_model_name in unet_model_names
            ],
        ]
        for row in comparison_rows
    ]

    window_counts = sorted(set(split_windows.values()))
    print("## Throughput")
    print()
    if len(window_counts) == 1:
        print(f"Tested {window_counts[0]} common windows per split.")
    else:
        joined = ", ".join(f"{label}={count}" for label, count in split_windows.items())
        print(f"Selected common windows per split: {joined}")
    print()
    print(_format_markdown_table(throughput_headers, throughput_rows))
    print()
    print(_format_markdown_table(comparison_headers, comparison_markdown_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
