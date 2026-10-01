from __future__ import annotations

import importlib
import sys
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_TRT_OPSET = 17
DEFAULT_TRT_WORKSPACE_BYTES = 1 << 30
DEFAULT_TRT_BUILDER_OPT_LEVEL = 5
DEFAULT_TRT_MAX_NUM_TACTICS = -1
_TENSORRT_LOGGER = None

if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from pipeline_model import (  # pylint: disable=wrong-import-position
    DEFAULT_FULL_PIPELINE_MONAI_CONFIG,
    FULL_PIPELINE_MONAI_CONFIGS,
    build_full_pipeline_unet,
    load_model_checkpoint,
)


ModelType = Literal["unet"]
ModelBackend = Literal["eager", "tensorrt_engine"]
_MODEL_CACHE: dict[tuple[Path, str, str, int, str], object] = {}


@dataclass(frozen=True)
class ModelSpec:
    model_type: ModelType = "unet"
    monai_config: str = DEFAULT_FULL_PIPELINE_MONAI_CONFIG
    checkpoint_path: Path | None = None
    compiled_engine_path: Path | None = None


@dataclass(frozen=True)
class LoadedModel:
    module: object
    outputs_qmap: bool
    backend: ModelBackend
    max_batch_size: int | None = None


class QMapInferenceWrapper(nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        raw = self.model(features)
        if isinstance(raw, (tuple, list)):
            raw = raw[0]
        return F.softplus(raw)


def resolve_device(name: str) -> torch.device:
    requested = str(name).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("--device must be one of: auto, cpu, cuda")
    return torch.device(requested)


def _validate_model_spec(spec: ModelSpec) -> None:
    if spec.checkpoint_path is None and spec.compiled_engine_path is None:
        raise ValueError("ModelSpec must provide either checkpoint_path or compiled_engine_path.")


def _clean_state_dict_items(items) -> OrderedDict[str, torch.Tensor]:
    cleaned: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, value in items:
        cleaned[str(key).replace("module.", "")] = value
    return cleaned


def _safe_torch_load(path: Path | str, *, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _infer_num_input_channels_from_state_dict(state_dict: OrderedDict[str, torch.Tensor], *, source: Path) -> int:
    conv_weights = [
        (key, value)
        for key, value in state_dict.items()
        if isinstance(value, torch.Tensor) and value.ndim == 4
    ]
    if not conv_weights:
        raise RuntimeError(f"Could not infer input channels from model without any convolution weights: {source}")

    first_key, first_weight = conv_weights[0]
    last_key, last_weight = conv_weights[-1]
    first_in_channels = int(first_weight.shape[1])
    last_out_channels = int(last_weight.shape[0])
    if first_in_channels == last_out_channels:
        return first_in_channels

    if any(int(weight.shape[0]) == first_in_channels for _key, weight in conv_weights):
        return first_in_channels
    if any(int(weight.shape[1]) == last_out_channels for _key, weight in conv_weights):
        return last_out_channels

    raise RuntimeError(
        "Could not infer num_input_channels from model boundary convolutions: "
        f"{source} first={first_key}{tuple(first_weight.shape)} "
        f"last={last_key}{tuple(last_weight.shape)}"
    )


def infer_checkpoint_input_channels(checkpoint_path: Path) -> int:
    checkpoint_path = checkpoint_path.resolve()
    info = _safe_torch_load(checkpoint_path, map_location="cpu")
    state_dict = info.get("state_dict", info)
    cleaned = _clean_state_dict_items(state_dict.items())
    return _infer_num_input_channels_from_state_dict(cleaned, source=checkpoint_path)


def _load_onnx_module():
    try:
        return importlib.import_module("onnx")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "TensorRT engine compilation requires the Python 'onnx' package. "
            "Install ONNX in the environment that runs full-pipeline/compile_models.py."
        ) from exc


def _load_tensorrt_module():
    try:
        return importlib.import_module("tensorrt")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "TensorRT engine compilation/runtime requires the Python 'tensorrt' package. "
            "Install TensorRT in the environment that runs full-pipeline/compile_models.py and full-pipeline/run.py."
        ) from exc


def _get_tensorrt_logger(trt_module):
    global _TENSORRT_LOGGER
    if _TENSORRT_LOGGER is None:
        _TENSORRT_LOGGER = trt_module.Logger(trt_module.Logger.WARNING)
    return _TENSORRT_LOGGER


def _torch_dtype_to_trt_dtype(torch_dtype: torch.dtype, trt_module):
    mapping = {
        torch.float32: trt_module.float32,
        torch.float16: trt_module.float16,
        torch.int32: trt_module.int32,
        torch.int8: trt_module.int8,
        torch.bool: trt_module.bool,
    }
    try:
        return mapping[torch_dtype]
    except KeyError as exc:
        raise RuntimeError(f"Unsupported torch dtype for TensorRT: {torch_dtype}") from exc


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


def _collect_parser_errors(parser) -> str:
    count = int(parser.num_errors)
    messages = []
    for idx in range(count):
        messages.append(str(parser.get_error(idx)))
    return "\n".join(messages) if messages else "<no parser errors reported>"


def _build_eager_model_from_checkpoint(
    spec: ModelSpec,
    *,
    num_input_channels: int,
    device: torch.device,
) -> torch.nn.Module:
    if spec.checkpoint_path is None:
        raise ValueError("checkpoint_path is required to build an eager model.")

    cache_key = (
        spec.checkpoint_path.resolve(),
        str(spec.model_type),
        str(spec.monai_config),
        int(num_input_channels),
        str(device),
    )
    cached = _MODEL_CACHE.get(cache_key)
    if isinstance(cached, torch.nn.Module):
        cached.eval()
        return cached

    if spec.model_type != "unet":
        raise ValueError(f"Unsupported model_type for full pipeline: {spec.model_type}")
    if spec.monai_config not in FULL_PIPELINE_MONAI_CONFIGS:
        raise ValueError(
            f"Unknown MONAI config '{spec.monai_config}'. Expected one of {sorted(FULL_PIPELINE_MONAI_CONFIGS)}"
        )

    model = build_full_pipeline_unet(
        num_input_channels=num_input_channels,
        monai_config=str(spec.monai_config),
    )
    model.to(device)
    model.eval()
    load_model_checkpoint(spec.checkpoint_path, model, device)
    _MODEL_CACHE[cache_key] = model
    return model


class TensorRTEngineModule:
    def __init__(self, engine_path: Path, *, device: torch.device) -> None:
        if device.type != "cuda":
            raise RuntimeError("TensorRT engine inference requires a CUDA device.")
        self.engine_path = engine_path.resolve()
        self.device = torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)
        # Run TensorRT on a dedicated non-default CUDA stream to avoid implicit
        # default-stream synchronizations inside enqueueV3/execute_async_v2.
        self.execution_stream = torch.cuda.Stream(device=self.device)
        self.trt = _load_tensorrt_module()
        self.logger = _get_tensorrt_logger(self.trt)
        self.runtime = self.trt.Runtime(self.logger)
        serialized_engine = self.engine_path.read_bytes()
        self.engine = self.runtime.deserialize_cuda_engine(serialized_engine)
        if self.engine is None:
            raise RuntimeError(f"Could not deserialize TensorRT engine: {self.engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"Could not create TensorRT execution context: {self.engine_path}")
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
        if int(self.input_max_shape[2]) != int(self.input_max_shape[3]):
            raise RuntimeError(
                "Only square TensorRT input shapes are supported for the full pipeline, "
                f"got {tuple(int(v) for v in self.input_max_shape)}"
            )

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
                    "Full-pipeline TensorRT engines must expose exactly one input and one output tensor, "
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
                "Full-pipeline TensorRT engines must expose exactly one input and one output binding, "
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
                raise RuntimeError(f"TensorRT could not set input shape {shape} for {self.engine_path}")
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
            raise RuntimeError(f"TensorRT engine execution failed: {self.engine_path}")
        caller_stream.wait_stream(self.execution_stream)

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        if features.device.type != "cuda":
            raise RuntimeError("TensorRT engine inference requires CUDA tensors.")
        if features.device.index != self.device.index:
            raise RuntimeError(
                f"TensorRT engine was loaded on cuda:{self.device.index} but received features on {features.device}"
            )
        if features.ndim != 4:
            raise ValueError(f"TensorRT full-pipeline features must have shape [B, C, H, W], got {tuple(features.shape)}")

        batch_size = int(features.shape[0])
        if batch_size <= 0:
            raise ValueError(f"TensorRT full-pipeline batch size must be positive, got {batch_size}")
        if batch_size > int(self.max_batch_size):
            raise ValueError(
                f"TensorRT engine {self.engine_path} supports max batch {self.max_batch_size}, got {batch_size}"
            )
        if int(features.shape[1]) != int(self.num_input_channels):
            raise ValueError(
                f"TensorRT engine {self.engine_path} expects {self.num_input_channels} channels, got {int(features.shape[1])}"
            )
        if int(features.shape[2]) != int(self.target_size) or int(features.shape[3]) != int(self.target_size):
            raise ValueError(
                f"TensorRT engine {self.engine_path} expects spatial {self.target_size}x{self.target_size}, "
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


def build_model(spec: ModelSpec, *, num_input_channels: int, device: torch.device) -> LoadedModel:
    _validate_model_spec(spec)

    if spec.compiled_engine_path is not None:
        compiled_path = spec.compiled_engine_path.resolve()
        cache_key = (
            compiled_path,
            "tensorrt_engine",
            str(spec.monai_config),
            int(num_input_channels),
            str(device),
        )
        cached = _MODEL_CACHE.get(cache_key)
        if not isinstance(cached, TensorRTEngineModule):
            engine = TensorRTEngineModule(compiled_path, device=device)
            _MODEL_CACHE[cache_key] = engine
            cached = engine
        return LoadedModel(
            module=cached,
            outputs_qmap=True,
            backend="tensorrt_engine",
            max_batch_size=int(cached.max_batch_size),
        )

    eager = _build_eager_model_from_checkpoint(
        spec,
        num_input_channels=num_input_channels,
        device=device,
    )
    eager.eval()
    return LoadedModel(module=eager, outputs_qmap=False, backend="eager", max_batch_size=None)


TRT_PRECISIONS = ("fp16", "bf16", "fp32")


def _set_engine_precision(config, builder, trt, precision: str) -> None:
    """fp16: fastest, but activations above 65504 overflow (the released coupling model reaches
    ~5e6 in its deepest U-Net levels and returns NaN); bf16: FP32 range with bf16 storage (Ampere or
    newer); fp32: TF32 tensor-core math."""
    if precision not in TRT_PRECISIONS:
        raise ValueError(f"Unsupported TensorRT precision '{precision}', expected one of {TRT_PRECISIONS}")
    if precision == "fp16":
        if not bool(builder.platform_has_fast_fp16):
            raise RuntimeError(
                "TensorRT FP16 compilation was requested but the current platform does not advertise fast FP16 support."
            )
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "bf16":
        if not hasattr(trt.BuilderFlag, "BF16"):
            raise RuntimeError("This TensorRT version does not support BF16 engines.")
        config.set_flag(trt.BuilderFlag.BF16)


def compile_eager_qmap_model_to_tensorrt_engine(
    eager: torch.nn.Module,
    *,
    num_input_channels: int,
    device: torch.device,
    output_path: Path,
    target_size: int,
    opt_batch_size: int,
    max_batch_size: int,
    model_label: str = "eager_model",
    precision: str = "fp16",
) -> Path:
    if device.type != "cuda":
        raise RuntimeError("TensorRT engine compilation requires CUDA.")
    if int(target_size) <= 0:
        raise ValueError(f"target_size must be positive, got {target_size}")
    if int(opt_batch_size) <= 0:
        raise ValueError(f"opt_batch_size must be positive, got {opt_batch_size}")
    if int(max_batch_size) <= 0:
        raise ValueError(f"max_batch_size must be positive, got {max_batch_size}")
    if int(opt_batch_size) > int(max_batch_size):
        raise ValueError(
            f"opt_batch_size must not exceed max_batch_size, got opt={opt_batch_size} max={max_batch_size}"
        )

    _load_onnx_module()
    trt = _load_tensorrt_module()

    eager = eager.to(device=device, dtype=torch.float32)
    eager.eval()
    wrapper = QMapInferenceWrapper(eager).eval()

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_name = "features"
    output_name = "qmap"
    dynamic_axes = None
    if int(max_batch_size) > 1:
        dynamic_axes = {
            input_name: {0: "batch"},
            output_name: {0: "batch"},
        }

    example_input = torch.randn(
        (1, int(num_input_channels), int(target_size), int(target_size)),
        device=device,
        dtype=torch.float32,
    )

    with tempfile.TemporaryDirectory(prefix="cnncap_flash_trt_") as tmp_dir_str:
        onnx_path = Path(tmp_dir_str) / f"{output_path.stem}.onnx"
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
                opset_version=int(DEFAULT_TRT_OPSET),
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
                f"TensorRT could not parse exported ONNX for {model_label}:\n{_collect_parser_errors(parser)}"
            )

        config = builder.create_builder_config()
        _set_engine_precision(config, builder, trt, precision)
        if hasattr(config, "builder_optimization_level"):
            config.builder_optimization_level = int(DEFAULT_TRT_BUILDER_OPT_LEVEL)
        if hasattr(config, "max_num_tactics"):
            config.max_num_tactics = int(DEFAULT_TRT_MAX_NUM_TACTICS)
        if hasattr(config, "set_memory_pool_limit") and hasattr(trt, "MemoryPoolType"):
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(DEFAULT_TRT_WORKSPACE_BYTES))
        elif hasattr(config, "max_workspace_size"):
            config.max_workspace_size = int(DEFAULT_TRT_WORKSPACE_BYTES)

        if int(max_batch_size) > 1:
            profile = builder.create_optimization_profile()
            profile.set_shape(
                input_name,
                (
                    1,
                    int(num_input_channels),
                    int(target_size),
                    int(target_size),
                ),
                (
                    int(opt_batch_size),
                    int(num_input_channels),
                    int(target_size),
                    int(target_size),
                ),
                (
                    int(max_batch_size),
                    int(num_input_channels),
                    int(target_size),
                    int(target_size),
                ),
            )
            config.add_optimization_profile(profile)

        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError(f"TensorRT failed to build an engine for {model_label}")
        output_path.write_bytes(bytes(serialized_engine))
    return output_path


def compile_model_to_tensorrt_engine(
    spec: ModelSpec,
    *,
    num_input_channels: int,
    device: torch.device,
    output_path: Path,
    target_size: int,
    opt_batch_size: int,
    max_batch_size: int,
    precision: str = "fp16",
) -> Path:
    if spec.checkpoint_path is None:
        raise ValueError("compile_model_to_tensorrt_engine requires spec.checkpoint_path.")

    eager = _build_eager_model_from_checkpoint(
        spec,
        num_input_channels=num_input_channels,
        device=device,
    )
    return compile_eager_qmap_model_to_tensorrt_engine(
        eager,
        num_input_channels=num_input_channels,
        device=device,
        output_path=output_path,
        target_size=target_size,
        opt_batch_size=opt_batch_size,
        max_batch_size=max_batch_size,
        model_label=str(spec.checkpoint_path),
        precision=precision,
    )


def forward_qmap(model: LoadedModel, features: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        raw = model.module(features)
        if model.outputs_qmap:
            return raw
        if isinstance(raw, (tuple, list)):
            raw = raw[0]
        return F.softplus(raw)
