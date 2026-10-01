#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Sequence, Tuple

import numpy as np
import torch


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
DEFAULT_TARGET_SIZE = 224
DEFAULT_DEVICE_NAME = "cuda"
DEFAULT_PREFERRED_BATCH_SIZE = 24
DEFAULT_ENV_BATCH_SIZE = 64
# Tiles staged per native DEF pass; bounded so full-chip runs fit in GPU memory.
DEFAULT_TILE_STREAM_SIZE = 64
MODEL_OUTPUT_TO_FARADS = 1e-15

for extra_path in (str(REPO_ROOT), str(THIS_DIR)):
    if extra_path not in sys.path:
        sys.path.insert(0, extra_path)

from idmap_cuda_runtime import load_idmap_expand_cuda_extension  # pylint: disable=wrong-import-position
from def_fast_density import load_fast_lefdef_parser_extension  # pylint: disable=wrong-import-position
from flash_common.grouped_sparse_reduce import (  # pylint: disable=wrong-import-position
    load_sparse_reduce_cuda_extension,
    reduce_qmap_to_all_conductors_sparse,
)
from model_runtime import (  # pylint: disable=wrong-import-position
    ModelSpec,
    build_model,
    forward_qmap,
    resolve_device,
)
from spef_runtime import (  # pylint: disable=wrong-import-position
    IndexedSpefAccumulator,
)
from tile_utils import (  # pylint: disable=wrong-import-position
    DEFAULT_TILE_CONTEXT_UM,
    DEFAULT_TILE_SIZE_UM,
    TilingSummary,
    TileJob,
    build_tiled_window_jobs,
    build_tiling_summary,
)
from window_runtime import (  # pylint: disable=wrong-import-position
    build_runtime_config,
    materialize_window_features,
    prepare_tiled_window_rasters,
    prepare_window_features,
)


@dataclass(frozen=True)
class PipelineConfig:
    tech_path: Path
    target_size: int
    device_name: str
    total_batch_size: int
    master_batch_size: int
    tile_stream_size: int
    compiled_model_manifest: Path
    total_compiled_model: Path
    env_compiled_model: Path


@dataclass(frozen=True)
class StagedWindow:
    window_id: str
    occupied: torch.Tensor
    full_local_map: torch.Tensor
    owned_local_counts: torch.Tensor
    owned_sparse_indices: torch.Tensor
    owned_sparse_counts: torch.Tensor
    owned_query_ids: torch.Tensor
    owned_query_output_indices: Tuple[int, ...]
    local_output_indices: torch.Tensor
    visible_master_local_ids: Tuple[int, ...]
    visible_master_output_indices: Tuple[int, ...]


@dataclass(frozen=True)
class EnvWorkItem:
    window_index: int
    master_local_id: int
    master_output_index: int


@dataclass(frozen=True)
class TileTensorCache:
    occupied: torch.Tensor
    full_local_map: torch.Tensor
    pixel_output_indices: torch.Tensor
    owned_local_counts: torch.Tensor
    owned_sparse_indices: torch.Tensor
    owned_sparse_counts: torch.Tensor
    local_output_indices: torch.Tensor


@dataclass(frozen=True)
class EnvWorkPlan:
    items: Tuple[EnvWorkItem, ...]
    window_indices: torch.Tensor
    master_output_indices: torch.Tensor


@dataclass(frozen=True)
class WindowJob:
    window_id: str
    solve_target_size: int
    pixel_resolution_um: float | None


@dataclass(frozen=True)
class StageTimingBreakdown:
    tile_planning_seconds: float
    read_and_stage_to_gpu_seconds: float
    total_inference_and_reduction_seconds: float
    env_feature_build_seconds: float
    env_inference_and_reduction_seconds: float
    gpu_to_cpu_and_aggregate_seconds: float
    spef_write_seconds: float

    @property
    def gpu_inference_and_reduction_seconds(self) -> float:
        return (
            float(self.total_inference_and_reduction_seconds)
            + float(self.env_feature_build_seconds)
            + float(self.env_inference_and_reduction_seconds)
        )

    @property
    def total_seconds(self) -> float:
        return (
            float(self.tile_planning_seconds)
            + float(self.read_and_stage_to_gpu_seconds)
            + float(self.total_inference_and_reduction_seconds)
            + float(self.env_feature_build_seconds)
            + float(self.env_inference_and_reduction_seconds)
            + float(self.gpu_to_cpu_and_aggregate_seconds)
            + float(self.spef_write_seconds)
        )


@dataclass(frozen=True)
class WindowSolveResult:
    window_id: str
    output_path: Path
    elapsed_seconds: float
    timing: StageTimingBreakdown
    tile_count: int
    env_work_item_count: int
    total_batch_size: int
    env_batch_size: int
    tile_stream_size: int


@dataclass(frozen=True)
class StartupTiming:
    device: str
    model_load_seconds: float
    extension_load_seconds: float
    startup_seconds: float


@dataclass(frozen=True)
class CompiledModelMetadata:
    artifact_format: str
    num_input_channels: int
    active_layers: Tuple[str, ...]
    target_size: int
    total_max_batch_size: int
    env_max_batch_size: int


def _read_def_diearea_um(def_path: Path) -> Tuple[float, float, float, float]:
    units = 2000.0
    with def_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line.startswith("UNITS DISTANCE MICRONS"):
                tokens = line.replace(";", " ").split()
                if len(tokens) >= 4:
                    units = float(tokens[3])
            elif line.startswith("DIEAREA"):
                tokens = line.replace("(", " ").replace(")", " ").replace(";", " ").split()
                numeric = [float(token) for token in tokens[1:] if token]
                if len(numeric) < 4:
                    raise RuntimeError(f"Could not parse DIEAREA line in {def_path}: {line}")
                return (
                    numeric[0] / units,
                    numeric[1] / units,
                    numeric[2] / units,
                    numeric[3] / units,
                )
    raise RuntimeError(f"DEF file did not contain a DIEAREA line: {def_path}")


def _build_window_job(def_path: Path, config: PipelineConfig) -> WindowJob:
    return WindowJob(
        window_id=def_path.stem,
        solve_target_size=int(config.target_size),
        pixel_resolution_um=None,
    )


def _plan_tiled_design(def_path: Path, config: PipelineConfig) -> tuple[TilingSummary, Tuple[TileJob, ...]]:
    die_bounds_um = _read_def_diearea_um(def_path)
    tile_jobs = build_tiled_window_jobs(
        def_path.stem,
        die_bounds_um=die_bounds_um,
        target_size=int(config.target_size),
        tile_size_um=float(DEFAULT_TILE_SIZE_UM),
        tile_context_um=float(DEFAULT_TILE_CONTEXT_UM),
    )
    summary = build_tiling_summary(
        def_path.stem,
        die_bounds_um=die_bounds_um,
        target_size=int(config.target_size),
        tile_jobs=tile_jobs,
        tile_size_um=float(DEFAULT_TILE_SIZE_UM),
        tile_context_um=float(DEFAULT_TILE_CONTEXT_UM),
    )
    return summary, tile_jobs


def _load_compiled_model_metadata(manifest_path: Path) -> CompiledModelMetadata:
    manifest_path = manifest_path.resolve()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Compiled model manifest was not found: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Compiled model manifest is not valid JSON: {manifest_path}") from exc

    if not isinstance(payload, dict):
        raise RuntimeError(f"Compiled model manifest must contain a JSON object: {manifest_path}")

    try:
        artifact_format = str(payload["artifact_format"])
    except KeyError as exc:
        raise RuntimeError(f"Compiled model manifest is missing artifact_format: {manifest_path}") from exc
    if artifact_format != "tensorrt_engine":
        raise RuntimeError(
            "Compiled model manifest does not describe TensorRT engines: "
            f"artifact_format={artifact_format} manifest={manifest_path}"
        )

    try:
        num_input_channels = int(payload["num_input_channels"])
        target_size = int(payload["target_size"])
    except KeyError as exc:
        raise RuntimeError(f"Compiled model manifest is missing a required top-level field: {manifest_path}") from exc
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Compiled model manifest has invalid numeric metadata: {manifest_path}") from exc

    active_layers_raw = payload.get("active_layers")
    if not isinstance(active_layers_raw, list) or not active_layers_raw:
        raise RuntimeError(f"Compiled model manifest must provide a non-empty active_layers list: {manifest_path}")
    active_layers = tuple(str(layer) for layer in active_layers_raw)
    if len(active_layers) != int(num_input_channels):
        raise RuntimeError(
            f"Compiled model manifest active_layers length mismatch: "
            f"num_input_channels={num_input_channels} active_layers={len(active_layers)} manifest={manifest_path}"
        )

    models_payload = payload.get("models")
    if not isinstance(models_payload, dict):
        raise RuntimeError(f"Compiled model manifest must provide a models object: {manifest_path}")
    try:
        total_max_batch_size = int(models_payload["total"]["max_batch_size"])
        env_max_batch_size = int(models_payload["env"]["max_batch_size"])
    except KeyError as exc:
        raise RuntimeError(f"Compiled model manifest is missing per-model max_batch_size metadata: {manifest_path}") from exc
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Compiled model manifest has invalid max_batch_size metadata: {manifest_path}") from exc

    if num_input_channels <= 0:
        raise RuntimeError(
            f"Compiled model manifest must declare a positive num_input_channels, got {num_input_channels}: {manifest_path}"
        )
    if target_size <= 0:
        raise RuntimeError(f"Compiled model manifest must declare a positive target_size, got {target_size}: {manifest_path}")
    if total_max_batch_size <= 0 or env_max_batch_size <= 0:
        raise RuntimeError(
            f"Compiled model manifest must declare positive max_batch_size values: {manifest_path}"
        )

    return CompiledModelMetadata(
        artifact_format=artifact_format,
        num_input_channels=num_input_channels,
        active_layers=active_layers,
        target_size=target_size,
        total_max_batch_size=total_max_batch_size,
        env_max_batch_size=env_max_batch_size,
    )


def _resolve_runtime_config(config: PipelineConfig, metadata: CompiledModelMetadata):
    if int(metadata.target_size) != int(config.target_size):
        raise RuntimeError(
            f"Compiled model target size mismatch: manifest={metadata.target_size} runtime={config.target_size}"
        )

    full_runtime_config = build_runtime_config(config.tech_path)
    available_layers = list(full_runtime_config.channel_layers)
    missing_layers = [str(layer) for layer in metadata.active_layers if str(layer) not in available_layers]
    if missing_layers:
        raise RuntimeError(
            "Compiled model active_layers are not available in the technology stack: "
            f"missing={missing_layers}"
        )
    return build_runtime_config(config.tech_path, selected_layers=metadata.active_layers)


def _synchronize_device(device: torch.device | None) -> None:
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure_elapsed(action, *, device: torch.device | None = None):
    _synchronize_device(device)
    started_at = time.perf_counter()
    result = action()
    _synchronize_device(device)
    return result, float(time.perf_counter() - started_at)


def _chunked(values: Sequence, chunk_size: int) -> Iterable[Sequence]:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    for start in range(0, len(values), chunk_size):
        yield values[start:start + chunk_size]


def _pad_long_1d_tensors(
    tensors: Sequence[torch.Tensor],
    *,
    fill_value: int = 0,
) -> torch.Tensor:
    if not tensors:
        raise ValueError("Cannot pad an empty tensor list.")
    device = tensors[0].device
    max_len = max(int(tensor.shape[0]) for tensor in tensors)
    out = torch.full((len(tensors), max_len), int(fill_value), device=device, dtype=torch.long)
    for batch_idx, tensor in enumerate(tensors):
        length = int(tensor.shape[0])
        if length > 0:
            out[batch_idx, :length] = tensor.to(device=device, dtype=torch.long)
    return out


def _pad_long_2d_tensors(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    if not tensors:
        raise ValueError("Cannot pad an empty tensor list.")
    device = tensors[0].device
    max_rows = max(int(tensor.shape[0]) for tensor in tensors)
    max_cols = max(int(tensor.shape[1]) for tensor in tensors)
    out = torch.zeros((len(tensors), max_rows, max_cols), device=device, dtype=torch.long)
    for batch_idx, tensor in enumerate(tensors):
        rows = int(tensor.shape[0])
        cols = int(tensor.shape[1])
        if rows > 0 and cols > 0:
            out[batch_idx, :rows, :cols] = tensor.to(device=device, dtype=torch.long)
    return out


def _build_tile_tensor_cache(staged_windows: Sequence[StagedWindow]) -> TileTensorCache:
    if not staged_windows:
        raise ValueError("Cannot build tile tensors from an empty window list.")
    occupied = torch.stack([staged.occupied for staged in staged_windows], dim=0).to(dtype=torch.float32).contiguous()
    full_local_map = torch.stack([staged.full_local_map for staged in staged_windows], dim=0).to(dtype=torch.long).contiguous()
    local_output_indices = _pad_long_1d_tensors(
        [staged.local_output_indices for staged in staged_windows],
        fill_value=-1,
    ).contiguous()
    pixel_output_indices = torch.gather(
        local_output_indices,
        1,
        full_local_map.reshape(len(staged_windows), -1),
    ).reshape_as(full_local_map).contiguous()
    return TileTensorCache(
        occupied=occupied,
        full_local_map=full_local_map,
        pixel_output_indices=pixel_output_indices,
        owned_local_counts=_pad_long_1d_tensors([staged.owned_local_counts for staged in staged_windows]).contiguous(),
        owned_sparse_indices=_pad_long_2d_tensors([staged.owned_sparse_indices for staged in staged_windows]).contiguous(),
        owned_sparse_counts=_pad_long_1d_tensors([staged.owned_sparse_counts for staged in staged_windows]).contiguous(),
        local_output_indices=local_output_indices,
    )


def _build_env_work_plan(staged_windows: Sequence[StagedWindow], *, device: torch.device) -> EnvWorkPlan:
    work_items: List[EnvWorkItem] = []
    window_indices: List[int] = []
    master_output_indices: List[int] = []
    for window_index, staged in enumerate(staged_windows):
        if not staged.owned_query_output_indices:
            continue
        for master_local_id, master_output_index in zip(
            staged.visible_master_local_ids,
            staged.visible_master_output_indices,
        ):
            work_items.append(
                EnvWorkItem(
                    window_index=int(window_index),
                    master_local_id=int(master_local_id),
                    master_output_index=int(master_output_index),
                )
            )
            window_indices.append(int(window_index))
            master_output_indices.append(int(master_output_index))
    return EnvWorkPlan(
        items=tuple(work_items),
        window_indices=torch.as_tensor(window_indices, device=device, dtype=torch.long),
        master_output_indices=torch.as_tensor(master_output_indices, device=device, dtype=torch.long),
    )


def _build_env_feature_batch_from_cache(
    cache: TileTensorCache,
    window_indices: torch.Tensor | Sequence[EnvWorkItem],
    master_output_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    if not isinstance(window_indices, torch.Tensor):
        work_items = tuple(window_indices)
        window_indices = torch.as_tensor(
            [int(item.window_index) for item in work_items],
            device=cache.occupied.device,
            dtype=torch.long,
        )
        master_output_indices = torch.as_tensor(
            [int(item.master_output_index) for item in work_items],
            device=cache.occupied.device,
            dtype=torch.long,
        )
    if master_output_indices is None:
        raise ValueError("master_output_indices must be provided with tensor window_indices.")
    if int(window_indices.numel()) == 0:
        raise ValueError("Cannot build an env feature batch from an empty work list.")
    batch = cache.occupied.index_select(0, window_indices).clone()
    pixel_output_indices = cache.pixel_output_indices.index_select(0, window_indices)
    batch.sub_(pixel_output_indices.eq(master_output_indices.view(-1, 1, 1, 1)).to(dtype=batch.dtype) * 2.0)
    return batch.contiguous()


def _reduce_batched_owned_queries(
    q_map: torch.Tensor,
    cache: TileTensorCache,
    window_indices: Sequence[int] | torch.Tensor,
) -> torch.Tensor:
    if isinstance(window_indices, torch.Tensor):
        if int(window_indices.numel()) == 0:
            raise ValueError("Cannot reduce an empty window batch.")
        index_tensor = window_indices.to(device=cache.owned_local_counts.device, dtype=torch.long, non_blocking=True)
    else:
        if not window_indices:
            raise ValueError("Cannot reduce an empty window batch.")
        index_tensor = torch.as_tensor(window_indices, device=cache.owned_local_counts.device, dtype=torch.long)
    local_counts = cache.owned_local_counts.index_select(0, index_tensor)
    sparse_indices = cache.owned_sparse_indices.index_select(0, index_tensor)
    sparse_counts = cache.owned_sparse_counts.index_select(0, index_tensor)
    reduced, _areas = reduce_qmap_to_all_conductors_sparse(
        q_map,
        local_counts,
        sparse_indices,
        sparse_counts,
        reduction="sum",
    )
    return reduced


def _preload_pipeline_extensions(device: torch.device) -> None:
    load_fast_lefdef_parser_extension()
    if device.type == "cuda":
        load_idmap_expand_cuda_extension()
        load_sparse_reduce_cuda_extension()


def _accumulate_total_predictions(
    spef_accumulator: IndexedSpefAccumulator,
    staged: StagedWindow,
    preds: torch.Tensor,
) -> None:
    if not staged.owned_query_output_indices:
        return
    values = preds[0].detach().cpu().numpy().astype(np.float64, copy=False)
    spef_accumulator.add_total_values(
        staged.owned_query_output_indices,
        values,
        scale=float(MODEL_OUTPUT_TO_FARADS),
    )


def _accumulate_coupling_predictions(
    spef_accumulator: IndexedSpefAccumulator,
    staged: StagedWindow,
    master_output_indices: Sequence[int],
    preds: torch.Tensor,
) -> None:
    if not staged.owned_query_output_indices or not master_output_indices:
        return
    spef_accumulator.add_directed_values(
        master_output_indices,
        staged.owned_query_output_indices,
        preds.detach().cpu().numpy().astype(np.float64, copy=False),
        scale=float(MODEL_OUTPUT_TO_FARADS),
    )


def _accumulate_total_prediction_batch(
    spef_accumulator: IndexedSpefAccumulator,
    staged_windows: Sequence[StagedWindow],
    window_indices: Sequence[int],
    reduced: torch.Tensor,
) -> None:
    reduced_cpu = reduced.detach().cpu().numpy().astype(np.float64, copy=False)
    for row_idx, window_index in enumerate(window_indices):
        staged = staged_windows[int(window_index)]
        if not staged.owned_query_output_indices:
            continue
        query_ids = staged.owned_query_ids.detach().cpu().numpy()
        values = reduced_cpu[row_idx, query_ids]
        spef_accumulator.add_total_values(
            staged.owned_query_output_indices,
            values,
            scale=float(MODEL_OUTPUT_TO_FARADS),
        )


def _accumulate_env_prediction_batch(
    spef_accumulator: IndexedSpefAccumulator,
    staged_windows: Sequence[StagedWindow],
    work_items: Sequence[EnvWorkItem],
    reduced: torch.Tensor,
) -> None:
    if not work_items:
        return
    reduced_cpu = reduced.detach().cpu().numpy().astype(np.float64, copy=False)
    rows_by_window: dict[int, list[int]] = {}
    for row_idx, item in enumerate(work_items):
        rows_by_window.setdefault(int(item.window_index), []).append(row_idx)
    for window_index, row_indices in rows_by_window.items():
        staged = staged_windows[int(window_index)]
        if not staged.owned_query_output_indices:
            continue
        query_ids = staged.owned_query_ids.detach().cpu().numpy()
        values = reduced_cpu[np.asarray(row_indices, dtype=np.int64)[:, None], query_ids[None, :]]
        spef_accumulator.add_directed_values(
            [int(work_items[row_idx].master_output_index) for row_idx in row_indices],
            staged.owned_query_output_indices,
            values,
            scale=float(MODEL_OUTPUT_TO_FARADS),
        )


def _unique_master_queries_by_output(
    local_ids: Sequence[int],
    output_indices: Sequence[int],
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    unique_local_ids: List[int] = []
    unique_output_indices: List[int] = []
    seen_outputs: set[int] = set()
    for local_id in local_ids:
        local_index = int(local_id)
        if local_index <= 0 or local_index > len(output_indices):
            raise ValueError(
                f"master local id {local_index} is outside the available conductor range "
                f"[1, {len(output_indices)}]"
            )
        output_index = int(output_indices[local_index - 1])
        if output_index in seen_outputs:
            continue
        seen_outputs.add(output_index)
        unique_local_ids.append(local_index)
        unique_output_indices.append(output_index)
    return tuple(unique_local_ids), tuple(unique_output_indices)


def _build_staged_window(window, register_nets: Callable[[Sequence[str]], Tuple[int, ...]]) -> StagedWindow:
    output_indices = register_nets(window.real_conductor_names)
    owned_query_local_ids = window.owned_query_local_ids.contiguous()
    visible_master_local_ids, visible_master_output_indices = _unique_master_queries_by_output(
        window.visible_master_local_ids.tolist(),
        output_indices,
    )
    local_output_indices = torch.full(
        (int(window.full_local_map.max().item()) + 1,),
        -1,
        device=window.full_local_map.device,
        dtype=torch.long,
    )
    if output_indices:
        local_output_indices[1:len(output_indices) + 1] = torch.as_tensor(
            output_indices,
            device=window.full_local_map.device,
            dtype=torch.long,
        )
    return StagedWindow(
        window_id=window.window_id,
        occupied=window.occupied.contiguous(),
        full_local_map=window.full_local_map.contiguous(),
        owned_local_counts=window.owned_local_counts.contiguous(),
        owned_sparse_indices=window.owned_sparse_indices.contiguous(),
        owned_sparse_counts=window.owned_sparse_counts.contiguous(),
        owned_query_ids=owned_query_local_ids,
        owned_query_output_indices=tuple(
            int(output_indices[int(local_id) - 1])
            for local_id in owned_query_local_ids.tolist()
        ),
        local_output_indices=local_output_indices.contiguous(),
        visible_master_local_ids=visible_master_local_ids,
        visible_master_output_indices=visible_master_output_indices,
    )


class InferenceSession:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.device = resolve_device(config.device_name)
        self.metadata = _load_compiled_model_metadata(config.compiled_model_manifest)
        self.runtime_config = _resolve_runtime_config(config, self.metadata)
        self.num_input_channels = len(self.runtime_config.channel_layers)
        self.total_model = build_model(
            ModelSpec(
                compiled_engine_path=config.total_compiled_model.resolve(),
            ),
            num_input_channels=self.num_input_channels,
            device=self.device,
        )
        self.env_model = build_model(
            ModelSpec(
                compiled_engine_path=config.env_compiled_model.resolve(),
            ),
            num_input_channels=self.num_input_channels,
            device=self.device,
        )
        if int(config.master_batch_size) > int(self.metadata.env_max_batch_size):
            raise RuntimeError(
                f"Runtime master_batch_size={config.master_batch_size} exceeds env engine max_batch_size={self.metadata.env_max_batch_size}"
            )
        if int(config.total_batch_size) > int(self.metadata.total_max_batch_size):
            raise RuntimeError(
                f"Runtime total_batch_size={config.total_batch_size} exceeds total engine max_batch_size={self.metadata.total_max_batch_size}"
            )
        if int(config.tile_stream_size) < 0:
            raise ValueError(f"Runtime tile_stream_size must be non-negative, got {config.tile_stream_size}")

    def _stage_window(
        self,
        def_path: Path,
        window_job: WindowJob,
        *,
        register_nets: Callable[[Sequence[str]], Tuple[int, ...]],
    ) -> StagedWindow:
        window = prepare_window_features(
            def_path,
            tech_path=self.config.tech_path,
            device=self.device,
            target_size=int(window_job.solve_target_size),
            pixel_resolution=window_job.pixel_resolution_um,
            selected_layers=self.runtime_config.channel_layers,
            include_supply_nets=True,
            window_id=window_job.window_id,
        )
        return _build_staged_window(window, register_nets)

    def _stage_tile(
        self,
        def_path: Path,
        tile_job: TileJob,
        *,
        register_nets: Callable[[Sequence[str]], Tuple[int, ...]],
    ) -> StagedWindow:
        window = prepare_window_features(
            def_path,
            tech_path=self.config.tech_path,
            device=self.device,
            target_size=int(tile_job.solve_target_size),
            pixel_resolution=float(tile_job.pixel_resolution_um),
            raster_bounds=tile_job.raster_bounds,
            ownership_bounds=tile_job.ownership_bounds,
            selected_layers=self.runtime_config.channel_layers,
            include_supply_nets=True,
            window_id=tile_job.window_id,
            patch_grid_origin=tile_job.patch_grid_origin,
            patch_size_um=float(tile_job.patch_size_um),
            patch_grid_shape=tile_job.patch_grid_shape,
            fragment_keep_bounds=tile_job.margin_bounds,
            reduce_visible_conductors=False,
            master_conductors_from_ownership=True,
            restrict_masters_to_routes=False,
        )
        return _build_staged_window(window, register_nets)

    def _stage_tile_batch(
        self,
        def_path: Path,
        tile_jobs: Sequence[TileJob],
        *,
        register_nets: Callable[[Sequence[str]], Tuple[int, ...]],
    ) -> List[StagedWindow]:
        prepared_windows = prepare_tiled_window_rasters(
            def_path,
            tech_path=self.config.tech_path,
            tile_jobs=tile_jobs,
            selected_layers=self.runtime_config.channel_layers,
            include_supply_nets=True,
        )
        staged_windows: List[StagedWindow] = []
        for prepared_window in prepared_windows:
            window = materialize_window_features(prepared_window, device=self.device)
            staged_windows.append(_build_staged_window(window, register_nets))
        return staged_windows

    def _reduce_owned_queries(self, q_map: torch.Tensor, staged: StagedWindow) -> torch.Tensor:
        batch_size = int(q_map.shape[0])
        if int(staged.owned_query_ids.numel()) == 0:
            return torch.empty((batch_size, 0), device=q_map.device, dtype=q_map.dtype)
        if q_map.device.type != "cuda":
            raise RuntimeError(
                f"Full-pipeline inference only supports the CUDA sparse-reduction path, got q_map on {q_map.device}."
            )
        reduced, _areas = reduce_qmap_to_all_conductors_sparse(
            q_map,
            staged.owned_local_counts.unsqueeze(0).expand(batch_size, -1),
            staged.owned_sparse_indices.unsqueeze(0).expand(batch_size, -1, -1),
            staged.owned_sparse_counts.unsqueeze(0).expand(batch_size, -1),
            reduction="sum",
        )
        return torch.gather(reduced, 1, staged.owned_query_ids.unsqueeze(0).expand(batch_size, -1))

    def _predict_total_queries(self, staged: StagedWindow) -> torch.Tensor:
        q_map = forward_qmap(self.total_model, staged.occupied.unsqueeze(0))
        return self._reduce_owned_queries(q_map, staged)

    def _build_env_feature_batch(self, staged: StagedWindow, master_local_ids: Sequence[int]) -> torch.Tensor:
        if not master_local_ids:
            raise ValueError("master_local_ids cannot be empty")
        master_ids = torch.as_tensor(master_local_ids, device=staged.full_local_map.device, dtype=torch.long)
        if bool((master_ids <= 0).any()):
            raise ValueError(f"master local ids must be positive, got {tuple(int(v) for v in master_ids.tolist())}")
        if bool((master_ids > int(staged.full_local_map.max().item())).any()) and int(staged.full_local_map.max().item()) > 0:
            raise ValueError(
                f"master local ids exceed available local ids: ids={tuple(int(v) for v in master_ids.tolist())} "
                f"available_max={int(staged.full_local_map.max().item())}"
            )
        batch = staged.occupied.unsqueeze(0).expand(len(master_local_ids), -1, -1, -1).clone()
        master_output_indices = staged.local_output_indices.index_select(0, master_ids)
        if bool((master_output_indices < 0).any()):
            raise ValueError(
                "master local ids must map to real output nets, got "
                f"{tuple(int(v) for v in master_ids.tolist())}"
            )
        pixel_output_indices = staged.local_output_indices.index_select(0, staged.full_local_map.reshape(-1))
        pixel_output_indices = pixel_output_indices.reshape_as(staged.full_local_map)
        master_mask = pixel_output_indices.unsqueeze(0).eq(master_output_indices.view(-1, 1, 1, 1))
        batch.sub_(master_mask.to(dtype=batch.dtype) * 2.0)
        return batch.contiguous()

    def _predict_coupling_chunk(self, staged: StagedWindow, master_local_ids: Sequence[int]) -> torch.Tensor:
        batch = self._build_env_feature_batch(staged, master_local_ids)
        q_map = forward_qmap(self.env_model, batch)
        return self._reduce_owned_queries(q_map, staged)

    def run_tiled_design(
        self,
        def_path: Path,
        tile_jobs: Sequence[TileJob],
        out_spef: Path,
        *,
        tile_planning_seconds: float,
        c_unit: str,
    ) -> WindowSolveResult:
        if not tile_jobs:
            raise ValueError("tile_jobs cannot be empty")

        spef_accumulator = IndexedSpefAccumulator()
        read_and_stage_to_gpu_seconds = 0.0
        total_inference_and_reduction_seconds = 0.0
        env_feature_build_seconds = 0.0
        env_inference_and_reduction_seconds = 0.0
        gpu_to_cpu_and_aggregate_seconds = 0.0
        env_work_item_count = 0

        effective_tile_stream_size = len(tile_jobs) if int(self.config.tile_stream_size) == 0 else int(self.config.tile_stream_size)

        for tile_job_chunk in _chunked(tuple(tile_jobs), max(1, effective_tile_stream_size)):
            staged_windows, elapsed_s = _measure_elapsed(
                lambda jobs=tuple(tile_job_chunk): self._stage_tile_batch(
                    def_path,
                    jobs,
                    register_nets=spef_accumulator.register_nets,
                ),
                device=self.device,
            )
            read_and_stage_to_gpu_seconds += elapsed_s

            cache, elapsed_s = _measure_elapsed(
                lambda windows=tuple(staged_windows): _build_tile_tensor_cache(windows),
                device=self.device,
            )
            read_and_stage_to_gpu_seconds += elapsed_s

            active_window_indices = [
                window_index
                for window_index, staged in enumerate(staged_windows)
                if staged.owned_query_output_indices
            ]
            for window_index_chunk in _chunked(
                tuple(active_window_indices),
                max(1, int(self.config.total_batch_size)),
            ):
                window_index_tensor = torch.as_tensor(
                    tuple(window_index_chunk),
                    device=cache.occupied.device,
                    dtype=torch.long,
                )
                reduced, elapsed_s = _measure_elapsed(
                    lambda indices=window_index_tensor: _reduce_batched_owned_queries(
                        forward_qmap(
                            self.total_model,
                            cache.occupied.index_select(0, indices),
                        ),
                        cache,
                        indices,
                    ),
                    device=self.device,
                )
                total_inference_and_reduction_seconds += elapsed_s

                _, elapsed_s = _measure_elapsed(
                    lambda indices=tuple(window_index_chunk), current_reduced=reduced: _accumulate_total_prediction_batch(
                        spef_accumulator,
                        staged_windows,
                        indices,
                        current_reduced,
                    )
                )
                gpu_to_cpu_and_aggregate_seconds += elapsed_s

            env_work_plan = _build_env_work_plan(staged_windows, device=cache.occupied.device)
            env_work_item_count += len(env_work_plan.items)
            for chunk_start in range(
                0,
                len(env_work_plan.items),
                max(1, int(self.config.master_batch_size)),
            ):
                chunk_end = min(
                    len(env_work_plan.items),
                    chunk_start + max(1, int(self.config.master_batch_size)),
                )
                work_item_chunk = env_work_plan.items[chunk_start:chunk_end]
                window_indices = env_work_plan.window_indices[chunk_start:chunk_end]
                master_output_indices = env_work_plan.master_output_indices[chunk_start:chunk_end]
                env_batch, elapsed_s = _measure_elapsed(
                    lambda indices=window_indices, masters=master_output_indices: _build_env_feature_batch_from_cache(
                        cache,
                        indices,
                        masters,
                    ),
                    device=self.device,
                )
                env_feature_build_seconds += elapsed_s

                reduced, elapsed_s = _measure_elapsed(
                    lambda indices=window_indices, batch=env_batch: _reduce_batched_owned_queries(
                        forward_qmap(self.env_model, batch),
                        cache,
                        indices,
                    ),
                    device=self.device,
                )
                env_inference_and_reduction_seconds += elapsed_s

                _, elapsed_s = _measure_elapsed(
                    lambda items=work_item_chunk, current_reduced=reduced: _accumulate_env_prediction_batch(
                        spef_accumulator,
                        staged_windows,
                        items,
                        current_reduced,
                    )
                )
                gpu_to_cpu_and_aggregate_seconds += elapsed_s

        _, spef_write_seconds = _measure_elapsed(
            lambda: spef_accumulator.write(
                out_spef,
                window_id=def_path.stem,
                c_unit=c_unit,
                derive_total_cap_from_couplings=False,
            )
        )

        timing = StageTimingBreakdown(
            tile_planning_seconds=float(tile_planning_seconds),
            read_and_stage_to_gpu_seconds=float(read_and_stage_to_gpu_seconds),
            total_inference_and_reduction_seconds=float(total_inference_and_reduction_seconds),
            env_feature_build_seconds=float(env_feature_build_seconds),
            env_inference_and_reduction_seconds=float(env_inference_and_reduction_seconds),
            gpu_to_cpu_and_aggregate_seconds=float(gpu_to_cpu_and_aggregate_seconds),
            spef_write_seconds=float(spef_write_seconds),
        )
        return WindowSolveResult(
            window_id=def_path.stem,
            output_path=Path(out_spef),
            elapsed_seconds=float(timing.total_seconds),
            timing=timing,
            tile_count=len(tile_jobs),
            env_work_item_count=int(env_work_item_count),
            total_batch_size=int(self.config.total_batch_size),
            env_batch_size=int(self.config.master_batch_size),
            tile_stream_size=int(effective_tile_stream_size),
        )

    def run_window(self, def_path: Path, out_spef_dir: Path, *, c_unit: str) -> WindowSolveResult:
        spef_accumulator = IndexedSpefAccumulator()
        read_and_stage_to_gpu_seconds = 0.0
        total_inference_and_reduction_seconds = 0.0
        env_feature_build_seconds = 0.0
        env_inference_and_reduction_seconds = 0.0
        gpu_to_cpu_and_aggregate_seconds = 0.0
        env_work_item_count = 0

        window_job, elapsed_s = _measure_elapsed(lambda: _build_window_job(def_path, self.config))
        read_and_stage_to_gpu_seconds += elapsed_s

        staged, elapsed_s = _measure_elapsed(
            lambda: self._stage_window(
                def_path,
                window_job,
                register_nets=spef_accumulator.register_nets,
            ),
            device=self.device,
        )
        read_and_stage_to_gpu_seconds += elapsed_s

        if staged.owned_query_output_indices:
            total_preds, elapsed_s = _measure_elapsed(
                lambda: self._predict_total_queries(staged),
                device=self.device,
            )
            total_inference_and_reduction_seconds += elapsed_s

            _, elapsed_s = _measure_elapsed(
                lambda: _accumulate_total_predictions(spef_accumulator, staged, total_preds)
            )
            gpu_to_cpu_and_aggregate_seconds += elapsed_s

        if staged.owned_query_output_indices and staged.visible_master_local_ids:
            chunk_size = max(1, int(self.config.master_batch_size))
            env_work_item_count = len(staged.visible_master_local_ids)
            for chunk_start in range(0, len(staged.visible_master_local_ids), chunk_size):
                master_chunk_tuple = staged.visible_master_local_ids[chunk_start:chunk_start + chunk_size]
                master_output_chunk = staged.visible_master_output_indices[chunk_start:chunk_start + chunk_size]
                env_batch, elapsed_s = _measure_elapsed(
                    lambda chunk=master_chunk_tuple: self._build_env_feature_batch(staged, chunk),
                    device=self.device,
                )
                env_feature_build_seconds += elapsed_s

                preds, elapsed_s = _measure_elapsed(
                    lambda batch=env_batch: self._reduce_owned_queries(forward_qmap(self.env_model, batch), staged),
                    device=self.device,
                )
                env_inference_and_reduction_seconds += elapsed_s

                _, elapsed_s = _measure_elapsed(
                    lambda chunk=master_output_chunk, chunk_preds=preds: _accumulate_coupling_predictions(
                        spef_accumulator,
                        staged,
                        chunk,
                        chunk_preds,
                    )
                )
                gpu_to_cpu_and_aggregate_seconds += elapsed_s

        output_path = out_spef_dir / f"{def_path.stem}.spef"
        _, spef_write_seconds = _measure_elapsed(
            lambda: spef_accumulator.write(
                output_path,
                window_id=def_path.stem,
                c_unit=c_unit,
            )
        )

        timing = StageTimingBreakdown(
            tile_planning_seconds=0.0,
            read_and_stage_to_gpu_seconds=float(read_and_stage_to_gpu_seconds),
            total_inference_and_reduction_seconds=float(total_inference_and_reduction_seconds),
            env_feature_build_seconds=float(env_feature_build_seconds),
            env_inference_and_reduction_seconds=float(env_inference_and_reduction_seconds),
            gpu_to_cpu_and_aggregate_seconds=float(gpu_to_cpu_and_aggregate_seconds),
            spef_write_seconds=float(spef_write_seconds),
        )
        return WindowSolveResult(
            window_id=def_path.stem,
            output_path=output_path,
            elapsed_seconds=float(timing.total_seconds),
            timing=timing,
            tile_count=1,
            env_work_item_count=int(env_work_item_count),
            total_batch_size=1,
            env_batch_size=int(self.config.master_batch_size),
            tile_stream_size=1,
        )


def _add_input_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--def", dest="def_path", type=Path, required=True, help="Single input full-layout DEF file.")


def _add_pipeline_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tech", type=Path, required=True, help="Technology YAML, for example tech/nangate45.yaml.")
    parser.add_argument(
        "--master-batch-size",
        type=int,
        default=None,
        help="Batch size for env-model master queries. Defaults to min(64, env engine max_batch_size).",
    )
    parser.add_argument(
        "--total-batch-size",
        type=int,
        default=None,
        help="Batch size for total-model tile queries. Defaults to min(24, total engine max_batch_size).",
    )
    parser.add_argument(
        "--tile-stream-size",
        type=int,
        default=DEFAULT_TILE_STREAM_SIZE,
        help="Number of staged model-window tiles solved as one stream chunk. Use 0 to stage all tiles in one native DEF pass.",
    )
    parser.add_argument(
        "--compiled-model-manifest",
        type=Path,
        required=True,
        help="Manifest written by compile_models.py, for example compiled_models/compiled_models.json.",
    )
    parser.add_argument(
        "--total-compiled-model",
        type=Path,
        required=True,
        help="TensorRT total-model engine generated for full-pipeline inference.",
    )
    parser.add_argument(
        "--env-compiled-model",
        type=Path,
        required=True,
        help="TensorRT env-model engine generated for full-pipeline inference.",
    )


def _build_pipeline_config(args: argparse.Namespace) -> PipelineConfig:
    metadata = _load_compiled_model_metadata(args.compiled_model_manifest.resolve())
    total_batch_size = int(
        min(DEFAULT_PREFERRED_BATCH_SIZE, int(metadata.total_max_batch_size))
        if args.total_batch_size is None
        else args.total_batch_size
    )
    master_batch_size = int(
        min(DEFAULT_ENV_BATCH_SIZE, int(metadata.env_max_batch_size))
        if args.master_batch_size is None
        else args.master_batch_size
    )
    tile_stream_size = int(args.tile_stream_size)
    if total_batch_size <= 0:
        raise ValueError(f"--total-batch-size must be positive, got {total_batch_size}")
    if master_batch_size <= 0:
        raise ValueError(f"--master-batch-size must be positive, got {master_batch_size}")
    if tile_stream_size < 0:
        raise ValueError(f"--tile-stream-size must be non-negative, got {tile_stream_size}")
    if total_batch_size > int(metadata.total_max_batch_size):
        raise RuntimeError(
            f"Runtime total_batch_size={total_batch_size} exceeds total engine max_batch_size={metadata.total_max_batch_size}"
        )
    if master_batch_size > int(metadata.env_max_batch_size):
        raise RuntimeError(
            f"Runtime master_batch_size={master_batch_size} exceeds env engine max_batch_size={metadata.env_max_batch_size}"
        )
    return PipelineConfig(
        tech_path=args.tech.resolve(),
        target_size=int(DEFAULT_TARGET_SIZE),
        device_name=str(DEFAULT_DEVICE_NAME),
        total_batch_size=total_batch_size,
        master_batch_size=master_batch_size,
        tile_stream_size=tile_stream_size,
        compiled_model_manifest=args.compiled_model_manifest.resolve(),
        total_compiled_model=args.total_compiled_model.resolve(),
        env_compiled_model=args.env_compiled_model.resolve(),
    )


def _resolve_def_path_from_namespace(args: argparse.Namespace) -> Path:
    def_path = args.def_path.resolve()
    if not def_path.exists():
        raise FileNotFoundError(f"DEF file not found: {def_path}")
    return def_path


def _normalize_cli_argv(argv: Sequence[str]) -> List[str]:
    if not argv:
        return ["run"]
    if argv[0] == "run":
        return list(argv)
    if argv[0].startswith("-"):
        return ["run", *argv]
    return list(argv)


def _measure_startup(config: PipelineConfig) -> tuple[InferenceSession, StartupTiming]:
    load_started_at = time.perf_counter()
    session = InferenceSession(config)
    model_load_seconds = time.perf_counter() - load_started_at
    extension_load_started_at = time.perf_counter()
    _preload_pipeline_extensions(session.device)
    extension_load_seconds = time.perf_counter() - extension_load_started_at

    timing = StartupTiming(
        device=str(session.device),
        model_load_seconds=float(model_load_seconds),
        extension_load_seconds=float(extension_load_seconds),
        startup_seconds=float(model_load_seconds + extension_load_seconds),
    )
    return session, timing


def _format_float(value: float) -> str:
    return f"{float(value):.3f}"


def _format_percent(value: float) -> str:
    return f"{float(value):.1f}%"


def _print_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(str(value)))

    def _format_row(values: Sequence[str]) -> str:
        cells = []
        for idx, value in enumerate(values):
            text = str(value)
            if idx == 0:
                cells.append(text.ljust(widths[idx]))
            else:
                cells.append(text.rjust(widths[idx]))
        return "  " + "  ".join(cells)

    print(_format_row(headers))
    print("  " + "  ".join("-" * width for width in widths))
    for row in rows:
        print(_format_row(row))


def _runtime_breakdown_rows(startup: StartupTiming, result: WindowSolveResult) -> List[Tuple[str, str, str]]:
    entries = [
        ("1. Load compiled models", float(startup.model_load_seconds)),
        ("2. Load native extensions", float(startup.extension_load_seconds)),
        ("3. Plan model-window tiles", float(result.timing.tile_planning_seconds)),
        ("4. Read DEF tiles and send to GPU", float(result.timing.read_and_stage_to_gpu_seconds)),
        ("5. Total-model inference + reduction", float(result.timing.total_inference_and_reduction_seconds)),
        ("6. Env feature batch construction", float(result.timing.env_feature_build_seconds)),
        ("7. Env-model inference + reduction", float(result.timing.env_inference_and_reduction_seconds)),
        ("8. Get tile results back and merge conductors", float(result.timing.gpu_to_cpu_and_aggregate_seconds)),
        ("9. Write merged SPEF output", float(result.timing.spef_write_seconds)),
    ]
    total_seconds = sum(value for _label, value in entries)
    rows: List[Tuple[str, str, str]] = []
    for label, value in entries:
        percent = 0.0 if total_seconds <= 0.0 else (value / total_seconds) * 100.0
        rows.append((label, _format_float(value), _format_percent(percent)))
    rows.append(("Total", _format_float(total_seconds), _format_percent(100.0 if total_seconds > 0.0 else 0.0)))
    return rows


def _runtime_summary_dict(
    *,
    startup: StartupTiming,
    windowing: TilingSummary,
    result: WindowSolveResult,
) -> dict:
    stages = {
        "model_load_seconds": float(startup.model_load_seconds),
        "extension_load_seconds": float(startup.extension_load_seconds),
        "tile_planning_seconds": float(result.timing.tile_planning_seconds),
        "read_and_stage_to_gpu_seconds": float(result.timing.read_and_stage_to_gpu_seconds),
        "total_inference_and_reduction_seconds": float(result.timing.total_inference_and_reduction_seconds),
        "env_feature_build_seconds": float(result.timing.env_feature_build_seconds),
        "env_inference_and_reduction_seconds": float(result.timing.env_inference_and_reduction_seconds),
        "gpu_to_cpu_and_aggregate_seconds": float(result.timing.gpu_to_cpu_and_aggregate_seconds),
        "spef_write_seconds": float(result.timing.spef_write_seconds),
    }
    return {
        "design_id": str(result.window_id),
        "output_path": str(result.output_path),
        "device": str(startup.device),
        "tile_count": int(result.tile_count),
        "tile_count_x": int(windowing.tile_count_x),
        "tile_count_y": int(windowing.tile_count_y),
        "env_work_items": int(result.env_work_item_count),
        "total_batch_size": int(result.total_batch_size),
        "env_batch_size": int(result.env_batch_size),
        "tile_stream_size": int(result.tile_stream_size),
        "startup_seconds": float(startup.startup_seconds),
        "solve_seconds": float(result.elapsed_seconds),
        "total_reported_seconds": float(startup.startup_seconds + result.elapsed_seconds),
        "stages": stages,
    }


def _print_startup_report(timing: StartupTiming) -> None:
    print("Startup")
    print(
        "  "
        f"device={timing.device} "
        f"model_load_seconds={_format_float(timing.model_load_seconds)} "
        f"extension_load_seconds={_format_float(timing.extension_load_seconds)} "
        f"startup_seconds={_format_float(timing.startup_seconds)}"
    )


def _print_windowing_report(summary: TilingSummary) -> None:
    print("Tiling")
    fields = [
        f"design_id={summary.design_id}",
        f"mode={summary.mode}",
        f"die_width_um={_format_float(summary.die_width_um)}",
        f"die_height_um={_format_float(summary.die_height_um)}",
        f"tile_width_um={_format_float(summary.tile_width_um)}",
        f"tile_context_um={_format_float(summary.tile_context_um)}",
        f"stride_um={_format_float(summary.stride_um)}",
        f"tile_count_x={summary.tile_count_x}",
        f"tile_count_y={summary.tile_count_y}",
        f"tile_count={summary.tile_count}",
        f"solve_target_size={summary.solve_target_size}",
    ]
    print("  " + " ".join(fields))


def _print_runtime_breakdown(startup: StartupTiming, result: WindowSolveResult) -> None:
    print("Runtime Breakdown")
    _print_table(("Stage", "Time (s)", "Percent"), _runtime_breakdown_rows(startup, result))


def _print_solve_summary(result: WindowSolveResult) -> None:
    print("Solve Summary")
    print(
        "  "
        f"design_id={result.window_id} "
        f"output_path={result.output_path} "
        f"tile_count={result.tile_count} "
        f"env_work_items={result.env_work_item_count} "
        f"total_batch_size={result.total_batch_size} "
        f"env_batch_size={result.env_batch_size} "
        f"tile_stream_size={result.tile_stream_size} "
        f"elapsed_ms={_format_float(result.elapsed_seconds * 1000.0)}"
    )


def _run_mode(args: argparse.Namespace) -> int:
    config = _build_pipeline_config(args)
    def_path = _resolve_def_path_from_namespace(args)
    session, startup_timing = _measure_startup(config)
    out_spef = args.out_spef.resolve()
    out_spef.parent.mkdir(parents=True, exist_ok=True)
    (windowing, tile_jobs), tile_planning_seconds = _measure_elapsed(lambda: _plan_tiled_design(def_path, config))
    _print_startup_report(startup_timing)
    _print_windowing_report(windowing)
    result = session.run_tiled_design(
        def_path,
        tile_jobs,
        out_spef,
        tile_planning_seconds=float(tile_planning_seconds),
        c_unit=args.c_unit,
    )
    _print_runtime_breakdown(startup_timing, result)
    _print_solve_summary(result)
    if args.runtime_json is not None:
        runtime_json_path = args.runtime_json.resolve()
        runtime_json_path.parent.mkdir(parents=True, exist_ok=True)
        runtime_json_path.write_text(
            json.dumps(
                _runtime_summary_dict(
                    startup=startup_timing,
                    windowing=windowing,
                    result=result,
                ),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run grouped U-Net capacitance inference from one full DEF file and emit one merged SPEF.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="Run one-shot tiled inference and write one merged SPEF.")
    _add_input_args(run_parser)
    run_parser.add_argument("--out-spef", type=Path, required=True, help="Output path for the merged SPEF.")
    run_parser.add_argument("--c-unit", type=str, default="PF", help="Capacitance unit used in the written SPEF.")
    run_parser.add_argument("--runtime-json", type=Path, default=None, help="Optional path for a machine-readable runtime report.")
    _add_pipeline_config_args(run_parser)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    normalized_argv = _normalize_cli_argv(sys.argv[1:] if argv is None else argv)
    args = _build_parser().parse_args(normalized_argv)
    if args.command == "run":
        return _run_mode(args)
    raise RuntimeError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
