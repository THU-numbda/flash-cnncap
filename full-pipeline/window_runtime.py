from __future__ import annotations

import sys
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent

for extra_path in (str(REPO_ROOT), str(THIS_DIR)):
    if extra_path not in sys.path:
        sys.path.insert(0, extra_path)

from idmap_cuda_runtime import rasterize_packed_rects_with_sparse_cuda  # pylint: disable=wrong-import-position
from def_fast_density import (  # pylint: disable=wrong-import-position
    PreparedDefRasterInput,
    CONDUCTOR_SOURCE_SYNTHETIC_LEF,
    RECT_SOURCE_ROUTE,
    RECT_SOURCE_SPECIAL_ROUTE,
    build_compiled_def_runtime_config,
    prepare_fast_def_raster_input,
    prepare_fast_def_raster_inputs,
)
@dataclass(frozen=True)
class WindowFeatures:
    window_id: str
    occupied: torch.Tensor
    full_local_map: torch.Tensor
    owned_local_counts: torch.Tensor
    owned_sparse_indices: torch.Tensor
    owned_sparse_counts: torch.Tensor
    owned_query_local_ids: torch.Tensor
    visible_local_counts: torch.Tensor
    visible_sparse_indices: torch.Tensor
    visible_sparse_counts: torch.Tensor
    visible_query_local_ids: torch.Tensor
    visible_master_local_ids: torch.Tensor
    real_conductor_names: List[str]
    real_conductor_ids: np.ndarray
    prepared: PreparedDefRasterInput

    @property
    def num_real_conductors(self) -> int:
        return len(self.real_conductor_names)


@dataclass(frozen=True)
class PreparedWindowRaster:
    window_id: str
    prepared: PreparedDefRasterInput
    packed_rects: np.ndarray
    ownership_bounds: tuple[float, float, float, float] | None
    reduce_visible_conductors: bool
    master_conductors_from_ownership: bool
    restrict_masters_to_routes: bool


def _rect_to_pixel_bounds(
    *,
    rect: tuple[float, float, float, float],
    window_x0: float,
    window_y0: float,
    pixel_resolution: float,
    target_size: int,
) -> tuple[int, int, int, int] | None:
    px_min = max(0, min(int(target_size), int(np.floor((rect[0] - window_x0) / pixel_resolution))))
    px_max = max(0, min(int(target_size), int(np.ceil((rect[2] - window_x0) / pixel_resolution))))
    py_min = max(0, min(int(target_size), int(np.floor((rect[1] - window_y0) / pixel_resolution))))
    py_max = max(0, min(int(target_size), int(np.ceil((rect[3] - window_y0) / pixel_resolution))))
    if px_min >= px_max or py_min >= py_max:
        return None
    return px_min, px_max, py_min, py_max


def _patch_index_for_point(
    *,
    x: float,
    y: float,
    patch_grid_origin: Sequence[float],
    patch_size_um: float,
    patch_grid_shape: Sequence[int],
) -> tuple[int, int]:
    if len(patch_grid_origin) != 2:
        raise ValueError(f"patch_grid_origin must contain 2 values, got {patch_grid_origin}")
    if len(patch_grid_shape) != 2:
        raise ValueError(f"patch_grid_shape must contain 2 values, got {patch_grid_shape}")
    if patch_size_um <= 0.0:
        raise ValueError(f"patch_size_um must be positive, got {patch_size_um}")
    origin_x, origin_y = (float(value) for value in patch_grid_origin)
    rows, cols = (int(value) for value in patch_grid_shape)
    if rows <= 0 or cols <= 0:
        raise ValueError(f"patch_grid_shape must be positive, got {patch_grid_shape}")
    col = int(np.floor((float(x) - origin_x) / float(patch_size_um)))
    row = int(np.floor((float(y) - origin_y) / float(patch_size_um)))
    return max(0, min(row, rows - 1)), max(0, min(col, cols - 1))


def _patch_boundary_pixels(
    *,
    rect_min_px: int,
    rect_max_px: int,
    rect_min_world: float,
    rect_max_world: float,
    axis_window_min: float,
    axis_origin: float,
    patch_size_um: float,
    target_size: int,
    pixel_resolution: float,
) -> list[int]:
    first_boundary = int(np.floor((rect_min_world - axis_origin) / patch_size_um)) + 1
    last_boundary = int(np.floor((rect_max_world - axis_origin) / patch_size_um))
    cuts = [int(rect_min_px), int(rect_max_px)]
    for boundary_index in range(first_boundary, last_boundary + 1):
        boundary = axis_origin + (float(boundary_index) * patch_size_um)
        if boundary <= rect_min_world or boundary >= rect_max_world:
            continue
        pixel = int(np.floor((boundary - axis_window_min) / pixel_resolution))
        pixel = max(0, min(int(target_size), pixel))
        if rect_min_px < pixel < rect_max_px:
            cuts.append(pixel)
    return sorted(set(cuts))


def _remap_packed_rects_to_patch_fragments(
    prepared: PreparedDefRasterInput,
    *,
    patch_grid_origin: Sequence[float],
    patch_size_um: float,
    patch_grid_shape: Sequence[int],
    fragment_keep_bounds: Sequence[float] | None = None,
) -> PreparedDefRasterInput:
    packed_rects = np.asarray(prepared.packed_rects, dtype=np.int32)
    if packed_rects.size == 0:
        return prepared
    if prepared.conductor_names_sorted is None:
        raise RuntimeError("Patch-fragment remapping requires conductor names.")
    original_real_conductor_count = int(prepared.real_conductor_ids_sorted.shape[0])
    if original_real_conductor_count <= 0:
        return prepared

    target_size = int(prepared.target_size)
    pixel_resolution = float(prepared.pixel_resolution)
    window_x0 = float(prepared.window_bounds[0])
    window_y0 = float(prepared.window_bounds[1])
    rows, cols = (int(value) for value in patch_grid_shape)
    origin_x, origin_y = (float(value) for value in patch_grid_origin)
    keep_bounds = None
    if fragment_keep_bounds is not None:
        if len(fragment_keep_bounds) != 4:
            raise ValueError(f"fragment_keep_bounds must contain 4 values, got {fragment_keep_bounds}")
        keep_bounds = tuple(float(value) for value in fragment_keep_bounds)

    pending_rows: list[tuple[tuple[str, str, int, int], int, int, int, int, int, int]] = []
    real_keys: list[tuple[str, str, int, int]] = []
    synthetic_keys: list[tuple[str, str, int, int]] = []
    real_key_seen: set[tuple[str, str, int, int]] = set()
    synthetic_key_seen: set[tuple[str, str, int, int]] = set()
    source_codes = np.asarray(prepared.rect_source_kind_codes, dtype=np.uint8)

    for row_idx, rect_row in enumerate(packed_rects.tolist()):
        layer, conductor_id, px_min, px_max, py_min, py_max = (int(value) for value in rect_row)
        if conductor_id <= 0 or conductor_id > len(prepared.conductor_names_sorted):
            continue
        if px_min >= px_max or py_min >= py_max:
            continue

        x0 = window_x0 + (float(px_min) * pixel_resolution)
        x1 = window_x0 + (float(px_max) * pixel_resolution)
        y0 = window_y0 + (float(py_min) * pixel_resolution)
        y1 = window_y0 + (float(py_max) * pixel_resolution)
        x_cuts = _patch_boundary_pixels(
            rect_min_px=px_min,
            rect_max_px=px_max,
            rect_min_world=x0,
            rect_max_world=x1,
            axis_window_min=window_x0,
            axis_origin=origin_x,
            patch_size_um=float(patch_size_um),
            target_size=target_size,
            pixel_resolution=pixel_resolution,
        )
        y_cuts = _patch_boundary_pixels(
            rect_min_px=py_min,
            rect_max_px=py_max,
            rect_min_world=y0,
            rect_max_world=y1,
            axis_window_min=window_y0,
            axis_origin=origin_y,
            patch_size_um=float(patch_size_um),
            target_size=target_size,
            pixel_resolution=pixel_resolution,
        )

        is_real = conductor_id <= original_real_conductor_count
        source_name = str(prepared.conductor_names_sorted[conductor_id - 1])
        key_kind = "real" if is_real else "synthetic"
        for x_left, x_right in zip(x_cuts[:-1], x_cuts[1:]):
            if x_left >= x_right:
                continue
            for y_bottom, y_top in zip(y_cuts[:-1], y_cuts[1:]):
                if y_bottom >= y_top:
                    continue
                frag_x0 = window_x0 + (float(x_left) * pixel_resolution)
                frag_x1 = window_x0 + (float(x_right) * pixel_resolution)
                frag_y0 = window_y0 + (float(y_bottom) * pixel_resolution)
                frag_y1 = window_y0 + (float(y_top) * pixel_resolution)
                if keep_bounds is not None:
                    keep_x0, keep_y0, keep_x1, keep_y1 = keep_bounds
                    if frag_x1 <= keep_x0 or frag_x0 >= keep_x1 or frag_y1 <= keep_y0 or frag_y0 >= keep_y1:
                        continue
                mid_x = window_x0 + ((float(x_left + x_right) * 0.5) * pixel_resolution)
                mid_y = window_y0 + ((float(y_bottom + y_top) * 0.5) * pixel_resolution)
                owner_row, owner_col = _patch_index_for_point(
                    x=mid_x,
                    y=mid_y,
                    patch_grid_origin=(origin_x, origin_y),
                    patch_size_um=float(patch_size_um),
                    patch_grid_shape=(rows, cols),
                )
                key = (key_kind, source_name, int(owner_row), int(owner_col))
                if is_real:
                    if key not in real_key_seen:
                        real_key_seen.add(key)
                        real_keys.append(key)
                else:
                    if key not in synthetic_key_seen:
                        synthetic_key_seen.add(key)
                        synthetic_keys.append(key)
                source_code = int(source_codes[row_idx]) if row_idx < int(source_codes.shape[0]) else 0
                pending_rows.append((key, layer, x_left, x_right, y_bottom, y_top, source_code))

    key_to_id: dict[tuple[str, str, int, int], int] = {}
    conductor_names: list[str] = []
    for key in real_keys:
        key_to_id[key] = len(conductor_names) + 1
        conductor_names.append(key[1])
    real_count = len(conductor_names)
    synthetic_names: list[str] = []
    for key in synthetic_keys:
        key_to_id[key] = real_count + len(synthetic_names) + 1
        synthetic_names.append(f"{key[1]}@r{key[2]:03d}_c{key[3]:03d}")

    if len(conductor_names) + len(synthetic_names) > np.iinfo(np.int16).max:
        raise RuntimeError(
            "Patch-fragment staging produced too many local conductors for int16 raster maps: "
            f"{len(conductor_names) + len(synthetic_names)}"
        )

    remapped_rects = np.zeros((len(pending_rows), packed_rects.shape[1]), dtype=np.int32)
    remapped_source_codes = np.zeros((len(pending_rows),), dtype=np.uint8)
    for out_idx, (key, layer, x_left, x_right, y_bottom, y_top, source_code) in enumerate(pending_rows):
        remapped_rects[out_idx] = (
            int(layer),
            int(key_to_id[key]),
            int(x_left),
            int(x_right),
            int(y_bottom),
            int(y_top),
        )
        remapped_source_codes[out_idx] = np.uint8(source_code)

    conductor_count = len(conductor_names) + len(synthetic_names)
    conductor_is_synthetic = np.zeros((conductor_count,), dtype=bool)
    if synthetic_names:
        conductor_is_synthetic[real_count:] = True
    conductor_source_kind_codes = np.zeros((conductor_count,), dtype=np.uint8)
    if synthetic_names:
        conductor_source_kind_codes[real_count:] = CONDUCTOR_SOURCE_SYNTHETIC_LEF

    return replace(
        prepared,
        conductor_names_sorted=conductor_names + synthetic_names,
        conductor_ids_sorted=np.arange(1, conductor_count + 1, dtype=np.int16),
        conductor_is_synthetic=conductor_is_synthetic,
        conductor_source_kind_codes=conductor_source_kind_codes,
        packed_rects=remapped_rects,
        packed_rects_torch=None,
        rect_source_kind_codes=remapped_source_codes,
        net_name_to_gpu_id=None,
        active_rectangles=int(remapped_rects.shape[0]),
    )


def build_runtime_config(tech_path: Path | str, *, selected_layers: Sequence[str] | None = None):
    return build_compiled_def_runtime_config(tech_path, selected_layers=selected_layers)


def _bounds_to_pixel_slice(
    *,
    window_bounds: Sequence[float],
    pixel_resolution: float,
    target_size: int,
    keep_bounds: Sequence[float],
) -> tuple[int, int, int, int]:
    if len(window_bounds) < 5:
        raise ValueError(f"window_bounds must contain at least 5 values, got {window_bounds}")
    if len(keep_bounds) != 4:
        raise ValueError(f"keep_bounds must contain 4 values, got {keep_bounds}")
    if pixel_resolution <= 0.0:
        raise ValueError(f"pixel_resolution must be positive, got {pixel_resolution}")
    if target_size <= 0:
        raise ValueError(f"target_size must be positive, got {target_size}")

    x_min = float(window_bounds[0])
    y_min = float(window_bounds[1])
    x_max = float(window_bounds[3])
    y_max = float(window_bounds[4])
    keep_x0, keep_y0, keep_x1, keep_y1 = (float(value) for value in keep_bounds)

    keep_x0 = min(max(keep_x0, x_min), x_max)
    keep_x1 = min(max(keep_x1, x_min), x_max)
    keep_y0 = min(max(keep_y0, y_min), y_max)
    keep_y1 = min(max(keep_y1, y_min), y_max)

    px_min = max(0, min(int(target_size), int(np.floor((keep_x0 - x_min) / float(pixel_resolution)))))
    px_max = max(0, min(int(target_size), int(np.ceil((keep_x1 - x_min) / float(pixel_resolution)))))
    py_min = max(0, min(int(target_size), int(np.floor((keep_y0 - y_min) / float(pixel_resolution)))))
    py_max = max(0, min(int(target_size), int(np.ceil((keep_y1 - y_min) / float(pixel_resolution)))))
    return px_min, px_max, py_min, py_max


def _route_master_local_ids(
    prepared: PreparedDefRasterInput,
    packed_rects: np.ndarray,
) -> np.ndarray:
    rect_codes = np.asarray(prepared.rect_source_kind_codes, dtype=np.uint8)
    if packed_rects.shape[0] != rect_codes.shape[0]:
        raise ValueError(
            "packed_rects and rect_source_kind_codes must have the same number of rows, "
            f"got {packed_rects.shape[0]} and {rect_codes.shape[0]}"
        )

    route_mask = (rect_codes == RECT_SOURCE_ROUTE) | (rect_codes == RECT_SOURCE_SPECIAL_ROUTE)
    if not bool(np.any(route_mask)):
        return np.empty((0,), dtype=np.int64)

    real_conductor_count = int(prepared.real_conductor_ids_sorted.shape[0])
    conductor_ids = np.asarray(packed_rects[route_mask, 1], dtype=np.int64)
    conductor_ids = conductor_ids[(conductor_ids > 0) & (conductor_ids <= real_conductor_count)]
    if conductor_ids.size == 0:
        return np.empty((0,), dtype=np.int64)
    return np.unique(conductor_ids)


def prepare_window_raster(
    def_path: Path | str,
    *,
    tech_path: Path | str,
    target_size: int,
    pixel_resolution: float | None = None,
    raster_bounds: Sequence[float] | None = None,
    ownership_bounds: Sequence[float] | None = None,
    selected_layers: Sequence[str] | None = None,
    include_supply_nets: bool = False,
    window_id: str | None = None,
    patch_grid_origin: Sequence[float] | None = None,
    patch_size_um: float | None = None,
    patch_grid_shape: Sequence[int] | None = None,
    fragment_keep_bounds: Sequence[float] | None = None,
    reduce_visible_conductors: bool = False,
    master_conductors_from_ownership: bool = False,
    restrict_masters_to_routes: bool = True,
) -> PreparedWindowRaster:
    def_path = Path(def_path).resolve()
    prepared = prepare_fast_def_raster_input(
        def_path=def_path,
        tech_path=tech_path,
        target_size=target_size,
        pixel_resolution=pixel_resolution,
        selected_layers=selected_layers,
        raster_bounds=raster_bounds,
        include_supply_nets=include_supply_nets,
        include_conductor_names=True,
    )
    if prepared.conductor_names_sorted is None:
        raise RuntimeError(f"Prepared DEF input for {def_path} did not include conductor names.")
    if patch_grid_origin is not None or patch_size_um is not None or patch_grid_shape is not None:
        if patch_grid_origin is None or patch_size_um is None or patch_grid_shape is None:
            raise ValueError("patch_grid_origin, patch_size_um, and patch_grid_shape must be provided together.")
        prepared = _remap_packed_rects_to_patch_fragments(
            prepared,
            patch_grid_origin=patch_grid_origin,
            patch_size_um=float(patch_size_um),
            patch_grid_shape=patch_grid_shape,
            fragment_keep_bounds=fragment_keep_bounds,
        )

    resolved_ownership_bounds = None
    if ownership_bounds is not None:
        if len(ownership_bounds) != 4:
            raise ValueError(f"ownership_bounds must contain 4 values, got {ownership_bounds}")
        resolved_ownership_bounds = tuple(float(value) for value in ownership_bounds)

    return PreparedWindowRaster(
        window_id=str(window_id) if window_id is not None else def_path.stem,
        prepared=prepared,
        packed_rects=np.asarray(prepared.packed_rects, dtype=np.int32),
        ownership_bounds=resolved_ownership_bounds,
        reduce_visible_conductors=bool(reduce_visible_conductors),
        master_conductors_from_ownership=bool(master_conductors_from_ownership),
        restrict_masters_to_routes=bool(restrict_masters_to_routes),
    )


def prepare_tiled_window_rasters(
    def_path: Path | str,
    *,
    tech_path: Path | str,
    tile_jobs: Sequence[object],
    selected_layers: Sequence[str] | None = None,
    include_supply_nets: bool = False,
) -> tuple[PreparedWindowRaster, ...]:
    if not tile_jobs:
        return tuple()

    tile_specs = []
    for tile_job in tile_jobs:
        tile_specs.append({
            "target_size": int(tile_job.solve_target_size),
            "pixel_resolution": float(tile_job.pixel_resolution_um),
            "raster_bounds": tuple(float(value) for value in tile_job.raster_bounds),
            "patch_grid_origin": tuple(float(value) for value in tile_job.patch_grid_origin),
            "patch_size_um": float(tile_job.patch_size_um),
            "patch_grid_shape": tuple(int(value) for value in tile_job.patch_grid_shape),
            "fragment_keep_bounds": tuple(float(value) for value in tile_job.margin_bounds),
        })

    prepared_items = prepare_fast_def_raster_inputs(
        def_path=def_path,
        tech_path=tech_path,
        tile_specs=tile_specs,
        selected_layers=selected_layers,
        include_supply_nets=include_supply_nets,
        include_conductor_names=True,
    )
    if len(prepared_items) != len(tile_jobs):
        raise RuntimeError(
            f"Prepared {len(prepared_items)} tiled windows for {len(tile_jobs)} tile jobs"
        )

    windows: list[PreparedWindowRaster] = []
    for tile_job, prepared in zip(tile_jobs, prepared_items):
        if prepared.conductor_names_sorted is None:
            raise RuntimeError(f"Prepared DEF input for {def_path} did not include conductor names.")
        windows.append(
            PreparedWindowRaster(
                window_id=str(tile_job.window_id),
                prepared=prepared,
                packed_rects=np.asarray(prepared.packed_rects, dtype=np.int32),
                ownership_bounds=tuple(float(value) for value in tile_job.ownership_bounds),
                reduce_visible_conductors=False,
                master_conductors_from_ownership=True,
                restrict_masters_to_routes=False,
            )
        )
    return tuple(windows)


def materialize_window_features(
    prepared_window: PreparedWindowRaster,
    *,
    device: torch.device,
) -> WindowFeatures:
    if device.type != "cuda":
        raise ValueError(f"materialize_window_features requires a CUDA device, got {device}")
    prepared = prepared_window.prepared
    packed_rects = prepared_window.packed_rects
    own_px_min = 0
    own_px_max = int(prepared.target_size)
    own_py_min = 0
    own_py_max = int(prepared.target_size)
    if prepared_window.ownership_bounds is not None:
        own_px_min, own_px_max, own_py_min, own_py_max = _bounds_to_pixel_slice(
            window_bounds=prepared.window_bounds,
            pixel_resolution=float(prepared.pixel_resolution),
            target_size=int(prepared.target_size),
            keep_bounds=prepared_window.ownership_bounds,
        )

    real_conductor_count = int(prepared.real_conductor_ids_sorted.shape[0])
    route_master_local_ids = _route_master_local_ids(prepared, packed_rects)
    if int(packed_rects.shape[0]) == 0:
        full_local_map = torch.zeros(
            (len(prepared.channel_layers), int(prepared.target_size), int(prepared.target_size)),
            device=device,
            dtype=torch.long,
        )
        occupied = torch.zeros_like(full_local_map, dtype=torch.float32)
        owned_local_counts = torch.zeros((real_conductor_count + 1,), device=device, dtype=torch.long)
        owned_sparse_indices = torch.zeros((real_conductor_count + 1, 1), device=device, dtype=torch.long)
        owned_sparse_counts = torch.zeros((real_conductor_count + 1,), device=device, dtype=torch.long)
        owned_query_local_ids = torch.zeros((0,), device=device, dtype=torch.long)
        visible_local_counts = owned_local_counts
        visible_sparse_indices = owned_sparse_indices
        visible_sparse_counts = owned_sparse_counts
        visible_query_local_ids = owned_query_local_ids
        visible_master_local_ids = torch.zeros((0,), device=device, dtype=torch.long)
    else:
        packed_rects_cuda = torch.from_numpy(packed_rects).to(
            device=device,
            dtype=torch.int32,
            non_blocking=True,
        ).contiguous()
        (
            occupied_u8,
            full_local_map_i16,
            _owned_local_map_i16,
            owned_local_counts,
            owned_sparse_indices,
            owned_sparse_counts,
            owned_query_local_ids,
            visible_master_local_ids,
        ) = rasterize_packed_rects_with_sparse_cuda(
            packed_rects_cuda,
            num_layers=len(prepared.channel_layers),
            target_size=int(prepared.target_size),
            real_conductor_count=real_conductor_count,
            own_x0=own_px_min,
            own_x1=own_px_max,
            own_y0=own_py_min,
            own_y1=own_py_max,
        )
        occupied = occupied_u8.to(dtype=torch.float32)
        full_local_map = full_local_map_i16.to(dtype=torch.long)

        if prepared_window.reduce_visible_conductors:
            (
                _visible_occupied_u8,
                _visible_full_local_map_i16,
                _visible_owned_local_map_i16,
                visible_local_counts,
                visible_sparse_indices,
                visible_sparse_counts,
                visible_query_local_ids,
                _visible_master_ids,
            ) = rasterize_packed_rects_with_sparse_cuda(
                packed_rects_cuda,
                num_layers=len(prepared.channel_layers),
                target_size=int(prepared.target_size),
                real_conductor_count=real_conductor_count,
                own_x0=0,
                own_x1=int(prepared.target_size),
                own_y0=0,
                own_y1=int(prepared.target_size),
            )
        else:
            visible_local_counts = owned_local_counts
            visible_sparse_indices = owned_sparse_indices
            visible_sparse_counts = owned_sparse_counts
            visible_query_local_ids = owned_query_local_ids

    if prepared_window.master_conductors_from_ownership:
        visible_master_local_ids = owned_query_local_ids
    elif int(visible_master_local_ids.numel()) > 0 and prepared_window.restrict_masters_to_routes:
        if route_master_local_ids.size == 0:
            visible_master_local_ids = torch.zeros((0,), device=device, dtype=torch.long)
        else:
            master_allowed = torch.zeros((real_conductor_count + 1,), device=device, dtype=torch.bool)
            master_allowed[
                torch.from_numpy(route_master_local_ids).to(
                    device=device,
                    dtype=torch.long,
                    non_blocking=True,
                )
            ] = True
            visible_master_local_ids = visible_master_local_ids[
                master_allowed.index_select(0, visible_master_local_ids)
            ]

    occupied = occupied.to(dtype=torch.float32).contiguous()
    full_local_map = full_local_map.contiguous()
    owned_local_counts = owned_local_counts.to(dtype=torch.long).contiguous()
    owned_sparse_indices = owned_sparse_indices.to(dtype=torch.long).contiguous()
    owned_sparse_counts = owned_sparse_counts.to(dtype=torch.long).contiguous()
    owned_query_local_ids = owned_query_local_ids.to(dtype=torch.long).contiguous()
    visible_local_counts = visible_local_counts.to(dtype=torch.long).contiguous()
    visible_sparse_indices = visible_sparse_indices.to(dtype=torch.long).contiguous()
    visible_sparse_counts = visible_sparse_counts.to(dtype=torch.long).contiguous()
    visible_query_local_ids = visible_query_local_ids.to(dtype=torch.long).contiguous()
    visible_master_local_ids = visible_master_local_ids.to(dtype=torch.long).contiguous()

    return WindowFeatures(
        window_id=prepared_window.window_id,
        occupied=occupied,
        full_local_map=full_local_map,
        owned_local_counts=owned_local_counts,
        owned_sparse_indices=owned_sparse_indices.contiguous(),
        owned_sparse_counts=owned_sparse_counts.contiguous(),
        owned_query_local_ids=owned_query_local_ids,
        visible_local_counts=visible_local_counts,
        visible_sparse_indices=visible_sparse_indices.contiguous(),
        visible_sparse_counts=visible_sparse_counts.contiguous(),
        visible_query_local_ids=visible_query_local_ids,
        visible_master_local_ids=visible_master_local_ids,
        real_conductor_names=list(prepared.conductor_names_sorted[:real_conductor_count]),
        real_conductor_ids=np.asarray(prepared.real_conductor_ids_sorted, dtype=np.int16),
        prepared=prepared,
    )


def prepare_window_features(
    def_path: Path | str,
    *,
    tech_path: Path | str,
    target_size: int,
    device: torch.device,
    pixel_resolution: float | None = None,
    raster_bounds: Sequence[float] | None = None,
    ownership_bounds: Sequence[float] | None = None,
    selected_layers: Sequence[str] | None = None,
    include_supply_nets: bool = False,
    window_id: str | None = None,
    patch_grid_origin: Sequence[float] | None = None,
    patch_size_um: float | None = None,
    patch_grid_shape: Sequence[int] | None = None,
    fragment_keep_bounds: Sequence[float] | None = None,
    reduce_visible_conductors: bool = False,
    master_conductors_from_ownership: bool = False,
    restrict_masters_to_routes: bool = True,
) -> WindowFeatures:
    prepared_window = prepare_window_raster(
        def_path,
        tech_path=tech_path,
        target_size=target_size,
        pixel_resolution=pixel_resolution,
        raster_bounds=raster_bounds,
        ownership_bounds=ownership_bounds,
        selected_layers=selected_layers,
        include_supply_nets=include_supply_nets,
        window_id=window_id,
        patch_grid_origin=patch_grid_origin,
        patch_size_um=patch_size_um,
        patch_grid_shape=patch_grid_shape,
        fragment_keep_bounds=fragment_keep_bounds,
        reduce_visible_conductors=reduce_visible_conductors,
        master_conductors_from_ownership=master_conductors_from_ownership,
        restrict_masters_to_routes=restrict_masters_to_routes,
    )
    return materialize_window_features(prepared_window, device=device)


def build_total_features(window: WindowFeatures) -> torch.Tensor:
    return window.occupied.to(dtype=torch.float32).contiguous()


def build_env_feature_batch(window: WindowFeatures, master_local_ids: Sequence[int]) -> torch.Tensor:
    if not master_local_ids:
        raise ValueError("master_local_ids cannot be empty")
    master_ids = torch.as_tensor(master_local_ids, device=window.full_local_map.device, dtype=torch.long)
    if bool((master_ids <= 0).any()):
        raise ValueError(f"master local ids must be positive, got {tuple(int(v) for v in master_ids.tolist())}")
    if window.full_local_map.numel() == 0:
        raise ValueError("Cannot build env features from an empty local map.")
    if bool((master_ids > int(window.full_local_map.max().item())).any()) and int(window.full_local_map.max().item()) > 0:
        raise ValueError(
            f"master local ids exceed available local ids: ids={tuple(int(v) for v in master_ids.tolist())} "
            f"available_max={int(window.full_local_map.max().item())}"
        )
    batch = window.occupied.unsqueeze(0).expand(len(master_local_ids), -1, -1, -1).clone()
    master_mask = window.full_local_map.unsqueeze(0).eq(master_ids.view(-1, 1, 1, 1))
    batch.sub_(master_mask.to(dtype=batch.dtype) * 2.0)
    return batch.contiguous()
