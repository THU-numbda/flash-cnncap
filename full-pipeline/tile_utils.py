from __future__ import annotations

from dataclasses import dataclass
import math
from typing import MutableMapping, Sequence, Tuple


DEFAULT_TILE_SIZE_UM = 10.0
DEFAULT_TILE_CONTEXT_UM = 1.0
DEFAULT_TILE_STRIDE_UM = DEFAULT_TILE_SIZE_UM - (2.0 * DEFAULT_TILE_CONTEXT_UM)
DEFAULT_PATCH_SIZE_UM = DEFAULT_TILE_STRIDE_UM
# Model window of each technology's CapBench large windows (224 px). Tiles must use the window the
# model was trained on: a Sky130HD model sees 20 um per window, so a 10 um tile doubles every feature.
TILE_SIZE_UM_BY_TECH = {"nangate45": 10.0, "sky130hd": 20.0}
TILE_CONTEXT_FRACTION = DEFAULT_TILE_CONTEXT_UM / DEFAULT_TILE_SIZE_UM
_EPSILON = 1e-9


def tile_geometry_for_tech(tech_name: str) -> Tuple[float, float]:
    """(tile_size_um, tile_context_um) for a technology YAML stem; unknown stems keep the defaults."""
    tile_size_um = TILE_SIZE_UM_BY_TECH.get(str(tech_name), DEFAULT_TILE_SIZE_UM)
    return tile_size_um, tile_size_um * TILE_CONTEXT_FRACTION


@dataclass(frozen=True)
class TileJob:
    window_id: str
    row_index: int
    col_index: int
    solve_target_size: int
    pixel_resolution_um: float
    raster_bounds: Tuple[float, float, float, float]
    ownership_bounds: Tuple[float, float, float, float]
    patch_bounds: Tuple[float, float, float, float]
    margin_bounds: Tuple[float, float, float, float]
    solve_bounds: Tuple[float, float, float, float]
    patch_grid_origin: Tuple[float, float]
    patch_grid_shape: Tuple[int, int]
    patch_size_um: float
    margin_um: float


@dataclass(frozen=True)
class TilingSummary:
    design_id: str
    mode: str
    die_width_um: float
    die_height_um: float
    solve_target_size: int
    tile_width_um: float
    tile_height_um: float
    tile_context_um: float
    stride_um: float
    tile_count_x: int
    tile_count_y: int
    tile_count: int


@dataclass(frozen=True)
class _AxisTile:
    index: int
    raster_min: float
    raster_max: float
    ownership_min: float
    ownership_max: float


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(float(value), upper))


def _build_axis_tiles(
    axis_min: float,
    axis_max: float,
    *,
    tile_size_um: float,
    stride_um: float,
) -> Tuple[_AxisTile, ...]:
    axis_min = float(axis_min)
    axis_max = float(axis_max)
    tile_size_um = float(tile_size_um)
    stride_um = float(stride_um)
    if axis_max <= axis_min:
        raise ValueError(f"Axis bounds must be increasing, got ({axis_min}, {axis_max})")
    if tile_size_um <= 0.0:
        raise ValueError(f"tile_size_um must be positive, got {tile_size_um}")
    if stride_um <= 0.0:
        raise ValueError(f"stride_um must be positive, got {stride_um}")

    span_um = axis_max - axis_min
    starts = []
    if span_um <= tile_size_um + _EPSILON:
        center = 0.5 * (axis_min + axis_max)
        starts.append(center - (0.5 * tile_size_um))
    else:
        start = axis_min
        while True:
            starts.append(start)
            if (start + tile_size_um) >= (axis_max - _EPSILON):
                break
            start += stride_um

    centers = [start + (0.5 * tile_size_um) for start in starts]
    tiles = []
    for idx, start in enumerate(starts):
        if idx == 0:
            ownership_min = axis_min
        else:
            ownership_min = 0.5 * (centers[idx - 1] + centers[idx])
        if idx == len(starts) - 1:
            ownership_max = axis_max
        else:
            ownership_max = 0.5 * (centers[idx] + centers[idx + 1])

        ownership_min = _clamp(ownership_min, axis_min, axis_max)
        ownership_max = _clamp(ownership_max, axis_min, axis_max)
        if ownership_max <= ownership_min:
            raise RuntimeError(
                "Axis ownership collapsed while building fixed-size tiles: "
                f"index={idx} ownership=({ownership_min}, {ownership_max})"
            )

        tiles.append(
            _AxisTile(
                index=idx,
                raster_min=float(start),
                raster_max=float(start + tile_size_um),
                ownership_min=float(ownership_min),
                ownership_max=float(ownership_max),
            )
        )
    return tuple(tiles)


def build_tiled_window_jobs(
    design_id: str,
    *,
    die_bounds_um: Sequence[float],
    target_size: int,
    tile_size_um: float = DEFAULT_TILE_SIZE_UM,
    tile_context_um: float = DEFAULT_TILE_CONTEXT_UM,
) -> Tuple[TileJob, ...]:
    if len(die_bounds_um) != 4:
        raise ValueError(f"die_bounds_um must contain 4 values, got {die_bounds_um}")
    if target_size <= 0:
        raise ValueError(f"target_size must be positive, got {target_size}")

    tile_size_um = float(tile_size_um)
    tile_context_um = float(tile_context_um)
    patch_size_um = tile_size_um - (2.0 * tile_context_um)
    if patch_size_um <= 0.0:
        raise ValueError(
            f"tile_context_um={tile_context_um} is too large for tile_size_um={tile_size_um}; stride must stay positive"
        )

    die_x0, die_y0, die_x1, die_y1 = (float(value) for value in die_bounds_um)
    if die_x1 <= die_x0 or die_y1 <= die_y0:
        raise ValueError(f"die_bounds_um must be increasing, got {die_bounds_um}")

    tile_count_x = max(1, int(math.ceil((die_x1 - die_x0) / patch_size_um)))
    tile_count_y = max(1, int(math.ceil((die_y1 - die_y0) / patch_size_um)))
    pixel_resolution_um = tile_size_um / float(target_size)

    jobs = []
    for row_index in range(tile_count_y):
        grid_y0 = die_y0 + (float(row_index) * patch_size_um)
        grid_y1 = grid_y0 + patch_size_um
        patch_y0 = max(die_y0, grid_y0)
        patch_y1 = min(die_y1, grid_y1)
        for col_index in range(tile_count_x):
            grid_x0 = die_x0 + (float(col_index) * patch_size_um)
            grid_x1 = grid_x0 + patch_size_um
            patch_x0 = max(die_x0, grid_x0)
            patch_x1 = min(die_x1, grid_x1)
            if patch_x1 <= patch_x0 or patch_y1 <= patch_y0:
                continue
            patch_bounds = (float(patch_x0), float(patch_y0), float(patch_x1), float(patch_y1))
            margin_bounds = (
                float(grid_x0 - tile_context_um),
                float(grid_y0 - tile_context_um),
                float(grid_x1 + tile_context_um),
                float(grid_y1 + tile_context_um),
            )
            solve_bounds = margin_bounds
            raster_width_um = float(solve_bounds[2] - solve_bounds[0])
            raster_height_um = float(solve_bounds[3] - solve_bounds[1])
            if not math.isclose(raster_width_um, tile_size_um, rel_tol=0.0, abs_tol=1e-9):
                raise RuntimeError(
                    "Tile raster width does not match the model input window size: "
                    f"width={raster_width_um} expected={tile_size_um}"
                )
            if not math.isclose(raster_height_um, tile_size_um, rel_tol=0.0, abs_tol=1e-9):
                raise RuntimeError(
                    "Tile raster height does not match the model input window size: "
                    f"height={raster_height_um} expected={tile_size_um}"
                )
            if not math.isclose(raster_width_um / float(target_size), pixel_resolution_um, rel_tol=0.0, abs_tol=1e-12):
                raise RuntimeError(
                    "Tile raster pixel resolution does not match the model input scale: "
                    f"width={raster_width_um} target_size={target_size} pixel_resolution={pixel_resolution_um}"
                )
            if not math.isclose(raster_height_um / float(target_size), pixel_resolution_um, rel_tol=0.0, abs_tol=1e-12):
                raise RuntimeError(
                    "Tile raster pixel resolution does not match the model input scale: "
                    f"height={raster_height_um} target_size={target_size} pixel_resolution={pixel_resolution_um}"
                )
            jobs.append(
                TileJob(
                    window_id=f"{design_id}__r{row_index:03d}_c{col_index:03d}",
                    row_index=row_index,
                    col_index=col_index,
                    solve_target_size=int(target_size),
                    pixel_resolution_um=float(pixel_resolution_um),
                    raster_bounds=solve_bounds,
                    ownership_bounds=patch_bounds,
                    patch_bounds=patch_bounds,
                    margin_bounds=margin_bounds,
                    solve_bounds=solve_bounds,
                    patch_grid_origin=(float(die_x0), float(die_y0)),
                    patch_grid_shape=(int(tile_count_y), int(tile_count_x)),
                    patch_size_um=float(patch_size_um),
                    margin_um=float(tile_context_um),
                )
            )
    return tuple(jobs)


def build_tiling_summary(
    design_id: str,
    *,
    die_bounds_um: Sequence[float],
    target_size: int,
    tile_jobs: Sequence[TileJob],
    tile_size_um: float = DEFAULT_TILE_SIZE_UM,
    tile_context_um: float = DEFAULT_TILE_CONTEXT_UM,
) -> TilingSummary:
    if len(die_bounds_um) != 4:
        raise ValueError(f"die_bounds_um must contain 4 values, got {die_bounds_um}")
    if not tile_jobs:
        raise ValueError("tile_jobs cannot be empty")

    row_indices = {int(job.row_index) for job in tile_jobs}
    col_indices = {int(job.col_index) for job in tile_jobs}
    die_x0, die_y0, die_x1, die_y1 = (float(value) for value in die_bounds_um)
    return TilingSummary(
        design_id=str(design_id),
        mode="patch-margin-model-window-tiling",
        die_width_um=float(die_x1 - die_x0),
        die_height_um=float(die_y1 - die_y0),
        solve_target_size=int(target_size),
        tile_width_um=float(tile_size_um - (2.0 * tile_context_um)),
        tile_height_um=float(tile_size_um - (2.0 * tile_context_um)),
        tile_context_um=float(tile_context_um),
        stride_um=float(tile_size_um - (2.0 * tile_context_um)),
        tile_count_x=len(col_indices),
        tile_count_y=len(row_indices),
        tile_count=len(tile_jobs),
    )


def accumulate_named_values(
    accumulator: MutableMapping[str, float],
    names: Sequence[str],
    values: Sequence[float],
    *,
    scale: float = 1.0,
) -> None:
    if len(names) != len(values):
        raise ValueError(f"names and values must match in length, got {len(names)} vs {len(values)}")
    for name, value in zip(names, values):
        key = str(name)
        accumulator[key] = accumulator.get(key, 0.0) + (float(value) * float(scale))


def accumulate_directed_values(
    accumulator: MutableMapping[tuple[str, str], float],
    master_names: Sequence[str],
    victim_names: Sequence[str],
    values: Sequence[Sequence[float]],
    *,
    scale: float = 1.0,
) -> None:
    if len(master_names) != len(values):
        raise ValueError(
            f"master_names and values row count must match, got {len(master_names)} vs {len(values)}"
        )
    for row_index, (master_name, row_values) in enumerate(zip(master_names, values)):
        if len(victim_names) != len(row_values):
            raise ValueError(
                "victim_names and values column count must match for each row, "
                f"row={row_index} {len(victim_names)} vs {len(row_values)}"
            )
        master = str(master_name)
        for victim_name, raw_value in zip(victim_names, row_values):
            victim = str(victim_name)
            if master == victim:
                continue
            value = float(raw_value) * float(scale)
            if value == 0.0:
                continue
            key = (master, victim)
            accumulator[key] = accumulator.get(key, 0.0) + value
