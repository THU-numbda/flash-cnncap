#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent

for extra_path in (str(REPO_ROOT), str(THIS_DIR)):
    if extra_path not in sys.path:
        sys.path.insert(0, extra_path)

from flash_common.grouped_sparse_reduce import reduce_qmap_to_all_conductors_sparse  # pylint: disable=wrong-import-position
import run as single_run  # pylint: disable=wrong-import-position
from spef_runtime import IndexedSpefAccumulator  # pylint: disable=wrong-import-position
from window_runtime import (  # pylint: disable=wrong-import-position
    PreparedWindowRaster,
    materialize_window_features,
    prepare_window_raster,
)


DEFAULT_PREFERRED_BATCH_SIZE = 24
DEFAULT_ENV_BATCH_SIZE = 64
DEFAULT_WINDOW_STREAM_SIZE = 24


@dataclass(frozen=True)
class MultiPipelineConfig:
    tech_path: Path
    target_size: int
    device_name: str
    total_batch_size: int
    master_batch_size: int
    max_windows: int
    window_stream_size: int
    compiled_model_manifest: Path
    total_compiled_model: Path
    env_compiled_model: Path


@dataclass
class PreparedWindow:
    def_path: Path
    window_id: str
    occupied: torch.Tensor
    full_local_map: torch.Tensor
    owned_local_counts: torch.Tensor
    owned_sparse_indices: torch.Tensor
    owned_sparse_counts: torch.Tensor
    owned_query_ids: torch.Tensor
    owned_query_output_indices: Tuple[int, ...]
    visible_master_local_ids: Tuple[int, ...]
    visible_master_output_indices: Tuple[int, ...]
    spef_accumulator: IndexedSpefAccumulator = field(default_factory=IndexedSpefAccumulator)


@dataclass(frozen=True)
class EnvWorkItem:
    window_index: int
    master_local_id: int
    master_output_index: int


@dataclass(frozen=True)
class BulkTimingBreakdown:
    model_load_seconds: float
    extension_load_seconds: float
    discover_windows_seconds: float
    preprocess_windows_seconds: float
    solve_gpu_seconds: float
    write_spef_seconds: float

    @property
    def total_seconds(self) -> float:
        return (
            float(self.model_load_seconds)
            + float(self.extension_load_seconds)
            + float(self.discover_windows_seconds)
            + float(self.preprocess_windows_seconds)
            + float(self.solve_gpu_seconds)
            + float(self.write_spef_seconds)
        )


@dataclass(frozen=True)
class BulkSolveResult:
    window_count: int
    env_work_item_count: int
    total_batch_size: int
    env_batch_size: int
    window_stream_size: int
    elapsed_seconds: float
    timing: BulkTimingBreakdown


@dataclass(frozen=True)
class ChunkSolveStats:
    env_work_item_count: int


@dataclass(frozen=True)
class StreamRunStats:
    preprocess_windows_seconds: float
    solve_gpu_seconds: float
    write_spef_seconds: float
    env_work_item_count: int
    written_windows: int


@dataclass(frozen=True)
class ChunkTensorCache:
    occupied: torch.Tensor
    full_local_map: torch.Tensor
    owned_local_counts: torch.Tensor
    owned_sparse_indices: torch.Tensor
    owned_sparse_counts: torch.Tensor


def _print_progress(message: str) -> None:
    print(message, flush=True)


def _chunked(values: Sequence, chunk_size: int) -> Iterable[Sequence]:
    for start in range(0, len(values), chunk_size):
        yield values[start:start + chunk_size]


def _to_single_config(config: MultiPipelineConfig) -> single_run.PipelineConfig:
    return single_run.PipelineConfig(
        tech_path=config.tech_path,
        target_size=int(config.target_size),
        device_name=str(config.device_name),
        total_batch_size=int(config.total_batch_size),
        master_batch_size=int(config.master_batch_size),
        tile_stream_size=1,
        compiled_model_manifest=config.compiled_model_manifest,
        total_compiled_model=config.total_compiled_model,
        env_compiled_model=config.env_compiled_model,
    )


def _resolve_def_paths(def_dir: Path, *, max_windows: int) -> List[Path]:
    resolved_dir = def_dir.resolve()
    if not resolved_dir.exists():
        raise FileNotFoundError(f"DEF directory not found: {resolved_dir}")
    if not resolved_dir.is_dir():
        raise NotADirectoryError(f"DEF path is not a directory: {resolved_dir}")
    if int(max_windows) <= 0:
        raise ValueError(f"--max-windows must be positive, got {max_windows}")

    def_paths = sorted(path.resolve() for path in resolved_dir.glob("*.def"))
    if not def_paths:
        raise RuntimeError(f"No DEF files were found in {resolved_dir}")
    return def_paths[: int(max_windows)]


def _pad_long_1d_tensors(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    if not tensors:
        raise ValueError("Cannot pad an empty tensor list.")
    device = tensors[0].device
    max_len = max(int(tensor.shape[0]) for tensor in tensors)
    out = torch.zeros((len(tensors), max_len), device=device, dtype=torch.long)
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


def _build_chunk_tensor_cache(windows: Sequence[PreparedWindow]) -> ChunkTensorCache:
    if not windows:
        raise ValueError("Cannot build chunk tensors from an empty window list.")
    return ChunkTensorCache(
        occupied=torch.stack([window.occupied for window in windows], dim=0).to(dtype=torch.float32).contiguous(),
        full_local_map=torch.stack([window.full_local_map for window in windows], dim=0).to(dtype=torch.long).contiguous(),
        owned_local_counts=_pad_long_1d_tensors([window.owned_local_counts for window in windows]).contiguous(),
        owned_sparse_indices=_pad_long_2d_tensors([window.owned_sparse_indices for window in windows]).contiguous(),
        owned_sparse_counts=_pad_long_1d_tensors([window.owned_sparse_counts for window in windows]).contiguous(),
    )


def _build_env_feature_batch(
    cache: ChunkTensorCache,
    work_items: Sequence[EnvWorkItem],
) -> torch.Tensor:
    if not work_items:
        raise ValueError("Cannot build an env feature batch from an empty work list.")
    window_indices = torch.as_tensor(
        [int(item.window_index) for item in work_items],
        device=cache.occupied.device,
        dtype=torch.long,
    )
    batch = cache.occupied.index_select(0, window_indices).clone()
    full_local_maps = cache.full_local_map.index_select(0, window_indices)
    master_ids = torch.as_tensor(
        [int(item.master_local_id) for item in work_items],
        device=batch.device,
        dtype=torch.long,
    )
    batch.sub_(full_local_maps.eq(master_ids.view(-1, 1, 1, 1)).to(dtype=batch.dtype) * 2.0)
    return batch.contiguous()


def _reduce_batched_qmap(
    q_map: torch.Tensor,
    cache: ChunkTensorCache,
    window_indices: Sequence[int],
) -> torch.Tensor:
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


def _accumulate_total_batch(
    windows: Sequence[PreparedWindow],
    window_indices: Sequence[int],
    reduced: torch.Tensor,
) -> None:
    for row_idx, window_index in enumerate(window_indices):
        window = windows[int(window_index)]
        if not window.owned_query_output_indices:
            continue
        values = reduced[row_idx].index_select(0, window.owned_query_ids).detach().cpu().numpy().astype(np.float64, copy=False)
        window.spef_accumulator.add_total_values(
            window.owned_query_output_indices,
            values,
            scale=float(single_run.MODEL_OUTPUT_TO_FARADS),
        )


def _accumulate_env_batch(
    windows: Sequence[PreparedWindow],
    work_items: Sequence[EnvWorkItem],
    reduced: torch.Tensor,
) -> None:
    for row_idx, item in enumerate(work_items):
        window = windows[item.window_index]
        if not window.owned_query_output_indices:
            continue
        values = reduced[row_idx].index_select(0, window.owned_query_ids).detach().cpu().numpy().astype(np.float64, copy=False)
        window.spef_accumulator.add_directed_row(
            int(item.master_output_index),
            window.owned_query_output_indices,
            values,
            scale=float(single_run.MODEL_OUTPUT_TO_FARADS),
        )


def _prepare_window_cpu(
    session: single_run.InferenceSession,
    config: MultiPipelineConfig,
    def_path: Path,
) -> PreparedWindowRaster:
    window_job = single_run._build_window_job(def_path, _to_single_config(config))
    return prepare_window_raster(
        def_path,
        tech_path=config.tech_path,
        target_size=int(window_job.solve_target_size),
        pixel_resolution=window_job.pixel_resolution_um,
        selected_layers=session.runtime_config.channel_layers,
        include_supply_nets=True,
        window_id=window_job.window_id,
    )


def _materialize_window_gpu(
    session: single_run.InferenceSession,
    prepared_window: PreparedWindowRaster,
    def_path: Path,
) -> PreparedWindow:
    window = materialize_window_features(
        prepared_window,
        device=session.device,
    )
    owned_query_local_ids = window.owned_query_local_ids.contiguous()
    visible_master_local_ids = tuple(int(value) for value in window.visible_master_local_ids.tolist())
    return PreparedWindow(
        def_path=def_path.resolve(),
        window_id=window.window_id,
        occupied=window.occupied.contiguous(),
        full_local_map=window.full_local_map.contiguous(),
        owned_local_counts=window.owned_local_counts.contiguous(),
        owned_sparse_indices=window.owned_sparse_indices.contiguous(),
        owned_sparse_counts=window.owned_sparse_counts.contiguous(),
        owned_query_ids=owned_query_local_ids,
        owned_query_output_indices=tuple(int(local_id) - 1 for local_id in owned_query_local_ids.tolist()),
        visible_master_local_ids=visible_master_local_ids,
        visible_master_output_indices=tuple(int(local_id) - 1 for local_id in visible_master_local_ids),
        spef_accumulator=IndexedSpefAccumulator.from_net_names(window.real_conductor_names),
    )


def _build_env_work_items(windows: Sequence[PreparedWindow]) -> List[EnvWorkItem]:
    items: List[EnvWorkItem] = []
    for window_index, window in enumerate(windows):
        if not window.owned_query_output_indices:
            continue
        for master_local_id, master_output_index in zip(
            window.visible_master_local_ids,
            window.visible_master_output_indices,
        ):
            items.append(
                EnvWorkItem(
                    window_index=window_index,
                    master_local_id=int(master_local_id),
                    master_output_index=int(master_output_index),
                )
            )
    return items


def _solve_chunk(
    session: single_run.InferenceSession,
    windows: Sequence[PreparedWindow],
    cache: ChunkTensorCache,
    *,
    total_batch_size: int,
    env_batch_size: int,
) -> ChunkSolveStats:
    active_window_indices = [window_index for window_index, window in enumerate(windows) if window.owned_query_output_indices]
    work_items = _build_env_work_items(windows)
    total_batches = list(_chunked(active_window_indices, int(total_batch_size)))
    env_batches = list(_chunked(work_items, int(env_batch_size)))

    for window_index_chunk in total_batches:
        index_tensor = torch.as_tensor(window_index_chunk, device=cache.occupied.device, dtype=torch.long)
        features = cache.occupied.index_select(0, index_tensor)
        q_map = single_run.forward_qmap(session.total_model, features)
        reduced = _reduce_batched_qmap(q_map, cache, window_index_chunk)
        _accumulate_total_batch(windows, window_index_chunk, reduced)

    for work_chunk in env_batches:
        reduction_window_indices = [int(item.window_index) for item in work_chunk]
        features = _build_env_feature_batch(cache, work_chunk)
        q_map = single_run.forward_qmap(session.env_model, features)
        reduced = _reduce_batched_qmap(q_map, cache, reduction_window_indices)
        _accumulate_env_batch(windows, work_chunk, reduced)

    return ChunkSolveStats(env_work_item_count=len(work_items))


def _write_chunk(
    windows: Sequence[PreparedWindow],
    *,
    out_spef_dir: Path,
    c_unit: str,
) -> float:
    out_spef_dir.mkdir(parents=True, exist_ok=True)
    write_started_at = time.perf_counter()
    for window in windows:
        window.spef_accumulator.write(
            out_spef_dir / f"{window.window_id}.spef",
            window_id=window.window_id,
            c_unit=c_unit,
        )
    return float(time.perf_counter() - write_started_at)


def _run_streamed_chunks(
    session: single_run.InferenceSession,
    config: MultiPipelineConfig,
    *,
    def_paths: Sequence[Path],
    out_spef_dir: Path,
    c_unit: str,
) -> StreamRunStats:
    preprocess_windows_seconds = 0.0
    solve_gpu_seconds = 0.0
    write_spef_seconds = 0.0
    env_work_item_count = 0
    total_windows = len(def_paths)
    total_stream_chunks = max(1, (total_windows + int(config.window_stream_size) - 1) // int(config.window_stream_size))
    written_windows = 0

    with tqdm(
        total=total_stream_chunks,
        desc="Chunks",
        unit="chunk",
        dynamic_ncols=True,
    ) as progress_bar:
        for def_chunk in _chunked(def_paths, int(config.window_stream_size)):
            preprocess_started_at = time.perf_counter()
            prepared_windows: List[PreparedWindowRaster] = [
                _prepare_window_cpu(session, config, def_path)
                for def_path in def_chunk
            ]
            preprocess_windows_seconds += float(time.perf_counter() - preprocess_started_at)

            solve_started_at = time.perf_counter()
            windows_chunk = [
                _materialize_window_gpu(session, prepared_window, def_path)
                for prepared_window, def_path in zip(prepared_windows, def_chunk)
            ]
            chunk_cache = _build_chunk_tensor_cache(windows_chunk)
            solve_stats = _solve_chunk(
                session,
                windows_chunk,
                chunk_cache,
                total_batch_size=int(config.total_batch_size),
                env_batch_size=int(config.master_batch_size),
            )
            solve_gpu_seconds += float(time.perf_counter() - solve_started_at)
            env_work_item_count += int(solve_stats.env_work_item_count)

            write_spef_seconds += _write_chunk(
                windows_chunk,
                out_spef_dir=out_spef_dir,
                c_unit=c_unit,
            )
            written_windows += len(windows_chunk)
            progress_bar.update(1)
            del chunk_cache
            del windows_chunk

    return StreamRunStats(
        preprocess_windows_seconds=float(preprocess_windows_seconds),
        solve_gpu_seconds=float(solve_gpu_seconds),
        write_spef_seconds=float(write_spef_seconds),
        env_work_item_count=int(env_work_item_count),
        written_windows=int(written_windows),
    )


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


def _print_summary(result: BulkSolveResult) -> None:
    print("Bulk Runtime Breakdown")
    rows = [
        ("1. Load compiled models", _format_float(result.timing.model_load_seconds), _format_percent(100.0 * result.timing.model_load_seconds / result.elapsed_seconds if result.elapsed_seconds > 0.0 else 0.0)),
        ("2. Load native extensions", _format_float(result.timing.extension_load_seconds), _format_percent(100.0 * result.timing.extension_load_seconds / result.elapsed_seconds if result.elapsed_seconds > 0.0 else 0.0)),
        ("3. Discover windows", _format_float(result.timing.discover_windows_seconds), _format_percent(100.0 * result.timing.discover_windows_seconds / result.elapsed_seconds if result.elapsed_seconds > 0.0 else 0.0)),
        ("4. Preprocess streamed windows", _format_float(result.timing.preprocess_windows_seconds), _format_percent(100.0 * result.timing.preprocess_windows_seconds / result.elapsed_seconds if result.elapsed_seconds > 0.0 else 0.0)),
        ("5. GPU inference + reduction", _format_float(result.timing.solve_gpu_seconds), _format_percent(100.0 * result.timing.solve_gpu_seconds / result.elapsed_seconds if result.elapsed_seconds > 0.0 else 0.0)),
        ("6. Write SPEFs", _format_float(result.timing.write_spef_seconds), _format_percent(100.0 * result.timing.write_spef_seconds / result.elapsed_seconds if result.elapsed_seconds > 0.0 else 0.0)),
        ("Total", _format_float(result.elapsed_seconds), _format_percent(100.0 if result.elapsed_seconds > 0.0 else 0.0)),
    ]
    _print_table(("Stage", "Time (s)", "Percent"), rows)
    print("Bulk Solve Summary")
    windows_per_second = 0.0 if result.elapsed_seconds <= 0.0 else float(result.window_count) / float(result.elapsed_seconds)
    print(
        "  "
        f"window_count={result.window_count} "
        f"env_work_items={result.env_work_item_count} "
        f"total_batch_size={result.total_batch_size} "
        f"env_batch_size={result.env_batch_size} "
        f"window_stream_size={result.window_stream_size} "
        f"windows_per_second={_format_float(windows_per_second)}"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run streamed grouped U-Net capacitance inference over many DEF windows and emit one SPEF per window.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--def-dir", type=Path, required=True, help="Directory containing input DEF windows.")
    parser.add_argument("--out-spef-dir", type=Path, required=True, help="Output directory for emitted SPEFs.")
    parser.add_argument("--tech", type=Path, required=True, help="Technology YAML, for example tech/nangate45.yaml.")
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
    parser.add_argument("--max-windows", type=int, default=1024, help="Maximum number of DEF windows to process.")
    parser.add_argument(
        "--window-stream-size",
        type=int,
        default=DEFAULT_WINDOW_STREAM_SIZE,
        help="Maximum number of windows processed per streamed chunk.",
    )
    parser.add_argument(
        "--total-batch-size",
        type=int,
        default=None,
        help=(
            "Batch size for total-model window inference. "
            "Defaults to min(24, total engine max_batch_size) from the manifest."
        ),
    )
    parser.add_argument(
        "--master-batch-size",
        type=int,
        default=None,
        help=(
            "Batch size for env-model master queries. "
            "Defaults to min(64, env engine max_batch_size) from the manifest."
        ),
    )
    parser.add_argument("--c-unit", type=str, default="PF", help="Capacitance unit used in the written SPEFs.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(sys.argv[1:] if argv is None else argv)

    if int(args.window_stream_size) <= 0:
        raise ValueError(f"--window-stream-size must be positive, got {args.window_stream_size}")

    base_config = single_run.PipelineConfig(
        tech_path=args.tech.resolve(),
        target_size=int(single_run.DEFAULT_TARGET_SIZE),
        device_name=str(single_run.DEFAULT_DEVICE_NAME),
        total_batch_size=1,
        master_batch_size=1,
        tile_stream_size=1,
        compiled_model_manifest=args.compiled_model_manifest.resolve(),
        total_compiled_model=args.total_compiled_model.resolve(),
        env_compiled_model=args.env_compiled_model.resolve(),
    )
    session, startup = single_run._measure_startup(base_config)
    total_batch_size = int(
        min(DEFAULT_PREFERRED_BATCH_SIZE, int(session.metadata.total_max_batch_size))
        if args.total_batch_size is None
        else args.total_batch_size
    )
    master_batch_size = int(
        min(DEFAULT_ENV_BATCH_SIZE, int(session.metadata.env_max_batch_size))
        if args.master_batch_size is None
        else args.master_batch_size
    )
    if total_batch_size <= 0:
        raise ValueError(f"--total-batch-size must be positive, got {total_batch_size}")
    if master_batch_size <= 0:
        raise ValueError(f"--master-batch-size must be positive, got {master_batch_size}")
    if total_batch_size > int(session.metadata.total_max_batch_size):
        raise RuntimeError(
            f"Runtime total_batch_size={total_batch_size} exceeds total engine max_batch_size={session.metadata.total_max_batch_size}"
        )
    if master_batch_size > int(session.metadata.env_max_batch_size):
        raise RuntimeError(
            f"Runtime master_batch_size={master_batch_size} exceeds env engine max_batch_size={session.metadata.env_max_batch_size}"
        )

    config = MultiPipelineConfig(
        tech_path=args.tech.resolve(),
        target_size=int(single_run.DEFAULT_TARGET_SIZE),
        device_name=str(single_run.DEFAULT_DEVICE_NAME),
        total_batch_size=total_batch_size,
        master_batch_size=master_batch_size,
        max_windows=int(args.max_windows),
        window_stream_size=int(args.window_stream_size),
        compiled_model_manifest=args.compiled_model_manifest.resolve(),
        total_compiled_model=args.total_compiled_model.resolve(),
        env_compiled_model=args.env_compiled_model.resolve(),
    )

    _print_progress(
        "Loaded compiled models: "
        f"device={startup.device} "
        f"model_load_seconds={_format_float(startup.model_load_seconds)} "
        f"extension_load_seconds={_format_float(startup.extension_load_seconds)} "
        f"total_batch_size={config.total_batch_size} "
        f"env_batch_size={config.master_batch_size} "
        f"window_stream_size={config.window_stream_size}"
    )

    discover_started_at = time.perf_counter()
    def_paths = _resolve_def_paths(args.def_dir, max_windows=config.max_windows)
    discover_windows_seconds = float(time.perf_counter() - discover_started_at)
    _print_progress(f"Selected {len(def_paths)} DEF windows from {args.def_dir.resolve()}")

    stream_stats = _run_streamed_chunks(
        session,
        config,
        def_paths=def_paths,
        out_spef_dir=args.out_spef_dir.resolve(),
        c_unit=args.c_unit,
    )

    timing = BulkTimingBreakdown(
        model_load_seconds=float(startup.model_load_seconds),
        extension_load_seconds=float(startup.extension_load_seconds),
        discover_windows_seconds=discover_windows_seconds,
        preprocess_windows_seconds=float(stream_stats.preprocess_windows_seconds),
        solve_gpu_seconds=float(stream_stats.solve_gpu_seconds),
        write_spef_seconds=float(stream_stats.write_spef_seconds),
    )
    result = BulkSolveResult(
        window_count=int(stream_stats.written_windows),
        env_work_item_count=int(stream_stats.env_work_item_count),
        total_batch_size=int(config.total_batch_size),
        env_batch_size=int(config.master_batch_size),
        window_stream_size=int(config.window_stream_size),
        elapsed_seconds=float(timing.total_seconds),
        timing=timing,
    )
    _print_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
