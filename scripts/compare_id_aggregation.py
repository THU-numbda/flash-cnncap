#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable, Iterator, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile, record_function
from tqdm import tqdm


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flash_common.grouped_sparse_reduce import (  # pylint: disable=wrong-import-position
    build_sparse_index_tensors,
    load_sparse_reduce_cuda_extension,
    reduce_qmap_to_all_conductors_sparse,
)


BENCHMARK_DTYPE = torch.float16


@dataclass
class WindowReduceSample:
    window_id: str
    num_conductors: int
    local_map: torch.Tensor  # [C, H, W], long
    local_counts: torch.Tensor  # [L], long
    sparse_indices: torch.Tensor  # [L, P], long
    sparse_counts: torch.Tensor  # [L], long


@dataclass
class BatchReduceData:
    window_ids: List[str]
    local_map_cpu: torch.Tensor  # [B, C, H, W], long
    local_counts_cpu: torch.Tensor  # [B, L], long
    sparse_indices_cpu: torch.Tensor  # [B, L, P], long
    sparse_counts_cpu: torch.Tensor  # [B, L], long

    @property
    def batch_size(self) -> int:
        return int(self.local_map_cpu.shape[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare simple PyTorch ID aggregation vs the old CUDA sparse reduction kernel "
            "using real binary-masks windows and random Softplus q_maps."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("datasets/nangate45/small"),
        help="Dataset root containing binary-masks/",
    )
    parser.add_argument(
        "--binary-masks-dir",
        type=Path,
        default=None,
        help="Override path to binary-masks directory",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=1024,
        help="Number of windows to load",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size used for throughput/correctness/trace",
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu", "auto"),
        default="cuda",
        help="Device to run on. CUDA is required for the kernel path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Base seed for deterministic random Softplus generation",
    )
    parser.add_argument(
        "--warmup-batches",
        type=int,
        default=8,
        help="Warmup batches before throughput timing",
    )
    parser.add_argument(
        "--throughput-passes",
        type=int,
        default=4,
        help="How many full passes over loaded windows to time for throughput",
    )
    parser.add_argument(
        "--trace-warmup-iters",
        type=int,
        default=2,
        help="Warmup iterations before trace recording",
    )
    parser.add_argument(
        "--trace-out-dir",
        type=Path,
        default=Path("traces/id_aggregation_compare"),
        help="Directory where trace JSON is written",
    )
    parser.add_argument(
        "--correctness-trials",
        type=int,
        default=2,
        help="Random Softplus trials per batch for correctness",
    )
    parser.add_argument(
        "--correctness-atol",
        type=float,
        default=1e-4,
        help="Absolute tolerance for correctness",
    )
    parser.add_argument(
        "--correctness-rtol",
        type=float,
        default=1e-4,
        help="Relative tolerance for correctness",
    )
    return parser.parse_args()


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _is_via_layer(layer_name: str) -> bool:
    return "VIA" in str(layer_name).upper()


def _window_paths(binary_masks_dir: Path, max_windows: int) -> List[Path]:
    all_paths = sorted(binary_masks_dir.glob("*.npz"))
    if len(all_paths) < max_windows:
        raise RuntimeError(
            f"Requested {max_windows} windows, but only {len(all_paths)} were found in {binary_masks_dir}."
        )
    return all_paths[:max_windows]


def _chunked(items: Sequence[WindowReduceSample], batch_size: int) -> Iterator[Sequence[WindowReduceSample]]:
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def _prepare_samples(binary_masks_dir: Path, max_windows: int) -> Tuple[List[WindowReduceSample], Tuple[int, int, int], float]:
    selected_paths = _window_paths(binary_masks_dir, max_windows)

    raw_maps: List[Tuple[str, List[np.ndarray]]] = []
    max_channels = 0
    max_h = 0
    max_w = 0

    for npz_path in tqdm(selected_paths, desc="Loading binary-masks", unit="win"):
        with np.load(npz_path, allow_pickle=True) as data:
            layers = [str(layer) for layer in data["layers"]]
            id_maps: List[np.ndarray] = []
            for layer_name in layers:
                if _is_via_layer(layer_name):
                    continue
                key = f"{layer_name}_idx"
                if key not in data:
                    continue
                id_map = data[key].astype(np.int64, copy=False)
                id_maps.append(id_map)

            if not id_maps:
                raise RuntimeError(f"{npz_path} has no non-VIA *_idx maps.")

        max_channels = max(max_channels, len(id_maps))
        max_h = max(max_h, int(id_maps[0].shape[0]))
        max_w = max(max_w, int(id_maps[0].shape[1]))
        raw_maps.append((npz_path.stem, id_maps))

    samples: List[WindowReduceSample] = []
    conductor_counts: List[int] = []

    for window_id, id_maps in raw_maps:
        channels = len(id_maps)
        h, w = id_maps[0].shape

        local_map = torch.zeros((max_channels, max_h, max_w), dtype=torch.long)
        stacked = np.stack(id_maps, axis=0).astype(np.int64, copy=False)
        local_map[:channels, :h, :w] = torch.from_numpy(stacked)

        max_local_id = int(local_map.max().item())
        local_counts = torch.bincount(local_map.reshape(-1), minlength=max_local_id + 1).to(dtype=torch.long)
        sparse_indices, sparse_counts = build_sparse_index_tensors([local_map], [local_counts])
        sparse_indices = sparse_indices.squeeze(0).contiguous()
        sparse_counts = sparse_counts.squeeze(0).contiguous()

        num_conductors = int((local_counts[1:] > 0).sum().item()) if local_counts.shape[0] > 1 else 0
        conductor_counts.append(num_conductors)
        samples.append(
            WindowReduceSample(
                window_id=window_id,
                num_conductors=num_conductors,
                local_map=local_map.contiguous(),
                local_counts=local_counts.contiguous(),
                sparse_indices=sparse_indices,
                sparse_counts=sparse_counts,
            )
        )

    avg_conductors = float(np.mean(conductor_counts)) if conductor_counts else 0.0
    return samples, (max_channels, max_h, max_w), avg_conductors


def _build_batches(samples: Sequence[WindowReduceSample], batch_size: int) -> List[BatchReduceData]:
    batches: List[BatchReduceData] = []
    for chunk in _chunked(samples, batch_size):
        chunk_list = list(chunk)
        local_map_batch = torch.stack([sample.local_map for sample in chunk_list], dim=0).contiguous()

        max_local = max(int(sample.local_counts.shape[0]) for sample in chunk_list)
        max_sparse_points = max(int(sample.sparse_indices.shape[1]) for sample in chunk_list)

        local_counts_batch = torch.zeros((len(chunk_list), max_local), dtype=torch.long)
        sparse_counts_batch = torch.zeros((len(chunk_list), max_local), dtype=torch.long)
        sparse_indices_batch = torch.zeros((len(chunk_list), max_local, max_sparse_points), dtype=torch.long)

        for batch_idx, sample in enumerate(chunk_list):
            local_slots = int(sample.local_counts.shape[0])
            sparse_points = int(sample.sparse_indices.shape[1])
            local_counts_batch[batch_idx, :local_slots] = sample.local_counts
            sparse_counts_batch[batch_idx, :local_slots] = sample.sparse_counts
            sparse_indices_batch[batch_idx, :local_slots, :sparse_points] = sample.sparse_indices

        batches.append(
            BatchReduceData(
                window_ids=[sample.window_id for sample in chunk_list],
                local_map_cpu=local_map_batch,
                local_counts_cpu=local_counts_batch.contiguous(),
                sparse_indices_cpu=sparse_indices_batch.contiguous(),
                sparse_counts_cpu=sparse_counts_batch.contiguous(),
            )
        )
    return batches


def _move_batch_to_device(batch: BatchReduceData, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        batch.local_map_cpu.to(device=device, non_blocking=True),
        batch.local_counts_cpu.to(device=device, non_blocking=True),
        batch.sparse_indices_cpu.to(device=device, non_blocking=True),
        batch.sparse_counts_cpu.to(device=device, non_blocking=True),
    )


def _softplus_random_q_map(
    shape: Tuple[int, int, int, int],
    *,
    device: torch.device,
    out_dtype: torch.dtype,
    generator: torch.Generator,
) -> torch.Tensor:
    raw = torch.randn(shape, device=device, dtype=torch.float32, generator=generator)
    positive = F.softplus(raw)
    if out_dtype != torch.float32:
        positive = positive.to(dtype=out_dtype)
    return positive


def _reduce_pytorch_simple(
    q_map: torch.Tensor,
    local_map: torch.Tensor,
    local_counts: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    flat_q = q_map.reshape(q_map.shape[0], -1)
    flat_ids = local_map.reshape(local_map.shape[0], -1).long()
    sums = torch.zeros((q_map.shape[0], local_counts.shape[1]), dtype=q_map.dtype, device=q_map.device)
    sums.scatter_add_(1, flat_ids, flat_q)
    if sums.shape[1] > 0:
        sums[:, 0] = 0
    return sums, local_counts.to(dtype=q_map.dtype)


def _reduce_cuda_kernel(
    q_map: torch.Tensor,
    local_counts: torch.Tensor,
    sparse_indices: torch.Tensor,
    sparse_counts: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return reduce_qmap_to_all_conductors_sparse(
        q_map,
        local_counts,
        sparse_indices,
        sparse_counts,
        reduction="sum",
        force_fp32=False,
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _warmup_reducers(
    *,
    batches: Sequence[BatchReduceData],
    warmup_batches: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> None:
    if not batches:
        return

    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 2000)
    warmup_limit = min(warmup_batches, len(batches))
    for batch in batches[:warmup_limit]:
        local_map, local_counts, sparse_indices, sparse_counts = _move_batch_to_device(batch, device)
        shape = (
            batch.batch_size,
            int(local_map.shape[1]),
            int(local_map.shape[2]),
            int(local_map.shape[3]),
        )
        q_map = _softplus_random_q_map(shape, device=device, out_dtype=dtype, generator=generator)
        _reduce_pytorch_simple(q_map, local_map, local_counts)
        _reduce_cuda_kernel(
            q_map,
            local_counts,
            sparse_indices,
            sparse_counts,
        )
    _synchronize(device)


def run_correctness(
    *,
    batches: Sequence[BatchReduceData],
    device: torch.device,
    seed: int,
    trials: int,
    atol: float,
    rtol: float,
) -> None:
    print()
    print("Running correctness comparison...")
    print(f"  batches={len(batches)} trials_per_batch={trials} atol={atol} rtol={rtol}")

    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 1000)

    max_abs_diff = 0.0
    max_rel_diff = 0.0
    checks = 0

    for batch_idx, batch in enumerate(tqdm(batches, desc="Correctness", unit="batch")):
        local_map, local_counts, sparse_indices, sparse_counts = _move_batch_to_device(batch, device)
        shape = (
            batch.batch_size,
            int(local_map.shape[1]),
            int(local_map.shape[2]),
            int(local_map.shape[3]),
        )
        for _ in range(trials):
            q_map = _softplus_random_q_map(shape, device=device, out_dtype=torch.float32, generator=generator)
            ref_pred, ref_area = _reduce_pytorch_simple(q_map, local_map, local_counts)
            ker_pred, ker_area = _reduce_cuda_kernel(
                q_map,
                local_counts,
                sparse_indices,
                sparse_counts,
            )

            ref_pred_cmp = ref_pred.to(dtype=ker_pred.dtype)
            ref_area_cmp = ref_area.to(dtype=ker_area.dtype)
            diff = torch.abs(ker_pred - ref_pred_cmp)
            rel = diff / ref_pred_cmp.abs().clamp(min=1e-12)

            max_abs_diff = max(max_abs_diff, float(diff.max().item()))
            max_rel_diff = max(max_rel_diff, float(rel.max().item()))
            checks += 1

            if not torch.allclose(ker_pred, ref_pred_cmp, atol=atol, rtol=rtol):
                raise AssertionError(
                    f"Prediction mismatch at batch={batch_idx}: "
                    f"max_abs_diff={float(diff.max().item()):.6g}, "
                    f"max_rel_diff={float(rel.max().item()):.6g}"
                )
            if not torch.allclose(ker_area, ref_area_cmp, atol=0.0, rtol=0.0):
                area_diff = torch.abs(ker_area - ref_area_cmp)
                raise AssertionError(
                    f"Area mismatch at batch={batch_idx}: max_area_diff={float(area_diff.max().item()):.6g}"
                )

    print(
        "Correctness PASS:",
        f"checks={checks}",
        f"max_abs_diff={max_abs_diff:.6g}",
        f"max_rel_diff={max_rel_diff:.6g}",
    )


def run_throughput(
    *,
    batches: Sequence[BatchReduceData],
    windows_count: int,
    passes: int,
    warmup_batches: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> None:
    print()
    print("Running throughput comparison...")
    print(f"  passes={passes} warmup_batches={warmup_batches} dtype={dtype}")

    _warmup_reducers(
        batches=batches,
        warmup_batches=warmup_batches,
        device=device,
        dtype=dtype,
        seed=seed,
    )

    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 3000)

    total_batches = len(batches) * passes
    total_windows = windows_count * passes
    pytorch_seconds = 0.0
    kernel_seconds = 0.0

    if device.type == "cuda":
        pytorch_events: List[Tuple[torch.cuda.Event, torch.cuda.Event]] = []
        kernel_events: List[Tuple[torch.cuda.Event, torch.cuda.Event]] = []
        iterator: Iterable[BatchReduceData] = batches * passes
        for batch in tqdm(iterator, total=total_batches, desc="Throughput", unit="batch"):
            local_map, local_counts, sparse_indices, sparse_counts = _move_batch_to_device(batch, device)
            shape = (
                batch.batch_size,
                int(local_map.shape[1]),
                int(local_map.shape[2]),
                int(local_map.shape[3]),
            )
            q_map = _softplus_random_q_map(shape, device=device, out_dtype=dtype, generator=generator)

            pyt_start = torch.cuda.Event(enable_timing=True)
            pyt_end = torch.cuda.Event(enable_timing=True)
            pyt_start.record()
            _reduce_pytorch_simple(q_map, local_map, local_counts)
            pyt_end.record()
            pytorch_events.append((pyt_start, pyt_end))

            ker_start = torch.cuda.Event(enable_timing=True)
            ker_end = torch.cuda.Event(enable_timing=True)
            ker_start.record()
            _reduce_cuda_kernel(
                q_map,
                local_counts,
                sparse_indices,
                sparse_counts,
            )
            ker_end.record()
            kernel_events.append((ker_start, ker_end))

        _synchronize(device)
        pytorch_seconds = sum(start.elapsed_time(end) for start, end in pytorch_events) / 1000.0
        kernel_seconds = sum(start.elapsed_time(end) for start, end in kernel_events) / 1000.0
    else:
        iterator = batches * passes
        for batch in tqdm(iterator, total=total_batches, desc="Throughput", unit="batch"):
            local_map, local_counts, sparse_indices, sparse_counts = _move_batch_to_device(batch, device)
            shape = (
                batch.batch_size,
                int(local_map.shape[1]),
                int(local_map.shape[2]),
                int(local_map.shape[3]),
            )
            q_map = _softplus_random_q_map(shape, device=device, out_dtype=dtype, generator=generator)

            start = perf_counter()
            _reduce_pytorch_simple(q_map, local_map, local_counts)
            pytorch_seconds += perf_counter() - start

            start = perf_counter()
            _reduce_cuda_kernel(
                q_map,
                local_counts,
                sparse_indices,
                sparse_counts,
            )
            kernel_seconds += perf_counter() - start

    pytorch_ms_per_batch = (pytorch_seconds / max(total_batches, 1)) * 1000.0
    kernel_ms_per_batch = (kernel_seconds / max(total_batches, 1)) * 1000.0
    pytorch_windows_per_s = total_windows / max(pytorch_seconds, 1e-12)
    kernel_windows_per_s = total_windows / max(kernel_seconds, 1e-12)

    print()
    print("Method                 Total Time (s)   Avg ms/batch    Windows/s")
    print("-------------------------------------------------------------------")
    print(
        f"{'pytorch_scatter':<22} "
        f"{pytorch_seconds:>14.6f} "
        f"{pytorch_ms_per_batch:>14.3f} "
        f"{pytorch_windows_per_s:>12.3f}"
    )
    print(
        f"{'cuda_sparse_kernel':<22} "
        f"{kernel_seconds:>14.6f} "
        f"{kernel_ms_per_batch:>14.3f} "
        f"{kernel_windows_per_s:>12.3f}"
    )
    print(f"Speedup (kernel / pytorch): {pytorch_seconds / max(kernel_seconds, 1e-12):.3f}x")


def run_trace(
    *,
    batches: Sequence[BatchReduceData],
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    trace_warmup_iters: int,
    trace_out_dir: Path,
) -> None:
    if not batches:
        raise RuntimeError("No batches available for tracing.")

    batch = batches[0]
    local_map, local_counts, sparse_indices, sparse_counts = _move_batch_to_device(batch, device)
    shape = (
        batch.batch_size,
        int(local_map.shape[1]),
        int(local_map.shape[2]),
        int(local_map.shape[3]),
    )

    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 4000)

    print()
    print("Running trace comparison...")
    print(f"  trace_warmup_iters={trace_warmup_iters} dtype={dtype} batch={batch.batch_size}")

    for _ in range(trace_warmup_iters):
        q_map = _softplus_random_q_map(shape, device=device, out_dtype=dtype, generator=generator)
        _reduce_pytorch_simple(q_map, local_map, local_counts)
        _reduce_cuda_kernel(
            q_map,
            local_counts,
            sparse_indices,
            sparse_counts,
        )
    _synchronize(device)

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    trace_out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = trace_out_dir / f"id_aggregation_compare_bs{batch.batch_size}.trace.json"

    with profile(activities=activities, record_shapes=True, profile_memory=True) as prof:
        with record_function("id_reduce.random_softplus"):
            q_map = _softplus_random_q_map(shape, device=device, out_dtype=dtype, generator=generator)
        with record_function("id_reduce.pytorch_scatter"):
            _reduce_pytorch_simple(q_map, local_map, local_counts)
        with record_function("id_reduce.sync_after_pytorch"):
            _synchronize(device)
        with record_function("id_reduce.cuda_sparse_kernel"):
            _reduce_cuda_kernel(
                q_map,
                local_counts,
                sparse_indices,
                sparse_counts,
            )
        _synchronize(device)

    prof.export_chrome_trace(str(trace_path))
    print(f"Saved trace: {trace_path}")


def main() -> int:
    args = parse_args()
    device = _resolve_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("This comparison requires CUDA because the kernel path is CUDA-only.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    dtype = BENCHMARK_DTYPE
    binary_masks_dir = (args.binary_masks_dir or (args.dataset_path / "binary-masks")).resolve()
    if not binary_masks_dir.exists():
        raise FileNotFoundError(f"binary-masks directory not found: {binary_masks_dir}")

    load_sparse_reduce_cuda_extension()

    samples, (channels, height, width), avg_conductors = _prepare_samples(binary_masks_dir, args.max_windows)
    batches = _build_batches(samples, args.batch_size)
    if not batches:
        raise RuntimeError("No batches were built from selected windows.")

    print("Loaded aggregation comparison workload:")
    print(f"  window_dir={binary_masks_dir}")
    print(f"  windows={len(samples)}")
    print(f"  batch_size={args.batch_size}")
    print(f"  padded_shape=({channels}, {height}, {width})")
    print(f"  avg_conductors={avg_conductors:.3f}")
    print("  dtype=fp16")
    print("  force_fp32_kernel=False")

    run_correctness(
        batches=batches,
        device=device,
        seed=args.seed,
        trials=args.correctness_trials,
        atol=args.correctness_atol,
        rtol=args.correctness_rtol,
    )
    run_throughput(
        batches=batches,
        windows_count=len(samples),
        passes=args.throughput_passes,
        warmup_batches=args.warmup_batches,
        device=device,
        dtype=dtype,
        seed=args.seed,
    )
    run_trace(
        batches=batches,
        device=device,
        dtype=dtype,
        seed=args.seed,
        trace_warmup_iters=args.trace_warmup_iters,
        trace_out_dir=args.trace_out_dir,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
