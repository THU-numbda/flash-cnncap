from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence, Tuple

import torch

from native_extension_utils import ensure_torch_cuda_arch_list


REPO_ROOT = Path(__file__).resolve().parents[1]
NATIVE_SOURCE_DIR = REPO_ROOT / "full-pipeline" / "native"
DEFAULT_TORCH_EXTENSION_DIR = REPO_ROOT / ".cache" / "torch_extensions"
DEFAULT_SPARSE_REDUCE_EXTENSION_NAME = "grouped_sparse_reduce_cuda_v2"

_SPARSE_REDUCE_CUDA_MODULE = None
_SPARSE_REDUCE_CUDA_LOAD_ATTEMPTED = False


def load_sparse_reduce_cuda_extension():
    global _SPARSE_REDUCE_CUDA_MODULE, _SPARSE_REDUCE_CUDA_LOAD_ATTEMPTED

    if _SPARSE_REDUCE_CUDA_MODULE is not None:
        return _SPARSE_REDUCE_CUDA_MODULE
    if _SPARSE_REDUCE_CUDA_LOAD_ATTEMPTED:
        raise RuntimeError("The grouped sparse CUDA reduction extension failed to initialize earlier in this process.")

    _SPARSE_REDUCE_CUDA_LOAD_ATTEMPTED = True

    if not torch.cuda.is_available():
        raise RuntimeError("The grouped sparse CUDA reduction extension requires CUDA.")

    ensure_torch_cuda_arch_list()

    try:
        from torch.utils.cpp_extension import load
    except ImportError as exc:
        raise RuntimeError("torch.utils.cpp_extension is required to build the grouped sparse CUDA reduction extension.") from exc

    source_paths = [
        NATIVE_SOURCE_DIR / "grouped_sparse_reduce_bindings.cpp",
        NATIVE_SOURCE_DIR / "grouped_sparse_reduce_cuda.cu",
    ]
    missing_sources = [str(path) for path in source_paths if not path.exists()]
    if missing_sources:
        raise RuntimeError(
            "Grouped sparse CUDA reduction sources are missing: "
            + ", ".join(missing_sources)
        )

    build_dir = DEFAULT_TORCH_EXTENSION_DIR / DEFAULT_SPARSE_REDUCE_EXTENSION_NAME
    build_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(DEFAULT_TORCH_EXTENSION_DIR))

    print("Building grouped sparse CUDA reduction extension...")
    print(f"Extension cache dir: {build_dir}")
    _SPARSE_REDUCE_CUDA_MODULE = load(
        name=DEFAULT_SPARSE_REDUCE_EXTENSION_NAME,
        sources=[str(path) for path in source_paths],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        build_directory=str(build_dir),
        verbose=True,
    )
    print("Grouped sparse CUDA reduction extension ready.")
    return _SPARSE_REDUCE_CUDA_MODULE


def sparse_reduce_cuda_forward(
    flat_q: torch.Tensor,
    sparse_indices: torch.Tensor,
    sparse_counts: torch.Tensor,
) -> torch.Tensor:
    module = load_sparse_reduce_cuda_extension()
    return module.sparse_reduce_forward(
        flat_q.contiguous(),
        sparse_indices.contiguous(),
        sparse_counts.contiguous(),
    )


def build_sparse_index_tensors(
    local_maps: Sequence[torch.Tensor],
    local_counts: Sequence[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = len(local_maps)
    if batch_size == 0:
        raise ValueError("Cannot build sparse index tensors from an empty batch.")
    if len(local_counts) != batch_size:
        raise ValueError(
            f"local_maps and local_counts must have the same batch size, got {batch_size} and {len(local_counts)}"
        )

    device = local_maps[0].device
    for local_map, counts in zip(local_maps, local_counts):
        if local_map.device != device or counts.device != device:
            raise ValueError("All local_maps and local_counts must live on the same device.")

    max_local = max(int(counts.shape[0]) for counts in local_counts)
    max_sparse_points = 0
    per_sample_indices = []

    for local_map, counts in zip(local_maps, local_counts):
        flat_local = local_map.reshape(-1)
        slot_count = int(counts.shape[0])
        sample_indices = [torch.zeros((0,), device=device, dtype=torch.long) for _ in range(slot_count)]
        present_local_ids = (
            torch.nonzero(counts[1:] > 0, as_tuple=False).reshape(-1).to(device="cpu", dtype=torch.long).tolist()
        )
        for local_offset in present_local_ids:
            local_id = int(local_offset) + 1
            indices = torch.nonzero(flat_local == local_id, as_tuple=False).reshape(-1).to(
                device=device,
                dtype=torch.long,
            )
            sample_indices[local_id] = indices
            max_sparse_points = max(max_sparse_points, int(indices.numel()))
        if slot_count < max_local:
            sample_indices.extend(
                torch.zeros((0,), device=device, dtype=torch.long)
                for _ in range(max_local - slot_count)
            )
        per_sample_indices.append(sample_indices)

    max_sparse_points = max(1, max_sparse_points)
    sparse_indices = torch.zeros((batch_size, max_local, max_sparse_points), device=device, dtype=torch.long)
    sparse_counts = torch.zeros((batch_size, max_local), device=device, dtype=torch.long)

    for batch_idx, sample_indices in enumerate(per_sample_indices):
        for local_id, indices in enumerate(sample_indices):
            if local_id == 0:
                continue
            count = int(indices.numel())
            sparse_counts[batch_idx, local_id] = count
            if count > 0:
                sparse_indices[batch_idx, local_id, :count] = indices

    return sparse_indices, sparse_counts


def reduce_qmap_to_all_conductors_sparse(
    q_map: torch.Tensor,
    local_counts: torch.Tensor,
    sparse_indices: torch.Tensor,
    sparse_counts: torch.Tensor,
    *,
    reduction: str = "sum",
    force_fp32: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if q_map.device.type != "cuda":
        raise RuntimeError("Grouped sparse CUDA reduction requires CUDA tensors.")

    reduce_dtype = torch.float32 if force_fp32 else q_map.dtype
    flat_q = q_map.to(dtype=reduce_dtype).reshape(q_map.shape[0], -1)
    effective_counts = local_counts.to(dtype=reduce_dtype)
    sums = sparse_reduce_cuda_forward(
        flat_q,
        sparse_indices.to(device=q_map.device, non_blocking=True),
        sparse_counts.to(device=q_map.device, non_blocking=True),
    )
    if reduction == "mean":
        return sums / effective_counts.clamp(min=1.0), effective_counts
    if reduction == "sum":
        return sums, effective_counts
    raise ValueError(f"Unsupported reduction: {reduction}")


def reduce_qmap_to_queries_sparse(
    q_map: torch.Tensor,
    local_counts: torch.Tensor,
    sparse_indices: torch.Tensor,
    sparse_counts: torch.Tensor,
    query_local_ids: torch.Tensor,
    *,
    reduction: str = "sum",
    force_fp32: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    reduced, areas = reduce_qmap_to_all_conductors_sparse(
        q_map,
        local_counts,
        sparse_indices,
        sparse_counts,
        reduction=reduction,
        force_fp32=force_fp32,
    )
    qids = query_local_ids.long()
    return torch.gather(reduced, 1, qids), torch.gather(areas, 1, qids)
