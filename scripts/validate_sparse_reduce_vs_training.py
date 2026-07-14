#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
TRAINING_DIR = REPO_ROOT / "training"

for extra_path in (str(REPO_ROOT), str(TRAINING_DIR), str(THIS_DIR)):
    if extra_path not in sys.path:
        sys.path.insert(0, extra_path)

from flash_common.grouped_sparse_reduce import (  # pylint: disable=wrong-import-position
    build_sparse_index_tensors,
    load_sparse_reduce_cuda_extension,
    reduce_qmap_to_all_conductors_sparse,
)
from capbench.window_id_map_dataset import IdMapWindowDataset  # pylint: disable=wrong-import-position
import train as train_mod  # pylint: disable=wrong-import-position


DTYPE_MAP = {
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the fused CUDA sparse reduction used by flash_common/grouped_sparse_reduce.py "
            "against the current _reduce_qmap_to_queries() implementation in training/train.py. "
            "Background slot 0 is intentionally excluded because the benchmark kernel skips it."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("datasets/nangate45/small"),
        help="Dataset root containing density_maps and label directories",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=64,
        help="Maximum number of windows to load for validation",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=64,
        help="Maximum number of grouped total samples to validate",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=3,
        help="Number of random q_map trials to run per sample and per dtype",
    )
    parser.add_argument(
        "--dtype",
        choices=("fp16", "fp32", "both"),
        default="both",
        help="Floating-point dtype(s) to validate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Base random seed used for reproducible q_map generation",
    )
    parser.add_argument(
        "--atol-fp16",
        type=float,
        default=5e-2,
        help="Absolute tolerance used for fp16 comparisons",
    )
    parser.add_argument(
        "--rtol-fp16",
        type=float,
        default=5e-2,
        help="Relative tolerance used for fp16 comparisons",
    )
    parser.add_argument(
        "--atol-fp32",
        type=float,
        default=1e-4,
        help="Absolute tolerance used for fp32 comparisons",
    )
    parser.add_argument(
        "--rtol-fp32",
        type=float,
        default=1e-4,
        help="Relative tolerance used for fp32 comparisons",
    )
    return parser.parse_args()


def select_dtype_names(dtype_arg: str) -> List[str]:
    if dtype_arg == "both":
        return ["fp32", "fp16"]
    return [dtype_arg]


def build_total_dataset(args: argparse.Namespace):
    window_dir = (args.dataset_path / "density_maps").resolve()
    spef_dir = (args.dataset_path / "labels_rwcap").resolve()
    if not window_dir.exists():
        raise FileNotFoundError(f"Window directory not found: {window_dir}")
    if not spef_dir.exists():
        raise FileNotFoundError(f"SPEF directory not found: {spef_dir}")

    window_ids = IdMapWindowDataset.discover_limited_windows(
        window_dir=window_dir,
        max_windows=args.max_windows,
        spef_dir=spef_dir,
    )
    if not window_ids:
        raise RuntimeError("No windows with both density_maps and SPEF labels were found.")

    print("Building real-window U-Net dataset for validation...")
    base_dataset = IdMapWindowDataset(
        window_dir=window_dir,
        spef_dir=spef_dir,
        window_ids=window_ids,
        goal="self",
        solver_preference="rwcap",
        build_workers=2,
        trim_margin=True,
    )
    total_dataset = train_mod.CapBenchTotalDataset(base_dataset)
    input_channels = base_dataset.num_layers

    print(
        "Validation workload:",
        f"windows={len(base_dataset.get_window_ids())}",
        f"total_cases={len(total_dataset)}",
        f"channels={input_channels}",
        f"slots={total_dataset.num_local_slots}",
        f"max_conductors={total_dataset.max_conductors}",
    )
    return base_dataset, total_dataset, input_channels


def validate_reduction_for_dtype(
    *,
    dtype_name: str,
    torch_dtype: torch.dtype,
    args: argparse.Namespace,
    base_dataset,
    total_dataset,
    device: torch.device,
) -> Tuple[float, float]:
    atol = args.atol_fp16 if dtype_name == "fp16" else args.atol_fp32
    rtol = args.rtol_fp16 if dtype_name == "fp16" else args.rtol_fp32
    sample_limit = min(len(total_dataset), args.max_samples)
    if sample_limit <= 0:
        raise RuntimeError("No grouped total samples are available for validation.")

    max_pred_diff = 0.0
    max_area_diff = 0.0
    total_checks = 0

    print()
    print(
        f"Validating dtype={dtype_name}",
        f"samples={sample_limit}",
        f"trials={args.trials}",
        f"atol={atol}",
        f"rtol={rtol}",
    )

    for sample_idx in range(sample_limit):
        sample = total_dataset[sample_idx]
        _xs, local_map, local_counts, query_ids, _targets, _valid = sample
        sparse_indices, sparse_counts = build_sparse_index_tensors([local_map], [local_counts])

        local_counts_gpu = local_counts.unsqueeze(0).to(device=device, dtype=torch_dtype, non_blocking=True)
        sparse_indices_gpu = sparse_indices.to(device=device, non_blocking=True)
        sparse_counts_gpu = sparse_counts.to(device=device, non_blocking=True)
        local_map_gpu = local_map.to(device=device, dtype=torch.long, non_blocking=True).unsqueeze(0)

        query_ids = query_ids.to(device=device, dtype=torch.long, non_blocking=True).unsqueeze(0)
        if query_ids.shape[1] == 0:
            continue

        q_shape = (1, *local_map.shape)

        for trial_idx in range(args.trials):
            trial_seed = args.seed + (sample_idx * args.trials) + trial_idx
            generator = torch.Generator()
            generator.manual_seed(trial_seed)
            logits = torch.randn(q_shape, generator=generator, dtype=torch.float32).to(
                device=device,
                dtype=torch_dtype,
            )
            q_map = F.softplus(logits)

            kernel_preds_sum, kernel_areas = reduce_qmap_to_all_conductors_sparse(
                q_map,
                local_counts_gpu,
                sparse_indices_gpu,
                sparse_counts_gpu,
                reduction="sum",
            )
            ref_preds_sum, ref_areas = train_mod._reduce_qmap_to_queries(
                q_map,
                local_map_gpu,
                local_counts_gpu,
                query_ids,
                reduction="sum",
            )
            kernel_selected_sum = torch.gather(kernel_preds_sum, 1, query_ids)
            kernel_selected_areas = torch.gather(kernel_areas, 1, query_ids)

            if not torch.allclose(kernel_selected_sum, ref_preds_sum, atol=atol, rtol=rtol):
                pred_diff = float(torch.max(torch.abs(kernel_selected_sum - ref_preds_sum)).item())
                raise AssertionError(
                    f"Sum reduction mismatch for dtype={dtype_name}, sample={sample_idx}, trial={trial_idx}: "
                    f"max_pred_diff={pred_diff:.6g}"
                )
            if not torch.allclose(kernel_selected_areas, ref_areas, atol=atol, rtol=rtol):
                area_diff = float(torch.max(torch.abs(kernel_selected_areas - ref_areas)).item())
                raise AssertionError(
                    f"Area mismatch for dtype={dtype_name}, sample={sample_idx}, trial={trial_idx}: "
                    f"max_area_diff={area_diff:.6g}"
                )

            kernel_preds_mean, _ = reduce_qmap_to_all_conductors_sparse(
                q_map,
                local_counts_gpu,
                sparse_indices_gpu,
                sparse_counts_gpu,
                reduction="mean",
            )
            ref_preds_mean, _ = train_mod._reduce_qmap_to_queries(
                q_map,
                local_map_gpu,
                local_counts_gpu,
                query_ids,
                reduction="mean",
            )
            kernel_selected_mean = torch.gather(kernel_preds_mean, 1, query_ids)
            if not torch.allclose(kernel_selected_mean, ref_preds_mean, atol=atol, rtol=rtol):
                pred_diff = float(torch.max(torch.abs(kernel_selected_mean - ref_preds_mean)).item())
                raise AssertionError(
                    f"Mean reduction mismatch for dtype={dtype_name}, sample={sample_idx}, trial={trial_idx}: "
                    f"max_pred_diff={pred_diff:.6g}"
                )

            max_pred_diff = max(
                max_pred_diff,
                float(torch.max(torch.abs(kernel_selected_sum - ref_preds_sum)).item()),
                float(torch.max(torch.abs(kernel_selected_mean - ref_preds_mean)).item()),
            )
            max_area_diff = max(
                max_area_diff,
                float(torch.max(torch.abs(kernel_selected_areas - ref_areas)).item()),
            )
            total_checks += 1

    print(
        f"PASS dtype={dtype_name}",
        f"checks={total_checks}",
        f"max_pred_diff={max_pred_diff:.6g}",
        f"max_area_diff={max_area_diff:.6g}",
    )
    return max_pred_diff, max_area_diff


def main() -> int:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This validation script requires CUDA because the fused sparse reduction kernel is CUDA-only.")

    device = torch.device("cuda")

    # Build once up front so compile time is not interleaved with the checks.
    load_sparse_reduce_cuda_extension()

    base_dataset, total_dataset, _input_channels = build_total_dataset(args)

    dtype_names = select_dtype_names(args.dtype)
    summary: Dict[str, Tuple[float, float]] = {}
    for dtype_name in dtype_names:
        summary[dtype_name] = validate_reduction_for_dtype(
            dtype_name=dtype_name,
            torch_dtype=DTYPE_MAP[dtype_name],
            args=args,
            base_dataset=base_dataset,
            total_dataset=total_dataset,
            device=device,
        )

    print()
    print("Validation summary:")
    for dtype_name in dtype_names:
        pred_diff, area_diff = summary[dtype_name]
        print(
            f"  {dtype_name}:",
            f"max_pred_diff={pred_diff:.6g}",
            f"max_area_diff={area_diff:.6g}",
        )
    print("Background slot 0 was intentionally excluded from comparison.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
