#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
FULL_PIPELINE_DIR = REPO_ROOT / "full-pipeline"
TRAINING_DIR = REPO_ROOT / "training"

for extra_path in (str(REPO_ROOT), str(FULL_PIPELINE_DIR), str(TRAINING_DIR)):
    if extra_path not in sys.path:
        sys.path.insert(0, extra_path)


def _load_module(module_name: str, relative_path: str):
    module_path = FULL_PIPELINE_DIR / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

model_runtime = _load_module("full_pipeline_model_runtime_sweep", "model_runtime.py")

DEFAULT_SYNTHETIC_WARMUP_ITERS = 16
DEFAULT_SYNTHETIC_BENCHMARK_ITERS = 128
DEFAULT_RANDOM_SEED = 11037


def _print_progress(message: str) -> None:
    print(message, flush=True)


def _parse_batch_sizes(text: str) -> list[int]:
    values: list[int] = []
    for raw in str(text).split(","):
        item = raw.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError(f"Batch sizes must be positive, got {value}")
        values.append(value)
    if not values:
        raise ValueError("At least one batch size must be provided.")
    deduped: list[int] = []
    seen = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile TensorRT total engines for several opt_batch_size values and benchmark synthetic total-engine inference runtime.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for engines and results.")
    parser.add_argument(
        "--tech",
        type=Path,
        default=None,
        help="Legacy compatibility flag. Ignored; the sweep now benchmarks only the total engine.",
    )
    parser.add_argument(
        "--def-dir",
        type=Path,
        default=None,
        help="Legacy compatibility flag. Ignored; synthetic benchmarking no longer reads DEF files.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="32,64,100,128",
        help="Comma-separated total-engine opt/max batch sizes to sweep.",
    )
    parser.add_argument(
        "--max-defs",
        type=int,
        default=0,
        help="Legacy compatibility flag. Ignored for synthetic benchmarking.",
    )
    parser.add_argument("--total-checkpoint", type=Path, required=True)
    parser.add_argument("--env-checkpoint", type=Path, default=None, help="Legacy compatibility flag. Ignored.")
    parser.add_argument("--total-model-type", choices=("unet",), default="unet")
    parser.add_argument("--env-model-type", choices=("unet",), default="unet", help="Legacy compatibility flag. Ignored.")
    parser.add_argument(
        "--total-monai-config",
        type=str,
        choices=sorted(model_runtime.FULL_PIPELINE_MONAI_CONFIGS),
        default=model_runtime.DEFAULT_FULL_PIPELINE_MONAI_CONFIG,
    )
    parser.add_argument(
        "--env-monai-config",
        type=str,
        choices=sorted(model_runtime.FULL_PIPELINE_MONAI_CONFIGS),
        default=model_runtime.DEFAULT_FULL_PIPELINE_MONAI_CONFIG,
        help="Legacy compatibility flag. Ignored.",
    )
    parser.add_argument("--target-size", type=int, default=224)
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=DEFAULT_SYNTHETIC_WARMUP_ITERS,
        help="Untimed synthetic total-engine warmup iterations per candidate.",
    )
    parser.add_argument(
        "--benchmark-iters",
        type=int,
        default=DEFAULT_SYNTHETIC_BENCHMARK_ITERS,
        help="Timed synthetic total-engine iterations per candidate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Base random seed used to generate deterministic synthetic feature tensors.",
    )
    parser.add_argument(
        "--c-unit",
        type=str,
        default="PF",
        help="Legacy compatibility flag. Ignored for synthetic benchmarking.",
    )
    return parser.parse_args()


def _summarize_elapsed_times_ms(values_s: Sequence[float]) -> dict[str, float]:
    if not values_s:
        return {
            "mean_ms": 0.0,
            "median_ms": 0.0,
            "p95_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
        }
    values_ms = [1000.0 * float(value) for value in values_s]
    return {
        "mean_ms": float(sum(values_ms) / len(values_ms)),
        "median_ms": float(statistics.median(values_ms)),
        "p95_ms": float(np.percentile(np.asarray(values_ms, dtype=np.float64), 95.0)),
        "min_ms": float(min(values_ms)),
        "max_ms": float(max(values_ms)),
    }


def _compile_total_engine_candidate(
    *,
    total_spec,
    num_input_channels: int,
    device: torch.device,
    target_size: int,
    opt_batch_size: int,
    output_path: Path,
) -> tuple[Path, float]:
    _print_progress(f"[opt={opt_batch_size}] Compiling total engine...")
    started_at = time.perf_counter()
    output = model_runtime.compile_model_to_tensorrt_engine(
        total_spec,
        num_input_channels=num_input_channels,
        device=device,
        output_path=output_path,
        target_size=int(target_size),
        opt_batch_size=int(opt_batch_size),
        max_batch_size=int(opt_batch_size),
    )
    elapsed_s = float(time.perf_counter() - started_at)
    _print_progress(f"[opt={opt_batch_size}] Total engine compile finished in {elapsed_s:.3f}s")
    return output, elapsed_s


def _synchronize_device(device: torch.device | None) -> None:
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure_elapsed_seconds(action, *, device: torch.device | None = None):
    _synchronize_device(device)
    started_at = time.perf_counter()
    result = action()
    _synchronize_device(device)
    return result, float(time.perf_counter() - started_at)


def _build_synthetic_feature_batch(
    *,
    batch_size: int,
    num_input_channels: int,
    target_size: int,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    features = torch.randn(
        (int(batch_size), int(num_input_channels), int(target_size), int(target_size)),
        generator=generator,
        dtype=torch.float32,
    )
    return features.to(device=device, dtype=torch.float32).contiguous()


def _load_compiled_total_model(
    *,
    total_engine_path: Path,
    num_input_channels: int,
    device: torch.device,
) -> tuple[object, float]:
    started_at = time.perf_counter()
    total_model = model_runtime.build_model(
        model_runtime.ModelSpec(compiled_engine_path=total_engine_path.resolve()),
        num_input_channels=int(num_input_channels),
        device=device,
    )
    return total_model, float(time.perf_counter() - started_at)


def _benchmark_synthetic_total_engine(
    *,
    total_model,
    device: torch.device,
    opt_batch_size: int,
    num_input_channels: int,
    target_size: int,
    warmup_iterations: int,
    benchmark_iterations: int,
    seed: int,
) -> dict:
    input_shape = (
        int(opt_batch_size),
        int(num_input_channels),
        int(target_size),
        int(target_size),
    )
    features = _build_synthetic_feature_batch(
        batch_size=int(opt_batch_size),
        num_input_channels=int(num_input_channels),
        target_size=int(target_size),
        device=device,
        seed=int(seed) + int(opt_batch_size),
    )

    output_shape: tuple[int, ...] | None = None
    if int(warmup_iterations) > 0:
        _print_progress(
            f"[opt={opt_batch_size}] Warmup total engine with {int(warmup_iterations)} synthetic iterations..."
        )
        for _ in range(int(warmup_iterations)):
            q_map = model_runtime.forward_qmap(total_model, features)
            output_shape = tuple(int(value) for value in q_map.shape)
        _synchronize_device(device)

    batch_wall_seconds: list[float] = []
    _print_progress(
        f"[opt={opt_batch_size}] Benchmarking total engine with {int(benchmark_iterations)} synthetic iterations..."
    )
    benchmark_started_at = time.perf_counter()
    for _ in range(int(benchmark_iterations)):
        q_map, elapsed_s = _measure_elapsed_seconds(
            lambda: model_runtime.forward_qmap(total_model, features),
            device=device,
        )
        output_shape = tuple(int(value) for value in q_map.shape)
        batch_wall_seconds.append(float(elapsed_s))
    benchmark_wall_seconds = float(time.perf_counter() - benchmark_started_at)

    return {
        "input_shape": [int(value) for value in input_shape],
        "output_shape": [int(value) for value in (output_shape or ())],
        "timed_batch_count": int(benchmark_iterations),
        "timed_sample_count": int(opt_batch_size) * int(benchmark_iterations),
        "benchmark_seconds_wall": float(benchmark_wall_seconds),
        "batch_latency": _summarize_elapsed_times_ms(batch_wall_seconds),
        "timed_batch_wall_seconds": [float(value) for value in batch_wall_seconds],
    }


def _benchmark_candidate(
    *,
    candidate_dir: Path,
    opt_batch_size: int,
    target_size: int,
    total_checkpoint: Path,
    total_model_type: str,
    total_monai_config: str,
    num_input_channels: int,
    device: torch.device,
    warmup_iterations: int,
    benchmark_iterations: int,
    seed: int,
) -> dict:
    candidate_dir.mkdir(parents=True, exist_ok=True)
    total_engine_path = candidate_dir / "total_qmap.engine"

    total_spec = model_runtime.ModelSpec(
        model_type=str(total_model_type),
        monai_config=str(total_monai_config),
        checkpoint_path=total_checkpoint,
    )

    _, compile_seconds = _compile_total_engine_candidate(
        total_spec=total_spec,
        num_input_channels=int(num_input_channels),
        device=device,
        target_size=int(target_size),
        opt_batch_size=int(opt_batch_size),
        output_path=total_engine_path,
    )

    total_model, startup_wall_seconds = _load_compiled_total_model(
        total_engine_path=total_engine_path,
        num_input_channels=int(num_input_channels),
        device=device,
    )
    _print_progress(f"[opt={opt_batch_size}] Startup finished in {startup_wall_seconds:.3f}s")

    benchmark = _benchmark_synthetic_total_engine(
        total_model=total_model,
        device=device,
        opt_batch_size=int(opt_batch_size),
        num_input_channels=int(num_input_channels),
        target_size=int(target_size),
        warmup_iterations=int(warmup_iterations),
        benchmark_iterations=int(benchmark_iterations),
        seed=int(seed),
    )
    benchmark_wall_seconds = float(benchmark["benchmark_seconds_wall"])
    total_wall_seconds = float(startup_wall_seconds + benchmark_wall_seconds)

    del total_model
    model_runtime._MODEL_CACHE.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    timed_batch_count = int(benchmark["timed_batch_count"])
    timed_sample_count = int(benchmark["timed_sample_count"])
    return {
        "benchmark_mode": "synthetic_total_random",
        "opt_batch_size": int(opt_batch_size),
        "compile_seconds": float(compile_seconds),
        "startup_seconds_reported": float(startup_wall_seconds),
        "startup_seconds_wall": float(startup_wall_seconds),
        "benchmark_seconds_wall": float(benchmark_wall_seconds),
        "total_end_to_end_seconds_wall": float(total_wall_seconds),
        "warmup_iterations": int(warmup_iterations),
        "benchmark_iterations": int(benchmark_iterations),
        "timed_batch_count": int(timed_batch_count),
        "timed_sample_count": int(timed_sample_count),
        "end_to_end_batches_per_second": (
            float(timed_batch_count / total_wall_seconds) if total_wall_seconds > 0.0 else 0.0
        ),
        "steady_state_batches_per_second": (
            float(timed_batch_count / benchmark_wall_seconds) if benchmark_wall_seconds > 0.0 else 0.0
        ),
        "end_to_end_samples_per_second": (
            float(timed_sample_count / total_wall_seconds) if total_wall_seconds > 0.0 else 0.0
        ),
        "steady_state_samples_per_second": (
            float(timed_sample_count / benchmark_wall_seconds) if benchmark_wall_seconds > 0.0 else 0.0
        ),
        "batch_latency": dict(benchmark["batch_latency"]),
        "timed_batch_wall_seconds": list(benchmark["timed_batch_wall_seconds"]),
        "input_shape": list(benchmark["input_shape"]),
        "output_shape": list(benchmark["output_shape"]),
        "total_engine_path": str(total_engine_path),
    }


def _print_summary_table(results: Sequence[dict]) -> None:
    if not results:
        return
    headers = (
        "opt_batch",
        "compile_s",
        "startup_s",
        "e2e_s",
        "steady_samp/s",
        "steady_batch/s",
        "mean_batch_ms",
        "p95_batch_ms",
    )
    rows = []
    for result in results:
        rows.append(
            (
                str(result["opt_batch_size"]),
                f'{float(result["compile_seconds"]):.3f}',
                f'{float(result["startup_seconds_wall"]):.3f}',
                f'{float(result["total_end_to_end_seconds_wall"]):.3f}',
                f'{float(result["steady_state_samples_per_second"]):.3f}',
                f'{float(result["steady_state_batches_per_second"]):.3f}',
                f'{float(result["batch_latency"]["mean_ms"]):.3f}',
                f'{float(result["batch_latency"]["p95_ms"]):.3f}',
            )
        )

    widths = [len(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def _format_row(values: Sequence[str]) -> str:
        return "  " + "  ".join(value.rjust(widths[idx]) for idx, value in enumerate(values))

    print("Sweep Summary")
    print(_format_row(headers))
    print("  " + "  ".join("-" * width for width in widths))
    for row in rows:
        print(_format_row(row))


def main() -> int:
    args = _parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    total_checkpoint = args.total_checkpoint.resolve()
    batch_sizes = _parse_batch_sizes(args.batch_sizes)
    device = model_runtime.resolve_device("cuda")
    warmup_iterations = int(args.warmup_iters)
    benchmark_iterations = int(args.benchmark_iters)
    if warmup_iterations < 0:
        raise ValueError(f"--warmup-iters must be non-negative, got {warmup_iterations}")
    if benchmark_iterations <= 0:
        raise ValueError(f"--benchmark-iters must be positive, got {benchmark_iterations}")

    if args.tech is not None or args.def_dir is not None or args.env_checkpoint is not None:
        _print_progress(
            "Ignoring legacy tech/DEF/env flags; the sweep now compiles and benchmarks only the total engine."
        )
    _print_progress(f"Total opt_batch_size sweep: {batch_sizes}")
    _print_progress(
        "Synthetic benchmark config: "
        f"warmup_iters={warmup_iterations} benchmark_iters={benchmark_iterations} seed={int(args.seed)}"
    )

    num_input_channels = int(model_runtime.infer_checkpoint_input_channels(total_checkpoint))

    sweep_results: list[dict] = []
    for opt_batch_size in batch_sizes:
        candidate_dir = out_dir / f"total_opt_{int(opt_batch_size):03d}"
        result = _benchmark_candidate(
            candidate_dir=candidate_dir,
            opt_batch_size=int(opt_batch_size),
            target_size=int(args.target_size),
            total_checkpoint=total_checkpoint,
            total_model_type=str(args.total_model_type),
            total_monai_config=str(args.total_monai_config),
            num_input_channels=num_input_channels,
            device=device,
            warmup_iterations=warmup_iterations,
            benchmark_iterations=benchmark_iterations,
            seed=int(args.seed),
        )
        sweep_results.append(result)

    results_payload = {
        "ok": True,
        "benchmark_mode": "synthetic_total_random",
        "device": str(device),
        "legacy_tech_path": (str(args.tech.resolve()) if args.tech is not None else None),
        "legacy_def_dir": (str(args.def_dir.resolve()) if args.def_dir is not None else None),
        "legacy_env_checkpoint": (str(args.env_checkpoint.resolve()) if args.env_checkpoint is not None else None),
        "legacy_env_model_type": str(args.env_model_type),
        "legacy_env_monai_config": str(args.env_monai_config),
        "batch_sizes": [int(value) for value in batch_sizes],
        "target_size": int(args.target_size),
        "num_input_channels": int(num_input_channels),
        "warmup_iterations": int(warmup_iterations),
        "benchmark_iterations": int(benchmark_iterations),
        "random_seed": int(args.seed),
        "total_checkpoint": str(total_checkpoint),
        "total_model_type": str(args.total_model_type),
        "total_monai_config": str(args.total_monai_config),
        "results": sweep_results,
    }
    results_path = out_dir / "sweep_results.json"
    results_path.write_text(json.dumps(results_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _print_summary_table(sweep_results)
    _print_progress(f"Wrote detailed results: {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
