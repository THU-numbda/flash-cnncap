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

for extra_path in (str(REPO_ROOT), str(FULL_PIPELINE_DIR)):
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


model_runtime = _load_module("full_pipeline_model_runtime_unet_sweep", "model_runtime.py")

DEFAULT_WARMUP_ITERS = 16
DEFAULT_BENCHMARK_ITERS = 128
DEFAULT_RANDOM_SEED = 11037
DEFAULT_BATCH_SIZES = "8,16,24,32,48,64"


def _print_progress(message: str) -> None:
    print(message, flush=True)


def _parse_int_csv(text: str, *, field_name: str) -> list[int]:
    values: list[int] = []
    for raw in str(text).split(","):
        item = raw.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError(f"{field_name} values must be positive, got {value}")
        values.append(value)
    if not values:
        raise ValueError(f"{field_name} requires at least one value.")
    deduped: list[int] = []
    seen = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped


def _parse_config_csv(text: str | None, *, checkpoint_path: Path | None) -> list[str]:
    if text is None:
        if checkpoint_path is not None:
            return [str(model_runtime.DEFAULT_FULL_PIPELINE_MONAI_CONFIG)]
        return sorted(str(name) for name in model_runtime.FULL_PIPELINE_MONAI_CONFIGS)
    values: list[str] = []
    allowed = set(str(name) for name in model_runtime.FULL_PIPELINE_MONAI_CONFIGS)
    for raw in str(text).split(","):
        item = raw.strip()
        if not item:
            continue
        if item not in allowed:
            raise ValueError(f"Unknown MONAI config '{item}'. Expected one of {sorted(allowed)}")
        values.append(item)
    if not values:
        raise ValueError("--monai-configs requires at least one config.")
    deduped: list[str] = []
    seen = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile and benchmark TensorRT full-pipeline U-Net variants on synthetic inputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for engines and results.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint to load before export. Omit to benchmark random-weight architectures only.",
    )
    parser.add_argument(
        "--num-input-channels",
        type=int,
        default=None,
        help="Required when --checkpoint is omitted. Ignored otherwise unless used for validation.",
    )
    parser.add_argument(
        "--monai-configs",
        type=str,
        default=None,
        help=(
            "Comma-separated MONAI configs to benchmark. "
            "Defaults to D4_D_k5 when --checkpoint is set, otherwise all available configs."
        ),
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default=DEFAULT_BATCH_SIZES,
        help="Comma-separated opt/max batch sizes to sweep.",
    )
    parser.add_argument("--target-size", type=int, default=224)
    parser.add_argument("--warmup-iters", type=int, default=DEFAULT_WARMUP_ITERS)
    parser.add_argument("--benchmark-iters", type=int, default=DEFAULT_BENCHMARK_ITERS)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
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


def _resolve_num_input_channels(checkpoint_path: Path | None, explicit_channels: int | None) -> int:
    if checkpoint_path is not None:
        inferred = int(model_runtime.infer_checkpoint_input_channels(checkpoint_path))
        if explicit_channels is not None and int(explicit_channels) != inferred:
            raise ValueError(
                f"--num-input-channels={int(explicit_channels)} does not match checkpoint channels={inferred}"
            )
        return inferred
    if explicit_channels is None or int(explicit_channels) <= 0:
        raise ValueError("--num-input-channels must be provided and positive when --checkpoint is omitted.")
    return int(explicit_channels)


def _build_eager_model(
    *,
    monai_config: str,
    checkpoint_path: Path | None,
    num_input_channels: int,
    device: torch.device,
) -> torch.nn.Module:
    model = model_runtime.build_full_pipeline_unet(
        num_input_channels=int(num_input_channels),
        monai_config=str(monai_config),
    )
    model.to(device=device, dtype=torch.float32)
    model.eval()
    if checkpoint_path is not None:
        model_runtime.load_model_checkpoint(checkpoint_path, model, device)
    return model


def _load_compiled_model(
    *,
    engine_path: Path,
    num_input_channels: int,
    device: torch.device,
) -> tuple[object, float]:
    started_at = time.perf_counter()
    loaded = model_runtime.build_model(
        model_runtime.ModelSpec(compiled_engine_path=engine_path.resolve()),
        num_input_channels=int(num_input_channels),
        device=device,
    )
    return loaded, float(time.perf_counter() - started_at)


def _benchmark_synthetic_engine(
    *,
    compiled_model,
    device: torch.device,
    batch_size: int,
    num_input_channels: int,
    target_size: int,
    warmup_iterations: int,
    benchmark_iterations: int,
    seed: int,
) -> dict:
    input_shape = (
        int(batch_size),
        int(num_input_channels),
        int(target_size),
        int(target_size),
    )
    features = _build_synthetic_feature_batch(
        batch_size=int(batch_size),
        num_input_channels=int(num_input_channels),
        target_size=int(target_size),
        device=device,
        seed=int(seed) + int(batch_size),
    )

    output_shape: tuple[int, ...] | None = None
    if int(warmup_iterations) > 0:
        for _ in range(int(warmup_iterations)):
            q_map = model_runtime.forward_qmap(compiled_model, features)
            output_shape = tuple(int(value) for value in q_map.shape)
        _synchronize_device(device)

    batch_wall_seconds: list[float] = []
    benchmark_started_at = time.perf_counter()
    for _ in range(int(benchmark_iterations)):
        q_map, elapsed_s = _measure_elapsed_seconds(
            lambda: model_runtime.forward_qmap(compiled_model, features),
            device=device,
        )
        output_shape = tuple(int(value) for value in q_map.shape)
        batch_wall_seconds.append(float(elapsed_s))
    benchmark_wall_seconds = float(time.perf_counter() - benchmark_started_at)

    return {
        "input_shape": [int(value) for value in input_shape],
        "output_shape": [int(value) for value in (output_shape or ())],
        "timed_batch_count": int(benchmark_iterations),
        "timed_sample_count": int(batch_size) * int(benchmark_iterations),
        "benchmark_seconds_wall": float(benchmark_wall_seconds),
        "batch_latency": _summarize_elapsed_times_ms(batch_wall_seconds),
    }


def _cleanup_candidate(*objects) -> None:
    for obj in objects:
        if obj is not None:
            del obj
    model_runtime._MODEL_CACHE.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_candidate(
    *,
    out_dir: Path,
    monai_config: str,
    batch_size: int,
    checkpoint_path: Path | None,
    num_input_channels: int,
    target_size: int,
    device: torch.device,
    warmup_iterations: int,
    benchmark_iterations: int,
    seed: int,
) -> dict:
    checkpoint_label = str(checkpoint_path) if checkpoint_path is not None else "random_weights"
    candidate_dir = out_dir / monai_config / f"bs_{int(batch_size):03d}"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    engine_path = candidate_dir / "qmap.engine"

    eager_model = None
    compiled_model = None
    try:
        eager_model = _build_eager_model(
            monai_config=str(monai_config),
            checkpoint_path=checkpoint_path,
            num_input_channels=int(num_input_channels),
            device=device,
        )
        _, compile_seconds = _measure_elapsed_seconds(
            lambda: model_runtime.compile_eager_qmap_model_to_tensorrt_engine(
                eager_model,
                num_input_channels=int(num_input_channels),
                device=device,
                output_path=engine_path,
                target_size=int(target_size),
                opt_batch_size=int(batch_size),
                max_batch_size=int(batch_size),
                model_label=f"{checkpoint_label}:{monai_config}",
            ),
            device=device,
        )

        compiled_model, startup_seconds = _load_compiled_model(
            engine_path=engine_path,
            num_input_channels=int(num_input_channels),
            device=device,
        )
        benchmark = _benchmark_synthetic_engine(
            compiled_model=compiled_model,
            device=device,
            batch_size=int(batch_size),
            num_input_channels=int(num_input_channels),
            target_size=int(target_size),
            warmup_iterations=int(warmup_iterations),
            benchmark_iterations=int(benchmark_iterations),
            seed=int(seed),
        )
        benchmark_wall_seconds = float(benchmark["benchmark_seconds_wall"])
        timed_batch_count = int(benchmark["timed_batch_count"])
        timed_sample_count = int(benchmark["timed_sample_count"])
        return {
            "ok": True,
            "monai_config": str(monai_config),
            "batch_size": int(batch_size),
            "checkpoint": None if checkpoint_path is None else str(checkpoint_path),
            "weight_source": "checkpoint" if checkpoint_path is not None else "random_weights",
            "compile_seconds": float(compile_seconds),
            "startup_seconds_wall": float(startup_seconds),
            "benchmark_seconds_wall": float(benchmark_wall_seconds),
            "steady_state_batches_per_second": (
                float(timed_batch_count / benchmark_wall_seconds) if benchmark_wall_seconds > 0.0 else 0.0
            ),
            "steady_state_samples_per_second": (
                float(timed_sample_count / benchmark_wall_seconds) if benchmark_wall_seconds > 0.0 else 0.0
            ),
            "batch_latency": dict(benchmark["batch_latency"]),
            "input_shape": list(benchmark["input_shape"]),
            "output_shape": list(benchmark["output_shape"]),
            "engine_path": str(engine_path),
        }
    except Exception as exc:  # pylint: disable=broad-except
        return {
            "ok": False,
            "monai_config": str(monai_config),
            "batch_size": int(batch_size),
            "checkpoint": None if checkpoint_path is None else str(checkpoint_path),
            "weight_source": "checkpoint" if checkpoint_path is not None else "random_weights",
            "error": str(exc),
        }
    finally:
        _cleanup_candidate(eager_model, compiled_model)


def _print_summary_table(results: Sequence[dict]) -> None:
    if not results:
        return
    headers = (
        "config",
        "batch",
        "status",
        "steady_samp/s",
        "mean_ms",
        "p95_ms",
        "compile_s",
    )
    rows = []
    for result in results:
        if not bool(result.get("ok")):
            rows.append(
                (
                    str(result["monai_config"]),
                    str(result["batch_size"]),
                    "error",
                    "-",
                    "-",
                    "-",
                    "-",
                )
            )
            continue
        rows.append(
            (
                str(result["monai_config"]),
                str(result["batch_size"]),
                "ok",
                f'{float(result["steady_state_samples_per_second"]):.3f}',
                f'{float(result["batch_latency"]["mean_ms"]):.3f}',
                f'{float(result["batch_latency"]["p95_ms"]):.3f}',
                f'{float(result["compile_seconds"]):.3f}',
            )
        )

    widths = [len(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def _format_row(values: Sequence[str]) -> str:
        return "  " + "  ".join(value.rjust(widths[idx]) for idx, value in enumerate(values))

    print("U-Net TRT Sweep Summary")
    print(_format_row(headers))
    print("  " + "  ".join("-" * width for width in widths))
    for row in rows:
        print(_format_row(row))


def main() -> int:
    args = _parse_args()
    checkpoint_path = None if args.checkpoint is None else args.checkpoint.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    batch_sizes = _parse_int_csv(args.batch_sizes, field_name="--batch-sizes")
    monai_configs = _parse_config_csv(args.monai_configs, checkpoint_path=checkpoint_path)
    num_input_channels = _resolve_num_input_channels(checkpoint_path, args.num_input_channels)
    warmup_iterations = int(args.warmup_iters)
    benchmark_iterations = int(args.benchmark_iters)
    if warmup_iterations < 0:
        raise ValueError(f"--warmup-iters must be non-negative, got {warmup_iterations}")
    if benchmark_iterations <= 0:
        raise ValueError(f"--benchmark-iters must be positive, got {benchmark_iterations}")
    if int(args.target_size) <= 0:
        raise ValueError(f"--target-size must be positive, got {args.target_size}")

    device = model_runtime.resolve_device("cuda")
    _print_progress(
        "Sweep config: "
        f"checkpoint={'none' if checkpoint_path is None else checkpoint_path} "
        f"channels={num_input_channels} target_size={int(args.target_size)} "
        f"configs={monai_configs} batch_sizes={batch_sizes}"
    )

    results: list[dict] = []
    for monai_config in monai_configs:
        for batch_size in batch_sizes:
            _print_progress(f"[config={monai_config} batch={batch_size}] compiling and benchmarking...")
            result = _run_candidate(
                out_dir=out_dir,
                monai_config=str(monai_config),
                batch_size=int(batch_size),
                checkpoint_path=checkpoint_path,
                num_input_channels=int(num_input_channels),
                target_size=int(args.target_size),
                device=device,
                warmup_iterations=int(warmup_iterations),
                benchmark_iterations=int(benchmark_iterations),
                seed=int(args.seed),
            )
            results.append(result)
            if bool(result.get("ok")):
                _print_progress(
                    f"[config={monai_config} batch={batch_size}] "
                    f"steady_state_samples_per_second={float(result['steady_state_samples_per_second']):.3f} "
                    f"mean_batch_ms={float(result['batch_latency']['mean_ms']):.3f}"
                )
            else:
                _print_progress(f"[config={monai_config} batch={batch_size}] failed: {result['error']}")

    payload = {
        "ok": True,
        "device": str(device),
        "checkpoint": None if checkpoint_path is None else str(checkpoint_path),
        "weight_source": "checkpoint" if checkpoint_path is not None else "random_weights",
        "num_input_channels": int(num_input_channels),
        "target_size": int(args.target_size),
        "batch_sizes": [int(value) for value in batch_sizes],
        "monai_configs": [str(value) for value in monai_configs],
        "warmup_iterations": int(warmup_iterations),
        "benchmark_iterations": int(benchmark_iterations),
        "random_seed": int(args.seed),
        "results": results,
    }
    results_path = out_dir / "sweep_results.json"
    results_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _print_summary_table(results)
    _print_progress(f"Wrote detailed results: {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
