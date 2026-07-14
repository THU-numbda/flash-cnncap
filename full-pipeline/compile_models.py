#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent

for extra_path in (str(REPO_ROOT), str(THIS_DIR)):
    if extra_path not in sys.path:
        sys.path.insert(0, extra_path)

from model_runtime import (  # pylint: disable=wrong-import-position
    DEFAULT_TRT_BUILDER_OPT_LEVEL,
    DEFAULT_TRT_MAX_NUM_TACTICS,
    ModelSpec,
    compile_model_to_tensorrt_engine,
    infer_checkpoint_input_channels,
    resolve_device,
)
from pipeline_model import (  # pylint: disable=wrong-import-position
    DEFAULT_FULL_PIPELINE_MONAI_CONFIG,
    FULL_PIPELINE_MONAI_CONFIGS,
)
from window_runtime import build_runtime_config  # pylint: disable=wrong-import-position


DEFAULT_PREFERRED_BATCH_SIZE = 24


def _print_progress(message: str) -> None:
    print(message, flush=True)


def _safe_torch_load(path: Path | str, *, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile full-pipeline U-Net checkpoints into FP16 TensorRT Q-map engines.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for compiled artifacts and manifest.")
    parser.add_argument(
        "--tech",
        type=Path,
        required=True,
        help="Technology YAML used to resolve exact active layers when checkpoint metadata does not provide them.",
    )
    parser.add_argument("--target-size", type=int, default=224, help="Fixed full-pipeline target size.")

    parser.add_argument("--total-checkpoint", type=Path, required=True)
    parser.add_argument("--env-checkpoint", type=Path, required=True)
    parser.add_argument("--total-model-type", choices=("unet",), default="unet")
    parser.add_argument("--env-model-type", choices=("unet",), default="unet")
    parser.add_argument(
        "--total-monai-config",
        type=str,
        choices=sorted(FULL_PIPELINE_MONAI_CONFIGS),
        default=None,
        help=(
            "Override the total-model MONAI config. Defaults to checkpoint metadata when available, "
            f"otherwise {DEFAULT_FULL_PIPELINE_MONAI_CONFIG}."
        ),
    )
    parser.add_argument(
        "--env-monai-config",
        type=str,
        choices=sorted(FULL_PIPELINE_MONAI_CONFIGS),
        default=None,
        help=(
            "Override the env-model MONAI config. Defaults to checkpoint metadata when available, "
            f"otherwise {DEFAULT_FULL_PIPELINE_MONAI_CONFIG}."
        ),
    )
    parser.add_argument("--total-max-batch-size", type=int, default=DEFAULT_PREFERRED_BATCH_SIZE)
    parser.add_argument("--env-max-batch-size", type=int, default=DEFAULT_PREFERRED_BATCH_SIZE)
    parser.add_argument("--total-opt-batch-size", type=int, default=DEFAULT_PREFERRED_BATCH_SIZE)
    parser.add_argument("--env-opt-batch-size", type=int, default=DEFAULT_PREFERRED_BATCH_SIZE)
    parser.add_argument("--total-output-name", type=str, default="total_qmap.engine")
    parser.add_argument("--env-output-name", type=str, default="env_qmap.engine")
    return parser.parse_args()


def _load_checkpoint_training_metadata(checkpoint_path: Path) -> dict | None:
    payload = _safe_torch_load(checkpoint_path, map_location="cpu")
    metadata = payload.get("metadata")
    if metadata is None:
        return None
    if not isinstance(metadata, dict):
        raise RuntimeError(f"Checkpoint metadata must be a dictionary: {checkpoint_path}")
    return metadata


def _validate_checkpoint_metadata(
    checkpoint_path: Path,
    metadata: dict | None,
    *,
    expected_goal: str,
    num_input_channels: int,
    expected_monai_config: str | None = None,
) -> None:
    if metadata is None:
        return

    goal = metadata.get("goal")
    if goal is not None and str(goal) != expected_goal:
        raise RuntimeError(
            f"Checkpoint goal metadata mismatch for {checkpoint_path}: expected {expected_goal}, found {goal}"
        )

    metadata_channels = metadata.get("num_input_channels")
    if metadata_channels is not None and int(metadata_channels) != int(num_input_channels):
        raise RuntimeError(
            "Checkpoint metadata num_input_channels mismatch for "
            f"{checkpoint_path}: inferred={num_input_channels} metadata={metadata_channels}"
        )

    active_layers = metadata.get("active_layers")
    if active_layers is not None and len(list(active_layers)) != int(num_input_channels):
        raise RuntimeError(
            f"Checkpoint active_layers length mismatch for {checkpoint_path}: "
            f"expected {num_input_channels}, found {len(list(active_layers))}"
        )

    if expected_monai_config is not None:
        metadata_monai_config = metadata.get("monai_config")
        if metadata_monai_config is not None and str(metadata_monai_config) != str(expected_monai_config):
            raise RuntimeError(
                f"Checkpoint monai_config mismatch for {checkpoint_path}: "
                f"expected {expected_monai_config}, found {metadata_monai_config}"
            )


def _resolve_monai_config(
    cli_value: str | None,
    metadata: dict | None,
    *,
    role: str,
) -> str:
    if cli_value is not None:
        return str(cli_value)

    if metadata is not None:
        metadata_value = metadata.get("monai_config")
        if metadata_value is not None:
            resolved = str(metadata_value)
            if resolved not in FULL_PIPELINE_MONAI_CONFIGS:
                raise RuntimeError(
                    f"{role} checkpoint metadata requested unsupported MONAI config '{resolved}'. "
                    f"Expected one of {sorted(FULL_PIPELINE_MONAI_CONFIGS)}"
                )
            return resolved

    return DEFAULT_FULL_PIPELINE_MONAI_CONFIG


def _resolve_active_layers(
    *,
    tech_path: Path,
    num_input_channels: int,
    total_metadata: dict | None,
    env_metadata: dict | None,
) -> list[str]:
    total_layers = total_metadata.get("active_layers") if total_metadata is not None else None
    env_layers = env_metadata.get("active_layers") if env_metadata is not None else None

    if total_layers is not None and env_layers is not None:
        resolved_total = [str(layer) for layer in total_layers]
        resolved_env = [str(layer) for layer in env_layers]
        if resolved_total != resolved_env:
            raise RuntimeError(
                "Total/env checkpoints declare different active layer sets: "
                f"total={resolved_total} env={resolved_env}"
            )
        return resolved_total

    if total_layers is not None:
        return [str(layer) for layer in total_layers]
    if env_layers is not None:
        return [str(layer) for layer in env_layers]

    runtime_config = build_runtime_config(tech_path)
    available_layers = list(runtime_config.channel_layers)
    if int(num_input_channels) > len(available_layers):
        raise RuntimeError(
            "Checkpoint input channel count exceeds the technology stack: "
            f"channels={num_input_channels} tech_layers={len(available_layers)}"
        )
    return [str(layer) for layer in available_layers[: int(num_input_channels)]]


def main() -> int:
    args = _parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device("cuda")
    tech_path = args.tech.resolve()
    total_checkpoint = args.total_checkpoint.resolve()
    env_checkpoint = args.env_checkpoint.resolve()
    _print_progress("Inspecting checkpoint channel counts and metadata...")
    total_input_channels = infer_checkpoint_input_channels(total_checkpoint)
    env_input_channels = infer_checkpoint_input_channels(env_checkpoint)
    if total_input_channels != env_input_channels:
        raise RuntimeError(
            "Total/env checkpoints use different input channel counts: "
            f"total={total_input_channels} env={env_input_channels}"
        )
    num_input_channels = int(total_input_channels)
    total_metadata = _load_checkpoint_training_metadata(total_checkpoint)
    env_metadata = _load_checkpoint_training_metadata(env_checkpoint)
    total_monai_config = _resolve_monai_config(args.total_monai_config, total_metadata, role="total")
    env_monai_config = _resolve_monai_config(args.env_monai_config, env_metadata, role="env")
    _validate_checkpoint_metadata(
        total_checkpoint,
        total_metadata,
        expected_goal="total",
        num_input_channels=num_input_channels,
        expected_monai_config=total_monai_config,
    )
    _validate_checkpoint_metadata(
        env_checkpoint,
        env_metadata,
        expected_goal="env",
        num_input_channels=num_input_channels,
        expected_monai_config=env_monai_config,
    )
    active_layers = _resolve_active_layers(
        tech_path=tech_path,
        num_input_channels=num_input_channels,
        total_metadata=total_metadata,
        env_metadata=env_metadata,
    )
    if len(active_layers) != num_input_channels:
        raise RuntimeError(
            f"Resolved active layers length mismatch: expected {num_input_channels}, got {len(active_layers)}"
        )
    _print_progress(
        "Resolved TensorRT build contract: "
        f"channels={num_input_channels} target_size={int(args.target_size)} "
        f"active_layers={active_layers} "
        f"total_monai_config={total_monai_config} env_monai_config={env_monai_config}"
    )

    total_spec = ModelSpec(
        model_type=str(args.total_model_type),
        monai_config=str(total_monai_config),
        checkpoint_path=total_checkpoint,
    )
    env_spec = ModelSpec(
        model_type=str(args.env_model_type),
        monai_config=str(env_monai_config),
        checkpoint_path=env_checkpoint,
    )

    started_at = time.perf_counter()
    _print_progress(
        "Building total engine: "
        f"checkpoint={total_checkpoint} opt_batch={int(args.total_opt_batch_size)} max_batch={int(args.total_max_batch_size)} "
        f"builder_optimization_level={int(DEFAULT_TRT_BUILDER_OPT_LEVEL)}"
    )
    total_started_at = time.perf_counter()
    total_output = compile_model_to_tensorrt_engine(
        total_spec,
        num_input_channels=num_input_channels,
        device=device,
        output_path=out_dir / str(args.total_output_name),
        target_size=int(args.target_size),
        opt_batch_size=int(args.total_opt_batch_size),
        max_batch_size=int(args.total_max_batch_size),
    )
    _print_progress(
        f"Built total engine: artifact={total_output} elapsed_seconds={time.perf_counter() - total_started_at:.3f}"
    )
    _print_progress(
        "Building env engine: "
        f"checkpoint={env_checkpoint} opt_batch={int(args.env_opt_batch_size)} max_batch={int(args.env_max_batch_size)} "
        f"builder_optimization_level={int(DEFAULT_TRT_BUILDER_OPT_LEVEL)}"
    )
    env_started_at = time.perf_counter()
    env_output = compile_model_to_tensorrt_engine(
        env_spec,
        num_input_channels=num_input_channels,
        device=device,
        output_path=out_dir / str(args.env_output_name),
        target_size=int(args.target_size),
        opt_batch_size=int(args.env_opt_batch_size),
        max_batch_size=int(args.env_max_batch_size),
    )
    _print_progress(
        f"Built env engine: artifact={env_output} elapsed_seconds={time.perf_counter() - env_started_at:.3f}"
    )
    elapsed_s = time.perf_counter() - started_at

    manifest = {
        "ok": True,
        "artifact_format": "tensorrt_engine",
        "device": str(device),
        "precision": "fp16",
        "num_input_channels": int(num_input_channels),
        "active_layers": list(active_layers),
        "target_size": int(args.target_size),
        "builder_optimization_level": int(DEFAULT_TRT_BUILDER_OPT_LEVEL),
        "max_num_tactics": int(DEFAULT_TRT_MAX_NUM_TACTICS),
        "elapsed_seconds": float(elapsed_s),
        "models": {
            "total": {
                "checkpoint_path": str(total_checkpoint),
                "artifact_path": str(total_output),
                "model_type": str(args.total_model_type),
                "monai_config": str(total_monai_config),
                "opt_batch_size": int(args.total_opt_batch_size),
                "max_batch_size": int(args.total_max_batch_size),
                "checkpoint_metadata": total_metadata,
            },
            "env": {
                "checkpoint_path": str(env_checkpoint),
                "artifact_path": str(env_output),
                "model_type": str(args.env_model_type),
                "monai_config": str(env_monai_config),
                "opt_batch_size": int(args.env_opt_batch_size),
                "max_batch_size": int(args.env_max_batch_size),
                "checkpoint_metadata": env_metadata,
            },
        },
    }
    manifest_path = out_dir / "compiled_models.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _print_progress(f"Wrote manifest: {manifest_path}")
    print(json.dumps({**manifest, "manifest_path": str(manifest_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
