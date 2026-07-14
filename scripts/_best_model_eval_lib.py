"""Parallel queue runner for the post-ablation evaluation matrix."""

from __future__ import annotations

import csv
import math
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Lock, Thread
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / "training" / "train.py"
DEFAULT_NUM_GPUS = 7
DEFAULT_JOBS_PER_GPU = 2
DEFAULT_EVAL_REPEATS = 1
DEFAULT_EVAL_EPOCHS = 100
DEFAULT_BASE_SEED = 11037
DEFAULT_LR = 3e-4
DEFAULT_BATCH_SIZE = 16
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_WARMUP_EPOCHS = 5
DEFAULT_NUM_WORKERS: int | None = None
DEFAULT_PREFETCH_FACTOR = 4
DEFAULT_WINDOW_CACHE_SIZE = 4
DEFAULT_LEGACY_LAYERS = "POLY1_MET1_MET2"
DEFAULT_LEGACY_PADDING = 12
TRAIN_SUMMARY_PREFIX = "TRAIN_SUMMARY "
JOBLOG_HEADER = (
    "run_name\tcase\tgoal\tdataset_format\tdataset_path\trepeat_index\tseed\tgpu_id\tstatus\texit_code\t"
    "started_at\tfinished_at\tcommand\tstdout_log\tcheckpoint_dir\tmare\tflops\tsamples_per_s\tparams\n"
)


@dataclass(frozen=True)
class EvalModelSpec:
    key: str
    family: str
    model_type: str
    monai_config: str | None = None


@dataclass(frozen=True)
class EvalCaseSpec:
    name: str
    goal: str
    dataset_format: str
    dataset: str
    layers: str | None = None


@dataclass(frozen=True)
class ResolvedEvalCaseSpec:
    name: str
    goal: str
    dataset_format: str
    dataset_path: Path
    tech_path: Path | None = None
    layers: str | None = None


@dataclass(frozen=True)
class EvalRunSpec:
    case: ResolvedEvalCaseSpec
    repeat_index: int
    seed: int
    model: EvalModelSpec

    @property
    def run_name(self) -> str:
        return f"{self.case.name}_{self.model.key}_seed{self.seed}"


@dataclass(frozen=True)
class PendingEvalRun:
    run: EvalRunSpec
    resume_path: Path | None = None
    resume_epoch: int | None = None


@dataclass(frozen=True)
class EvalResult:
    run_name: str
    case_name: str
    goal: str
    model_key: str
    model_type: str
    monai_config: str | None
    dataset_format: str
    dataset_path: Path
    repeat_index: int
    seed: int
    gpu_id: int
    status: str
    exit_code: int
    started_at: str
    finished_at: str
    command: tuple[str, ...]
    stdout_path: Path
    checkpoint_dir: Path
    mare: float | None
    ratio_gt_5pct: float | None
    ratio_gt_10pct: float | None
    flops: float | None
    samples_per_second: float | None
    params: float | None


MODEL_SPECS: tuple[EvalModelSpec, ...] = (
    EvalModelSpec("D4_B", "unet", "unet", monai_config="D4_B"),
    EvalModelSpec("D4_D_k5", "unet", "unet", monai_config="D4_D_k5"),
    EvalModelSpec("resnet34", "resnet", "resnet34"),
    EvalModelSpec("resnet50", "resnet", "resnet50"),
)
DEFAULT_MODEL_KEYS: tuple[str, ...] = tuple(model.key for model in MODEL_SPECS)


CASE_SPECS: tuple[EvalCaseSpec, ...] = (
    EvalCaseSpec("total_small_nangate45", "total", "capbench", "nangate45/small"),
    EvalCaseSpec("total_small_sky130hd", "total", "capbench", "sky130hd/small"),
    EvalCaseSpec("total_legacy_cnncap", "total", "cnncap_legacy", "legacy_cnncap", layers=DEFAULT_LEGACY_LAYERS),
    EvalCaseSpec("total_medium_nangate45", "total", "capbench", "nangate45/medium"),
    EvalCaseSpec("total_medium_sky130hd", "total", "capbench", "sky130hd/medium"),
    EvalCaseSpec("total_large_nangate45", "total", "capbench", "nangate45/large"),
    EvalCaseSpec("total_large_sky130hd", "total", "capbench", "sky130hd/large"),
    EvalCaseSpec("env_small_nangate45", "env", "capbench", "nangate45/small"),
    EvalCaseSpec("env_small_sky130hd", "env", "capbench", "sky130hd/small"),
    EvalCaseSpec("env_legacy_cnncap", "env", "cnncap_legacy", "legacy_cnncap", layers=DEFAULT_LEGACY_LAYERS),
    EvalCaseSpec("env_medium_nangate45", "env", "capbench", "nangate45/medium"),
    EvalCaseSpec("env_medium_sky130hd", "env", "capbench", "sky130hd/medium"),
    EvalCaseSpec("env_large_nangate45", "env", "capbench", "nangate45/large"),
    EvalCaseSpec("env_large_sky130hd", "env", "capbench", "sky130hd/large"),
)


def select_model_specs(keys: Sequence[str] | None = None) -> list[EvalModelSpec]:
    model_by_key = {model.key: model for model in MODEL_SPECS}
    if keys:
        unknown = [key for key in keys if key not in model_by_key]
        if unknown:
            available = ", ".join(model_by_key)
            raise KeyError(f"Unknown model(s): {', '.join(unknown)}. Available: {available}")
        return [model_by_key[key] for key in keys]
    return list(MODEL_SPECS)


def list_case_specs() -> list[EvalCaseSpec]:
    return list(CASE_SPECS)


def select_case_specs(names: Sequence[str] | None = None, *, group: str = "all") -> list[EvalCaseSpec]:
    case_by_name = {case.name: case for case in CASE_SPECS}
    if names:
        unknown = [name for name in names if name not in case_by_name]
        if unknown:
            available = ", ".join(sorted(case_by_name))
            raise KeyError(f"Unknown case(s): {', '.join(unknown)}. Available: {available}")
        return [case_by_name[name] for name in names]

    normalized_group = group.strip().lower()
    if normalized_group not in {"all", "total", "env"}:
        raise ValueError(f"group must be one of all/total/env, got {group}")
    if normalized_group == "all":
        return list(CASE_SPECS)
    return [case for case in CASE_SPECS if case.goal == normalized_group]


def _build_seed_offsets(repeats: int) -> tuple[int, ...]:
    if repeats <= 0:
        raise ValueError(f"repeats must be positive, got {repeats}")
    return tuple(range(repeats))


def _build_gpu_ids(num_gpus: int) -> tuple[int, ...]:
    if num_gpus <= 0:
        raise ValueError(f"num_gpus must be positive, got {num_gpus}")
    return tuple(range(num_gpus))


def _resolve_gpu_ids(num_gpus: int, gpu_ids: Sequence[int] | None = None) -> tuple[int, ...]:
    if gpu_ids is None:
        return _build_gpu_ids(num_gpus)
    resolved = tuple(int(gpu_id) for gpu_id in gpu_ids)
    if not resolved:
        raise ValueError("gpu_ids must contain at least one GPU id.")
    if any(gpu_id < 0 for gpu_id in resolved):
        raise ValueError(f"gpu_ids must be non-negative, got {resolved}")
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"gpu_ids must not contain duplicates, got {resolved}")
    return resolved


def _extract_model_signature(command: Sequence[str]) -> tuple[str, str | None]:
    model_type = _extract_command_option(command, "--model_type")
    monai_config = _extract_command_option(command, "--monai-config")
    if model_type is None:
        model_type = "unet"
    return str(model_type), None if monai_config is None else str(monai_config)


def _resolve_model_key(model_type: str, monai_config: str | None) -> str:
    if model_type == "unet":
        return str(monai_config or "unet")
    return str(model_type)


def _build_eval_runs(
    cases: Sequence[ResolvedEvalCaseSpec],
    *,
    base_seed: int,
    repeats: int,
    models: Sequence[EvalModelSpec],
) -> list[EvalRunSpec]:
    runs: list[EvalRunSpec] = []
    phase_order = (
        ("unet", "total"),
        ("unet", "env"),
        ("resnet", "total"),
        ("resnet", "env"),
    )
    for family, goal in phase_order:
        phase_cases = [case for case in cases if case.goal == goal]
        phase_models = [model for model in models if model.family == family]
        for case in phase_cases:
            for model in phase_models:
                for repeat_index, seed_offset in enumerate(_build_seed_offsets(repeats), start=1):
                    runs.append(
                        EvalRunSpec(
                            case=case,
                            repeat_index=repeat_index,
                            seed=base_seed + seed_offset,
                            model=model,
                        )
                    )
    return runs


def _resolve_cpu_count() -> int:
    cpu_count = os.cpu_count()
    if cpu_count is None or cpu_count <= 0:
        return 1
    return int(cpu_count)


def _resolve_runtime_settings(
    *,
    num_workers: int | None,
    jobs_per_gpu: int,
    gpu_ids: Sequence[int],
) -> tuple[int, int, int, int, int]:
    if num_workers is not None and num_workers < 0:
        raise ValueError(f"num_workers must be non-negative, got {num_workers}")
    if jobs_per_gpu <= 0:
        raise ValueError(f"jobs_per_gpu must be positive, got {jobs_per_gpu}")

    cpu_count = _resolve_cpu_count()
    slot_count = max(1, len(gpu_ids) * jobs_per_gpu)
    cpu_budget_per_slot = max(1, cpu_count // slot_count)
    resolved_num_workers = int(num_workers) if num_workers is not None else max(2, min(4, cpu_budget_per_slot))
    resolved_build_workers = max(1, min(2, cpu_budget_per_slot))
    return (
        resolved_num_workers,
        DEFAULT_PREFETCH_FACTOR,
        resolved_build_workers,
        DEFAULT_WINDOW_CACHE_SIZE,
        cpu_count,
    )


def _build_train_command(
    run: EvalRunSpec,
    *,
    epochs: int,
    num_workers: int,
    prefetch_factor: int,
    build_workers: int,
    window_cache_size: int,
    resume_path: Path | None = None,
    save_base: Path,
    log_dir: Path,
    tb_dir: Path,
) -> tuple[list[str], Path, Path]:
    save_dir = save_base / run.model.key / run.case.name / f"seed_{run.seed}"
    log_file = log_dir / f"{run.run_name}.txt"
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--dataset-format",
        str(run.case.dataset_format),
        "--dataset-path",
        str(run.case.dataset_path),
        "--goal",
        str(run.case.goal),
        "--model_type",
        str(run.model.model_type),
        "--epoch",
        str(epochs),
        "--seed",
        str(run.seed),
        "--deterministic",
        "--num-workers",
        str(num_workers),
        "--prefetch-factor",
        str(prefetch_factor),
        "--build-workers",
        str(build_workers),
        "--window-cache-size",
        str(window_cache_size),
        "--lr",
        str(DEFAULT_LR),
        "--batch_size",
        str(DEFAULT_BATCH_SIZE),
        "--weight-decay",
        str(DEFAULT_WEIGHT_DECAY),
        "--warmup-epochs",
        str(DEFAULT_WARMUP_EPOCHS),
        "--save-dir",
        str(save_dir),
        "--savename",
        f"{run.run_name}.pth",
        "--logfile",
        str(log_file),
        "--tb-logdir",
        str(tb_dir / run.model.key / run.case.name),
    ]
    if run.model.monai_config is not None:
        command.extend(
            [
                "--monai-config",
                str(run.model.monai_config),
            ]
        )
    if run.case.dataset_format == "capbench":
        if run.case.tech_path is None:
            raise ValueError(f"CapBench case {run.case.name} is missing a tech_path")
        command.extend(
            [
                "--tech",
                str(run.case.tech_path),
                "--labels-solver",
                "rwcap",
            ]
        )
    else:
        command.extend(
            [
                "--layers",
                str(run.case.layers or DEFAULT_LEGACY_LAYERS),
                "--padding",
                str(DEFAULT_LEGACY_PADDING),
            ]
        )
    if resume_path is not None:
        command.extend(["--resume", str(resume_path)])
    return command, save_dir, log_file


def _extract_command_option(command: Sequence[str], option: str) -> str | None:
    for idx, token in enumerate(command):
        if token == option and idx + 1 < len(command):
            return str(command[idx + 1])
    return None


def _extract_command_epoch(command: Sequence[str]) -> int | None:
    raw_epoch = _extract_command_option(command, "--epoch")
    if raw_epoch is None:
        return None
    try:
        return int(raw_epoch)
    except ValueError:
        return None


def _parse_optional_metric(raw_value: str | None) -> float | None:
    if raw_value is None:
        return None
    lowered = raw_value.strip().lower()
    if lowered in {"na", "n/a", "none", "null"}:
        return None
    return float(raw_value)


def _extract_run_metrics(
    stdout_path: Path,
) -> tuple[float | None, float | None, float | None, float | None, float | None, float | None]:
    summary_line: str | None = None
    with stdout_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line.startswith(TRAIN_SUMMARY_PREFIX):
                summary_line = line

    if summary_line is None:
        return None, None, None, None, None, None

    fields: dict[str, str] = {}
    for token in summary_line.split()[1:]:
        key, sep, value = token.partition("=")
        if sep == "=" and key:
            fields[key] = value

    return (
        _parse_optional_metric(fields.get("mare")),
        _parse_optional_metric(fields.get("ratio_gt_5pct")),
        _parse_optional_metric(fields.get("ratio_gt_10pct")),
        _parse_optional_metric(fields.get("flops")),
        _parse_optional_metric(fields.get("samples_per_s")),
        _parse_optional_metric(fields.get("params")),
    )


def _extract_checkpoint_resume_epoch(checkpoint_path: Path) -> int | None:
    if not checkpoint_path.exists() or not checkpoint_path.is_file():
        return None
    try:
        import torch
    except ModuleNotFoundError:
        return None
    try:
        info = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        info = torch.load(checkpoint_path, map_location="cpu")
    except Exception:
        return None
    if not isinstance(info, dict):
        return None
    raw_epoch = info.get("epoch")
    if raw_epoch is None:
        return None
    try:
        return int(raw_epoch) + 1
    except (TypeError, ValueError):
        return None


def _extract_logged_model_type(stdout_path: Path) -> str | None:
    with stdout_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line.startswith("Model type:"):
                continue
            _, _, value = line.partition(":")
            model_type = value.strip()
            return model_type or None
    return None


def _extract_logged_monai_config(stdout_path: Path) -> str | None:
    with stdout_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if "config_id=" not in line:
                continue
            for token in line.split():
                key, sep, value = token.partition("=")
                if sep == "=" and key == "config_id" and value:
                    return value
    return None


def _build_completed_checkpoint_result(
    run: EvalRunSpec,
    *,
    command: Sequence[str],
    save_dir: Path,
    checkpoint_path: Path,
    stdout_path: Path,
) -> EvalResult:
    metrics = (None, None, None, None, None, None)
    if stdout_path.exists() and stdout_path.is_file():
        metrics = _extract_run_metrics(stdout_path)
    mare, ratio_gt_5pct, ratio_gt_10pct, flops, samples_per_second, params = metrics
    finished_source = stdout_path if stdout_path.exists() and stdout_path.is_file() else checkpoint_path
    finished_at = datetime.fromtimestamp(finished_source.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    return EvalResult(
        run_name=run.run_name,
        case_name=run.case.name,
        goal=run.case.goal,
        model_key=run.model.key,
        model_type=run.model.model_type,
        monai_config=run.model.monai_config,
        dataset_format=run.case.dataset_format,
        dataset_path=run.case.dataset_path,
        repeat_index=run.repeat_index,
        seed=run.seed,
        gpu_id=-1,
        status="skipped_existing",
        exit_code=0,
        started_at=finished_at,
        finished_at=finished_at,
        command=tuple(command),
        stdout_path=stdout_path,
        checkpoint_dir=save_dir,
        mare=mare,
        ratio_gt_5pct=ratio_gt_5pct,
        ratio_gt_10pct=ratio_gt_10pct,
        flops=flops,
        samples_per_second=samples_per_second,
        params=params,
    )


def _load_existing_joblog_results(joblog_path: Path) -> list[EvalResult]:
    if not joblog_path.exists():
        return []

    results: list[EvalResult] = []
    with joblog_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if not row:
                continue
            command_text = row.get("command") or ""
            parsed_command = tuple(shlex.split(command_text)) if command_text else ()
            model_type, monai_config = _extract_model_signature(parsed_command) if parsed_command else ("", None)
            stdout_text = str(row.get("stdout_log") or "")
            stdout_path = Path(stdout_text) if stdout_text else Path(".")
            summary_metrics = (None, None, None, None, None, None)
            if stdout_text and stdout_path.exists() and stdout_path.is_file():
                summary_metrics = _extract_run_metrics(stdout_path)
            mare, ratio_gt_5pct, ratio_gt_10pct, flops, samples_per_second, params = summary_metrics
            results.append(
                EvalResult(
                    run_name=str(row.get("run_name") or ""),
                    case_name=str(row.get("case") or ""),
                    goal=str(row.get("goal") or ""),
                    model_key=_resolve_model_key(model_type, monai_config) if model_type else "",
                    model_type=model_type,
                    monai_config=monai_config,
                    dataset_format=str(row.get("dataset_format") or ""),
                    dataset_path=Path(str(row.get("dataset_path") or ".")),
                    repeat_index=int(row.get("repeat_index") or 0),
                    seed=int(row.get("seed") or 0),
                    gpu_id=int(row.get("gpu_id") or -1),
                    status=str(row.get("status") or ""),
                    exit_code=int(row.get("exit_code") or 0),
                    started_at=str(row.get("started_at") or ""),
                    finished_at=str(row.get("finished_at") or ""),
                    command=parsed_command,
                    stdout_path=stdout_path,
                    checkpoint_dir=Path(str(row.get("checkpoint_dir") or ".")),
                    mare=_parse_optional_metric(row.get("mare")) if row.get("mare") else mare,
                    ratio_gt_5pct=ratio_gt_5pct,
                    ratio_gt_10pct=ratio_gt_10pct,
                    flops=_parse_optional_metric(row.get("flops")) if row.get("flops") else flops,
                    samples_per_second=_parse_optional_metric(row.get("samples_per_s"))
                    if row.get("samples_per_s")
                    else samples_per_second,
                    params=_parse_optional_metric(row.get("params")) if row.get("params") else params,
                )
            )
    return results


def _find_existing_success_result(
    run: EvalRunSpec,
    *,
    command: Sequence[str],
    save_dir: Path,
    stdout_path: Path,
    existing_joblog_results: Sequence[EvalResult],
    target_epochs: int,
) -> EvalResult | None:
    expected_model_type, expected_monai_config = _extract_model_signature(command)
    checkpoint_path = save_dir / f"{run.run_name}.pth"
    checkpoint_resume_epoch = _extract_checkpoint_resume_epoch(checkpoint_path)
    for existing in reversed(existing_joblog_results):
        if existing.case_name != run.case.name or int(existing.seed) != int(run.seed):
            continue
        if int(existing.exit_code) != 0:
            continue
        existing_model_type = existing.model_type or _extract_model_signature(existing.command)[0]
        existing_monai_config = existing.monai_config
        if existing_model_type != expected_model_type or existing_monai_config != expected_monai_config:
            continue
        existing_epochs = _extract_command_epoch(existing.command)
        if existing_epochs is not None and existing_epochs < target_epochs:
            continue
        if checkpoint_resume_epoch is not None and checkpoint_resume_epoch < target_epochs:
            continue
        return replace(existing, status="skipped_existing", gpu_id=-1)

    checkpoint_matches = sorted(save_dir.glob(f"{run.run_name}*.pth"))
    if checkpoint_resume_epoch is not None and checkpoint_resume_epoch >= target_epochs and checkpoint_matches:
        return _build_completed_checkpoint_result(
            run,
            command=command,
            save_dir=save_dir,
            checkpoint_path=checkpoint_path,
            stdout_path=stdout_path,
        )
    return None


def _average_metric(values: Sequence[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


def _mean_and_std_metric(values: Sequence[float | None]) -> tuple[float | None, float | None]:
    present = [value for value in values if value is not None]
    if not present:
        return None, None
    mean = sum(present) / len(present)
    if len(present) == 1:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in present) / (len(present) - 1)
    return mean, math.sqrt(variance)


def _format_summary_value(value: float | None, *, decimals: int, integer_like: bool = False) -> str:
    if value is None:
        return "na"
    if integer_like:
        return str(int(round(value)))
    return f"{value:.{decimals}f}"


def _format_joblog_value(value: float | None, *, decimals: int = 6, integer_like: bool = False) -> str:
    if value is None:
        return ""
    if integer_like:
        return str(int(round(value)))
    return f"{value:.{decimals}f}"


def _build_case_summary_line(results: Sequence[EvalResult]) -> str:
    successful = [result for result in results if result.exit_code == 0]
    mare_mean, mare_std = _mean_and_std_metric([result.mare for result in successful])
    ratio_gt_5pct_mean, ratio_gt_5pct_std = _mean_and_std_metric([result.ratio_gt_5pct for result in successful])
    ratio_gt_10pct_mean, ratio_gt_10pct_std = _mean_and_std_metric([result.ratio_gt_10pct for result in successful])
    flops = _average_metric([result.flops for result in successful])
    samples_per_second = _average_metric([result.samples_per_second for result in successful])
    params = _average_metric([result.params for result in successful])
    first = results[0]
    return (
        f"RUN_SUMMARY case={first.case_name} model={first.model_key} goal={first.goal} "
        f"ok={len(successful)}/{len(results)} "
        f"mare={_format_summary_value(mare_mean, decimals=6)}+-{_format_summary_value(mare_std, decimals=6)} "
        f"ratio_gt_5pct={_format_summary_value(ratio_gt_5pct_mean, decimals=6)}+-{_format_summary_value(ratio_gt_5pct_std, decimals=6)} "
        f"ratio_gt_10pct={_format_summary_value(ratio_gt_10pct_mean, decimals=6)}+-{_format_summary_value(ratio_gt_10pct_std, decimals=6)} "
        f"flops={_format_summary_value(flops, decimals=0, integer_like=True)} "
        f"samples_per_s={_format_summary_value(samples_per_second, decimals=3)} "
        f"params={_format_summary_value(params, decimals=0, integer_like=True)}"
    )


def run_best_model_queue(
    *,
    cases: Sequence[ResolvedEvalCaseSpec],
    num_gpus: int = DEFAULT_NUM_GPUS,
    gpu_ids: Sequence[int] | None = None,
    epochs: int = DEFAULT_EVAL_EPOCHS,
    seed: int = DEFAULT_BASE_SEED,
    num_workers: int | None = DEFAULT_NUM_WORKERS,
    repeats: int = DEFAULT_EVAL_REPEATS,
    model_keys: Sequence[str] | None = None,
    log_dir: Path | None = None,
    save_base: Path | None = None,
    tb_dir: Path | None = None,
) -> int:
    if not cases:
        raise ValueError("At least one evaluation case must be selected")
    if num_gpus <= 0:
        raise ValueError(f"num_gpus must be positive, got {num_gpus}")
    if repeats <= 0:
        raise ValueError(f"repeats must be positive, got {repeats}")
    if not TRAIN_SCRIPT.exists():
        raise FileNotFoundError(f"Training entrypoint not found: {TRAIN_SCRIPT}")

    models = select_model_specs(model_keys)
    runs = _build_eval_runs(cases, base_seed=seed, repeats=repeats, models=models)
    resolved_log_dir = (log_dir or (REPO_ROOT / "log" / "best_model_eval")).resolve()
    resolved_save_base = (save_base or (REPO_ROOT / "saved_models" / "best_model_eval")).resolve()
    resolved_tb_dir = (tb_dir or (REPO_ROOT / "runs" / "best_model_eval")).resolve()
    joblog_path = resolved_log_dir / "joblog.tsv"

    resolved_log_dir.mkdir(parents=True, exist_ok=True)
    resolved_save_base.mkdir(parents=True, exist_ok=True)
    resolved_tb_dir.mkdir(parents=True, exist_ok=True)
    existing_joblog_results = _load_existing_joblog_results(joblog_path)

    gpu_ids = _resolve_gpu_ids(num_gpus, gpu_ids)
    jobs_per_gpu = DEFAULT_JOBS_PER_GPU
    resolved_num_workers, prefetch_factor, build_workers, window_cache_size, cpu_count = _resolve_runtime_settings(
        num_workers=num_workers,
        jobs_per_gpu=jobs_per_gpu,
        gpu_ids=gpu_ids,
    )

    pending_runs: list[PendingEvalRun] = []
    existing_results: list[EvalResult] = []
    resumed_run_count = 0
    for run in runs:
        command, save_dir, _ = _build_train_command(
            run,
            epochs=epochs,
            num_workers=resolved_num_workers,
            prefetch_factor=prefetch_factor,
            build_workers=build_workers,
            window_cache_size=window_cache_size,
            resume_path=None,
            save_base=resolved_save_base,
            log_dir=resolved_log_dir,
            tb_dir=resolved_tb_dir,
        )
        stdout_path = resolved_log_dir / f"{run.run_name}_stdout.txt"
        existing = _find_existing_success_result(
            run,
            command=command,
            save_dir=save_dir,
            stdout_path=stdout_path,
            existing_joblog_results=existing_joblog_results,
            target_epochs=epochs,
        )
        if existing is not None:
            existing_results.append(existing)
            continue
        checkpoint_path = save_dir / f"{run.run_name}.pth"
        resume_epoch = _extract_checkpoint_resume_epoch(checkpoint_path)
        if resume_epoch is not None and resume_epoch < epochs:
            pending_runs.append(PendingEvalRun(run=run, resume_path=checkpoint_path, resume_epoch=resume_epoch))
            resumed_run_count += 1
            continue
        pending_runs.append(PendingEvalRun(run=run))

    effective_num_gpus = len(gpu_ids)
    worker_slots = tuple((gpu_id, slot_idx) for gpu_id in gpu_ids for slot_idx in range(jobs_per_gpu))

    print(
        f"Best-model evaluation: {len(cases)} cases, {len(runs)} training runs, "
        f"{effective_num_gpus} GPUs ({jobs_per_gpu} jobs/GPU, {len(worker_slots)} total slots)"
    )
    print(f"GPU ids: {', '.join(str(gpu_id) for gpu_id in gpu_ids)}")
    print(f"Models: {', '.join(model.key for model in models)}")
    print(f"Seeds/run: {', '.join(str(seed + offset) for offset in _build_seed_offsets(repeats))}")
    print(
        f"Epochs/run: {epochs} | Loader workers/job: {resolved_num_workers} | "
        f"Build workers/job: {build_workers} | Prefetch/job: {prefetch_factor} | "
        f"Window cache/job: {window_cache_size}"
    )
    print(f"Host CPU threads detected: {cpu_count}")
    print(f"Existing successful runs to skip: {len(existing_results)}")
    print(f"Pending runs: {len(pending_runs)}")
    print(f"Runs resuming from checkpoints: {resumed_run_count}")
    print(f"Fresh runs without checkpoints: {len(pending_runs) - resumed_run_count}")
    print("Case order:")
    for case in cases:
        print(f"  {case.name} | goal={case.goal} | format={case.dataset_format} | dataset={case.dataset_path}")
    print(f"Job log: {joblog_path}")
    print("")

    jobs: Queue[PendingEvalRun] = Queue()
    for pending_run in pending_runs:
        jobs.put(pending_run)

    active_processes: dict[tuple[int, int], subprocess.Popen[bytes]] = {}
    results: list[EvalResult] = list(existing_results)
    worker_errors: list[BaseException] = []
    lock = Lock()
    stop_requested = Event()
    summarized_run_groups: set[tuple[str, str]] = set()

    def append_joblog(result: EvalResult) -> None:
        row = [
            result.run_name,
            result.case_name,
            result.goal,
            result.dataset_format,
            str(result.dataset_path),
            str(result.repeat_index),
            str(result.seed),
            str(result.gpu_id),
            result.status,
            str(result.exit_code),
            result.started_at,
            result.finished_at,
            shlex.join(result.command),
            str(result.stdout_path),
            str(result.checkpoint_dir),
            _format_joblog_value(result.mare),
            _format_joblog_value(result.flops, decimals=0, integer_like=True),
            _format_joblog_value(result.samples_per_second, decimals=3),
            _format_joblog_value(result.params, decimals=0, integer_like=True),
        ]
        with joblog_path.open("a", encoding="utf-8") as handle:
            handle.write("\t".join(row) + "\n")

    def maybe_print_case_summary(case_name: str, model_key: str) -> None:
        summary_key = (case_name, model_key)
        if summary_key in summarized_run_groups:
            return
        case_results = [result for result in results if result.case_name == case_name and result.model_key == model_key]
        if len(case_results) != repeats:
            return
        print(_build_case_summary_line(case_results))
        summarized_run_groups.add(summary_key)

    def worker(gpu_id: int, slot_idx: int) -> None:
        slot_key = (gpu_id, slot_idx)
        while not stop_requested.is_set():
            try:
                pending_run = jobs.get_nowait()
            except Empty:
                return

            run = pending_run.run
            try:
                command, save_dir, _ = _build_train_command(
                    run,
                    epochs=epochs,
                    num_workers=resolved_num_workers,
                    prefetch_factor=prefetch_factor,
                    build_workers=build_workers,
                    window_cache_size=window_cache_size,
                    resume_path=pending_run.resume_path,
                    save_base=resolved_save_base,
                    log_dir=resolved_log_dir,
                    tb_dir=resolved_tb_dir,
                )
                save_dir.mkdir(parents=True, exist_ok=True)
                stdout_path = resolved_log_dir / f"{run.run_name}_stdout.txt"
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                env["OMP_NUM_THREADS"] = "1"
                env["MKL_NUM_THREADS"] = "1"
                env["OPENBLAS_NUM_THREADS"] = "1"
                env["NUMEXPR_NUM_THREADS"] = "1"
                started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with lock:
                    resume_suffix = ""
                    if pending_run.resume_path is not None and pending_run.resume_epoch is not None:
                        resume_suffix = f" resume_epoch={pending_run.resume_epoch} checkpoint={pending_run.resume_path}"
                    print(
                        f"[{started_at}] START {run.run_name} | "
                        f"case={run.case.name} model={run.model.key} goal={run.case.goal} repeat={run.repeat_index} "
                        f"seed={run.seed} gpu={gpu_id} slot={slot_idx + 1}/{jobs_per_gpu}{resume_suffix}"
                    )

                stdout_mode = "a" if pending_run.resume_path is not None else "w"
                with stdout_path.open(stdout_mode, encoding="utf-8") as stdout_handle:
                    if pending_run.resume_path is not None and pending_run.resume_epoch is not None:
                        stdout_handle.write(
                            f"\n[launcher] Resuming {run.run_name} from {pending_run.resume_path} at epoch {pending_run.resume_epoch}\n"
                        )
                        stdout_handle.flush()
                    process = subprocess.Popen(
                        command,
                        cwd=str(REPO_ROOT),
                        env=env,
                        stdout=stdout_handle,
                        stderr=subprocess.STDOUT,
                    )
                    with lock:
                        active_processes[slot_key] = process
                    exit_code = process.wait()

                finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                status = "ok" if exit_code == 0 else "failed"
                mare, ratio_gt_5pct, ratio_gt_10pct, flops, samples_per_second, params = _extract_run_metrics(stdout_path)
                result = EvalResult(
                    run_name=run.run_name,
                    case_name=run.case.name,
                    goal=run.case.goal,
                    model_key=run.model.key,
                    model_type=run.model.model_type,
                    monai_config=run.model.monai_config,
                    dataset_format=run.case.dataset_format,
                    dataset_path=run.case.dataset_path,
                    repeat_index=run.repeat_index,
                    seed=run.seed,
                    gpu_id=gpu_id,
                    status=status,
                    exit_code=exit_code,
                    started_at=started_at,
                    finished_at=finished_at,
                    command=tuple(command),
                    stdout_path=stdout_path,
                    checkpoint_dir=save_dir,
                    mare=mare,
                    ratio_gt_5pct=ratio_gt_5pct,
                    ratio_gt_10pct=ratio_gt_10pct,
                    flops=flops,
                    samples_per_second=samples_per_second,
                    params=params,
                )
                with lock:
                    active_processes.pop(slot_key, None)
                    results.append(result)
                    append_joblog(result)
                    print(f"[{finished_at}] DONE {run.run_name} | status={status}, exit={exit_code}")
                    maybe_print_case_summary(run.case.name, run.model.key)
            except BaseException as exc:  # pragma: no cover - worker failure path
                with lock:
                    worker_errors.append(exc)
                stop_requested.set()
            finally:
                with lock:
                    active_processes.pop(slot_key, None)
                jobs.task_done()

    if not joblog_path.exists():
        with joblog_path.open("w", encoding="utf-8") as handle:
            handle.write(JOBLOG_HEADER)

    summary_order: list[tuple[str, str]] = []
    seen_summary_keys: set[tuple[str, str]] = set()
    for run in runs:
        summary_key = (run.case.name, run.model.key)
        if summary_key in seen_summary_keys:
            continue
        seen_summary_keys.add(summary_key)
        summary_order.append(summary_key)

    for case_name, model_key in summary_order:
        maybe_print_case_summary(case_name, model_key)

    threads = [Thread(target=worker, args=(gpu_id, slot_idx), daemon=True) for gpu_id, slot_idx in worker_slots]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    except KeyboardInterrupt:  # pragma: no cover - interactive interruption path
        print("\nInterrupted; terminating active training jobs...")
        stop_requested.set()
        with lock:
            active = list(active_processes.values())
        for process in active:
            process.terminate()
        for process in active:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        for thread in threads:
            thread.join()
        return 130

    if worker_errors:
        raise RuntimeError(f"Best-model evaluation queue failed: {worker_errors[0]}")

    print("")
    print("═══════════════════════════════════════════════════")
    print(" BEST MODEL EVALUATION SUMMARY")
    print("═══════════════════════════════════════════════════")

    total = len(results)
    failed_results = [result for result in results if result.exit_code != 0]
    passed = total - len(failed_results)

    print(f"Passed: {passed} / {total}")
    if failed_results:
        print(f"Failed: {len(failed_results)} / {total}")
        print("")
        print("Failed runs:")
        for result in failed_results:
            print(f"  {result.run_name}")
        print("")
        print(f"Check logs in {resolved_log_dir}/")
    print("")
    print(f"Logs:        {resolved_log_dir}/")
    print(f"Checkpoints: {resolved_save_base}/")
    print(f"TensorBoard: tensorboard --logdir {resolved_tb_dir} --bind_all")
    print(f"Job log:     {joblog_path}")

    return 1 if failed_results else 0
