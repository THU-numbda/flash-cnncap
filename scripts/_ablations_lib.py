"""Repository-local ablation queue runner for Flash-CNNCap."""

from __future__ import annotations

import os
import math
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Lock, Thread
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / "training" / "train.py"
DEFAULT_NUM_GPUS = 8
DEFAULT_ABLATION_REPEATS = 5
TRAIN_SUMMARY_PREFIX = "TRAIN_SUMMARY "


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    flags: tuple[str, ...]


@dataclass(frozen=True)
class TrainingRunSpec:
    experiment: ExperimentSpec
    repeat_index: int
    seed: int

    @property
    def run_name(self) -> str:
        return f"{self.experiment.name}_seed{self.seed}"


@dataclass(frozen=True)
class ExperimentResult:
    run_name: str
    experiment_name: str
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


EXPERIMENTS: tuple[ExperimentSpec, ...] = (
    # Exact rows of the accepted paper's architecture-ablation table.
    ExperimentSpec("D3_B", ("--monai-config", "D3_B")),
    ExperimentSpec("D3_C", ("--monai-config", "D3_C")),
    ExperimentSpec("D4_A", ("--monai-config", "D4_A")),
    ExperimentSpec("D4_B", ("--monai-config", "D4_B")),
    ExperimentSpec("D4_C", ("--monai-config", "D4_C")),
    ExperimentSpec("D4_D", ("--monai-config", "D4_D")),
    ExperimentSpec("D4_E", ("--monai-config", "D4_E")),
    ExperimentSpec("D4_F", ("--monai-config", "D4_F")),
    ExperimentSpec("D4_C_k5", ("--monai-config", "D4_C_k5")),
    ExperimentSpec("D4_D_k5", ("--monai-config", "D4_D_k5")),
    ExperimentSpec("D4_F_k5", ("--monai-config", "D4_F_k5")),
    ExperimentSpec("D5_B", ("--monai-config", "D5_B")),
    ExperimentSpec("A4_A", ("--monai-config", "A4_A")),
)


def list_experiment_names() -> list[str]:
    return [experiment.name for experiment in EXPERIMENTS]


def select_experiments(names: Sequence[str] | None) -> list[ExperimentSpec]:
    if not names:
        return list(EXPERIMENTS)

    experiment_by_name = {experiment.name: experiment for experiment in EXPERIMENTS}
    unknown = [name for name in names if name not in experiment_by_name]
    if unknown:
        available = ", ".join(sorted(experiment_by_name))
        raise KeyError(f"Unknown experiment(s): {', '.join(unknown)}. Available: {available}")
    return [experiment_by_name[name] for name in names]


def _build_seed_offsets(repeats: int) -> tuple[int, ...]:
    if repeats <= 0:
        raise ValueError(f"repeats must be positive, got {repeats}")
    return tuple(range(repeats))


def _build_gpu_ids(num_gpus: int) -> tuple[int, ...]:
    if num_gpus <= 0:
        raise ValueError(f"num_gpus must be positive, got {num_gpus}")
    return tuple(range(num_gpus))


def _build_training_runs(
    experiments: Sequence[ExperimentSpec],
    *,
    base_seed: int,
    repeats: int,
) -> list[TrainingRunSpec]:
    runs: list[TrainingRunSpec] = []
    seed_offsets = _build_seed_offsets(repeats)
    for experiment in experiments:
        for repeat_index, seed_offset in enumerate(seed_offsets, start=1):
            runs.append(
                TrainingRunSpec(
                    experiment=experiment,
                    repeat_index=repeat_index,
                    seed=base_seed + seed_offset,
                )
            )
    return runs


def _build_train_command(
    run: TrainingRunSpec,
    *,
    dataset_path: Path,
    tech_path: Path,
    epochs: int,
    num_workers: int,
    save_base: Path,
    log_dir: Path,
    tb_dir: Path,
) -> tuple[list[str], Path, Path]:
    save_dir = save_base / run.experiment.name / f"seed_{run.seed}"
    log_file = log_dir / f"{run.run_name}.txt"
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        *run.experiment.flags,
        "--dataset-path",
        str(dataset_path),
        "--tech",
        str(tech_path),
        "--epoch",
        str(epochs),
        "--seed",
        str(run.seed),
        "--deterministic",
        "--num-workers",
        str(num_workers),
        "--save-dir",
        str(save_dir),
        "--savename",
        f"{run.run_name}.pth",
        "--logfile",
        str(log_file),
        "--tb-logdir",
        str(tb_dir / run.experiment.name),
    ]
    return command, save_dir, log_file


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
        if sep != "=" or not key:
            continue
        fields[key] = value

    return (
        _parse_optional_metric(fields.get("mare")),
        _parse_optional_metric(fields.get("ratio_gt_5pct")),
        _parse_optional_metric(fields.get("ratio_gt_10pct")),
        _parse_optional_metric(fields.get("flops")),
        _parse_optional_metric(fields.get("samples_per_s")),
        _parse_optional_metric(fields.get("params")),
    )


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


def _build_experiment_summary_line(results: Sequence[ExperimentResult]) -> str:
    if not results:
        raise ValueError("Expected at least one result to summarize")

    successful = [result for result in results if result.exit_code == 0]
    mare_mean, mare_std = _mean_and_std_metric([result.mare for result in successful])
    ratio_gt_5pct_mean, ratio_gt_5pct_std = _mean_and_std_metric([result.ratio_gt_5pct for result in successful])
    ratio_gt_10pct_mean, ratio_gt_10pct_std = _mean_and_std_metric([result.ratio_gt_10pct for result in successful])
    flops = _average_metric([result.flops for result in successful])
    samples_per_second = _average_metric([result.samples_per_second for result in successful])
    params = _average_metric([result.params for result in successful])
    experiment_name = results[0].experiment_name
    return (
        f"RUN_SUMMARY experiment={experiment_name} "
        f"ok={len(successful)}/{len(results)} "
        f"mare={_format_summary_value(mare_mean, decimals=6)}+-{_format_summary_value(mare_std, decimals=6)} "
        f"ratio_gt_5pct={_format_summary_value(ratio_gt_5pct_mean, decimals=6)}+-{_format_summary_value(ratio_gt_5pct_std, decimals=6)} "
        f"ratio_gt_10pct={_format_summary_value(ratio_gt_10pct_mean, decimals=6)}+-{_format_summary_value(ratio_gt_10pct_std, decimals=6)} "
        f"flops={_format_summary_value(flops, decimals=0, integer_like=True)} "
        f"samples_per_s={_format_summary_value(samples_per_second, decimals=3)} "
        f"params={_format_summary_value(params, decimals=0, integer_like=True)}"
    )


def run_ablation_queue(
    *,
    dataset_path: Path,
    tech_path: Path,
    num_gpus: int = DEFAULT_NUM_GPUS,
    epochs: int = 100,
    seed: int = 11037,
    num_workers: int = 0,
    repeats: int = DEFAULT_ABLATION_REPEATS,
    log_dir: Path | None = None,
    save_base: Path | None = None,
    tb_dir: Path | None = None,
    experiment_names: Sequence[str] | None = None,
) -> int:
    if num_gpus <= 0:
        raise ValueError(f"num_gpus must be positive, got {num_gpus}")
    if repeats <= 0:
        raise ValueError(f"repeats must be positive, got {repeats}")
    if not TRAIN_SCRIPT.exists():
        raise FileNotFoundError(f"Training entrypoint not found: {TRAIN_SCRIPT}")
    if not tech_path.exists():
        raise FileNotFoundError(f"Technology file not found: {tech_path}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    selected = select_experiments(experiment_names)
    seed_offsets = _build_seed_offsets(repeats)
    runs = _build_training_runs(selected, base_seed=seed, repeats=repeats)
    resolved_log_dir = (log_dir or (REPO_ROOT / "log" / "ablation")).resolve()
    resolved_save_base = (save_base or (REPO_ROOT / "saved_models" / "ablation")).resolve()
    resolved_tb_dir = (tb_dir or (REPO_ROOT / "runs" / "ablation")).resolve()
    joblog_path = resolved_log_dir / "joblog.tsv"

    resolved_log_dir.mkdir(parents=True, exist_ok=True)
    resolved_save_base.mkdir(parents=True, exist_ok=True)
    resolved_tb_dir.mkdir(parents=True, exist_ok=True)

    gpu_ids = _build_gpu_ids(num_gpus)

    print(f"Ablation study: {len(selected)} experiments, {len(runs)} training runs, {num_gpus} GPUs (1 job/GPU)")
    print(f"GPU ids: {', '.join(str(gpu_id) for gpu_id in gpu_ids)}")
    print(f"Seeds/run: {', '.join(str(seed + offset) for offset in seed_offsets)}")
    print(f"Dataset: {dataset_path} | Epochs: {epochs} | Workers/job: {num_workers}")
    print(f"Job log: {joblog_path}")
    print("")

    jobs: Queue[TrainingRunSpec] = Queue()
    for run in runs:
        jobs.put(run)

    active_processes: dict[int, subprocess.Popen[bytes]] = {}
    results: list[ExperimentResult] = []
    worker_errors: list[BaseException] = []
    lock = Lock()
    stop_requested = Event()
    summarized_experiments: set[str] = set()

    def append_joblog(result: ExperimentResult) -> None:
        row = [
            result.run_name,
            result.experiment_name,
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

    def maybe_print_experiment_summary(experiment_name: str) -> None:
        if experiment_name in summarized_experiments:
            return

        experiment_results = [result for result in results if result.experiment_name == experiment_name]
        if len(experiment_results) != repeats:
            return

        print(_build_experiment_summary_line(experiment_results))
        summarized_experiments.add(experiment_name)

    def worker(gpu_id: int) -> None:
        while not stop_requested.is_set():
            try:
                run = jobs.get_nowait()
            except Empty:
                return

            try:
                if stop_requested.is_set():
                    return
                command, save_dir, _ = _build_train_command(
                    run,
                    dataset_path=dataset_path,
                    tech_path=tech_path,
                    epochs=epochs,
                    num_workers=num_workers,
                    save_base=resolved_save_base,
                    log_dir=resolved_log_dir,
                    tb_dir=resolved_tb_dir,
                )
                save_dir.mkdir(parents=True, exist_ok=True)
                stdout_path = resolved_log_dir / f"{run.run_name}_stdout.txt"
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with lock:
                    print(
                        f"[{started_at}] START {run.run_name} | "
                        f"experiment={run.experiment.name} repeat={run.repeat_index} "
                        f"seed={run.seed} gpu={gpu_id} {' '.join(run.experiment.flags)}"
                    )

                with stdout_path.open("w", encoding="utf-8") as stdout_handle:
                    process = subprocess.Popen(
                        command,
                        cwd=str(REPO_ROOT),
                        env=env,
                        stdout=stdout_handle,
                        stderr=subprocess.STDOUT,
                    )
                    with lock:
                        active_processes[gpu_id] = process
                    exit_code = process.wait()

                finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                status = "ok" if exit_code == 0 else "failed"
                mare, ratio_gt_5pct, ratio_gt_10pct, flops, samples_per_second, params = _extract_run_metrics(stdout_path)
                result = ExperimentResult(
                    run_name=run.run_name,
                    experiment_name=run.experiment.name,
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
                    active_processes.pop(gpu_id, None)
                    results.append(result)
                    append_joblog(result)
                    print(f"[{finished_at}] DONE {run.run_name} | status={status}, exit={exit_code}")
                    maybe_print_experiment_summary(run.experiment.name)
            except BaseException as exc:  # pragma: no cover - worker failure path
                with lock:
                    worker_errors.append(exc)
                stop_requested.set()
            finally:
                with lock:
                    active_processes.pop(gpu_id, None)
                jobs.task_done()

    with joblog_path.open("w", encoding="utf-8") as handle:
        handle.write(
            "run_name\texperiment\trepeat_index\tseed\tgpu_id\tstatus\texit_code\tstarted_at\tfinished_at\tcommand\tstdout_log\tcheckpoint_dir\tmare\tflops\tsamples_per_s\tparams\n"
        )

    threads = [Thread(target=worker, args=(gpu_id,), daemon=True) for gpu_id in gpu_ids]
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
        raise RuntimeError(f"Ablation queue failed: {worker_errors[0]}")

    print("")
    print("═══════════════════════════════════════════════════")
    print(" ABLATION RESULTS SUMMARY")
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
