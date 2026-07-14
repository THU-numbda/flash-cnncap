#!/usr/bin/env python3
"""Unified best-model evaluation launcher with total-first scheduling."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _best_model_eval_lib import (
    DEFAULT_MODEL_KEYS,
    DEFAULT_EVAL_EPOCHS,
    DEFAULT_EVAL_REPEATS,
    DEFAULT_NUM_GPUS,
    DEFAULT_NUM_WORKERS,
    DEFAULT_BASE_SEED,
    ResolvedEvalCaseSpec,
    run_best_model_queue,
    select_case_specs,
)


REPO_ROOT = SCRIPT_DIR.parent


def _parse_gpu_ids(raw: str | None) -> tuple[int, ...] | None:
    if raw is None:
        return None
    values: list[int] = []
    for item in str(raw).split(","):
        token = item.strip()
        if not token:
            continue
        value = int(token)
        if value < 0:
            raise ValueError(f"--gpu-ids values must be non-negative, got {value}")
        values.append(value)
    if not values:
        raise ValueError("--gpu-ids requires at least one GPU id.")
    return tuple(values)


def _resolve_requested_models(models: Sequence[str] | None, monai_config: str | None) -> Sequence[str] | None:
    if models:
        if monai_config is not None and monai_config not in models:
            raise ValueError("--monai-config cannot conflict with explicit --model selections.")
        return list(models)
    if monai_config is not None:
        return [str(monai_config)]
    return None


def _load_capbench_dataset_resolver():
    try:
        from capbench.datasets import resolve_dataset_path
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "CapBench is not installed. Install it first from its checkout with "
            "`pip install -e \".[all]\"`."
        ) from exc
    return resolve_dataset_path


def _load_capbench_tech_root() -> Path:
    try:
        from capbench.paths import TECH_ROOT
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "CapBench is not installed. Install it first from its checkout with "
            "`pip install -e \".[all]\"`."
        ) from exc
    return TECH_ROOT


def _extract_process_node_from_path(dataset_path: Path) -> str:
    known_nodes = ("nangate45", "asap7", "sky130hd", "gf180", "tsmc28")
    for part in reversed(dataset_path.parts):
        lowered = part.lower()
        for node in known_nodes:
            if node in lowered:
                return node
    raise ValueError(f"Could not infer a process node from dataset path: {dataset_path}")


def _resolve_capbench_dataset_path(dataset: str) -> Path:
    candidate = Path(dataset).expanduser()
    if candidate.exists():
        dataset_path = candidate.resolve()
        required = [
            dataset_path / "density_maps",
            dataset_path / "labels_rwcap",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Dataset path is missing required evaluation artifacts: "
                + ", ".join(missing)
            )
        return dataset_path

    resolve_dataset_path = _load_capbench_dataset_resolver()
    return resolve_dataset_path(dataset, artifacts=["density_maps", "labels_rwcap"]).resolve()


def _resolve_tech_path(dataset_path: Path) -> Path:
    tech_root = _load_capbench_tech_root()
    process_node = _extract_process_node_from_path(dataset_path)
    tech_path = tech_root / f"{process_node}.yaml"
    if not tech_path.exists():
        raise FileNotFoundError(f"Could not find technology YAML for process node '{process_node}' at {tech_path}")
    return tech_path.resolve()


def _default_legacy_cnncap_root() -> Path:
    return (REPO_ROOT.parent / "CapBench" / "reference" / "datasets" / "cnncap_small").resolve()


def _resolve_legacy_cnncap_root(raw_path: Path | None) -> Path:
    candidate = raw_path.expanduser().resolve() if raw_path is not None else _default_legacy_cnncap_root()
    if not candidate.exists():
        raise FileNotFoundError(
            "Legacy CNNCap dataset root was not found. "
            f"Expected {candidate}. Pass --legacy-cnncap-root to override."
        )
    required = [
        candidate / "label",
        candidate / "POLY1.npz",
        candidate / "MET1.npz",
        candidate / "MET2.npz",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Legacy CNNCap dataset root is missing required files: " + ", ".join(missing)
        )
    return candidate


def _resolve_cases(case_names: Sequence[str] | None, *, group: str, legacy_cnncap_root: Path | None) -> list[ResolvedEvalCaseSpec]:
    selected = select_case_specs(case_names, group=group)
    resolved_legacy_root = _resolve_legacy_cnncap_root(legacy_cnncap_root) if any(
        case.dataset_format == "cnncap_legacy" for case in selected
    ) else None

    resolved: list[ResolvedEvalCaseSpec] = []
    for case in selected:
        if case.dataset_format == "capbench":
            dataset_path = _resolve_capbench_dataset_path(case.dataset)
            resolved.append(
                ResolvedEvalCaseSpec(
                    name=case.name,
                    goal=case.goal,
                    dataset_format=case.dataset_format,
                    dataset_path=dataset_path,
                    tech_path=_resolve_tech_path(dataset_path),
                    layers=None,
                )
            )
        else:
            if resolved_legacy_root is None:
                raise RuntimeError("Legacy CNNCap root resolution unexpectedly failed")
            resolved.append(
                ResolvedEvalCaseSpec(
                    name=case.name,
                    goal=case.goal,
                    dataset_format=case.dataset_format,
                    dataset_path=resolved_legacy_root,
                    tech_path=None,
                    layers=case.layers,
                )
            )
    return resolved


def _run(args: argparse.Namespace) -> int:
    resolved_cases = _resolve_cases(
        args.case,
        group=args.group,
        legacy_cnncap_root=args.legacy_cnncap_root,
    )
    return run_best_model_queue(
        cases=resolved_cases,
        num_gpus=args.num_gpus,
        gpu_ids=_parse_gpu_ids(args.gpu_ids),
        epochs=args.epochs,
        seed=args.seed,
        num_workers=args.num_workers,
        repeats=args.repeats,
        model_keys=_resolve_requested_models(args.model, args.monai_config),
        log_dir=args.log_dir,
        save_base=args.save_base,
        tb_dir=args.tb_dir,
    )


def _list(args: argparse.Namespace) -> int:
    selected = select_case_specs(args.case, group=args.group)
    for case in selected:
        print(f"{case.name}\tgoal={case.goal}\tformat={case.dataset_format}\tdataset={case.dataset}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/best_model_eval.py",
        description="Run the fixed multi-model evaluation matrix with total-first scheduling.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run the fixed multi-model evaluation matrix.")
    run.add_argument(
        "--group",
        choices=("all", "total", "env"),
        default="all",
        help="Select a default subset when --case is not provided. Default: all.",
    )
    run.add_argument(
        "--case",
        action="append",
        default=None,
        help="Run only the named case. Repeat to preserve an explicit case order.",
    )
    run.add_argument(
        "--legacy-cnncap-root",
        type=Path,
        default=None,
        help="Override the legacy CNNCap dataset root. Defaults to ../CapBench/reference/datasets/cnncap_small.",
    )
    run.add_argument("--num-gpus", type=int, default=DEFAULT_NUM_GPUS, help="Number of concurrent GPU workers.")
    run.add_argument(
        "--gpu-ids",
        type=str,
        default=None,
        help="Optional comma-separated physical GPU ids to use, for example 1,4. Overrides --num-gpus.",
    )
    run.add_argument(
        "--model",
        action="append",
        type=str,
        choices=DEFAULT_MODEL_KEYS,
        default=None,
        help="Run only the named model. Repeat to select a subset. Defaults to all four models.",
    )
    run.add_argument(
        "--monai-config",
        type=str,
        choices=("D4_B", "D4_D_k5"),
        default=None,
        help="Deprecated alias for selecting a single U-Net model.",
    )
    run.add_argument("--epochs", type=int, default=DEFAULT_EVAL_EPOCHS, help="Epochs per training run.")
    run.add_argument("--repeats", type=int, default=DEFAULT_EVAL_REPEATS, help="Number of seed repeats per case.")
    run.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS, help="DataLoader workers per process.")
    run.add_argument("--seed", type=int, default=DEFAULT_BASE_SEED, help="Base seed for repeated runs.")
    run.add_argument("--log-dir", type=Path, default=None, help="Override the evaluation log directory.")
    run.add_argument("--save-base", type=Path, default=None, help="Override the checkpoint output root.")
    run.add_argument("--tb-dir", type=Path, default=None, help="Override the TensorBoard output root.")

    list_parser = subparsers.add_parser("list", help="List the fixed evaluation cases.")
    list_parser.add_argument(
        "--group",
        choices=("all", "total", "env"),
        default="all",
        help="Filter the listed cases by goal.",
    )
    list_parser.add_argument(
        "--case",
        action="append",
        default=None,
        help="List only the named case(s).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "run":
            return _run(args)
        if args.command == "list":
            return _list(args)
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    parser.error("Unhandled command")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
