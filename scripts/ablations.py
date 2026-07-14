#!/usr/bin/env python3
"""Repo-local ablation launcher for Flash-CNNCap."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _ablations_lib import list_experiment_names, run_ablation_queue


def _load_capbench_dataset_resolver():
    try:
        from capbench.datasets import resolve_dataset_path
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError(
            "CapBench is not installed. Install it first from its checkout with "
            "`pip install -e \".[all]\"`."
        ) from exc
    return resolve_dataset_path


def _load_capbench_tech_root() -> Path:
    try:
        from capbench.paths import TECH_ROOT
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on local environment
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


def _resolve_dataset_path(dataset: str) -> Path:
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
                "Dataset path is missing required ablation artifacts: "
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


def _run_ablations(args: argparse.Namespace) -> int:
    dataset_path = _resolve_dataset_path(args.dataset)
    tech_path = args.tech.resolve() if args.tech is not None else _resolve_tech_path(dataset_path)

    return run_ablation_queue(
        dataset_path=dataset_path,
        tech_path=tech_path,
        num_gpus=args.num_gpus,
        epochs=args.epochs,
        repeats=args.repeats,
        num_workers=args.num_workers,
        seed=args.seed,
        experiment_names=args.experiment,
        log_dir=args.log_dir,
        save_base=args.save_base,
        tb_dir=args.tb_dir,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/ablations.py",
        description="Repo-local ablation launcher for Flash-CNNCap.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Resolve the dataset through CapBench and launch the ablation queue.")
    run.add_argument(
        "--dataset",
        default="nangate45/small",
        help="Registered CapBench dataset id or explicit dataset path.",
    )
    run.add_argument("--tech", type=Path, default=None, help="Optional override for the CapBench tech YAML.")
    run.add_argument(
        "--num-gpus",
        type=int,
        default=8,
        help="Number of concurrent GPU workers assigned contiguously from GPU 0 upward.",
    )
    run.add_argument("--epochs", type=int, default=100, help="Epochs per ablation run.")
    run.add_argument("--repeats", type=int, default=5, help="Number of seed repeats per experiment.")
    run.add_argument("--num-workers", type=int, default=1, help="DataLoader workers per training process.")
    run.add_argument(
        "--seed",
        type=int,
        default=11037,
        help="Base training seed; each ablation runs with seeds [seed, seed+1, ..., seed+repeats-1].",
    )
    run.add_argument(
        "--experiment",
        action="append",
        default=None,
        help="Run only the named experiment. Repeat the flag to run a subset.",
    )
    run.add_argument("--log-dir", type=Path, default=None, help="Override the ablation log directory.")
    run.add_argument("--save-base", type=Path, default=None, help="Override the checkpoint output root.")
    run.add_argument("--tb-dir", type=Path, default=None, help="Override the TensorBoard output root.")

    subparsers.add_parser("list", help="List the registered ablation experiment names.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "run":
            return _run_ablations(args)
        if args.command == "list":
            for name in list_experiment_names():
                print(name)
            return 0
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    parser.error("Unhandled command")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
