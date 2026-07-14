#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple


def _load_runtime(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_name(name_map: dict[str, str], token: str) -> str:
    return name_map.get(token, token)


def _parse_spef(path: Path) -> tuple[Dict[str, float], Dict[Tuple[str, str], float]]:
    name_map: dict[str, str] = {}
    totals: Dict[str, float] = {}
    couplings: Dict[Tuple[str, str], float] = {}
    mode = ""
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("*NAME_MAP"):
            mode = "name_map"
            continue
        if line.startswith("*D_NET"):
            parts = line.split()
            if len(parts) >= 3:
                totals[_resolve_name(name_map, parts[1])] = float(parts[2])
            mode = ""
            continue
        if line.startswith("*CAP"):
            mode = "cap"
            continue
        if line.startswith("*"):
            if mode == "name_map":
                parts = line.split(maxsplit=1)
                if len(parts) == 2 and parts[0][1:].isdigit():
                    name_map[parts[0]] = parts[1]
                    continue
            mode = ""
            continue
        if mode == "cap":
            parts = line.split()
            if len(parts) != 4:
                continue
            left = _resolve_name(name_map, parts[1])
            right = _resolve_name(name_map, parts[2])
            if left == right:
                continue
            key = tuple(sorted((left, right)))
            couplings[key] = couplings.get(key, 0.0) + float(parts[3])
    return totals, couplings


def _max_relative_diff(left: Dict, right: Dict) -> float:
    max_rel = 0.0
    for key in set(left) & set(right):
        denom = max(abs(float(left[key])), 1e-30)
        max_rel = max(max_rel, abs(float(right[key]) - float(left[key])) / denom)
    return max_rel


def _compare_spef(reference: Path, candidate: Path, *, max_rel_diff: float) -> dict:
    ref_totals, ref_couplings = _parse_spef(reference)
    cand_totals, cand_couplings = _parse_spef(candidate)
    total_missing = sorted(set(ref_totals) - set(cand_totals))
    total_extra = sorted(set(cand_totals) - set(ref_totals))
    coupling_missing = sorted(set(ref_couplings) - set(cand_couplings))
    coupling_extra = sorted(set(cand_couplings) - set(ref_couplings))
    total_max_rel = _max_relative_diff(ref_totals, cand_totals)
    coupling_max_rel = _max_relative_diff(ref_couplings, cand_couplings)
    result = {
        "total_count_reference": len(ref_totals),
        "total_count_candidate": len(cand_totals),
        "coupling_count_reference": len(ref_couplings),
        "coupling_count_candidate": len(cand_couplings),
        "total_missing": len(total_missing),
        "total_extra": len(total_extra),
        "coupling_missing": len(coupling_missing),
        "coupling_extra": len(coupling_extra),
        "total_max_relative_diff": total_max_rel,
        "coupling_max_relative_diff": coupling_max_rel,
    }
    if total_missing or total_extra or coupling_missing or coupling_extra:
        raise RuntimeError(f"SPEF key regression: {result}")
    if total_max_rel > max_rel_diff or coupling_max_rel > max_rel_diff:
        raise RuntimeError(f"SPEF value regression above {max_rel_diff}: {result}")
    return result


def _run_once(args: argparse.Namespace, *, run_index: int, measured: bool) -> dict:
    label = f"run{run_index:02d}" if measured else f"warmup{run_index:02d}"
    spef_path = args.out_dir / f"{label}.spef"
    runtime_path = args.out_dir / f"{label}.runtime.json"
    cmd = [
        str(args.python),
        str(args.runner),
        "run",
        "--def",
        str(args.def_path),
        "--out-spef",
        str(spef_path),
        "--tech",
        str(args.tech),
        "--compiled-model-manifest",
        str(args.compiled_model_manifest),
        "--total-compiled-model",
        str(args.total_compiled_model),
        "--env-compiled-model",
        str(args.env_compiled_model),
        "--runtime-json",
        str(runtime_path),
    ]
    if args.total_batch_size is not None:
        cmd.extend(["--total-batch-size", str(args.total_batch_size)])
    if args.master_batch_size is not None:
        cmd.extend(["--master-batch-size", str(args.master_batch_size)])
    if args.tile_stream_size is not None:
        cmd.extend(["--tile-stream-size", str(args.tile_stream_size)])
    subprocess.run(cmd, check=True)
    runtime = _load_runtime(runtime_path)
    runtime["spef_path"] = str(spef_path)
    runtime["runtime_json_path"] = str(runtime_path)
    runtime["measured"] = bool(measured)
    return runtime


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run repeated full-pipeline benchmarks and enforce optional runtime/SPEF regression gates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable), help="Python executable used to run full-pipeline/run.py.")
    parser.add_argument("--runner", type=Path, default=Path("full-pipeline/run.py"), help="Path to full-pipeline/run.py.")
    parser.add_argument("--def", dest="def_path", type=Path, required=True, help="Input full-layout DEF.")
    parser.add_argument("--tech", type=Path, required=True, help="Technology YAML.")
    parser.add_argument("--compiled-model-manifest", type=Path, required=True, help="Compiled model manifest.")
    parser.add_argument("--total-compiled-model", type=Path, required=True, help="Total model TensorRT engine.")
    parser.add_argument("--env-compiled-model", type=Path, required=True, help="Environment model TensorRT engine.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for SPEFs, per-run JSON, and summary JSON.")
    parser.add_argument("--warmups", type=int, default=1, help="Number of untimed warmup runs.")
    parser.add_argument("--runs", type=int, default=3, help="Number of measured runs.")
    parser.add_argument("--total-batch-size", type=int, default=None)
    parser.add_argument("--master-batch-size", type=int, default=None)
    parser.add_argument("--tile-stream-size", type=int, default=None)
    parser.add_argument("--baseline-runtime-json", type=Path, default=None, help="Optional baseline runtime JSON for median solve regression checks.")
    parser.add_argument("--max-solve-regression", type=float, default=0.03, help="Allowed median solve runtime regression fraction.")
    parser.add_argument("--baseline-spef", type=Path, default=None, help="Optional reference SPEF for output regression checks.")
    parser.add_argument("--max-spef-rel-diff", type=float, default=1e-5, help="Allowed maximum relative SPEF value diff.")
    args = parser.parse_args(argv)

    if args.warmups < 0 or args.runs <= 0:
        raise ValueError("--warmups must be non-negative and --runs must be positive.")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    for idx in range(args.warmups):
        runs.append(_run_once(args, run_index=idx + 1, measured=False))
    measured = []
    for idx in range(args.runs):
        runtime = _run_once(args, run_index=idx + 1, measured=True)
        measured.append(runtime)
        runs.append(runtime)

    solve_values = [float(item["solve_seconds"]) for item in measured]
    total_values = [float(item["total_reported_seconds"]) for item in measured]
    summary = {
        "runs": runs,
        "measured_count": len(measured),
        "median_solve_seconds": statistics.median(solve_values),
        "best_solve_seconds": min(solve_values),
        "median_total_reported_seconds": statistics.median(total_values),
        "best_total_reported_seconds": min(total_values),
    }

    if args.baseline_runtime_json is not None:
        baseline = _load_runtime(args.baseline_runtime_json)
        baseline_solve = float(baseline["solve_seconds"])
        allowed = baseline_solve * (1.0 + float(args.max_solve_regression))
        summary["baseline_solve_seconds"] = baseline_solve
        summary["allowed_solve_seconds"] = allowed
        if summary["median_solve_seconds"] > allowed:
            raise RuntimeError(
                f"Median solve runtime regressed: {summary['median_solve_seconds']:.6f}s > {allowed:.6f}s"
            )

    if args.baseline_spef is not None:
        summary["spef_compare"] = _compare_spef(
            args.baseline_spef,
            Path(measured[-1]["spef_path"]),
            max_rel_diff=float(args.max_spef_rel_diff),
        )

    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
