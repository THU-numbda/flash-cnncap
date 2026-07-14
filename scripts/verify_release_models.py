#!/usr/bin/env python3
"""Verify Flash-CNNCap release hashes, metadata, and model compatibility."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import torch


FORBIDDEN_TEXT = ("/home/", "/data/", "saved_models")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pipeline_module(repo_root: Path):
    module_path = repo_root / "full-pipeline" / "pipeline_model.py"
    spec = importlib.util.spec_from_file_location("flash_cnncap_pipeline_model", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_metadata(metadata: dict[str, Any], artifact: str) -> None:
    required = {
        "goal",
        "monai_config",
        "num_input_channels",
        "active_layers",
        "release",
        "dataset",
        "target",
        "seed",
        "best_epoch",
        "paper_metrics",
        "source_checkpoint_sha256",
    }
    missing = sorted(required - metadata.keys())
    if missing:
        raise ValueError(f"{artifact}: missing metadata fields {missing}")
    if metadata["release"] != "v1.0.0" or metadata["monai_config"] != "D4_D_k5":
        raise ValueError(f"{artifact}: unexpected release or architecture metadata")
    expected_goal = "env" if metadata["target"] == "coupling" else "total"
    if metadata["goal"] != expected_goal:
        raise ValueError(f"{artifact}: expected goal={expected_goal!r}")
    if len(metadata["active_layers"]) != int(metadata["num_input_channels"]):
        raise ValueError(f"{artifact}: active layer count does not match input channels")
    encoded = json.dumps(metadata, sort_keys=True)
    if any(value in encoded for value in FORBIDDEN_TEXT):
        raise ValueError(f"{artifact}: metadata contains a private or training-only path")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models_dir", type=Path)
    parser.add_argument(
        "--load-weights",
        action="store_true",
        help="instantiate the paper U-Net with MONAI and strictly load every state dict",
    )
    args = parser.parse_args()

    models_dir = args.models_dir.expanduser().resolve()
    manifest = json.loads((models_dir / "release_manifest.json").read_text(encoding="utf-8"))
    artifacts = manifest.get("artifacts", [])
    if len(artifacts) != 14:
        raise ValueError(f"Expected 14 paper checkpoints, found {len(artifacts)}")

    pipeline = None
    if args.load_weights:
        pipeline = load_pipeline_module(Path(__file__).resolve().parents[1])

    for item in artifacts:
        artifact = str(item["artifact"])
        path = models_dir / artifact
        actual_hash = sha256(path)
        if actual_hash != item["sha256"]:
            raise ValueError(f"{artifact}: SHA-256 mismatch")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("format_version") != 1 or not isinstance(payload.get("state_dict"), dict):
            raise ValueError(f"{artifact}: unsupported checkpoint format")
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"{artifact}: missing metadata")
        check_metadata(metadata, artifact)

        if pipeline is not None:
            model = pipeline.build_full_pipeline_unet(
                num_input_channels=int(metadata["num_input_channels"]),
                monai_config=str(metadata["monai_config"]),
            )
            pipeline.load_model_checkpoint(path, model, torch.device("cpu"), strict=True)

        print(f"ok  {artifact}")

    print(f"Verified {len(artifacts)} Flash-CNNCap paper checkpoints.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
