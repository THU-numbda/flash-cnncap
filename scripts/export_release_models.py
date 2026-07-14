#!/usr/bin/env python3
"""Export optimizer-free, path-sanitized Flash-CNNCap release checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


SAFE_SOURCE_METADATA = (
    "goal",
    "model_type",
    "monai_config",
    "monai_arch",
    "num_input_channels",
    "active_layers",
    "dataset_format",
    "labels_solver",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_spec(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError(f"Unsupported model specification: {path}")
    if not isinstance(payload.get("models"), list) or not payload["models"]:
        raise ValueError(f"Model specification contains no models: {path}")
    return payload


def export_model(source_root: Path, out_dir: Path, entry: dict[str, Any], seed: int) -> dict[str, Any]:
    source = (source_root / str(entry["source_relative"])).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)

    payload = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
        raise ValueError(f"Checkpoint has no state_dict: {source}")
    source_metadata = payload.get("metadata", {})
    if not isinstance(source_metadata, dict):
        raise ValueError(f"Checkpoint metadata is not a dictionary: {source}")
    expected_goal = "env" if entry["target"] == "coupling" else "total"
    if source_metadata.get("goal") != expected_goal:
        raise ValueError(
            f"Goal mismatch for {source}: expected {expected_goal}, found {source_metadata.get('goal')}"
        )
    if source_metadata.get("monai_config") != "D4_D_k5":
        raise ValueError(f"Unexpected model configuration in {source}")

    metrics = {key: float(value) for key, value in dict(entry["metrics"]).items()}
    checkpoint_loss = float(payload.get("loss", metrics["mare"]))
    if abs(checkpoint_loss - metrics["mare"]) > 5e-7:
        raise ValueError(
            f"Paper MARE {metrics['mare']:.6f} does not match checkpoint loss {checkpoint_loss:.9f}: {source}"
        )

    metadata = {
        key: source_metadata[key]
        for key in SAFE_SOURCE_METADATA
        if key in source_metadata
    }
    metadata.update(
        {
            "release": "v1.0.0",
            "dataset": str(entry["dataset"]),
            "target": str(entry["target"]),
            "input_encoding": "binary occupancy; coupling master encoded as -1",
            "seed": int(seed),
            "best_epoch": int(payload.get("epoch", -1)),
            "paper_metrics": metrics,
            "source_checkpoint_sha256": sha256(source),
        }
    )
    released = {
        "format_version": 1,
        "state_dict": payload["state_dict"],
        "metadata": metadata,
    }

    destination = out_dir / str(entry["artifact"])
    torch.save(released, destination)
    return {
        "artifact": destination.name,
        "bytes": destination.stat().st_size,
        "sha256": sha256(destination),
        "source_checkpoint": str(entry["source_relative"]),
        "source_sha256": metadata["source_checkpoint_sha256"],
        "dataset": metadata["dataset"],
        "target": metadata["target"],
        "best_epoch": metadata["best_epoch"],
        "metrics": metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    source_root = args.source_root.expanduser().resolve()
    spec = load_spec(args.spec.expanduser().resolve())
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    exported = [
        export_model(source_root, out_dir, entry, int(spec["seed"]))
        for entry in spec["models"]
    ]
    manifest = {
        "schema_version": 1,
        "release": "v1.0.0",
        "architecture": spec["architecture"],
        "seed": int(spec["seed"]),
        "artifacts": exported,
    }
    manifest_path = out_dir / "release_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    sums_path = out_dir / "SHA256SUMS"
    sums_path.write_text(
        "".join(f"{item['sha256']}  {item['artifact']}\n" for item in exported),
        encoding="utf-8",
    )
    print(f"Exported {len(exported)} models to {out_dir}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
