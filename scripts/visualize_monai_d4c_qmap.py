#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors
except ImportError as exc:  # pragma: no cover - explicit runtime guard
    raise RuntimeError("matplotlib is required for visualization.") from exc


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
TRAINING_DIR = REPO_ROOT / "training"

for path in (REPO_ROOT, TRAINING_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from capbench._internal.common.datasets import get_dataset_subdirs  # pylint: disable=wrong-import-position
from capbench.window_id_map_dataset import IdMapWindowDataset  # pylint: disable=wrong-import-position
from train import (  # pylint: disable=wrong-import-position
    CapBenchCouplingDataset,
    CapBenchTotalDataset,
    MONAI_ABLATION_CONFIGS,
    get_model,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize MONAI D4_C q-map outputs for CapBench total/env cases.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint.")
    parser.add_argument("--dataset-path", type=Path, required=True, help="CapBench dataset root path.")
    parser.add_argument("--goal", choices=("total", "env"), required=True, help="Visualization goal.")
    parser.add_argument("--sample-index", type=int, default=0, help="Sample index to visualize.")
    parser.add_argument("--seed", type=int, default=11037, help="Seed for deterministic env extra-master selection.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="Execution device.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/monai_d4c_viz"),
        help="Base output directory for rendered figures.",
    )
    parser.add_argument("--dpi", type=int, default=200, help="Figure DPI.")
    parser.add_argument(
        "--qmap-scale",
        choices=("linear", "log"),
        default="log",
        help="Color scaling for q-map visualization.",
    )
    return parser.parse_args()


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(name)


def _choose_spef_dir(dataset_dirs: Dict[str, Path]) -> Path:
    rwcap = dataset_dirs["labels_rwcap"]
    raphael = dataset_dirs["labels_raphael"]
    if rwcap.exists():
        return rwcap
    if raphael.exists():
        return raphael
    raise FileNotFoundError(
        f"Could not find labels directory. Expected one of: {rwcap} or {raphael}"
    )


def _extract_state_dict(payload: object) -> Dict[str, torch.Tensor]:
    if not isinstance(payload, dict):
        raise RuntimeError("Unsupported checkpoint format: expected dict-like object.")

    for key in ("state_dict", "model_state_dict", "model"):
        maybe = payload.get(key)
        if isinstance(maybe, dict) and maybe and all(isinstance(k, str) for k in maybe.keys()):
            if any(torch.is_tensor(v) for v in maybe.values()):
                return maybe  # type: ignore[return-value]

    if payload and all(isinstance(k, str) for k in payload.keys()) and any(torch.is_tensor(v) for v in payload.values()):
        return payload  # type: ignore[return-value]

    raise RuntimeError("Could not locate model state_dict in checkpoint.")


def _clean_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key.replace("module.", ""): value for key, value in state_dict.items()}


def _remap_legacy_monai_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Remap older MONAI UNet key patterns to newer module naming.

    Known migrations handled:
    - .sub0, .sub1, ... -> .submodule.0, .submodule.1, ...
    - .subconv -> .submodule.conv
    - .subresidual -> .submodule.residual
    """
    remapped: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = re.sub(r"\.sub(\d+)", lambda match: f".submodule.{match.group(1)}", key)
        new_key = new_key.replace(".subconv.", ".submodule.conv.")
        new_key = new_key.replace(".subresidual.", ".submodule.residual.")
        remapped[new_key] = value
    return remapped


def _build_d4c_model(num_input_channels: int, device: torch.device) -> torch.nn.Module:
    cfg = MONAI_ABLATION_CONFIGS["D4_C"]
    model = get_model(
        "unet",
        num_input_channels=num_input_channels,
        base_ch=int(cfg["base_ch"]),
        depth=int(cfg["depth"]),
        monai_channels=tuple(int(v) for v in cfg["channels"]),
        monai_strides=tuple(int(v) for v in cfg["strides"]),
        monai_num_res_units=int(cfg["num_res_units"]),
    )
    model.to(device)
    model.eval()
    return model


def _load_checkpoint_into_model(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = _clean_state_dict(_extract_state_dict(payload))

    try:
        model.load_state_dict(state_dict, strict=True)
        return
    except RuntimeError as exc:
        strict_err = str(exc)

    remapped = _remap_legacy_monai_keys(state_dict)
    try:
        model.load_state_dict(remapped, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "Failed to load checkpoint after MONAI legacy-key remap.\n"
            f"Original strict-load error:\n{strict_err}\n\n"
            f"Remapped strict-load error:\n{exc}"
        ) from exc


def _run_qmap(model: torch.nn.Module, features: np.ndarray, device: torch.device) -> np.ndarray:
    x = torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        raw = model(x)
        if isinstance(raw, (tuple, list)):
            raw = raw[0]
        q_map = F.softplus(raw)[0]
    return q_map.detach().cpu().numpy()


def _sanitize_filename(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", text)


def _occupied_masks_for_window(base_dataset: IdMapWindowDataset, window_idx: int) -> np.ndarray:
    local_map, _local_counts, _actual_to_local = base_dataset._build_window_local_state(window_idx)  # pylint: disable=protected-access
    return (local_map > 0).astype(np.float32, copy=False)


def _boundary_mask(mask: np.ndarray) -> np.ndarray:
    m = (mask > 0)
    if m.size == 0:
        return m

    up = np.zeros_like(m, dtype=bool)
    up[1:, :] = m[:-1, :]
    down = np.zeros_like(m, dtype=bool)
    down[:-1, :] = m[1:, :]
    left = np.zeros_like(m, dtype=bool)
    left[:, 1:] = m[:, :-1]
    right = np.zeros_like(m, dtype=bool)
    right[:, :-1] = m[:, 1:]

    interior = m & up & down & left & right
    return m & (~interior)


def _dilate8(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(bool, copy=False)
    out = m.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            shifted = np.zeros_like(m, dtype=bool)
            y_src_start = max(0, -dy)
            y_src_end = m.shape[0] - max(0, dy)
            x_src_start = max(0, -dx)
            x_src_end = m.shape[1] - max(0, dx)
            y_dst_start = max(0, dy)
            y_dst_end = y_dst_start + (y_src_end - y_src_start)
            x_dst_start = max(0, dx)
            x_dst_end = x_dst_start + (x_src_end - x_src_start)
            shifted[y_dst_start:y_dst_end, x_dst_start:x_dst_end] = m[y_src_start:y_src_end, x_src_start:x_src_end]
            out |= shifted
    return out


def _overlay_black_white_outline(ax, mask: np.ndarray) -> None:
    boundary = _boundary_mask(mask)
    if not np.any(boundary):
        return

    halo = _dilate8(boundary)
    h, w = boundary.shape

    black = np.zeros((h, w, 4), dtype=np.float32)
    black[..., 3] = halo.astype(np.float32) * 0.95
    ax.imshow(black, origin="lower", interpolation="nearest")

    white = np.ones((h, w, 4), dtype=np.float32)
    white[..., 3] = boundary.astype(np.float32) * 0.95
    ax.imshow(white, origin="lower", interpolation="nearest")


def _draw_density_layer(
    ax,
    layer_density: np.ndarray,
    occupied_mask: np.ndarray,
    highlight_mask: np.ndarray,
    title: str,
) -> None:
    _ = layer_density  # Input style is intentionally binary: conductors over white background.

    h, w = occupied_mask.shape
    base = np.ones((h, w, 3), dtype=np.float32)  # white background
    occ = occupied_mask > 0
    base[occ] = 0.0  # black conductors
    ax.imshow(base, origin="lower", interpolation="nearest")

    if np.any(highlight_mask > 0):
        alpha = 0.62 * (highlight_mask > 0).astype(np.float32)
        overlay = np.zeros((highlight_mask.shape[0], highlight_mask.shape[1], 4), dtype=np.float32)
        overlay[..., 0] = 1.0
        overlay[..., 3] = alpha
        ax.imshow(overlay, origin="lower", interpolation="nearest")

    ax.set_title(title)
    ax.set_axis_off()


def _save_visualizations(
    *,
    features: np.ndarray,
    q_map: np.ndarray,
    occupied_masks: np.ndarray,
    highlight_masks: np.ndarray,
    layer_labels: Sequence[str],
    output_dir: Path,
    title_prefix: str,
    dpi: int,
    qmap_scale: str,
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    num_layers = int(features.shape[0])
    masked_q_map = np.where(occupied_masks > 0, q_map, np.nan)
    finite_vals = masked_q_map[np.isfinite(masked_q_map)]
    positive_vals = finite_vals[finite_vals > 0]

    norm = None
    vmin = None
    vmax = None
    if qmap_scale == "log" and positive_vals.size > 0:
        log_vmin = max(float(np.min(positive_vals)), 1e-12)
        log_vmax = max(float(np.max(positive_vals)), log_vmin * 1.000001)
        norm = mcolors.LogNorm(vmin=log_vmin, vmax=log_vmax)
    else:
        if finite_vals.size > 0:
            lin_vmin = float(np.min(finite_vals))
            lin_vmax = float(np.max(finite_vals))
        else:
            lin_vmin = float(np.min(q_map))
            lin_vmax = float(np.max(q_map))
        if abs(lin_vmax - lin_vmin) < 1e-12:
            lin_vmax = lin_vmin + 1e-12
        vmin = lin_vmin
        vmax = lin_vmax

    q_cmap = plt.cm.get_cmap("viridis").copy()
    q_cmap.set_bad(color="white")

    saved: List[Path] = []

    for layer_idx in range(num_layers):
        layer_name = layer_labels[layer_idx] if layer_idx < len(layer_labels) else f"L{layer_idx}"
        safe_layer = _sanitize_filename(layer_name)

        input_path = output_dir / f"input_layer_{layer_idx:02d}_{safe_layer}.png"
        fig_in, ax_in = plt.subplots(1, 1, figsize=(5.0, 5.0))
        _draw_density_layer(
            ax_in,
            features[layer_idx],
            occupied_masks[layer_idx],
            highlight_masks[layer_idx],
            f"{title_prefix} | Input {layer_name}",
        )
        fig_in.tight_layout()
        fig_in.savefig(input_path, dpi=dpi)
        plt.close(fig_in)
        saved.append(input_path)

        qmap_path = output_dir / f"qmap_layer_{layer_idx:02d}_{safe_layer}.png"
        fig_q, ax_q = plt.subplots(1, 1, figsize=(5.0, 5.0))
        img = ax_q.imshow(
            masked_q_map[layer_idx],
            cmap=q_cmap,
            origin="lower",
            interpolation="nearest",
            norm=norm,
            vmin=vmin,
            vmax=vmax,
        )
        ax_q.set_title(f"{title_prefix} | Q-map {layer_name} ({qmap_scale})")
        ax_q.set_axis_off()
        fig_q.colorbar(img, ax=ax_q, fraction=0.046, pad=0.04)
        fig_q.tight_layout()
        fig_q.savefig(qmap_path, dpi=dpi)
        plt.close(fig_q)
        saved.append(qmap_path)

    # One large figure containing all input/output layers.
    fig_all, axes = plt.subplots(num_layers, 2, figsize=(12.0, max(4.0, 3.0 * num_layers)), squeeze=False)
    last_img = None
    for layer_idx in range(num_layers):
        layer_name = layer_labels[layer_idx] if layer_idx < len(layer_labels) else f"L{layer_idx}"
        _draw_density_layer(
            axes[layer_idx, 0],
            features[layer_idx],
            occupied_masks[layer_idx],
            highlight_masks[layer_idx],
            f"Input {layer_name}",
        )
        last_img = axes[layer_idx, 1].imshow(
            masked_q_map[layer_idx],
            cmap=q_cmap,
            origin="lower",
            interpolation="nearest",
            norm=norm,
            vmin=vmin,
            vmax=vmax,
        )
        axes[layer_idx, 1].set_title(f"Q-map {layer_name} ({qmap_scale})")
        axes[layer_idx, 1].set_axis_off()
    if last_img is not None:
        fig_all.colorbar(last_img, ax=axes[:, 1].ravel().tolist(), fraction=0.02, pad=0.01)
    fig_all.suptitle(f"{title_prefix} | All Layers", fontsize=12)
    fig_all.tight_layout(rect=(0, 0, 1, 0.97))
    all_path = output_dir / "all_layers.png"
    fig_all.savefig(all_path, dpi=dpi)
    plt.close(fig_all)
    saved.append(all_path)

    return saved


def _resolve_layer_labels(base_dataset: IdMapWindowDataset) -> List[str]:
    return list(base_dataset.active_layers)


def _render_total_case(
    *,
    model: torch.nn.Module,
    total_dataset: CapBenchTotalDataset,
    base_dataset: IdMapWindowDataset,
    sample_index: int,
    device: torch.device,
    out_dir: Path,
    layer_labels: Sequence[str],
    dpi: int,
    qmap_scale: str,
) -> Dict[str, object]:
    if sample_index < 0 or sample_index >= len(total_dataset):
        raise IndexError(f"sample-index={sample_index} out of range [0, {len(total_dataset) - 1}] for total goal.")

    window_idx = sample_index
    features, _conductor_masks, master_masks, _other_masks, _targets = total_dataset.get_visualization_case(sample_index)
    q_map = _run_qmap(model, features, device)
    occupied_masks = _occupied_masks_for_window(base_dataset, window_idx)
    no_highlight = np.zeros_like(master_masks, dtype=np.float32)

    output_dir = out_dir / "total" / f"sample_{sample_index:05d}" / "total_view"
    saved_paths = _save_visualizations(
        features=features,
        q_map=q_map,
        occupied_masks=occupied_masks,
        highlight_masks=no_highlight,
        layer_labels=layer_labels,
        output_dir=output_dir,
        title_prefix=f"total | sample={sample_index}",
        dpi=dpi,
        qmap_scale=qmap_scale,
    )

    return {
        "window_idx": int(window_idx),
        "selected_cid": None,
        "output_dir": str(output_dir),
        "saved_files": [str(path) for path in saved_paths],
    }


def _build_env_master_view(
    coupling_dataset: CapBenchCouplingDataset,
    window_idx: int,
    master_id: int,
) -> Tuple[np.ndarray, np.ndarray]:
    features = coupling_dataset._build_raw_features(window_idx)  # pylint: disable=protected-access
    _window, cached = coupling_dataset.base._get_window_and_cache(window_idx)  # pylint: disable=protected-access
    coupling_dataset.base._apply_highlight(features, cached, master_id, positive=False)  # pylint: disable=protected-access
    highlight_masks = coupling_dataset._build_conductor_masks(window_idx, [master_id])[0]  # pylint: disable=protected-access
    return features, highlight_masks.astype(np.float32)


def _sample_extra_master_ids(
    *,
    coupling_dataset: CapBenchCouplingDataset,
    window_idx: int,
    selected_master_id: int,
    seed: int,
) -> List[int]:
    window_ranges = coupling_dataset.get_window_sample_ranges()
    if window_idx < 0 or window_idx >= len(window_ranges):
        raise IndexError(f"window_idx={window_idx} out of range [0, {len(window_ranges) - 1}]")
    start, end = window_ranges[window_idx]
    master_pool = sorted(
        {
            int(coupling_dataset.base.get_grouped_coupling_case(case_idx)[1])
            for case_idx in range(start, end)
        }
    )
    candidates = [cid for cid in master_pool if cid != selected_master_id]
    rng = random.Random(seed)
    if not candidates:
        return []
    count = min(3, len(candidates))
    return list(rng.sample(candidates, k=count))


def _render_env_master_case(
    *,
    model: torch.nn.Module,
    coupling_dataset: CapBenchCouplingDataset,
    base_dataset: IdMapWindowDataset,
    window_idx: int,
    master_id: int,
    case_label: str,
    sample_index: int,
    device: torch.device,
    out_dir: Path,
    layer_labels: Sequence[str],
    dpi: int,
    qmap_scale: str,
) -> Dict[str, object]:
    features, highlight_masks = _build_env_master_view(coupling_dataset, window_idx, master_id)
    q_map = _run_qmap(model, features, device)
    occupied_masks = _occupied_masks_for_window(base_dataset, window_idx)

    output_dir = out_dir / "env" / f"sample_{sample_index:05d}" / f"{case_label}_master_{master_id}"
    saved_paths = _save_visualizations(
        features=features,
        q_map=q_map,
        occupied_masks=occupied_masks,
        highlight_masks=highlight_masks,
        layer_labels=layer_labels,
        output_dir=output_dir,
        title_prefix=f"env | sample={sample_index} | {case_label} | master={master_id}",
        dpi=dpi,
        qmap_scale=qmap_scale,
    )

    return {
        "window_idx": int(window_idx),
        "master_id": int(master_id),
        "case_label": case_label,
        "output_dir": str(output_dir),
        "saved_files": [str(path) for path in saved_paths],
    }


def _render_env_cases(
    *,
    model: torch.nn.Module,
    coupling_dataset: CapBenchCouplingDataset,
    base_dataset: IdMapWindowDataset,
    sample_index: int,
    seed: int,
    device: torch.device,
    out_dir: Path,
    layer_labels: Sequence[str],
    dpi: int,
    qmap_scale: str,
) -> Dict[str, object]:
    if sample_index < 0 or sample_index >= len(coupling_dataset):
        raise IndexError(f"sample-index={sample_index} out of range [0, {len(coupling_dataset) - 1}] for env goal.")

    window_idx, selected_master_id, _slave_ids, _targets, _valid = coupling_dataset.base.get_grouped_coupling_case(
        sample_index
    )
    window_idx = int(window_idx)
    selected_master_id = int(selected_master_id)

    selected_info = _render_env_master_case(
        model=model,
        coupling_dataset=coupling_dataset,
        base_dataset=base_dataset,
        window_idx=window_idx,
        master_id=selected_master_id,
        case_label="selected",
        sample_index=sample_index,
        device=device,
        out_dir=out_dir,
        layer_labels=layer_labels,
        dpi=dpi,
        qmap_scale=qmap_scale,
    )

    extra_master_ids = _sample_extra_master_ids(
        coupling_dataset=coupling_dataset,
        window_idx=window_idx,
        selected_master_id=selected_master_id,
        seed=seed,
    )

    extras: List[Dict[str, object]] = []
    for idx, master_id in enumerate(extra_master_ids, start=1):
        extras.append(
            _render_env_master_case(
                model=model,
                coupling_dataset=coupling_dataset,
                base_dataset=base_dataset,
                window_idx=window_idx,
                master_id=int(master_id),
                case_label=f"extra_{idx}",
                sample_index=sample_index,
                device=device,
                out_dir=out_dir,
                layer_labels=layer_labels,
                dpi=dpi,
                qmap_scale=qmap_scale,
            )
        )

    return {
        "window_idx": window_idx,
        "selected_master_id": selected_master_id,
        "extra_master_ids": [int(cid) for cid in extra_master_ids],
        "selected": selected_info,
        "extras": extras,
    }


def main() -> int:
    args = _parse_args()
    device = _resolve_device(args.device)

    dataset_path = args.dataset_path.resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    dataset_dirs = get_dataset_subdirs(dataset_path)
    window_dir = dataset_dirs["density_maps"].resolve()
    if not window_dir.exists():
        raise FileNotFoundError(f"density_maps directory not found: {window_dir}")
    spef_dir = _choose_spef_dir(dataset_dirs).resolve()

    dataset_goal = "self" if args.goal == "total" else "coupling"
    base_dataset = IdMapWindowDataset(
        window_dir=window_dir,
        spef_dir=spef_dir,
        goal=dataset_goal,
        trim_margin=True,
    )

    num_input_channels = base_dataset.num_layers
    model = _build_d4c_model(num_input_channels=num_input_channels, device=device)
    _load_checkpoint_into_model(model, args.checkpoint.resolve(), device)

    layer_labels = _resolve_layer_labels(base_dataset)
    out_dir = args.out_dir.resolve()

    if args.goal == "total":
        wrapped = CapBenchTotalDataset(base_dataset)
        results = _render_total_case(
            model=model,
            total_dataset=wrapped,
            base_dataset=base_dataset,
            sample_index=int(args.sample_index),
            device=device,
            out_dir=out_dir,
            layer_labels=layer_labels,
            dpi=int(args.dpi),
            qmap_scale=str(args.qmap_scale),
        )
    else:
        wrapped = CapBenchCouplingDataset(base_dataset)
        results = _render_env_cases(
            model=model,
            coupling_dataset=wrapped,
            base_dataset=base_dataset,
            sample_index=int(args.sample_index),
            seed=int(args.seed),
            device=device,
            out_dir=out_dir,
            layer_labels=layer_labels,
            dpi=int(args.dpi),
            qmap_scale=str(args.qmap_scale),
        )

    metadata = {
        "checkpoint": str(args.checkpoint.resolve()),
        "dataset_path": str(dataset_path),
        "goal": args.goal,
        "sample_index": int(args.sample_index),
        "seed": int(args.seed),
        "device": str(device),
        "qmap_scale": str(args.qmap_scale),
        "d4c_config": {
            "channels": list(MONAI_ABLATION_CONFIGS["D4_C"]["channels"]),
            "strides": list(MONAI_ABLATION_CONFIGS["D4_C"]["strides"]),
            "num_res_units": int(MONAI_ABLATION_CONFIGS["D4_C"]["num_res_units"]),
        },
        "layer_labels": list(layer_labels),
        "results": results,
    }

    root = out_dir / args.goal / f"sample_{int(args.sample_index):05d}"
    root.mkdir(parents=True, exist_ok=True)
    metadata_path = root / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved visualization metadata: {metadata_path}")
    print(f"Goal: {args.goal} | Sample: {args.sample_index} | Layers: {len(layer_labels)} | Device: {device}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
