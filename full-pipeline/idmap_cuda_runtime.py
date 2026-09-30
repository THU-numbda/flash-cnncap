from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from native_extension_utils import ensure_torch_cuda_arch_list


REPO_ROOT = Path(__file__).resolve().parents[1]
NATIVE_SOURCE_DIR = Path(__file__).resolve().parent / "native"
DEFAULT_TORCH_EXTENSION_DIR = REPO_ROOT / ".cache" / "torch_extensions"
DEFAULT_IDMAP_EXPAND_EXTENSION_NAME = "cnncap_flash_idmap_expand_cuda_v2"
DEFAULT_TARGET_SIZE = 224
INT16_MAX = np.iinfo(np.int16).max

RECT_COL_LAYER = 0
RECT_COL_CONDUCTOR_ID = 1
RECT_COL_PX_MIN = 2
RECT_COL_PX_MAX = 3
RECT_COL_PY_MIN = 4
RECT_COL_PY_MAX = 5
RECT_COL_COUNT = 6

_IDMAP_EXPAND_CUDA_MODULE = None
_IDMAP_EXPAND_CUDA_LOAD_ATTEMPTED = False


def load_idmap_expand_cuda_extension():
    global _IDMAP_EXPAND_CUDA_MODULE, _IDMAP_EXPAND_CUDA_LOAD_ATTEMPTED

    if _IDMAP_EXPAND_CUDA_MODULE is not None:
        return _IDMAP_EXPAND_CUDA_MODULE
    if _IDMAP_EXPAND_CUDA_LOAD_ATTEMPTED:
        raise RuntimeError("The CUDA ID-map expansion extension failed to initialize earlier in this process.")

    _IDMAP_EXPAND_CUDA_LOAD_ATTEMPTED = True

    if not torch.cuda.is_available():
        raise RuntimeError("The CUDA ID-map expansion extension requires CUDA.")

    ensure_torch_cuda_arch_list()

    try:
        from torch.utils.cpp_extension import load
    except ImportError as exc:
        raise RuntimeError("torch.utils.cpp_extension is required to build the CUDA ID-map expansion extension.") from exc

    source_paths = [
        NATIVE_SOURCE_DIR / "idmap_expand_bindings.cpp",
        NATIVE_SOURCE_DIR / "idmap_expand_cuda.cu",
    ]
    missing_sources = [str(path) for path in source_paths if not path.exists()]
    if missing_sources:
        raise RuntimeError("CUDA ID-map expansion sources are missing: " + ", ".join(missing_sources))

    build_dir = DEFAULT_TORCH_EXTENSION_DIR / DEFAULT_IDMAP_EXPAND_EXTENSION_NAME
    build_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(DEFAULT_TORCH_EXTENSION_DIR))

    _IDMAP_EXPAND_CUDA_MODULE = load(
        name=DEFAULT_IDMAP_EXPAND_EXTENSION_NAME,
        sources=[str(path) for path in source_paths],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        build_directory=str(build_dir),
        verbose=True,
    )
    return _IDMAP_EXPAND_CUDA_MODULE


def rasterize_binary_masks_cpu(
    packed_rects: np.ndarray,
    *,
    num_layers: int,
    target_size: int,
    real_conductor_count: int,
) -> Tuple[np.ndarray, np.ndarray]:
    occupied = np.zeros((num_layers, target_size, target_size), dtype=np.uint8)
    master_masks = np.zeros((real_conductor_count, num_layers, target_size, target_size), dtype=np.uint8)
    packed_rects = np.asarray(packed_rects, dtype=np.int32)
    for idx in range(int(packed_rects.shape[0])):
        layer, cid, x0, x1, y0, y1 = packed_rects[idx]
        if layer < 0 or cid <= 0 or x0 >= x1 or y0 >= y1 or layer >= num_layers:
            continue
        occupied[layer, y0:y1, x0:x1] = np.uint8(1)
        if cid <= real_conductor_count:
            master_masks[cid - 1, layer, y0:y1, x0:x1] = np.uint8(1)
    return occupied, master_masks


def expand_fast_idmaps_cpu(prepared) -> np.ndarray:
    layers = len(prepared.channel_layers)
    size = int(prepared.target_size)
    out = np.zeros((layers, size, size), dtype=np.int16)
    if prepared.active_blocks == 0:
        return out

    for idx in range(prepared.active_blocks):
        layer, cid, x0, x1, y0, y1 = prepared.packed_rects[idx]
        if layer < 0 or cid <= 0 or x0 >= x1 or y0 >= y1:
            continue
        out[layer, y0:y1, x0:x1] = np.int16(cid)
    return out


def rasterize_binary_masks_cuda(
    packed_rects_cuda: torch.Tensor,
    *,
    num_layers: int,
    target_size: int,
    real_conductor_count: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    module = load_idmap_expand_cuda_extension()
    occupied, master_masks = module.rasterize_binary_masks(
        packed_rects_cuda,
        int(num_layers),
        int(target_size),
        int(target_size),
        int(real_conductor_count),
    )
    return occupied, master_masks


def rasterize_packed_rects_with_sparse_cuda(
    packed_rects_cuda: torch.Tensor,
    *,
    num_layers: int,
    target_size: int,
    real_conductor_count: int,
    own_x0: int,
    own_x1: int,
    own_y0: int,
    own_y1: int,
):
    module = load_idmap_expand_cuda_extension()
    return module.rasterize_idmaps_with_sparse(
        packed_rects_cuda,
        int(num_layers),
        int(target_size),
        int(target_size),
        int(real_conductor_count),
        int(own_x0),
        int(own_x1),
        int(own_y0),
        int(own_y1),
    )


def expand_fast_idmaps_cuda(prepared, device: torch.device) -> torch.Tensor:
    if device.type != "cuda":
        raise ValueError(f"expand_fast_idmaps_cuda requires a CUDA device, got: {device}")
    if prepared.active_blocks == 0:
        return torch.zeros((len(prepared.channel_layers), prepared.target_size, prepared.target_size), dtype=torch.int16, device=device)

    packed_rects_cuda = torch.from_numpy(prepared.packed_rects).to(device=device, dtype=torch.int32, non_blocking=True).contiguous()
    module = load_idmap_expand_cuda_extension()
    return module.rasterize_idmaps(
        packed_rects_cuda,
        int(len(prepared.channel_layers)),
        int(prepared.target_size),
        int(prepared.target_size),
    )
