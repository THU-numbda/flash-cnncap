"""Repo-local copy of the legacy CNNCap dataset format."""

from __future__ import annotations

from collections import namedtuple
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


DEFAULT_LAYERS = "POLY1_MET1_MET2"
DEFAULT_PADDING = 12
_WINDOW_SIZE = 200
_TARGET_SCALE = 1.0e16

_EnvEntry = namedtuple("EnvEntry", ["layer_index", "conductor_id", "value"])
_GroupedCase = namedtuple("GroupedCase", ["x_offset", "y_offset", "grid_x", "grid_y", "master_id", "envs"])
_TotalCase = namedtuple("TotalCase", ["x_offset", "y_offset", "grid_x", "grid_y", "master_id", "value"])


def resolve_label_path(dataset_root: Path, layers: str, goal: str, split: str) -> Path:
    return dataset_root / "label" / f"{layers}_{goal}_{split}.txt"


def summarize_targets(label_lines: Sequence[str], goal: str) -> str:
    values: list[float] = []
    for raw_line in label_lines:
        fields = raw_line.split()
        if not fields:
            continue
        if goal == "env":
            values.append(abs(float(fields[8]) * (-_TARGET_SCALE)))
        else:
            values.append(abs(float(fields[5]) * _TARGET_SCALE))

    if not values:
        return "Loaded target magnitudes: unavailable (no samples)"

    value_array = np.asarray(values, dtype=np.float64)
    return (
        "Loaded target magnitudes: "
        f"min={value_array.min():.6g} "
        f"median={np.median(value_array):.6g} "
        f"max={value_array.max():.6g}"
    )


def collate_grouped(batch):
    xs, env_masks, targets = zip(*batch)
    max_queries = max(mask.shape[0] for mask in env_masks)
    batch_size = len(batch)
    num_layers = env_masks[0].shape[1]
    height = env_masks[0].shape[2]
    width = env_masks[0].shape[3]

    masks_padded = torch.zeros((batch_size, max_queries, num_layers, height, width), dtype=torch.float32)
    targets_padded = torch.zeros((batch_size, max_queries), dtype=torch.float32)
    valid = torch.zeros((batch_size, max_queries), dtype=torch.float32)

    for batch_idx, (masks, values) in enumerate(zip(env_masks, targets)):
        query_count = masks.shape[0]
        masks_padded[batch_idx, :query_count] = masks
        targets_padded[batch_idx, :query_count] = values
        valid[batch_idx, :query_count] = 1.0

    return torch.stack(xs), masks_padded, targets_padded, valid


def collate_total(batch):
    xs, masks, targets = zip(*batch)
    return torch.stack(xs), torch.stack(masks), torch.stack(targets).squeeze(1)


class CouplingDataset(Dataset):
    def __init__(
        self,
        layers: str,
        labels: Sequence[str],
        dataset_root: str | Path,
        indices: Iterable[int] | None = None,
        *,
        padding: int = DEFAULT_PADDING,
    ) -> None:
        self.layer_names = layers.split("_")
        self.layer_name_map = {layer_name: idx for idx, layer_name in enumerate(self.layer_names)}
        self.window_size = _WINDOW_SIZE
        self.padding = int(padding)
        self.master_layer = self.layer_name_map.get("MET1", min(1, len(self.layer_names) - 1))

        density_maps = []
        index_maps = []
        dataset_root = Path(dataset_root)
        for layer_name in self.layer_names:
            arrays = np.load(dataset_root / f"{layer_name}.npz")
            density_maps.append(arrays["img"])
            index_maps.append(arrays["idx"])

        padding_vec = [(0, 0), (self.window_size, self.window_size), (self.window_size, self.window_size)]
        self.density_maps = np.pad(np.asarray(density_maps), padding_vec, "constant")
        self.index_maps = np.pad(np.asarray(index_maps), padding_vec, "constant")
        self.density_maps.setflags(write=0)
        self.index_maps.setflags(write=0)

        resolved_indices = list(range(len(labels))) if indices is None else list(indices)
        grouped: dict[tuple[int, int, int, int, int], list[_EnvEntry]] = {}
        for label_idx in resolved_indices:
            fields = labels[label_idx].split()
            total_value = float(fields[7]) * _TARGET_SCALE
            env_value = float(fields[8]) * (-_TARGET_SCALE)
            x_offset = int(fields[0])
            y_offset = int(fields[1])
            master_id = int(fields[4])
            conductor_id = int(fields[5])
            layer_index = self.layer_name_map[fields[6]]

            if conductor_id == 0 or total_value == 0.0:
                continue
            if x_offset % 80 != 0 or y_offset % 80 != 0:
                continue

            env_value = max(env_value, 0.0)
            key = (x_offset, y_offset, int(fields[2]), int(fields[3]), master_id)
            grouped.setdefault(key, []).append(_EnvEntry(layer_index, conductor_id, env_value))

        self.cases = [
            _GroupedCase(x_offset, y_offset, grid_x, grid_y, master_id, envs)
            for (x_offset, y_offset, grid_x, grid_y, master_id), envs in grouped.items()
            if envs
        ]
        print(f"{len(self.cases)} legacy CNNCap env cases loaded.")

    def __len__(self) -> int:
        return len(self.cases)

    def _build_case_tensors(self, index: int):
        case = self.cases[index]
        x_origin = case.grid_x * 160 + 180 + case.x_offset
        y_origin = case.grid_y * 160 + 180 + case.y_offset
        index_view = self.index_maps[:, x_origin : x_origin + 200, y_origin : y_origin + 200]
        density_view = np.copy(self.density_maps[:, x_origin : x_origin + 200, y_origin : y_origin + 200])

        main_layer = np.copy(density_view[self.master_layer])
        main_layer[index_view[self.master_layer] == case.master_id] += 1.0
        density_view[self.master_layer, 20:180, 20:180] = main_layer[20:180, 20:180]
        density_view = np.pad(
            density_view,
            ((0, 0), (self.padding, self.padding), (self.padding, self.padding)),
            "constant",
        )

        num_layers = len(self.layer_name_map)
        env_masks = np.zeros((len(case.envs), num_layers, density_view.shape[1], density_view.shape[2]), dtype=np.float32)
        env_values = np.zeros((len(case.envs),), dtype=np.float32)
        for env_idx, env in enumerate(case.envs):
            mask = (index_view[env.layer_index] == env.conductor_id).astype(np.float32)
            mask = np.pad(mask, ((self.padding, self.padding), (self.padding, self.padding)), "constant")
            env_masks[env_idx, env.layer_index] = mask
            env_values[env_idx] = env.value

        master_mask = (index_view[self.master_layer] == case.master_id).astype(np.float32)
        master_mask = np.pad(master_mask, ((self.padding, self.padding), (self.padding, self.padding)), "constant")
        master_masks = np.zeros((num_layers, density_view.shape[1], density_view.shape[2]), dtype=np.float32)
        master_masks[self.master_layer] = master_mask
        other_masks = (index_view > 0).astype(np.float32)
        other_masks[self.master_layer][index_view[self.master_layer] == case.master_id] = 0.0
        other_masks = np.pad(
            other_masks,
            ((0, 0), (self.padding, self.padding), (self.padding, self.padding)),
            "constant",
        )
        return density_view, master_masks, other_masks, env_masks, env_values

    def get_visualization_case(self, index: int):
        return self._build_case_tensors(index)

    def __getitem__(self, index: int):
        density_view, _master_masks, _other_masks, env_masks, env_values = self._build_case_tensors(index)
        return (
            torch.tensor(density_view, dtype=torch.float32),
            torch.tensor(env_masks, dtype=torch.float32),
            torch.tensor(env_values, dtype=torch.float32),
        )


class ScalarCouplingDataset(CouplingDataset):
    def __init__(
        self,
        layers: str,
        labels: Sequence[str],
        dataset_root: str | Path,
        indices: Iterable[int] | None = None,
        *,
        padding: int = DEFAULT_PADDING,
    ) -> None:
        super().__init__(
            layers,
            labels,
            dataset_root,
            indices=indices,
            padding=padding,
        )
        self.scalar_cases = [
            (case_idx, env_idx)
            for case_idx, case in enumerate(self.cases)
            for env_idx, _env in enumerate(case.envs)
        ]
        print(f"{len(self)} legacy CNNCap scalar env samples loaded.")

    def __len__(self) -> int:
        return len(self.scalar_cases)

    def __getitem__(self, index: int):
        case_idx, env_idx = self.scalar_cases[index]
        density_view, _master_masks, _other_masks, env_masks, env_values = self._build_case_tensors(case_idx)
        features = np.copy(density_view)
        env_mask = env_masks[env_idx] > 0.0
        features[env_mask] = -features[env_mask]
        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor([float(env_values[env_idx])], dtype=torch.float32),
        )


class TotalDataset(Dataset):
    def __init__(
        self,
        layers: str,
        labels: Sequence[str],
        dataset_root: str | Path,
        indices: Iterable[int] | None = None,
        *,
        padding: int = DEFAULT_PADDING,
    ) -> None:
        self.layer_names = layers.split("_")
        self.layer_name_map = {layer_name: idx for idx, layer_name in enumerate(self.layer_names)}
        self.window_size = _WINDOW_SIZE
        self.padding = int(padding)
        self.master_layer = self.layer_name_map.get("MET1", min(1, len(self.layer_names) - 1))

        density_maps = []
        index_maps = []
        dataset_root = Path(dataset_root)
        for layer_name in self.layer_names:
            arrays = np.load(dataset_root / f"{layer_name}.npz")
            density_maps.append(arrays["img"])
            index_maps.append(arrays["idx"])

        padding_vec = [(0, 0), (self.window_size, self.window_size), (self.window_size, self.window_size)]
        self.density_maps = np.pad(np.asarray(density_maps), padding_vec, "constant")
        self.index_maps = np.pad(np.asarray(index_maps), padding_vec, "constant")
        self.density_maps.setflags(write=0)
        self.index_maps.setflags(write=0)

        resolved_indices = list(range(len(labels))) if indices is None else list(indices)
        self.cases = [
            _TotalCase(
                int(fields[0]),
                int(fields[1]),
                int(fields[2]),
                int(fields[3]),
                int(fields[4]),
                float(fields[5]) * _TARGET_SCALE,
            )
            for fields in (labels[label_idx].split() for label_idx in resolved_indices)
            if fields
        ]
        print(f"{len(self)} legacy CNNCap total cases loaded.")

    def __len__(self) -> int:
        return len(self.cases)

    def _build_case_tensors(self, index: int):
        case = self.cases[index]
        x_origin = case.grid_x * 160 + 180 + case.x_offset
        y_origin = case.grid_y * 160 + 180 + case.y_offset
        index_view = self.index_maps[:, x_origin : x_origin + 200, y_origin : y_origin + 200]
        density_view = np.copy(self.density_maps[:, x_origin : x_origin + 200, y_origin : y_origin + 200])

        main_layer = np.copy(density_view[self.master_layer])
        main_layer[index_view[self.master_layer] == case.master_id] += 1.0
        density_view[self.master_layer, 20:180, 20:180] = main_layer[20:180, 20:180]
        density_view = np.pad(
            density_view,
            ((0, 0), (self.padding, self.padding), (self.padding, self.padding)),
            "constant",
        )

        num_layers = len(self.layer_name_map)
        master_mask = (index_view[self.master_layer] == case.master_id).astype(np.float32)
        master_mask = np.pad(master_mask, ((self.padding, self.padding), (self.padding, self.padding)), "constant")
        master_masks = np.zeros((num_layers, density_view.shape[1], density_view.shape[2]), dtype=np.float32)
        master_masks[self.master_layer] = master_mask
        other_masks = (index_view > 0).astype(np.float32)
        other_masks[self.master_layer][index_view[self.master_layer] == case.master_id] = 0.0
        other_masks = np.pad(
            other_masks,
            ((0, 0), (self.padding, self.padding), (self.padding, self.padding)),
            "constant",
        )
        return density_view, master_masks, other_masks, case.value

    def get_visualization_case(self, index: int):
        return self._build_case_tensors(index)

    def __getitem__(self, index: int):
        density_view, master_masks, _other_masks, target = self._build_case_tensors(index)
        return (
            torch.tensor(density_view, dtype=torch.float32),
            torch.tensor(master_masks, dtype=torch.float32),
            torch.tensor([target], dtype=torch.float32),
        )


class ScalarTotalDataset(TotalDataset):
    def __getitem__(self, index: int):
        density_view, _master_masks, _other_masks, target = self._build_case_tensors(index)
        return (
            torch.tensor(density_view, dtype=torch.float32),
            torch.tensor([target], dtype=torch.float32),
        )
