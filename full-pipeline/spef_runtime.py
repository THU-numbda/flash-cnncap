from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np


_PACKED_INDEX_BITS = 32
_PACKED_INDEX_MASK = (1 << _PACKED_INDEX_BITS) - 1


@dataclass(frozen=True)
class WindowPredictionResult:
    window_id: str
    total_cap_f: Dict[str, float]
    coupling_cap_f: Dict[Tuple[str, str], float]


class IndexedSpefAccumulator:
    def __init__(self, net_names: Sequence[str] = ()) -> None:
        self._net_names: list[str] = []
        self._name_to_index: Dict[str, int] = {}
        self._total_cap_f = np.zeros((0,), dtype=np.float64)
        self._directed_coupling_f: Dict[int, float] = {}
        self._directed_std_rel: Dict[int, float] = {}
        if net_names:
            self.register_nets(net_names)

    @classmethod
    def from_net_names(cls, net_names: Sequence[str]) -> IndexedSpefAccumulator:
        return cls(net_names)

    @property
    def net_names(self) -> Tuple[str, ...]:
        return tuple(self._net_names)

    @property
    def net_count(self) -> int:
        return len(self._net_names)

    def _ensure_total_capacity(self, required: int) -> None:
        if required <= int(self._total_cap_f.shape[0]):
            return
        new_capacity = max(int(required), max(16, int(self._total_cap_f.shape[0]) * 2))
        grown = np.zeros((new_capacity,), dtype=np.float64)
        if self.net_count > 0:
            grown[: self.net_count] = self._total_cap_f[: self.net_count]
        self._total_cap_f = grown

    def register_nets(self, net_names: Sequence[str]) -> Tuple[int, ...]:
        indices: list[int] = []
        for raw_name in net_names:
            name = str(raw_name)
            index = self._name_to_index.get(name)
            if index is None:
                index = self.net_count
                self._ensure_total_capacity(index + 1)
                self._name_to_index[name] = index
                self._net_names.append(name)
            indices.append(index)
        return tuple(indices)

    def add_total_values(
        self,
        net_indices: Sequence[int],
        values_f: Sequence[float],
        *,
        scale: float = 1.0,
    ) -> None:
        index_array = np.asarray(net_indices, dtype=np.int64)
        value_array = np.asarray(values_f, dtype=np.float64)
        if index_array.ndim != 1 or value_array.ndim != 1:
            raise ValueError("net_indices and values_f must be rank-1 sequences.")
        if index_array.shape[0] != value_array.shape[0]:
            raise ValueError(
                f"net_indices and values_f must have the same length, got {index_array.shape[0]} vs {value_array.shape[0]}"
            )
        if index_array.size == 0:
            return
        if int(index_array.min()) < 0 or int(index_array.max()) >= self.net_count:
            raise ValueError("net_indices contain values outside the registered net range.")
        np.add.at(self._total_cap_f, index_array, value_array * float(scale))

    def add_directed_row(
        self,
        master_index: int,
        victim_indices: Sequence[int],
        values_f: Sequence[float],
        *,
        scale: float = 1.0,
        std_rel: float | Sequence[float] = 1.0,
    ) -> None:
        if master_index < 0 or master_index >= self.net_count:
            raise ValueError(f"master_index must be in [0, {self.net_count - 1}], got {master_index}")
        victim_array = np.asarray(victim_indices, dtype=np.int64)
        value_array = np.asarray(values_f, dtype=np.float64)
        if victim_array.ndim != 1 or value_array.ndim != 1:
            raise ValueError("victim_indices and values_f must be rank-1 sequences.")
        if victim_array.shape[0] != value_array.shape[0]:
            raise ValueError(
                f"victim_indices and values_f must have the same length, got {victim_array.shape[0]} vs {value_array.shape[0]}"
            )
        if victim_array.size == 0:
            return
        if int(victim_array.min()) < 0 or int(victim_array.max()) >= self.net_count:
            raise ValueError("victim_indices contain values outside the registered net range.")

        scaled_values = value_array * float(scale)
        active = (victim_array != int(master_index)) & (scaled_values != 0.0)
        if not bool(np.any(active)):
            return

        active_victims = victim_array[active]
        active_values = scaled_values[active]
        std_array = np.asarray(std_rel, dtype=np.float64)
        if std_array.ndim == 0:
            active_stds = None if float(std_array) == 1.0 else np.full(active_values.shape, float(std_array), dtype=np.float64)
        elif std_array.ndim == 1 and std_array.shape[0] == victim_array.shape[0]:
            active_stds = std_array[active]
        else:
            raise ValueError(
                "std_rel must be a scalar or a rank-1 sequence with the same length as victim_indices."
            )

        if active_stds is None:
            for victim_index, value_f in zip(active_victims.tolist(), active_values.tolist()):
                key = _pack_directed_edge(master_index, int(victim_index))
                self._directed_coupling_f[key] = self._directed_coupling_f.get(key, 0.0) + float(value_f)
            return

        for victim_index, value_f, std_value in zip(active_victims.tolist(), active_values.tolist(), active_stds.tolist()):
            key = _pack_directed_edge(master_index, int(victim_index))
            self._directed_coupling_f[key] = self._directed_coupling_f.get(key, 0.0) + float(value_f)
            if float(std_value) != 1.0:
                self._directed_std_rel[key] = max(0.0, float(std_value))

    def add_directed_values(
        self,
        master_indices: Sequence[int],
        victim_indices: Sequence[int],
        values_f,
        *,
        scale: float = 1.0,
        std_rel=1.0,
    ) -> None:
        master_array = np.asarray(master_indices, dtype=np.int64)
        victim_array = np.asarray(victim_indices, dtype=np.int64)
        value_matrix = np.asarray(values_f, dtype=np.float64)
        if master_array.ndim != 1 or victim_array.ndim != 1 or value_matrix.ndim != 2:
            raise ValueError("master_indices, victim_indices, and values_f must be shaped [R], [C], and [R, C].")
        expected_shape = (master_array.shape[0], victim_array.shape[0])
        if value_matrix.shape != expected_shape:
            raise ValueError(f"values_f shape mismatch: expected {expected_shape}, got {value_matrix.shape}")
        if victim_array.size and (int(victim_array.min()) < 0 or int(victim_array.max()) >= self.net_count):
            raise ValueError("victim_indices contain values outside the registered net range.")
        std_matrix = np.asarray(std_rel, dtype=np.float64)
        if std_matrix.ndim == 0 and float(std_matrix) == 1.0:
            for row_index, master_index in enumerate(master_array.tolist()):
                master_i = int(master_index)
                if master_i < 0 or master_i >= self.net_count:
                    raise ValueError(f"master_index must be in [0, {self.net_count - 1}], got {master_i}")
                row_values = value_matrix[row_index] * float(scale)
                active = (victim_array != master_i) & (row_values != 0.0)
                if not bool(np.any(active)):
                    continue
                key_prefix = np.uint64(master_i << _PACKED_INDEX_BITS)
                keys = key_prefix | victim_array[active].astype(np.uint64, copy=False)
                for key, value_f in zip(keys.tolist(), row_values[active].tolist()):
                    key_i = int(key)
                    self._directed_coupling_f[key_i] = self._directed_coupling_f.get(key_i, 0.0) + float(value_f)
            return

        for row_index, master_index in enumerate(master_array.tolist()):
            if std_matrix.ndim == 0:
                row_std = float(std_matrix)
            elif std_matrix.shape == value_matrix.shape:
                row_std = std_matrix[row_index]
            else:
                raise ValueError(
                    f"std_rel must be scalar or match values_f shape {value_matrix.shape}, got {std_matrix.shape}"
                )
            self.add_directed_row(
                int(master_index),
                victim_array,
                value_matrix[row_index],
                scale=float(scale),
                std_rel=row_std,
            )

    def _serialize(
        self,
        *,
        sort_nets: bool,
    ) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        net_count = self.net_count
        net_names = list(self._net_names)
        total_cap_f = self._total_cap_f[:net_count].copy()

        if self._directed_coupling_f:
            packed_keys = np.fromiter(
                self._directed_coupling_f.keys(),
                dtype=np.uint64,
                count=len(self._directed_coupling_f),
            )
            directed_values_f = np.fromiter(
                self._directed_coupling_f.values(),
                dtype=np.float64,
                count=len(self._directed_coupling_f),
            )
            directed_std_rel = np.fromiter(
                (self._directed_std_rel.get(key, 1.0) for key in self._directed_coupling_f.keys()),
                dtype=np.float64,
                count=len(self._directed_coupling_f),
            )
            left_indices = (packed_keys >> np.uint64(_PACKED_INDEX_BITS)).astype(np.int64, copy=False)
            right_indices = (packed_keys & np.uint64(_PACKED_INDEX_MASK)).astype(np.int64, copy=False)
        else:
            left_indices = np.empty((0,), dtype=np.int64)
            right_indices = np.empty((0,), dtype=np.int64)
            directed_values_f = np.empty((0,), dtype=np.float64)
            directed_std_rel = np.empty((0,), dtype=np.float64)

        if sort_nets and net_count > 1:
            order = np.asarray(sorted(range(net_count), key=net_names.__getitem__), dtype=np.int64)
            identity = np.arange(net_count, dtype=np.int64)
            if not np.array_equal(order, identity):
                inverse = np.empty_like(order)
                inverse[order] = identity
                net_names = [net_names[int(index)] for index in order.tolist()]
                total_cap_f = total_cap_f[order]
                if left_indices.size:
                    left_indices = inverse[left_indices]
                    right_indices = inverse[right_indices]

        if left_indices.size:
            edge_order = np.lexsort((right_indices, left_indices))
            left_indices = left_indices[edge_order]
            right_indices = right_indices[edge_order]
            directed_values_f = directed_values_f[edge_order]
            directed_std_rel = directed_std_rel[edge_order]

        return net_names, total_cap_f, left_indices, right_indices, directed_values_f, directed_std_rel

    def write(
        self,
        output_path: Path,
        *,
        window_id: str,
        c_unit: str = "PF",
        sort_nets: bool = True,
        derive_total_cap_from_couplings: bool = False,
    ) -> None:
        net_names, total_cap_f, left_indices, right_indices, directed_values_f, directed_std_rel = self._serialize(sort_nets=sort_nets)
        write_indexed_window_spef(
            output_path,
            window_id=window_id,
            net_names=net_names,
            total_cap_f=total_cap_f,
            directed_left_indices=left_indices,
            directed_right_indices=right_indices,
            directed_values_f=directed_values_f,
            directed_std_rel=directed_std_rel,
            derive_total_cap_from_couplings=derive_total_cap_from_couplings,
            c_unit=c_unit,
        )


def _pack_directed_edge(left_index: int, right_index: int) -> int:
    left = int(left_index)
    right = int(right_index)
    if left < 0 or right < 0:
        raise ValueError(f"Directed edge indices must be non-negative, got ({left}, {right})")
    if left > _PACKED_INDEX_MASK or right > _PACKED_INDEX_MASK:
        raise ValueError(f"Directed edge indices exceed the {32}-bit packing limit: ({left}, {right})")
    return (left << _PACKED_INDEX_BITS) | right


def symmetrize_couplings(
    conductor_names: Sequence[str],
    directed_matrix_f,
) -> Dict[Tuple[str, str], float]:
    couplings: Dict[Tuple[str, str], float] = {}
    size = len(conductor_names)
    for left in range(size):
        for right in range(left + 1, size):
            a = float(directed_matrix_f[left][right])
            b = float(directed_matrix_f[right][left])
            couplings[(str(conductor_names[left]), str(conductor_names[right]))] = 0.5 * (a + b)
    return couplings


def symmetrize_directed_indexed_couplings(
    *,
    net_count: int,
    directed_left_indices,
    directed_right_indices,
    directed_values_f,
    directed_std_rel=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left_array = np.asarray(directed_left_indices, dtype=np.int64)
    right_array = np.asarray(directed_right_indices, dtype=np.int64)
    value_array = np.asarray(directed_values_f, dtype=np.float64)
    if directed_std_rel is None:
        std_array = np.ones_like(value_array, dtype=np.float64)
    else:
        std_array = np.asarray(directed_std_rel, dtype=np.float64)
    if left_array.ndim != 1 or right_array.ndim != 1 or value_array.ndim != 1 or std_array.ndim != 1:
        raise ValueError("Directed coupling arrays must all be rank-1.")
    if left_array.shape[0] != right_array.shape[0] or left_array.shape[0] != value_array.shape[0]:
        raise ValueError("Directed coupling index and value arrays must have the same length.")
    if std_array.shape[0] != value_array.shape[0]:
        raise ValueError("directed_std_rel must have the same length as directed_values_f.")
    if net_count < 0:
        raise ValueError(f"net_count must be non-negative, got {net_count}")

    pair_values: Dict[tuple[int, int], list[float]] = {}
    pair_stds: Dict[tuple[int, int], list[float]] = {}
    for left, right, value, std in zip(
        left_array.tolist(),
        right_array.tolist(),
        value_array.tolist(),
        std_array.tolist(),
    ):
        left_i = int(left)
        right_i = int(right)
        if left_i == right_i or float(value) == 0.0:
            continue
        if left_i < 0 or right_i < 0 or left_i >= int(net_count) or right_i >= int(net_count):
            raise ValueError("Directed coupling indices are out of bounds for the provided net list.")
        low = min(left_i, right_i)
        high = max(left_i, right_i)
        slot = 0 if left_i == low else 1
        key = (low, high)
        if key not in pair_values:
            pair_values[key] = [0.0, 0.0]
            pair_stds[key] = [1.0, 1.0]
        pair_values[key][slot] += float(value)
        pair_stds[key][slot] = max(0.0, float(std))

    out_left: list[int] = []
    out_right: list[int] = []
    out_values: list[float] = []
    for key in sorted(pair_values):
        forward, reverse = pair_values[key]
        forward_present = forward != 0.0
        reverse_present = reverse != 0.0
        if forward_present and reverse_present:
            forward_std, reverse_std = pair_stds[key]
            forward_var = float(forward_std) * float(forward_std)
            reverse_var = float(reverse_std) * float(reverse_std)
            denom = forward_var + reverse_var
            value = 0.5 * (forward + reverse) if denom == 0.0 else (
                (forward * reverse_var) + (reverse * forward_var)
            ) / denom
        elif forward_present:
            value = forward
        elif reverse_present:
            value = reverse
        else:
            continue
        if value == 0.0:
            continue
        out_left.append(key[0])
        out_right.append(key[1])
        out_values.append(float(value))

    return (
        np.asarray(out_left, dtype=np.int64),
        np.asarray(out_right, dtype=np.int64),
        np.asarray(out_values, dtype=np.float64),
    )


def totals_from_unordered_couplings(
    net_count: int,
    left_indices,
    right_indices,
    coupling_values_f,
) -> np.ndarray:
    totals = np.zeros((int(net_count),), dtype=np.float64)
    left_array = np.asarray(left_indices, dtype=np.int64)
    right_array = np.asarray(right_indices, dtype=np.int64)
    value_array = np.asarray(coupling_values_f, dtype=np.float64)
    if left_array.shape[0] != right_array.shape[0] or left_array.shape[0] != value_array.shape[0]:
        raise ValueError("Coupling index and value arrays must have the same length.")
    for left, right, value in zip(left_array.tolist(), right_array.tolist(), value_array.tolist()):
        left_i = int(left)
        right_i = int(right)
        if left_i == right_i or float(value) == 0.0:
            continue
        if left_i < 0 or right_i < 0 or left_i >= int(net_count) or right_i >= int(net_count):
            raise ValueError("Coupling indices are out of bounds for the provided net list.")
        totals[left_i] += float(value)
        totals[right_i] += float(value)
    return totals


def symmetrize_directed_couplings(
    directed_cap_f: Dict[Tuple[str, str], float],
) -> Dict[Tuple[str, str], float]:
    names = sorted({str(name) for pair in directed_cap_f for name in pair})
    index = {name: idx for idx, name in enumerate(names)}
    left_indices: list[int] = []
    right_indices: list[int] = []
    values: list[float] = []
    for (left, right), value in directed_cap_f.items():
        if str(left) == str(right):
            continue
        left_indices.append(index[str(left)])
        right_indices.append(index[str(right)])
        values.append(float(value))
    out_left, out_right, out_values = symmetrize_directed_indexed_couplings(
        net_count=len(names),
        directed_left_indices=left_indices,
        directed_right_indices=right_indices,
        directed_values_f=values,
    )
    return {
        (names[int(left)], names[int(right)]): float(value)
        for left, right, value in zip(out_left.tolist(), out_right.tolist(), out_values.tolist())
    }


def totals_from_predictions(
    conductor_names: Sequence[str],
    values_f: Sequence[float],
) -> Dict[str, float]:
    if len(conductor_names) != len(values_f):
        raise ValueError(
            f"conductor_names and values_f must have the same length, got {len(conductor_names)} vs {len(values_f)}"
        )
    return {str(name): float(value) for name, value in zip(conductor_names, values_f)}


def _load_native_spef_module():
    from def_fast_density import load_fast_lefdef_parser_extension

    return load_fast_lefdef_parser_extension()


def _load_native_spef_writer():
    return _load_native_spef_module().write_simple_spef_file


def _load_native_directed_spef_writer():
    return _load_native_spef_module().write_simple_spef_file_from_directed_edges


def write_indexed_window_spef(
    output_path: Path,
    *,
    window_id: str,
    net_names: Sequence[str],
    total_cap_f,
    directed_left_indices,
    directed_right_indices,
    directed_values_f,
    directed_std_rel=None,
    derive_total_cap_from_couplings: bool = False,
    c_unit: str = "PF",
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    net_name_list = [str(name) for name in net_names]
    total_cap_array = np.asarray(total_cap_f, dtype=np.float64)
    left_index_array = np.asarray(directed_left_indices, dtype=np.int64)
    right_index_array = np.asarray(directed_right_indices, dtype=np.int64)
    directed_value_array = np.asarray(directed_values_f, dtype=np.float64)
    directed_std_array = (
        np.ones_like(directed_value_array, dtype=np.float64)
        if directed_std_rel is None
        else np.asarray(directed_std_rel, dtype=np.float64)
    )

    if total_cap_array.ndim != 1:
        raise ValueError(f"total_cap_f must be rank-1, got shape {total_cap_array.shape}")
    if left_index_array.ndim != 1 or right_index_array.ndim != 1 or directed_value_array.ndim != 1:
        raise ValueError("Directed coupling arrays must all be rank-1.")
    if directed_std_array.ndim != 1:
        raise ValueError("directed_std_rel must be rank-1.")
    if total_cap_array.shape[0] != len(net_name_list):
        raise ValueError(
            f"net_names and total_cap_f must have the same length, got {len(net_name_list)} vs {total_cap_array.shape[0]}"
        )
    if left_index_array.shape[0] != right_index_array.shape[0] or left_index_array.shape[0] != directed_value_array.shape[0]:
        raise ValueError("Directed coupling arrays must have the same length.")
    if directed_std_array.shape[0] != directed_value_array.shape[0]:
        raise ValueError("directed_std_rel must have the same length as directed_values_f.")

    default_std_rel = directed_std_rel is None or bool(np.all(directed_std_array == 1.0))
    if not derive_total_cap_from_couplings and default_std_rel:
        writer = _load_native_directed_spef_writer()
        writer(
            str(output_path),
            str(window_id),
            net_name_list,
            np.ascontiguousarray(total_cap_array, dtype=np.float64),
            np.ascontiguousarray(left_index_array, dtype=np.int64),
            np.ascontiguousarray(right_index_array, dtype=np.int64),
            np.ascontiguousarray(directed_value_array, dtype=np.float64),
            str(c_unit).upper(),
            True,
        )
        return

    coupling_left, coupling_right, coupling_values = symmetrize_directed_indexed_couplings(
        net_count=len(net_name_list),
        directed_left_indices=left_index_array,
        directed_right_indices=right_index_array,
        directed_values_f=directed_value_array,
        directed_std_rel=directed_std_array,
    )
    if derive_total_cap_from_couplings:
        total_cap_array = totals_from_unordered_couplings(
            len(net_name_list),
            coupling_left,
            coupling_right,
            coupling_values,
        )

    writer = _load_native_spef_writer()
    writer(
        str(output_path),
        str(window_id),
        net_name_list,
        total_cap_array.tolist(),
        coupling_left.tolist(),
        coupling_right.tolist(),
        coupling_values.tolist(),
        str(c_unit).upper(),
        True,
    )


def write_window_spef(
    output_path: Path,
    *,
    window_id: str,
    total_cap_f: Dict[str, float],
    coupling_cap_f: Dict[Tuple[str, str], float],
    c_unit: str = "PF",
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    net_names = sorted(str(name) for name in total_cap_f)
    net_index = {name: idx for idx, name in enumerate(net_names)}
    total_caps = [float(total_cap_f[name]) for name in net_names]

    left_indices: list[int] = []
    right_indices: list[int] = []
    coupling_values: list[float] = []
    for (left_name, right_name), value_f in sorted(coupling_cap_f.items()):
        left = str(left_name)
        right = str(right_name)
        if left == right:
            continue
        if left not in net_index or right not in net_index:
            continue
        left_indices.append(net_index[left])
        right_indices.append(net_index[right])
        coupling_values.append(float(value_f))

    writer = _load_native_spef_writer()
    writer(
        str(output_path),
        str(window_id),
        net_names,
        total_caps,
        left_indices,
        right_indices,
        coupling_values,
        str(c_unit).upper(),
        True,
    )
