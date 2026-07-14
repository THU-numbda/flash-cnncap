"""Optional HDF5 cache for the paper's binary-occupancy CapBench inputs."""

from __future__ import annotations

import hashlib
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
from torch.utils.data import Dataset


BINARY_CACHE_FORMAT_VERSION = 1


@dataclass(frozen=True)
class BinaryHdf5CacheSpec:
    selector: str
    source_root: Path
    trim_margin: bool
    shape: tuple[int, int, int]
    active_layers: tuple[str, ...]
    window_ids: tuple[str, ...]

    @property
    def window_fingerprint(self) -> str:
        digest = hashlib.sha1()
        for window_id in self.window_ids:
            digest.update(window_id.encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()[:12]


@dataclass(frozen=True)
class BinaryHdf5CacheConfig:
    cache_dir: Path
    compression: str | None = "lzf"
    chunk_windows: int = 8
    rebuild: bool = False
    lock_timeout_seconds: float = 6 * 60 * 60


@dataclass(frozen=True)
class _BinaryWindowCache:
    base_features: np.ndarray
    local_map: np.ndarray
    local_counts: np.ndarray
    actual_to_local: Mapping[int, int]


def build_binary_cache_spec(
    dataset: Any,
    *,
    selector: str,
    source_root: Path,
    trim_margin: bool,
) -> BinaryHdf5CacheSpec:
    return BinaryHdf5CacheSpec(
        selector=str(selector),
        source_root=Path(source_root).expanduser().resolve(),
        trim_margin=bool(trim_margin),
        shape=tuple(int(value) for value in dataset.tensor_shape),
        active_layers=tuple(str(value) for value in dataset.active_layers),
        window_ids=tuple(str(value) for value in dataset.get_window_ids()),
    )


def normalize_hdf5_compression(value: str | None) -> str | None:
    normalized = "none" if value is None else str(value).strip().lower()
    if normalized in {"", "none", "null", "false"}:
        return None
    if normalized not in {"lzf", "gzip"}:
        raise ValueError(f"Unsupported HDF5 compression: {value}")
    return normalized


def _sanitize_cache_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip("/")).strip("_")


def binary_hdf5_cache_path(cache_dir: Path, spec: BinaryHdf5CacheSpec) -> Path:
    root_hash = hashlib.sha1(str(spec.source_root).encode("utf-8")).hexdigest()[:10]
    trim_tag = "trim" if spec.trim_margin else "notrim"
    selector = _sanitize_cache_name(spec.selector)
    return Path(cache_dir) / f"{selector}_binary_{trim_tag}_{spec.window_fingerprint}_{root_hash}.h5"


def _read_strings(dataset: Any) -> tuple[str, ...]:
    return tuple(item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in dataset[()])


def binary_hdf5_cache_is_valid(path: Path, spec: BinaryHdf5CacheSpec) -> bool:
    if not Path(path).is_file():
        return False
    try:
        import h5py  # type: ignore

        with h5py.File(path, "r") as handle:
            if int(handle.attrs.get("format_version", 0)) != BINARY_CACHE_FORMAT_VERSION:
                return False
            if str(handle.attrs.get("input_encoding", "")) != "binary_occupancy":
                return False
            if str(handle.attrs.get("source_root", "")) != str(spec.source_root):
                return False
            if str(handle.attrs.get("selector", "")) != spec.selector:
                return False
            if bool(handle.attrs.get("trim_margin", False)) != spec.trim_margin:
                return False
            if _read_strings(handle["window_ids"]) != spec.window_ids:
                return False
            if _read_strings(handle["active_layers"]) != spec.active_layers:
                return False
            expected_shape = (len(spec.window_ids), *spec.shape)
            if tuple(int(value) for value in handle["occupancy"].shape) != expected_shape:
                return False
            if tuple(int(value) for value in handle["local_maps"].shape) != expected_shape:
                return False
            if handle["occupancy"].dtype != np.dtype("uint8"):
                return False
            if int(handle["local_offsets"].shape[0]) != len(spec.window_ids) + 1:
                return False
            # A density dataset would silently change the release scope.
            if "density" in handle or "density_maps" in handle:
                return False
    except (KeyError, OSError, TypeError, ValueError):
        return False
    return True


def _actual_ids_by_local(actual_to_local: Mapping[int, int], count: int) -> np.ndarray:
    actual_ids = np.zeros((count,), dtype=np.int32)
    for actual_id, local_id in actual_to_local.items():
        local_idx = int(local_id)
        if local_idx <= 0 or local_idx >= count:
            raise ValueError(f"Invalid local conductor ID {local_id} for {count} entries")
        if actual_ids[local_idx] != 0:
            raise ValueError(f"Duplicate local conductor ID {local_id}")
        actual_ids[local_idx] = int(actual_id)
    if count > 1 and np.any(actual_ids[1:] <= 0):
        raise ValueError("Local conductor map is not contiguous")
    return actual_ids


def build_binary_hdf5_cache(
    dataset: Any,
    path: Path,
    spec: BinaryHdf5CacheSpec,
    *,
    compression: str | None,
    chunk_windows: int,
) -> None:
    """Build an occupancy-only cache; no fractional density arrays are written."""
    import h5py  # type: ignore

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    if tmp_path.exists():
        tmp_path.unlink()

    dataset_ids = tuple(str(value) for value in dataset.get_window_ids())
    if dataset_ids != spec.window_ids:
        raise ValueError("Dataset window order does not match the binary HDF5 cache specification")
    chunks = (max(1, min(int(chunk_windows), len(spec.window_ids))), *spec.shape)
    print(
        f"Building binary-occupancy HDF5 cache selector={spec.selector} "
        f"windows={len(spec.window_ids)} shape={spec.shape} path={path}",
        flush=True,
    )

    actual_id_chunks: list[np.ndarray] = []
    count_chunks: list[np.ndarray] = []
    offsets = [0]
    try:
        with h5py.File(tmp_path, "w") as handle:
            handle.attrs["format_version"] = BINARY_CACHE_FORMAT_VERSION
            handle.attrs["input_encoding"] = "binary_occupancy"
            handle.attrs["source_root"] = str(spec.source_root)
            handle.attrs["selector"] = spec.selector
            handle.attrs["trim_margin"] = spec.trim_margin
            handle.attrs["created_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
            string_dtype = h5py.string_dtype(encoding="utf-8")
            handle.create_dataset("window_ids", data=np.asarray(spec.window_ids, dtype=object), dtype=string_dtype)
            handle.create_dataset("active_layers", data=np.asarray(spec.active_layers, dtype=object), dtype=string_dtype)
            occupancy_ds = handle.create_dataset(
                "occupancy",
                shape=(len(spec.window_ids), *spec.shape),
                dtype=np.uint8,
                chunks=chunks,
                compression=compression,
            )
            local_maps_ds = handle.create_dataset(
                "local_maps",
                shape=(len(spec.window_ids), *spec.shape),
                dtype=np.uint32,
                chunks=chunks,
                compression=compression,
            )

            for row, window_id in enumerate(spec.window_ids):
                features = np.asarray(dataset._build_window_features(row))  # pylint: disable=protected-access
                if tuple(int(value) for value in features.shape) != spec.shape:
                    raise ValueError(f"Unexpected occupancy shape for {window_id}: {features.shape}")
                if not np.all((features == 0) | (features == 1)):
                    raise ValueError(
                        f"{window_id} contains non-binary model inputs; refusing to write a paper cache"
                    )
                local_map, local_counts, actual_to_local = dataset._build_window_local_state(row)  # pylint: disable=protected-access
                local_map = np.asarray(local_map)
                local_counts = np.asarray(local_counts, dtype=np.int64)
                if tuple(int(value) for value in local_map.shape) != spec.shape:
                    raise ValueError(f"Unexpected local-map shape for {window_id}: {local_map.shape}")
                if local_counts.ndim != 1 or local_counts.size == 0:
                    raise ValueError(f"Invalid local conductor counts for {window_id}")
                if int(local_map.max(initial=0)) >= int(local_counts.size):
                    raise ValueError(f"Local conductor IDs exceed the count table for {window_id}")

                occupancy_ds[row] = features.astype(np.uint8, copy=False)
                local_maps_ds[row] = local_map.astype(np.uint32, copy=False)
                actual_id_chunks.append(_actual_ids_by_local(actual_to_local, int(local_counts.size)))
                count_chunks.append(local_counts)
                offsets.append(offsets[-1] + int(local_counts.size))

            handle.create_dataset("local_offsets", data=np.asarray(offsets, dtype=np.int64))
            handle.create_dataset("local_actual_ids", data=np.concatenate(actual_id_chunks), dtype=np.int32)
            handle.create_dataset("local_counts", data=np.concatenate(count_chunks), dtype=np.int64)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _wait_for_cache(path: Path, spec: BinaryHdf5CacheSpec, lock_path: Path, timeout: float) -> bool:
    start = time.monotonic()
    while lock_path.exists():
        if binary_hdf5_cache_is_valid(path, spec):
            return True
        if time.monotonic() - start > timeout:
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass
            return False
        time.sleep(5.0)
    return binary_hdf5_cache_is_valid(path, spec)


def ensure_binary_hdf5_cache(
    dataset: Any,
    spec: BinaryHdf5CacheSpec,
    config: BinaryHdf5CacheConfig,
) -> Path:
    cache_dir = Path(config.cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = binary_hdf5_cache_path(cache_dir, spec)
    if not config.rebuild and binary_hdf5_cache_is_valid(path, spec):
        return path

    lock_path = path.with_suffix(path.suffix + ".lock")
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        if not config.rebuild and _wait_for_cache(
            path,
            spec,
            lock_path,
            float(config.lock_timeout_seconds),
        ):
            return path
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)

    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(f"pid={os.getpid()}\nstarted_at={datetime.now().astimezone().isoformat(timespec='seconds')}\n")
    try:
        if config.rebuild or not binary_hdf5_cache_is_valid(path, spec):
            build_binary_hdf5_cache(
                dataset,
                path,
                spec,
                compression=normalize_hdf5_compression(config.compression),
                chunk_windows=int(config.chunk_windows),
            )
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
    return path


class BinaryHdf5CapBenchDataset(Dataset):
    """CapBench adapter that reads binary features and local maps from HDF5."""

    def __init__(self, base_dataset: Any, cache_path: Path) -> None:
        self.base_dataset = base_dataset
        self.cache_path = Path(cache_path).expanduser().resolve()
        self._hdf5_file: Any | None = None
        self._windows = base_dataset._windows  # pylint: disable=protected-access
        self._window_samples = base_dataset._window_samples  # pylint: disable=protected-access

        import h5py  # type: ignore

        with h5py.File(self.cache_path, "r") as handle:
            if int(handle.attrs.get("format_version", 0)) != BINARY_CACHE_FORMAT_VERSION:
                raise ValueError(f"Unsupported binary HDF5 cache: {self.cache_path}")
            if str(handle.attrs.get("input_encoding", "")) != "binary_occupancy":
                raise ValueError(f"Cache is not binary occupancy: {self.cache_path}")
            if "density" in handle or "density_maps" in handle:
                raise ValueError(f"Cache unexpectedly includes density inputs: {self.cache_path}")
            cache_ids = _read_strings(handle["window_ids"])
            cache_layers = _read_strings(handle["active_layers"])
            self._chunk_windows = int(handle["occupancy"].chunks[0]) if handle["occupancy"].chunks else 1
        if cache_layers != tuple(str(value) for value in base_dataset.active_layers):
            raise ValueError("Binary HDF5 active layers do not match the CapBench dataset")
        self._row_by_window = {window_id: row for row, window_id in enumerate(cache_ids)}
        missing = [window_id for window_id in self.get_window_ids() if window_id not in self._row_by_window]
        if missing:
            raise ValueError(f"Binary HDF5 cache is missing windows: {missing[:5]}")

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_hdf5_file"] = None
        return state

    def __getattr__(self, name: str):
        if name == "base_dataset":
            raise AttributeError(name)
        return getattr(self.base_dataset, name)

    def close(self) -> None:
        if self._hdf5_file is not None:
            self._hdf5_file.close()
        self._hdf5_file = None

    def _open_hdf5(self):
        if self._hdf5_file is None:
            import h5py  # type: ignore

            self._hdf5_file = h5py.File(self.cache_path, "r")
        return self._hdf5_file

    @property
    def active_layers(self) -> list[str]:
        return list(self.base_dataset.active_layers)

    @property
    def tensor_shape(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.base_dataset.tensor_shape)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        return self.base_dataset[index]

    def get_window_ids(self) -> list[str]:
        return [str(value) for value in self.base_dataset.get_window_ids()]

    def get_window_shard_ids(self) -> list[int]:
        return [self._row_by_window[window_id] // self._chunk_windows for window_id in self.get_window_ids()]

    def create_window_subset(self, window_ids: Sequence[str]) -> "BinaryHdf5CapBenchDataset":
        return BinaryHdf5CapBenchDataset(
            self.base_dataset.create_window_subset(list(window_ids)),
            self.cache_path,
        )

    def _cache_row(self, window_idx: int) -> int:
        return self._row_by_window[self.get_window_ids()[int(window_idx)]]

    def _load_window_cache(self, window_idx: int) -> _BinaryWindowCache:
        handle = self._open_hdf5()
        row = self._cache_row(window_idx)
        offsets = handle["local_offsets"]
        start = int(offsets[row])
        end = int(offsets[row + 1])
        actual_ids = np.asarray(handle["local_actual_ids"][start:end], dtype=np.int32)
        actual_to_local = {
            int(actual_id): local_id
            for local_id, actual_id in enumerate(actual_ids)
            if local_id > 0 and int(actual_id) > 0
        }
        return _BinaryWindowCache(
            base_features=np.asarray(handle["occupancy"][row], dtype=np.float32),
            local_map=np.asarray(handle["local_maps"][row], dtype=np.int64),
            local_counts=np.asarray(handle["local_counts"][start:end], dtype=np.int64),
            actual_to_local=actual_to_local,
        )

    def _get_window_and_cache(self, window_idx: int):
        window = self._windows[int(window_idx)]
        return window, self._load_window_cache(window_idx)

    def _build_window_features(self, window_idx: int) -> np.ndarray:
        return self._load_window_cache(window_idx).base_features.copy()

    def _build_window_local_state(self, window_idx: int):
        cached = self._load_window_cache(window_idx)
        return cached.local_map, cached.local_counts, dict(cached.actual_to_local)

    def _build_window_conductor_masks(self, window_idx: int, conductor_ids: Sequence[int]) -> np.ndarray:
        cached = self._load_window_cache(window_idx)
        masks = np.zeros((len(conductor_ids), *self.tensor_shape), dtype=np.float32)
        for slot, conductor_id in enumerate(conductor_ids):
            local_id = cached.actual_to_local.get(int(conductor_id), 0)
            if local_id > 0:
                masks[slot] = (cached.local_map == local_id).astype(np.float32, copy=False)
        return masks

    def _apply_highlight(
        self,
        features: np.ndarray,
        cached: _BinaryWindowCache,
        conductor_id: int,
        *,
        positive: bool,
    ) -> None:
        local_id = cached.actual_to_local.get(int(conductor_id), 0)
        if local_id <= 0:
            return
        mask = cached.local_map == local_id
        if positive:
            features[mask] += 1.0
        else:
            features[mask] = -features[mask]
