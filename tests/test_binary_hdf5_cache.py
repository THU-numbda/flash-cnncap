from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from training.binary_hdf5_cache import (
    BinaryHdf5CacheConfig,
    BinaryHdf5CapBenchDataset,
    binary_hdf5_cache_is_valid,
    build_binary_cache_spec,
    ensure_binary_hdf5_cache,
    normalize_hdf5_compression,
)


class _FakeDataset:
    active_layers = ["M1"]
    tensor_shape = (1, 2, 3)

    def __init__(self, window_ids=("W0",)):
        self._window_ids = list(window_ids)
        self._windows = [SimpleNamespace(name=window_id) for window_id in self._window_ids]
        self._window_samples = [[object()] for _ in self._window_ids]

    def __len__(self):
        return len(self._window_ids)

    def __getitem__(self, index):
        return index

    def get_window_ids(self):
        return self._window_ids.copy()

    def get_window_sample_ranges(self):
        return [(idx, idx + 1) for idx in range(len(self))]

    def create_window_subset(self, window_ids):
        return _FakeDataset(tuple(window_ids))

    def _build_window_features(self, row):
        del row
        return np.asarray([[[0, 1, 1], [0, 0, 1]]], dtype=np.float32)

    def _build_window_local_state(self, row):
        del row
        local_map = np.asarray([[[0, 1, 2], [0, 0, 2]]], dtype=np.int64)
        return local_map, np.asarray([1, 1, 2]), {7: 1, 11: 2}


def test_binary_cache_has_no_density_dataset_and_round_trips(tmp_path: Path):
    source = _FakeDataset()
    spec = build_binary_cache_spec(
        source,
        selector="nangate45/small",
        source_root=tmp_path / "source",
        trim_margin=True,
    )
    path = ensure_binary_hdf5_cache(
        source,
        spec,
        BinaryHdf5CacheConfig(cache_dir=tmp_path / "cache", compression=None),
    )

    assert binary_hdf5_cache_is_valid(path, spec)
    with h5py.File(path, "r") as handle:
        assert handle.attrs["input_encoding"] == "binary_occupancy"
        assert "occupancy" in handle
        assert "local_maps" in handle
        assert "density" not in handle
        assert "density_maps" not in handle
        assert handle["occupancy"].dtype == np.dtype("uint8")

    cached = BinaryHdf5CapBenchDataset(source, path)
    np.testing.assert_array_equal(cached._build_window_features(0), source._build_window_features(0))
    local_map, counts, mapping = cached._build_window_local_state(0)
    expected_map, expected_counts, expected_mapping = source._build_window_local_state(0)
    np.testing.assert_array_equal(local_map, expected_map)
    np.testing.assert_array_equal(counts, expected_counts)
    assert mapping == expected_mapping

    _window, runtime = cached._get_window_and_cache(0)
    features = cached._build_window_features(0)
    cached._apply_highlight(features, runtime, 11, positive=False)
    assert features.tolist() == [[[0.0, 1.0, -1.0], [0.0, 0.0, -1.0]]]


def test_cache_rejects_non_binary_features(tmp_path: Path):
    source = _FakeDataset()
    source._build_window_features = lambda row: np.asarray([[[0, 0.5, 1], [0, 0, 1]]], dtype=np.float32)
    spec = build_binary_cache_spec(
        source,
        selector="sky130hd/large",
        source_root=tmp_path / "source",
        trim_margin=False,
    )

    try:
        ensure_binary_hdf5_cache(source, spec, BinaryHdf5CacheConfig(cache_dir=tmp_path / "cache"))
    except ValueError as exc:
        assert "non-binary" in str(exc)
    else:
        raise AssertionError("Expected fractional model inputs to be rejected")


def test_normalize_hdf5_compression():
    assert normalize_hdf5_compression(None) is None
    assert normalize_hdf5_compression("none") is None
    assert normalize_hdf5_compression("lzf") == "lzf"
    assert normalize_hdf5_compression("gzip") == "gzip"
