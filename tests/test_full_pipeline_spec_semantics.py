from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
FULL_PIPELINE = REPO_ROOT / "full-pipeline"
if str(FULL_PIPELINE) not in sys.path:
    sys.path.insert(0, str(FULL_PIPELINE))

from spef_runtime import (  # noqa: E402
    symmetrize_directed_indexed_couplings,
    totals_from_unordered_couplings,
)
from tile_utils import build_tiled_window_jobs  # noqa: E402


def test_patch_tiling_uses_owned_patch_margin_and_10um_model_window() -> None:
    jobs = build_tiled_window_jobs(
        "tiny",
        die_bounds_um=(0.0, 0.0, 16.0, 8.0),
        target_size=224,
        tile_size_um=10.0,
        tile_context_um=1.0,
    )

    assert len(jobs) == 2
    assert jobs[0].ownership_bounds == (0.0, 0.0, 8.0, 8.0)
    assert jobs[0].patch_bounds == (0.0, 0.0, 8.0, 8.0)
    assert jobs[0].margin_bounds == (-1.0, -1.0, 9.0, 9.0)
    assert jobs[0].solve_bounds == (-1.0, -1.0, 9.0, 9.0)
    assert jobs[0].raster_bounds == jobs[0].solve_bounds
    assert jobs[0].pixel_resolution_um == (10.0 / 224.0)

    assert jobs[1].ownership_bounds == (8.0, 0.0, 16.0, 8.0)
    assert jobs[1].patch_bounds == (8.0, 0.0, 16.0, 8.0)
    assert jobs[1].margin_bounds == (7.0, -1.0, 17.0, 9.0)
    assert jobs[1].solve_bounds == (7.0, -1.0, 17.0, 9.0)


def test_edge_patch_keeps_full_10um_model_window() -> None:
    jobs = build_tiled_window_jobs(
        "gcd",
        die_bounds_um=(0.0, 0.0, 36.24, 36.24),
        target_size=224,
        tile_size_um=10.0,
        tile_context_um=1.0,
    )

    assert len(jobs) == 25
    last = jobs[-1]
    assert last.ownership_bounds == (32.0, 32.0, 36.24, 36.24)
    assert last.raster_bounds == (31.0, 31.0, 41.0, 41.0)
    assert last.solve_bounds == last.raster_bounds
    assert last.pixel_resolution_um == (10.0 / 224.0)


def test_patch_fragment_remap_splits_rectangles_crossing_patch_boundary() -> None:
    try:
        from def_fast_density import PreparedDefRasterInput
        from window_runtime import _remap_packed_rects_to_patch_fragments
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise unittest.SkipTest("full pipeline staging tests require torch") from exc
        raise

    prepared = PreparedDefRasterInput(
        def_path=Path("tiny.def"),
        lef_path=Path("tiny.lef"),
        lef_paths=(Path("tiny.lef"),),
        channel_layers=["M1"],
        backend="test",
        target_size=20,
        pixel_resolution=1.0,
        window_bounds=np.asarray((0.0, 0.0, 0.0, 20.0, 10.0, 0.0), dtype=np.float64),
        conductor_names_sorted=["netA"],
        conductor_ids_sorted=np.asarray([1], dtype=np.int16),
        conductor_is_synthetic=np.asarray([False], dtype=bool),
        conductor_source_kind_codes=np.asarray([0], dtype=np.uint8),
        packed_rects=np.asarray([[0, 1, 8, 12, 0, 5]], dtype=np.int32),
        packed_rects_torch=None,
        rect_source_kind_codes=np.asarray([0], dtype=np.uint8),
        net_name_to_gpu_id={"netA": 1},
        total_segments=1,
        total_endpoint_extensions=0,
        active_rectangles=1,
        parse_ms=0.0,
        prepare_ms=0.0,
        component_resolution_stats={},
    )

    remapped = _remap_packed_rects_to_patch_fragments(
        prepared,
        patch_grid_origin=(0.0, 0.0),
        patch_size_um=10.0,
        patch_grid_shape=(1, 2),
    )

    assert remapped.conductor_names_sorted == ["netA", "netA"]
    assert remapped.real_conductor_ids_sorted.tolist() == [1, 2]
    assert remapped.packed_rects.tolist() == [
        [0, 1, 8, 10, 0, 5],
        [0, 2, 10, 12, 0, 5],
    ]


def test_env_master_queries_are_unique_by_output_net() -> None:
    try:
        import torch
        from run import _build_staged_window
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise unittest.SkipTest("full pipeline staging tests require torch") from exc
        raise

    window = SimpleNamespace(
        window_id="dedup",
        occupied=torch.zeros((1, 2, 2), dtype=torch.float32),
        full_local_map=torch.as_tensor([[[0, 1], [2, 4]]], dtype=torch.long),
        owned_local_counts=torch.zeros((5,), dtype=torch.long),
        owned_sparse_indices=torch.zeros((5, 1), dtype=torch.long),
        owned_sparse_counts=torch.zeros((5,), dtype=torch.long),
        owned_query_local_ids=torch.as_tensor([1, 2, 3], dtype=torch.long),
        visible_master_local_ids=torch.as_tensor([1, 2, 3, 4], dtype=torch.long),
        real_conductor_names=["netA", "netA", "netB", "netA"],
    )

    def register_nets(names):
        name_to_index = {}
        indices = []
        for name in names:
            if name not in name_to_index:
                name_to_index[name] = len(name_to_index)
            indices.append(name_to_index[name])
        return tuple(indices)

    staged = _build_staged_window(window, register_nets)

    assert staged.visible_master_local_ids == (1, 3)
    assert staged.visible_master_output_indices == (0, 1)
    assert staged.owned_query_output_indices == (0, 0, 1)


def test_batched_env_features_mask_all_fragments_for_master_output_net() -> None:
    try:
        import torch
        from run import EnvWorkItem, _build_env_feature_batch_from_cache, _build_tile_tensor_cache
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise unittest.SkipTest("full pipeline batching tests require torch") from exc
        raise

    staged_a = SimpleNamespace(
        occupied=torch.zeros((1, 2, 2), dtype=torch.float32),
        full_local_map=torch.as_tensor([[[0, 1], [2, 3]]], dtype=torch.long),
        owned_local_counts=torch.zeros((4,), dtype=torch.long),
        owned_sparse_indices=torch.zeros((4, 1), dtype=torch.long),
        owned_sparse_counts=torch.zeros((4,), dtype=torch.long),
        local_output_indices=torch.as_tensor([-1, 0, 0, 1], dtype=torch.long),
    )
    staged_b = SimpleNamespace(
        occupied=torch.ones((1, 2, 2), dtype=torch.float32),
        full_local_map=torch.as_tensor([[[1, 2], [0, 1]]], dtype=torch.long),
        owned_local_counts=torch.zeros((3,), dtype=torch.long),
        owned_sparse_indices=torch.zeros((3, 1), dtype=torch.long),
        owned_sparse_counts=torch.zeros((3,), dtype=torch.long),
        local_output_indices=torch.as_tensor([-1, 2, 3], dtype=torch.long),
    )

    cache = _build_tile_tensor_cache([staged_a, staged_b])
    batch = _build_env_feature_batch_from_cache(
        cache,
        [
            EnvWorkItem(window_index=0, master_local_id=1, master_output_index=0),
            EnvWorkItem(window_index=1, master_local_id=2, master_output_index=3),
        ],
    )

    assert batch.tolist() == [
        [[
            [0.0, -2.0],
            [-2.0, 0.0],
        ]],
        [[
            [1.0, -1.0],
            [1.0, 1.0],
        ]],
    ]


def test_batched_total_accumulation_maps_rows_back_to_staged_windows() -> None:
    try:
        import torch
        from run import _accumulate_total_prediction_batch
        from spef_runtime import IndexedSpefAccumulator
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise unittest.SkipTest("full pipeline batching tests require torch") from exc
        raise

    accumulator = IndexedSpefAccumulator.from_net_names(["a", "b", "c"])
    staged_windows = [
        SimpleNamespace(
            owned_query_ids=torch.as_tensor([1, 2], dtype=torch.long),
            owned_query_output_indices=(0, 1),
        ),
        SimpleNamespace(
            owned_query_ids=torch.as_tensor([1], dtype=torch.long),
            owned_query_output_indices=(2,),
        ),
    ]
    reduced = torch.as_tensor(
        [
            [0.0, 10.0, 20.0],
            [0.0, 30.0, 40.0],
        ],
        dtype=torch.float32,
    )

    _accumulate_total_prediction_batch(accumulator, staged_windows, (0, 1), reduced)
    _names, totals, _left, _right, _values, _std = accumulator._serialize(sort_nets=False)

    assert np.allclose(totals, np.asarray([10.0e-15, 20.0e-15, 30.0e-15], dtype=np.float64))


def test_coupling_accumulation_uses_owned_victim_outputs_only() -> None:
    try:
        import torch
        from run import _accumulate_coupling_predictions
        from spef_runtime import IndexedSpefAccumulator
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise unittest.SkipTest("full pipeline accumulation tests require torch") from exc
        raise

    accumulator = IndexedSpefAccumulator.from_net_names(["master", "owned", "context"])
    staged = SimpleNamespace(
        owned_query_output_indices=(1,),
    )

    _accumulate_coupling_predictions(
        accumulator,
        staged,
        master_output_indices=(0,),
        preds=torch.as_tensor([[5.0]], dtype=torch.float32),
    )

    names, _totals, left, right, values, _std = accumulator._serialize(sort_nets=False)

    assert names == ["master", "owned", "context"]
    assert left.tolist() == [0]
    assert right.tolist() == [1]
    assert np.allclose(values, np.asarray([5.0e-15], dtype=np.float64))


def test_directed_merge_uses_single_direction_without_halving() -> None:
    left, right, values = symmetrize_directed_indexed_couplings(
        net_count=2,
        directed_left_indices=[0],
        directed_right_indices=[1],
        directed_values_f=[7.0],
    )

    assert left.tolist() == [0]
    assert right.tolist() == [1]
    assert values.tolist() == [7.0]


def test_directed_merge_uses_spec_weighted_bidirectional_formula() -> None:
    left, right, values = symmetrize_directed_indexed_couplings(
        net_count=2,
        directed_left_indices=[0, 1],
        directed_right_indices=[1, 0],
        directed_values_f=[10.0, 4.0],
        directed_std_rel=[2.0, 1.0],
    )

    assert left.tolist() == [0]
    assert right.tolist() == [1]
    assert np.allclose(values, np.asarray([5.2], dtype=np.float64))


def test_totals_are_derived_from_final_unordered_couplings() -> None:
    totals = totals_from_unordered_couplings(
        3,
        left_indices=[0, 0, 1],
        right_indices=[1, 2, 2],
        coupling_values_f=[1.5, 2.5, 4.0],
    )

    assert np.allclose(totals, np.asarray([4.0, 5.5, 6.5], dtype=np.float64))
