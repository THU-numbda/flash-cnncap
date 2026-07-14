# Optional binary-occupancy HDF5 cache

The accepted paper uses binary per-layer occupancy as the model input. The optional cache in `training/binary_hdf5_cache.py` preserves that protocol: it stores `uint8` occupancy tensors plus conductor-local maps needed for highlighting and output reduction. It refuses non-binary feature arrays and rejects cache files containing `density` or `density_maps` datasets.

No fractional density map is written to the HDF5 file. The cache is an I/O optimization only and does not affect model inputs, targets, splitting, or metrics.

Enable it on any paper training command:

```bash
python training/train.py \
  --dataset-format capbench \
  --dataset-path nangate45/small \
  --goal total \
  --model_type unet \
  --monai-config D4_D_k5 \
  --binary-hdf5-cache-dir .cache/flash-cnncap
```

The first run builds a provenance-checked cache. Later runs reuse it, including total and coupling runs with the same window set. Available controls are:

- `--binary-hdf5-compression {none,lzf,gzip}` (default: `lzf`)
- `--binary-hdf5-chunk-windows N` (default: `8`)
- `--rebuild-binary-hdf5-cache`

Cache files and lock files live under the requested cache directory and are ignored by Git. Each DataLoader worker lazily opens its own HDF5 handle.

Run the CPU-only cache test with:

```bash
python -m pytest -q tests/test_binary_hdf5_cache.py
```
