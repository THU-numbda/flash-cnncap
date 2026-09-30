# Unreleased

## Exact DEF geometry in the deployment rasterizer

The native DEF rasterizer now reproduces the layout that the training windows and RWCap references were built from (the GDS written by OpenROAD-flow-scripts), instead of an approximation:

- vias come from the LEF and DEF VIAS definitions (including VIARULE-generated arrays) rather than being inferred from overlapping metal;
- DEF wire extensions are honoured and special wires have flush ends;
- non-default-rule widths, routing RECT patches, and IO pin shapes are included;
- standard-cell metal comes from the cell GDS; cell-internal metal is visible to the model but never queried;
- the design is expanded once and indexed spatially, so tiled full-chip runs no longer re-process the whole design per tile.

The rasterized geometry matches GDS-derived CAP3D exactly on the Nangate45 gcd die and a Sky130HD ibex clip. On gcd, the released models' full-chip error against RWCap drops from 9.6% to 3.7% MARE for totals and from 16.9% to 10.6% for couplings >= 1 aF (FP32 coupling engine; the released FP16 coupling engine overflows). End-to-end numbers reported in the paper were produced with the v1.0.0 rasterizer; use the `v1.0.0` tag to reproduce them.

# Flash-CNNCap v1.0.0

Initial artifact release for the accepted ICCAD 2026 paper, “Flash-CNNCap: Capacitance Extraction via Image Mapping.”

## Included

- Binary-occupancy training and evaluation code for the reported protocol
- The exact 13-model architecture ablation registry
- GPU-accelerated DEF-to-SPEF deployment code
- Nangate45 and Sky130HD technology-stack configurations
- An optional binary-occupancy HDF5 cache (no fractional density inputs)
- Fourteen optimizer-free D4_D_k5 checkpoints covering total and coupling prediction for CNN-Cap legacy and all six reported CapBench subsets
- SHA-256 hashes linking the release files to the original paper-run checkpoints

## Excluded

Post-paper mixed-PDK, combined-target, density-input, JPEG-holdout, and other exploratory experiments are not part of this artifact. Datasets, run logs, optimizer states, TensorRT engines, private paths, and unrelated documents are also excluded.

See `README.md`, `docs/reproduction.md`, and `models/README.md` for setup, reproduction, and checkpoint verification.
