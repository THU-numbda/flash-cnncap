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
