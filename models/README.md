# Released models

The `v1.0.0` GitHub release provides the 14 selected D4_D_k5 checkpoints from the accepted paper's main accuracy table. The artifacts are intentionally not committed to Git.

Download all models:

```bash
gh release download v1.0.0 \
  --repo THU-numbda/flash-cnncap \
  --pattern 'flash-cnncap-d4-d-k5-*.pth' \
  --dir models
```

Each file contains:

- `format_version`: release checkpoint schema version
- `state_dict`: model weights
- `metadata`: model type, MONAI configuration, active layers, dataset, goal, seed, best epoch, paper metrics, and source hash

Optimizer state and private training paths have been removed. The files remain directly compatible with `full-pipeline/compile_models.py` and the repository's checkpoint loaders.

`paper_models.json` identifies the original checkpoint selected for every paper row. `release_manifest.json` records artifact sizes and SHA-256 hashes generated during export.

Verify downloads:

```bash
(cd models && sha256sum -c SHA256SUMS)
```

Only Nangate45 and Sky130HD CapBench checkpoints are used by the DEF-to-SPEF deployment path. The CNN-Cap legacy pair is included because it appears in the main accuracy table.
