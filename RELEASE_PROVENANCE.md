# Release provenance

This is a curated paper-artifact repository rather than a copy of the private development history.

- Accepted-paper code lineage: `HectorRguez/CNNCap-flash`
- Clean source snapshot: `84148e9e295af3de00cc5201e46d119bca343d90`
- Training lineage at the time of the reported runs: `673a6d977e3c1c94bde3e29bf7bc7e4dbe356b83`
- CapBench revision used by the reported runs: `33939cb5002575c06f19c237e2eae6105e331634`
- Optional binary-occupancy HDF5 cache: release-only implementation informed by the later I/O optimization, with density-map storage removed
- Release-model seed: `11037`
- Ablation seeds: `11037`, `11038`, `11039`, `11040`, `11041`

The clean release excludes post-paper mixed-PDK, combined-target, density-input, JPEG-holdout, and other exploratory experiments. It also excludes private paths, run logs, datasets, TensorRT engines, optimizer states, and unrelated documents.

The model release manifest records SHA-256 hashes of both the original OutGPU3 checkpoints and the optimizer-free public artifacts. This provides a verifiable link between the paper runs and the published files without retaining private filesystem metadata.
