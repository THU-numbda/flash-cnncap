# Flash-CNNCap: Capacitance Extraction via Image Mapping

Reference implementation and trained models for the paper accepted at ICCAD 2026. Flash-CNNCap predicts spatial capacitance-contribution maps and reduces them over conductor masks, reducing full-matrix reconstruction from quadratic to linear model evaluations.

## Release scope

This repository is a clean artifact of the accepted paper. It contains:

- the binary-occupancy D4_D_k5 U-Net training and evaluation code used for the reported results;
- the 13-architecture, five-seed ablation launcher;
- the GPU-accelerated DEF-to-SPEF pipeline;
- pinned Nangate45 and Sky130HD technology-stack configurations;
- an optional HDF5 cache for the same binary-occupancy inputs; and
- a manifest for the 14 released D4_D_k5 checkpoints from the paper's main results table.

Post-paper mixed-PDK, combined-target, density-input, JPEG-holdout, and other exploratory experiments are intentionally excluded.

## Models

The `v1.0.0` GitHub release contains optimizer-free PyTorch checkpoints for total and coupling prediction on the CNN-Cap legacy dataset and all six CapBench subsets. Each artifact is approximately 72 MB and includes the architecture, active layers, seed, epoch, paper metrics, and source-checkpoint hash.

```bash
gh release download v1.0.0 \
  --repo HectorRguez/flash-cnncap \
  --pattern 'flash-cnncap-d4-d-k5-*.pth' \
  --dir models
```

See [`models/README.md`](models/README.md) and [`models/paper_models.json`](models/paper_models.json) for the mapping from datasets to artifacts.

## Environment

The reported runs used the NVIDIA PyTorch 26.02 container, PyTorch `2.11.0a0+eb65b36914.nv26.02`, MONAI `1.5.2`, TensorRT `10.15.1.29`, one RTX 5090 for inference, and seed `11037` unless otherwise stated. The ablation uses seeds `11037` through `11041`.

```bash
apptainer pull pytorch_26.02-py3.sif \
  docker://nvcr.io/nvidia/pytorch:26.02-py3

apptainer exec --nv pytorch_26.02-py3.sif bash
python -m pip install -r requirements.txt
```

Install the exact CapBench revision used for the reported runs:

```bash
git clone https://github.com/THU-numbda/CapBench.git
cd CapBench
git checkout 33939cb5002575c06f19c237e2eae6105e331634
python -m pip install -e '.[all]'
```

Install a dataset through CapBench, for example:

```bash
python -m capbench datasets install nangate45/small
```

## Paper reproduction

List or run the 13 architecture configurations:

```bash
python scripts/ablations.py list
python scripts/ablations.py run \
  --dataset nangate45/small \
  --num-gpus 8 \
  --epochs 100 \
  --repeats 5
```

Run the selected D4_D_k5 model across the registered evaluation matrix:

```bash
python scripts/best_model_eval.py run \
  --model D4_D_k5 \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --epochs 100 \
  --seed 11037
```

Full commands, splits, metrics, and deployment instructions are in [`docs/reproduction.md`](docs/reproduction.md) and [`full-pipeline/README.md`](full-pipeline/README.md).

## Repository layout

- `training/`: model definitions and training code
- `scripts/`: paper ablations, accuracy evaluation, runtime benchmarks, and validation utilities
- `full-pipeline/`: TensorRT compilation and DEF-to-SPEF inference
- `flash_common/`: shared sparse-reduction code
- `tech/`: pinned technology-stack configurations
- `models/`: release model manifest and verification instructions
- `tests/`: CPU-safe semantic and cache tests; CUDA tests are exercised in the paper environment

The optional HDF5 cache stores binary occupancy and conductor-local maps only—never fractional density maps—and does not alter the paper protocol. See [`docs/hdf5-cache.md`](docs/hdf5-cache.md).

## Citation

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). Please also cite CapBench when using its datasets.

## License

Flash-CNNCap is released under the Apache License 2.0. PDK-derived generated files retain their upstream attribution; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
