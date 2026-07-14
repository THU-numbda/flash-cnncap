# Flash-CNNCap: Capacitance Extraction via Image Mapping

**Accepted at ICCAD 2026.**

Flash-CNNCap predicts spatial capacitance-contribution maps and reduces them over conductor masks, lowering full-matrix reconstruction from $O(n^2)$ to $O(n)$ model evaluations.

This artifact contains the paper's binary-occupancy training code, exact 13-model ablation, GPU DEF-to-SPEF pipeline, and 14 reported D4_D_k5 checkpoints. Post-paper experiments are intentionally excluded.

## Setup

The reported runs used the NVIDIA PyTorch 26.02 container and CapBench revision `33939cb5002575c06f19c237e2eae6105e331634`.

```bash
python -m pip install -r requirements.txt
git clone https://github.com/THU-numbda/CapBench.git
git -C CapBench checkout 33939cb5002575c06f19c237e2eae6105e331634
python -m pip install -e './CapBench[all]'
python -m capbench datasets install nangate45/small
```

## Models

Download the optimizer-free paper checkpoints from the [`v1.0.0` release](https://github.com/THU-numbda/flash-cnncap/releases/tag/v1.0.0):

```bash
gh release download v1.0.0 \
  --repo THU-numbda/flash-cnncap \
  --pattern 'flash-cnncap-d4-d-k5-*.pth' \
  --dir models
```

The model-to-dataset mapping and checksums are in [`models/`](models/).

## Reproduce

```bash
python scripts/ablations.py list
python scripts/ablations.py run --dataset nangate45/small --num-gpus 8 --epochs 100 --repeats 5
python scripts/best_model_eval.py run --model D4_D_k5 --gpu-ids 0,1,2,3,4,5,6,7 --epochs 100
```

See [`docs/reproduction.md`](docs/reproduction.md) for the full protocol and [`full-pipeline/README.md`](full-pipeline/README.md) for deployment. The optional [HDF5 cache](docs/hdf5-cache.md) stores binary occupancy only and does not change the paper input encoding.

## Citation

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for technology-data attribution.
