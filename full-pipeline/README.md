# DEF-to-SPEF pipeline

The deployment path reads routed DEF geometry, stages fixed-size model windows, performs TensorRT inference and sparse conductor reduction on CUDA, merges tiled results, and writes SPEF.

## Requirements

- CUDA-capable NVIDIA GPU
- the NVIDIA PyTorch 26.02 container described in the root README
- TensorRT `10.15.1.29` and ONNX `1.18.0`
- one total-capacitance and one coupling checkpoint for the same PDK/window size

TensorRT engines are specific to the GPU/CUDA/TensorRT stack. The release distributes portable PyTorch checkpoints, not serialized engines.

## Compile models

```bash
python full-pipeline/compile_models.py \
  --out-dir artifacts/compiled-models \
  --tech tech/nangate45.yaml \
  --total-checkpoint models/flash-cnncap-d4-d-k5-nangate45-large-total.pth \
  --env-checkpoint models/flash-cnncap-d4-d-k5-nangate45-large-coupling.pth
```

This writes FP16 TensorRT engines plus `compiled_models.json`, which records the active layer ordering and model contract.

## Full-layout inference

```bash
python full-pipeline/run.py run \
  --def /path/to/block.def \
  --out-spef artifacts/full-pipeline-spef/block.spef \
  --tech tech/nangate45.yaml \
  --compiled-model-manifest artifacts/compiled-models/compiled_models.json \
  --total-compiled-model artifacts/compiled-models/total_qmap.engine \
  --env-compiled-model artifacts/compiled-models/env_qmap.engine
```

The full-layout runner partitions the die into owned patches, adds model context, deduplicates coupling queries by final net, reduces predictions over owned fragments, and merges directed coupling estimates into unordered SPEF pairs.

## Paper's 1,024-window benchmark

The accepted paper's end-to-end comparison uses pre-windowed Nangate45 large DEF inputs and `run_multi.py`:

```bash
python full-pipeline/run_multi.py \
  --def-dir /path/to/nangate45-large/def \
  --out-spef-dir artifacts/paper-spef \
  --tech tech/nangate45.yaml \
  --total-compiled-model artifacts/compiled-models/total_qmap.engine \
  --env-compiled-model artifacts/compiled-models/env_qmap.engine \
  --max-windows 1024
```

Run `python full-pipeline/run_multi.py --help` for batching and output controls. The paper reports 51.23 s total time on one RTX 5090, including 1.31 s initialization, 0.12 s DEF parsing, 36.06 s GPU processing, and 13.75 s SPEF writing.

## Technology support

The release includes the Nangate45 and Sky130HD stack YAML files used by the paper. Standard-cell geometry needed by the native parser is compiled into `native/lefdef_compiled_cell_recipes*.h`; provenance is documented in the repository's third-party notices.
