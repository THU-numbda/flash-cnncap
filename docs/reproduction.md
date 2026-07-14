# Reproducing the accepted-paper results

## Fixed protocol

- input: binary per-layer occupancy; the coupling master is encoded with `-1`
- split: fixed 80/20 window-level split, seed `42`
- training seeds: `11037` for the main comparison and `11037`-`11041` for the ablation
- epochs: `100`
- optimizer: AdamW, learning rate `3e-4`, weight decay `1e-4`
- batch size: `16`
- schedule: five-epoch linear warmup followed by cosine decay to 10% of the base learning rate
- loss: mean squared relative error
- gradient clipping: maximum norm `1.0`
- selected model: `D4_D_k5`
- inference hardware: one NVIDIA GeForce RTX 5090

The paper's environment and dependency versions are pinned in the root README and `requirements.txt`.

## Datasets

Install the six CapBench subsets:

```bash
for selector in \
  nangate45/small nangate45/medium nangate45/large \
  sky130hd/small sky130hd/medium sky130hd/large
do
  python -m capbench datasets install "$selector"
done
```

The split is window-disjoint but not design-disjoint, matching the accepted paper.

## Architecture ablation

```bash
python scripts/ablations.py run \
  --dataset nangate45/small \
  --num-gpus 8 \
  --num-workers 1 \
  --epochs 100 \
  --repeats 5 \
  --seed 11037
```

This runs the 13 configurations reported in the paper. `python scripts/ablations.py list` prints their registered names.

## Main accuracy matrix

```bash
python scripts/best_model_eval.py run \
  --model D4_D_k5 \
  --model resnet34 \
  --model resnet50 \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --epochs 100 \
  --repeats 1 \
  --seed 11037
```

The released Flash-CNNCap checkpoints reproduce the selected-model rows:

| Dataset | PDK | Target | MARE | >5% | >10% |
| --- | --- | --- | ---: | ---: | ---: |
| CNN-Cap | legacy | total | 0.007180 | 0.009644 | 0.000000 |
| CNN-Cap | legacy | coupling | 0.027770 | 0.140140 | 0.042292 |
| Small | Nangate45 | total | 0.028677 | 0.109946 | 0.030063 |
| Small | Nangate45 | coupling | 0.033026 | 0.194993 | 0.039319 |
| Small | Sky130HD | total | 0.020509 | 0.076611 | 0.018437 |
| Small | Sky130HD | coupling | 0.036328 | 0.219390 | 0.056606 |
| Medium | Nangate45 | total | 0.017157 | 0.051802 | 0.009639 |
| Medium | Nangate45 | coupling | 0.033331 | 0.213767 | 0.040460 |
| Medium | Sky130HD | total | 0.014582 | 0.040450 | 0.006630 |
| Medium | Sky130HD | coupling | 0.030242 | 0.183963 | 0.032332 |
| Large | Nangate45 | total | 0.029380 | 0.154556 | 0.042010 |
| Large | Nangate45 | coupling | 0.045550 | 0.314262 | 0.090010 |
| Large | Sky130HD | total | 0.030994 | 0.171157 | 0.050576 |
| Large | Sky130HD | coupling | 0.044690 | 0.314317 | 0.097076 |

## Runtime

`scripts/benchmark_window_model_runtime.py` measures full-matrix inference on 256 windows. The paper reports 4.7 s for D4_D_k5 versus 82.2 s for ResNet-34 on large windows, a 17.5x speedup on the same RTX 5090.

The end-to-end 1,024-window pipeline command is documented in `full-pipeline/README.md`. Preserve the same hardware, TensorRT version, engine batch size `24`, and pre-windowed inputs when comparing the reported 51.23 s runtime.
