# Third-party notices

Flash-CNNCap is licensed under Apache-2.0. The repository also contains generated or adapted data derived from the following open technology resources.

## Nangate45

`full-pipeline/native/lefdef_compiled_cell_recipes.h` is generated from Nangate45 LEF geometry. The corresponding OpenROAD-flow-scripts Nangate45 platform is distributed under the Apache License 2.0:

- [OpenROAD-flow-scripts Nangate45 platform](https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts/tree/master/flow/platforms/nangate45)
- [Nangate45 platform license](https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts/blob/master/flow/platforms/nangate45/LICENSE)

The generated file has been transformed into C++ lookup tables for Flash-CNNCap's native DEF parser.

## SkyWater SKY130

`full-pipeline/native/lefdef_compiled_cell_recipes_sky130hd.h` is generated from SKY130 HD LEF geometry. The SkyWater open-source PDK is distributed under the Apache License 2.0:

- [SkyWater open-source PDK](https://github.com/google/skywater-pdk)
- [SkyWater PDK license](https://github.com/google/skywater-pdk/blob/main/LICENSE)

The generated file has been transformed into C++ lookup tables for Flash-CNNCap's native DEF parser.

## CapBench

Dataset discovery, shared loaders, and the pinned technology YAML source are provided by CapBench:

- [CapBench](https://github.com/THU-numbda/CapBench)
- paper-run revision: `33939cb5002575c06f19c237e2eae6105e331634`

The two YAML files in `tech/` are preserved from that revision to make the released deployment commands self-contained.

## Python and CUDA dependencies

Python, CUDA, TensorRT, MONAI, PyTorch, and other dependencies retain their respective upstream licenses. See `requirements.txt` and the NVIDIA PyTorch container documentation for exact versions.
