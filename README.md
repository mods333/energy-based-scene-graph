# Energy-Based Learning for Scene Graph Generation
This repository contains the code for our paper Energy-Based Learning for Scene Graph Generation.

## Envirioment setup
To setup the environment with all the required dependancies run `create_env.sh`. 
\
**Note**: By default the `cudatoolkit` version is set to 10.0. When creating an environment on your machine check you cuda compiler version by running `nvcc --version` and adjust the `cudatoolkit` version appopriately. Version mismatches can lead to the `build` failing or `segmentaion fault` error when running the code.

## DATASET
Check [Dataset.md](https://github.com/mods333/energy-based-scene-graph/blob/master/DATASET.md) for details on downloading the datasets.

## Pre-Trained Models

We realsed the weights for the pretained VCTree model on the Visual Genome dataset trained using both cross-entropy based and energy-based training.

| EBM                | CE                 |
|--------------------|--------------------|
| [VCTree-Predcls](https://tinyurl.com/vctree-ebm-predcls) | [VCTree-PredCLS](https://tinyurl.com/yxpt4n7w) |
| [VCTree-SGCLS](https://tinyurl.com/vctree-ebm-sgcls)   | [VCTree-SGCLS](https://tinyurl.com/vctree-ce-sgcls)   |
| [VCTree-SGDET](https://tinyurl.com/vctree-ebm-sgdet)   | [VCTree-SGDET](https://tinyurl.com/vctree-ce-sgdet)   |



## Acknowledgment
This repository is developed on top of the scene graph benchmarking framwork develped by [KaihuaTang](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch)
