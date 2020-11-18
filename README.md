# Energy-Based Learning for Scene Graph Generation
This repository contains the code for our paper Energy-Based Learning for Scene Graph Generation.

## Envirioment setup
To setup the environment with all the required dependancies run `create_env.sh`. 
\
**Note**: By default the `cudatoolkit` version is set to 10.0. When creating an environment on your machine check you cuda compiler version by running `nvcc --version` and adjust the `cudatoolkit` version appopriately. Version mismatches can lead to the `build` failing or `segmentaion fault` error when running the code.

## DATASET
Check Dataset.md for details on downloading the datasets.

## Acknowledgment
This repository is developed on top of the scene graph benchmarking framwork develped by of [KaihuaTang](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch)
