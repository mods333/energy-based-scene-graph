# Energy-Based Learning for Scene Graph Generation
This repository contains the code for our paper Energy-Based Learning for Scene Graph Generation.

## Envirioment setup
To setup the environment with all the required dependancies run `create_env.sh`. 
\
**Note**: By default the `cudatoolkit` version is set to 10.0. When creating an environment on your machine check you cuda compiler version by running `nvcc --version` and adjust the `cudatoolkit` version appopriately. Version mismatches can lead to the `build` failing or `segmentaion fault` error when running the code.

## DATASET
The following is adapted from [Danfei Xu](https://github.com/danfeiX/scene-graph-TF-release/blob/master/data_tools/README.md) and [neural-motifs](https://github.com/rowanz/neural-motifs).

Note that our codebase intends to support attribute-head too, so our ```VG-SGG.h5``` and ```VG-SGG-dicts.json``` are different with their original versions in [Danfei Xu](https://github.com/danfeiX/scene-graph-TF-release/blob/master/data_tools/README.md) and [neural-motifs](https://github.com/rowanz/neural-motifs). We add attribute information and rename them to be ```VG-SGG-with-attri.h5``` and ```VG-SGG-dicts-with-attri.json```. The code we use to generate them is located at ```datasets/vg/generate_attribute_labels.py```. Although, we encourage later researchers to explore the value of attribute features, in our paper "Unbiased Scene Graph Generation from Biased Training", we follow the conventional setting to turn off the attribute head in both detector pretraining part and relationship prediction part for fair comparison, so does the default setting of this codebase.

### Download:
1. Download the VG images [part1 (9 Gb)](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip) [part2 (5 Gb)](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip). Extract these images to the file `datasets/vg/VG_100K`. If you want to use other directory, please link it in `DATASETS['VG_stanford_filtered']['img_dir']` of `maskrcnn_benchmark/config/paths_catelog.py`. 
2. Download the [scene graphs](https://onedrive.live.com/embed?cid=22376FFAD72C4B64&resid=22376FFAD72C4B64%21779871&authkey=AA33n7BRpB1xa3I) and extract them to `datasets/vg/VG-SGG-with-attri.h5`, or you can edit the path in `DATASETS['VG_stanford_filtered_with_attribute']['roidb_file']` of `maskrcnn_benchmark/config/paths_catalog.py`.

## Acknowledgment
This repository is developed on top of the scene graph benchmarking framwork develped by of [KaihuaTang](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch)
