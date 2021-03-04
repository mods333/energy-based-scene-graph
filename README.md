[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/energy-based-learning-for-scene-graph/scene-graph-generation-on-visual-genome)](https://paperswithcode.com/sota/scene-graph-generation-on-visual-genome?p=energy-based-learning-for-scene-graph)
# Energy-Based Learning for Scene Graph Generation
This repository contains the code for our paper [Energy-Based Learning for Scene Graph Generation](https://arxiv.org/abs/2103.02221) accepted at CVPR 2021.

## Envirioment setup
To setup the environment with all the required dependancies follow the steps in [Install.md](https://github.com/mods333/energy-based-scene-graph/blob/master/INSTALL.md). 
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

To train you own models you can obtain the weights for the pretrained detectron from [this repository](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch).

## Training for Energy Based Scene Graph Generation

```bash
python -m torch.distributed.launch --master_port 10001 --nproc_per_node=4 \
    tools/energy_joint_train_cd.py --config-file configs/e2e_relation_X_101_32_8_FPN_1x.yaml \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
    MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor \
    SOLVER.IMS_PER_BATCH 16  TEST.IMS_PER_BATCH 4 \
    DTYPE float16 SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 2000 \
    SOLVER.CHECKPOINT_PERIOD 2000 \
    GLOVE_DIR $GLOVE_DIR \
    MODEL.PRETRAINED_DETECTOR_CKPT $PRETRAINED_DETECTOR_PATH \
    OUTPUT_DIR $OUTPUT_DIR \
    SOLVER.BASE_LR 0.001 SAMPLER.LR 1.0 SAMPLER.ITERS 20 SAMPLER.VAR 0.001 SAMPLER.GRAD_CLIP 0.01 MODEL.DEV_RUN False
```

The above scripts trains a model using 4 GPUs. Here how to change the training behavior for various requirements.
1. **Scene Graph Genration Tasks**
    1. For PredCLS set \
     `MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True`
    2. For SGCLS set \
    `MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False`
    3. For SGDet set \
    `MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False`
2. **Changing scene graph prediction model** \
  Change `MODEL.ROI_RELATION_HEAD.PREDICTOR` to one of the available models
    - `VCTreePredictor`
    - `MotifPredictor`
    - `IMPPredictor`
    - `TransformerPredictor` (change SOLVER.BASE_LR to 0.001, SOLVER.SCHEDULE.TYPE to WarmupMultiStepLR, SOLVER.MAX_ITER to 16000, SOLVER.IMS_PER_BATCH to 16, SOLVER.STEPS to (10000, 16000). From [this repo](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch))
    - If you would like to implement your own scene graph prection model just add the implementaion to [maskrcnn_benchmark/modeling/roi_heads/relation_head/roi_relation_predictors.py](https://github.com/mods333/energy-based-scene-graph/blob/master/maskrcnn_benchmark/modeling/roi_heads/relation_head/roi_relation_predictors.py)
3. **Modifying Sampler** \
    Current implementation only has a single sampler (SGLD). You can implement samplers of your choice in [maskrcnn_benchmark/modeling/energy_head/sampler.py](https://github.com/mods333/energy-based-scene-graph/blob/master/maskrcnn_benchmark/modeling/energy_head/sampler.py). To change the parametes of the sampler use the fields under `SAMPLER` in the config.

## Acknowledgment
This repository is developed on top of the scene graph benchmarking framwork develped by [KaihuaTang](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch)
