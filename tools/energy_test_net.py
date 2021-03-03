# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
import time
import datetime

import torch
from torch.nn.utils import clip_grad_norm_
import wandb

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader, get_dataset_statistics
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.engine.trainer import reduce_loss_dict
from maskrcnn_benchmark.engine.inference import inference, energy_inference
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.modeling.energy_head import build_energy_model
from maskrcnn_benchmark.modeling.energy_head import detection2graph, gt2graph
from maskrcnn_benchmark.modeling.energy_head import build_loss_function, build_sampler
from maskrcnn_benchmark.utils.checkpoint import EBMCheckpointer
from maskrcnn_benchmark.utils.checkpoint import clip_grad_norm
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank, all_gather
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger, debug_print
from maskrcnn_benchmark.utils.miscellaneous import mkdir, save_config, get_mode
from maskrcnn_benchmark.utils.metric_logger import MetricLogger


# Check if we can enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for mixed precision via apex.amp')


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="/private/home/fmassa/github/detectron.pytorch_v2/configs/e2e_faster_rcnn_R_50_C4_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    
        
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # #Wandb Setup
    if get_rank() == 0:
        if cfg.MODEL.DEV_RUN or cfg.WANDB.MUTE:
            os.environ['WANDB_MODE'] = 'dryrun'

        wandb.init(project="sgebm")

    cfg.DATASETS.NUM_OBJ_CLASSES = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
    cfg.DATASETS.NUM_REL_CLASSES = cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

    cfg.freeze()

    save_dir = ""
    logger = setup_logger("maskrcnn_benchmark", save_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    base_model = build_detection_model(cfg)
    base_model.to(cfg.MODEL.DEVICE)

    energy_model = build_energy_model(cfg, base_model.roi_heads.relation.box_feature_extractor.out_channels)
    energy_model.to(cfg.MODEL.DEVICE)

    sampler = build_sampler(cfg)

    # Initialize mixed-precision if necessary
    use_mixed_precision = cfg.DTYPE == 'float16'
    amp_handle = amp.init(enabled=use_mixed_precision, verbose=cfg.AMP_VERBOSE)

    output_dir = cfg.OUTPUT_DIR

    checkpointer = EBMCheckpointer(
        cfg=cfg, base_model=base_model, energy_model=energy_model, 
        base_optimizer=None, energy_optimizer=None, 
        base_scheduler=None, energy_scheduler=None, 
        save_dir=output_dir
    )

    _ = checkpointer.load(cfg.MODEL.WEIGHT)

    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    if cfg.MODEL.RELATION_ON:
        iou_types = iou_types + ("relations", )
    if cfg.MODEL.ATTRIBUTE_ON:
        iou_types = iou_types + ("attributes", )
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST

    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    
    data_loaders_val = make_data_loader(cfg, mode="test", is_distributed=distributed)

    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        inference(
            cfg,
            base_model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
        )
        # energy_inference(
        #     cfg,
        #     base_model,
        #     energy_model,
        #     sampler,
        #     data_loader_val,
        #     dataset_name=dataset_name,
        #     with_sample=False,
        #     iou_types=iou_types,
        #     box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
        #     device=cfg.MODEL.DEVICE,
        #     expected_results=cfg.TEST.EXPECTED_RESULTS,
        #     expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
        #     output_folder=output_folder,
        #     logger=logger,
        # )
        
        synchronize()


if __name__ == "__main__":
    main()
