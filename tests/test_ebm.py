from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import datetime
import json
import logging
import os
import time

import torch
import wandb
from torch.nn.utils import clip_grad_norm_
from torchvision.utils import save_image

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import get_dataset_statistics, make_data_loader
from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from maskrcnn_benchmark.engine.inference import energy_inference, inference, _accumulate_predictions_from_multiple_gpus
from maskrcnn_benchmark.engine.trainer import reduce_loss_dict
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.modeling.energy_head import (build_energy_model,
                                                     build_loss_function,
                                                     build_sampler,
                                                     detection2graph, gt2graph)
from maskrcnn_benchmark.solver import make_lr_scheduler, make_optimizer
from maskrcnn_benchmark.utils.checkpoint import EBMCheckpointer, clip_grad_norm
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import all_gather, get_rank, synchronize
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import debug_print, setup_logger
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.utils.miscellaneous import get_mode, mkdir, save_config

logger = logging.getLogger(__name__)
# See if we can use apex.DistributedDataParallel instead of the torch default,
# and enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for multi-precision via apex.amp')

out_log = f = open("ouput.txt", "w")

def get_info(groundtruth, prediction, vocab_file):
    # boxes
    boxes = groundtruth.bbox
    # object labels
    idx2label = vocab_file['idx_to_label']
    labels = ['{}-{}'.format(idx,idx2label[str(i)]) for idx, i in enumerate(groundtruth.get_field('labels').tolist())]
    pred_labels = ['{}-{}'.format(idx,idx2label[str(i)]) for idx, i in enumerate(prediction.get_field('pred_labels').tolist())]
    # groundtruth relation triplet
    idx2pred = vocab_file['idx_to_predicate']
    gt_rel_idx = groundtruth.get_field('relation').nonzero()
    gt_rels = groundtruth.get_field('relation')[groundtruth.get_field('relation')!=0]
    gt_rels = torch.cat((gt_rels[:, None], gt_rel_idx), dim=1)

    gt_rels = [(labels[i[1].item()], idx2pred[str(i[0].item())], labels[i[2].item()]) for i in gt_rels]
    # prediction relation triplet
    pred_rel_pair = prediction.get_field('rel_pair_idxs').tolist()
    pred_rel_label = prediction.get_field('pred_rel_scores')
    pred_rel_label[:,0] = 0
    pred_rel_score, pred_rel_label = pred_rel_label.max(-1)
    #mask = pred_rel_score > thres
    #pred_rel_score = pred_rel_score[mask]
    #pred_rel_label = pred_rel_label[mask]
    #import ipdb; ipdb.set_trace()
    pred_rels = [(pred_labels[i[0]], idx2pred[str(j)], pred_labels[i[1]]) for i, j in zip(pred_rel_pair, pred_rel_label.tolist())]
    return boxes, labels, gt_rels, pred_rels, pred_rel_score, pred_rel_label

def print_list(name, input_list):
    for i, item in enumerate(input_list):
        out_log.write(name + ' ' + str(i) + ': ' + str(item))
        out_log.write('\n')

def draw_image(boxes, labels, gt_rels, pred_rels, pred_rel_score, pred_rel_label, print_img=True):
    num_obj = boxes.shape[0]

    if print_img:
        out_log.write('*' * 50)
        out_log.write('\n')
        print_list('gt_boxes', labels)
        out_log.write('\n')
        out_log.write('*' * 50)
        out_log.write('\n')
        print_list('gt_rels', gt_rels)
        out_log.write('\n')
        out_log.write('*' * 50)
        out_log.write('\n')
    print_list('pred_rels', pred_rels[:20])
    out_log.write('\n')
    out_log.write('*' * 50)
    out_log.write('\n')

def draw_image_pre(boxes, labels, gt_rels, pred_rels, pred_rel_score, pred_rel_label, print_img=True):
    num_obj = boxes.shape[0]
    out_log.write('>>>>>>>>>>>>>>>>>>>>>Pre Refinement <<<<<<<<<<<<<<<<<\n')
    print_list('pred_rels', pred_rels[:20])
    out_log.write('\n')
    out_log.write('*' * 50)
    out_log.write('\n')

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

    statistics = get_dataset_statistics(cfg)
    cfg.DATASETS.NUM_OBJ_CLASSES = len(statistics['obj_classes'])
    cfg.DATASETS.NUM_REL_CLASSES = len(statistics['rel_classes'])

    cfg.freeze()

    save_dir = ""
    logger = setup_logger("maskrcnn_benchmark", save_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())
   
    device = torch.device(cfg.MODEL.DEVICE)
    cpu_device = torch.device("cpu")

    base_model = build_detection_model(cfg)
    base_model.to(cfg.MODEL.DEVICE)

    energy_model = build_energy_model(cfg, base_model.roi_heads.relation.box_feature_extractor.out_channels)
    energy_model.to(cfg.MODEL.DEVICE)

    sampler = build_sampler(cfg)
    mode = get_mode(cfg)

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

    vocab_file = json.load(open('/home/ubuntu/Scene-Graph-Benchmark.pytorch/datasets/vg/VG-SGG-dicts-with-attri.json'))

    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    
    data_loaders_val = make_data_loader(cfg, mode="test", is_distributed=distributed)[0]
    

    start_iter = 1
    results_dict1 = {}
    results_dict2 = {}

    extra_args = dict(
        box_only=False,
        iou_types=iou_types,
        expected_results=cfg.TEST.EXPECTED_RESULTS,
        expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
    )
    for iteration, (images, targets, image_ids) in enumerate(data_loaders_val, start_iter):

        if iteration < 5: 
            continue

        base_model.eval()
        energy_model.eval()

        images = images.to(device)
        targets = [target.to(device) for target in targets]

        #################################################################################################################################
        # Random Graph Initialization
        detections = base_model(images, targets)
        edge_init = tuple([torch.rand_like(detections[0][i]).uniform_(-10,2) for i in range(len(detections[0]))])
        node_init = detections[1]
        pair_idx = detections[2]
        proposals = detections[3]

        init_states = (edge_init, node_init, pair_idx, proposals)
        #################################################################################################################################

        pre_detections = energy_model.post_processor((edge_init, node_init), pair_idx, proposals)
        _, classes = pre_detections[0].get_field('pred_rel_scores').max(-1)
        # classes = [vocab_file['idx_to_predicate'][str(j)] for j in  classes.tolist()]
        print(classes)
        groundtruth = targets[0]
        draw_image_pre(*get_info(groundtruth, pre_detections[0], vocab_file))

        results_dict1.update(
                {img_id: result for img_id, result in zip(image_ids, pre_detections)}
            )
        results_dict1 = _accumulate_predictions_from_multiple_gpus(results_dict1, synchronize_gather=cfg.TEST.RELATION.SYNC_GATHER)

        # import ipdb; ipdb.set_trace()
        evaluate(cfg=cfg,
                dataset=data_loaders_val.dataset,
                predictions=results_dict1,
                output_folder=output_folder,
                logger=logger,
                **extra_args)
        #################################################################################################################################
        # Energy Computation for gt graph and random graph
        pred_im_graph, pred_scene_graph, pred_bbox = detection2graph(images, init_states, base_model, cfg.DATASETS.NUM_OBJ_CLASSES, mode)
        gt_im_graph, gt_scene_graph, gt_bbox = gt2graph(images, targets, base_model, cfg.DATASETS.NUM_OBJ_CLASSES, cfg.DATASETS.NUM_REL_CLASSES)

        gt_energy = energy_model(gt_im_graph, gt_scene_graph, gt_bbox)
        print(">>>>>>>>>>> GT Energy {}".format(gt_energy.item()))
        energy_1 = energy_model(pred_im_graph, pred_scene_graph, pred_bbox)
        print(">>>>>>>>>>> Pre Energy {}".format(energy_1.item()))
        #################################################################################################################################
        
        #Sampling stage
        pred_scene_graph = sampler.sample(energy_model, pred_im_graph, pred_scene_graph, pred_bbox, mode)

        #################################################################################################################################
        #Updating Graph State
        if mode == 'predcls':
            num_rels = [r.shape[0] for r in detections[0]]
            relation_logits = pred_scene_graph.edge_states.split(num_rels)
            object_logits = detections[1]

        else:
            num_rels = [r.shape[0] for r in detections[0]]
            num_objs = [o.shape[0] for o in detections[1]]
            relation_logits = pred_scene_graph.edge_states.split(num_rels)
            object_logits = pred_scene_graph.node_states.split(num_objs)

        init_states = (relation_logits, object_logits, detections[2], detections[3])

        #################################################################################################################################
        #Calcualting energy after refinement
        pred_im_graph, pred_scene_graph, pred_bbox = detection2graph(images, init_states, base_model, cfg.DATASETS.NUM_OBJ_CLASSES, mode)        
        energy_2 = energy_model(pred_im_graph, pred_scene_graph, pred_bbox)
        print('>>>>>> Post Energy {}'.format(energy_2.item()))

        #################################################################################################################################
        
        #################################################################################################################################
        #Post Processing
        output = energy_model.post_processor((relation_logits, object_logits), detections[2], detections[3])
        output = [o.to(cpu_device) for o in output]
        results_dict2.update(
                {img_id: result for img_id, result in zip(image_ids, output)}
            )
        results_dict2 = _accumulate_predictions_from_multiple_gpus(results_dict2, synchronize_gather=cfg.TEST.RELATION.SYNC_GATHER)

        evaluate(cfg=cfg,
                dataset=data_loaders_val.dataset,
                predictions=results_dict2,
                output_folder=output_folder,
                logger=logger,
                **extra_args)
        #################################################################################################################################

        #Checking classes of detection
        _, classes_new = output[0].get_field('pred_rel_scores').max(-1)
        # classes_new = [vocab_file['idx_to_predicate'][str(j)] for j in  classes_new.tolist()]
        print(classes_new)

        #################################################################################################################################
        #Writing to File
        prediction = output[0]
        draw_image(*get_info(groundtruth, prediction, vocab_file))
        save_image(images.tensors.to(cpu_device)[0], 'img1.png')
        #################################################################################################################################
        break
    
    out_log.close()

if __name__ == "__main__":
    main()
