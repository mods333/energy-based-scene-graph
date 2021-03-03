# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT
import time

import torch
from torch_scatter import scatter

from maskrcnn_benchmark.modeling.energy_head.graph import Graph
from maskrcnn_benchmark.modeling.roi_heads.relation_head.utils_motifs import (
    encode_box_info, to_onehot)

import logging
logger = logging.getLogger(__name__)

def normalize_states(states):
    states = states - torch.min(states, dim=-1, keepdim=True)[0]
    states = states/torch.max(states, dim=1, keepdim=True)[0]
    return states

def get_predicted_sg(detections, num_obj_classes, mode, noise_var):
    '''
    This function converts the detction in scene grpah strucuter 
    Parameters:
    -----------
        detection: A tuple of (relation_logits, object_logits, rel_pair_idxs, proposals)
    '''
    offset = 0
    pair_list = []
    batch_list = []
    edge_batch_list = []

    ################################################################################################
    rel_list = torch.cat(detections[0], dim= 0)
    rel_list = normalize_states(rel_list)
    # rel_list = (rel_list - torch.min(rel_list, dim=-1, keepdim=True)[0])
    # rel_list = rel_list/torch.max(rel_list, dim=1, keepdim=True)[0]
    # if detach:
    #     rel_list.detach()

    node_list = torch.cat(detections[1], dim= 0)
    if mode == 'predcls':
        #Add small noise to the input
        node_noise = torch.rand_like(node_list).normal_(0, noise_var)
        node_list.data.add_(node_noise)
    else:
        node_list = normalize_states(node_list)

    # if detach:
    #     node_list.detach()
    ################################################################################################

    for i in range(len(detections[0])):
        pair_list.append(detections[2][i] + offset)
        batch_list.append(torch.full((detections[1][i].shape[0], ) , i, dtype=torch.long))
        edge_batch_list.append(torch.full( (detections[0][i].shape[0], ), i, dtype=torch.long))
        offset += detections[1][i].shape[0]
    
    pair_list = torch.cat(pair_list, dim=0)
    batch_list = torch.cat(batch_list, dim=0).to(node_list.device)
    edge_batch_list = torch.cat(edge_batch_list, dim=0).to(node_list.device)

    return node_list, rel_list, pair_list, batch_list, edge_batch_list

def get_gt_scene_graph(targets, num_obj_classes, num_rel_classes, noise_var):
    '''
    Converts gorund truth annotations into graph structure
    '''
    offset = 0
    pair_list = []
    node_list = []
    rel_list = []
    batch_list = []
    edge_batch_list = []

    for i, target in enumerate(targets):
        rel = target.get_field('relation')
        rel_pair_idxs = torch.nonzero(rel)
        pair_list.append(rel_pair_idxs + offset)
        
        node_list.append(target.get_field('labels'))
        rel_list.append(rel[rel_pair_idxs[:,0], rel_pair_idxs[:,1]])

        batch_list.extend([i]*len(target))
        # import ipdb; ipdb.set_trace()
        edge_batch_list.extend([i]*len(pair_list[-1]))
        offset += len(target)

    node_list = to_onehot(torch.cat(node_list, dim=0), num_obj_classes)
    node_noise = torch.rand_like(node_list).normal_(0, noise_var)
    node_list.data.add_(node_noise)

    rel_list = to_onehot(torch.cat(rel_list, dim=0), num_rel_classes)
    rel_noise = torch.rand_like(rel_list).normal_(0, noise_var)
    rel_list.data.add_(rel_noise)
    batch_list = torch.tensor(batch_list).to(node_list.device)
    pair_list = torch.cat(pair_list, dim=0)
    edge_batch_list = torch.tensor(edge_batch_list).to(node_list.device)
    
    # adj_matrix = torch.zeros(size=(node_list.shape[0], node_list.shape[0])).to(node_list.device)
    # adj_matrix[pair_list[:,0], pair_list[:,1]] = 1

    # rel_list = torch.sparse.FloatTensor(pair_list.t(), rel_list, torch.Size([adj_matrix.shape[0], adj_matrix.shape[0], rel_list.shape[-1]])).to_dense()
    
    return node_list, rel_list,  pair_list, batch_list, edge_batch_list

def get_gt_im_graph(node_states, images, detections, base_model, noise_var):
    #Extract region feature from the target bbox
    # import ipdb; ipdb.set_trace()
    if node_states is None:
        features = base_model.backbone(images.tensors)
        node_states = base_model.roi_heads.relation.box_feature_extractor(features, detections)

    node_noise = torch.rand_like(node_states).normal_(0, noise_var)
    node_states.data.add_(node_noise)

    return node_states

def get_pred_im_graph(node_states, images, detections, base_model, noise_var, detach=True):
    #Extract region feature from the predictions
    if node_states is None:
        features = base_model.backbone(images.tensors)
        node_states = base_model.roi_heads.relation.box_feature_extractor(features, detections[-1])

    node_noise = torch.rand_like(node_states).normal_(0, noise_var)
    node_states.data.add_(node_noise)
    if detach:
        node_states.detach()

    return node_states

def detection2graph(node_states, images, detections, base_model, num_obj_classes, mode, noise_var):

    '''
    Create image graph and scene graph given the detections
    Parameters:
    ----------
        images: Batch of input images
        detection: A tuple of (relation_logits, object_logits, rel_pair_idxs, proposals)
        base_model: realtion predcition model (Used of extracting features)
        num_obj_classes: Number of object classes in the dataset(Used for ocnvertin to one hot encoding)
    Return:
    ----------
        im_graph: A graph corresponding to the image
        scene_graph: A graph corresponding to the scene graph
    '''
    #Scene graph Creation
    
    sg_node_states, sg_rel_states, adj_matrix, batch_list, edge_batch_list = get_predicted_sg(detections, num_obj_classes, mode, noise_var)
        
    #Iage graph generation
    im_node_states = get_pred_im_graph(node_states, images, detections, base_model, noise_var)
    
    scene_graph = Graph(sg_node_states, adj_matrix, batch_list, sg_rel_states, edge_batch_list)
    im_graph = Graph(im_node_states, adj_matrix, batch_list)

    return im_graph, scene_graph, encode_box_info(detections[-1])

def gt2graph(node_states, images, targets, base_model, num_obj_classes, num_rel_classes, noise_var):

    '''
    Create image graph and scene graph given the detections
    Parameters:
    ----------
        images: Batch of input images
        target: Gt Target
        base_model: realtion predcition model (Used of extracting features)
        num_obj_classes: Number of object classes in the dataset(Used for ocnvertin to one hot encoding)
    Return:
    ----------
        im_graph: A graph corresponding to the image
        scene_graph: A graph corresponding to the scene graph
    '''

    sg_node_states, sg_edge_states, adj_matrix, batch_list, edge_batch_list = get_gt_scene_graph(targets, num_obj_classes, num_rel_classes, noise_var)

    im_node_states = get_gt_im_graph(node_states, images, targets, base_model, noise_var)

    sg_graph = Graph(sg_node_states, adj_matrix, batch_list, sg_edge_states, edge_batch_list)
    im_graph = Graph(im_node_states, adj_matrix, batch_list)
    
    return im_graph, sg_graph, encode_box_info(targets),
