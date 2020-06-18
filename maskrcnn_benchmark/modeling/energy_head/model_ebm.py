import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.make_layers import make_fc
from maskrcnn_benchmark.modeling.utils import cat

from .layers import EGNNLayer
from .pooling import GatedPooling


class GraphEnergyModel(nn.Module):
    '''
    Energy Model class
    This class takes as input two graphs
    The first graph corresponds to the features extracted from the images
    The second graph corresponds to an intialization of the scene graph
    The goal is the leans an energy functin such that the enegry is low for scene graph that match the image
    Parameters:
    -----------
        config: Experiment configuration
        obj_classes: The object classes in the dataset
        rel_classes: All the realtion classes in the dataset
        obj_feature_dim: Dimension to which the object features will be embedded
        rel_feature_dim: Dimension to which the relation features will be embedded
        in_channels: Dimension for features extracted form the detector
    '''

    def __init__(self, config, obj_classes, rel_classes, in_channels):

        super(GraphEnergyModel, self).__init__()
        self.config = config

        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.num_obj_classes = len(obj_classes)
        self.num_rel_classes = len(rel_classes)

        self.obj_embed_dim = self.config.MODEL.ROI_RELATION_HEAD.EBM.OBJ_EMBED_DIM
        self.rel_embed_dim = self.config.MODEL.ROI_RELATION_HEAD.EBM.REL_EMBED_DIM

        self.obj_label_embed_dim = self.config.MODEL.ROI_RELATION_HEAD.EBM.OBJ_EMBED_DIM
        self.rel_label_embed_dim = self.config.MODEL.ROI_RELATION_HEAD.EBM.REL_EMBED_DIM

        self.pooling_dim = self.config.MODEL.ROI_RELATION_HEAD.EBM.POOLING_DIM
        # Obtain the generatin model based on availabel indformation
        if self.config.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.config.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'

        ##########################################################################################
        #Embedding layers
        #These layers are required we want to incorporate the bounding box information 
        #Size of the embedding vector
        self.obj_dim = in_channels
        self.rel_dim = in_channels

        #Embedding for the features extracted from detector
        self.obj_emdedding = nn.Linear(self.obj_dim, self.obj_embed_dim)
        self.rel_embeding = nn.Linear(self.rel_dim, self.rel_embed_dim)

        #Embedding for the scene graph representaions
        self.obj_label_embedding = nn.Linear(self.num_obj_classes + 128, self.obj_label_embed_dim) #128 is for the positional encoding which will be appended to the object lables
        self.rel_label_embedding = nn.Linear(self.num_rel_classes, self.rel_label_embed_dim)

        # position embedding
        self.pos_embed = nn.Sequential(*[
            nn.Linear(9, 32), nn.BatchNorm1d(32, momentum= 0.001),
            nn.Linear(32, 128), nn.ReLU(inplace=True),
        ])
        ##########################################################################################

        self.sg_layer = EGNNLayer(self.obj_label_embed_dim, self.rel_label_embed_dim)
        self.im_layer = EGNNLayer(self.obj_embed_dim, self.rel_embed_dim)

        self.sg_pooler = GatedPooling(self.obj_label_embed_dim, self.rel_label_embed_dim, self.pooling_dim)
        self.im_pooler = GatedPooling(self.obj_embed_dim, self.rel_embed_dim, self.pooling_dim)

        self.energy = nn.Sequential(
            nn.Linear(self.pooling_dim*2, self.pooling_dim),
            nn.ReLU(), 
            nn.Linear(self.pooling_dim, 1)
        )
    def get_contiguous_rel_pair_idx(self, rel_pair_idxs, proposals):
        '''
        This function converts the list of rel_pair_idxs into a single tensor 
        For example: if we have a batch with two images and the first images has two objects and the second image has two object then
        an exampele rel_pair_idxs would be 
            [ [[0,1],[1, 0]] , 
              [[0,1],[1,0]]]
        This funciont will convert it such that the 
         [
             [0,1], [1,0], [2,3], [3,2]
         ]
         This will help us define a single matrix that consist of all the objects and relations from all the images in the batch.
         We can then make sure that information does not flow between nodes/edges belonging to different images by make the adjacency matrix accordingly
        '''
        offset = 0
        pair_list = []
        for i, proposal in enumerate(proposals): 
            pair_list.append(rel_pair_idxs[i] + offset)
            offset += len(proposal)
        
        return torch.cat(pair_list, dim=0)

    def forward(self, x, proposals, union_features, rel_pair_idxs, obj_labels, rel_labels):

        #Embedding the bounding boxes
        assert proposals[0].mode == 'xyxy'
        pos_embed = self.pos_embed(encode_box_info(proposals))
        
        #Adjacency matrix
        rel_pair_idx = self.get_contiguous_rel_pair_idx(rel_pair_idxs, proposals)
        
        adj_matrix = torch.zeros(size=(obj_pre_rep.shape[0], obj_pre_rep.shape[0])).to(x.device)
        adj_matrix[rel_pair_idx[:,0], rel_pair_idx[:,1]] = 1

        #Obtain the states for the image graph
        im_node_states = self.obj_emdedding(x)
        im_edge_states = self.edge_embeding(union_features)
        im_edge_states = torch.sparse.FloatTensor(rel_pair_idx.t(), im_edge_states, torch.Size([adj_matrix.shape[0], adj_matrix.shape[0], im_edge_states.shape[-1]])).to_dense()

        #Obtain the states of the scene graph
        sg_node_states = self.obj_label_embedding(cat((obj_labels, pos_embed), -1))
        sg_edge_states = self.rel_label_embedding(rel_labels)

        #Refine the states of the image graph
        im_node_states, im_edge_states = self.im_layer(im_node_states, im_edge_states)
        #Refine the states of the scene graph
        sg_node_states, sg_edge_states = self.sg_layer(sg_node_states, sg_edge_states)
        
        #Pooling the states
        im_pooled = self.im_pooler(im_node_states, im_edge_states)
        sg_pooled = self.sg_pooler(sg_node_states, sg_edge_states)

        energy = self.energy(cat((im_pooled, sg_pooled), -1))

        return energy
