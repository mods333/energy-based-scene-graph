from maskrcnn_benchmark.modeling import registry
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence
from torch.nn import functional as F
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.modeling.make_layers import make_fc

class EGNNLayer(nn.Module):
    '''
    Graph Layer to apply edged graph convolution on the specified graphs
    '''

    def __init__(self, node_dim, edge_dim, iters=3):
        
        super(EGNNLayer, self).__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.iters = iters

        ################################
        #Define the kernels

        self.kernel_activation = nn.ReLU()
        self.node2node_kernel = nn.Sequential(
            nn.Linear(self.node_dim, self.node_dim),
            self.kernel_activation,
            nn.Linear(self.node_dim, self.node_dim),
            self.kernel_activation
            )
        
        self.edge2node_kernel = nn.Sequential(
            nn.Linear(self.edge_dim, self.node_dim),
            self.kernel_activation, 
            self.nn.Linear(self.node_dim, self.node_dim),
            self.kernel_activation
        )

        self.node2edge_kernel = nn.Sequential(
            nn.Conv2d(2*self.node_dim, self.edge_dim, kernel_size=1, stride=1),
            self.kernel_activation, 
            nn.Conv2d(self.edge_dim, self.edge_dim, kernel_size=1, stride=1),
            self.kernel_activation, 
        )

    def node2node_mp(self, node_states, adj_matrix):
        '''
        Function to aggreagate message from all the neighbouring nodes in the graph
        '''
        node2node_messages = torch.mm(adj_matrix, node_states)
        #Message normalization
        node2node_messages = node2node_messages/(torch.sum(adj_matrix, dim=1, keepdim=True) + 1e-6)
        return node2node_messages
    
    def edge2node_mp(self, edge_states, adj_matrix):
        '''
        FUnction to aggregate messages form incoming edges to nodes 
        '''
        edge2node_messages = (adj_matrix[:,:,None] * edge_states).sum(dim=0)
        return edge2node_messages

    def node2edge_mp(self, node_states, adj_matrix):
        '''
        Funciton to pass message form nodes to edges that connect the nodes
        (The concatination should be direction aware) 
        '''
        n = node_states.shape[0]
        ek_input = torch.cat([node_states.repeat(1,n).view(n*n, -1), node_states.repeat(n,1)], dim=1).view(n,n, 2*self.node_dim) * adj_matrix[:,:,None]
        ek_input = ek_input.transpose(0,2).unsqueeze(0)
        node2edge_messages = self.node2edge_kernel(ek_input).squeeze(0).transpose(0,2)
        return node2edge_messages  
    
    def node_update(self, node_states, node2node_messages, edge2node_messages):
        return self.node_gate(self.alpha*node2node_messages + (1-self.alpha)*edge2node_messages, node_states)

    def edge_update(self, edge_states, node2edge_messages):
        m = node2edge_messages.shape[0]
        edge_states = self.edge_gate(node2edge_messages.reshape(m*m, -1), edge_states.reshape(m*m, -1)).view(m,m,-1)
        return edge_states

    def forward(self, node_states, edge_states, adj_matrix):

        for _ in range(self.iters):

            #Aggregate node to node infromations
            node2node_messages = self.node2node_mp(node_states, adj_matrix)
            #Aggregate edge to node informations
            edge2node_messages = self.edge2node_mp(edge_states, adj_matrix)
            #Aggregate node to edge infromations
            node2edge_messages = self.node2edge_mp(node_states, adj_matrix)

            #Apply kernels to the recieves messages
            node2node_messages = self.node2node_kernel(node2node_messages)
            edge2node_messages = self.edge2node_kernel(edge2node_messages)

            node_states = self.node_update(node_states, node2node_messages, edge2node_messages)
            edge_states = self.edge_update(edge_states, node2edge_messages)
        
        return node_states, edge_states

class GatedPooling(nn.Module):
    '''
    Modified Version of Global Pooling Layer from the “Gated Graph Sequence Neural Networks” paper
    Parameters:
    ----------
        node_dim: Dimension of node features
        edge_dim: Dimension of edge features
    '''

    def __init__(self, node_dim, edge_dim, pooling_dim):
        super(GatedPooling, self).__init__()

        ###############################################################
        # Gates to compute attention scores
        self.hgate_node = nn.Sequential(
            nn.Linear(node_dim, 1)
        )
        self.hgate_edge = nn.Sequential(
            nn.Linear(edge_dim, 1)
        )

        ##############################################################
        #Layers to tranfrom features before combinig
        self.htheta_node = nn.Sequential(
            nn.Linear(node_dim, node_dim)
            nn.ReLU()
        )
        self.htheta_edge = nn.Sequential(
            nn.Linear(edge_dim, edge_dim)
            nn.ReLU()
        )

        ################################################################
        #Final pooling layer
        self.poolingLayer = nn.Sequential(
            nn.Linear(node_dim + edge_dim, pooling_dim)
        )
    def forward(self, node_features, edge_features):

        node_alpha = self.hgate_node(node_features)
        edge_alpha = self.hgate_edge(edge_features)
        import ipdb; ipdb.set_trace()
        node_pool = torch.sum(node_alpha*node_features, dim=0)
        edge_pool = torch.sum(edge_alpha*edge_features, dim=0)

        return self.poolingLayer(cat((node_pool, edge_pool), -1))

class EnergyModel(nn.Module):
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
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
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

        self.sg_pooler = GatedPooling(self.obj_label_embed_dim, self.rel_label_embed_dim)
        self.im_pooler = GatedPooling(self.obj_embed_dim, self.rel_embed_dim)

        self.energy = nn.Sequential(
            nn.Linear(self.pooling_dim*2, 1)
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