from maskrcnn_benchmark.modeling import registry
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence
from torch.nn import functional as F
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.modeling.make_layers import make_fc
from .model_motifs import FrequencyBias
from .utils_motifs import obj_edge_vectors, encode_box_info, to_onehot, nms_overlaps


class EGNNContext(nn.Module):
    '''
    An edge-based message passing algorithm for scene graph generation 
    Parameters:
        ----------
            config(DictConfig): experiment configration
            obj_classes: list of all the object classes in the dataset
            rel_classes: list of all the relation classes in the dataset
            in_channels: 
    '''

    def __init__(self, config, obj_classes, rel_classes, in_channels, num_iter=3):

        super(EGNNContext, self).__init__()

        self.cfg = config
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.num_iter = num_iter
        self.alpha = 0.5
        self.nms_thresh = self.cfg.TEST.RELATION.LATER_NMS_PREDICTION_THRES

        self.node_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.edge_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.hidden_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM

        self.num_obj_classes = len(obj_classes)
        self.num_rel_classes = len(rel_classes)

        self.dropout_rate = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_DROPOUT_RATE

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
        #Size of the embedding vector
        self.embed_dim = self.cfg.MODEL.ROI_RELATION_HEAD.EMBED_DIM
        #Obtain weights for the embeding layers from glove
        obj_embed_vecs = obj_edge_vectors(self.obj_classes, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.embed_dim)

        self.obj_embed1 = nn.Embedding(self.num_obj_classes, self.embed_dim)
        self.obj_embed2 = nn.Embedding(self.num_obj_classes, self.embed_dim)
        with torch.no_grad():
            self.obj_embed1.weight.copy_(obj_embed_vecs, non_blocking=True)
            self.obj_embed2.weight.copy_(obj_embed_vecs, non_blocking=True)

        self.obj_dim = in_channels + self.embed_dim + 128 #128 correspond to size of position embedding
        self.rel_dim = in_channels

        #Node Embedding
        self.node_embedding = nn.Linear(self.obj_dim, self.node_dim)
        self.node_embedding2 = nn.Linear(in_channels + self.embed_dim + self.node_dim, self.node_dim)
        #Edge embedding
        self.edge_embeding = nn.Linear(self.rel_dim, self.edge_dim)
        # position embedding
        self.pos_embed = nn.Sequential(*[
            nn.Linear(9, 32), nn.BatchNorm1d(32, momentum= 0.001),
            nn.Linear(32, 128), nn.ReLU(inplace=True),
        ])
        ##########################################################################################

        ##########################################################################################
        #Gating Layers
        #Gating functin for updating the node and edge states
        self.node_gate = nn.GRUCell(self.node_dim, self.node_dim)
        self.edge_gate = nn.GRUCell(self.edge_dim, self.edge_dim, bias=False)
        # self.edge_gate = nn.Parameter(torch.zeros(self.edge_dim*2, self.edge_dim*2))
        # self.edge_transform = nn.Parameter(torch.zeros(self.edge_dim*2, self.edge_dim))
        # nn.init.xavier_uniform_(self.edge_gate, gain=1.414)
        # nn.init.xavier_uniform_(self.edge_transform, gain=1.414)
        ##########################################################################################

        ##########################################################################################
        # Message Passing Layers
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
            nn.Linear(self.node_dim, self.node_dim),
            self.kernel_activation
        )

        # self.node2edge_kernel = nn.Parameter(torch.zeros(self.node_dim*2, self.edge_dim))
        # nn.init.xavier_uniform_(self.node2edge_kernel, gain=1.414)

        self.node2edge_kernel = nn.Sequential(
            nn.Conv2d(2*self.node_dim, self.edge_dim, kernel_size=1, stride=1),
            self.kernel_activation, 
            nn.Conv2d(self.edge_dim, self.edge_dim, kernel_size=1, stride=1),
            self.kernel_activation, 
        )
        ##########################################################################################

        #Object classifier
        self.obj_classifier = make_fc(self.hidden_dim, self.num_obj_classes)
        self.rel_classifier = make_fc(self.hidden_dim, self.num_rel_classes)

    def get_contiguous_rel_pair_idx(self, rel_pair_idxs, proposals):
        '''
        This function converts the list of rel_pair_idxs into a single tensor 
        '''
        offset = 0
        pair_list = []
        for i, proposal in enumerate(proposals): 
            pair_list.append(rel_pair_idxs[i] + offset)
            offset += len(proposal)
        
        return torch.cat(pair_list, dim=0)
    
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
        ek_input = ek_input.transpose(0, 2).unsqueeze(0)
        node2edge_messages = self.node2edge_kernel(ek_input).squeeze(0).transpose(0,2)
        # node2edge_messages = self.kernel_activation(torch.matmul(ek_input, self.node2edge_kernel))
        return node2edge_messages    
    
    def node_update(self, node_states, node2node_messages, edge2node_messages):
        return self.node_gate(self.alpha*node2node_messages + (1-self.alpha)*edge2node_messages, node_states)

    def edge_update(self, edge_states, node2edge_messages):
        m = node2edge_messages.shape[0]
        edge_states = self.edge_gate(node2edge_messages.reshape(m*m, -1), edge_states.reshape(m*m, -1)).view(m,m,-1)
        return edge_states

    def forward(self, x, proposals, union_features, rel_pair_idxs, logger=None, all_average=False):
        '''
        Parameters:
        ----------
            x : Roi Features
            proposals: Region proposals
            rel_pair_idx: List of index pair suggesting possible realtions
        '''

        #########################################################################################
        #Obtain ground truth object labels or prdicted object logits from 
        #the detected and  pass through embedding layer
        
        if self.training or self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            obj_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        else:
            obj_labels = None
        
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            obj_embed = self.obj_embed1(obj_labels.long())
        else:
            obj_logits = cat([proposal.get_field("predict_logits") for proposal in proposals], dim=0).detach()
            obj_embed = F.softmax(obj_logits, dim=1) @ self.obj_embed1.weight
        #########################################################################################

        #Embedding the bounding boxes
        assert proposals[0].mode == 'xyxy'
        pos_embed = self.pos_embed(encode_box_info(proposals))

        #Object embedding/Node embedding
        obj_pre_rep = cat((x, obj_embed, pos_embed), -1)
        node_states = self.node_embedding(obj_pre_rep)

        # boxes_per_cls = None
        # if self.mode == 'sgdet' and not self.training:
        #     boxes_per_cls = cat([proposal.get_field('boxes_per_cls') for proposal in proposals], dim=0) # comes from post process of box_head

        #Adjacency matrix
        rel_pair_idx = self.get_contiguous_rel_pair_idx(rel_pair_idxs, proposals)
        
        adj_matrix = torch.zeros(size=(obj_pre_rep.shape[0], obj_pre_rep.shape[0])).to(x.device)
        adj_matrix[rel_pair_idx[:,0], rel_pair_idx[:,1]] = 1

        #Embedding the edges representation
        edge_rep = self.edge_embeding(union_features)
        edge_states = torch.sparse.FloatTensor(rel_pair_idx.t(), edge_rep, torch.Size([adj_matrix.shape[0], adj_matrix.shape[0], edge_rep.shape[-1]])).to_dense()
        
        #########################################################################################
        # Message Passing to update node states
        for _ in range(self.num_iter):
            
            #Aggregate node to node infromations
            node2node_messages = self.node2node_mp(node_states, adj_matrix)
            #Aggregate edge to node informations
            edge2node_messages = self.edge2node_mp(edge_states, adj_matrix)
            #Aggregate node to edge infromations
            # node2edge_messages = self.node2edge_mp(node_states, adj_matrix)

            #Apply kernels to the recieves messages
            node2node_messages = self.node2node_kernel(node2node_messages)
            edge2node_messages = self.edge2node_kernel(edge2node_messages)

            node_states = self.node_update(node_states, node2node_messages, edge2node_messages)
            # edge_states = self.edge_update(edge_states, node2edge_messages)
        
        #########################################################################################
        # Object Classification
        if self.mode != 'predcls':
            obj_dists = self.obj_classifier(node_states)
            
            use_decoder_nms = self.mode == 'sgdet' and not self.training
            if use_decoder_nms:
                num_objs = [len(p) for p in proposals]
                boxes_per_cls = [proposal.get_field('boxes_per_cls') for proposal in proposals]
                obj_preds = self.nms_per_cls(obj_dists, boxes_per_cls, num_objs)
            else:
                obj_preds = obj_dists[:, 1:].max(1)[1] + 1
        else:
            assert obj_labels is not None
            obj_preds = obj_labels
            obj_dists = to_onehot(obj_preds, self.num_obj_classes)

        #Obtain new node states to refine the edge states besed on the either the refined node states and predicted/gt object labels
        node_states = cat((x, node_states, self.obj_embed2(obj_preds)), dim=-1)
        node_states = self.node_embedding2(node_states) #to change the dimesion

        #########################################################################################
        #Message Passing to Update the edge states 
        for _ in range(self.num_iter):
            #Aggregate node to edge infromations
            node2edge_messages = self.node2edge_mp(node_states, adj_matrix)
            #Update the edge states
            edge_states = self.edge_update(edge_states, node2edge_messages)

        #########################################################################################
        # Relatoion Classifications
        # Flattent 2d matrix
        edge_states = edge_states[rel_pair_idx[:, 0], rel_pair_idx[:,1]]
        rel_dists = self.rel_classifier(edge_states)

        return obj_dists, rel_dists        

    def nms_per_cls(self, obj_dists, boxes_per_cls, num_objs):
        obj_dists = obj_dists.split(num_objs, dim=0)
        obj_preds = []
        for i in range(len(num_objs)):
            is_overlap = nms_overlaps(boxes_per_cls[i]).cpu().numpy() >= self.nms_thresh # (#box, #box, #class)

            out_dists_sampled = F.softmax(obj_dists[i], -1).cpu().numpy()
            out_dists_sampled[:, 0] = -1

            out_label = obj_dists[i].new(num_objs[i]).fill_(0)

            for i in range(num_objs[i]):
                box_ind, cls_ind = np.unravel_index(out_dists_sampled.argmax(), out_dists_sampled.shape)
                out_label[int(box_ind)] = int(cls_ind)
                out_dists_sampled[is_overlap[box_ind,:,cls_ind], cls_ind] = 0.0
                out_dists_sampled[box_ind] = -1.0 # This way we won't re-sample

            obj_preds.append(out_label.long())
        obj_preds = torch.cat(obj_preds, dim=0)
        return obj_preds

        
