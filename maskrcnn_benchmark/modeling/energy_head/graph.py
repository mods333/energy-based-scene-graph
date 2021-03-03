# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT
import torch

class Graph(object):
    '''
    A wrapper class for graph
    '''

    def __init__(self, node_states, adj_matrix, batch_list, edge_states=None, edge_batch_list=None):

        self.node_states = node_states
        self.adj_matrix = adj_matrix
        self.batch_list = batch_list
        self.edge_states = edge_states
        self.edge_batch_list = edge_batch_list
        self.device = node_states.device
        
        self.num_nodes = node_states.shape[0]

    def get_states(self):
        '''
        Return node and edge_states
        '''
        return self.node_states, self.edge_states
    
    def get_adj(self):
        return self.adj_matrix
    
    def get_batch_list(self):
        return self.batch_list
    
    def get_edge_batch_list(self):
        return self.edge_batch_list
    
    def adj_type(self):
        if self.adj_matrix.shape[-1] == 2:
            return 'pair'
        else:
            return 'matrix'
    def detach(self):
        self.node_states.detach()
        self.edge_states.detach()
        
    def requires_grad(self, mode):

        if mode == 'predcls':
            self.edge_states.requires_grad = True
        else:
            self.node_states.requires_grad = True
            self.edge_states.requires_grad = True
    
    def pair2matrix(self):
        
        assert self.adj_type() == 'pair', "Trying to convert adj list to matrix but the adjacency  is not a pair list and has shape {}".format(self.adj_matrix.shape)

        adj_matrix = torch.zeros(size=(self.num_nodes, self.num_nodes)).to(self.device)
        adj_matrix[self.adj_matrix[:,0], self.adj_matrix[:,1]] = 1

        return adj_matrix
    def __len__(self):
        return self.node_states.shape[0]