# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT
import torch
from torch import nn

class EGNNLayer(nn.Module):
    '''
    Graph Layer to apply edged graph convolution on the specified graphs
    '''

    def __init__(self, node_dim, edge_dim, iters=3):
        
        super(EGNNLayer, self).__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.iters = iters
        self.alpha = 0.5
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
            nn.Linear(self.node_dim, self.node_dim),
            self.kernel_activation
        )

        self.node2edge_kernel = nn.Sequential(
            nn.Conv2d(2*self.node_dim, self.edge_dim, kernel_size=1, stride=1),
            self.kernel_activation, 
            nn.Conv2d(self.edge_dim, self.edge_dim, kernel_size=1, stride=1),
            self.kernel_activation, 
        )

        self.node_gate = nn.GRUCell(self.node_dim, self.node_dim)
        self.edge_gate = nn.GRUCell(self.edge_dim, self.edge_dim, bias=False)


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

class GNNLayer(nn.Module):
    '''
    Graph Layer to apply edged graph convolution on the specified graphs
    '''

    def __init__(self, node_dim, iters=3):
        
        super(GNNLayer, self).__init__()

        self.node_dim = node_dim
        self.iters = iters

        ################################
        #Define the kernels

        self.kernel_activation = nn.ReLU()
        self.node_kernel = nn.Sequential(
            nn.Linear(self.node_dim, self.node_dim),
            self.kernel_activation,
            nn.Linear(self.node_dim, self.node_dim),
            self.kernel_activation
            )
        
        self.node_gate = nn.GRUCell(self.node_dim, self.node_dim)

    def node_mp(self, node_states, adj_matrix):
        '''
        Function to aggreagate message from all the neighbouring nodes in the graph
        '''
        node_messages = torch.mm(adj_matrix, node_states)
        #Message normalization
        node_messages = node_messages/(torch.sum(adj_matrix, dim=1, keepdim=True) + 1e-6)
        return node_messages
    
    
    def node_update(self, node_states, node_messages):
        return self.node_gate(node_messages, node_states)

    def forward(self, node_states, adj_matrix):

        for _ in range(self.iters):

            #Aggregate node to node infromations
            node_messages = self.node_mp(node_states, adj_matrix)
            node_messages = self.node_kernel(node_messages)
            node_states = self.node_update(node_states, node_messages)
        
        return node_states