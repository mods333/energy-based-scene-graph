import torch
import torch.nn as nn

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
            nn.Linear(node_dim, node_dim),
            nn.ReLU()
        )
        self.htheta_edge = nn.Sequential(
            nn.Linear(edge_dim, edge_dim),
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
