from torch import nn
import torch.nn.functional as F

from torch import nn, random
import torch.nn.functional as F
import numpy as np
import torch
import random


fix_seed = 2023
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

class GraphWaveletNeuralNetwork(nn.Module):
    def __init__(self, feature_dims, hidden_dims,device,dropout_rate=0.2): #分别为节点数，节点特征维度，小波基，逆小波基
        super(GraphWaveletNeuralNetwork, self).__init__()
        self.feature_dims = feature_dims
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.device=device

        self.conv_1 = GWNNLayer(self.feature_dims,
                                self.hidden_dims,self.device)

        self.conv_2 = GWNNLayer(self.hidden_dims,
                                self.hidden_dims,self.device)

    def forward(self, input,wavelets,wavelets_inv):
        output_1 = F.dropout(F.relu(self.conv_1(input,wavelets,wavelets_inv)),
                             training=self.training,
                             p=self.dropout_rate)
        # output_1 = self.conv_1(input, wavelets, wavelets_inv)
        output_2 = self.conv_2(output_1,wavelets,wavelets_inv)

        return output_2


import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F


class GWNNLayer(nn.Module):

    def __init__(self, in_channels, out_channels,device):
        super(GWNNLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.device=device
        self.filter=None

        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.in_channels, self.out_channels)).to(self.device)

        init.xavier_uniform_(self.weight_matrix)

        self.to(self.device)

    def _initial_filter(self, node_num):
        self.filter = nn.Parameter(torch.Tensor(node_num)).to(self.device)
        init.uniform_(self.filter, 0.9, 1.1)
        return

    def forward(self, features,wavelets,wavelets_inv):

        if self.filter is None:
            self._initial_filter(wavelets.size(1))
        # print(self.filter.shape) # torch.Size([706])
        # print(wavelets.shape) # torch.Size([1, 706, 706])
        # print(features.shape)
        # print(self.weight_matrix.shape)
        transformed_features = torch.matmul(features, self.weight_matrix)
        output = torch.matmul(torch.matmul(wavelets, torch.diag(self.filter[:wavelets.size(1)])),
                          torch.matmul(wavelets_inv, transformed_features))
        return output
