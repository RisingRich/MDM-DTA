import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import global_add_pool
from transformers import T5EncoderModel, T5Tokenizer, EsmModel, EsmTokenizer
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

from .densenet import DenseNet


class MPNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(MPNNLayer, self).__init__(aggr='add')
        self.node_linear = nn.Linear(in_channels, out_channels)
        self.update_mlp = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.node_linear(x)
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        return self.update_mlp(aggr_out)


class MPNNNet(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=78, output_dim=128, dropout=0.2):
        super(MPNNNet, self).__init__()

        self.x = None
        dim = 32
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # MPNN层提取smile原子特征
        self.conv1 = MPNNLayer(num_features_xd, dim)
        self.bn1 = torch.nn.BatchNorm1d(dim)
        self.conv2 = MPNNLayer(dim, dim)
        self.bn2 = torch.nn.BatchNorm1d(dim)
        self.conv3 = MPNNLayer(dim, dim)
        self.bn3 = torch.nn.BatchNorm1d(dim)
        self.conv4 = MPNNLayer(dim, dim)
        self.bn4 = torch.nn.BatchNorm1d(dim)
        self.conv5 = MPNNLayer(dim, dim)
        self.bn5 = torch.nn.BatchNorm1d(dim)
        self.fc1_xd = nn.Linear(dim, output_dim)

        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 128)
        self.out = nn.Linear(128, 1)


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 逐层传递消息
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1_xd(x))
        x = F.dropout(x, p=0.2, training=self.training)

        

        return x
