import torch
from torch import nn
import torch.nn.functional as F


class DescriptorModel(nn.Module):
    def __init__(self, output_dim=128, descriptor_size=8, dropout=0.2):
        super(DescriptorModel, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.descriptor_conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.descriptor_conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.descriptor_conv3 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.descriptor_pool = nn.MaxPool1d(kernel_size=2)

        linear_input_dim = 32 * self._descriptor_linear_size(descriptor_size)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.fc_d = nn.Linear(linear_input_dim,128)

    def _descriptor_linear_size(self, descriptor_size):

        size = descriptor_size // 2 
        size = size // 2  
        size = size // 2 
        return size

    def forward(self, data):
        data.d = data.d.to(self.device)
        descriptors = data.d.unsqueeze(1) 
        descriptors = F.relu(self.descriptor_conv1(descriptors))
        descriptors = self.descriptor_pool(descriptors)
        descriptors = F.relu(self.descriptor_conv2(descriptors))
        descriptors = self.descriptor_pool(descriptors)
        descriptors = F.relu(self.descriptor_conv3(descriptors))
        descriptors = self.descriptor_pool(descriptors)
        descriptors = descriptors.view(descriptors.size(0), -1)
        descriptors = self.fc_d(descriptors)

        return descriptors
