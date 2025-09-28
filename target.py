from torch import nn
import torch
from .densenet import DenseNet
from .efficientnet1d import EfficientNet1D
import torch.nn.functional as F


class XtEmbeddingModel(nn.Module):
    def __init__(self, num_features_xt=1000, embed_dim=128, n_filters=128,num_classes=480,output_dim=128, dropout=0.2):
        super(XtEmbeddingModel, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt_1 = nn.Conv1d(in_channels=embed_dim, out_channels=1024, kernel_size=5)
        self.conv_xt_2 = nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=3)
        self.conv_xt_3 = nn.Conv1d(in_channels=512, out_channels=n_filters, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1_xt = nn.Linear(15744, output_dim)
        self.densenet = DenseNet()
        # self.efficientnet = EfficientNet1D(num_classes=num_classes, input_channels=n_filters)
        self.fc1 = nn.Linear(1024, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        target = data.target
        emb = self.embedding_xt(target)
        xt = emb.permute(0, 2, 1)
        xt = F.relu(self.conv_xt_1(xt))
        xt = self.pool(xt)
        xt = F.relu(self.conv_xt_2(xt))
        xt = self.pool(xt)
        xt = F.relu(self.conv_xt_3(xt))
        xt = self.pool(xt)
        xt = xt.view(xt.size(0), -1)
        xt = F.relu(self.fc1_xt(xt))
        xt = xt.unsqueeze(1)
        # xt = xt.unsqueeze(2)
        xt = self.densenet(xt)
        # xt = self.efficientnet(xt)
        xt = self.fc1(xt)
        xt = self.relu(xt)
        xt = self.dropout(xt)


        return xt
