import os
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F


class PEModel(nn.Module):
    def __init__(self, dropout=0.2):
        super(PEModel, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cwd = os.getcwd()
        model_folder = 'esm'
        model_path = os.path.join(cwd, model_folder)
        self.esm_model = AutoModel.from_pretrained(model_path)
        self.esm_tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.esm_embeddings =None

    def forward(self, data):
        protein_sequences = data.protein_sequences
        device = torch.device(self.device)
        # 直接把模型移到目标设备
        self.esm_model = self.esm_model.to(device)

        # 创建列表来存储输入 IDs
        protein_input_ids = []
        for seq in protein_sequences:
            tokens = self.esm_tokenizer(seq, return_tensors='pt', padding='max_length', max_length=128, truncation=True)
            # 将 token 移动到指定的设备
            protein_input_ids.append(tokens['input_ids'].to(device))

        # 将输入 ID 合并为一个大的张量，合并时指定维度（dim=0）
        protein_input_ids = torch.cat(protein_input_ids, dim=0)


        # 将输入传入模型，获得输出
        esm_outputs = self.esm_model(input_ids=protein_input_ids)
        # 获取模型的最后隐藏状态并计算平均值
        self.esm_embeddings = esm_outputs.last_hidden_state.mean(dim=1)

        return self.esm_embeddings



# class PEModel(nn.Module):
#     def __init__(self, model_folder='esm', output_dim=480, dropout=0.2):
#         super(PEModel, self).__init__()
#         self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#         cwd = os.getcwd()
#         model_path = os.path.join(cwd, model_folder) if os.path.isdir(os.path.join(cwd, model_folder)) else model_folder
#
#         self.encoder = AutoModel.from_pretrained(model_path).to(self.device)
#         self.tokenizer = AutoTokenizer.from_pretrained(model_path)
#
#         hidden_size = self.encoder.config.hidden_size
#         self.fc = nn.Linear(hidden_size, output_dim)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, data):
#         sequences = data.protein_sequences  # list[str]
#
#         # 批量 tokenize
#         encoding = self.tokenizer(
#             sequences,
#             return_tensors='pt',
#             padding='max_length',
#             truncation=True,
#             max_length=128
#         )
#         input_ids = encoding['input_ids'].to(self.device)
#         attention_mask = encoding['attention_mask'].to(self.device)
#
#         # 前向传播
#         outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
#         last_hidden = outputs.last_hidden_state  # [batch, seq_len, hidden_size]
#
#         # mask 平均池化
#         mask = attention_mask.unsqueeze(-1)  # [batch, seq_len, 1]
#         masked_hidden = last_hidden * mask
#         embeddings = masked_hidden.sum(dim=1) / mask.sum(dim=1)
#
#         # Dropout + FC + ReLU
#         embeddings = self.dropout(self.relu(self.fc(embeddings)))
#         return embeddings
