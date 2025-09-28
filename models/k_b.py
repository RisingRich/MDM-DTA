import os

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import torch.nn.functional as F

class K_B_Model(nn.Module):
    def __init__(self, output_dim=128, dropout=0.2):
        super(K_B_Model, self).__init__()
        # 加载smiles大语言模型
        # 加载预训练好的BERT模型和tokenizer
        # 普通bert
        cwd = os.getcwd()
        model_folder = 'bert'
        model_path = os.path.join(cwd, model_folder)
        # self.k_bert = BertModel.from_pretrained('bert-base-uncased')
        from transformers import DistilBertTokenizer, DistilBertModel
        # model_name = '/distilbert-base-uncased'
        # model_name = 'huawei-noah/TinyBERT_General_4L_312D'
        # self.k_bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.k_bert = BertModel.from_pretrained(model_path)
        # k-bert
        model_path2 = 'models/pretrain_k_bert_epoch_7.pth'
        model_path2 = os.path.join(cwd, model_path2)
        state_dict = torch.load(model_path2)
        self.k_bert.load_state_dict(state_dict, strict=False)
        # model_path = './models/k_bert.pth'
        # self.k_bert = torch.load(model_path)
        # self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.fc =nn.Linear(312,480)
        self.smiles_embeddings = None
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        smiles = data.smiles
        max_length = 128
        input_ids = []
        attention_masks = []
        for s in smiles:
            s = s[0]
            tokens = self.tokenizer.tokenize(s)[:max_length - 2]
            tokens = ['[CLS]'] + tokens + ['[SEP]']
            ids = self.tokenizer.convert_tokens_to_ids(tokens)
            padding = [0] * (max_length - len(ids))
            ids += padding
            input_ids.append(ids)
            mask = [1] * len(tokens) + [0] * len(padding)
            attention_masks.append(mask)
        device = torch.device('cuda:0')
        input_ids = torch.tensor(input_ids).to(device)
        attention_masks = torch.tensor(attention_masks).to(device)
        outputs = self.k_bert(input_ids, attention_mask=attention_masks)
        last_hidden_states = outputs[0]
        self.smiles_embeddings = torch.mean(last_hidden_states, dim=1).squeeze()
        smiles_embeddings = F.relu(self.fc(self.smiles_embeddings))
        smiles_embeddings=self.dropout(smiles_embeddings)
        return smiles_embeddings




# import os
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from transformers import BertModel, BertTokenizer
#
# class K_B_Model(nn.Module):
#     def __init__(self, model_folder='bert', output_dim=480, max_length=128, dropout=0.2):
#         super(K_B_Model, self).__init__()
#         self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         self.max_length = max_length
#
#         # 加载本地BERT模型
#         cwd = os.getcwd()
#         model_path = os.path.join(cwd, model_folder)
#         self.bert = BertModel.from_pretrained(model_path).to(self.device)
#         self.tokenizer = BertTokenizer.from_pretrained(model_path)
#
#         # 如果有微调好的权重
#         pretrained_weights = os.path.join(cwd, 'models/pretrain_k_bert_epoch_7.pth')
#         if os.path.exists(pretrained_weights):
#             state_dict = torch.load(pretrained_weights, map_location=self.device)
#             self.bert.load_state_dict(state_dict, strict=False)
#
#         self.fc = nn.Linear(self.bert.config.hidden_size, output_dim)
#         self.dropout = nn.Dropout(dropout)
#         self.relu = nn.ReLU()
#
#     def forward(self, data):
#         smiles_list = data.smiles  # list of strings
#
#         # 保证输入格式
#         if isinstance(smiles_list[0], (list, tuple)):
#             smiles_list = [s[0] for s in smiles_list]
#         elif not isinstance(smiles_list[0], str):
#             smiles_list = [str(s) for s in smiles_list]
#
#         # 批量 tokenization
#         encoding = self.tokenizer(
#             smiles_list,
#             return_tensors='pt',
#             padding='max_length',
#             truncation=True,
#             max_length=self.max_length
#         )
#         input_ids = encoding['input_ids'].to(self.device)
#         attention_mask = encoding['attention_mask'].to(self.device)
#
#         # BERT 前向传播
#         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
#         last_hidden = outputs.last_hidden_state  # [batch, seq_len, hidden_size]
#
#         # 带 mask 平均池化
#         mask = attention_mask.unsqueeze(-1)  # [batch, seq_len, 1]
#         masked_hidden = last_hidden * mask
#         embeddings = masked_hidden.sum(dim=1) / mask.sum(dim=1)
#
#         # FC + ReLU + Dropout
#         embeddings = self.dropout(self.relu(self.fc(embeddings)))
#         return embeddings
