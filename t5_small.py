import os
import torch.nn.functional as F
import torch
from torch import nn
from transformers import T5EncoderModel, T5Tokenizer
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class SmilesEmbeddingModel(nn.Module):
    def __init__(self, output_dim=128,dropout=0.2):
        super(SmilesEmbeddingModel, self).__init__()
        # 加载smiles大语言模型
        cwd = os.getcwd()
        model_folder = 't5-small'
        model_path = os.path.join(cwd, model_folder)
        self.t5_model = T5EncoderModel.from_pretrained(model_path)
        self.tokenizer = T5Tokenizer.from_pretrained(model_path, legacy=False)
        self.fc1 = nn.Linear(512, 480)


    def forward(self, data):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # 处理化合物 SMILES 分子
        smiles = data.smiles
        max_length = 128
        input_ids = []
        for s in smiles:
            s = s[0]
            encoded = self.tokenizer.encode(s, max_length=max_length, truncation=True, padding='max_length',
                                            return_tensors='pt')
            input_ids.append(encoded)
        input_ids = torch.cat(input_ids, dim=0).to(device)
        outputs = self.t5_model(input_ids)
        last_hidden_states = outputs.last_hidden_state
        smiles_embeddings = torch.mean(last_hidden_states, dim=1).squeeze()
        smiles_embeddings = self.fc1(smiles_embeddings)
        return smiles_embeddings
