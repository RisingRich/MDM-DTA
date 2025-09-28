import os
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch
from torch import nn
import torch.nn.functional as F


class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_heads,device):
        super(CrossAttentionFusion, self).__init__()
        self.device = device
        self.cross_attention_esm_to_smiles = nn.MultiheadAttention(embed_dim, num_heads)
        self.cross_attention_smiles_to_esm = nn.MultiheadAttention(embed_dim, num_heads)
        self.fusion_linear = nn.Linear(embed_dim * 2, embed_dim)
        self.activation = nn.ReLU()

    def forward(self, esm_embeddings, smiles_embeddings):
        esm_embeddings = esm_embeddings.to(self.device)
        smiles_embeddings = smiles_embeddings.to(self.device)
        esm_to_smiles_attention, _ = self.cross_attention_esm_to_smiles(
            query=esm_embeddings,
            key=smiles_embeddings,
            value=smiles_embeddings
        )
        smiles_to_esm_attention, _ = self.cross_attention_smiles_to_esm(
            query=smiles_embeddings,
            key=esm_embeddings,
            value=esm_embeddings
        )
        concatenated = torch.cat(
            (esm_to_smiles_attention.mean(dim=0), smiles_to_esm_attention.mean(dim=0)),
            dim=-1
        )
        fused_embeddings = self.activation(self.fusion_linear(concatenated))
        return fused_embeddings


class GIModel(nn.Module):
    def __init__(self, num_experts=3, in_features=480, dropout=0.2, embed_dim=480, num_heads=8):
        super(GIModel, self, ).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.cross_attention_fusion = CrossAttentionFusion(embed_dim, num_heads,self.device)
        self.gate = nn.Linear(in_features, num_experts)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(480,embed_dim)

    def forward(self, esm_embeddings, smiles_embeddings):
        esm_embeddings = esm_embeddings.to(self.device)
        smiles_embeddings = smiles_embeddings.to(self.device)
        esm_embeddings = self.fc1(esm_embeddings)
        fused_embeddings = self.cross_attention_fusion(esm_embeddings, smiles_embeddings)
        out = F.relu(self.gate(fused_embeddings))
        out = self.dropout(out)


        return out
