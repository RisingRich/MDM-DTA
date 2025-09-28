import os
import numpy as np
from math import sqrt
from scipy import stats
from torch_geometric.data import InMemoryDataset
from torch_geometric import data as DATA
import torch
from torch_geometric.loader import DataLoader

class TestbedDataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='davis',
                 xd=None, xt=None,xl =None, y=None, transform=None,
                 pre_transform=None, smile_graph=None,d = None):

        # root is required for save preprocessed data, default is '/tmp'
        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        # benchmark dataset, default = 'davis'
        self.dataset = dataset


        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(xd, xt, xl ,y, smile_graph,d)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pass
        # return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    # Customize the process method to fit the task of drug-target affinity prediction
    # Inputs:
    # XD - list of SMILES, XT: list of encoded target (categorical or one-hot),
    # Y: list of labels (i.e. affinity)
    # Return: PyTorch-Geometric format processed data
    def process(self, xd, xt,xl, y, smile_graph, d):
        assert (len(xd) == len(xt) and len(xt) == len(y)), "The three lists must be the same length!"
        data_list = []
        data_len = len(xd)
        for i in range(data_len):
            print('Converting SMILES to graph: {}/{}'.format(i + 1, data_len))
            protein_sequences = xl[i]
            smiles = xd[i]
            target = xt[i]
            labels = y[i]
            # Convert SMILES to molecular representation using RDKit
            c_size, features, edge_index,edge_attr = smile_graph[smiles]
            descriptors = d[smiles]

            # Make the graph ready for PyTorch Geometric's GCN algorithms
            GCNData = DATA.Data(x=torch.Tensor(features),
                                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                edge_attr=torch.FloatTensor(edge_attr),  # 边特征张量
                                y=torch.FloatTensor([labels]),
                                d = torch.FloatTensor(descriptors).unsqueeze(0)
                                )
            GCNData.protein_sequences = [protein_sequences]
            GCNData.target = torch.LongTensor([target])
            GCNData.smiles = [smiles]
            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))

            # Append graph, label, and target sequence to data list
            data_list.append(GCNData)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)

        # Save preprocessed data
        torch.save((data, slices), self.processed_paths[0])


def rmse(y, f):
    rmse = sqrt(((y - f) ** 2).mean(axis=0))
    return rmse


def mse(y, f):
    mse = ((y - f) ** 2).mean(axis=0)
    return mse


def pearson(y, f):
    rp = np.corrcoef(y, f)[0, 1]
    return rp


def spearman(y, f):
    rs = stats.spearmanr(y, f)[0]
    return rs

def rm2(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 计算 R^2
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - ss_res / ss_tot

    # 计算 k
    k = np.sum(y_true * y_pred) / np.sum(y_pred**2)

    # 计算 R0^2
    ss_res0 = np.sum((y_true - k * y_pred)**2)
    r0_2 = 1 - ss_res0 / ss_tot

    # 计算 rm2
    rm2_val = r2 * (1 - np.sqrt(abs(r2 - r0_2)))

    return rm2_val

def ci(y, f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y) - 1
    j = i - 1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z + 1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i - 1
    ci = S / z
    return ci


def rae(y, f):
    y_mean = np.mean(y)
    numerator = np.sum(np.abs(y - f))
    denominator = np.sum(np.abs(y - y_mean))
    rae = numerator / denominator
    return rae


def rse(y, f):
    y_mean = np.mean(y)
    numerator = np.sum((y - f) ** 2)
    denominator = np.sum((y - y_mean) ** 2)
    rse = numerator / denominator
    return rse


def rmsle(y, f):
    epsilon = 1e-10
    y_log = np.log1p(y + epsilon)
    f_log = np.log1p(f + epsilon)
    rmsle = sqrt(((y_log - f_log) ** 2).mean(axis=0))
    return rmsle
