import json  # 用于处理JSON格式数据
import pickle  # 用于序列化和反序列化数据
from collections import OrderedDict  # 用于创建有序字典
import networkx as nx  # 用于图形处理
from rdkit.Chem import Descriptors, rdMolDescriptors
from sklearn.preprocessing import StandardScaler

from utils import *  # 导入自定义的工具函数
import os
import pandas as pd
from rdkit import Chem, DataStructs
import torch
import numpy as np



# 计算分子描述符
def calculate_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    return [
        # 分子量 (MW)
        Descriptors.MolWt(mol),

        # ALogP (常用的LogP计算方法之一)
        Descriptors.MolLogP(mol),

        # 氢键供体数 (HBDS)
        Descriptors.NumHDonors(mol),

        # 氢键受体数 (HBAS)
        Descriptors.NumHAcceptors(mol),

        # 极性表面积 (PSA)
        Descriptors.TPSA(mol),

        # 可旋转键数 (ROTBS)
        Descriptors.NumRotatableBonds(mol),

        # 芳香环数量 (AROMS)
        rdMolDescriptors.CalcNumAromaticRings(mol),

        # 警告片段数量 (ALERTS)
        Descriptors.fr_Al_COO(mol) + Descriptors.fr_Ar_COO(mol) + Descriptors.fr_Ar_N(mol) +
                  Descriptors.fr_Ar_NH(mol) + Descriptors.fr_NH2(mol) + Descriptors.fr_Ar_OH(mol)
    ]

def atom_features(atom):
    # 定义函数：获取原子特征向量
    return np.array(
        one_of_k_encoding_unk(atom.GetSymbol(),
                              ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I',
                               'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li',
                               'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
        one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
        one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
        one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
        [atom.GetIsAromatic()]  # 获取原子的芳香性信息（布尔值）
    )


def one_of_k_encoding(x, allowable_set):
    # 定义函数：将输入编码为one-hot向量
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))  # 将x转换为one-hot向量


def one_of_k_encoding_unk(x, allowable_set):
    # 定义函数：将输入编码为one-hot向量，未知输入映射到允许集合的最后一个元素
    if x not in allowable_set:
        x = allowable_set[-1]  # 如果x不在允许集合中，则将x设置为允许集合中的最后一个元素
    return list(map(lambda s: x == s, allowable_set))  # 将x转换为one-hot向量


def smile_feature(smile):
    mol = Chem.MolFromSmiles(smile)
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)  # 获取每个原子的特征向量
        features.append(feature / sum(feature))  # 归一化特征向量并添加到列表中
    return features

def bond_features(bond):
    # 提取化学键的特征
    bond_type = bond.GetBondTypeAsDouble()  # 获取键的类型
    return np.array([bond_type])  # 可以根据需要扩展特征

def smile_to_graph(smile):
    # 定义函数：将SMILES字符串转换为图形表示
    mol = Chem.MolFromSmiles(smile)  # 从SMILES字符串创建分子对象

    c_size = mol.GetNumAtoms()  # 获取分子中的原子数

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)  # 获取每个原子的特征向量
        features.append(feature / sum(feature))  # 归一化特征向量并添加到列表中

    edges = []
    edge_attr = []  # 初始化边特征列表
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])  # 获取所有化学键的起始和结束原子索引
        edge_attr.append(bond_features(bond))
    g = nx.Graph(edges).to_directed()  # 构建图形对象并转换为有向图
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])  # 获取图形的边索引

    return c_size, features, edge_index,edge_attr  # 返回原子数、特征向量列表和边索引列表


def seq_cat(prot):
    # 定义函数：将蛋白质序列编码为固定长度的向量
    x = np.zeros(max_seq_len)  # 创建全零数组，长度为最大序列长度
    for i, ch in enumerate(prot[:max_seq_len]):
        x[i] = seq_dict[ch]  # 使用seq_dict将蛋白质序列的字符映射为数字
    return x  # 返回编码后的向量


# 处理DeepDTA数据
all_prots = []  # 存储所有蛋白质的列表
datasets = ['kiba', 'davis']  # 数据集列表
for dataset in datasets:
    print('convert data from DeepDTA for ', dataset)
    fpath = 'data/' + dataset + '/'  # 数据文件路径
    train_fold = json.load(open(fpath + "folds/train_fold_setting1.txt"))  # 加载训练集折叠数据
    train_fold = [ee for e in train_fold for ee in e]  # 将嵌套列表展开为一维列表
    valid_fold = json.load(open(fpath + "folds/test_fold_setting1.txt"))  # 加载测试集折叠数据
    ligands = json.load(open(fpath + "ligands_can.txt"), object_pairs_hook=OrderedDict)  # 加载化合物数据
    proteins = json.load(open(fpath + "proteins.txt"), object_pairs_hook=OrderedDict)  # 加载蛋白质数据
    affinity = pickle.load(open(fpath + "Y", "rb"), encoding='latin1')  # 加载亲和力数据
    drugs = []
    prots = []
    for d in ligands.keys():
        lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]),
                              isomericSmiles=True)  # 将化合物的Canonical SMILES转换为分子对象后再转换为SMILES
        drugs.append(lg)  # 将转换后的SMILES字符串添加到列表中
    for t in proteins.keys():
        prots.append(proteins[t])  # 将蛋白质序列添加到列表中
    if dataset == 'davis':
        affinity = [-np.log10(y / 1e9) for y in affinity]  # 将Davis数据集的亲和力转换为-pKa值
    affinity = np.asarray(affinity)  # 将亲和力列表转换为NumPy数组
    opts = ['train', 'test']  # 数据集操作类型列表
    for opt in opts:
        rows, cols = np.where(np.isnan(affinity) == False)  # 获取非NaN值的行列索引
        if opt == 'train':
            rows, cols = rows[train_fold], cols[train_fold]  # 获取训练集的行列索引
        elif opt == 'test':
            rows, cols = rows[valid_fold], cols[valid_fold]  # 获取测试集的行列索引
        with open('data/' + dataset + '_' + opt + '.csv', 'w') as f:
            f.write('compound_iso_smiles,target_sequence,affinity\n')  # 写入CSV文件的标题行
            for pair_ind in range(len(rows)):
                ls = []
                ls += [drugs[rows[pair_ind]]]  # 添加化合物SMILES
                ls += [prots[cols[pair_ind]]]  # 添加蛋白质序列
                ls += [affinity[rows[pair_ind], cols[pair_ind]]]  # 添加亲和力值
                f.write(','.join(map(str, ls)) + '\n')  # 将列表转换为字符串并写入文件
    print('\ndataset:', dataset)
    print('train_fold:', len(train_fold))
    print('test_fold:', len(valid_fold))
    print('len(set(drugs)),len(set(prots)):', len(set(drugs)), len(set(prots)))  # 输出唯一化合物和蛋白质的数量
    all_prots += list(set(prots))  # 将当前数据集中的蛋白质添加到all_prots列表中

seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"  # 蛋白质序列的字母表
seq_dict = {v: (i + 1) for i, v in enumerate(seq_voc)}  # 将字母映射为数字的字典
seq_dict_len = len(seq_dict)  # 字典的长度
max_seq_len = 1000  # 最大蛋白质序列长度

compound_iso_smiles = []  # 存储所有化合物SMILES的列表
for dt_name in ['case1']:
    opts = ['train', 'test']
    for opt in opts:
        df = pd.read_csv('data/' + dt_name + '_' + opt + '.csv')  # 读取CSV文件
        compound_iso_smiles.extend(df['compound_iso_smiles'])  # 更高效地添加化合物SMILES
compound_iso_smiles = list(set(compound_iso_smiles))
smile_graph = {}  # 存储化合物图形表示的字典
for smile in compound_iso_smiles:
    g = smile_to_graph(smile)  # 获取化合物的图形表示
    smile_graph[smile] = g  # 将化合物和其图形表示添加到字典中,

descriptors_dir = {}
for smile in compound_iso_smiles:
    descriptors = calculate_descriptors(smile)
    descriptors_dir[smile] = descriptors



cuda_name = "cuda:0"
datasets = ['case1']  # 数据集列表
device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
# 转换为PyTorch数据格式
for dataset in datasets:
    processed_data_file_train = 'data/processed/' + dataset + '_train.pt'  # 训练集的PyTorch数据文件路径
    processed_data_file_test = 'data/processed/' + dataset + '_test.pt'  # 测试集的PyTorch数据文件路径
    if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
        df = pd.read_csv('data/' + dataset + '_train.csv')  # 读取训练集CSV文件
        train_drugs, train_prots, train_Y = list(df['compound_iso_smiles']), list(df['target_sequence']), list(
            df['affinity'])  # 获取训练集的化合物SMILES、蛋白质序列和亲和力值

        XT = [seq_cat(t) for t in train_prots]  # 将训练集的蛋白质序列编码为向量

        train_drugs, train_prots,train_prots_list, train_Y = np.asarray(train_drugs), np.asarray(XT),np.asarray(train_prots), np.asarray(train_Y)  # 转换为NumPy数组

        print('preparing ', dataset + '_train.pt in pytorch format!')
        train_data = TestbedDataset(root='data', dataset=dataset + '_train', xd=train_drugs, xt=train_prots,xl =train_prots_list, y=train_Y,
                                    smile_graph=smile_graph, d=descriptors_dir)

        df = pd.read_csv('data/' + dataset + '_test.csv')  # 读取测试集CSV文件
        test_drugs, test_prots, test_Y = list(df['compound_iso_smiles']), list(df['target_sequence']), list(
            df['affinity'])  # 获取测试集的化合物SMILES、蛋白质序列和亲和力值
        XT = [seq_cat(t) for t in test_prots]  # 将测试集的蛋白质序列编码为向量
        test_drugs, test_prots, test_prots_list,test_Y = np.asarray(test_drugs), np.asarray(XT),np.asarray(test_prots), np.asarray(test_Y)  # 转换为NumPy数组

        print('preparing ', dataset + '_test.pt in pytorch format!')
        test_data = TestbedDataset(root='data', dataset=dataset + '_test', xd=test_drugs, xt=test_prots,xl =test_prots_list, y=test_Y,
                                   smile_graph=smile_graph, d=descriptors_dir)
        print(processed_data_file_train, ' and ', processed_data_file_test, ' have been created')
    else:
        print(processed_data_file_train, ' and ', processed_data_file_test, ' are already created')

