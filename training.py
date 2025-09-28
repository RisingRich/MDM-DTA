import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
from models.moe import MoeLayer
from utils import *
from torch.utils.data import random_split




# training function at each epoch
def train(model, device, train_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data.x),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))


def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


datasets = [['davis', 'kiba','metz','BindingDB'][int(sys.argv[1])]]
modeling = [MoeLayer][int(sys.argv[2])]
model_st = modeling.__name__

cuda_name = "cuda:0"
if len(sys.argv) > 3:
    cuda_name = "cuda:" + str(int(sys.argv[3]))
print('cuda_name:', cuda_name)

TRAIN_BATCH_SIZE = 100
TEST_BATCH_SIZE = 100
LR = 0.0005
LOG_INTERVAL = 20
NUM_EPOCHS = 50

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

# Main program: iterate over different datasets
for dataset in datasets:
    print('\nrunning on ', model_st + '_' + dataset)

    # Define processed data file paths
    processed_data_file_train = 'data/processed/' + dataset + '_train.pt'
    processed_data_file_test = 'data/processed/' + dataset + '_test.pt'

    # Check if processed data exists
    if not (os.path.isfile(processed_data_file_train) and os.path.isfile(processed_data_file_test)):
        print('Please run create_data.py to prepare data in pytorch format!')
    else:
        # Load the dataset
        train_data = TestbedDataset(root='data', dataset=dataset + '_train')
        test_data = TestbedDataset(root='data', dataset=dataset + '_test')

        train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)

        # Prepare model and optimizer
        device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.enabled = False  # Disabling the cudnn for determinism
        model = modeling().to(device)  # Assuming 'modeling' is the model instantiation function
        loss_fn = nn.SmoothL1Loss(beta=1.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        # Initialize variables to track the best results
        best_mse = 1000
        best_ci = 0
        best_epoch = -1
        model_file_name = 'model_' + model_st + '_' + dataset + '.model'
        result_file_name = 'result_' + model_st + '_' + dataset + '.csv'

        # Train the model
        for epoch in range(NUM_EPOCHS):
            train(model, device, train_loader, optimizer, epoch + 1)

            # Predict on the test data
            G, P = predicting(model, device, test_loader)
            ret = [rmse(G, P), mse(G, P), pearson(G, P), spearman(G, P), ci(G, P)]

            # Check for improvement in MSE and save the model if improved
            if ret[1] < best_mse:
                torch.save(model.state_dict(), model_file_name)
                with open(result_file_name, 'w') as f:
                    f.write(','.join(map(str, ret)))

                best_epoch = epoch + 1
                best_mse = ret[1]
                best_ci = ret[-1]
                print(
                    f'RMSE improved at epoch {best_epoch}; best_mse, best_ci: {best_mse}, {best_ci} {model_st} {dataset}')
            else:
                print(
                    f'{ret[1]} No improvement since epoch {best_epoch}; best_mse, best_ci: {best_mse}, {best_ci} {model_st} {dataset}')

# import numpy as np
# import pandas as pd
# import sys, os
# from random import shuffle
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, Subset
# from sklearn.model_selection import KFold
#
# from models.moe import MoeLayer
# from utils import *
#
# # -------------------------------
# # Training function
# # -------------------------------
# def train(model, device, train_loader, optimizer, epoch):
#     print('Training on {} samples...'.format(len(train_loader.dataset)))
#     model.train()
#     for batch_idx, data in enumerate(train_loader):
#         data = data.to(device)
#         optimizer.zero_grad()
#         output = model(data)
#         loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
#         loss.backward()
#         optimizer.step()
#         if batch_idx % LOG_INTERVAL == 0:
#             print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch,
#                 batch_idx * len(data.x),
#                 len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader),
#                 loss.item()
#             ))
#
# # -------------------------------
# # Prediction function
# # -------------------------------
# def predicting(model, device, loader):
#     model.eval()
#     total_preds = torch.Tensor()
#     total_labels = torch.Tensor()
#     print('Make prediction for {} samples...'.format(len(loader.dataset)))
#     with torch.no_grad():
#         for data in loader:
#             data = data.to(device)
#             output = model(data)
#             total_preds = torch.cat((total_preds, output.cpu()), 0)
#             total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
#     return total_labels.numpy().flatten(), total_preds.numpy().flatten()
#
# # -------------------------------
# # Hyperparameters & settings
# # -------------------------------
# datasets = [['davis', 'kiba', 'metz', 'BindingDB'][int(sys.argv[1])]]
# modeling = [MoeLayer][int(sys.argv[2])]
# model_st = modeling.__name__
#
# cuda_name = "cuda:0"
# if len(sys.argv) > 3:
#     cuda_name = "cuda:" + str(int(sys.argv[3]))
# device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
# torch.backends.cudnn.enabled = False  # for determinism
#
# TRAIN_BATCH_SIZE = 100
# TEST_BATCH_SIZE = 100
# LR = 0.0005
# LOG_INTERVAL = 20
# NUM_EPOCHS = 10
# NUM_FOLDS = 5
#
# # -------------------------------
# # Main program
# # -------------------------------
# for dataset in datasets:
#     print('\nrunning on ', model_st + '_' + dataset)
#
#     # Define processed data file paths
#     processed_data_file_train = 'data/processed/' + dataset + '_train.pt'
#     processed_data_file_test  = 'data/processed/' + dataset + '_test.pt'
#
#     if not (os.path.isfile(processed_data_file_train) and os.path.isfile(processed_data_file_test)):
#         print('Please run create_data.py to prepare data in pytorch format!')
#         continue
#
#     # Load datasets
#     train_data = TestbedDataset(root='data', dataset=dataset + '_train')
#     test_data  = TestbedDataset(root='data', dataset=dataset + '_test')
#
#     test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)
#
#     # -------------------------------
#     # 全局预训练模型文件路径
#     # -------------------------------
#     global_pretrained_model = f'model_{model_st}_{dataset}.model'
#     if not os.path.isfile(global_pretrained_model):
#         print(f"⚠️ No global pretrained model found at {global_pretrained_model}, training from scratch.")
#     else:
#         print(f"Using global pretrained model: {global_pretrained_model}")
#
#     # -------------------------------
#     # 5-Fold CV on train_data
#     # -------------------------------
#     kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
#
#     fold_results = []
#
#     for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(train_data)))):
#         print(f"\n===== Fold {fold + 1} / {NUM_FOLDS} =====")
#
#         train_subset = Subset(train_data, train_idx)
#         val_subset   = Subset(train_data, val_idx)
#
#         train_loader = DataLoader(train_subset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
#         val_loader   = DataLoader(val_subset, batch_size=TEST_BATCH_SIZE, shuffle=False)
#
#         # Initialize model, optimizer, loss_fn
#         model = modeling().to(device)
#
#         # ---- 加载全局预训练模型 ----
#         if os.path.isfile(global_pretrained_model):
#             print(f"Loading global pretrained weights from {global_pretrained_model} ...")
#             model.load_state_dict(
#                 torch.load(global_pretrained_model, map_location=device),
#                 strict=False
#             )
#
#         loss_fn = nn.SmoothL1Loss(beta=1.0)
#         optimizer = torch.optim.Adam(model.parameters(), lr=LR)
#
#         best_val_mse = 1000
#         best_epoch = -1
#         model_file_name  = f'model_{model_st}_{dataset}_fold{fold + 1}.model'
#         result_file_name = f'result_{model_st}_{dataset}_fold{fold + 1}.csv'
#
#         # Training loop
#         for epoch in range(NUM_EPOCHS):
#             train(model, device, train_loader, optimizer, epoch + 1)
#
#             # Evaluate on validation fold
#             G_val, P_val = predicting(model, device, val_loader)
#             ret_val = [rmse(G_val, P_val), mse(G_val, P_val), pearson(G_val, P_val), spearman(G_val, P_val), ci(G_val, P_val)]
#
#             # Save best model (by validation MSE)
#             if ret_val[1] < best_val_mse:
#                 torch.save(model.state_dict(), model_file_name)
#                 with open(result_file_name, 'w') as f:
#                     f.write(','.join(map(str, ret_val)))
#
#                 best_val_mse = ret_val[1]
#                 best_epoch = epoch + 1
#                 print(f'Val RMSE improved at epoch {best_epoch}; best_val_mse: {best_val_mse}, CI: {ret_val[-1]}')
#
#         # Load best model for this fold and evaluate on test set
#         model.load_state_dict(torch.load(model_file_name, map_location=device))
#         G_test, P_test = predicting(model, device, test_loader)
#         test_ret = [rmse(G_test, P_test), mse(G_test, P_test), pearson(G_test, P_test), spearman(G_test, P_test), ci(G_test, P_test)]
#         print(f"Fold {fold + 1} Test results: RMSE={test_ret[0]:.4f}, MSE={test_ret[1]:.4f}, CI={test_ret[-1]:.4f}")
#         fold_results.append(test_ret)
#
#     # -------------------------------
#     # Print 5-fold average on test set
#     # -------------------------------
#     fold_results = np.array(fold_results)
#     mean_results = np.mean(fold_results, axis=0)
#     print(f"\n===== 5-Fold Test Set Average ({model_st}_{dataset}) =====")
#     print(f"Mean RMSE: {mean_results[0]:.4f}, Mean MSE: {mean_results[1]:.4f}, Mean CI: {mean_results[-1]:.4f}")
