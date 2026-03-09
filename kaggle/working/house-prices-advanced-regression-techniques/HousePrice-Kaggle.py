import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(42)

#***********************************#
#          Data  Preprocess         #
#***********************************#

# Training set

train_file_path = os.path.join(SCRIPT_DIR, "../../input/house-prices-advanced-regression-techniques/train.csv")
dataset_df = pd.read_csv(train_file_path)
dataset_df = dataset_df.drop('Id', axis=1)
num_df_idx = dataset_df.select_dtypes(include=['float64', 'int64']).columns
cat_df_idx = dataset_df.select_dtypes(exclude=['float64', 'int64']).columns

# numerical features (exclude SalePrice from standardization)
dataset_df['SalePrice'] = np.log1p(dataset_df['SalePrice'])
num_features = [c for c in num_df_idx if c != 'SalePrice']
data_mean = dataset_df[num_features].mean()
data_std = dataset_df[num_features].std()
dataset_df[num_features] = (dataset_df[num_features] - data_mean) / data_std
dataset_df[num_features] = dataset_df[num_features].fillna(0)
# catagorical features
dataset_df = pd.get_dummies(dataset_df, dummy_na=True, dtype=float)
# 确保 SalePrice 在最后一列
cols = [c for c in dataset_df.columns if c != 'SalePrice'] + ['SalePrice']
dataset_df = dataset_df[cols]
dataset = torch.tensor(dataset_df.astype('float64').to_numpy(), dtype=torch.float32)

# Test set

test_file_path = os.path.join(SCRIPT_DIR, "../../input/house-prices-advanced-regression-techniques/test.csv")
testset_df = pd.read_csv(test_file_path)
testset_id = testset_df['Id']
testset_df = testset_df.drop('Id', axis=1)
# numerical features (use same mean/std from training set)
testset_df[num_features] = (testset_df[num_features] - data_mean) / data_std
testset_df[num_features] = testset_df[num_features].fillna(0)
# catagorical features
testset_df = pd.get_dummies(testset_df, dummy_na=True, dtype=float)
# align test columns with train columns (drop SalePrice which is in train only)
train_feature_cols = [c for c in dataset_df.columns if c != 'SalePrice']
testset_df = testset_df.reindex(columns=train_feature_cols, fill_value=0)
testset = torch.tensor(testset_df.astype('float64').to_numpy(), dtype=torch.float32)
                               
class MyDataset(Dataset):
    def __init__(self, dataset):
        self.X = dataset[:, :-1]
        self.y = dataset[:, -1:]
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


#*******************************#
#            Train              #
#*******************************#

print(f"pandas:\ndataset: {dataset_df.shape}\ntestset: {testset_df.shape}")
print(f"torch:\ndataset: {dataset.shape}\ntestset: {testset.shape}")

in_features = dataset.shape[1] - 1
def make_net(in_features):
    return nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Linear(256, 1),
    )
loss = nn.MSELoss()

def log_rmse(net, features, labels):
    # target 已经是 log1p 空间，直接算 RMSE 就等于 RMSLE
    pred = net(features)
    return torch.sqrt(loss(pred, labels))

def train(net, dataset, dataset_vali, epochs, lr, weight_decay, batch_size):
    """
    net: nn.Sequential
    dataset: torch.Tensor
    dataset_vali: torch.Tensor
    epochs: int, num of epoch
    lr: float
    weight_decay: float
    batch_size: int
    """
    batch_dataset = MyDataset(dataset)
    batch_dataset_vali = MyDataset(dataset_vali)
    dataloader = DataLoader(batch_dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(params=net.parameters(), lr=lr, weight_decay=weight_decay)
    
    train_loss = []
    vali_loss = []
    
    for _ in range(epochs):
        net.train()
        for X, y in dataloader:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        net.eval()
        with torch.no_grad():
            X = dataset[:, :-1]
            y = dataset[:, -1:]
            X_vali = dataset_vali[:, :-1]
            y_vali = dataset_vali[:, -1:]
            train_loss.append(log_rmse(net, X, y).item())
            vali_loss.append(log_rmse(net, X_vali, y_vali).item())

    return train_loss, vali_loss

epochs, lr, weight_decay, batch_size = 200, 5e-4, 1e-3, 64
# net = make_net(in_features)
# train(net, epochs, lr, weight_decay, batch_size)


#******************************#
#           predict            #
#******************************#


def train_and_predict(dataset_train, dataset_test, epochs, lr, weight_decay, batch_size):
    """
    params: 
        dataset_train: torch.tensor, training set, n*(d+1)
        dataset_test: torch.tensor, test features, n*d
        other params same as train()
    return:
        y_pred: torch.tensor
    """
    in_features = dataset_train.shape[1] - 1
    net = make_net(in_features)
    
    batch_dataset = MyDataset(dataset_train)
    dataloader = DataLoader(batch_dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(params=net.parameters(), lr=lr, weight_decay=weight_decay)
    
    # train
    for _ in range(epochs):
        net.train()
        for X, y in dataloader:
            # targets already log1p transformed during preprocessing
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
    
    # predict
    net.eval()
    with torch.no_grad():
        y_pred = torch.expm1(net(dataset_test))
    return y_pred

y_pred = train_and_predict(dataset, testset, epochs, lr, weight_decay, batch_size)
submission = pd.DataFrame({
    'Id': testset_id,
    'SalePrice': y_pred.squeeze().numpy()
})
submission.to_csv(os.path.join(SCRIPT_DIR, '../../output/house-prices-advanced-regression-techniques/submission.csv'), index=False)
