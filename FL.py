#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
import torch.optim as optim
from models import BaselineModel
from datasets import FastTensorDataLoader
import os
import random
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.metrics import r2_score
import sys

sys.path.append("./")
from FL.fedavg import FedAvg

def set_seed(seed):
    """Set all random seeds and settings for reproducibility (deterministic behavior)."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def load_data(dpath, bs=512, n_clients=2):
    # loaded_data = np.load(dpath)
    # feature_name = loaded_data['feature_name']
    # features = loaded_data['features'][:60000, 6:]
    # labels = loaded_data['labels'][:60000]

    loaded_data = np.load("data/debug_data.npz")
    features, labels = loaded_data['features'], loaded_data['labels']
    
    # remove instance with nan labels
    clean_idx = ~np.isnan(labels)
    features, labels = features[clean_idx], labels[clean_idx]


    # standardize the input features 
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    ## create splits for clients
    sample_per_client = features.shape[0]//n_clients
    all_dls, all_stats = {}, {}
    for i in range(n_clients):
        tmp_features = features[i*sample_per_client:(i+1)*sample_per_client]
        tmp_labels = labels[i*sample_per_client:(i+1)*sample_per_client]

        ## split train and test
        total_n = tmp_features.shape[0]
        train_n = int(total_n * 0.8)
        idx = list(range(total_n))
        np.random.shuffle(idx)
        train_X, train_y = tmp_features[idx[:train_n]], tmp_labels[idx[:train_n]]
        test_X, test_y = tmp_features[idx[train_n:]], tmp_labels[idx[train_n:]]

        train_dl = FastTensorDataLoader(train_X, train_y, batch_size=bs, shuffle=True)
        test_dl = FastTensorDataLoader(test_X, test_y, batch_size=bs, shuffle=False)
        dls = {"train": train_dl, "test": test_dl}
        stats = {"data_size": features.shape[0], "feature_dim": features.shape[1]}

        all_dls[f'client_{i}'] = dls
        all_stats[f'client_{i}'] = stats
    return all_dls, all_stats

def main():
    seed = 0
    set_seed(seed)

    #global LOG_FILE_NAME
    log_dir = f"./logs/FL/{seed}/"
    os.makedirs(log_dir, exist_ok=True)
    n_clients = 2
   
    dls, stats = load_data(dpath='data/data.npz')
    
    # Training setups
    criterions = {f"client_{i}": nn.MSELoss(reduction='mean') for i in range(n_clients)}
    client_weights = {f"client_{i}": 1/n_clients for i in range(n_clients)}

    # criterion = nn.L1Loss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    lrs ={f"client_{i}": 1e-3 for i in range(n_clients)}
    max_epochs = 10
    aggregation_freq=1
    l1_lambda = 1e-3

    config = {'feature_dim': stats['client_1']['feature_dim'], 'l1_lambda': l1_lambda}

    fedavg = FedAvg(dls, lrs, criterions, max_epochs, client_weights, aggregation_freq, device, saved_dir="log/FL/", config=config)
    
    fedavg.simulation()

if __name__ == "__main__":
    main()
