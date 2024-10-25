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
from sklearn.datasets import load_iris
from models import MLP
from collections import Counter

sys.path.append("./")
from FL.fedavg import FedAvg
from FL.fedavg_phe import FedAvgWithPHE
from FL.utils import set_seed

def create_fl_split(features, labels, n_clients, bs, shuffle=True):
    if shuffle:
        idx = list(range(features.shape[0]))
        np.random.shuffle(idx)
        features = features[idx]
        labels = labels[idx]

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
        stats = {"data_size": tmp_features.shape[0], "feature_dim": tmp_features.shape[1], "num_labels": len(set(tmp_labels)), "label_distribution": Counter(tmp_labels)}

        all_dls[f'client_{i}'] = dls
        all_stats[f'client_{i}'] = stats
    return all_dls, all_stats


def load_rps_data(bs=512, n_clients=2):
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
    all_dls, all_stats = create_fl_split(features, labels, n_clients, bs)

    return all_dls, all_stats

def load_uci_data(bs=512, n_clients=2):
    iris_data = load_iris()
    features, labels = iris_data.data, iris_data.target
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    all_dls, all_stats = create_fl_split(features, labels, n_clients, bs)
    return all_dls, all_stats


def main():
    seed = 1
    set_seed(seed)

    #global LOG_FILE_NAME
    n_clients = 5
    log_dir = f"./logs/FL_PHE/iris_{n_clients}/{seed}/"
    os.makedirs(log_dir, exist_ok=True)
   
    # dls, stats = load_data(dpath='data/data.npz')
    dls, stats = load_uci_data(n_clients=n_clients, bs=128)
    
    # Training setups
    criterions = {f"client_{i}": nn.CrossEntropyLoss() for i in range(n_clients)}
    client_weights = {f"client_{i}": 1.0/n_clients for i in range(n_clients)}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    lrs ={f"client_{i}": 1e-3 for i in range(n_clients)}
    max_epochs = 50
    aggregation_freq = 1

    config = {'feature_dim': stats['client_0']['feature_dim'], 'num_labels': stats['client_0']['num_labels']}

    model = MLP(config['feature_dim'], hidden_dim=100, num_layers=10, output_dim=config['num_labels'])

    fedavg = FedAvg(model, dls, lrs, criterions, max_epochs, client_weights, aggregation_freq, device, saved_dir=log_dir, config=config)
    
    fedavg.simulation()

if __name__ == "__main__":
    main()
