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

LOG_FILE_NAME = ""
eps = 1e-5


def myprint(s):
    global LOG_FILE_NAME
    print(s)
    with open(LOG_FILE_NAME, 'a') as file:
        file.write(s)
        file.write('\n')

def set_seed(seed):
    """Set all random seeds and settings for reproducibility (deterministic behavior)."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def load_data(dpath, bs=512):
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


    ## split train and test
    total_n = features.shape[0]
    train_n = int(total_n * 0.8)
    idx = list(range(total_n))
    np.random.shuffle(idx)
    train_X, train_y = features[idx[:train_n]], labels[idx[:train_n]]
    test_X, test_y = features[idx[train_n:]], labels[idx[train_n:]]


    train_dl = FastTensorDataLoader(train_X, train_y, batch_size=bs, shuffle=True)
    test_dl = FastTensorDataLoader(test_X, test_y, batch_size=bs, shuffle=False)
    dls = {"train": train_dl, "test": test_dl}
    stats = {"data_size": features.shape[0], "feature_dim": features.shape[1]}
    return dls, stats

def train_step(train_dl, model, criterion, optimizer, device):
    model = model.to(device)
    model.train()
    epoch_wise_loss = 0
    for X, y in tqdm(train_dl):
        optimizer.zero_grad()  # Clear gradients

        y = y.float().unsqueeze(dim=1).to(device)
        X = X.float().to(device)
        y_pred, l1_reg = model(X)
        loss = criterion(y_pred, y) + l1_reg

        # Backward pass and optimization
        
        loss.backward()        # Backpropagation
        optimizer.step()       # Update weights
        
        epoch_wise_loss += loss.item()
    return epoch_wise_loss


@torch.no_grad()
def eval_step(test_dl, model, device, criterion=None):
    model = model.to(device)
    model.eval()
    all_predictions = []
    all_targets = []
    all_loss = 0
    for X, y in tqdm(test_dl):
        y = y.float().unsqueeze(dim=1).to(device)
        X = X.float().to(device)
        y_pred, l1_reg = model(X)
        if criterion != None:
            loss = criterion(y_pred, y) + l1_reg
            all_loss += loss.item()
        all_predictions.extend(y_pred.cpu().numpy())
        all_targets.extend(y.cpu().numpy())
    # Calculate R2 score
    r2 = r2_score(all_targets, all_predictions)
    return all_loss, r2
    



def train(train_dl, model, criterion, optimizer, epochs, device, test_dl=None):
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        
        # Forward pass
        epoch_wise_loss = train_step(train_dl, model, criterion, optimizer, device)
        
        # eval by r2 score
        loss, r2 = eval_step(train_dl, model, device, criterion)
        print_str = ""
        print_str += f'Epoch {epoch+1}/{epochs}\nTrain Loss: {loss} \t R2: {r2}'
        if test_dl != None:
            loss, r2 = eval_step(test_dl, model, device, criterion)
            print_str += f' \t Test Loss: {loss} \t R2: {r2}'
        myprint(print_str)
        
    
    return model


def main():

    seed = 0
    set_seed(seed)

    #global LOG_FILE_NAME
    log_dir = f"./logs/baseline1/{seed}/"
    os.makedirs(log_dir, exist_ok=True)

    global LOG_FILE_NAME

    LOG_FILE_NAME += f"{log_dir}/log.txt"
    with open(LOG_FILE_NAME, 'w') as file:
        file.write(f'========= start to log ================\n')

    # load data from saved npz file
    dls, stats = load_data(dpath='data/data.npz')
    
    # create baseline model with specified feature dimension
    model = BaselineModel(stats['feature_dim'], l1_lambda=1e-3)


    # Training setups
    epochs = 300
    criterion = nn.MSELoss(reduce='mean')
    optimizer = optim.SGD(model.parameters(), lr=1e-3)

    # criterion = nn.L1Loss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train(dls['train'], model, criterion, optimizer, epochs, device, test_dl=dls['test'])


    torch.save(model.state_dict(), f"{log_dir}/final.pt")
    

if __name__ == "__main__":
    main()
