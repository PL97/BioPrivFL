import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import r2_score, balanced_accuracy_score, accuracy_score
import random
import os
import numpy as np

def set_seed(seed):
    """Set all random seeds and settings for reproducibility (deterministic behavior)."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def _shared_train_step(train_dl, model, criterion, optimizer, device):
    model = model.to(device)
    model.train()
    epoch_wise_loss = 0
    for X, y in tqdm(train_dl):
        optimizer.zero_grad()  # Clear gradients

        y = y.to(device)
        X = X.float().to(device)
        y_pred = model(X)
        loss = criterion(y_pred, y)

        # Backward pass and optimization
        
        loss.backward()        # Backpropagation
        optimizer.step()       # Update weights
        
        epoch_wise_loss += loss.item()
    return model


@torch.no_grad()
def _shared_eval_step(test_dl, model, device, criterion=None):
    model = model.to(device)
    model.eval()
    all_predictions = []
    all_targets = []
    all_loss = 0
    for X, y in tqdm(test_dl):
        y = y.to(device)
        X = X.float().to(device)
        y_pred = model(X)
        if criterion != None:
            loss = criterion(y_pred, y)
            all_loss += loss.item()
        all_predictions.extend(y_pred.cpu().numpy())
        all_targets.extend(y.cpu().numpy())
    # Calculate R2 score
    # r2 = r2_score(all_targets, all_predictions)
    all_y_pred = np.argmax(all_predictions, axis=1)
    acc = accuracy_score(all_targets, all_y_pred)
    results = {
        "acc": acc,
        "loss": all_loss,
    }
    return results