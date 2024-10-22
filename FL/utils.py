import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import r2_score


def _shared_train_step(train_dl, model, criterion, optimizer, device):
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
def _shared_eval_step(test_dl, model, device, criterion=None):
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