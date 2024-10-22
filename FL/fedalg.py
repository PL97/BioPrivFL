import torch
import sys
sys.path.append("../")
import torch
import torch.nn as nn
import numpy as np
import copy
from torch.optim import SGD, AdamW
from transformers import get_linear_schedule_with_warmup
import os
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
import gc
import torch

class Aggregator():
    def __init__(self, client_weights, device):
        self.client_weights = client_weights
        self.clients_model_weights = {}
        self.device = device

    def fetch_local_model(self, client_state_dict):
        self.client_state_dict = client_state_dict

    def aggregate(**args):
        pass


class Server():
    def __init__(self, aggregator: Aggregator):
        self.agg = aggregator
        self.clients_model_weights = {}
        self.client_state_dict = {}

    def init_model(self, **args):
        pass

    def global_validate(self):
        pass

    def aggregate(self):
        pass

    

class Client():
    def __init__(self, dl, criterion, lr, device, saved_dir, amp=False, **args):
        self.saved_dir = saved_dir
        self.dl = dl
        self.lr = lr
        self.criterion = criterion
        self.device = device
        self.args = args
        
        ## AMP
        self.scaler = torch.cuda.amp.GradScaler() if amp else None
    
    def get_copy_of_model_weights(self):
        return copy.deepcopy(self.model.state_dict())
    
    def init_model(self, **args):
        pass
    
    def local_train(self):
        pass
    
    def local_validate(self, idx):
        pass
    
    def save_models(self, file_name="best.pt"):
        torch.save(self.client_state_dict[client_idx], f"./{self.saved_dir}/site-{client_idx+1}/{file_name}")