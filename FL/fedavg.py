from FL.fedalg import Server, Client, Aggregator
from FL.utils import _shared_eval_step, _shared_train_step
import torch.optim as optim
import torch.nn as nn
import torch
import copy

class BaselineModel(nn.Module):
    def __init__(self, N_var, l1_lambda=5e-5):
        super(BaselineModel, self).__init__()

        # Define layers
        self.dropout1 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(N_var, 50)
        self.sigmoid1 = nn.Sigmoid()

        self.dropout2 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(50, 20)
        self.sigmoid2 = nn.Sigmoid()

        self.fc3 = nn.Linear(20, 10)
        self.sigmoid3 = nn.Sigmoid()

        self.fc4 = nn.Linear(10, 5)
        self.sigmoid4 = nn.Sigmoid()

        self.fc5 = nn.Linear(5, 1)
        self.sigmoid5 = nn.Sigmoid()

        self.l1_lambda = l1_lambda  # L1 regularization coefficient

    def forward(self, x):
        # Forward pass
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.sigmoid1(x)

        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.sigmoid2(x)

        x = self.fc3(x)
        x = self.sigmoid3(x)

        x = self.fc4(x)
        x = self.sigmoid4(x)

        x = self.fc5(x)
        x = self.sigmoid5(x)

        # Calculate L1 regularization term
        l1_reg = torch.tensor(0., requires_grad=True)
        for param in self.parameters():
            l1_reg = l1_reg + torch.sum(torch.abs(param))

        # Scale L1 regularization
        l1_loss = self.l1_lambda * l1_reg

        return x, l1_loss

class FedAvgAggregator(Aggregator):
    @torch.no_grad()
    def aggregate(self):
        # Sanity check: Ensure all models have the same parameter keys
        model_keys = [set(state_dict.keys()) for _, state_dict in self.clients_model_weights.items()]
        if not all(keys == set(self.dummy_model.state_dict().keys()) for keys in model_keys):
            raise ValueError("Mismatch in model parameters! Ensure all models have the same structure.")

        for key in self.dummy_model.state_dict().keys():
            # num_batches_tracked is a non trainable LongTensor and
            # num_batches_tracked are the same for all clients for the given datasets
            if 'num_batches_tracked' not in key:
                aggregated_weights = torch.zeros_like(self.dummy_model.state_dict()[key], dtype=torch.float32).to(self.device)
                # temp = torch.zeros_like(server_model.state_dict()[key], dtype=type(server_model.state_dict()[key]))
                for cname, state_dict in self.clients_model_weights.items():
                    aggregated_weights += self.client_weights[cname] * state_dict[key]
                self.dummy_model.state_dict()[key].data.copy_(aggregated_weights)

class FedAvgServer(Server):
    def init_model(self, N_var, l1_lambda):
        self.model = BaselineModel(N_var=N_var, l1_lambda=l1_lambda)
        self.agg.dummy_model = BaselineModel(N_var=N_var, l1_lambda=l1_lambda)
    
    def update_client_model_weight(self, cname, model_state_dict):
        self.agg.clients_model_weights[cname] = copy.deepcopy(model_state_dict)
    
    def aggregate(self):
        self.agg.fetch_local_model(self.client_state_dict)
        self.agg.aggregate()
        self.model.load_state_dict(self.agg.dummy_model.state_dict())

class FedAvgClient(Client):
    def init_model(self, N_var, l1_lambda):
        self.model = BaselineModel(N_var=N_var, l1_lambda=l1_lambda)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)

    def local_train(self):
        _shared_eval_step(self.dl['train'], self.model, self.device, self.criterion)

    def local_validate(self, epoch):
        loss, r2 = _shared_eval_step(self.dl['test'], self.model, self.device, criterion=None)
        return loss, r2

class FedAvg:
    def __init__(self, dls, lrs, criterions, max_epochs, client_weights, aggregation_freq, device, saved_dir, amp=False, **args):
        self.config = args['config']
        self.max_epochs = max_epochs
        client_names = dls.keys()
        num_clients = len(dls.keys())

        ## setup models and training configs
        aggregator = FedAvgAggregator(client_weights, device)
        self.server = FedAvgServer(aggregator)
        self.clients = {}
        for tmp_cname in client_names:
            self.clients[tmp_cname] = FedAvgClient(saved_dir=f"{saved_dir}/{tmp_cname}/",
                                                    dl=dls[tmp_cname], 
                                                    criterion=criterions[tmp_cname],
                                                    lr=lrs[tmp_cname],
                                                    device=device,
                                                    amp=amp,
                                                    args=args)

        ## AMP
        self.scaler = torch.cuda.amp.GradScaler() if amp else None
    
    def simulation(self):
        self.server.init_model(self.config['feature_dim'], l1_lambda=self.config['l1_lambda'])
        for _, c in self.clients.items():
            c.init_model(self.config['feature_dim'], l1_lambda=self.config['l1_lambda'])
        for _ in range(self.max_epochs):
            print(f"========== start FL iteration: {_} ==========")
            ## initialize all the models
            for cname, client in self.clients.items():
                print(f"start training {cname}")
                client.local_train()
                self.server.update_client_model_weight(cname, client.get_copy_of_model_weights())
            
            print(f"start aggregation at server")
            self.server.aggregate()
            
            ## send udpates back to clients
            for cname, client in self.clients.items():
                # self.client_state_dict[client_idx][key].data.copy_(server_model.state_dict()[key])
                for key, param in self.server.model.state_dict().items():
                    if 'num_batches_tracked' not in key:
                        client.model.state_dict()[key].data.copy_(param)

