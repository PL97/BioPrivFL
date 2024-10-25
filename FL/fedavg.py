from FL.fedalg import Server, Client, Aggregator
from FL.utils import _shared_eval_step, _shared_train_step
import torch.optim as optim
import torch.nn as nn
import torch
import copy
from torch.utils.tensorboard import SummaryWriter

class FedAvgAggregator(Aggregator):
    @torch.no_grad()
    def aggregate(self):
        '''
        here to define the aggregation rule
        Fedavg: we aggregate the weight using weighted sum 
        '''
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
                self.agg_weights[key] = aggregated_weights


class FedAvgServer(Server):
    def init_model(self, model):
        '''
        input: model. The model shared by all clients
        Here we initialize the server model and also dummy_model in aggregator
        '''
        self.model = copy.deepcopy(model)
        self.agg.dummy_model = copy.deepcopy(model)
    
    def update_client_model_weight(self, cname, model_state_dict):
        '''
        input: cname (str), the clients names that need to to updates to server (send to server)
        model_state_dict (dict), the model weights of client "cname"
        '''
        self.agg.clients_model_weights[cname] = copy.deepcopy(model_state_dict)
    
    def aggregate(self):
        """
        aggregate the model, typically following the step:
            1) fetch all the local updates from all clients
            2) aggregate received the udpates and aggregate the weights
            3) server received the aggregated model from aggregator
        """
        self.agg.aggregate()
    
    def get_agg_weights(self):
        return self.agg.agg_weights

class FedAvgClient(Client):
    def init_model(self, model):
        """
        input: model, model shared by all clients
        initialize the client model, optimizer, and logging variables (epoch and tensorflow writer)
        """
        self.epoch = 0
        self.model = copy.deepcopy(model)
        # self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.writer = SummaryWriter(log_dir=f"{self.saved_dir}/tb_events/")

    def local_train(self, aggregation_freq):
        for _ in range(aggregation_freq):
            _shared_train_step(self.dl['train'], self.model, self.criterion, self.optimizer, self.device)

    def local_validate(self):
        """
        evaluate the local model by test on its training and validation data and also log the result to tensorboard
        """
        self.epoch += 1
        train_results = _shared_eval_step(self.dl['train'], self.model, self.device, criterion=self.criterion)
        test_results = _shared_eval_step(self.dl['test'], self.model, self.device, criterion=self.criterion)
        for k in train_results:
            self.writer.add_scalar(f'train_{k}', train_results[k], self.epoch)
            self.writer.add_scalar(f'test_{k}', test_results[k], self.epoch)
        return train_results, test_results

    def get_copy_of_model_weights(self):
        return copy.deepcopy(self.model.state_dict())
    
    def update_local_model(self, state_dict):
        for key, param in state_dict.items():
            if 'num_batches_tracked' not in key:
                self.model.state_dict()[key].data.copy_(param)

class FedAvg:
    def __init__(self, model, dls, lrs, criterions, max_epochs, client_weights, aggregation_freq, device, saved_dir, amp=False, **args):
        """
        input: model
        dls (dict): keys are the name of the clients, values are the dataloader for the corresponding client
        lrs (dict): same as above but the value is the learning rate
        criterion (dict): same as above but the value is the loss function
        max_epochs (int): number of aggregation round
        client_weights (dict): same as dls but the vlaue is the weight in [0, 1] used for aggregation
        aggregation_freq (int): number of local updates (epoch-wise)
        device (torch.device): cuda or cpu
        saved_dir (str): logging path
        amp (bool): turn on mix precision for acceraltion of local training
        """
        self.config = args['config']
        self.max_epochs = max_epochs
        client_names = dls.keys()
        num_clients = len(dls.keys())
        self.model = model
        self.aggregation_freq = aggregation_freq

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
        """
        federated learning simulation on one device
        basically follows the steps as:
            1. initialize server and clients models
            2. clients do local update (parallel)
            3. clients send updates to server and awaiting aggregation
            4. server aggregate the updates
            5. server send updates to clients, and clients sync the local model
        """
        self.server.init_model(self.model)
        for _, c in self.clients.items():
            c.init_model(self.model)
        for _ in range(self.max_epochs):
            print(f"========== start FL iteration: {_} ==========")
            ## initialize all the models
            for cname, client in self.clients.items():
                print(f"start training {cname}")
                client.local_train(self.aggregation_freq)
                train_score, test_score = client.local_validate()
                print(f"train: {train_score} \t test: {test_score}")
                self.server.update_client_model_weight(cname, client.get_copy_of_model_weights())
            
            print(f"start aggregation at server")
            self.server.aggregate()
            
            ## send udpates back to clients
            for cname, client in self.clients.items():
                client.update_local_model(self.server.get_agg_weights())


