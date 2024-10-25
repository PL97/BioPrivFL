from FL.fedavg import FedAvgAggregator, FedAvgClient, FedAvgServer
from FL.phe_utils import ArrayEncrypt, DigitEncrypt
import torch
import copy
import phe.paillier as paillier

class FedAvgAggregatorWithPHE(FedAvgAggregator):

    def setpubkey(self, pubkey):
        self.pubkey = pubkey
    @torch.no_grad()
    def aggregate(self):
        '''
        here to define the aggregation rule
        Fedavg: we aggregate the weight using weighted sum, to simplify the computation, we only perform sum in the aggregation step
        assuming the weights already applied to the clients
        '''
        # Sanity check: Ensure all models have the same parameter keys
        model_keys = [set(state_dict.keys()) for _, state_dict in self.clients_model_weights.items()]
        if not all(keys == set(self.dummy_model.state_dict().keys()) for keys in model_keys):
            raise ValueError("Mismatch in model parameters! Ensure all models have the same structure.")

        agg_state_dict = {}
        for key in self.dummy_model.state_dict().keys():
            # num_batches_tracked is a non trainable LongTensor and
            # num_batches_tracked are the same for all clients for the given datasets
            if 'num_batches_tracked' not in key:
                aggregated_weights = torch.zeros_like(self.dummy_model.state_dict()[key], dtype=torch.float32).tolist()
                aggregated_weights = ArrayEncrypt(aggregated_weights, self.pubkey)
                # temp = torch.zeros_like(server_model.state_dict()[key], dtype=type(server_model.state_dict()[key]))
                for cname, state_dict in self.clients_model_weights.items():
                    aggregated_weights += state_dict[key] * self.client_weights[cname]
                agg_state_dict[key] = aggregated_weights


class FedAvgServerWithPHE(FedAvgServer):
    pass

class FedAvgClientrWithPHE(FedAvgClient):
    
    def setkeypairs(self, pubkey, privatekey):
        self.pubkey = pubkey
        self.privatekey = privatekey

    def get_copy_of_model_weights(self):
        ## encrpt the model weights before share
        encrpted_weights = {}
        for k, w in self.model.state_dict().items():
            if w.dim() > 0:
                encrpted_weights[k] = ArrayEncrypt(w.detach().cpu().numpy().tolist(), self.pubkey)
            else:
                encrpted_weights[k] = DigitEncrypt(w.item(), self.pubkey)
        return copy.deepcopy(encrpted_weights)
    
    def update_local_model(self, state_dict):
        for k, v in state_dict:
            plaintext_weights = torch.tensor(v.get_plaintext(self.privatekey))
            self.model.state_dict()[k].data.copy_(plaintext_weights)

class FedAvgWithPHE:
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

        ## generate pubkey and private key
        pubkey, privatekey = paillier.generate_paillier_keypair()

        ## setup models and training configs
        aggregator = FedAvgAggregatorWithPHE(client_weights, device)
        aggregator.setpubkey(pubkey)
        self.server = FedAvgServerWithPHE(aggregator)
        self.clients = {}
        for tmp_cname in client_names:
            self.clients[tmp_cname] = FedAvgClientrWithPHE(saved_dir=f"{saved_dir}/{tmp_cname}/",
                                                    dl=dls[tmp_cname], 
                                                    criterion=criterions[tmp_cname],
                                                    lr=lrs[tmp_cname],
                                                    device=device,
                                                    amp=amp,
                                                    args=args)
            self.clients[tmp_cname].setkeypairs(pubkey, privatekey)

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


