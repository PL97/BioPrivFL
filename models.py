import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=100, num_layers=10, output_dim=2):
        super().__init__()
        bias = False
        layers = [nn.Linear(input_dim, hidden_dim, bias=bias)]
        for _ in range(num_layers-2):
            layers.append(nn.BatchNorm1d(num_features = hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
        self.features = nn.Sequential(*layers)

        if output_dim == 1:
            self.fc_layers = nn.Sequential(
                        nn.Linear(hidden_dim, output_dim, bias=bias),
                        nn.Sigmoid())
        else:
            self.fc_layers = nn.Linear(hidden_dim, output_dim, bias=bias)

    def forward(self, X):
        x = self.features(X)
        return self.fc_layers(x)

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
                              
class Model(nn.Module):
    def __init__(self, N_var, l1_lambda=5e-5):
        super(BaselineModel, self).__init__()

        # Define layers
        self.fc1 = nn.Linear(N_var, 100)
        self.sigmoid1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(100)

        self.fc2 = nn.Linear(100, 5)
        self.sigmoid2 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(5)

        self.fc3 = nn.Linear(5, 1)

        self.l1_lambda = l1_lambda  # L1 regularization coefficient

    def forward(self, x):
        # Forward pass
        # x = self.dropout1(x)
        x = self.fc1(x)
        x = self.sigmoid1(x)
        x = self.bn1(x)

        # x = self.dropout2(x)
        x = self.fc2(x)
        x = self.sigmoid2(x)
        x = self.bn2(x)

        x = self.fc3(x)

        # Calculate L1 regularization term
        l1_reg = torch.tensor(0., requires_grad=True)
        for param in self.parameters():
            l1_reg = l1_reg + torch.sum(torch.abs(param))

        # Scale L1 regularization
        l1_loss = self.l1_lambda * l1_reg

        return x, l1_loss
                              
