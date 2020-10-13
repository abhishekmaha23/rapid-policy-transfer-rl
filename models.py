import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import torch


class MLPActor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, batch_norm=False, num_layers=1):
        super().__init__()
        self.batch_norm = batch_norm
        self.batch_norm_momentum: float = 0.0
        self.num_layers = num_layers
        self.fc_1 = nn.Linear(input_dim, hidden_dim)
        self.bn_1 = nn.BatchNorm1d(hidden_dim, momentum=self.batch_norm_momentum)
        self.fc_mid = nn.Linear(hidden_dim, hidden_dim)
        self.bn_mid = nn.BatchNorm1d(hidden_dim, momentum=self.batch_norm_momentum)
        self.fc_2 = nn.Linear(hidden_dim, output_dim)
        self.bn_2 = nn.BatchNorm1d(hidden_dim, momentum=self.batch_norm_momentum)
        # self.dropout = nn.Dropout(dropout)
        self.apply(init_weights)

    def forward(self, x):
        x = self.fc_1(x)
        # print(x.shape)
        if self.batch_norm is True:
            x = self.bn_1(x)
        # x = self.dropout(x)
        x = F.relu(x)
        if self.batch_norm is True:
            x = self.bn_2(x)
        x = F.softmax(self.fc_2(x), dim=-1)
        return x


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, batch_norm=False):
        super().__init__()
        self.batch_norm = batch_norm
        self.batch_norm_momentum: float = 0.0
        self.fc_1 = nn.Linear(input_dim, hidden_dim)
        self.bn_1 = nn.BatchNorm1d(hidden_dim, momentum=self.batch_norm_momentum)
        self.fc_2 = nn.Linear(hidden_dim, output_dim)
        self.bn_2 = nn.BatchNorm1d(output_dim, momentum=self.batch_norm_momentum)
        # self.dropout = nn.Dropout(dropout)
        self.apply(init_weights)

    def forward(self, x):
        x = self.fc_1(x)
        if self.batch_norm is True:
            x = self.bn_1(x)
        # x = self.dropout(x)

        x = F.relu(x)
        x = self.fc_2(x)
        if self.batch_norm is True:
            x = self.bn_2(x)

        return x


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0)


def get_random_agent(config, batch_norm=True):
    # random_size = np.random.randint(10, 200)
    random_size = np.random.randint(30, 80)
    # num_layers = random.choice([1, 2])
    num_layers = 1
    # random_size = 64
    return MLPActor(config.state_size, random_size, config.action_size, batch_norm, num_layers)
