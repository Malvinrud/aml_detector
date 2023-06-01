import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from edges_nodes import create_nodes_edges
from data import clean_data


######## TF imports here ########



class FraudGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(FraudGNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x.squeeze(-1)
