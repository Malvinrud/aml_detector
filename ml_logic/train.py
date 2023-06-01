import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from edges_nodes import create_nodes_edges
from data import clean_data
from model import FraudGNN


######## TF imports here ########




def train_model():

    FraudGNN = FraudGNN
    G = create_nodes_edges()
    df = clean_data()
    # Prepare the data for input into the model
    edge_list = list(G.edges(data=True))
    x = []
    for edge in edge_list:
        edge_values = list(edge[2].values())
        print(edge_values)
        edge_values = [float(i[0]) if type(i) == tuple and type(i[0]) == str else i[0] if type(i) == tuple else i for i in edge_values]
        x.append(edge_values)
    x = torch.tensor(x, dtype=torch.float)

    target = torch.tensor(df['is_laundering'].values, dtype=torch.float)

    input_dim = len(x[0])
    hidden_dim = 16
    model = FraudGNN(input_dim, hidden_dim)
    num_epochs=201

    # Define the loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for i in range(num_epochs):
        # Forward pass
        output = model(x)
        # Compute the loss
        loss = criterion(output, target)
        if i % 20 == 0:
            print(f'Epoch: {i}, Loss: {loss.item()}')
        # Zero the gradients
        optimizer.zero_grad()
        # Perform backpropagation
        loss.backward()
        # Update the parameters
        optimizer.step()

    return model, target
