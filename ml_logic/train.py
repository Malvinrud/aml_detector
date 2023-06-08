import numpy as np

import torch
import torch.nn as nn



######## TF imports here ########




def train_model(FraudGNN, x, target):

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


    return optimizer.step, model, target
