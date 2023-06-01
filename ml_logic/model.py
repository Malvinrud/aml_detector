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


def evaluate_model(y_pred, y_test):
    """
    Evaluate trained model performance on the y_test
    """

    accuracy = BinaryAccuracy()

    precision = BinaryPrecision()

    recall = BinaryRecall()

    f1 = BinaryF1Score()

    auroc = BinaryAUROC()

    accuracy = accuracy(y_pred, y_test)

    precision = precision(y_pred, y_test)

    recall = recall(y_pred, y_test)

    f1 = f1(y_pred, y_test)

    auroc = auroc(y_pred, y_test)


    print(f"✅ Model evaluated")
    print(f"accuracy: {accuracy}")
    print(f"precision: {precision}")
    print(f"recall: {recall}")
    print(f"f1: {f1}")
    print(f"auroc: {auroc}")

    return (accuracy, precision, recall, f1, auroc)
