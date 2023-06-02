import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from edges_nodes import create_nodes_edges
from data import clean_data
from train import train_model

def predict_model():
    model, target = train_model()

    with torch.no_grad():
        predictions = torch.sigmoid(model(test_data))  # Apply sigmoid to obtain probabilities

    # Convert probabilities to binary predictions
    threshold = 0.5  # Set the threshold for classification
    binary_predictions = (predictions >= threshold).int()

    return binary_predictions


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
