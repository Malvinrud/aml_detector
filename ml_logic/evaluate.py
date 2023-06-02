import torch
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryF1Score, BinaryRecall, BinaryAUROC

def predict(model, x):

    model.eval()



    # Forward pass through the model to obtain predictions
    with torch.no_grad():
        predictions = torch.sigmoid(model(x))

    # Apply sigmoid to obtain probabilities

    # Convert probabilities to binary predictions
    threshold = 0.5  # Set the threshold for classification
    binary_predictions = (predictions >= threshold).int()

    # Print the binary predictions
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
