import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ml_logic.data import *
from ml_logic.preprocessor import preprocess_features
from ml_logic.edges_nodes import *
from ml_logic.edges_nodes import create_nodes_edges
from ml_logic.model import *
from ml_logic.model import FraudGNN
from ml_logic.model import model_data
from ml_logic.model import train_test
from ml_logic.train import train_model
from ml_logic.evaluate import predict
from ml_logic.evaluate import evaluate_model



app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/predict")
def aml_detector():
    """Dummy function for prediction in production"""
    df = get_data_local()
    df = clean_data(df)
    df = data_reduction(df)
    df = preprocess_features(df)
    G = create_nodes_edges(df)
    x, target = model_data(G, df)
    x, test_x, target, test_target = train_test(x, target)
    optimizer, model, target = train_model(FraudGNN, x, target)
    binary_predictions = predict(model, test_x)
    result = evaluate_model(test_target, binary_predictions)
    return {'Prediction': f"{binary_predictions}",
            'Accuracy': f"{result[0].item()}",
            'Precision': f"{result[1].item()}",
            'Recall': f"{result[1].item()}",
            'f1': f"{result[1].item()}",
            'AUROC': f"{result[1].item()}"}



@app.get("/")
def root():
    data = get_data_local()
    size = data.shape
    return {'greeting': "Hello" }
