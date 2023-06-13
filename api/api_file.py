import pandas as pd
import json
from io import StringIO
from fastapi import FastAPI, File, UploadFile
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
from ml_logic.worklfow import *


api = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@api.post("/predict")
def aml_detector(myfile: UploadFile = File(...)):


    contents = myfile.file.read()

    decoded_str = StringIO(contents.decode('utf-8'))
    df = pd.read_csv(decoded_str, sep=",")

    df = clean_data(df)




    # Theoretically correct
    #############################################################



    # df = clean_data(df)
    # df = preprocess_features(df)
    # G = create_nodes_edges(df)
    # x, target = model_data(G, df)
    # x, test_x, target, test_target = train_test(x, target)
    # optimizer, model, target = train_model(FraudGNN, x, target)
    # binary_predictions = predict(model, test_x)
    # binary_predictions = binary_predictions.numpy().tolist()
    # print(binary_predictions)
    # return binary_predictions, len(binary_predictions)


    # Only for train/predict in production

    #############################################################

    y = df["is_laundering"]

    X = df.drop(["is_laundering"], axis=1)

    X, X_test, y, y_test = train_test_split(X, y, test_size=0.2)

    model, X_test_tensor, y_test_tensor = model_workflow(X, X_test, y, y_test)

    predictions = predict(model, x=X_test_tensor)

    #############################################################

    return predictions


# @api.post("/plot")
# def aml_detector(myfile: UploadFile = File(...)):





# @api.get("/")
# def root():
#     data = get_data_local()
#     size = data.shape
#     return {'greeting': "Hello" }
