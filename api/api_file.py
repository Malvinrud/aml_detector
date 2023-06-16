import pandas as pd
import json
from io import StringIO
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware


from ml_logic.preprocessor import preprocess_features
from ml_logic.edges_nodes2 import *
from ml_logic.model import *
from ml_logic.evaluate import predict
from ml_logic.worklfow import *

import networkx as nx
import pandas as pd
import plotly.graph_objects as go

app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.post("/predict")
def aml_detector(myfile: UploadFile = File(...)):


    contents = myfile.file.read()

    decoded_str = contents.decode('utf-8')

    df = pd.read_json(decoded_str)

    ###################################################

    df = clean_data(df)

    y = df["is_laundering"]

    X = df.drop(["is_laundering"], axis=1)

    X, X_test, y, y_test = train_test_split(X, y, test_size=0.2)

    model, X_test_tensor, y_test_tensor = model_workflow(X, X_test, y, y_test)

    results = predict(model, X_test_tensor)

    print(results)

    ###################################################


    ###### If predict function is used:

    results = results.tolist()

    print(type(results))

    return results



@app.post("/plot")
def aml_plot(myfile: UploadFile = File(...)):

    pass



@app.get("/")
def root():
    data = get_data_local()
    size = data.shape
    return {'greeting': "Hello" }
