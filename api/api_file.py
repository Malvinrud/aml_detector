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



app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.post('/uploaded_csv_json')
def upload_file(myfile: UploadFile = File(...)):
    contents = myfile.file.read() # Reading content of 'myfile' in bytes
    decoded_str = contents.decode('utf-8') # Decoding contents into str type
    print(decoded_str)
    #breakpoint()
    decoded_str = StringIO(contents.decode('utf-8')) # Alternative using StringIO
    #df_json = json.loads(decoded_str) # Reading string and converting to json (dictionary)
    df = pd.DataFrame(decoded_str) # Reading dictionary and converting into dataframe

    return df

@app.post("/predict")
def aml_detector(myfile: UploadFile = File(...)):

    contents = myfile.file.read()
    decoded_str = contents.decode('latin-1')
    decoded_str = StringIO(contents.decode('utf-8'))

    df = pd.read_csv(decoded_str, sep=",")

    print(df.columns)
    print(df.shape)

    #breakpoint()

    df = clean_data(df)
    df = preprocess_features(df)
    G = create_nodes_edges(df)
    x, target = model_data(G, df)
    x, test_x, target, test_target = train_test(x, target)
    optimizer, model, target = train_model(FraudGNN, x, target)
    binary_predictions = predict(model, test_x)
    binary_predictions = binary_predictions.numpy().tolist()
    print(binary_predictions)
    return binary_predictions



@app.get("/")
def root():
    data = get_data_local()
    size = data.shape
    return {'greeting': "Hello" }
