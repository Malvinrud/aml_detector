import pandas as pd
from ml_logic.model import *
from sklearn.model_selection import train_test_split
from ml_logic.preprocessor import *
from ml_logic.edges_nodes2 import *
from ml_logic.train import *
from ml_logic.evaluate import *


def data_workflow(size="Small", fraud="HI", test_size=0.2):

    df = get_data_local(size="Small", fraud="HI")

    df = clean_data(df)

    y = df["is_laundering"]

    X = df.drop(["is_laundering"], axis=1)

    X, X_test, y, y_test = train_test_split(X, y, test_size=test_size)

    return X, X_test, y, y_test


def model_workflow(X, X_test, y, y_test):

    X = preprocess_features(X)

    X_test = preprocess_features(X_test)

    G = create_nodes_edges(X)

    G_test = create_nodes_edges(X_test)

    y = pd.DataFrame(y)

    y_test = pd.DataFrame(y_test)

    ####### don't forget to insert model params here and in functione defintion

    X_tensor, y_tensor = model_data(G, y)

    X_test_tensor, y_test_tensor = model_data(G_test, y_test)

    model = FraudGNN

    opti, model, target = train_model(FraudGNN, X_tensor, y_tensor)

    return model, X_test_tensor, y_test_tensor


def plot_prep(X_test, y_pred):

    y = y_pred.tolist()
    X_test["is_laundering"] = y

    return graph_df


def prediction_workflow(model, X_test, y_test):

    X_test = preprocess_features(X_test)

    G_test = create_nodes_edges(X_test)

    y = pd.DataFrame(y)

    X_test_tensor, y_test_tensor = model_data(G_test, y_test)

    prediction = predict(model, X_test)

    prediction = prediction.tolist()

    return prediction
