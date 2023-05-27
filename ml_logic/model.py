import numpy as np

from colorama import Fore, Style
from typing import Tuple

from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryF1Score, BinaryRecall, BinaryAUROC

# Timing the TF import
print(Fore.BLUE + "\nLoading TensorFlow..." + Style.RESET_ALL)
start = time.perf_counter()

######## TF imports here ########

from tensorflow import keras
from keras import Model, Sequential, layers, regularizers, optimizers
from keras.callbacks import EarlyStopping

end = time.perf_counter()
print(f"\n✅ TensorFlow loaded ({round(end - start, 2)}s)")



def initialize_model(df) -> Model:
    """
    Initialize the model
    """
    !pip install networkx
    import networkx as nx
    G = nx.MultiGraph()

        # Add nodes to the graph for each unique card_id, merchant_name
    G.add_nodes_from(df["Account"].unique(), type='Account')
    G.add_nodes_from(df["Account.1"].unique(), type='Account.1')


    for _, row in df_temp.iterrows():
    # Create a variable for each properties for each edge

        year = row["Year"],
        month = row["Month"],
        day = row["Day"],
        hour = row["Hour"],
        minute =row["Minute"],
        amount_paid = row["Amount Paid USD"],
        payment_format_ach =  row["Payment Format_ACH"],
        payment_format_bitcoin = row["Payment Format_Bitcoin"],
        payment_format_cash = row["Payment Format_Cash"],
        Payment_format_Cheque = row["Payment Format_Cheque"],
        Payment_Format_Credit_Card = row["Payment Format_Credit Card"],
        Payment_Format_Reinvestment = row["Payment Format_Reinvestment"],
        Payment_Format_Wire =row["Payment Format_Wire"],
        Currency_Code_BRL =  row["Currency Code_BRL"],
        Currency_Code_BTC = row["Currency Code_BTC"],
        Currency_Code_CHF = row["Currency Code_CHF"],
        Currency_Code_EUR = row["Currency Code_EUR"],
        Currency_Code_GBP = row["Currency Code_GBP"],
        Currency_Code_ILS =row["Currency Code_ILS"],
        Currency_Code_INR = row["Currency Code_INR"],
        Currency_Code_JPY =  row["Currency Code_JPY"],
        Currency_Code_MXN = row["Currency Code_MXN"],
        Currency_Code_RUB =row["Currency Code_RUB"],
        Currency_Code_SAR = row["Currency Code_SAR"],
        Currency_Code_USD =  row["Currency Code_USD"]

        G.add_edge(row['Account'], row['Account.1'], year = year , month = month , day = day ,
              hour = hour , minute = minute , amount_paid = amount_paid, payment_format_ach =  payment_format_ach,
              payment_format_bitcoin = payment_format_bitcoin,
            payment_format_cash = payment_format_cash,
            Payment_format_Cheque = Payment_format_Cheque,
            Payment_Format_Credit_Card = Payment_Format_Credit_Card,
            Payment_Format_Reinvestment = Payment_Format_Reinvestment,
            Payment_Format_Wire = Payment_Format_Wire,
            Currency_Code_BRL =  Currency_Code_BRL,
            Currency_Code_BTC = Currency_Code_BTC,
            Currency_Code_CHF = Currency_Code_CHF,
            Currency_Code_EUR = Currency_Code_EUR,
            Currency_Code_GBP = Currency_Code_GBP,
            Currency_Code_ILS = Currency_Code_ILS,
            Currency_Code_INR = Currency_Code_INR,
            Currency_Code_JPY =  Currency_Code_JPY,
            Currency_Code_MXN = Currency_Code_MXN,
            Currency_Code_RUB = Currency_Code_RUB,
            Currency_Code_SAR = Currency_Code_SAR,
            Currency_Code_USD =  Currency_Code_USD)

    # Get the number of nodes and edges in the graph
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    # Convert the graph to an adjacency matrix
    adj_matrix = nx.adjacency_matrix(G).todense()

    import collections
    # Retrieve the properties errors of all the edges
    edge_properties = nx.get_edge_attributes(G, 'errors')

    # Count the number of edges by property value
    edge_count_by_property = collections.Counter(edge_properties.values())

    # Print the count of edges by property value
    for property_value, count in edge_count_by_property.items():

    print("Number of nodes:", num_nodes)
    print("Number of edges:", num_edges)

    print("✅ Model initialized")

    return model


def compile_model(model: Model, learning_rate=0.0005) -> Model:
    """
    Compile the model
    """


    print("✅ Model compiled")

    return model

def train_model():
    """
    Fit the model and return a tuple (fitted_model, history)
    """

    ####################### Left as a dummy, might be useful ######################
    print(Fore.BLUE + "\nTraining model..." + Style.RESET_ALL)

    es = EarlyStopping()

    history = model.fit(
        X,
        y,
        validation_data=validation_data,
        validation_split=validation_split,
        epochs=100,
        batch_size=batch_size,
        callbacks=[es],
        verbose=0
    )

    print(f"✅ Model trained on {len(X)} rows with min val MAE: {round(np.min(history.history['val_mae']), 2)}")

    return model, history


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
