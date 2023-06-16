import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx


######## TF imports here ########

def create_nodes_edges(df):
    """
    Initialize the model
    """
    G = nx.MultiGraph()

    # Add nodes to the graph for each unique card_id, merchant_name
    G.add_nodes_from(df["from_account"].unique(), type='from_account')
    G.add_nodes_from(df["to_account"].unique(), type='to_account')

    for _, row in df.iterrows():
        edge_attrs = dict()
        columns = [
            'from_bank', 'to_bank', 'amount_paid_USD', 'amount_received_USD', 'month',
            'day', 'hour', 'minute', 'payment_format_ACH', 'payment_format_Bitcoin',
            'payment_format_Cash', 'payment_format_Cheque', 'payment_format_Credit Card',
            'receiving_currency_BRL', 'receiving_currency_CAD', 'receiving_currency_CHF',
            'receiving_currency_CNY', 'receiving_currency_EUR', 'receiving_currency_GBP',
            'receiving_currency_ILS', 'receiving_currency_INR', 'receiving_currency_JPY',
            'receiving_currency_MXN', 'receiving_currency_RUB', 'receiving_currency_SAR',
            'receiving_currency_USD', 'receiving_currency_XBT', 'receiving_currency_nan',
            'payment_currency_BRL', 'payment_currency_CAD', 'payment_currency_CHF',
            'payment_currency_CNY', 'payment_currency_EUR', 'payment_currency_GBP',
            'payment_currency_ILS', 'payment_currency_INR', 'payment_currency_JPY',
            'payment_currency_MXN', 'payment_currency_RUB', 'payment_currency_SAR',
            'payment_currency_USD', 'payment_currency_XBT', 'payment_currency_nan'
        ]

        for column in columns:
            try:
                edge_attrs[column] = row[column]
            except KeyError:
                continue

        G.add_edge(row['from_account'], row['to_account'], **edge_attrs)

    # Get the number of nodes and edges in the graph
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    # Print the number of nodes and edges
    print("✅ Number of nodes:", num_nodes)
    print("✅ Number of edges:", num_edges)

    return G
