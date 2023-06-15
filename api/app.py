import streamlit as st
import numpy as np
import pandas as pd
import plotly as px
import requests
import json
from ml_logic.edges_nodes2 import *
from ml_logic.network_plot import *
from ml_logic.data import clean_data

from pyvis.network import Network


############
import networkx as nx
import pandas as pd
import plotly.graph_objects as go
############

st.markdown("""# AML detector""")

st.divider()

st.markdown("""💸💸💸“Money is usually attracted, not pursued.”\t💸💸💸""")

stop = True


# Uploading CSV file
uploaded_csv = st.file_uploader("Choose a CSV file", type="csv", accept_multiple_files=False)

if uploaded_csv is not None:
    if st.button('Get money laundering analysis'):
        with st.spinner('Data analysis in process...'):
            # load the csv as dataframe
            df = pd.read_csv(uploaded_csv)
            print(df)
            df_byte = df.to_json().encode() # .to_json() converts dataframe into json object
                                            # .encode() converts json object into bytes, encoded using UTF-8

            results = requests.post(url="http://localhost:8000/predict", files={"myfile": df_byte})
            #response = pd.DataFrame.from_dict(response.text)
            print ("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            results = results.text
            results = json.loads(results)

        stop = False
        st.success("analysis ready")

if stop == True:
    st.stop()

# results need to be merged to df!

st.write(df)

st.write(sum(results))

stop = True


# plotly

df = clean_data(df)

# first, create the directed multigraph
G = directed_multidigraph(df)

# next, calculate degrees for each node in the graph
node_in_degrees, node_out_degrees, node_degrees = calculate_degrees(G)

# finally, draw the directed multigraph
draw_directed_multigraph(G)

network = Network(directed=True, notebook=False)
network.from_nx(G)

edge_list = nx.to_pandas_edgelist(G)

# Create a Plotly figure
fig = go.Figure(data=[go.Scatter(x=edge_list['source'], y=edge_list['target'], mode='lines')])


# Render the network visualization in Streamlit
st.plotly_chart(fig)
