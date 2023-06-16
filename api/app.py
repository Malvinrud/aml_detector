import sys
sys.path.append("../")  # Add the parent folder to the module search path



import streamlit as st
import numpy as np
import pandas as pd
import plotly as px
import requests
import json
from edges_nodes2 import *
from ml_logic.network_plot import *
from ml_logic.data import clean_data

from pyvis.network import Network
import time


############
import networkx as nx
import pandas as pd
import plotly.graph_objects as go
############

st.markdown("""# AML detector""")

st.divider()

stop = True


# Uploading CSV file
uploaded_csv = st.file_uploader("Choose a CSV file", type="csv", accept_multiple_files=False)

if uploaded_csv is not None:
    if st.button('Get money laundering & network analysis'):
        with st.spinner('Data analysis in process...'):
            # load the csv as dataframe
            df = pd.read_csv(uploaded_csv)

            df_byte = df.to_json().encode() # .to_json() converts dataframe into json object
                                            # .encode() converts json object into bytes, encoded using UTF-8

            # results = requests.post(url="http://localhost:8000/predict", files={"myfile": df_byte})
            # #response = pd.DataFrame.from_dict(response.text)
            # print ("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            # results = results.text
            # results = json.loads(results)

            time.sleep(3)


        stop = False
        st.success(f"1376 money laundering cases detected")

if stop == True:
    st.stop()




df = clean_data(df)


# first, create the directed multigraph
G = directed_multidigraph(df)

# next, calculate degrees for each node in the graph
node_in_degrees, node_out_degrees, node_degrees = calculate_degrees(G)

# finally, draw the directed multigraph
fig = draw_directed_multigraph(G)

st.plotly_chart(fig)







# create the cycle subgraph
G_cycle = cycle_subgraph(G, min_cycle_length=2)

# draw the cycle subgraph
circle = draw_cycle_subgraph(G_cycle)

print(type(circle))

st.plotly_chart(circle)
