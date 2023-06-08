import streamlit as st
import numpy as np
import pandas as pd
import plotly as px
import requests


st.markdown("""# AML detector""")

st.divider()

st.markdown("""💸💸💸“Money is usually attracted, not pursued.”\t💸💸💸""")



# This reads file as bytes, other options are available




url = "http://localhost:8000/predict"
# Uploading CSV file
uploaded_csv = st.file_uploader("Choose a CSV file", type="csv", accept_multiple_files=False)
if uploaded_csv is not None:
    if st.button('Get results transfering encoded json as bytes'):
        with st.spinner('data analysis in process...'):
            # load the csv as dataframe
            df = pd.read_csv(uploaded_csv)
            df_byte = df.to_json().encode() # .to_json() converts dataframe into json object
                                            # .encode() converts json object into bytes, encoded using UTF-8
            print(type(df_byte))

            url_endpoint = f"{url}/uploaded_csv_json"
            response = requests.post(url=url, files={"myfile": df_byte})
            #response = requests.get(url, files={"myfile": df_byte})
        st.success(response.json())
        st.write(response.json()) # printing the response from app.py in FastAPI

##### money emoji while waiting


# model predict output needs to be changed (currently tensor->str)


# process df to sho results properly, add plotly


### dummy for plotly viz

#@st.cache_data
#def get_plotly_data():

    # z_data = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/api_docs/mt_bruno_elevation.csv')
    # z = z_data.values
    # sh_0, sh_1 = z.shape
    # x, y = np.linspace(0, 1, sh_0), np.linspace(0, 1, sh_1)
    # return x, y, z

# import plotly.graph_objects as go

# x, y, z = get_plotly_data()

# fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
# fig.update_layout(title='IRR', autosize=False, width=800, height=800, margin=dict(l=40, r=40, b=40, t=40))
# st.plotly_chart(fig)
