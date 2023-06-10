import streamlit as st
import numpy as np
import pandas as pd
import plotly as px
import requests

st.markdown("""# AML detector""")

st.divider()

st.markdown("""💸💸💸“Money is usually attracted, not pursued.”\t💸💸💸""")

stop = True

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


            response = requests.post(url=url, files={"myfile": df_byte})
            #response = pd.DataFrame.from_dict(response.text)
            print(response)
        stop = False
        st.success("analysis ready")

#process_df = pd.read_csv(uploaded_csv)

if stop == True:
    st.stop()

stop = True


# process df to show results properly, add plotly


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
