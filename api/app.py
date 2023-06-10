import streamlit as st
import numpy as np
import pandas as pd
import plotly as px

st.markdown("""# AML detector""")

st.divider()

st.markdown("""💸💸💸“Money is usually attracted, not pursued.”\t💸💸💸""")



# This reads file as bytes, other options are available


with st.form("upload_form"):

    uploaded_file = st.file_uploader("Upload file for money laundering detection")
    if uploaded_file is not None:
        # To read file as bytes:
        bytes_data = uploaded_file.getvalue()
        st.write(bytes_data)

    # Every form must have a submit button.
    submitted = st.form_submit_button("Upload")
    if submitted:
       st.write("""Your file is being processed""")


### dummy for plotly viz

@st.cache_data
def get_plotly_data():

    z_data = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/api_docs/mt_bruno_elevation.csv')
    z = z_data.values
    sh_0, sh_1 = z.shape
    x, y = np.linspace(0, 1, sh_0), np.linspace(0, 1, sh_1)
    return x, y, z

import plotly.graph_objects as go

x, y, z = get_plotly_data()

fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
fig.update_layout(title='IRR', autosize=False, width=800, height=800, margin=dict(l=40, r=40, b=40, t=40))
st.plotly_chart(fig)




with st.form("upload_form"):
    st.write("Inside the form")

    uploaded_file = st.file_uploader("Upload file for money laundering detection")
    if uploaded_file is not None:
        # To read file as bytes:
        bytes_data = uploaded_file.getvalue()
        st.write(bytes_data)

    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:
       st.write("slider", slider_val, "checkbox", checkbox_val)
