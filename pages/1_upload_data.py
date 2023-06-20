import streamlit as st
import io
import pandas as pd
import numpy as np
from streamlit import session_state
from streamlit_extras.add_vertical_space import add_vertical_space


def display_data_content(df):
    # data in dataframe form
    st.dataframe(df)
    st.write(f'Shape of the data : {df.shape}')

    # data info
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_text = buffer.getvalue()
    st.text(info_text)

    # data description
    st.write("<h6 style = 'text-align : center ;'> Analysis of all numerical data </h6>",
             unsafe_allow_html=True)
    st.dataframe(df.describe())


def display_upload_data_page():

    # Title of page
    st.write("<h1 style = 'text-align : center;'> Upload the data </h1> ",
             unsafe_allow_html=True)
    st.write("---")

    # File uploading
    training_data_file_name = st.file_uploader(
        "**Upload the training data**", ["csv"])

    # Storing data in backend
    if training_data_file_name is not None:
        df_train = pd.read_csv(training_data_file_name)
        session_state.df_train = df_train
    
    # Displaying data
    if "df_train" in session_state:
        display_data_content(session_state.df_train)
    st.write("---")


if __name__ == "__main__":
    display_upload_data_page()
