import joblib
import streamlit as st
import pandas as pd
from streamlit import session_state
from streamlit_extras.add_vertical_space import add_vertical_space


def show_model_comparison():

    # Title
    st.write("<h1 style = 'text-align : center';> Compare different models </h1>",
             unsafe_allow_html=True)
    st.write("---")

    st.subheader("Upload models for comparison")
    models = st.file_uploader(
        "Upload models for comparison", accept_multiple_files=True, label_visibility="collapsed")
    ml_models = list()
    if models is not None:
        for pickle_model in models:
            model = joblib.load(pickle_model)
            ml_models.append(model)
        session_state.ml_models = ml_models

    for model in session_state.ml_models:
        st.text(model)

    st.write("---")


if ("df_train" not in session_state):
    st.write("<h2 style = 'text-align : center'; > Please upload the data first to use this feature! </h2>",
             unsafe_allow_html=True)
else:
    show_model_comparison()
