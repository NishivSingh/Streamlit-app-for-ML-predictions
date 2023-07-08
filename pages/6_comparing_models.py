import random
import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
import streamlit as st
import pandas as pd
from streamlit import session_state
from streamlit_extras.add_vertical_space import add_vertical_space


def evaluation(y, y_, model_type):
    if model_type == "Regression model":
        mae = mean_absolute_error(y, y_)
        mse = mean_squared_error(y, y_)
        rmse = np.sqrt(mean_squared_error(y, y_))
        r_squared = r2_score(y, y_)
        return mae, mse, rmse, r_squared
    else:
        return accuracy_score(y_true=y, y_pred=y_)


def get_results(model, compare_data_dict, model_type):
    compare_data_dict['model name'].append(str(model))
    model.fit(session_state.final_X_train,
              np.ravel(session_state.y_train))
    y_pred = model.predict(session_state.final_X_test)

    if model_type == "Regression model":
        mae, mse, rmse, r2_score = evaluation(
            session_state.y_test, y_pred, model_type)  # type: ignore
        compare_data_dict["mae"].append(mae)
        compare_data_dict["mse"].append(mse)
        compare_data_dict["rmse"].append(rmse)
        compare_data_dict["r2 score"].append(r2_score)
    else:
        compare_data_dict["accuracy score"].append(
            evaluation(session_state.y_test, y_pred, model_type))


def show_model_comparison():
    random.seed(session_state.seed_val)
    # Title
    st.write("<h1 style = 'text-align : center';> Compare different models </h1>",
             unsafe_allow_html=True)
    st.write("---")

    # Upload models
    st.subheader("Upload models for comparison")
    models = st.file_uploader(
        "Upload models for comparison", [".pkl"], accept_multiple_files=True, label_visibility="collapsed")
    ml_models = dict()
    ml_models[str(session_state.model)] = session_state.model
    if models is not None:
        for pickle_model in models:
            model = joblib.load(pickle_model)
            if (str(model) not in ml_models.keys()):
                ml_models[str(model)] = model
        session_state.ml_models = ml_models

    st.write("---")

    # Available models
    st.write("### Available unique models")
    add_vertical_space(1)
    for model in session_state.ml_models.values():
        st.text(model)
    add_vertical_space(2)
    compare_btn = st.button("Compare")
    st.write("---")

    # Results
    if session_state.model_type == "Regression model":
        compare_data_dict = {"model name": list(), "rmse": list(
        ), "mae": list(), "mse": list(), "r2 score": list()}
    else:
        compare_data_dict = {"model name": list(), "accuracy score": list()}

    if compare_btn:
        for model in session_state.ml_models.values():
            get_results(model, compare_data_dict, session_state.model_type)
        session_state.compare_data_dict = compare_data_dict

    if "compare_data_dict" in session_state:
        st.write("### Results")
        st.dataframe(session_state.compare_data_dict, use_container_width=True)
        st.write("---")

    st.write("#### Choose final model from available models")
    final_model = st.selectbox(
        "Select", session_state.ml_models.keys(), label_visibility="collapsed")
    final_model_btn = st.button("Use this model for prediction")

    if final_model_btn:
        session_state.final_model = session_state.ml_models[final_model]
        st.text(session_state.final_model)


if ("df_train" not in session_state):
    st.write("<h2 style = 'text-align : center'; > Please upload the data first to use this feature! </h2>",
             unsafe_allow_html=True)
else:
    show_model_comparison()
