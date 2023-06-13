import streamlit as st
from streamlit import session_state
from streamlit_extras.add_vertical_space import add_vertical_space


def parameters(model):
    params_text = dict()
    if model == "Support Vector Machine":
        params_text = {'kernal': 'rbf', 'degree': 3, 'gamma': 'scale', 'coef0': 0.0, 'tol': 0.001,
                       'C': 1.0, 'epsilon': 0.1, 'shrinking': True, 'cache_size': 200, 'verbose': False, 'max_iter': -1}
    elif model == "Linear Regression":
        params_text = {'fit_intercept': True, 'copy_X': True,
                       'n_jobs': None, 'positive': False}
    elif model == "Logistic Regression":
        params_text = {'penalty': 'l2', 'dual': False, 'tol': 0.0001, 'C': 1.0, 'fit_intercept': True, 'intercept_scaling': 1, 'class_weight': None,
                       'random_state': None, 'solver': 'lbfgs', 'max_iter': 100, 'multi_class': 'auto', 'verbose': 0, 'warm_start': False, 'n_jobs': None, 'l1_ratio': None}
    return params_text


def show_feature_selection():
    st.write("<h1 style = 'text-align : center';> Select model for prediction </h1>",
             unsafe_allow_html=True)
    st.write("---")
    st.write(" <h4> Select model </h4>", unsafe_allow_html=True)
    add_vertical_space(1)
    st.write("<h6>Select type of model</h6>", unsafe_allow_html=True)
    model_type = st.radio(
        "Select", ["Regression model", "Classification model"], label_visibility="collapsed")
    model_text = str()
    if model_type == "Regression model":
        st.write(" <h6> Regression model </h6>", unsafe_allow_html=True)
        regression_model = st.selectbox(
            "Select", ["Support Vector Machine", "Linear Regression"], label_visibility="collapsed"
        )
        if regression_model:
            model_text = regression_model
    else:
        st.write(" <h6> Classification model </h6>", unsafe_allow_html=True)
        classification_model = st.selectbox(
            "Select", ["Logistic Regression"], label_visibility="collapsed"
        )
        if classification_model:
            model_text = classification_model

    st.write("---")

    st.write("<h4> Set hyper-parameters </h4>", unsafe_allow_html=True)
    model_params = parameters(model_text)
    st.write("---")
    st.write("<h4> Additional operations </h4>", unsafe_allow_html=True)
    st.write("---")


if ("df_train" not in session_state):
    st.write("<h2 style = 'text-align : center'; > Please upload the data first for visualization! </h2>",
             unsafe_allow_html=True)
else:
    df_train = session_state.df_train
    show_feature_selection()
