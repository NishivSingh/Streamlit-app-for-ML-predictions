import traceback
import numpy as np
import pandas as pd
from sklearn.model_selection import LeavePOut, ShuffleSplit, StratifiedKFold, train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import streamlit as st
from streamlit import session_state
from streamlit_extras.add_vertical_space import add_vertical_space


def parameters(model):
    params_text = dict()
    options_dict = dict()
    if model == "Support Vector Machine":
        params_text = {'kernel': 'rbf', 'degree': 3, 'gamma': 'scale', 'coef0': 0.0, 'tol': 0.001,
                       'C': 1.0, 'epsilon': 0.1, 'shrinking': True, 'cache_size': 200, 'verbose': False, 'max_iter': -1}
        options_dict = {'kernel': [
            'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'], 'gamma': ['scale', 'auto']}
    elif model == "Linear Regression":
        params_text = {'fit_intercept': True, 'copy_X': True,
                       'n_jobs': 0, 'positive': False}
        options_dict = {}
    elif model == "Logistic Regression":
        params_text = {'penalty': 'l2', 'dual': False, 'tol': 0.0001, 'C': 1.0, 'fit_intercept': True, 'intercept_scaling': 1, 'class_weight': "balanced",
                       'random_state': 42, 'solver': 'lbfgs', 'max_iter': 100, 'multi_class': 'auto', 'verbose': 0, 'warm_start': False, 'n_jobs': 1, 'l1_ratio': 0}
        options_dict = {'penalty': ['l2', 'l1', 'elasticnet', None], 'solver': [
            'lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'], 'multi_class': ['auto', 'ovr', 'multinomial'],'class_weight' : ['balanced']}
    return params_text, options_dict


def spilt_data(df_train, targets, features):
    session_state.X = df_train[features]
    session_state.y = df_train[targets]
    X_train, X_test, y_train, y_test = train_test_split(
        session_state.X, session_state.y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def create_model(model_text, model_params):
    if (model_text == "Support Vector Machine"):
        return SVR(**model_params)

    elif (model_text == "Linear Regression"):
        return LinearRegression(**model_params)

    return LogisticRegression(**model_params)


def transform():
    if ("scaler" in session_state):
        session_state.final_X = session_state.scaler.transform(session_state.X)
        session_state.final_X_train = session_state.scaler.transform(
            session_state.X_train)
        session_state.final_X_test = session_state.scaler.transform(
            session_state.X_test)

    else:
        session_state.final_X = session_state.X
        session_state.final_X_train = session_state.X_train
        session_state.final_X_test = session_state.X_test


def show_feature_selection():

    # Title
    st.write("<h1 style = 'text-align : center';> Select model for prediction </h1>",
             unsafe_allow_html=True)
    st.write("---")

    # Splitting data
    X_train, X_test, y_train, y_test = spilt_data(
        df_train, session_state.targets, session_state.features)

    # Storing the splitted data
    session_state.X_train = X_train
    session_state.X_test = X_test
    session_state.y_train = y_train
    session_state.y_test = y_test

    # Model selection
    st.write(" <h4> Select model </h4>", unsafe_allow_html=True)
    add_vertical_space(1)
    st.write("<h6>Select type of model</h6>", unsafe_allow_html=True)
    model_type = st.radio(
        "Select", ["Regression model", "Classification model"], label_visibility="collapsed")
    
    session_state.model_type = model_type
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
    session_state.model_text = model_text
    st.write("---")

    # Setting hyper-parameters
    st.write("<h4> Set hyper-parameters </h4>", unsafe_allow_html=True)
    model_params, option_dict = parameters(model_text)

    keys = list(model_params.keys())
    values = list(model_params.values())

    for i in range(len(keys)):

        col1, col2, col3 = st.columns([10, 1, 10])
        col1.text_input(
            "Key", keys[i], key=f"key_input_{i}", label_visibility="collapsed", disabled=True)
        value = values[i]
        col2.text_input(":", "=", key=i,
                        label_visibility="collapsed", disabled=True)

        if isinstance(value, str):
            new_value = col3.selectbox(
                "Value", options=option_dict[keys[i]], key=f"value_input_{i}", label_visibility="collapsed",)
        elif isinstance(value, bool):
            new_value = col3.checkbox(
                "Value", value=value, key=f"value_input_{i}", label_visibility="collapsed")
        else:
            new_value = col3.number_input(
                "Value", value, key=f"value_input_{i}", label_visibility="collapsed")

        model_params[keys[i]] = new_value

    session_state.model_params = model_params

    add_vertical_space(2)
    st.write("<h5>Current hyper-parameters</h5>", unsafe_allow_html=True)
    st.write(session_state.model_params)
    st.write("---")

    # Storing the model
    model = create_model(session_state.model_text, session_state.model_params)
    session_state.model = model

    # Additional operations
    st.write("<h4> Additional operations </h4>", unsafe_allow_html=True)
    add_vertical_space(1)
    st.write("<h6> Please select the operations to perform</h6>",
             unsafe_allow_html=True)
    norm_val = st.checkbox("Feature scaling")
    cross_val = st.checkbox("Cross-validation")
    add_vertical_space(4)
    if norm_val:
        st.write("<h5> Feature Scaling </h5>", unsafe_allow_html=True)
        st.write("<h6>select type of feature scaling</h6>",
                 unsafe_allow_html=True)
        scaler_text = st.selectbox(
            "select", ["MinMaxScaler", "StandardScaler"], label_visibility="collapsed")
        perform_btn = st.button("Perform", key="norm")
        if perform_btn:
            if scaler_text == "MinMaxScaler":
                scaler = MinMaxScaler()
                scaler.fit(session_state.X_train)
                session_state.scaler = scaler
            else:
                scaler = StandardScaler()
                scaler.fit(session_state.X_train)
                session_state.scaler = scaler

    transform()

    if cross_val:
        add_vertical_space(1)
        st.write("<h5> Cross Validation </h5>", unsafe_allow_html=True)
        st.write("<h6>select type of cross validation</h6>",
                 unsafe_allow_html=True)
        cross_val_options = ["K-fold", "Monte Carlo"]
        cross_val_type = st.selectbox(
            "select", cross_val_options, label_visibility="collapsed")
        perform_btn = st.button("Perform", key="cross")

        if perform_btn:
            try:
                session_state.cross_val_used = True
                score_data = list()
                predict = list()
                if cross_val_type == "K-fold":
                    score_data = cross_val_score(
                        session_state.model, session_state.final_X, np.ravel(session_state.y), cv=10)
                    predict = cross_val_predict(
                        session_state.model, session_state.final_X, np.ravel(session_state.y), cv=10)
                    
                elif cross_val_type == "Monte Carlo":
                    shuffle_split = ShuffleSplit(
                        test_size=0.3, train_size=0.7, n_splits=10)
                    score_data = cross_val_score(session_state.model, session_state.final_X, np.ravel(
                        session_state.y), cv=shuffle_split)
                    predict = cross_val_predict(session_state.model, session_state.final_X, np.ravel(
                        session_state.y), cv=shuffle_split)
                    
                session_state.y_pred = predict
                session_state.y_true = session_state.y
                st.write("**Results of cross validation**")
                st.text(f"cross validation scores : {score_data}")
                st.text(f"maximum score achieved : {np.max(score_data)}")
                st.text(f"minimum score achieved : {np.min(score_data)}")
                st.text(f"average score : {np.average(score_data)}")
            except ValueError as e:
                traceback_str = str(traceback.format_exc())
                last_line = traceback_str.strip().split('\n')[-1]
                st.write(last_line)


    st.write("---")


if ("df_train" not in session_state):  # type: ignore
    st.write("<h2 style = 'text-align : center'; > Please upload the data first to use this feature! </h2>",
             unsafe_allow_html=True)
elif ("features" not in session_state or "targets" not in session_state):
    st.write("<h2> Please select features and targets from the dataset to continue! </h2>",
             unsafe_allow_html=True)
else:
    df_train = session_state.df_train
    show_feature_selection()
