import pandas as pd
from sklearn.model_selection import train_test_split
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
        options_dict = {'kernel' : ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],'gamma':['scale', 'auto']}
    elif model == "Linear Regression":
        params_text = {'fit_intercept': True, 'copy_X': True,
                       'n_jobs': None, 'positive': False}
        options_dict = {}
    elif model == "Logistic Regression":
        params_text = {'penalty': 'l2', 'dual': False, 'tol': 0.0001, 'C': 1.0, 'fit_intercept': True, 'intercept_scaling': 1, 'class_weight': None,
                       'random_state': None, 'solver': 'lbfgs', 'max_iter': 100, 'multi_class': 'auto', 'verbose': 0, 'warm_start': False, 'n_jobs': None, 'l1_ratio': None}
        options_dict = {'penalty' : ['l1', 'l2', 'elasticnet', None],'solver':['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],'multi_class':['auto', 'ovr', 'multinomial']}
    return params_text,options_dict

def spilt_data(df_train,targets,features):
    X = df_train[features]
    y = df_train[targets]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    return X_train,X_test,y_train,y_test

def create_model(model_text,model_params):
    if (model_text == "Support Vector Machine"):
        return SVR(**model_params)
    
    elif (model_text == "Linear Regression"):
        return LinearRegression(**model_params)
    
    return LogisticRegression(**model_params)

def show_feature_selection():
    
    # Title
    st.write("<h1 style = 'text-align : center';> Select model for prediction </h1>",
             unsafe_allow_html=True)
    st.write("---")

    # Splitting data
    X_train,X_test,y_train,y_test = spilt_data(df_train,session_state.targets,session_state.features)
    
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
    model_params,option_dict = parameters(model_text)

    keys = list(model_params.keys())
    values = list(model_params.values())

    for i in range(len(keys)):

        col1,col2, col3 = st.columns([10,1,10])
        col1.text_input("Key", keys[i], key=f"key_input_{i}",label_visibility="collapsed",disabled=True)
        value = values[i]
        col2.text_input(":","=",key=i, label_visibility="collapsed",disabled=True)

        if isinstance(value, str):
            new_value = col3.selectbox("Value", options=option_dict[keys[i]], key=f"value_input_{i}",label_visibility="collapsed")
        elif isinstance(value, bool):
            new_value = col3.checkbox("Value", value=value, key=f"value_input_{i}",label_visibility="collapsed")
        else:
            new_value = col3.number_input("Value", value, key=f"value_input_{i}",label_visibility="collapsed")
        
        model_params[keys[i]] = new_value

    session_state.model_params = model_params

    add_vertical_space(2)
    st.write("<h5>Current hyper-parameters</h5>",unsafe_allow_html=True)
    st.write(session_state.model_params)
    st.write("---")

    # Storing the model
    model = create_model(session_state.model_text,session_state.model_params)
    session_state.model = model

    # Additional operations
    st.write("<h4> Additional operations </h4>", unsafe_allow_html=True)
    add_vertical_space(1)
    st.write("<h6> Please select the operations to perform</h6>",unsafe_allow_html=True)
    norm_val = st.checkbox("Feature scaling")
    cross_val = st.checkbox("Cross-validation")
    add_vertical_space(4)
    if norm_val:
        st.write("<h5> Feature Scaling </h5>",unsafe_allow_html=True)
        st.write("<h6>select type of feature scaling</h6>",unsafe_allow_html=True)
        scaler = st.selectbox("select",["MinMaxScaler","StandardScaler"],label_visibility="collapsed")
        perform_btn = st.button("Perform",key="norm")
    if cross_val:
        add_vertical_space(1)
        st.write("<h5> Cross Validation </h5>",unsafe_allow_html=True)
        st.write("<h6>select type of cross validation</h6>",unsafe_allow_html=True)
        scaler = st.selectbox("select",["K-fold","Stratified k-fold","Leave-p-out","Leave-one-out"],label_visibility="collapsed")
        perform_btn = st.button("Perform",key="cross")
    st.write("---")


if ("df_train" not in session_state):
    st.write("<h2 style = 'text-align : center'; > Please upload the data first to use this feature! </h2>",
             unsafe_allow_html=True)
elif ("features" not in session_state or "targets" not in session_state):
    st.write("<h2> Please select features and targets from the dataset to continue! </h2>",unsafe_allow_html=True)
else:
    df_train = session_state.df_train
    show_feature_selection()
