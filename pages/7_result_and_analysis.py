from matplotlib import pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score, accuracy_score
import streamlit as st
from streamlit import session_state
from streamlit_extras.add_vertical_space import add_vertical_space


def evaluation():
    y = session_state.y_true
    y_ = session_state.y_pred
    mae = mean_absolute_error(y, y_)
    mse = mean_squared_error(y, y_)
    rmse = np.sqrt(mean_squared_error(y, y_))
    r_squared = r2_score(y, y_)
    return mae, mse, rmse, r_squared


def scatterplot():
    y = session_state.y_true
    fig, ax = plt.subplots()
    ax.scatter(y, session_state.y_pred, edgecolors=(0, 0, 0))
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    ax.set_title(f"{session_state.targets[0]} plot")
    ax.set_xlabel("True Values")
    ax.set_ylabel("Predicted Values")
    st.write("<h4 style = 'text-align : center;'> Scatter plot between predicted and true values</h4>",
             unsafe_allow_html=True)
    st.write(fig)
    add_vertical_space(2)
    st.write("---")


def line_chart():
    # Line chart of actual vs predicted values
    index = pd.RangeIndex(start=0, stop=len(session_state.y_pred), step=1)
    fig = plt.figure(figsize=(8, 6))
    plt.plot(index, session_state.y_true, label='Actual')
    plt.plot(index, session_state.y_pred, label='Predicted')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Actual vs Predicted Values')
    plt.legend()
    st.write("<h4 style = 'text-align : center;'> Line chart of true vs predicted vales</h4>",
             unsafe_allow_html=True)
    st.pyplot(fig)
    add_vertical_space(2)
    st.write("---")


def histogram():
    # Histogram of predicted values
    fig = plt.figure(figsize=(8, 6))
    sns.histplot(session_state.y_pred, kde=True)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Predicted Values')
    st.write("<h4 style = 'text-align : center;'> Histogram of predicted values</h4>",
             unsafe_allow_html=True)
    st.pyplot(fig)
    add_vertical_space(2)
    st.write("---")


def confusion_matrix_plot():
    conf_mat = (confusion_matrix(
        y_true=session_state.y_true, y_pred=session_state.y_pred))
    fig, ax = plot_confusion_matrix(
        conf_mat=conf_mat, show_absolute=True, show_normed=True, colorbar=True)
    ax.set_title(f"{session_state.targets[0]}")
    st.write("<div style = 'font-size : 30px; margin : auto ;padding : 10px; color : white; border-radius : 5px; text-align : center; background-color : #08306b;'>Confusion matrix</div>", unsafe_allow_html=True)
    add_vertical_space(2)
    st.pyplot(fig, use_container_width=True)


def show_results():

    # Title
    st.write("<h1 style = 'text-align : center';> Result and Analysis </h1>",
             unsafe_allow_html=True)
    st.write("---")

    # Getting the predictions
    if "final_model" not in session_state:
        session_state.final_model = session_state.model

    st.write(
        f"<div style = 'display:flex;'><h5> Model used for predictions : </h5> <p>{session_state.final_model} </p> </div>", unsafe_allow_html=True)
    session_state.final_model.fit(
        session_state.final_X_train, np.ravel(session_state.y_train))
    session_state.y_pred = session_state.final_model.predict(
        session_state.final_X_test)
    session_state.y_true = session_state.y_test

    # Regression model results
    if (session_state.model_type == "Regression model"):
        st.write("<h4 style = 'text-align : center;'> Evaluation metrics</h4>",
                 unsafe_allow_html=True)
        mae, mse, rmse, r_squared = evaluation()
        eval_dict = {"evaluation method": ["mean absolute error", "mean squared error",
                                           "root mean squared error", "r-squared score"], "value": [mae, mse, rmse, r_squared]}
        col1, col2, col3 = st.columns([1, 2, 1])
        col2.dataframe(eval_dict, use_container_width=True)
        st.write("---")
        add_vertical_space(2)
        scatterplot()

        add_vertical_space(2)
        line_chart()

        add_vertical_space(2)
        histogram()

    # Classification model results
    else:
        confusion_matrix_plot()
        add_vertical_space(2)
        st.write(
            f"<div style = 'width:50%; margin:auto;'><span style = 'font-weight : bold;'>Accuracy score </span>: {accuracy_score(y_true=session_state.y_true,y_pred=session_state.y_pred)}</div>", unsafe_allow_html=True)

        add_vertical_space(2)
        st.write("<h4 style = 'text-align : center;'> Classification report </h4>",
                 unsafe_allow_html=True)

        class_report = pd.DataFrame(classification_report(
            y_true=session_state.y_true, y_pred=session_state.y_pred, output_dict=True))
        st.dataframe(class_report, use_container_width=True)  # type: ignore


if ("df_train" not in session_state):
    st.write("<h2 style = 'text-align : center'; > Please upload the data first to use this feature! </h2>",
             unsafe_allow_html=True)
else:
    df_train = session_state.df_train
    if "model" not in session_state:
        st.write("<h2 style = 'text-align : center'; > Please choose a model first to use this feature! </h2>",
                 unsafe_allow_html=True)
    else:
        show_results()
