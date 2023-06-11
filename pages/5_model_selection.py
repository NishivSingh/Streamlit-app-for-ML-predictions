import streamlit as st
from streamlit import session_state
from streamlit_extras.add_vertical_space import add_vertical_space

def show_feature_selection():
    st.write("<h1 style = 'text-align : center';> Select model for prediction </h1>", unsafe_allow_html= True)
    st.write("---")
    st.write(" <h4> Select model </h4>", unsafe_allow_html=True)
    add_vertical_space(2)
    st.write(" <h6> Regression model </h6>",unsafe_allow_html=True)
    regression_model = st.selectbox(
        "**Select column for horizontal axis**", ["Support Vector Machine ", "Linear Regression"], label_visibility="collapsed"
    )
    add_vertical_space(1)
    st.write(" <h6> Classification model </h6>",unsafe_allow_html=True)
    classification_model = st.selectbox(
        "**Select column for horizontal axis**", ["Logistic Regression "], label_visibility="collapsed"
    )
    st.write("---")

    st.write(" #### Select parameters")

    Feature_columns = st.multiselect(
        "**Select columns**", df_train.columns,key=3,label_visibility="collapsed"
    )
    st.write("---")
    st.write(" #### Additional operations")
    st.write("---")

if ("df_train" not in session_state):
    st.write("<h2 style = 'text-align : center'; > Please upload the data first for visualization! </h2>",
             unsafe_allow_html=True)
else:
    df_train = session_state.df_train
    show_feature_selection()