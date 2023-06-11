import streamlit as st
from streamlit import session_state

def show_feature_selection():
    st.write("<h1 style = 'text-align : center';> Select features and targets </h1>", unsafe_allow_html= True)
    st.write("---")
    st.write("<h4> Select feature columns </h4> ",unsafe_allow_html=True)
    Feature_columns = st.multiselect(
        "**Select columns**", df_train.columns,key=1,label_visibility="collapsed"
    )
    st.write("---")

    st.write("<h4> Select target columns </h4>" ,unsafe_allow_html=True)

    Feature_columns = st.multiselect(
        "**Select columns**", df_train.columns,key=2,label_visibility="collapsed"
    )
    st.write("---")

if ("df_train" not in session_state):
    st.write("<h2 style = 'text-align : center'; > Please upload the data first for visualization! </h2>",
             unsafe_allow_html=True)
else:
    df_train = session_state.df_train
    show_feature_selection()