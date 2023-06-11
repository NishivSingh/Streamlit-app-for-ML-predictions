import streamlit as st
from streamlit import session_state

def show_pre_processing():
    st.write("<h1 style = 'text-align : center';> Pre-processing of data </h1>", unsafe_allow_html= True)
    st.write("---")
    st.write("<h4> Handle missing values </h4>",unsafe_allow_html=True)
    st.write("<h6> Select a method </h6>",unsafe_allow_html=True)
    missing_text = st.selectbox("Select",["Delete columns having > 75% missing data", "Filling the missing data by mean of the column"],label_visibility="collapsed")
    st.write("---")

    st.write(" <h4> Encode categorical features </h4> " ,unsafe_allow_html=True)
    st.write(" <h6> Select a method for encoding </h6>", unsafe_allow_html= True)
    encoder_text = st.selectbox("Select",["Label Encoding", "One Hot Encoding"],label_visibility="collapsed")
    st.write("---")
if ("df_train" not in session_state):
    st.write("<h2 style = 'text-align : center'; > Please upload the data first for visualization! </h2>",
             unsafe_allow_html=True)
else:
    df_train = session_state.df_train
    show_pre_processing()