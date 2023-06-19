import streamlit as st
from streamlit import session_state

def show_results():
    
    # Title
    st.write("<h1 style = 'text-align : center';> Result and Analysis </h1>", unsafe_allow_html= True)
    st.write("---")

if ("df_train" not in session_state):
    st.write("<h2 style = 'text-align : center'; > Please upload the data first for visualization! </h2>",
             unsafe_allow_html=True)
else:
    df_train = session_state.df_train
    show_results()