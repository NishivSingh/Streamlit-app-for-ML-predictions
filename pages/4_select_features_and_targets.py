import streamlit as st
import re
import pandas as pd
from streamlit import session_state
from streamlit_extras.add_vertical_space import add_vertical_space


def show_feature_selection():

    # Title
    st.write("<h1 style = 'text-align : center';> Select features and targets </h1>",
             unsafe_allow_html=True)
    st.write("---")

    # Targets selection
    st.write("<h4> Select target columns </h4>", unsafe_allow_html=True)

    target_columns = st.multiselect(
        "Select", df_train.columns, key=3, label_visibility="collapsed"
    )

    apply_btn_target = st.button("Apply", key=4)

    if apply_btn_target:
        session_state.targets = target_columns
    if "targets" in session_state:
        st.dataframe(pd.DataFrame(session_state.targets,
                     columns=['selected columns']))
        
    st.write("---")
    
    # Features selection
    st.write("<h4> Select feature columns </h4> ", unsafe_allow_html=True)
    options_for_method = [None, "using column name",
                          "using indices of columns", "using RegEx query"]

    method_for_selection = st.selectbox(
        "Select", options_for_method, label_visibility="collapsed")

    feature_columns = list()

    if method_for_selection == options_for_method[1]:
        columns_names = st.multiselect(
            "Select", df_train.columns, key=1, label_visibility="collapsed"
        )
        feature_columns = columns_names

    elif method_for_selection == options_for_method[2]:
        indices = st.slider("choose the range", 0, len(
            df_train.columns)-1, (0, len(df_train.columns)-1), label_visibility="collapsed")
        for i in range(indices[0], indices[1]+1, 1):
            feature_columns.append(df_train.columns[i])

    elif method_for_selection == options_for_method[3]:
        RegEx_query = st.text_input("Type your query")
        if RegEx_query:
            word_re = re.compile(RegEx_query)
            for col in df_train.columns:
                if (word_re.search(col)):
                    feature_columns.append(col)
    apply_btn_feature = st.button("Apply", key=2)

    if apply_btn_feature:
        session_state.features = feature_columns
    if "features" in session_state:
        st.dataframe(pd.DataFrame(session_state.features,
                     columns=['selected columns']))
    
    st.write("---")


if ("df_train" not in session_state):
    st.write("<h2 style = 'text-align : center'; > Please upload the data first for visualization! </h2>",
             unsafe_allow_html=True)
else:
    df_train = session_state.df_train
    show_feature_selection()
