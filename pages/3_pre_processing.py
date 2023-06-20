import streamlit as st
from sklearn import preprocessing
import pandas as pd
from streamlit import session_state
from streamlit_extras.add_vertical_space import add_vertical_space


def detail(df):
    cat_detail = [list() for i in range(len(df.columns))]
    for i in range (len(df.dtypes)):
        cat_detail[i].append(i+1)
        cat_detail[i].append(df.columns[i])
        cat_detail[i].append(str(df.dtypes[i]))
    return cat_detail


def show_missing_info(df):
    st.write("<h6> Missing values information for the dataset </h6>",
             unsafe_allow_html=True)
    null_text = df.isnull().any()
    null_val = df.isnull().sum()
    null_dict = {'Sr. no.' : list(),'Column': list(), 'isNull/isNan': list(),
                 'Count': list()}
    for i in range(len(df.columns)):
        null_dict['Sr. no.'].append(i+1)
        null_dict['Column'].append(df.columns[i])
        if (null_text[i]):
            null_dict['isNull/isNan'].append("True")

        else:
            null_dict['isNull/isNan'].append("False")
        null_dict['Count'].append(null_val[i])
    st.dataframe(null_dict, width=500)
    contains_null = df.isnull().values.any()
    col_names = list()
    if contains_null:
        for i in range(len(df.columns)):
            if null_text[i]:
                col_names.append(df.columns[i])
    return col_names

def label_encode(df,col_names):
    label_encoder = preprocessing.LabelEncoder()

    for col in col_names:
        df[col] = label_encoder.fit_transform(df[col])
    

def show_pre_processing():
    
    # Title
    st.write("<h1 style = 'text-align : center';> Pre-processing of data </h1>",
             unsafe_allow_html=True)
    st.write("---")


    # Handling missing values
    st.write("<h4> Handle missing values </h4>", unsafe_allow_html=True)
    col_names = show_missing_info(df_train)
    if (len(col_names) == 0):
        st.write("The data does not have any missing values !")
    else:
        st.write("<h6> Choose columns </h6>", unsafe_allow_html=True)
        missing_columns = st.selectbox(
            "Select", col_names, label_visibility="collapsed")
        st.write("<h6> Select a method </h6>", unsafe_allow_html=True)
        missing_text = st.selectbox("Select", ["Delete columns having > 75% missing data",
                                    "Filling the missing data by mean of the column"], label_visibility="collapsed")
    st.write("---")


    # Handling categorical values
    st.write(" <h4> Encode categorical features </h4> ", unsafe_allow_html=True)
    categorical_detail = detail(df_train)
    st.write("<h6> Datatype of all columns in dataset</h6>",
             unsafe_allow_html=True)
    
    st.dataframe(pd.DataFrame(categorical_detail,columns=["Sr. no.","Column","Datatype"]),hide_index=True,width=400)

    col_names_categorical = list()
    for i in range(len(categorical_detail)):
        if (categorical_detail[i][2] == "object"):
            col_names_categorical.append(df_train.columns[i])

    if (len(col_names_categorical) == 0):
        st.write("The data does not have any categorical features !")
    else:
        st.write(" <h6> Select a method for encoding </h6>",
                 unsafe_allow_html=True)
        encoder_text = st.selectbox(
            "Select", ["Label Encoding", "One Hot Encoding"], label_visibility="collapsed")
        apply_btn = st.button("Apply")

        if apply_btn:
            if encoder_text == "Label Encoding":
                label_encode(df_train,col_names_categorical)
    st.write("---")


if ("df_train" not in session_state):
    st.write("<h2 style = 'text-align : center'; > Please upload the data first for pre processing! </h2>",
             unsafe_allow_html=True)
else:
    df_train = session_state.df_train
    show_pre_processing()
