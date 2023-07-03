import streamlit as st
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
import pandas as pd
from streamlit import session_state
from streamlit_extras.add_vertical_space import add_vertical_space


def detail(df):
    cat_detail = [list() for i in range(len(df.columns))]
    for i in range(len(df.dtypes)):
        cat_detail[i].append(i+1)
        cat_detail[i].append(df.columns[i])
        cat_detail[i].append(str(df.dtypes[i]))
        cat_detail[i].append(df[df.columns[i]].nunique())
    return cat_detail


def show_missing_info(df):
    st.write("<h6> Missing values information for the dataset </h6>",
             unsafe_allow_html=True)
    null_text = df.isnull().any()
    null_val = df.isnull().sum()
    total_val = df.shape[0]
    null_dict = {'Sr. no.': list(), 'Column': list(), 'isNull/isNan': list(),
                 'Count': list(), "missing data (%)": list()}
    for i in range(len(df.columns)):
        null_dict['Sr. no.'].append(i+1)
        null_dict['Column'].append(df.columns[i])
        if (null_text[i]):
            null_dict['isNull/isNan'].append("True")

        else:
            null_dict['isNull/isNan'].append("False")
        null_dict['Count'].append(null_val[i])
        null_dict['missing data (%)'].append(null_val[i]*100/total_val)

    st.dataframe(null_dict, width=500)
    contains_null = df.isnull().values.any()
    col_names = list()
    if contains_null:
        for i in range(len(df.columns)):
            if null_text[i]:
                col_names.append(df.columns[i])
    return col_names


def handle(missing_text, missing_columns):
    if (missing_text == "Delete selected columns"):
        session_state.df_train = session_state.df_train.drop(
            missing_columns, axis=1)
        if ("df_test" in session_state):
            session_state.df_test = session_state.df_test.drop(
                missing_columns, axis=1)
    else:
        # Imputation
        my_imputer = SimpleImputer()
        session_state.df_train = pd.DataFrame(my_imputer.fit_transform(
            session_state.df_train), columns=session_state.df_train.columns)

        if "df_test" in session_state:
            session_state.df_test = pd.DataFrame(my_imputer.transform(
                session_state.df_test), columns=session_state.df_test.columns)


def encode(text, encode_col_names):
    if (text == "Label Encoding"):
        label_encode(encode_col_names)
    else:
        one_hot_encode(encode_col_names)


def label_encode(col_names):
    label_encoder = preprocessing.LabelEncoder()

    for col in col_names:
        session_state.df_train[col] = label_encoder.fit_transform(
            session_state.df_train[col])
        if "df_test" in session_state:
            if (col in session_state.df_test.columns):
                session_state.df_test[col] = label_encoder.fit_transform(
                    session_state.df_test[col])


def one_hot_encode(col_names):
    session_state.df_train = pd.get_dummies(
        session_state.df_train, columns=col_names, drop_first=True)
    if "df_test" in session_state:
        test_columns = list()
        for col in col_names:
            if (col in session_state.df_test):
                test_columns.append(col)
        session_state.df_test = pd.get_dummies(
            session_state.df_test, columns=test_columns, drop_first=True)


def show_pre_processing():

    # Title
    st.write("<h1 style = 'text-align : center';> Pre-processing of data </h1>",
             unsafe_allow_html=True)
    st.write("---")

    # Handling categorical values
    st.write(" <h4> Encode categorical features </h4> ", unsafe_allow_html=True)
    categorical_detail = detail(session_state.df_train)
    st.write("<h6> Datatype of all columns in dataset</h6>",
             unsafe_allow_html=True)

    st.dataframe(pd.DataFrame(categorical_detail, columns=[
                 "Sr. no.", "Column", "Datatype", "Unique values count"]), hide_index=True)

    col_names_categorical = list()
    for i in range(len(categorical_detail)):
        if (categorical_detail[i][2] == "object"):
            col_names_categorical.append(session_state.df_train.columns[i])

    if (len(col_names_categorical) == 0):
        st.write("The data does not have any categorical features !")
    else:
        st.write("""
        ##### When to use a Label Encoding vs. One Hot Encoding
        This question generally depends on your dataset and the model which you wish to apply. But still, a few points to note before choosing the right encoding technique for your model:

        ###### We apply One-Hot Encoding when:
        1. The categorical feature is not ordinal (e.g. countries , gender )
        2. The number of categorical features is **less** so one-hot encoding can be effectively applied
        ###### We apply Label Encoding when:
        1. The categorical feature is ordinal (e.g. tall, short, primary school, high school)
        2. The number of categories is quite **large** as one-hot encoding can lead to high memory consumption
        """)
        add_vertical_space(2)
        st.write("<h6>Choose columns for encoding</h6>",
                 unsafe_allow_html=True)
        encode_col_names = st.multiselect(
            "Select", col_names_categorical, label_visibility="collapsed")
        st.write(" <h6> Select a method for encoding </h6>",
                 unsafe_allow_html=True)
        encoder_text = st.selectbox(
            "Select", ["Label Encoding", "One Hot Encoding"], label_visibility="collapsed")

        apply_btn = st.button("Apply")

        if apply_btn:
            encode(encoder_text, encode_col_names)
            st.experimental_rerun()
    add_vertical_space(2)
    st.write("---")
    add_vertical_space(2)

    # Handling missing values
    st.write("<h4> Handle missing values </h4>", unsafe_allow_html=True)
    col_names = show_missing_info(session_state.df_train)
    if (len(col_names) == 0):
        st.write("The data does not have any missing values !")
    else:
        st.write("<h6> Choose columns </h6>", unsafe_allow_html=True)
        missing_columns = st.selectbox(
            "Select", col_names, label_visibility="collapsed")
        st.write("<h6> Select a method </h6>", unsafe_allow_html=True)
        missing_text = st.selectbox("Select", ["Delete selected columns",
                                    "Impute the columns"], label_visibility="collapsed")

        apply_btn = st.button("Apply", key="missing_key")

        if apply_btn:
            handle(missing_text, missing_columns)
            st.experimental_rerun()

    st.write("---")


if ("df_train" not in session_state):
    st.write("<h2 style = 'text-align : center'; > Please upload the data first for pre processing! </h2>",
             unsafe_allow_html=True)
else:
    show_pre_processing()
