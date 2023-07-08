import random
import numpy as np
import pandas as pd
import streamlit as st
from streamlit import session_state


@st.cache_data
def convert_df_to_csv(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(index=False).encode('utf-8')


def show_results():
    random.seed(session_state.seed_val)
    # Title
    st.write("<h1 style = 'text-align : center';> Prediction on new data </h1>",
             unsafe_allow_html=True)
    st.write("---")

    # Testing data
    testing_data_file_name = st.file_uploader(
        "**Upload the testing data**", ["csv"])

    if testing_data_file_name is not None:
        df_test = pd.read_csv(testing_data_file_name)
        st.write("<h5 style = 'text-align : center;'> Test data </h5>",
                 unsafe_allow_html=True)
        st.dataframe(df_test)
        # Make predictions
        predict_btn = st.button("Predict the target values")

        st.write("---")
        if predict_btn:
            modified_df_test = df_test[session_state.features]
            if "scaler" in session_state:
                modified_df_test = session_state.scaler.transform(df_test)

            session_state.final_model.fit(session_state.final_X_train,
                                          np.ravel(session_state.y_train))
            predictions = session_state.final_model.predict(modified_df_test)
            final_data = df_test
            final_data.insert(
                0, f"Predicted {session_state.targets}", predictions, True)
            session_state.final_data = final_data

        if "final_data" in session_state:
            st.write("<h5 style = 'text-align : center;'> Predicted data </h5>",
                     unsafe_allow_html=True)
            st.dataframe(session_state.final_data)
            st.download_button(label="Download predictions", data=convert_df_to_csv(
                session_state.final_data), file_name="predictions.csv", mime="text/csv")
            st.write("---")


if ("df_train" not in session_state):
    st.write("<h2 style = 'text-align : center'; > Please upload the data first to use this feature! </h2>",
             unsafe_allow_html=True)
else:
    df_train = session_state.df_train
    show_results()
