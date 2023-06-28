import numpy as np
import pandas as pd
import streamlit as st
from streamlit import session_state

@st.cache_data
def convert_df_to_csv(df):
  # IMPORTANT: Cache the conversion to prevent computation on every rerun
  return df.to_csv(index = False).encode('utf-8')

def show_results():
    
    # Title
    st.write("<h1 style = 'text-align : center';> Prediction on new data </h1>", unsafe_allow_html= True)
    st.write("---")

    if "scaler" in session_state:
        session_state.final_X_test = session_state.scaler.transform(session_state.X_test)
    session_state.model.fit(session_state.final_X_train,np.ravel(session_state.y_train))
    predictions = pd.DataFrame(session_state.model.predict(session_state.final_X_test),columns=session_state.targets)
    st.download_button(label="Download predictions",data=convert_df_to_csv(predictions),file_name="predictions.csv",mime="text/csv")

    

if ("df_train" not in session_state):
    st.write("<h2 style = 'text-align : center'; > Please upload the data first to use this feature! </h2>",
             unsafe_allow_html=True)
else:
    df_train = session_state.df_train
    if "df_test" not in session_state:
        st.write("<h2 style = 'text-align : center'; > No test data is upload for prediction !</h2>",
             unsafe_allow_html=True)
    else:
        show_results()