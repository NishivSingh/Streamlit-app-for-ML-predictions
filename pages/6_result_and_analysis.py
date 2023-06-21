from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
import streamlit as st
from streamlit import session_state

def evaluation(y, predictions):
    mae = mean_absolute_error(y, predictions)
    mse = mean_squared_error(y, predictions)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    r_squared = r2_score(y, predictions)
    return mae, mse, rmse, r_squared

def show_graph(y, predictions):
    fig,ax = plt.subplots()
    ax.scatter(y,predictions,edgecolors=(0,0,0))
    ax.plot([y.min(),y.max()],[y.min(),y.max()],'k--',lw=4)
    ax.set_xlabel("Measured")
    ax.set_ylabel("Predicted")
    st.write(fig)
def show_results():
    
    # Title
    st.write("<h1 style = 'text-align : center';> Result and Analysis </h1>", unsafe_allow_html= True)
    st.write("---")

    if "y_pred" not in session_state:
        session_state.model.fit(session_state.final_X_train,np.ravel(session_state.y_train))
        session_state.y_pred = session_state.model.predict(session_state.final_X_test)
        session_state.y_true = session_state.y_test
    
    mae,mse,rmse,r_squared = evaluation(session_state.y_true,session_state.y_pred)
    eval_dict = {"evaluation method": ["mean absolute error","mean squared error","root mean squared error", "r-squared score"], "value":[mae,mse,rmse,r_squared]}
    col1,col2,col3 = st.columns([2,1,2])
    col2.dataframe(eval_dict)

    show_graph(session_state.y_true,session_state.y_pred)

if ("df_train" not in session_state):
    st.write("<h2 style = 'text-align : center'; > Please upload the data first to use this feature! </h2>",
             unsafe_allow_html=True)
else:
    df_train = session_state.df_train
    show_results()