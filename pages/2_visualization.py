import streamlit as st
import matplotlib.pyplot as plt
from streamlit import session_state
import seaborn as sns


def show_graph(x_column, y_columns):
    fig, ax = plt.subplots()
    for i in range(len(y_columns)):
        ax.scatter(df_train[x_column], df_train[y_columns[i]],label = y_columns[i])
    ax.legend()
    ax.grid(True)
    plt.title("Scatter plot")
    plt.xlabel(x_column)
    plt.ylabel(y_columns)
    st.pyplot(fig)

def show_correlation_matrix(columns):
    if (len(columns) == 0):
        st.write("Please select a column")
    else:

        if "All" not in columns:
            correlation = df_train[columns].corr()
        else:
            correlation = df_train.corr()
        
        
        fig = plt.figure(figsize=(25, 15))
        sns.heatmap(correlation, xticklabels=correlation.columns,
                yticklabels=correlation.columns, annot=True)
        st.write(fig)

def show_box_plot(column):
    fig = plt.figure()
    sns.boxplot(df_train[column])
    st.pyplot(fig)

def show_visualization_page():

    st.write("<h1 style = 'text-align : center;'>Data Visualization</h1>",unsafe_allow_html=True)
    st.markdown("---")

    st.subheader("Graphical correlation plot")
    graphical_correlation_plot_x_column = st.selectbox(
        "**Select column for horizontal axis**", df_train.columns
    )

    graphical_correlation_plot_y_columns = st.multiselect(
        "**Select columns for vertical axis**", df_train.columns,df_train.columns[0]
    )

    show_graph(graphical_correlation_plot_x_column,
               graphical_correlation_plot_y_columns)

    st.markdown("---")


    options = ["All"]
    for column in df_train.columns:
        options.append(column)
    
    st.subheader("Correlation matrix plot (heatmap)")
    correlation_matrix_plot_columns = st.multiselect(
        "**Select columns**",  options, options[0]
    )

    show_correlation_matrix(correlation_matrix_plot_columns)
    st.markdown("---")

    st.subheader("Box plot")
    box_plot_column = st.selectbox("**Select column**",df_train.columns)
    show_box_plot(box_plot_column)
    st.write("---")


if ("df_train" not in session_state):
    st.write("<h2 style = 'text-align : center'; > Please upload the data first for visualization! </h2>",
             unsafe_allow_html=True)
else:
    df_train = session_state.df_train
    show_visualization_page()
