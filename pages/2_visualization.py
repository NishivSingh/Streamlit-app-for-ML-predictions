import streamlit as st
from streamlit import session_state
from streamlit_extras.add_vertical_space import add_vertical_space
import matplotlib.pyplot as plt
import seaborn as sns


def filter_columns():
    col_names = list()
    categorical_detail = df_train.dtypes
    for i in range(len(categorical_detail)):
        if (categorical_detail[i] != "object"):
            col_names.append(df_train.columns[i])
    return col_names


def show_graph(x_column, y_columns):
    fig, ax = plt.subplots()
    for i in range(len(y_columns)):
        ax.scatter(df_train[x_column],
                   df_train[y_columns[i]], label=y_columns[i])
    ax.legend()
    ax.grid(True)
    plt.title("Scatter plot")
    plt.xlabel(x_column)
    plt.ylabel(y_columns)
    st.pyplot(fig)


def show_correlation_matrix(columns, filtered_col):
    if (len(columns) == 0):
        st.write("Please select a column")
    else:

        if "All" not in columns:
            correlation = df_train[columns].corr()
        else:
            correlation = df_train[filtered_col].corr()

        fig = plt.figure(figsize=(20, 20))
        sns.heatmap(correlation, xticklabels=correlation.columns,
                    yticklabels=correlation.columns, annot=True)
        st.write(fig)


def show_box_plot(column):
    fig = plt.figure()
    sns.boxplot(df_train[column])
    st.pyplot(fig)


def show_visualization_page():
    filtered_col = filter_columns()

    # Title
    st.write("<h1 style = 'text-align : center;'>Data Visualization</h1>",
             unsafe_allow_html=True)
    st.markdown("---")

    # Scatter plot
    st.subheader("Graphical correlation plot")
    graphical_correlation_plot_x_column = st.selectbox(
        "**Select column for horizontal axis**", filtered_col
    )

    graphical_correlation_plot_y_columns = st.multiselect(
        "**Select columns for vertical axis**", filtered_col, filtered_col[0]
    )

    show_graph(graphical_correlation_plot_x_column,
               graphical_correlation_plot_y_columns)

    st.markdown("---")

    # Heatmap
    options = ["All"]
    for column in filtered_col:
        options.append(column)

    st.subheader("Correlation matrix plot (heatmap)")
    correlation_matrix_plot_columns = st.multiselect(
        "**Select columns**",  options, options[0]
    )

    show_correlation_matrix(correlation_matrix_plot_columns, filtered_col)
    st.markdown("---")

    # Boxplot
    st.subheader("Box plot")
    box_plot_column = st.selectbox("**Select column**", filtered_col)
    show_box_plot(box_plot_column)
    st.write("---")


if ("df_train" not in session_state):
    st.write("<h2 style = 'text-align : center'; > Please upload the data first to use this feature! </h2>",
             unsafe_allow_html=True)
else:
    df_train = session_state.df_train
    show_visualization_page()
