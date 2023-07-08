import streamlit as st

# # Set the app layout to wide mode
# st.set_page_config(layout="wide")


def main():
    st.write("<h1 style = 'text-align : center';> ML Predictor </h1> ",
             unsafe_allow_html=True)
    st.write("---")
    col1, col2, col3 = st.columns([1, 2, 1])

    col2.image("streamlit_logo.png")
    st.subheader("Introduction")

    st.write("""
    **Streamlit app overview**<br>
    The Streamlit app is a web-based application that allows users to easily create and deploy machine learning models for materials engineering data.

    **Purpose of the app**<br>
    The main purpose of the app is to provide a user-friendly interface for engineers and researchers to make accurate predictions and analysis of material properties based on various input parameters.

    **Target Audience**<br>
    The target audience for the app includes engineers, researchers, and professionals in data related field who are interested in fast and efficient predictions of material properties.
    """, unsafe_allow_html=True)

    st.write("---")

    st.write("""
    #### Steps for using the app <br>
    1. Upload the data for training the models, the supported format is .csv files.
    2. In visualization section , you can visualize various types of graphical, matrix figure to understand the data better.
    3. After this if the data contains any categorical values or have some missing values in any feature , that can be handled here.
    4. Select features and target columns from the dataset. A random seed is provided for preserving the results.
    5. Select model according to the dataset type (classification or regression).The hyper-parameters of the model can be tuned using gridSearchCV.
    6. Save the model, and use feature scaling for normalizing/standardizing the data. Use cross-validation to get an initial picture of accuracy of model.
    7. Download the model as a pickle for future uses , this step is optional.
    8. Compare the models by uploading any saved model from the local system with the current model.
    9. Select final model for getting the predictions.
    10. Analyse the results obtained from the predictions.
    11. Upload new testing data , and make predictions.
    12. Download the predicted data as a csv file for future use.        """, unsafe_allow_html=True)

    st.write("---")


if __name__ == "__main__":
    main()
