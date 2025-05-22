import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import joblib
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from src.predict import predict_churn, load_model, load_features

# ---------------- Page Setup ---------------- #
st.set_page_config(page_title="Churn Prediction Dashboard", layout="centered")
st.title("Customer Churn Prediction App")
st.write("Predict whether a customer is likely to churn based on their profile.")

# ---------------- Load Model & Features ---------------- #
model = load_model(os.path.join("models", "CatBoost.pkl"))
feature_list = load_features(os.path.join("config","config.yaml"))

# ---------------- Sidebar Option ---------------- #
option = st.sidebar.radio("Choose Prediction Method:", ["Upload CSV", "Manual Input"])

# ---------------- CSV Upload Section ---------------- #
if option == "Upload CSV":
    st.subheader("Upload Your CSV File")
    
    with st.expander("CSV Format Instructions"):
        st.markdown("""
        Your CSV should contain the following **exact columns**:\n
        - All the required input features used in training (e.g., `SeniorCitizen`, `tenure`, `MonthlyCharges`, etc.)
        - Binary/dummy encoded values for categorical variables (like `OnlineSecurity_Yes`, `Contract_Two year` etc.)
        - No target column `Churn` required.
        - Refer to the [sample CSV here](https://your-link-to-sample.com) or download this sample below.
        """)
        with open("../data/sample_churn_data.csv", "rb") as file:
            st.download_button("Download Sample CSV", file, file_name="sample_churn_data.csv")

    uploaded_file = st.file_uploader("Upload CSV file for batch prediction", type="csv")

    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.subheader("Uploaded Data Preview")
        st.dataframe(data.head())

        # Basic stats
        with st.expander("Basic Feature Distributions"):
            numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
            for col in numeric_cols:
                if col in data.columns:
                    fig, ax = plt.subplots()
                    sns.histplot(data[col], kde=True, ax=ax)
                    ax.set_title(f'Distribution of {col}')
                    st.pyplot(fig)

        if st.button("Predict Churn"):
            preds, probs = predict_churn(data, model, feature_list)
            result_df = data.copy()
            result_df['Churn_Prediction'] = preds
            result_df['Churn_Probability'] = probs

            st.subheader("Predictions")
            st.dataframe(result_df[['Churn_Prediction', 'Churn_Probability']])

            # Visualize prediction results
            st.subheader("Churn Prediction Summary")
            fig2, ax2 = plt.subplots()
            result_df['Churn_Prediction'].value_counts().plot.pie(
                autopct='%1.1f%%', labels=["No Churn", "Churn"], ax=ax2, colors=["lightgreen", "salmon"]
            )
            ax2.set_ylabel('')
            ax2.set_title("Predicted Churn Distribution")
            st.pyplot(fig2)

            # Download
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions", data=csv, file_name="churn_predictions.csv")

# ---------------- Manual Input Section ---------------- #
else:
    st.subheader("Manual Customer Entry")
    st.markdown("Enter feature values below to predict churn for a single customer:")

    form_data = {}
    for feature in feature_list:
        if "Yes" in feature or "No" in feature or "_" in feature:
            form_data[feature] = st.selectbox(f"{feature}", [0, 1])
        else:
            form_data[feature] = st.number_input(f"{feature}", step=1.0)

    input_df = pd.DataFrame([form_data])

    if st.button("Predict Churn"):
        pred, prob = predict_churn(input_df, model, feature_list)
        st.success(f"Prediction: {'Churn' if pred[0]==1 else 'No Churn'}")
        st.info(f"Probability of churn: {prob[0]:.2f}")
