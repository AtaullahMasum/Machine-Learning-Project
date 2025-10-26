import streamlit as st
import pandas as pd
import joblib
from data_utils import inv_log_target

st.title("House Price Predictor")

model = joblib.load("models/xgb_pipeline.joblib")

st.write("Enter features as CSV row or upload a CSV file with same columns as training.")

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    X_train, y_true = df.drop(columns='price'), df['price']
    preds = inv_log_target(model.predict(X_train))
    
    st.write("Predictions")
    st.write(pd.DataFrame({"Prediction": preds}))
else:
    st.info("Upload CSV to predict.")
