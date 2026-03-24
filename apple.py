#!/usr/bin/env python
# coding: utf-8

# ===============================
# 🍎 Apple Stock Price Prediction
# Advanced Streamlit Application
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import timedelta

# ===============================
# STEP 1: PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Apple Stock Forecasting",
    layout="wide",
    page_icon="🍎"
)

# ===============================
# STEP 2: LOAD MODEL & FEATURES
# ===============================
@st.cache_resource
def load_model():
    model = joblib.load("xgb_model.pkl")
    features = joblib.load("feature_columns.pkl")
    return model, features

model, features = load_model()

# ===============================
# STEP 3: SIDEBAR NAVIGATION
# ===============================
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", ["📈 Prediction", "ℹ️ About App"])

# ===============================
# ===============================
# 📈 PREDICTION PAGE
# ===============================
# ===============================
if page == "📈 Prediction":

    st.title("🍎 Apple Stock Price Prediction")
    st.markdown(
        "Predict **next 30 days Apple stock prices** using an **advanced XGBoost Machine Learning model**."
    )

    # ===============================
    # STEP 4: USER INPUT
    # ===============================
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", pd.to_datetime("2018-01-01"))
    with col2:
        end_date = st.date_input("End Date", pd.to_datetime("2023-12-31"))

    # ===============================
    # STEP 5: LOAD DATA
    # ===============================
    @st.cache_data
    def load_data(start, end):
        data = yf.download("AAPL", start=start, end=end)
        return data

    df = load_data(start_date, end_date)

    if df.empty:
        st.error("❌ No data available for the selected date range.")
        st.stop()

    # ===============================
    # STEP 6: HISTORICAL VISUAL
    # ===============================
    st.subheader("📊 Historical Apple Stock Prices")
    st.line_chart(df["Close"])

    # ===============================
    # STEP 7: FEATURE ENGINEERING (SAFE & ADVANCED)
    # ===============================
    
    if "Adj Close" not in df.columns:
        df["Adj Close"] = df["Close"]

    df["Apple_Return"] = df["Adj Close"].pct_change()
    df["Daily_Return"] = df["Adj Close"].pct_change()

    df["MA_7"] = df["Adj Close"].rolling(7).mean()
    df["MA_14"] = df["Adj Close"].rolling(14).mean()
    df["MA_30"] = df["Adj Close"].rolling(30).mean()

    df["Volatility_7"] = df["Adj Close"].rolling(7).std()
    df["Volatility_30"] = df["Adj Close"].rolling(30).std()

    df["Lag_1"] = df["Adj Close"].shift(1)
    df["Lag_7"] = df["Adj Close"].shift(7)

    df["EMA_14"] = df["Adj Close"].ewm(span=14, adjust=False).mean()

    df["Month"] = df.index.month
    df["Year"] = df.index.year

    # External feature placeholders (model alignment)
    external_features = [
        "SP500_Close", "SP500_Return", "Relative_Return",
        "SP500_MA_30", "Earnings_Event", "Inflation_Impact",
        "SP500_Lag_1", "SP500_Lag_7"
    ]

    for col in external_features:
        if col not in df.columns:
            df[col] = 0

    df.dropna(inplace=True)

    if len(df) < 30:
            st.warning("⚠️ Please select a date range of at least 1 months for accurate prediction.")
            st.stop()


    # ===============================
    # STEP 8: MODEL INPUT
    # ===============================
    X_input = df[features].tail(1).values

    # ===============================
    # STEP 9: FUTURE FORECAST
    # ===============================
    future_days = 30
    future_predictions = []

    last_features = X_input.copy()

    for _ in range(future_days):
        pred = model.predict(last_features)
        future_predictions.append(float(pred[0]))

    future_dates = pd.date_range(df.index[-1] + timedelta(days=1), periods=future_days)

    forecast_df = pd.DataFrame(
        {"Predicted Price": future_predictions},
        index=future_dates
    )

    # ===============================
    # STEP 10: METRICS (FIXED ERROR)
    # ===============================
    last_close = float(df["Close"].iloc[-1])

    max_price = float(forecast_df["Predicted Price"].max())
    min_price = float(forecast_df["Predicted Price"].min())

    st.subheader("📌 Forecast Summary")
    c1, c2, c3 = st.columns(3)

    c1.metric("Last Closing Price", f"${last_close:.2f}")
    c2.metric("Max Forecast Price", f"${max_price:.2f}")
    c3.metric("Min Forecast Price", f"${min_price:.2f}")

    # ===============================
    # STEP 11: FORECAST VISUAL
    # ===============================
    st.subheader("📈 30-Day Price Forecast")
    st.line_chart(forecast_df)

    # ===============================
    # STEP 12: COMBINED PLOT
    # ===============================
    st.subheader("📉 Historical vs Forecast Comparison")

    plt.figure(figsize=(12, 5))
    plt.plot(df.index, df["Close"], label="Historical Price")
    plt.plot(forecast_df.index, forecast_df["Predicted Price"], linestyle="--", label="Forecast")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

    st.success("✅ Forecast generated successfully!")

# ===============================
# ===============================
# ℹ️ ABOUT PAGE
# ===============================
# ===============================
if page == "ℹ️ About App":

    st.title("ℹ️ About This Application")

    st.markdown("""
    ### 🍎 Apple Stock Price Forecasting System

    This application is an **advanced machine learning–based stock forecasting system**
    developed using **XGBoost**, **Python**, and **Streamlit**.

    ### 🔍 Key Features
    - Advanced Feature Engineering (Returns, Volatility, Moving Averages)
    - Machine Learning–based Forecasting (XGBoost)
    - Interactive Data Visualization
    - Defensive Error Handling
    - Professional Dashboard UI

    ### 🧠 Technologies Used
    - Python
    - Pandas, NumPy
    - XGBoost
    - Yahoo Finance API
    - Streamlit

    ### 👨‍💻 Project Team
    - **Amit Kumar Raychoudhury**
    - **Abhishek Hanchinal**
    - **Suyash Ramesh Patil**
    - **Tilna Kuriakose**
    - **Kumaresh T**
    - **Abhishek**
    - **Muthyala Manish Reddy**

    ### 🚀 Future Enhancements
    - LSTM / Hybrid Models
    - SHAP Explainability
    - Login & User Profiles
    - Multi-stock Support
    - Cloud Deployment

    ---
    📌 *This project is suitable for academic submission, portfolio, and real-world demonstrations.*
    """)

    st.success("Thank you for exploring our application! 🙌")
