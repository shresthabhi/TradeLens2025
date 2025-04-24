# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

import datetime
import backend_diversification as backend  
from frontend_diversification import diversification_tab
from frontend_screening import screening_tab
from frontend_kmeans import clustering_tab

# Page settings
st.set_page_config(page_title="Financial Analytics Dashboard", layout="wide")
# st.title("📊 Financial Analytics Dashboard")

tabs = ["Stock Screening", "Custom ML Models", "Backtesting", "Diversification"]
choice = st.sidebar.radio("Navigation", tabs)

@st.cache_data
def load_data():
    df = pd.read_csv("financial_ratios_2010_2024_v3.csv")
    df.columns = df.columns.str.lower()  # standardize columns
    df['public_date'] = pd.to_datetime(df['public_date'], errors='coerce')  # parse date once
    return df

df = load_data()


if choice == "Stock Screening":
    screening_tab(df)

elif choice == "Custom ML Models":
    clustering_tab()

elif(choice == "Diversification"):
    diversification_tab()

# Backtesting (Placeholder)
elif choice == "Backtesting":
    st.header("Strategy Backtesting")
    st.info("Coming soon: Upload a portfolio and define entry/exit rules.")

