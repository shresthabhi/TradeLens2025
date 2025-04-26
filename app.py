# app.py
import streamlit as st
import pandas as pd
# Make sure all necessary imports are here
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import datetime

import backend_diversification as backend
from frontend_diversification import diversification_tab
from frontend_screening import screening_tab
from frontend_kmeans import clustering_tab # Your KMeans function using internal tabs
from sentiment_panel import sentiment_analysis_panel
from home import home

from ui_backtest.backtest_tab import backtest_tab # Your Backtesting function


# Page settings
st.set_page_config(page_title="Financial Analytics Dashboard", layout="wide")
# st.title("📊 Financial Analytics Dashboard")

# --- Navigation ---
# Remove "Diversification" from the main sidebar navigation
tabs = ["Home", "Stock Screening", "Analytical Tools", "Backtesting"]
choice = st.sidebar.radio("Navigation", tabs)


# --- Data Loading ---
@st.cache_data
def load_data():
    # Ensure this path is correct
    try:
        df = pd.read_csv("financial_ratios.csv")
        df.columns = df.columns.str.lower()  # standardize columns
        df['public_date'] = pd.to_datetime(df['public_date'], errors='coerce')  # parse date once
        return df
    except FileNotFoundError:
        st.error("Error: financial_ratios_2010_2024_v3.csv not found. Please ensure the file is in the correct directory.")
        st.stop() # Stop execution if file is not found
    except Exception as e:
         st.error(f"Error loading data: {e}")
         st.stop()


df = load_data()

if choice == "Home":
    home()

elif choice == "Stock Screening":
    st.subheader("Stock Screening") # Optional: Add a subheader for clarity
    screening_tab(df)

elif choice == "Analytical Tools":
    st.subheader("Analytical Tools") # Optional: Add a main subheader

    if 'model_sub_choice' not in st.session_state:
        st.session_state.model_sub_choice = 'Sentiment' # Set a default view

    col1, col2, col3 = st.columns(3)


    with col1:
        if st.button("🔍 Sentiment Analysis", use_container_width=True):
            st.session_state.model_sub_choice = 'Sentiment'
            st.rerun()

    with col3:
        if st.button("📈 KMeans Clustering", use_container_width=True):
             st.session_state.model_sub_choice = 'KMeans'
             st.rerun()

    with col2:
        if st.button("🌱 Diversification Analysis", use_container_width=True):
             st.session_state.model_sub_choice = 'Diversification'
             st.rerun()

    st.markdown("---") 

    if st.session_state.model_sub_choice == "Sentiment":
        st.write("Explore what the media is talking about")
        sentiment_analysis_panel()

    elif st.session_state.model_sub_choice == "KMeans":
        st.write("Explore KMeans Clustering here.")
        clustering_tab()

    elif st.session_state.model_sub_choice == "Diversification":
        st.write("Check if your portfolio is diversified!") 
        diversification_tab()


elif choice == "Backtesting":
    st.subheader("Backtesting Strategy") # Optional: Add a subheader
    backtest_tab()