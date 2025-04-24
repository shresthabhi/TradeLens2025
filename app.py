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

# Page settings
st.set_page_config(page_title="Financial Analytics Dashboard", layout="wide")
st.title("📊 Financial Analytics Dashboard")

tabs = ["Stock Screening", "Custom ML Models", "Backtesting", "Diversification"]
choice = st.sidebar.radio("Navigation", tabs)

@st.cache_data
def load_data():
    df = pd.read_csv("financial_ratios_2010_2024_v3.csv")
    df.columns = df.columns.str.lower()  # standardize columns
    df['public_date'] = pd.to_datetime(df['public_date'], errors='coerce')  # parse date once
    return df

df = load_data()

# Shared file uploader
uploaded_file = st.sidebar.file_uploader("Upload Excel or CSV file", type=["xlsx", "csv"])

# # Load data
# if uploaded_file:
#     if uploaded_file.name.endswith(".csv"):
#         df = pd.read_csv(uploaded_file)
#     else:
#         df = pd.read_excel(uploaded_file)

# Stock Screening
if choice == "Stock Screening":
    # st.header("Stock Screening & Filtering")

    # if df is not None:
    #     st.dataframe(df.head())

    #     numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    #     col1, col2 = st.columns(2)

    #     with col1:
    #         x_axis = st.selectbox("X-axis column", options=numeric_cols)
    #     with col2:
    #         y_axis = st.selectbox("Y-axis column", options=numeric_cols)

    #     fig = px.scatter(df, x=x_axis, y=y_axis, title=f"{y_axis} vs {x_axis}")
    #     st.plotly_chart(fig, use_container_width=True)
    # else:
    #     st.info("Please upload data to begin screening.")
    screening_tab(df)
# ML Models
elif choice == "Custom ML Models":
    st.header(" Machine Learning Models")

    if df is not None:
        st.subheader("Select Features for ML")
        ml_cols = st.multiselect("Choose columns for modeling", df.select_dtypes(include="number").columns)

        if len(ml_cols) >= 2:
            model_type = st.selectbox("Choose a model", ["K-Means Clustering", "Linear Regression"])

            if model_type == "K-Means Clustering":
                k = st.slider("Number of clusters", 2, 10, 3)
                kmeans = KMeans(n_clusters=k)
                clusters = kmeans.fit_predict(df[ml_cols])
                df["Cluster"] = clusters
                fig = px.scatter(df, x=ml_cols[0], y=ml_cols[1], color=df["Cluster"].astype(str),
                                 title="K-Means Clustering")
                st.plotly_chart(fig, use_container_width=True)

            elif model_type == "Linear Regression":
                x_col = st.selectbox("Feature (X)", ml_cols)
                y_col = st.selectbox("Target (Y)", ml_cols)
                model = LinearRegression()
                X = df[[x_col]]
                y = df[y_col]
                model.fit(X, y)
                y_pred = model.predict(X)
                df["Prediction"] = y_pred
                fig = px.scatter(df, x=x_col, y=y_col, title="Linear Regression")
                fig.add_scatter(x=df[x_col], y=y_pred, mode="lines", name="Regression Line")
                st.plotly_chart(fig, use_container_width=True)

                st.write("R-squared:", model.score(X, y))
        else:
            st.warning("Select at least two numeric columns.")
    else:
        st.info("Upload data to use ML models.")

elif(choice == "Diversification"):
    
    diversification_tab()

# Backtesting (Placeholder)
elif choice == "Backtesting":
    st.header("Strategy Backtesting")
    st.info("Coming soon: Upload a portfolio and define entry/exit rules.")

