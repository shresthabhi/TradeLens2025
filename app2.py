# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

import datetime
from frontend_diversification import diversification_tab
import sys

#sentiment analysis 
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import yfinance as yf

def get_news(ticker):
    stock = yf.Ticker(ticker)
    try:
        return stock.news
    except Exception:
        return []
    
if "torch" in sys.modules:
    del sys.modules["torch"]


# Page settings
st.set_page_config(page_title="Financial Analytics Dashboard", layout="wide")
st.title("📊 Financial Analytics Dashboard")

tabs = ["Stock Screening", "Custom ML Models", "Backtesting", "Diversification", "Sentiment Analysis"]
choice = st.sidebar.radio("Navigation", tabs)

# Shared file uploader
uploaded_file = st.sidebar.file_uploader("Upload Excel or CSV file", type=["xlsx", "csv"])

# Load FinBERT once (do this globally)
@st.cache_resource
def load_finbert():
    tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    model = BertForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    return sentiment_pipeline

finbert = load_finbert()

    

# Load data
df = None
if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

# Stock Screening
if choice == "Stock Screening":
    st.header("Stock Screening & Filtering")

    if df is not None:
        st.dataframe(df.head())

        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
        col1, col2 = st.columns(2)

        with col1:
            x_axis = st.selectbox("X-axis column", options=numeric_cols)
        with col2:
            y_axis = st.selectbox("Y-axis column", options=numeric_cols)

        fig = px.scatter(df, x=x_axis, y=y_axis, title=f"{y_axis} vs {x_axis}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Please upload data to begin screening.")

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

elif choice == "Sentiment Analysis":
    st.header("🧾 Company News + Sentiment Analysis")

    ticker = st.text_input("Enter stock ticker (e.g., AAPL, TSLA, NVDA)", value="AAPL")

    if st.button("Fetch & Analyze News"):
        news_items = get_news(ticker.upper())
        
        if not news_items:
            st.warning("No news found or invalid ticker.")
        else:
            results = []
            for item in news_items:
                title = item['title']
                sentiment = finbert(title)[0]
                results.append({
                    "Title": title,
                    "Sentiment": sentiment['label'],
                    "Confidence": round(sentiment['score'], 2)
                })

            results_df = pd.DataFrame(results)
            st.subheader(f" News Sentiment for {ticker.upper()}")
            st.dataframe(results_df)

            # Optional plot
            fig = px.histogram(results_df, x="Sentiment", title="Sentiment Distribution")
            st.plotly_chart(fig, use_container_width=True)


