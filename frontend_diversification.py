# diversification.py
import streamlit as st
import datetime
import pandas as pd
import backend_diversification as backend  

def diversification_tab():
    st.title("Portfolio Diversification & Optimization")
    
    # Predefined list of 20 major company ticker symbols
    tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "BRK-B", "JNJ", "V", "WMT",
        "JPM", "PG", "UNH", "HD", "DIS", "NVDA", "VZ", "ADBE", "NFLX", "INTC"
    ]
    
    # User selects stocks from a multiselect
    selected_tickers = st.multiselect("Select Stocks", tickers, default=tickers[:5])
    
    # Date range selection for historical data
    start_date = st.date_input("Start Date", datetime.date(2020, 1, 1))
    end_date = st.date_input("End Date", datetime.date.today())
    
    # Submit button to run the optimization
    if st.button("Optimize Portfolio"):
        if len(selected_tickers) < 2:
            st.error("Please select at least two stocks for diversification.")
        else:
            with st.spinner("Fetching data and optimizing portfolio..."):
                try:
                    # Fetch price data from the finance library
                    price_data = backend.fetch_data(selected_tickers, start_date, end_date)
                    # Calculate daily returns
                    returns_df = backend.calculate_returns(price_data)
                    # Compute optimized portfolio weights
                    weights = backend.optimize_portfolio(returns_df)
                    
                    # Display the results in a DataFrame
                    results_df = pd.DataFrame({
                        "Ticker": selected_tickers,
                        "Optimized Weight": weights
                    })
                    st.write("### Optimized Portfolio Weights")
                    st.dataframe(results_df)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
