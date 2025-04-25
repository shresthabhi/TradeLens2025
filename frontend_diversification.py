# import streamlit as st
# import datetime
# import pandas as pd
# import backend_diversification as backend

# def diversification_tab():
#     st.title("📈 Portfolio Diversification & Optimization")

#     st.markdown("Choose how you’d like to input your portfolio:")
#     input_mode = st.radio(
#         "Input Method",
#         ["📁 Upload My Portfolio", "🛠️ Manually Select Stocks"]
#     )

#     # Large set of popular tickers for ease
#     tickers = sorted([
#         "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "BRK-B", "JNJ", "V", "WMT",
#         "JPM", "PG", "UNH", "HD", "DIS", "NVDA", "VZ", "ADBE", "NFLX", "INTC",
#         "PEP", "KO", "PFE", "MRK", "CVX", "XOM", "T", "BA", "IBM", "CRM",
#         "NKE", "QCOM", "ORCL", "ABBV", "TMO", "ABT", "MDT", "LLY", "BMY", "COST",
#         "MCD", "HON", "UPS", "LOW", "GS", "AXP", "GE", "CAT", "DE", "BLK"
#     ])

#     portfolio_df = None
#     selected_tickers = []

#     if input_mode == "📁 Upload My Portfolio":
#         uploaded_file = st.file_uploader("Upload CSV with columns: `ticker`, `current_value`", type=["csv"])
#         if uploaded_file:
#             try:
#                 portfolio_df = pd.read_csv(uploaded_file)
#                 if {"ticker", "current_value"}.issubset(portfolio_df.columns):
#                     st.success("✅ Portfolio uploaded successfully!")
#                     st.dataframe(portfolio_df)
#                     selected_tickers = portfolio_df["ticker"].tolist()
#                 else:
#                     st.error("CSV must contain `ticker` and `current_value` columns.")
#             except Exception as e:
#                 st.error(f"Failed to read file: {e}")

#     elif input_mode == "🛠️ Manually Select Stocks":
#         selected_tickers = st.multiselect(
#             "Select or type tickers",
#             options=tickers,
#             default=[],
#             help="Choose from the list or type your own ticker (e.g., 'BABA', 'ZM')"
#         )

#     st.markdown("---")

#     st.subheader("📅 Time Period for Analysis")
#     col1, col2 = st.columns(2)
#     with col1:
#         start_date = st.date_input("Start Date", datetime.date(2020, 1, 1))
#     with col2:
#         end_date = st.date_input("End Date", datetime.date.today())

#     st.markdown("📉 Optional: Enter Risk-Free Rate (default: 0.02)")
#     risk_free_rate = st.number_input("Risk-Free Rate", value=0.02, min_value=0.0, max_value=1.0, step=0.01)

#     st.markdown("---")
    
#     if st.button("🚀 Optimize Portfolio"):
#         if len(selected_tickers) < 2:
#             st.error("Please select or upload at least two stocks.")
#         else:
#             with st.spinner("Fetching data and optimizing portfolio..."):
#                 try:
#                     price_data = backend.fetch_data(selected_tickers, start_date, end_date)
#                     returns_df = backend.calculate_returns(price_data)
#                     weights = backend.optimize_portfolio(returns_df, risk_free_rate=risk_free_rate)

#                     results_df = pd.DataFrame({
#                         "Ticker": selected_tickers,
#                         "Optimized Weight": weights
#                     })

#                     st.success("✅ Portfolio optimized successfully!")
#                     st.write("### 📊 Optimized Portfolio Weights")
#                     st.dataframe(results_df)

#                 except Exception as e:
#                     st.error(f"An error occurred: {e}")


import streamlit as st
import datetime
import pandas as pd
import plotly.express as px
import backend_diversification as backend

def diversification_tab():
    st.title("📈 Portfolio Diversification & Optimization")

    st.markdown("Choose how you’d like to input your portfolio:")
    input_mode = st.radio(
        "Input Method",
        ["📁 Upload My Portfolio", "🛠️ Manually Select Stocks"]
    )

    # Predefined large ticker set
    tickers = sorted([
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "BRK-B", "JNJ", "V", "WMT",
        "JPM", "PG", "UNH", "HD", "DIS", "NVDA", "VZ", "ADBE", "NFLX", "INTC",
        "PEP", "KO", "PFE", "MRK", "CVX", "XOM", "T", "BA", "IBM", "CRM",
        "NKE", "QCOM", "ORCL", "ABBV", "TMO", "ABT", "MDT", "LLY", "BMY", "COST",
        "MCD", "HON", "UPS", "LOW", "GS", "AXP", "GE", "CAT", "DE", "BLK"
    ])

    portfolio_df = None
    selected_tickers = []

    if input_mode == "📁 Upload My Portfolio":
        uploaded_file = st.file_uploader("Upload CSV with columns: `ticker`, `current_value`", type=["csv"])
        if uploaded_file:
            try:
                portfolio_df = pd.read_csv(uploaded_file)
                if {"ticker", "current_value"}.issubset(portfolio_df.columns):
                    st.success("✅ Portfolio uploaded successfully!")
                    st.dataframe(portfolio_df)
                    selected_tickers = portfolio_df["ticker"].tolist()
                else:
                    st.error("CSV must contain `ticker` and `current_value` columns.")
            except Exception as e:
                st.error(f"Failed to read file: {e}")

    elif input_mode == "🛠️ Manually Select Stocks":
        selected_tickers = st.multiselect(
            "Select or type tickers",
            options=tickers,
            default=[],
            help="Choose from the list or type your own ticker (e.g., 'BABA', 'ZM')"
        )

    st.markdown("---")

    st.subheader("📅 Time Period for Analysis")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime.date(2020, 1, 1))
    with col2:
        end_date = st.date_input("End Date", datetime.date(2023, 12, 1))

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    st.markdown("📉 Optional: Enter Risk-Free Rate (default: 0.02)")
    risk_free_rate = st.number_input("Risk-Free Rate", value=0.02, min_value=0.0, max_value=1.0, step=0.01)

    st.markdown("---")
    
    if st.button("🚀 Optimize Portfolio"):
        if len(selected_tickers) < 2:
            st.error("Please select or upload at least two stocks.")
        else:
            with st.spinner("Fetching data and optimizing portfolio..."):
                try:
                    # Optimization
                    max_sharpe_weights = backend.get_max_sharpe_portfolio(selected_tickers, start_date, end_date, risk_free_rate)
                    min_risk_weights = backend.get_min_volatility_portfolio(selected_tickers, start_date, end_date, risk_free_rate)

                    # Risk/Return Analysis
                    max_sharpe_stats = backend.analyze_existing_portfolio(selected_tickers, max_sharpe_weights, start_date, end_date, risk_free_rate)
                    min_risk_stats = backend.analyze_existing_portfolio(selected_tickers, min_risk_weights, start_date, end_date, risk_free_rate)

                    # Generate random portfolios for plotting
                    sim_df = backend.generate_risk_return_chart(selected_tickers, start_date, end_date, risk_free_rate)

                    # Format Optimized Portfolio Weights
                    sharpe_df = pd.DataFrame({
                        "Ticker": selected_tickers,
                        "Weight (%)": (max_sharpe_weights * 100).round(2)
                    })

                    risk_df = pd.DataFrame({
                        "Ticker": selected_tickers,
                        "Weight (%)": (min_risk_weights * 100).round(2)
                    })

                    st.success("✅ Portfolio optimized successfully!")
                    st.subheader("📊 Max Sharpe Ratio Portfolio Weights")
                    st.dataframe(sharpe_df)

                    st.subheader("📉 Minimum Risk Portfolio Weights")
                    st.dataframe(risk_df)

                    st.markdown("### 📈 Portfolio Statistics")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Max Sharpe Portfolio Return", f"{(max_sharpe_stats['Return']*100):.2f}%")
                        st.metric("Max Sharpe Portfolio Risk", f"{(max_sharpe_stats['Risk']*100):.2f}%")
                        st.metric("Max Sharpe Ratio", f"{max_sharpe_stats['Sharpe Ratio']:.2f}")
                    with col2:
                        st.metric("Min Risk Portfolio Return", f"{(min_risk_stats['Return']*100):.2f}%")
                        st.metric("Min Risk Portfolio Risk", f"{(min_risk_stats['Risk']*100):.2f}%")
                        st.metric("Min Risk Sharpe Ratio", f"{min_risk_stats['Sharpe Ratio']:.2f}")

                    # If user uploaded a portfolio, analyze it
                    user_stats = None
                    if portfolio_df is not None:
                        current_values = portfolio_df["current_value"].values
                        user_stats = backend.analyze_existing_portfolio(selected_tickers, current_values, start_date, end_date, risk_free_rate)
                        st.subheader("👤 User Portfolio Stats")
                        st.metric("User Portfolio Return", f"{(user_stats['Return']*100):.2f}%")
                        st.metric("User Portfolio Risk", f"{(user_stats['Risk']*100):.2f}%")
                        st.metric("User Sharpe Ratio", f"{user_stats['Sharpe Ratio']:.2f}")

                    # Plot Risk vs Return Scatter
                    st.subheader("📊 Risk vs Return Scatter Plot")
                    fig = px.scatter(sim_df, x="Risk", y="Return", color="Sharpe", title="Portfolio Diversification - Risk vs Return")

                    # Add markers for optimized portfolios
                    fig.add_scatter(x=[max_sharpe_stats['Risk']], y=[max_sharpe_stats['Return']], mode="markers+text",
                                    marker=dict(color="green", size=12),
                                    text=["Max Sharpe"], textposition="top center", name="Max Sharpe")
                    
                    fig.add_scatter(x=[min_risk_stats['Risk']], y=[min_risk_stats['Return']], mode="markers+text",
                                    marker=dict(color="blue", size=12),
                                    text=["Min Risk"], textposition="top center", name="Min Risk")

                    if user_stats:
                        fig.add_scatter(x=[user_stats['Risk']], y=[user_stats['Return']], mode="markers+text",
                                        marker=dict(color="red", size=12, symbol="diamond"),
                                        text=["Your Portfolio"], textposition="top center", name="User Portfolio")

                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"An error occurred: {e}")
