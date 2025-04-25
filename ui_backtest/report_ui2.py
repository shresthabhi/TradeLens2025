import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import date
import scipy.stats as stats
from .helper_functions import *


import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from finlib_data.manager.data_manager import DataManager




# --- The Updated Enhanced Report Function ---
def show_backtest_report():
    st.title("🚀 Backtest Report")

    # --- Get Data from Session State ---
    nav_df = st.session_state.get("nav_df")
    trade_log_df = st.session_state.get("trade_log_df")

    # Check if essential data exists
    if nav_df is None or nav_df.empty:
        st.warning("⚠️ NAV data not found. Please run a backtest first.")
        return
    if trade_log_df is None:
        st.info("ℹ️ Trade log data not found. Displaying NAV performance only.")
        trade_log_df = pd.DataFrame(columns=["Date", "Action", "Ticker", "Price", "Quantity"]).set_index("Date")


    # --- Configuration ---
    INITIAL_INVESTMENT = st.session_state.get("INITIAL_INVESTMENT", 100000)
    BENCHMARK_TICKER = st.session_state.get("BENCHMARK_TICKER", "^GSPC")
    BENCHMARK_NAME = st.session_state.get("BENCHMARK_NAME", "S&P 500")
    PLOT_TEMPLATE = "plotly_white"

    # Ensure index is DateTimeIndex and handle potential errors
    try:
        nav_df.index = pd.to_datetime(nav_df.index)
        if not trade_log_df.empty:
             trade_log_df.index = pd.to_datetime(trade_log_df.index)
    except Exception as e:
        st.error(f"Error processing data indices: {e}. Please check data format.")
        return

    # Ensure NAV column is numeric
    nav_df['NAV'] = pd.to_numeric(nav_df['NAV'], errors='coerce')
    nav_df.dropna(subset=['NAV'], inplace=True)

    if nav_df.empty:
         st.error("NAV data is empty after cleaning. Cannot generate report.")
         return

    # Ensure NAV starts from a positive value for meaningful metrics
    if nav_df['NAV'].iloc[0] <= 0:
        st.error("Initial NAV is not positive. Cannot generate meaningful performance metrics.")
        return


    # Extract date range from NAV
    start_date = nav_df.index.min()
    end_date = nav_df.index.max()
    if pd.isna(start_date) or pd.isna(end_date):
        st.error("Could not determine valid start/end dates from NAV data.")
        return

    # --- Fetch Benchmark Data ---
    benchmark_data = None
    benchmark_returns = pd.Series(dtype=float)
    try:
        try:
             DataManager
        except NameError:
             st.error("DataManager class not found. Please ensure it is imported correctly.")
             raise

        dm = DataManager()
        # Removed the fetching status message: st.info(f"Fetching {BENCHMARK_NAME} data...")
        benchmark_data_raw = dm.get_data(
            ticker=BENCHMARK_TICKER,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            frequency="daily",
            data_type="price_data",
            market="usa"
        )

        if benchmark_data_raw is None or benchmark_data_raw.empty or 'Close' not in benchmark_data_raw.columns:
             st.warning(f"No valid {BENCHMARK_NAME} data returned for the period. Cannot show comparison.")
             benchmark_data = None
        else:
             benchmark_data = benchmark_data_raw.copy()
             if not isinstance(benchmark_data.index, pd.DatetimeIndex):
                 benchmark_data.set_index("date", inplace=True)
                 benchmark_data.index = pd.to_datetime(benchmark_data.index)

             benchmark_data['Close'] = pd.to_numeric(benchmark_data['Close'], errors='coerce')
             benchmark_data.dropna(subset=['Close'], inplace=True)

             if benchmark_data.empty:
                 st.warning(f"No valid numeric 'Close' data found for {BENCHMARK_NAME} after cleaning.")
                 benchmark_data = None
             else:
                 benchmark_data = benchmark_data.loc[start_date:end_date]
                 if benchmark_data.empty:
                     st.warning(f"No {BENCHMARK_NAME} data available within the specific backtest range ({start_date.date()} to {end_date.date()}).")
                     benchmark_data = None
                 else:
                     # Removed the success message: st.success(f"{BENCHMARK_NAME} data loaded successfully.")
                     benchmark_returns = benchmark_data['Close'].pct_change()


    except Exception as e:
        st.error(f"An error occurred fetching benchmark data for {BENCHMARK_TICKER}: {e}")
        benchmark_data = None
        benchmark_returns = pd.Series(dtype=float)


    # --- Data Preparation ---
    if nav_df['NAV'].iloc[0] > 0:
        nav_df['Normalized NAV'] = (nav_df['NAV'] / nav_df['NAV'].iloc[0]) * 100
    else:
        st.warning("Initial NAV is not positive. Cannot normalize NAV for charting.")
        nav_df['Normalized NAV'] = np.nan


    benchmark_perf = pd.Series(dtype=float)
    if benchmark_data is not None and not benchmark_data.empty:
        first_valid_bm_index = benchmark_data['Close'].first_valid_index()
        if first_valid_bm_index is not None and nav_df.first_valid_index() is not None and first_valid_bm_index <= nav_df.first_valid_index():
            bm_start_base_value = benchmark_data['Close'].loc[first_valid_bm_index]
            if bm_start_base_value > 0:
                 benchmark_data['Normalized Close'] = (benchmark_data['Close'] / bm_start_base_value) * 100
                 benchmark_perf = benchmark_data['Normalized Close'].reindex(nav_df.index, method='ffill')
                 benchmark_perf.fillna(method='ffill', inplace=True)
            else:
                 st.warning(f"{BENCHMARK_NAME} starting value is not positive; cannot reliably normalize benchmark.")
                 benchmark_perf = pd.Series(dtype=float)
        else:
             st.warning(f"{BENCHMARK_NAME} data starts after NAV data or is empty; cannot reliably normalize benchmark relative to strategy start.")
             benchmark_perf = pd.Series(dtype=float)


    nav_df['Daily Return'] = nav_df['NAV'].pct_change()


    buy_trades = pd.DataFrame()
    sell_trades = pd.DataFrame()
    if not trade_log_df.empty:
        trade_log_df['Price'] = pd.to_numeric(trade_log_df['Price'], errors='coerce').fillna(0)
        trade_log_df['Quantity'] = pd.to_numeric(trade_log_df['Quantity'], errors='coerce').fillna(0)
        trade_log_df['Trade Value'] = trade_log_df['Price'] * trade_log_df['Quantity']

        buy_trades = trade_log_df[trade_log_df['Action'].str.lower() == 'buy']
        sell_trades = trade_log_df[trade_log_df['Action'].str.lower() == 'sell']


    # --- Core Metrics Calculations ---
    total_days = (end_date - start_date).days
    years = max(total_days / 365.25, 1/365.25)

    start_nav = nav_df['NAV'].iloc[0]
    final_nav = nav_df['NAV'].iloc[-1]
    strategy_total_return = (final_nav / start_nav) - 1 if start_nav else 0.0
    strategy_cagr = calculate_cagr(start_nav, final_nav, years)
    strategy_mdd = calculate_max_drawdown(nav_df['Daily Return'].iloc[1:])

    benchmark_total_return = 0.0
    benchmark_cagr = 0.0
    benchmark_mdd = 0.0
    if benchmark_data is not None and not benchmark_data.empty:
        bm_valid_indices = benchmark_data['Close'].dropna().index
        if not bm_valid_indices.empty:
             bm_start_date_calc = max(start_date, bm_valid_indices.min())
             bm_end_date_calc = min(end_date, bm_valid_indices.max())

             if bm_start_date_calc <= bm_end_date_calc:
                 bm_start_val = benchmark_data['Close'].loc[bm_start_date_calc]
                 bm_end_val = benchmark_data['Close'].loc[bm_end_date_calc]
                 bm_years_calc = max((bm_end_date_calc - bm_start_date_calc).days / 365.25, 1/365.25)

                 if bm_start_val > 0:
                      benchmark_total_return = (bm_end_val / bm_start_val) - 1
                      benchmark_cagr = calculate_cagr(bm_start_val, bm_end_val, bm_years_calc)

                 benchmark_mdd = calculate_max_drawdown(benchmark_data['Close'].pct_change().dropna())


    # --- UI Layout ---
    st.markdown("---")
    st.subheader("📊 Key Performance Indicators (KPIs)")

    kpi_cols = st.columns(3)
    kpi_cols[0].metric("Period", f"{start_date.strftime('%b-%y')} to {end_date.strftime('%b-%y')}")
    kpi_cols[1].metric("Duration (Years)", f"{years:.2f}")
    kpi_cols[2].metric("Number of Trades", f"{len(trade_log_df)}")

    st.markdown("---")

    # --- Performance Chart (Moved up) ---
    st.subheader("📈 Portfolio Growth vs Benchmark")

    fig = go.Figure()

    if not nav_df['Normalized NAV'].dropna().empty:
        fig.add_trace(go.Scatter(
            x=nav_df.index,
            y=nav_df['Normalized NAV'],
            mode='lines',
            name='Strategy NAV',
            line=dict(color='royalblue', width=2),
            hovertemplate='Date: %{x|%Y-%m-%d}<br>Strategy NAV: %{y:.2f}<extra></extra>'
        ))


    if not benchmark_perf.empty:
        fig.add_trace(go.Scatter(
            x=benchmark_perf.index,
            y=benchmark_perf,
            mode='lines',
            name=BENCHMARK_NAME,
            line=dict(color='grey', width=1.5, dash='dash'),
             hovertemplate='Date: %{x|%Y-%m-%d}<br>Benchmark: %{y:.2f}<extra></extra>'
        ))


    if not buy_trades.empty and not nav_df['Normalized NAV'].dropna().empty:
        buy_nav_values = nav_df['Normalized NAV'].reindex(buy_trades.index, method='ffill').dropna()
        buy_trades_aligned = buy_trades.loc[buy_nav_values.index]

        if not buy_trades_aligned.empty:
            fig.add_trace(go.Scatter(
                x=buy_trades_aligned.index,
                y=buy_nav_values,
                mode='markers',
                name='Buy Orders',
                marker=dict(
                    color='rgba(0, 128, 0, 0.7)',
                    symbol='triangle-up',
                    size=8,
                    line=dict(width=1, color='DarkSlateGrey')
                ),
                customdata=buy_trades_aligned[['Ticker', 'Price', 'Quantity', 'Trade Value']],
                hovertemplate=(
                    '<b>BUY</b> %{customdata[0]}<br>'
                    'Date: %{x|%Y-%m-%d}<br>'
                    'Price: $%{customdata[1]:.2f}<br>'
                    'Quantity: %{customdata[2]:.0f}<br>'
                    'Value: $%{customdata[3]:,.0f}'
                    '<extra></extra>'
                )
            ))


    if not sell_trades.empty and not nav_df['Normalized NAV'].dropna().empty:
        sell_nav_values = nav_df['Normalized NAV'].reindex(sell_trades.index, method='ffill').dropna()
        sell_trades_aligned = sell_trades.loc[sell_nav_values.index]

        if not sell_trades_aligned.empty:
            fig.add_trace(go.Scatter(
                x=sell_trades_aligned.index,
                y=sell_nav_values,
                mode='markers',
                name='Sell Orders',
                marker=dict(
                    color='rgba(255, 0, 0, 0.7)',
                    symbol='triangle-down',
                    size=8,
                    line=dict(width=1, color='DarkSlateGrey')
                ),
                customdata=sell_trades_aligned[['Ticker', 'Price', 'Quantity', 'Trade Value']],
                 hovertemplate=(
                    '<b>SELL</b> %{customdata[0]}<br>'
                    'Date: %{x|%Y-%m-%d}<br>'
                    'Price: $%{customdata[1]:.2f}<br>'
                    'Quantity: %{customdata[2]:.0f}<br>'
                    'Value: $%{customdata[3]:,.0f}'
                    '<extra></extra>'
                )
            ))


    fig.update_layout(
        title='Portfolio Value Over Time (Normalized to 100)',
        xaxis_title='Date',
        yaxis_title='Normalized Value',
        legend_title='Legend',
        hovermode='x unified',
        template=PLOT_TEMPLATE,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # --- Display Detailed Performance & Risk Metrics (Moved down) ---
    # Pass all calculated metrics to the display function
    display_risk_performance_metrics(
        nav_df, benchmark_returns, BENCHMARK_NAME,
        strategy_total_return, benchmark_total_return,
        strategy_cagr, benchmark_cagr,
        strategy_mdd, benchmark_mdd
    )


    # --- Drawdown Chart (Moved down) ---
    display_drawdown_chart(nav_df, PLOT_TEMPLATE)


    # --- Annual Returns Table (Moved down) ---
    display_annual_returns(nav_df, benchmark_data, BENCHMARK_NAME)


    # --- Monthly Returns Table (Moved down, heatmap removed) ---
    display_monthly_returns(nav_df, benchmark_data) # Pass benchmark_data DataFrame


    # --- Trade Log Details (Remains at the bottom) ---
    if not trade_log_df.empty:
        with st.expander("📜 View Detailed Trade Log", expanded=False):
            display_log = trade_log_df[['Action', 'Ticker', 'Price', 'Quantity', 'Trade Value']].sort_index()
            st.dataframe(display_log)
    else:
        st.info("No trades were logged during this backtest.")