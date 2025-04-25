import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import date

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from finlib_data.manager.data_manager import DataManager


def calculate_cagr(start_value, end_value, years):
    if years <= 0 or start_value <= 0: # Avoid division by zero or invalid calculation
        return 0.0
    if end_value <= 0:
        return -1.0 
    return (end_value / start_value) ** (1 / years) - 1


def calculate_max_drawdown(returns_series):
    if returns_series.empty:
        return 0.0

    returns_series = pd.to_numeric(returns_series, errors='coerce').fillna(0)
    cumulative_returns = (1 + returns_series).cumprod()
    if cumulative_returns.empty or not np.all(np.isfinite(cumulative_returns)):
        return 0.0 # Return 0 if cumulative returns are empty or contain non-finite values
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()
    return max_drawdown if pd.notna(max_drawdown) else 0.0


# --- The Enhanced Report Function ---
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
        st.warning("⚠️ Trade log data not found. Displaying NAV only.")
        trade_log_df = pd.DataFrame(columns=["Date", "Action", "Ticker", "Price", "Quantity"]).set_index("Date")


    # --- Configuration ---
    INITIAL_INVESTMENT = 100000 # Your starting capital (Consider getting this from session state if it varies)
    BENCHMARK_TICKER = "^GSPC"
    BENCHMARK_NAME = "S&P 500"
    PLOT_TEMPLATE = "plotly_white" # Try "plotly_dark", "seaborn", etc.

    # Ensure index is DateTimeIndex and handle potential errors
    try:
        nav_df.index = pd.to_datetime(nav_df.index)
        if not trade_log_df.empty:
             trade_log_df.index = pd.to_datetime(trade_log_df.index)
    except Exception as e:
        st.error(f"Error processing data indices: {e}. Please check data format.")
        return

    # Extract date range from NAV
    start_date = nav_df.index.min()
    end_date = nav_df.index.max()
    if pd.isna(start_date) or pd.isna(end_date):
        st.error("Could not determine valid start/end dates from NAV data.")
        return

    # --- Fetch Benchmark Data ---
    benchmark_data = None 
    try:
        if 'DataManager' not in globals() and 'DataManager' not in locals():
             raise NameError("DataManager class not found. Please ensure it is imported.")

        dm = DataManager() # Instantiate your data manager
        st.info(f"Fetching {BENCHMARK_NAME} data...")
        benchmark_data_raw = dm.get_data(
            ticker=BENCHMARK_TICKER,
            start_date=start_date.strftime("%Y-%m-%d"), # Pass date object if required by get_data
            end_date=end_date.strftime("%Y-%m-%d"),     # Pass date object if required by get_data
            frequency="daily",
            data_type="price_data",
            market="usa"
        )

        benchmark_data_raw = benchmark_data_raw.set_index("date")

        # Validate fetched data
        if benchmark_data_raw is None or benchmark_data_raw.empty or 'Close' not in benchmark_data_raw.columns:
             st.warning(f"No valid {BENCHMARK_NAME} data returned for the period. Cannot show comparison.")
             benchmark_data = None
        else:
             # Ensure index is DatetimeIndex and handle potential errors
             benchmark_data = benchmark_data_raw.copy() # Work on a copy
             if not isinstance(benchmark_data.index, pd.DatetimeIndex):
                 benchmark_data.index = pd.to_datetime(benchmark_data.index)
             benchmark_data['Close'] = pd.to_numeric(benchmark_data['Close'], errors='coerce')
             benchmark_data.dropna(subset=['Close'], inplace=True) # Drop rows where Close is invalid

             if benchmark_data.empty:
                 st.warning(f"No valid numeric 'Close' data found for {BENCHMARK_NAME} after cleaning.")
                 benchmark_data = None
             else:
                 # Filter to the exact backtest date range AFTER converting index
                 benchmark_data = benchmark_data.loc[start_date:end_date]
                 if benchmark_data.empty:
                     st.warning(f"No {BENCHMARK_NAME} data available within the specific backtest range ({start_date.date()} to {end_date.date()}).")
                     benchmark_data = None
                 else:
                     st.success(f"{BENCHMARK_NAME} data loaded successfully.")

    except NameError as e:
         st.error(e) # Show the NameError related to DataManager
         benchmark_data = None
    except Exception as e:
        st.error(f"An error occurred fetching benchmark data for {BENCHMARK_TICKER}: {e}")
        benchmark_data = None

    # --- Data Preparation ---
    # 1. Performance Normalization (Starting at 100)
    nav_df['Normalized NAV'] = (nav_df['NAV'] / nav_df['NAV'].iloc[0]) * 100
    benchmark_perf = pd.Series(dtype=float) # Initialize empty series
    if benchmark_data is not None and not benchmark_data.empty:
        # Ensure benchmark data starts on or before nav data for proper normalization base
        first_valid_bm_index = benchmark_data.first_valid_index()
        if first_valid_bm_index is not None and first_valid_bm_index <= nav_df.first_valid_index():
            benchmark_data['Normalized Close'] = (benchmark_data['Close'] / benchmark_data['Close'].loc[first_valid_bm_index]) * 100
            # Reindex benchmark to match NAV dates (forward fill for missing dates like weekends)
            benchmark_perf = benchmark_data['Normalized Close'].reindex(nav_df.index, method='ffill')
            benchmark_perf.fillna(method='ffill', inplace=True) # Fill potential leading NaNs if NAV starts before BM
        else:
             st.warning(f"{BENCHMARK_NAME} data starts after NAV data or is empty; cannot reliably normalize benchmark.")


    # 2. Daily/Periodic Returns
    nav_df['Daily Return'] = nav_df['NAV'].pct_change() # Leave NaNs for first entry
    benchmark_returns = pd.Series(dtype=float) # Initialize empty series
    if benchmark_data is not None and not benchmark_data.empty:
         benchmark_data['Daily Return'] = benchmark_data['Close'].pct_change()
         benchmark_returns = benchmark_data['Daily Return'].reindex(nav_df.index, method='ffill')


    # 3. Trade Log Enhancement (check if trade_log_df is not empty)
    buy_trades = pd.DataFrame()
    sell_trades = pd.DataFrame()
    if not trade_log_df.empty:
        trade_log_df['Trade Value'] = trade_log_df['Price'] * trade_log_df['Quantity']
        buy_trades = trade_log_df[trade_log_df['Action'].str.lower() == 'buy']
        sell_trades = trade_log_df[trade_log_df['Action'].str.lower() == 'sell']


    # --- Calculations ---
    # Time Period
    total_days = (end_date - start_date).days
    years = max(total_days / 365.25, 1/365.25) # Ensure years is not zero, use minimum of 1 day

    # Strategy Metrics
    start_nav = nav_df['NAV'].iloc[0]
    final_nav = nav_df['NAV'].iloc[-1]
    strategy_total_return = (final_nav / start_nav) - 1 if start_nav else 0.0
    strategy_cagr = calculate_cagr(start_nav, final_nav, years)
    # Pass series without the first NaN to max drawdown
    strategy_mdd = calculate_max_drawdown(nav_df['Daily Return'].iloc[1:])


    # Benchmark Metrics
    benchmark_total_return = 0.0
    benchmark_cagr = 0.0
    benchmark_mdd = 0.0
    if benchmark_data is not None and not benchmark_data.empty:
        # Use .loc for safety, find first/last valid index within range
        bm_valid_indices = benchmark_data['Close'].dropna().index
        if not bm_valid_indices.empty:
             bm_start_date = max(start_date, bm_valid_indices.min())
             bm_end_date = min(end_date, bm_valid_indices.max())
             bm_start_val = benchmark_data['Close'].loc[bm_start_date]
             bm_end_val = benchmark_data['Close'].loc[bm_end_date]
             bm_years = max((bm_end_date - bm_start_date).days / 365.25, 1/365.25)

             if bm_start_val > 0:
                  benchmark_total_return = (bm_end_val / bm_start_val) - 1
                  benchmark_cagr = calculate_cagr(bm_start_val, bm_end_val, bm_years)
             # Calculate drawdown on returns aligned with NAV dates
             benchmark_mdd = calculate_max_drawdown(benchmark_returns.iloc[1:])


    # --- UI Layout ---
    st.markdown("---")
    st.subheader("📊 Key Performance Indicators (KPIs)")

    kpi_cols = st.columns(3)
    kpi_cols[0].metric("Period", f"{start_date.strftime('%B-%Y')} to {end_date.strftime('%B-%Y')}")
    kpi_cols[1].metric("Duration (Years)", f"{years:.2f}")
    kpi_cols[2].metric("Number of Trades", f"{len(trade_log_df)}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Strategy Performance**")
        st.metric("Total Return", f"{strategy_total_return:.2%}")
        st.metric("CAGR", f"{strategy_cagr:.2%}")
        st.metric("Max Drawdown", f"{strategy_mdd:.2%}", delta_color="inverse") # Lower is better

    with col2:
        st.markdown(f"**{BENCHMARK_NAME} Performance**")
        if benchmark_data is not None:
             st.metric(f"{BENCHMARK_NAME} Total Return", f"{benchmark_total_return:.2%}")
             st.metric(f"{BENCHMARK_NAME} CAGR", f"{benchmark_cagr:.2%}")
             st.metric(f"{BENCHMARK_NAME} Max Drawdown", f"{benchmark_mdd:.2%}", delta_color="inverse")
        else:
             st.info(f"{BENCHMARK_NAME} data not available for comparison.")

    st.markdown("---")

    # --- Performance Chart ---
    st.subheader("📈 Portfolio Growth vs Benchmark")

    fig = go.Figure()

    # Add Strategy NAV Trace
    fig.add_trace(go.Scatter(
        x=nav_df.index,
        y=nav_df['Normalized NAV'],
        mode='lines',
        name='Strategy NAV',
        line=dict(color='royalblue', width=2),
        hovertemplate='Date: %{x|%Y-%m-%d}<br>Strategy NAV: %{y:.2f}<extra></extra>' # Custom hover
    ))

    # Add Benchmark Trace (if available and processed)
    if not benchmark_perf.empty:
        fig.add_trace(go.Scatter(
            x=benchmark_perf.index,
            y=benchmark_perf,
            mode='lines',
            name=BENCHMARK_NAME,
            line=dict(color='grey', width=1.5, dash='dash'),
             hovertemplate='Date: %{x|%Y-%m-%d}<br>Benchmark: %{y:.2f}<extra></extra>'
        ))

    # Add Trade Markers (if trades exist)
    if not buy_trades.empty:
        fig.add_trace(go.Scatter(
            x=buy_trades.index,
            y=nav_df['Normalized NAV'].reindex(buy_trades.index, method='ffill'), # Get NAV at trade time
            mode='markers',
            name='Buy Orders',
            marker=dict(
                color='rgba(0, 128, 0, 0.7)', # Semi-transparent green
                symbol='triangle-up',
                size=8, # Adjust size as needed
                line=dict(width=1, color='DarkSlateGrey')
            ),
            customdata=buy_trades[['Ticker', 'Price', 'Quantity', 'Trade Value']],
            hovertemplate=(
                '<b>BUY</b> %{customdata[0]}<br>'
                'Date: %{x|%Y-%m-%d}<br>'
                'Price: $%{customdata[1]:.2f}<br>'
                'Quantity: %{customdata[2]:.0f}<br>'
                'Value: $%{customdata[3]:,.0f}'
                '<extra></extra>' # Hides the trace name
            )
        ))

    if not sell_trades.empty:
        fig.add_trace(go.Scatter(
            x=sell_trades.index,
            y=nav_df['Normalized NAV'].reindex(sell_trades.index, method='ffill'),
            mode='markers',
            name='Sell Orders',
            marker=dict(
                color='rgba(255, 0, 0, 0.7)', # Semi-transparent red
                symbol='triangle-down',
                size=8,
                line=dict(width=1, color='DarkSlateGrey')
            ),
            customdata=sell_trades[['Ticker', 'Price', 'Quantity', 'Trade Value']],
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
        hovermode='x unified', # Show tooltips for all traces at a given x-coordinate
        template=PLOT_TEMPLATE,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1) # Legend on top
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Trade Log Details ---
    st.markdown("---")
    if not trade_log_df.empty:
        with st.expander("📜 View Detailed Trade Log", expanded=False):
            # Display relevant columns, sort by index (date)
            display_log = trade_log_df[['Action', 'Ticker', 'Price', 'Quantity', 'Trade Value']].sort_index()
            st.dataframe(display_log)
    else:
        st.info("No trades were logged during this backtest.")
