import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import date
import scipy.stats as stats
# Removed plotly.figure_factory as heatmap is removed

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from finlib_data.manager.data_manager import DataManager

# --- Helper Calculation Functions (No change needed here) ---

def calculate_cagr(start_value, end_value, years):
    if years <= 0 or start_value <= 0:
        return 0.0
    if end_value <= 0:
         return -1.0 if start_value > 0 else 0.0
    try:
        growth_factor = end_value / start_value
        if growth_factor < 0:
            return -1.0
        return (growth_factor) ** (1 / years) - 1
    except Exception as e:
        st.warning(f"Error calculating CAGR: {e}")
        return 0.0


def calculate_max_drawdown(returns_series):
    if returns_series.empty:
        return 0.0

    returns_series = pd.to_numeric(returns_series, errors='coerce').fillna(0)

    cumulative_returns = (1 + returns_series).cumprod()
    cumulative_returns = cumulative_returns[np.isfinite(cumulative_returns)]

    if cumulative_returns.empty:
        return 0.0

    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak.replace(0, np.nan)
    max_drawdown = drawdown.min()
    return max_drawdown if pd.notna(max_drawdown) else 0.0

def calculate_sharpe_ratio(returns_series, risk_free_rate_annualized=0.0):
    if returns_series.empty or returns_series.std() == 0:
        return 0.0

    returns_series = pd.to_numeric(returns_series, errors='coerce').dropna()

    if returns_series.empty:
        return 0.0

    annualization_factor = 252

    risk_free_rate_daily = (1 + risk_free_rate_annualized)**(1/annualization_factor) - 1

    excess_returns = returns_series - risk_free_rate_daily
    avg_excess_return = excess_returns.mean()
    std_dev_returns = returns_series.std()

    sharpe_ratio = (avg_excess_return / std_dev_returns) * np.sqrt(annualization_factor)

    return sharpe_ratio if pd.notna(sharpe_ratio) else 0.0

def calculate_volatility(returns_series):
    if returns_series.empty:
        return 0.0

    returns_series = pd.to_numeric(returns_series, errors='coerce').dropna()

    if returns_series.empty:
        return 0.0

    annualization_factor = 252

    volatility = returns_series.std() * np.sqrt(annualization_factor)
    return volatility if pd.notna(volatility) else 0.0

def calculate_alpha_beta(strategy_returns_series, benchmark_returns_series, risk_free_rate_annualized=0.0):
    strategy_returns = pd.to_numeric(strategy_returns_series, errors='coerce').dropna()
    benchmark_returns = pd.to_numeric(benchmark_returns_series, errors='coerce').dropna()

    common_index = strategy_returns.index.intersection(benchmark_returns.index)
    if common_index.empty or len(common_index) < 2:
        return 0.0, 0.0

    strategy_returns = strategy_returns.loc[common_index]
    benchmark_returns = benchmark_returns.loc[common_index]

    annualization_factor = 252
    risk_free_rate_daily = (1 + risk_free_rate_annualized)**(1/annualization_factor) - 1

    strategy_excess_returns = strategy_returns - risk_free_rate_daily
    benchmark_excess_returns = benchmark_returns - risk_free_rate_daily

    try:
         slope, intercept, r_value, p_value, std_err = stats.linregress(
             benchmark_excess_returns, strategy_excess_returns
         )
         alpha_annualized = intercept * annualization_factor
         beta = slope
         return alpha_annualized, beta
    except ValueError:
         return 0.0, 0.0
    except Exception as e:
         st.warning(f"Error during Alpha/Beta calculation: {e}")
         return 0.0, 0.0


def calculate_annual_returns(nav_series):
    if nav_series.empty:
        return pd.Series(dtype=float)

    nav = pd.to_numeric(nav_series, errors='coerce').dropna()
    if nav.empty: return pd.Series(dtype=float)

    year_end_nav = nav.resample('Y').ffill().dropna()

    if year_end_nav.empty or len(year_end_nav) < 2:
         return pd.Series(dtype=float)

    annual_returns = year_end_nav.pct_change().dropna()

    first_year = nav.index.min().year
    last_nav_first_year = year_end_nav.loc[year_end_nav.index.year == first_year].iloc[-1]
    first_nav = nav.iloc[0]

    if first_nav != last_nav_first_year:
         initial_year_return = (last_nav_first_year / first_nav) - 1
         annual_returns = pd.concat([pd.Series([initial_year_return], index=[pd.to_datetime(f'{first_year}-12-31')]), annual_returns])

    annual_returns.index = annual_returns.index.year
    return annual_returns


def calculate_monthly_returns(nav_series):
    if nav_series.empty:
        return pd.DataFrame()

    nav = pd.to_numeric(nav_series, errors='coerce').dropna()
    if nav.empty: return pd.DataFrame()

    month_end_nav = nav.resample('M').ffill().dropna()

    if month_end_nav.empty or len(month_end_nav) < 2:
         return pd.DataFrame()

    monthly_returns_series = month_end_nav.pct_change().dropna()

    first_month = nav.index.min().to_period('M')
    last_nav_first_month_end = month_end_nav.loc[month_end_nav.index.to_period('M') == first_month].iloc[-1]
    first_nav = nav.iloc[0]

    if first_nav != last_nav_first_month_end:
        initial_month_return = (last_nav_first_month_end / first_nav) - 1
        monthly_returns_series = pd.concat([pd.Series([initial_month_return], index=[month_end_nav.index[0]]), monthly_returns_series])


    monthly_returns_df = pd.DataFrame({
        'Year': monthly_returns_series.index.year,
        'Month': monthly_returns_series.index.month,
        'Return': monthly_returns_series.values
    })

    monthly_returns_pivot = monthly_returns_df.pivot(index='Year', columns='Month', values='Return')

    month_names = {
        1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
        7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
    }
    monthly_returns_pivot = monthly_returns_pivot.rename(columns=month_names)

    annual_total_returns = []
    for year in monthly_returns_pivot.index:
        year_months_returns = monthly_returns_pivot.loc[year].dropna()
        if not year_months_returns.empty:
             compounded_return = (1 + year_months_returns).prod() - 1
             annual_total_returns.append(compounded_return)
        else:
             annual_total_returns.append(np.nan)

    monthly_returns_pivot['Total'] = annual_total_returns


    return monthly_returns_pivot


def calculate_drawdown(nav_series):
    if nav_series.empty:
        return pd.Series(dtype=float)

    nav = pd.to_numeric(nav_series, errors='coerce').dropna()
    if nav.empty: return pd.Series(dtype=float)

    peak = nav.cummax()
    drawdown = (nav - peak) / peak.replace(0, np.nan)
    return drawdown if not drawdown.empty else pd.Series(dtype=float)

# --- Display Functions (Modified display_risk_performance_metrics and display_monthly_returns) ---
def display_risk_performance_metrics(nav_df, benchmark_returns, benchmark_name, strategy_total_return, benchmark_total_return, strategy_cagr, benchmark_cagr, strategy_mdd, benchmark_mdd):
    st.subheader("📊 Detailed Performance & Risk Metrics")

    strategy_daily_returns = nav_df['NAV'].pct_change().dropna()

    # Ensure there's benchmark data before proceeding with benchmark-dependent calcs and display
    benchmark_available = benchmark_returns is not None and not benchmark_returns.empty

    benchmark_returns_aligned = pd.Series(dtype=float)
    has_enough_aligned_data = False
    if benchmark_available:
        benchmark_returns_aligned = benchmark_returns.reindex(strategy_daily_returns.index).dropna()
        # Ensure there's enough aligned data for regression before calculating Alpha/Beta
        has_enough_aligned_data = len(strategy_daily_returns) >= 2 and len(benchmark_returns_aligned) >= 2


    # Calculate metrics for Strategy
    strategy_sharpe = calculate_sharpe_ratio(strategy_daily_returns)
    strategy_volatility = calculate_volatility(strategy_daily_returns)
    strategy_alpha = 0.0
    strategy_beta = 0.0
    if has_enough_aligned_data:
         strategy_alpha, strategy_beta = calculate_alpha_beta(strategy_daily_returns, benchmark_returns_aligned)

    # Calculate metrics for Benchmark (if available)
    benchmark_sharpe = 0.0
    benchmark_volatility = 0.0
    if benchmark_available:
         benchmark_sharpe = calculate_sharpe_ratio(benchmark_returns.dropna())
         benchmark_volatility = calculate_volatility(benchmark_returns.dropna())


    # --- Display Metrics in Two Main Columns ---

    # Create two main columns for Strategy and Benchmark metrics
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Strategy Performance**")
        st.metric("Total Return", f"{strategy_total_return:.2%}")
        st.metric("CAGR", f"{strategy_cagr:.2%}")
        # Add delta_color="inverse" for Max Drawdown as lower is better
        st.metric("Max Drawdown", f"{strategy_mdd:.2%}", delta_color="inverse")
        # Include Sharpe and Volatility for Strategy
        st.metric("Sharpe Ratio (Annualized)", f"{strategy_sharpe:.2f}")
        st.metric("Volatility (Annualized)", f"{strategy_volatility:.2%}")

        # Include Alpha and Beta here, as they are strategy metrics relative to the benchmark
        # Only show them if benchmark data was available for calculation
        if benchmark_available:
             st.metric(f"Alpha (vs {benchmark_name})", f"{strategy_alpha:.2f}")
             st.metric(f"Beta (vs {benchmark_name})", f"{strategy_beta:.2f}")
        else:
             st.info(f"Alpha and Beta require {benchmark_name} data for calculation.")


    with col2:
        # Use the actual benchmark name in the header
        st.markdown(f"**{benchmark_name} Performance**")
        if benchmark_available:
             # Display benchmark metrics
             st.metric("Total Return", f"{benchmark_total_return:.2%}")
             st.metric("CAGR", f"{benchmark_cagr:.2%}")
             st.metric("Max Drawdown", f"{benchmark_mdd:.2%}", delta_color="inverse") # Lower is better
             # Include Sharpe and Volatility for Benchmark
             st.metric("Sharpe Ratio (Annualized)", f"{benchmark_sharpe:.2f}")
             st.metric("Volatility (Annualized)", f"{benchmark_volatility:.2%}")
        else:
             # Display a message if benchmark data is not available
             st.info(f"{benchmark_name} data not available for comparison.")


    st.markdown("---") # Add a separator at the end

# def display_risk_performance_metrics(nav_df, benchmark_returns, benchmark_name, strategy_total_return, benchmark_total_return, strategy_cagr, benchmark_cagr, strategy_mdd, benchmark_mdd):
#     st.subheader("📊 Detailed Performance & Risk Metrics")

#     strategy_daily_returns = nav_df['NAV'].pct_change().dropna()
#     benchmark_returns_aligned = benchmark_returns.reindex(strategy_daily_returns.index).dropna()

#     # Ensure there's enough aligned data for regression before calculating Alpha/Beta
#     has_enough_aligned_data = len(strategy_daily_returns) >= 2 and len(benchmark_returns_aligned) >= 2

#     strategy_sharpe = calculate_sharpe_ratio(strategy_daily_returns)
#     strategy_volatility = calculate_volatility(strategy_daily_returns)
#     benchmark_sharpe = calculate_sharpe_ratio(benchmark_returns.dropna())
#     benchmark_volatility = calculate_volatility(benchmark_returns.dropna())

#     strategy_alpha = 0.0
#     strategy_beta = 0.0
#     if has_enough_aligned_data:
#          strategy_alpha, strategy_beta = calculate_alpha_beta(strategy_daily_returns, benchmark_returns_aligned)


#     # Display Metrics - Adjusted layout for Alpha/Beta
#     st.markdown("**Strategy Metrics**")
#     col1, col2, col3 = st.columns(3)
#     col1.metric("Total Return", f"{strategy_total_return:.2%}")
#     col2.metric("CAGR", f"{strategy_cagr:.2%}")
#     col3.metric("Max Drawdown", f"{strategy_mdd:.2%}", delta_color="inverse")

#     col1, col2, col3 = st.columns(3)
#     col1.metric("Sharpe Ratio (Annualized)", f"{strategy_sharpe:.2f}")
#     col2.metric("Volatility (Annualized)", f"{strategy_volatility:.2%}")
#     col3.metric(f"Alpha (vs {benchmark_name})", f"{strategy_alpha:.2f}")

#     col1, col2, col3 = st.columns(3) # Use new columns for Beta
#     col1.metric(f"Beta (vs {benchmark_name})", f"{strategy_beta:.2f}")


#     st.markdown(f"**{benchmark_name} Metrics**")
#     if benchmark_returns is not None and not benchmark_returns.empty:
#         col1, col2, col3 = st.columns(3) # Use new columns for benchmark metrics
#         col1.metric("Total Return", f"{benchmark_total_return:.2%}")
#         col2.metric("CAGR", f"{benchmark_cagr:.2%}")
#         col3.metric("Max Drawdown", f"{benchmark_mdd:.2%}", delta_color="inverse")

#         col1, col2, col3 = st.columns(3)
#         col1.metric("Sharpe Ratio (Annualized)", f"{benchmark_sharpe:.2f}")
#         col2.metric("Volatility (Annualized)", f"{benchmark_volatility:.2%}")

#     else:
#          st.info(f"{benchmark_name} data not available for detailed comparison.")

#     st.markdown("---")


def display_drawdown_chart(nav_df, plot_template):
    st.subheader("📉 Drawdown Over Time")

    strategy_drawdown = calculate_drawdown(nav_df['NAV'])

    if strategy_drawdown.empty:
        st.info("Not enough data to calculate or display drawdown.")
        return

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=strategy_drawdown.index,
        y=strategy_drawdown,
        mode='lines',
        name='Drawdown',
        line=dict(color='darkorange', width=2),
        fill='tozeroy',
        fillcolor='rgba(255, 140, 0, 0.3)',
        hovertemplate='Date: %{x|%Y-%m-%d}<br>Drawdown: %{y:.2%}<extra></extra>'
    ))

    fig.update_layout(
        title='Strategy Drawdown from Peak',
        xaxis_title='Date',
        yaxis_title='Drawdown (%)',
        yaxis_tickformat='.0%',
        template=plot_template,
        hovermode='x unified',
        shapes=[
            dict(
                type='line',
                xref='paper', yref='y',
                x0=0, y0=0,
                x1=1, y1=0,
                line=dict(color='grey', width=1, dash='dot'),
            )
        ],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")


def display_annual_returns(nav_df, benchmark_data, benchmark_name):
    st.subheader("📅 Annual Returns")

    strategy_annual_returns = calculate_annual_returns(nav_df['NAV'])
    benchmark_annual_returns = pd.Series(dtype=float)

    if benchmark_data is not None and not benchmark_data.empty:
        benchmark_annual_returns = calculate_annual_returns(benchmark_data['Close'])

    annual_returns_df = pd.DataFrame({
        'Strategy': strategy_annual_returns,
        benchmark_name: benchmark_annual_returns
    }).sort_index()


    if annual_returns_df.empty or annual_returns_df.dropna(how='all').empty:
        st.info("Not enough data to calculate or display annual returns.")
        return

    formatted_df = annual_returns_df.applymap(lambda x: f"{x:.2%}" if pd.notna(x) else '-')

    st.dataframe(formatted_df)
    st.markdown("---")


def display_monthly_returns(nav_df, benchmark_data): # Removed plot_template as heatmap is removed
    st.subheader("🗓️ Monthly Returns")

    strategy_monthly_returns = calculate_monthly_returns(nav_df['NAV'])
    benchmark_monthly_returns = pd.DataFrame()
    benchmark_name = st.session_state.get("BENCHMARK_NAME", "Benchmark") # Get benchmark name for display


    if isinstance(benchmark_data, pd.DataFrame) and not benchmark_data.empty:
         benchmark_monthly_returns = calculate_monthly_returns(benchmark_data['Close'])


    # --- Display Monthly Returns Table ---
    if not strategy_monthly_returns.empty:
         st.markdown("**Strategy Monthly Returns (%)**")
         formatted_strategy_monthly = strategy_monthly_returns.applymap(lambda x: f"{x:.2%}" if pd.notna(x) else '-')
         st.dataframe(formatted_strategy_monthly)
    else:
         st.info("Not enough data to calculate or display strategy monthly returns.")


    if not benchmark_monthly_returns.empty:
         st.markdown(f"**{benchmark_name} Monthly Returns (%)**")
         formatted_benchmark_monthly = benchmark_monthly_returns.applymap(lambda x: f"{x:.2%}" if pd.notna(x) else '-')
         st.dataframe(formatted_benchmark_monthly)

    # Removed the heatmap plotting code

    st.markdown("---")

