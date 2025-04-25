# backend_diversification.py
import yfinance as yf
import numpy as np
import pandas as pd
from backtest.rebalancer import Rebalancer  # Import your existing class

def fetch_data(tickers, start_date, end_date, **kwargs):
    """
    Fetch historical adjusted close price data from Yahoo Finance.
    """
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False, **kwargs)['Adj Close']
    return data

def get_max_sharpe_portfolio(tickers, start_date, end_date, risk_free_rate=0.03):
    """
    Optimize portfolio for maximum Sharpe ratio.
    """
    rb = Rebalancer(tickers, start_date, end_date, strategy="sharpe", risk_free_rate=risk_free_rate)
    optimized_weights = rb.compute_weights()
    return optimized_weights

def get_min_volatility_portfolio(tickers, start_date, end_date, risk_free_rate=0.03):
    """
    Optimize portfolio for minimum volatility (risk).
    """
    rb = Rebalancer(tickers, start_date, end_date, strategy="min_volatility", risk_free_rate=risk_free_rate)
    optimized_weights = rb.compute_weights()
    return optimized_weights

def generate_risk_return_chart(tickers, start_date, end_date, risk_free_rate=0.03, n_samples=10000):
    """
    Simulate random portfolios and compute their returns, volatility, and Sharpe ratios.
    """
    prices_df = fetch_data(tickers, start_date, end_date)
    returns_df = prices_df.pct_change().dropna()

    mean_returns = returns_df.mean().values
    cov_matrix = returns_df.cov().values
    N = len(tickers)

    results = []

    for _ in range(n_samples):
        weights = np.random.dirichlet(np.ones(N), size=1).flatten()  # Random weights that sum to 1
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std
        results.append({
            "Return": portfolio_return,
            "Risk": portfolio_std,
            "Sharpe": sharpe_ratio,
            "Weights": weights
        })

    return pd.DataFrame(results)

def analyze_existing_portfolio(tickers, current_values, start_date, end_date, risk_free_rate=0.03):
    """
    Analyze the current risk, return, and Sharpe ratio of a user-provided portfolio.
    """
    prices_df = fetch_data(tickers, start_date, end_date)
    returns_df = prices_df.pct_change().dropna()

    mean_returns = returns_df.mean().values
    cov_matrix = returns_df.cov().values
    daily_risk_free_rate = risk_free_rate/252

    # Normalize current values to get weights
    total_value = sum(current_values)
    weights = np.array(current_values) / total_value

    # Portfolio return and risk
    port_return = np.dot(weights, mean_returns)
    port_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (port_return - daily_risk_free_rate) / port_std

    return {
        "Return": port_return,
        "Risk": port_std,
        "Sharpe Ratio": sharpe_ratio,
        "Weights": weights
    }



# # backend.py
# import yfinance as yf
# import numpy as np
# import pandas as pd
# from scipy.optimize import minimize

# def fetch_data(tickers, start_date, end_date, **kwargs):
#     """
#     Fetch historical adjusted close price data from Yahoo Finance.
#     """
#     data = yf.download(tickers, start=start_date, end=end_date, auto_adjust = False, **kwargs)['Adj Close']
#     return data

# def calculate_returns(price_data):
#     """
#     Calculate daily returns from price data.
#     """
#     returns = price_data.pct_change().dropna()
#     return returns

# def optimize_portfolio(returns_df, risk_free_rate=0.0):
#     """
#     Compute optimal portfolio weights to maximize the Sharpe Ratio.
#     Constraints:
#       - Sum of weights = 1
#       - All weights >= 0 (no short-selling)
#     """
#     N = returns_df.shape[1]
#     mean_returns = returns_df.mean().values
#     cov_matrix = returns_df.cov().values

#     def neg_sharpe(weights):
#         portfolio_return = np.dot(weights, mean_returns)
#         portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
#         sharpe = (portfolio_return - risk_free_rate) / portfolio_std
#         return -sharpe  # Negative because we are minimizing

#     constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
#     bounds = tuple((0.0, 1.0) for _ in range(N))
#     init_guess = np.ones(N) / N

#     result = minimize(neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
#     if not result.success:
#         raise ValueError("Optimization failed: " + result.message)
#     return result.x


