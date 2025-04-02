# backend.py
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import minimize

def fetch_data(tickers, start_date, end_date, **kwargs):
    """
    Fetch historical adjusted close price data from Yahoo Finance.
    """
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust = False, **kwargs)['Adj Close']
    return data

def calculate_returns(price_data):
    """
    Calculate daily returns from price data.
    """
    returns = price_data.pct_change().dropna()
    return returns

def optimize_portfolio(returns_df, risk_free_rate=0.0):
    """
    Compute optimal portfolio weights to maximize the Sharpe Ratio.
    Constraints:
      - Sum of weights = 1
      - All weights >= 0 (no short-selling)
    """
    N = returns_df.shape[1]
    mean_returns = returns_df.mean().values
    cov_matrix = returns_df.cov().values

    def neg_sharpe(weights):
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = (portfolio_return - risk_free_rate) / portfolio_std
        return -sharpe  # Negative because we are minimizing

    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = tuple((0.0, 1.0) for _ in range(N))
    init_guess = np.ones(N) / N

    result = minimize(neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    if not result.success:
        raise ValueError("Optimization failed: " + result.message)
    return result.x


