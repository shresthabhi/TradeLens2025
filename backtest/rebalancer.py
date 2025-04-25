import pandas as pd
import numpy as np
from scipy.optimize import minimize

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from finlib_data.manager.data_manager import DataManager

class Rebalancer:
    def __init__(self, tickers, start_date, end_date, strategy, risk_free_rate = 0.03):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.strategy = strategy
        self.risk_free_rate = risk_free_rate
        self.daily_risk_free_rate = self.risk_free_rate / 252

        dm = DataManager()

        data = pd.DataFrame()
        for ticker in tickers:
            temp_data = dm.get_data(
                ticker=ticker,
                start_date=self.start_date,
                end_date=self.end_date,
                data_type="price_data",
                frequency="daily",
                market="usa"
            )

            if temp_data is not None and not temp_data.empty:
                temp_data["ticker"] = ticker
                columns_to_drop = [col for col in temp_data.columns if col not in ["date", "Close", "ticker"]]
                temp_data = temp_data.drop(columns=columns_to_drop)
                
                data  =  pd.concat([data, temp_data])

        data.columns =  [col.lower() for col in data.columns]
        data = data.pivot(index="date", values="close", columns="ticker")
        
        self.prices_df = data


    def compute_weights(self):
        returns_df = self.prices_df.pct_change().dropna()
        N = returns_df.shape[1]
        mean_returns = returns_df.mean().values
        cov_matrix = returns_df.cov().values

        if self.strategy == "sharpe":
            def objective(weights):
                port_return = np.dot(weights, mean_returns)
                port_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                epsilon = 1e-8
                return -(port_return - self.daily_risk_free_rate) / (port_std + 1e-8)
        elif self.strategy == "min_volatility":
            def objective(weights):
                return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        else:
            raise ValueError("Unknown strategy")

        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = tuple((0.0, 1.0) for _ in range(N))
        init_guess = np.ones(N) / N

        result = minimize(objective, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        if not result.success:
            raise ValueError("Optimization failed: " + result.message)
        return result.x
