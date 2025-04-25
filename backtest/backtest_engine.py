import pandas as pd
from .strategy import Strategy
from .portfolio import Portfolio
from .rebalancer import Rebalancer

from typing import Callable, Dict, List, Optional

class BacktestEngine:
    def __init__(
        self,
        strategy: Strategy,
        stock_prices: pd.DataFrame,
        fundamentals: pd.DataFrame,
        tickers: List[str],
        start_date: str,
        end_date: str,
        initial_cash: float = 100000
    ):
        self.strategy = strategy
        self.prices = stock_prices
        self.fundamentals = fundamentals
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.portfolio = Portfolio(initial_cash)
        self.nav_history = []

    def run(self):
        dates = sorted(self.prices['date'].unique())
        last_rebalance = None

        for date in dates:
            if date < self.start_date or date > self.end_date:
                continue

            day_prices = self.prices[self.prices['date'] == date].set_index('ticker')['close'].to_dict()
            day_fundamentals = self.fundamentals[self.fundamentals['date'] <= date].sort_values('date').groupby('ticker').last().to_dict("index")

            for ticker in list(self.portfolio.holdings.keys()):
                if ticker in day_fundamentals:
                    days_held = (pd.to_datetime(date) - pd.to_datetime(self.portfolio.holdings[ticker]['entry_date'])).days
                    if self.strategy.exit_condition(day_fundamentals[ticker]) or days_held >= self.strategy.hold_period:
                        if ticker in day_prices:
                            self.portfolio.sell(ticker, date, day_prices[ticker])

            rebalance_needed = False
            qualifying_stocks = []
            for ticker in self.tickers:
                if ticker not in self.portfolio.holdings and ticker in day_fundamentals and ticker in day_prices:
                    if self.strategy.entry_condition(day_fundamentals[ticker]):
                        qualifying_stocks.append(ticker)

            if self.strategy.rebalance_mode == "event" and qualifying_stocks:
                rebalance_needed = True
            elif self.strategy.rebalance_mode == "periodic":
                if not last_rebalance or (pd.to_datetime(date) - pd.to_datetime(last_rebalance)).days >= self.strategy.rebalance_frequency:
                    rebalance_needed = True

            if rebalance_needed:
                combined_stocks = set(qualifying_stocks + list(self.portfolio.holdings.keys()))
                valid_stocks = []
                for ticker in combined_stocks:
                    if ticker in day_prices:
                        valid_stocks.append(ticker)
                    elif ticker in self.portfolio.holdings:
                        self.portfolio.sell(ticker, date, 0)

                if valid_stocks:
                    start_date_to_rebalance = (pd.to_datetime(date) - pd.Timedelta(days=90)).strftime('%Y-%m-%d')
                    reb = Rebalancer(
                        tickers=valid_stocks,
                        start_date=start_date_to_rebalance,
                        end_date=date,
                        strategy=self.strategy.rebalance_strategy,
                        risk_free_rate=0.03  # or parameterize this
                    )
                    weights = reb.compute_weights()
                    

                    total_portfolio_value = self.portfolio.cash
                    current_values = {}
                    for ticker in valid_stocks:
                        price = day_prices[ticker]
                        quantity = self.portfolio.holdings.get(ticker, {}).get("quantity", 0)
                        value = price * quantity
                        current_values[ticker] = value
                        total_portfolio_value += value

                    for i, ticker in enumerate(valid_stocks):
                        target_value = weights[i] * total_portfolio_value
                        current_quantity = self.portfolio.holdings.get(ticker, {}).get("quantity", 0)
                        current_value = day_prices[ticker] * current_quantity
                        diff_value = target_value - current_value
                        diff_quantity = int(diff_value / day_prices[ticker])

                        if diff_quantity > 0:
                            cost = diff_quantity * day_prices[ticker]
                            if self.portfolio.cash >= cost:
                                self.portfolio.buy(ticker, date, day_prices[ticker], diff_quantity)
                        elif diff_quantity < 0:
                            self.portfolio.sell(ticker, date, day_prices[ticker], abs(diff_quantity))

                last_rebalance = date

            nav = self.portfolio.get_value(day_prices)
            self.nav_history.append((date, nav))

    def get_nav_history(self):
        return pd.DataFrame(self.nav_history, columns=["Date", "NAV"]).set_index("Date")

    def get_trade_log(self):
        return pd.DataFrame(self.portfolio.history, columns=["Date", "Action", "Ticker", "Price", "Quantity"]).set_index("Date")
