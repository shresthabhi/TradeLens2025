import pandas as pd
from .strategy import Strategy
from .portfolio import Portfolio
from .rebalancer import Rebalancer

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

            # Evaluate sell conditions
            for ticker in list(self.portfolio.holdings.keys()):
                if ticker in day_fundamentals:
                    days_held = (pd.to_datetime(date) - pd.to_datetime(self.portfolio.holdings[ticker]['entry_date'])).days
                    if self.strategy.exit_condition(day_fundamentals[ticker]) or days_held >= self.strategy.hold_period:
                        if ticker in day_prices:
                            self.portfolio.sell(ticker, date, day_prices[ticker])

            # Rebalance check
            should_rebalance = False
            if self.strategy.rebalance_mode == "after_purchase":
                should_rebalance = True
            elif self.strategy.rebalance_mode == "periodic":
                if not last_rebalance or (pd.to_datetime(date) - pd.to_datetime(last_rebalance)).days >= self.strategy.rebalance_frequency:
                    should_rebalance = True

            # Entry condition check
            if should_rebalance:
                for ticker in self.tickers:
                    if ticker in day_prices and ticker in day_fundamentals and ticker not in self.portfolio.holdings:
                        if self.strategy.entry_condition(day_fundamentals[ticker]):
                            quantity = int(self.portfolio.cash / len(self.tickers) / day_prices[ticker])
                            if quantity > 0:
                                self.portfolio.buy(ticker, date, day_prices[ticker], quantity)
                last_rebalance = date

            nav = self.portfolio.get_value(day_prices)
            self.nav_history.append((date, nav))

    def get_nav_history(self):
        return pd.DataFrame(self.nav_history, columns=["Date", "NAV"]).set_index("Date")

    def get_trade_log(self):
        return pd.DataFrame(self.portfolio.history, columns=["Date", "Action", "Ticker", "Price", "Quantity"]).set_index("Date")
