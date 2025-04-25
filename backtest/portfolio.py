from typing import Dict, Optional

class Portfolio:
    def __init__(self, initial_cash: float):
        self.cash = initial_cash
        self.holdings = {}  # ticker -> {entry_date, price, quantity}
        self.history = []

    def buy(self, ticker: str, date: str, price: float, quantity: int):
        cost = price * quantity
        if self.cash >= cost:
            self.cash -= cost
            self.holdings[ticker] = {"entry_date": date, "price": price, "quantity": quantity}
            self.history.append((date, "BUY", ticker, price, quantity))

    def sell(self, ticker: str, date: str, price: float):
        if ticker in self.holdings:
            quantity = self.holdings[ticker]["quantity"]
            self.cash += price * quantity
            del self.holdings[ticker]
            self.history.append((date, "SELL", ticker, price, quantity))

    def get_value(self, prices: Dict[str, float]) -> float:
        value = self.cash
        for ticker, pos in self.holdings.items():
            if ticker in prices:
                value += prices[ticker] * pos["quantity"]
        return value