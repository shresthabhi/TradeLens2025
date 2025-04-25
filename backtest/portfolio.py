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
            if ticker in self.holdings:
                self.holdings[ticker]["quantity"] += quantity
            else:
                self.holdings[ticker] = {"entry_date": date, "price": price, "quantity": quantity}
            self.history.append((date, "BUY", ticker, price, quantity))

    def sell(self, ticker: str, date: str, price: float, quantity: Optional[int] = None):
        if ticker in self.holdings:
            current_quantity = self.holdings[ticker]["quantity"]
            sell_quantity = quantity if quantity is not None else current_quantity
            sell_quantity = min(sell_quantity, current_quantity)
            if sell_quantity > 0:
                self.cash += price * sell_quantity
                self.history.append((date, "SELL", ticker, price, sell_quantity))
                if sell_quantity == current_quantity:
                    del self.holdings[ticker]
                else:
                    self.holdings[ticker]["quantity"] -= sell_quantity

    def get_value(self, prices: Dict[str, float]) -> float:
        value = self.cash
        for ticker, pos in self.holdings.items():
            if ticker in prices:
                value += prices[ticker] * pos["quantity"]
        return value