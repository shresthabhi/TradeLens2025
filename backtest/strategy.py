from typing import Callable, Dict, List, Optional


class Strategy:
    def __init__(
        self,
        entry_condition: Callable[[Dict], bool],
        exit_condition: Callable[[Dict], bool],
        hold_period: int = 0,
        rebalance_mode: str = "event",  # "event" or "periodic"
        rebalance_frequency: Optional[int] = 30,
        rebalance_strategy: str = "sharpe"  # "sharpe" or "min_volatility"
    ):
        self.entry_condition = entry_condition
        self.exit_condition = exit_condition
        self.hold_period = hold_period
        self.rebalance_mode = rebalance_mode
        self.rebalance_frequency = rebalance_frequency
        self.rebalance_strategy = rebalance_strategy