import pandas as pd
import numpy as np
import sys
import os
from typing import Callable, Dict, List, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backtest.strategy import Strategy
from backtest.backtest_engine import BacktestEngine
from finlib_data.manager.data_manager import DataManager
import streamlit as st

def get_fundamental_data(tickers_list):
    data_fundamentals = pd.read_csv("financial_ratios.csv")
    data_fundamentals["date"] = pd.to_datetime(data_fundamentals["public_date"])
    data_fundamentals.columns = [col.lower() for col in data_fundamentals.columns]
    return data_fundamentals[data_fundamentals.ticker.isin(tickers_list)].copy()

def get_stock_price_data(start_date, end_date, tickers_list):
    dm = DataManager()
    data_stock = pd.DataFrame()

    for ticker in tickers_list:
        temp_data = dm.get_data(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            frequency="daily",
            market="usa",
            data_type="price_data"
        )
        if temp_data is not None and not temp_data.empty:
            temp_data["ticker"] = ticker
            data_stock = pd.concat([data_stock, temp_data[["date", "Close", "ticker"]]])

    data_stock.columns = [col.lower() for col in data_stock.columns]
    return data_stock

def run_backtest_from_ui(tickers, start_date, end_date, entry_lambda, exit_lambda, rebalance_mode, rebalance_strategy, rebalance_frequency, holding_period):
    try:
        tickers = tickers
        start_date = start_date.strftime("%Y-%m-%d")
        end_date = end_date.strftime("%Y-%m-%d")

        data_fundamentals = get_fundamental_data(tickers)
        data_stock_prices = get_stock_price_data(start_date, end_date, tickers)


        print(tickers)
        print(start_date, end_date)
        print(data_fundamentals.head())
        print(data_stock_prices.head())
        print(entry_lambda)
        print(exit_lambda)

        strategy = Strategy(
            entry_condition=entry_lambda,
            exit_condition=exit_lambda,
            hold_period=holding_period,
            rebalance_mode=rebalance_mode,
            rebalance_frequency=rebalance_frequency,
            rebalance_strategy=rebalance_strategy,
        )

        engine = BacktestEngine(
            strategy=strategy,
            stock_prices=data_stock_prices,
            fundamentals=data_fundamentals,
            tickers=tickers,
            start_date=pd.Timestamp(start_date),
            end_date=pd.Timestamp(end_date),
            initial_cash=100000
        )

        engine.run()

        print(engine.get_nav_history().tail())
        return engine.get_nav_history(), engine.get_trade_log()

    except Exception as e:
        st.error(f"Backtest failed: {e}")
        return None, None
