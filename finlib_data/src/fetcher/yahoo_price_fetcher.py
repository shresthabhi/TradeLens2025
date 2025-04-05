import yfinance as yf
import pandas as pd
from typing import List
from .base_fetcher import BaseFetcher

class YahooPriceFetcher(BaseFetcher):
    """
    Fetches price data using yfinance.
    """

    def __init__(self):
        pass

    def fetch(
        self,
        ticker: str,
        frequency: str,
        missing_chunks: List[str],
        start_date: str,
        end_date: str,
        market: str
    ) -> pd.DataFrame:
        """
        Fetches data from Yahoo Finance for the specified period.
        Ignores `missing_chunks` for now; fetches the full date range.
        """

        interval_map = {
            "daily": "1d",
            "weekly": "1wk",
            "intraday_1h": "60m",
            "intraday_1min": "1m"
        }

        if frequency not in interval_map:
            raise ValueError(f"Frequency '{frequency}' not supported by yfinance.")

        interval = interval_map[frequency]

        df = yf.download(
            tickers=ticker,
            start=start_date,
            end=end_date,
            interval=interval,
            progress=False
        )

        if df.empty:
            return pd.DataFrame()

        df.columns = df.columns.get_level_values(0)
        
        df.reset_index(inplace=True)
        df.index.name = None

        df.rename(columns={"Date": "date"}, inplace=True)
        df["ticker"] = ticker

        print("Fetcher\n", df.head())

        return df
