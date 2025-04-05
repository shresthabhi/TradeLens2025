import os
import pandas as pd
from typing import List, Tuple
from ..config import DATA_ROOT, STORAGE_RULES
from .base_loader import BaseLoader
from ..utils.date_utils import (
    get_time_key_for_frequency,
    get_required_period_keys
)

class PriceDataLoader(BaseLoader):
    """
    Handles loading, saving, and checking availability of price data
    using frequency-aware, config-driven storage format.
    """

    def __init__(self):
        self.data_type = "price_data"

    def _get_ticker_path(self, ticker: str, frequency: str, market: str) -> str:
        return os.path.join(DATA_ROOT, market, self.data_type, ticker, frequency)

    def load(
        self,
        ticker: str,
        frequency: str,
        start_date: str,
        end_date: str,
        market: str
    ):
        """
        Loads locally available data and returns missing period chunks if any.
        """
        folder_path = self._get_ticker_path(ticker, frequency, market)
        if not os.path.exists(folder_path):
            return pd.DataFrame(), get_required_period_keys(start_date, end_date, self.data_type, frequency, STORAGE_RULES)

        rules = STORAGE_RULES[self.data_type][frequency]
        available = set(self.get_available_periods(ticker, frequency, market))
        needed = set(get_required_period_keys(start_date, end_date, self.data_type, frequency, STORAGE_RULES))

        to_load = sorted(available.intersection(needed))
        to_fetch = (needed - available) + [max(needed)] # Need to change this, because this is a quick fix

        frames = []
        for period_key in to_load:
            file_path = os.path.join(folder_path, f"{period_key}.parquet")
            if os.path.exists(file_path):
                df = pd.read_parquet(file_path)
                frames.append(df)

        if not frames:
            return pd.DataFrame(), to_fetch

        combined = pd.concat(frames)
        combined["date"] = pd.to_datetime(combined["date"])
        filtered = combined[
            (combined["date"] >= start_date) & (combined["date"] <= end_date)
        ].reset_index(drop=True)

        return filtered, to_fetch

    def save(
        self,
        ticker: str,
        frequency: str,
        data: pd.DataFrame,
        market: str
    ) -> None:
        folder_path = self._get_ticker_path(ticker, frequency, market)
        os.makedirs(folder_path, exist_ok=True)

        rules = STORAGE_RULES[self.data_type][frequency]

        print("Price saver: \n", data.head())

        data["date"] = pd.to_datetime(data["date"])
        data["chunk_key"] = data["date"].apply(
            lambda x: get_time_key_for_frequency(x, self.data_type, frequency, STORAGE_RULES)
        )

        for chunk_key, group in data.groupby("chunk_key"):
            file_path = os.path.join(folder_path, f"{chunk_key}.parquet")
            group.drop(columns=["chunk_key"]).to_parquet(file_path, index=False)

    def get_available_periods(self, ticker: str, frequency: str, market: str) -> List[str]:
        folder_path = self._get_ticker_path(ticker, frequency, market)
        if not os.path.exists(folder_path):
            return []

        return sorted([
            f.replace(".parquet", "") for f in os.listdir(folder_path) if f.endswith(".parquet")
        ])
