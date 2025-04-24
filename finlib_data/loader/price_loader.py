import os
import pandas as pd
from typing import List, Tuple
from ..config import DATA_ROOT, STORAGE_RULES
from .base_loader import BaseLoader
from ..utils.date_utils import (
    get_time_key_for_frequency,
    get_required_period_keys,
    get_date_range_for_key
)

import json

class PriceDataLoader(BaseLoader):
    """
    Handles loading, saving, and checking availability of price data
    using frequency-aware, config-driven storage format.
    """

    def __init__(self):
        self.data_type = "price_data"

    def _get_ticker_path(self, ticker: str, frequency: str, market: str) -> str:
        return os.path.join(DATA_ROOT, market, self.data_type, ticker, frequency)
    
    def _is_chunk_complete(self, chunk_key, data, frequency, rules):

        should_min_date, should_max_date = get_date_range_for_key(chunk_key, self.data_type, frequency, rules) 

        actual_min_date = data.date.min()
        actual_max_date = data.date.max()

        if(should_min_date != actual_min_date or should_max_date != actual_max_date):
            return False
        
        return True


    def _update_price_metadata_file(self, chunk_key, data, folder_path, frequency, rules):

        is_complete = self._is_chunk_complete(chunk_key, data, frequency, rules)
        file_path = os.path.jon(folder_path, "is_price_data_complete.json")
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = dict()
        else:
            data = dict()
        
        if(is_complete):
            data[chunk_key] = True
        else:
            data[chunk_key] = False
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)

        print(f"Updated JSON saved to {file_path}")

        return None

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
        # print(needed)

        to_load = sorted(available.intersection(needed))
        to_fetch = (needed - available).union(set([max(needed)])) # Need to change this, because this is a quick fix
        
        # Quick fix for daily price range
        # to_load = sorted([a for a in list(available) if a != "2025"])
        # to_fetch = ["2025"]
        # print(f"to load : {to_load}\n to fetch : {to_fetch}")

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

        # print("Price saver: \n", data.head())

        data["date"] = pd.to_datetime(data["date"])
        data["chunk_key"] = data["date"].apply(
            lambda x: get_time_key_for_frequency(x, self.data_type, frequency, STORAGE_RULES)
        )

        for chunk_key, group in data.groupby("chunk_key"):
            file_path = os.path.join(folder_path, f"{chunk_key}.parquet")
            group.drop(columns=["chunk_key"]).to_parquet(file_path, index=False)

            # self._update_price_metadata_file(chunk_key, group, folder_path, frequency, STORAGE_RULES)
            


    def get_available_periods(self, ticker: str, frequency: str, market: str) -> List[str]:
        folder_path = self._get_ticker_path(ticker, frequency, market)
        if not os.path.exists(folder_path):
            return []

        return sorted([
            f.replace(".parquet", "") for f in os.listdir(folder_path) if f.endswith(".parquet")
        ])
