from typing import Optional
import pandas as pd
from ..loader.price_loader import PriceDataLoader
from ..fetcher.yahoo_price_fetcher import YahooPriceFetcher

class DataManager:
    """
    Coordinates data loading and fetching.
    """

    def __init__(self):
        self.loaders = {
            "price_data": PriceDataLoader()
            # "financials": FinancialDataLoader(), etc.
        }

        self.fetchers = {
            "price_data": YahooPriceFetcher()
            # "financials": AlphaVantageFetcher(), etc.
        }

    def get_data(
        self,
        ticker: str,
        data_type: str,
        frequency: str,
        start_date: str,
        end_date: str,
        market: str = "usa"
    ) -> Optional[pd.DataFrame]:
        """
        Returns a DataFrame for the requested time range.
        Fetches and saves missing data if needed.
        """
        loader = self.loaders.get(data_type)
        fetcher = self.fetchers.get(data_type)

        if not loader or not fetcher:
            raise ValueError(f"No loader or fetcher defined for data type: {data_type}")

        # Step 1: Try loading locally available data
        local_data, missing_chunks = loader.load(ticker, frequency, start_date, end_date, market)

        # Step 2: If missing chunks, fetch them
        if missing_chunks:
            print(f"[INFO] Missing chunks for {ticker}: {missing_chunks}")
            fetched_data = fetcher.fetch(ticker, frequency, missing_chunks, start_date, end_date, market)
            
            if not fetched_data.empty:
                print("Data manager\n", fetched_data.head())
                print("this, this , this")
                loader.save(ticker, frequency, fetched_data, market)
                # Reload the final dataset to include fetched data
                full_data, _ = loader.load(ticker, frequency, start_date, end_date, market)
                return full_data
            else:
                print(f"[WARN] Fetcher returned no data for missing chunks.")
                return local_data  # Return what we have

        return local_data
