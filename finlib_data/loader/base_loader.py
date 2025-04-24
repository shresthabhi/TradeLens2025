from abc import ABC, abstractmethod
from typing import Any
import pandas as pd


class BaseLoader(ABC):
    """
    Abstract base class for all the data loaders
    Every loader must define how to load and save files
    """

    @abstractmethod
    def load(self, ticker: str, start_date: str, end_date: str, market: str):
        """
        Load data from local storate for a given ticker symbol and time range
        """
        pass

    @abstractmethod
    def save(self, ticker: str, strat_date: str, end_date:str, market: str) -> None:
        """
        Save data to local storage
        """
        pass
    
    @abstractmethod
    def get_available_periods(self, ticker: str, frequency:str, market:str) -> list:
        """
        Check what data is already available (e.g., list of years or months).
        """
        pass



