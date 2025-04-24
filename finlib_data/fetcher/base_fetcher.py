from abc import ABC, abstractmethod
import pandas as pd
from typing import List

class BaseFetcher(ABC):
    """
    Abstract base class for all data fetchers.
    """

    @abstractmethod
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
        Fetches missing data and returns a combined DataFrame.
        """
        pass
